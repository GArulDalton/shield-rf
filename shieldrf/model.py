import autograd.numpy as anp
from autograd import grad
import numpy as np
from autograd.scipy.special import logsumexp
from .data import augment_raw, featurize, perturb_impairments

def init_mlp(sizes, rng):
    return [(rng.normal(0, np.sqrt(2 / m), (m, n)).astype(np.float64), np.zeros(n)) for m, n in zip(sizes[:-1], sizes[1:])]

def mlp(params, x, final_act=False):
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1 or final_act:
            x = anp.maximum(x, 0.01 * x)   # leaky ReLU
    return x

def l2n(z):
    return z / anp.sqrt(anp.sum(z ** 2, axis=1, keepdims=True) + 1e-8)

class Adam:
    def __init__(self, params, lr=1e-3):
        self.lr = lr; self.t = 0
        self.m = [np.zeros_like(p) for p in flat(params)]
        self.v = [np.zeros_like(p) for p in flat(params)]
    def step(self, params, grads):
        self.t += 1
        ps, gs = flat(params), flat(grads)
        out = []
        for i, (p, g) in enumerate(zip(ps, gs)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + 1e-8))
        return unflat(params, out)

def flat(t):
    if isinstance(t, (list, tuple)):
        return [x for s in t for x in flat(s)]
    if isinstance(t, dict):
        return [x for k in sorted(t) for x in flat(t[k])]
    return [t]

def unflat(tmpl, vals, i=None):
    it = iter(vals) if i is None else i
    def rec(t):
        if isinstance(t, list): return [rec(s) for s in t]
        if isinstance(t, tuple): return tuple(rec(s) for s in t)
        if isinstance(t, dict): return {k: rec(t[k]) for k in sorted(t)}
        return next(it)
    return rec(tmpl)

SCALE = 16.0

def embed(P, x):
    return l2n(mlp(P['enc'], x))

def logits_fn(P, z):
    return SCALE * z @ anp.transpose(l2n(P['proto']))

def softmax_ce(logits, y):
    return -anp.mean(logits[np.arange(len(y)), y] - logsumexp(logits, axis=1))

def energy(logits, T=1.0):
    return -T * logsumexp(logits / T, axis=1)

def supcon(z1, z2, y, tau=0.1):
    z = anp.concatenate([z1, z2], 0)
    yy = np.concatenate([y, y])
    sim = z @ anp.transpose(z) / tau
    n = len(yy)
    eye = np.eye(n)
    sim = sim - 1e9 * eye
    logp = sim - logsumexp(sim, axis=1, keepdims=True)
    pos = (yy[:, None] == yy[None, :]).astype(float) - eye
    return -anp.mean(anp.sum(logp * pos, 1) / pos.sum(1))

# ---------------------------------------------------------------------------
def train(Xraw, Xf, y, c, n_cls, cfg, seed=0, log=print):
    """Unified trainer. cfg flags: contrast, adv, proto, energy_margin, cosine, aug, center."""
    rng = np.random.default_rng(seed)
    d_in = Xf.shape[1]; H = cfg.get('H', 256); D = cfg.get('D', 64)
    P = {'enc': init_mlp([d_in, H, H, D], rng),
         'proto': rng.normal(0, 1, (n_cls, D)),
         'adv': init_mlp([D, 64, 5], rng)}
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    P_norm = (mu, sd)
    lam_adv = cfg.get('lam_adv', 0.3); lam_p = cfg.get('lam_proto', 0.5); lam_e = cfg.get('lam_e', 0.1)
    m_in, m_out = cfg.get('m_in', -14.0), cfg.get('m_out', -9.0)

    def enc_loss(P, xa, xb, yb, cb, phase, xo=None):
        za = embed(P, xa)
        if phase == 'pre':
            zb = embed(P, xb)
            return supcon(za, zb, yb)
        L = 0.0
        lg = logits_fn(P, za)
        L = L + softmax_ce(lg, yb)
        if cfg.get('contrast_ft') and xb is not None:
            L = L + 0.5 * supcon(za, embed(P, xb), yb)
        if cfg.get('proto'):
            pn = l2n(P['proto'])[yb]
            L = L + lam_p * anp.mean(anp.sum((za - pn) ** 2, 1))
        if cfg.get('adv'):
            la = mlp(P['adv'], za)
            L = L - lam_adv * softmax_ce(la, cb)          # gradient reversal
        if cfg.get('energy_margin') and xo is not None:
            Ein = energy(lg); Eo = energy(logits_fn(P, embed(P, xo)))
            L = L + lam_e * (anp.mean(anp.maximum(Ein - m_in, 0) ** 2) + anp.mean(anp.maximum(m_out - Eo, 0) ** 2))
        elif cfg.get('energy_margin'):
            Ein = energy(lg)
            idx = rng.permutation(len(yb)); a = rng.uniform(0.3, 0.7, (len(yb), 1))
            zm = l2n(a * za + (1 - a) * za[idx])
            keep = (yb != yb[idx]).astype(float)
            Eout = energy(logits_fn(P, zm))
            L = L + lam_e * (anp.mean(anp.maximum(Ein - m_in, 0) ** 2) +
                             anp.sum(keep * anp.maximum(m_out - Eout, 0) ** 2) / (keep.sum() + 1))
        return L

    def adv_loss(Padv, z, cb):
        return softmax_ce(mlp(Padv, z), cb)

    g_enc = grad(enc_loss); g_adv = grad(adv_loss)
    opt = Adam(P, cfg.get('lr', 1e-3)); opt_a = Adam(P['adv'], 1e-3)
    norm = lambda X: (X - mu) / sd
    bs = cfg.get('bs', 256)
    phases = []
    if cfg.get('contrast'):
        phases += ['pre'] * cfg.get('pre_epochs', 8)
    phases += ['ft'] * cfg.get('epochs', 20)
    for ep, phase in enumerate(phases):
        use_aug = cfg.get('aug') or phase == 'pre' or cfg.get('contrast_ft')
        if use_aug:
            XA = norm(featurize(augment_raw(Xraw, rng)))
            XB = norm(featurize(augment_raw(Xraw, rng))) if (phase == 'pre' or cfg.get('contrast_ft')) else None
            if cfg.get('aug') and phase == 'ft' and not cfg.get('contrast_ft'):
                # mix clean + augmented
                msk = rng.random(len(XA)) < 0.5
                XA = np.where(msk[:, None], XA, norm(Xf))
        else:
            XA, XB = norm(Xf), None
        XO = None
        if cfg.get('ipos') and phase == 'ft':
            sub = rng.choice(len(y), len(y) // 2, replace=False)
            XO = norm(featurize(augment_raw(perturb_impairments(Xraw[sub], rng), rng)))
        perm = rng.permutation(len(y)); tot = 0
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            xb = XB[b] if XB is not None else None
            xo = XO[rng.choice(len(XO), len(b))] if XO is not None else None
            G = g_enc(P, XA[b], xb, y[b], c[b], phase, xo)
            tot += 1
            P = opt.step(P, G)
            if cfg.get('adv') and phase == 'ft':
                z = np.asarray(embed(P, XA[b]))
                P['adv'] = opt_a.step(P['adv'], g_adv(P['adv'], z, c[b]))
        if (ep + 1) % 5 == 0:
            log(f"  ep{ep+1} {phase}")
    P['_norm'] = P_norm
    return P

def infer(P, Xf):
    mu, sd = P['_norm']
    z = np.asarray(embed(P, (Xf - mu) / sd))
    lg = SCALE * z @ np.asarray(l2n(P['proto'])).T
    return z, lg
