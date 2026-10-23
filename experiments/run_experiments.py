import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
os.chdir(os.path.join(ROOT, "runs"))
import numpy as np, json, time, sys, pickle
from scipy.stats import weibull_min
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from shieldrf.data import *
from shieldrf.model import *

SEED = int(sys.argv[1]); PART = sys.argv[2]
devs = make_devices(45, seed=1)
KNOWN, UNK = list(range(30)), list(range(30, 40))
SEEN, UNSEEN = ['AWGN', 'Rician', 'TDL-A'], ['Rayleigh', 'TDL-C']
Xf, y, c, s, raw = generate(devs, KNOWN, SEEN, 150, (0, 30), seed=100 + SEED)
Xv, yv, cv, sv, _ = generate(devs, KNOWN, SEEN, 20, (0, 30), seed=200 + SEED)
test = {}
for tag, chans in [('seen', SEEN), ('unseen', UNSEEN)]:
    test[tag + '_k'] = generate(devs, KNOWN, chans, 40, (0, 30), seed=300 + SEED + len(tag))
    test[tag + '_u'] = generate(devs, UNK, chans, 40, (0, 30), seed=400 + SEED + len(tag))

FULL = dict(contrast=True, pre_epochs=10, epochs=30, proto=True, energy_margin=True, ipos=True, aug=True)
METHODS = {
    'CE-DNN': dict(epochs=30),
    'Center-Proto': dict(epochs=30, proto=True),
    'DANN': dict(epochs=30, adv=True),
    'SHIELD-RF': FULL,
}
ABL = {
    'w/o contrastive pretraining': {**FULL, 'contrast': False},
    'w/o prototype loss': {**FULL, 'proto': False},
    'IPOS -> embedding-mixup outliers': {**FULL, 'ipos': False},
    'w/o energy margin (no outliers)': {**FULL, 'energy_margin': False, 'ipos': False},
    '+ gradient-reversal channel adversary': {**FULL, 'adv': True},
    'w/o fine-tuning channel augmentation': {**FULL, 'aug': False},
}

def softmax(lg):
    e = np.exp(lg - lg.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def energy_np(lg): return -(np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1))

def oscr(pred_k, yk, sk, su):
    # Dhamija et al. 2018: area under CCR vs FPR
    thr = np.sort(np.concatenate([sk, su]))[::-1]
    corr = pred_k == yk
    ccr = [(corr & (sk >= t)).mean() for t in thr[::10]]
    fpr = [(su >= t).mean() for t in thr[::10]]
    return float(np.trapezoid([0] + ccr + [corr.mean()], [0] + fpr + [1]))

class OpenMax:
    def fit(self, lg, y, tail=20):
        self.mav, self.wb = [], []
        for k in range(lg.shape[1]):
            a = lg[(y == k) & (lg.argmax(1) == k)]
            m = a.mean(0); d = np.linalg.norm(a - m, axis=1)
            self.mav.append(m); self.wb.append(weibull_min.fit(np.sort(d)[-tail:], floc=0))
        self.mav = np.array(self.mav)
    def score(self, lg, alpha=5):
        out = np.zeros(len(lg))
        for i, v in enumerate(lg):
            rank = np.argsort(v)[::-1][:alpha]; w = np.ones_like(v)
            for r, k in enumerate(rank):
                d = np.linalg.norm(v - self.mav[k])
                w[k] = 1 - (alpha - r) / alpha * weibull_min.cdf(d, *self.wb[k])
            v2 = v * w; unk = np.sum(v * (1 - w))
            p = np.exp(np.append(v2, unk) - max(v2.max(), unk)); p /= p.sum()
            out[i] = 1 - p[-1]
        return out

def scores(name, lg, om=None):
    if name == 'CE-DNN (MSP)': return softmax(lg).max(1)
    if name == 'OpenMax': return om.score(lg)
    return -energy_np(lg)   # higher = more "known"

def evaluate(P, method, cfgname, om=None):
    res = {}
    _, lgv = infer(P, Xv); sv_score = scores(method, lgv, om)
    # global + SNR-adaptive thresholds at 95% TPR on validation (known only)
    gthr = np.quantile(sv_score, 0.05)
    bins = [0, 10, 20, 31]
    athr = [np.quantile(sv_score[(sv >= bins[i]) & (sv < bins[i + 1])], 0.05) for i in range(3)]
    for tag in ['seen', 'unseen']:
        Xk, yk, ck, sk_, _ = test[tag + '_k']; Xu, _, _, su_, _ = test[tag + '_u']
        _, lk = infer(P, Xk); _, lu = infer(P, Xu)
        sk, su = scores(method, lk, om), scores(method, lu, om)
        pred = lk.argmax(1)
        tk = np.array([athr[np.searchsorted(bins, v, 'right') - 1] for v in sk_])
        tu = np.array([athr[np.searchsorted(bins, v, 'right') - 1] for v in su_])
        def openacc(tk_, tu_):
            kc = ((pred == yk) & (sk >= tk_)).mean(); ur = (su < tu_).mean()
            return kc, ur, (kc * len(yk) + ur * len(su)) / (len(yk) + len(su))
        kg, ug, og = openacc(gthr, gthr); ka, ua, oa = openacc(tk, tu)
        res[tag] = dict(acc=float((pred == yk).mean()),
                        auroc=float(roc_auc_score(np.r_[np.ones(len(sk)), np.zeros(len(su))], np.r_[sk, su])),
                        oscr=oscr(pred, yk, sk, su),
                        fpr95=float((su >= np.quantile(sk, 0.05)).mean()),
                        urr_global=float(ug), openacc_global=float(og),
                        urr_adapt=float(ua), openacc_adapt=float(oa))
    return res

out = {}
if PART == 'main':
    for m, cfg in METHODS.items():
        t = time.time(); P = train(raw, Xf, y, c, 30, cfg, seed=SEED, log=lambda *a: None)
        tt = time.time() - t
        pickle.dump(P, open(f'P_{m}_{SEED}.pkl', 'wb'))
        if m == 'CE-DNN':
            _, lgt = infer(P, Xf); om = OpenMax(); om.fit(lgt, y)
            for mm in ['CE-DNN (MSP)', 'CE-DNN (Energy)', 'OpenMax']:
                out[mm] = evaluate(P, mm, m, om); out[mm]['train_s'] = tt
        else:
            nm = m if m != 'DANN' else 'DANN'
            out[m] = evaluate(P, m, m); out[m]['train_s'] = tt
        print(m, json.dumps(out.get(m, out.get('CE-DNN (MSP)'))), flush=True)
elif PART == 'abl':
    import os
    for i, (m, cfg) in enumerate(ABL.items()):
        fn = f'abl_{SEED}_{i}.json'
        if os.path.exists(fn): continue
        P = train(raw, Xf, y, c, 30, cfg, seed=SEED, log=lambda *a: None)
        json.dump({m: evaluate(P, m, m)}, open(fn, 'w')); print(m, flush=True)
    out = {}
    for i in range(len(ABL)): out.update(json.load(open(f'abl_{SEED}_{i}.json')))
if PART == 'main' or len(out) == len(ABL):
    json.dump(out, open(f'res_{PART}_{SEED}.json', 'w'), indent=1)
