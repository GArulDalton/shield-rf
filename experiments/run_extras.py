import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
os.chdir(os.path.join(ROOT, "runs"))
import numpy as np, pickle, json, time
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from shieldrf.data import *; from shieldrf.model import *

devs = make_devices(45, seed=1)
KNOWN, UNK = list(range(30)), list(range(30, 40))
SEEN, UNSEEN, ALL = ['AWGN', 'Rician', 'TDL-A'], ['Rayleigh', 'TDL-C'], CHANNELS
M = ['CE-DNN', 'Center-Proto', 'DANN', 'SHIELD-RF']
P = {m: pickle.load(open(f'P_{m}_0.pkl', 'rb')) for m in M}
negE = lambda lg: np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
R = {}

# 1) SNR sweep on unseen channels
snrs = [-5, 0, 5, 10, 15, 20, 25, 30]
R['snr'] = {m: {'acc': [], 'auroc': []} for m in M}
for i, sn in enumerate(snrs):
    Xk, yk, *_ = generate(devs, KNOWN, UNSEEN, 20, (sn, sn), seed=5000 + i)
    Xu, *_ = generate(devs, UNK, UNSEEN, 20, (sn, sn), seed=6000 + i)
    for m in M:
        _, lk = infer(P[m], Xk); _, lu = infer(P[m], Xu)
        R['snr'][m]['acc'].append(float((lk.argmax(1) == yk).mean()))
        R['snr'][m]['auroc'].append(float(roc_auc_score(np.r_[np.ones(len(lk)), np.zeros(len(lu))], np.r_[negE(lk), negE(lu)])))
R['snr']['snrs'] = snrs
print('snr done', flush=True)

# 2) Few-shot enrollment of the 10 new devices (no retraining)
Xe, ye, *_ = generate(devs, UNK, SEEN, 20, (10, 30), seed=7000)
Xn, yn, *_ = generate(devs, UNK, ALL, 40, (0, 30), seed=7001)
Xk, yk, *_ = generate(devs, KNOWN, ALL, 20, (0, 30), seed=7002)
R['fewshot'] = {}
rng = np.random.default_rng(0)
for m in M:
    ze, _ = infer(P[m], Xe); zn, _ = infer(P[m], Xn); zk, _ = infer(P[m], Xk)
    protoK = np.asarray(l2n(P[m]['proto']))
    res = {}
    for K in [1, 5, 10, 20]:
        an, ak = [], []
        for r in range(5):
            newp = []
            for d in UNK:
                idx = np.where(ye == d)[0]; pick = rng.choice(idx, K, replace=False)
                v = ze[pick].mean(0); newp.append(v / np.linalg.norm(v))
            PR = np.vstack([protoK, np.array(newp)])
            an.append(((zn @ PR.T).argmax(1) == yn).mean()); ak.append(((zk @ PR.T).argmax(1) == yk).mean())
        res[K] = dict(new=float(np.mean(an)), new_sd=float(np.std(an)), known=float(np.mean(ak)))
    R['fewshot'][m] = res
print('fewshot done', flush=True)

# 3) channel leakage: can a linear probe recover the channel type from the embedding?
Xl, yl, cl, *_ = generate(devs, KNOWN, ALL, 12, (10, 30), seed=8000)
R['leak'] = {}
for m in M:
    z, _ = infer(P[m], Xl)
    R['leak'][m] = float(cross_val_score(LogisticRegression(max_iter=2000), z, cl, cv=5).mean())
print('leak done', flush=True)

# 4) t-SNE (10 known devices, unseen channels)
Xt, yt, ct, *_ = generate(devs, list(range(10)), UNSEEN, 40, (15, 30), seed=9000)
emb = {}
for m in ['CE-DNN', 'SHIELD-RF']:
    z, _ = infer(P[m], Xt)
    emb[m] = TSNE(2, perplexity=30, random_state=0, init='pca').fit_transform(z)
np.savez('tsne.npz', ce=emb['CE-DNN'], sh=emb['SHIELD-RF'], y=yt, c=ct)

# 5) energy histograms on unseen channels (SNR 0-30)
Xk2, *_ = generate(devs, KNOWN, UNSEEN, 20, (0, 30), seed=9100)
Xu2, *_ = generate(devs, UNK, UNSEEN, 60, (0, 30), seed=9101)
H = {}
for m in ['CE-DNN', 'SHIELD-RF']:
    _, lk = infer(P[m], Xk2); _, lu = infer(P[m], Xu2)
    H[m + '_k'] = -negE(lk); H[m + '_u'] = -negE(lu)
np.savez('energy.npz', **H)

# 6) complexity
d_in = Xl.shape[1]
params = sum(W.size + b.size for W, b in P['SHIELD-RF']['enc']) + P['SHIELD-RF']['proto'].size
flops = sum(2 * W.size for W, b in P['SHIELD-RF']['enc']) + 2 * P['SHIELD-RF']['proto'].size
Xb = Xl[:1000]; t = time.time()
for _ in range(5): infer(P['SHIELD-RF'], Xb)
R['complexity'] = dict(d_in=int(d_in), params=int(params), flops=int(flops), ms_per_burst=(time.time() - t) / 5000 * 1e3)
json.dump(R, open('extras.json', 'w'), indent=1)
print('ALL DONE', flush=True)
