"""Quick start: train a small SHIELD-RF model, calibrate, authenticate and enroll.
Runs in about 1 minute on one CPU core (10 registered devices instead of 30; the paper's
full protocol is in experiments/run_experiments.py).

    python examples/quickstart.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shieldrf import make_devices, generate, train, SHIELDAuthenticator, REJECT

SEEN, UNSEEN = ['AWGN', 'Rician', 'TDL-A'], ['Rayleigh', 'TDL-C']
devs = make_devices(45, seed=1)
known, rogue = list(range(10)), list(range(30, 35))       # small demo: 10 registered, 5 rogue

# 1) training data (registered devices, seen channels only)
Xf, y, c, s, raw = generate(devs, known, SEEN, 150, (0, 30), seed=100)
cfg = dict(contrast=True, pre_epochs=10, epochs=30, proto=True, energy_margin=True, ipos=True, aug=True)
print('training SHIELD-RF (10 registered devices)...')
P = train(raw, Xf, y, c, len(known), cfg, seed=0, log=print)

# 2) calibration on legitimate validation bursts
Xv, yv, cv, sv, _ = generate(devs, known, SEEN, 15, (0, 30), seed=200)
auth = SHIELDAuthenticator(P)
print('SNR-band thresholds tau_b:', np.round(auth.calibrate(Xv, sv), 2))

# 3) authenticate registered and rogue bursts on UNSEEN channels
Xk, yk, _, sk, _ = generate(devs, known, UNSEEN, 20, (10, 30), seed=300)
Xu, _, _, su, _ = generate(devs, rogue, UNSEEN, 20, (10, 30), seed=301)
dk, _ = auth.authenticate(Xk, sk); du, _ = auth.authenticate(Xu, su)
print(f'registered: accepted+correct {np.mean(dk == yk):.1%}, wrongly rejected {np.mean(dk == REJECT):.1%}')
print(f'rogue:      rejected {np.mean(du == REJECT):.1%}')

# 4) few-shot enrollment of a new device (device 30) from 5 bursts, no retraining
Xe, *_ = generate(devs, [30], SEEN, 5, (15, 30), seed=400)
auth.enroll(Xe[:5], device_id=30)
Xn, yn, _, sn, _ = generate(devs, [30], SEEN + UNSEEN, 10, (10, 30), seed=401)
_, lg = auth._embed_logits(Xn)
print(f'enrolled device 30: identified in {np.mean(np.array(auth.labels)[lg.argmax(1)] == 30):.1%} of bursts')
