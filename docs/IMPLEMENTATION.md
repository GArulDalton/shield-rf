# Implementation details

This document maps every part of the paper *Channel Robust Open Set RF Fingerprint Authentication via Impairment Guided Outlier Synthesis* to the code.

## 1. Paper-to-code map

| Paper element | Code |
|---|---|
| Zadoff-Chu preamble, N = 128, u = 25 (Sec. 3.1) | `shieldrf/data.py: zc()`, constant `PRE` |
| Device fingerprint Θ_d (Table 1 ranges) | `shieldrf/data.py: make_devices()` |
| PA nonlinearity, Eq. (1); IQ imbalance, CFO, DC, phase noise, Eq. (2) | `shieldrf/data.py: tx()` |
| Channel model, Eq. (3); AWGN, Rician, TDL-A, Rayleigh, TDL-C | `shieldrf/data.py: channel()`, `awgn()` |
| Dual-view feature vector v ∈ R^640, Eq. (4) | `shieldrf/data.py: featurize()` |
| Dataset generation for given devices, channels, SNR range | `shieldrf/data.py: generate()` |
| Channel augmentation 𝒜 (random FIR, Doppler, noise), Sec. 4.1 | `shieldrf/data.py: augment_raw()` |
| IPOS impairment perturbation 𝒯_η, Eq. (9), Algorithm 2 | `shieldrf/data.py: perturb_impairments()` + `augment_raw()` |
| Encoder f_θ (MLP 640-256-256-64, leaky ReLU, ℓ2 norm) | `shieldrf/model.py: init_mlp(), mlp(), embed()` |
| Cosine prototype head ℓ_k = s zᵀp̂_k, s = 16 | `shieldrf/model.py: logits_fn()` |
| Free energy E(v), Eq. (5) | `shieldrf/model.py: energy()`, `shieldrf/authenticate.py: _energy()` |
| Supervised contrastive loss, Eq. (6) | `shieldrf/model.py: supcon()` |
| Total loss, Eq. (7); energy margin loss, Eq. (8) | `shieldrf/model.py: train() → enc_loss()` |
| Training procedure, Algorithm 1 | `shieldrf/model.py: train()` |
| SNR-adaptive calibration, authentication, enrollment, Algorithm 3 | `shieldrf/authenticate.py: SHIELDAuthenticator` |
| Gradient-reversal adversary (DANN baseline and ablation) | `shieldrf/model.py: train()` with `adv=True` |
| OpenMax baseline | `experiments/run_experiments.py: OpenMax` |
| Metrics: Acc, AUROC, OSCR, FPR95, URR, open-set accuracy | `experiments/run_experiments.py: evaluate(), oscr()` |
| Tables 2, 3, 4 (main results) | `run_experiments.py <seed> main` → `aggregate.py main` |
| Table 6 (ablation) | `run_experiments.py <seed> abl` → `aggregate.py abl` |
| Table 5 (channel leakage), Figs. 4-7, complexity | `experiments/run_extras.py` |
| Figs. 1-7 | `experiments/make_figures.py` |

## 2. Training configuration flags

`shieldrf.model.train(Xraw, Xf, y, c, n_cls, cfg, seed)` is a single trainer for SHIELD-RF, all baselines and all ablations. `cfg` keys:

| Key | Meaning | SHIELD-RF |
|---|---|---|
| `contrast`, `pre_epochs` | Stage 1 supervised contrastive pretraining and its epochs | True, 10 |
| `epochs` | Stage 2 fine-tuning epochs | 30 |
| `aug` | channel augmentation during fine-tuning (half clean, half augmented) | True |
| `proto`, `lam_proto` | prototype compactness loss and weight λ_p | True, 0.5 |
| `energy_margin`, `lam_e`, `m_in`, `m_out` | energy margin loss, weight λ_e, margins | True, 0.1, −14, −9 |
| `ipos` | use IPOS outliers in the energy margin (otherwise embedding mixup outliers) | True |
| `adv`, `lam_adv` | gradient-reversal channel adversary and weight | False, 0.3 |
| `lr`, `bs`, `H`, `D` | learning rate, batch size, hidden width, embedding size | 1e-3, 256, 256, 64 |

Baselines in the paper: CE-DNN = `dict(epochs=30)`, Center-Proto = `dict(epochs=30, proto=True)`, DANN = `dict(epochs=30, adv=True)`.

## 3. Data protocol

* 45 devices from `make_devices(45, seed=1)`: ids 0-29 registered, 30-39 rogue (test only), 40-44 development only.
* Training: seen channels AWGN, Rician, TDL-A; 150 bursts per device and channel; SNR U[0, 30] dB.
* Validation (threshold calibration only): 20 bursts per device and channel, seen channels.
* Test: 40 bursts per device and channel, seen and unseen (Rayleigh, TDL-C) channels, registered and rogue devices.
* Seeds 0, 1, 2 change data and initialisation; results are mean ± std over seeds.

## 4. Implementation notes

* Pure NumPy with `autograd` for gradients; Adam is implemented in `shieldrf/model.py: Adam`. No GPU or deep learning framework is needed.
* Gradient reversal is implemented by subtracting λ_adv × adversary loss from the encoder loss, while the adversary is updated separately to minimise its own loss.
* SNR is assumed known at the receiver with 10 dB band granularity for threshold selection; in deployment use a preamble-based SNR estimator.
* Thresholds are calibrated on seen channels. Under unseen channels the legitimate acceptance rate drifts below 95% (see paper Sec. 6.6); the quick start shows this effect.
* All random number generators are seeded, and a fresh run of seed 0 reproduces `results/` exactly.

## 5. Runtime (one CPU core)

| Step | Time |
|---|---|
| `examples/quickstart.py` | about 1 min |
| `run_experiments.py <seed> main` (4 models + OpenMax) | about 4 min |
| `run_experiments.py <seed> abl` (6 variants) | about 10 min |
| `run_extras.py` | about 2 min |
| Full reproduction, `scripts/run_all.sh` | about 1.5 h |
