# SHIELD-RF

Code for the paper **"Channel Robust Open Set RF Fingerprint Authentication via Impairment Guided Outlier Synthesis"** by G. Arul Dalton (Saveetha Engineering College, Chennai, India), submitted to *Annals of Telecommunications*.

SHIELD-RF is a gateway side radio frequency fingerprint identification (RFFI) authenticator for IoT networks. It (i) learns channel robust embeddings with receiver side channel augmentation and supervised contrastive pretraining, (ii) scores device familiarity with the free energy of a cosine prototype classifier, (iii) trains an energy margin on **IPOS** (impairment perturbation outlier synthesis) virtual rogue transmitters, (iv) rejects rogue devices with an SNR adaptive threshold, and (v) enrolls new devices from a few bursts without retraining.

## Repository layout

```
shieldrf/
  data.py          transmitter impairment model, channel models, features, channel augmentation, IPOS
  model.py         encoder, cosine prototype head, losses (SupCon, CE, prototype, energy margin), trainer
  authenticate.py  gateway API: SNR-adaptive calibration, authentication and few-shot enrollment (Algorithm 3)
examples/
  quickstart.py    train, calibrate, authenticate and enroll in about 1 minute
docs/
  IMPLEMENTATION.md  paper-to-code map, configuration flags, data protocol, runtimes
experiments/
  run_experiments.py   main comparison (part "main") and ablation study (part "abl") for one seed
  run_extras.py        SNR sweep, few shot enrollment, channel leakage probe, t-SNE, energy histograms, complexity
  aggregate.py         mean and std over seeds -> runs/agg_main.json, runs/agg_abl.json
  make_figures.py      all paper figures (block diagrams + result plots) -> figures/
scripts/run_all.sh     reproduces every table and figure
results/               aggregated results reported in the paper (agg_main.json, agg_abl.json, extras.json)
figures/               figures used in the paper (vector PDF and 600 dpi PNG)
```

## Installation

```bash
git clone https://github.com/GArulDalton/shield-rf.git
cd shield-rf
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No GPU is needed. Everything runs on a single CPU core with NumPy and autograd.

## Quick start

```bash
python examples/quickstart.py
```

```python
from shieldrf import make_devices, generate, train, SHIELDAuthenticator, REJECT
auth = SHIELDAuthenticator(trained_params)
auth.calibrate(X_val, snr_val)              # per-SNR-band thresholds
decisions, energy = auth.authenticate(X, snr_est)   # device id or REJECT (-1)
auth.enroll(X_new_device_bursts, device_id=30)      # no retraining
```

See `docs/IMPLEMENTATION.md` for the complete paper-to-code map.

## Reproducing the paper

```bash
bash scripts/run_all.sh        # about 1.5 h on one CPU core
```

or step by step:

```bash
python experiments/run_experiments.py 0 main   # seed 0 main comparison (~4 min)
python experiments/run_experiments.py 0 abl    # seed 0 ablations (~10 min)
python experiments/run_extras.py               # needs the seed 0 main models
python experiments/aggregate.py main
python experiments/aggregate.py abl
python experiments/make_figures.py
```

Intermediate files (trained models, per seed JSON) are written to `runs/`. All random number generators are seeded, so results are reproducible; a fresh run of seed 0 reproduces the numbers in `results/` exactly.

## Benchmark

* 45 virtual transmitters with IQ imbalance, CFO and drift, DC offset, cubic PA nonlinearity and Wiener phase noise; Zadoff-Chu preamble, N = 128.
* Devices 1-30 registered, 31-40 rogue (test only), 41-45 development only.
* Seen channels: AWGN, Rician, TDL-A like. Unseen channels: Rayleigh, TDL-C like with Doppler. SNR 0-30 dB.

## Main results (mean +/- std over three seeds, %)

| Method | Acc seen | Acc unseen | AUROC seen | AUROC unseen |
|---|---|---|---|---|
| CE-DNN (MSP) | 79.6 +/- 0.4 | 62.0 +/- 1.1 | 67.6 +/- 1.4 | 60.4 +/- 0.5 |
| OpenMax | 79.6 +/- 0.4 | 62.0 +/- 1.1 | 70.4 +/- 1.6 | 63.7 +/- 0.8 |
| DANN | 78.4 +/- 1.4 | 60.6 +/- 1.1 | 66.0 +/- 0.8 | 60.2 +/- 0.9 |
| **SHIELD-RF** | **85.2 +/- 0.8** | **74.3 +/- 0.9** | **84.8 +/- 0.9** | **76.0 +/- 1.0** |

See `results/agg_main.json` and `results/agg_abl.json` for all metrics (OSCR, FPR95, URR, open set accuracy).

## Citation

If you use this code, please cite the paper (full details will be added after publication); see `CITATION.cff`.

## License

MIT, see `LICENSE`.
