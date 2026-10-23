#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper (CPU only; ~1.5 h on one core).
set -e
cd "$(dirname "$0")/.."
for s in 0 1 2; do python experiments/run_experiments.py $s main; done
for s in 0 1 2; do python experiments/run_experiments.py $s abl;  done
python experiments/run_extras.py
python experiments/aggregate.py main
python experiments/aggregate.py abl
python experiments/make_figures.py
