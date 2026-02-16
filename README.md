# House Buying ML Lab

A multi-class machine learning project to classify whether a buyer can purchase a house (`no`, `neutral`, `yes`) using synthetic tabular data.

## What's included

- Dataset generator with controlled variation and split sizing (`generate_house_dataset.py`)
- Jupyter experiment notebook (`house_buying_nn_experiment.ipynb`)
- Model comparison across:
  - Neural networks
  - KNN
  - KMeans++ label model
  - K-Medoids label model
  - Bootstrap weighted ensemble
- Experiment logging and reporting utilities:
  - `record_experiment_results.py`
  - `export_results_visuals.py`
- Full results snapshots, visual history, and docs in `results/`

## Current promoted profile

See `results/experiment_results_log.md` and `results/docs/` for round-wise objectives, math/statistics reasoning, and future roadmap.
