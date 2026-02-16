# House Buying Classifier: Round-wise Objectives and Step-by-step Progress

Date range covered: February 16, 2026

## Overall objective

Build a multi-class house-buying decision model (`no`, `neutral`, `yes`) that generalizes well on unseen data, while keeping a strict experiment log and visual history for every round.

## Round 0 - Baseline notebook setup (initial phase)

Objective:
- Create an end-to-end notebook pipeline (data loading, preprocessing, model training, metrics, visuals).

Steps:
1. Created synthetic dataset with core features and target `can_buy`.
2. Trained multiple neural networks (`nn_small`, `nn_medium`, `nn_deep`).
3. Evaluated train/CV/test and added confusion matrix + ROC + comparison charts.

Outcome:
- Established the base modeling/evaluation framework.
- Detected overfitting risk from train vs CV/test gaps.

## Round 1 - Addition of model diversity + bootstrap ensemble

Objective:
- Improving generalization by adding model diversity and ensemble learning.

Steps:
1. Added non-NN models: KNN, KMeans++, K-Medoids.
2. Added bootstrap bagging and weighted soft voting.
3. Logged metrics in a centralized report.

Outcome:
- Ensemble helped in earlier small-data setup.
- Framework became multi-model, not NN-only.

## Round 2 - Dataset scaling and variance increase

Objective:
- To increase data scale and feature-space variation so models learn broader patterns.

Steps:
1. Rebuilt generator with richer scenario-driven sampling and more variance.
2. Regenerated splits to:
   - train = 3000
   - CV = 400
   - test = 300
3. Re-ran full pipeline and updated snapshots.

Outcome:
- Metrics became more stable and realistic.
- Better separation between robust and unstable model choices.

## Round 3 - Adaptive ensemble and model tuning

Objective:
- Reduce overfitting and focus ensemble on stronger model families.

Steps:
1. Tuned NN and neighbor/cluster configs.
2. Changed ensemble to adaptive top-family selection by CV rank.
3. Used CV-thresholded squared weighting in bootstrap voting.
4. Added automatic result snapshotting and figure generation.

Outcome:
- Promoted profile: `opt_round3_medium_tuned`.
- Best balanced single model became `nn_medium`.

Key promoted numbers (from `metrics_snapshot_2026-02-16_16-22-56.csv`):
- `nn_medium` test accuracy: `0.8767`
- `nn_medium` test macro-F1: `0.8779`
- `bootstrap_ensemble` test macro-F1: `0.8744`

## Round 4 - Exploratory optimization (not promoted)

Objective:
- To probe whether further gains are possible via seed sensitivity, stacking, and extra ensemble variants.

Steps:
1. Ran structured trials and stored outputs:
   - `csv/optimization_trials_round4_2026-02-16.csv`
   - `optimization_trials_round4_2026-02-16.md`
2. Tested seed sensitivity for tuned `nn_medium`.
3. Tested stacking and alternate ensemble family combinations.

Outcome:
- Best test-only candidate found: `nn_medium_rs=59` with test macro-F1 `0.8845`.
- Not promoted because CV macro-F1 dropped (`0.8540`), so robustness was weaker.

## Experiment governance improvements made

1. Central report with appended snapshots:
   - `results/experiment_results_log.md`
2. Per-run metrics snapshots:
   - `csv/metrics_snapshot_*.csv`
3. Persistent visual history archive by tag:
   - `results/history/<tag>/figures`
4. Auto-archive support added to:
   - `export_results_visuals.py --tag <snapshot_tag>`

