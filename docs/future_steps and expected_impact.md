# What Next: Action Plan and Expected Impact

## Current status checkpoint

- Promoted stable profile: `opt_round3_medium_tuned`.
- Best balanced model now: `nn_medium`.
- Strongest test-only candidate exists (`nn_medium_rs=59`) but lower CV robustness.

## Recommended next actions (with expected outcomes)

## 1) Repeat-seed evaluation protocol for promotion

Action:
- Run each top candidate over multiple seeds (for both model init and bootstrap sampling).
- Report mean and standard deviation of CV/test macro-F1.

Expected impact:
- More defensible model selection.
- Reduced risk of promoting a lucky seed.

## 2) Nested cross-validation for hyperparameter tuning

Action:
- Inner loop: tune hyperparameters.
- Outer loop: unbiased performance estimate.

Expected impact:
- Better estimate of true generalization.
- Lower overfitting to one CV split.

## 3) Calibrated probability modeling

Action:
- Apply probability calibration (Platt scaling or isotonic) on top candidates.

Expected impact:
- Improved log-loss and decision confidence quality.
- Better downstream threshold-based decisions (risk-sensitive policies).

## 4) Threshold policy for class decisions

Action:
- Instead of strict argmax, define risk-aware thresholds:
  - high confidence for `yes`
  - conservative assignment to `neutral` near boundaries

Expected impact:
- Better practical decision behavior in ambiguous cases.
- Lower costly false positives for `yes`.

## 5) Segment-wise validation (subpopulation performance)

Action:
- Evaluate metrics by feature segments:
  - income bands
  - credit bands
  - employment type
  - loan eligibility

Expected impact:
- Detect hidden weak spots.
- Targeted feature engineering/model fixes for specific cohorts.

## 6) Robustness and drift checks

Action:
- Simulate distribution shifts (e.g., lower credit regime, higher price-to-income regime).
- Re-evaluate model and ensemble behavior.

Expected impact:
- Better resilience to real-world market changes.
- Early warning for retraining triggers.

## 7) Improve synthetic data realism with constrained generation

Action:
- Enforce additional realistic correlations:
  - credit score vs loan eligibility
  - income vs EMI burden
  - location score vs price band

Expected impact:
- Reduced synthetic-data artifacts.
- Better transfer of learned relationships.

## 8) Add explainability layer

Action:
- Use SHAP/permutation importance on promoted models.
- Produce per-class explanation summaries.

Expected impact:
- Better interpretability and stakeholder trust.
- Faster debugging for failure cases.

## 9) Candidate deployment strategy

Action:
- Keep two profiles:
  1. `stable_profile` (current promoted model)
  2. `aggressive_profile` (higher test but weaker CV robustness)
- Use shadow evaluation before switching.

Expected impact:
- Safe iteration without losing current reliability.

## 10) Operational checklist for every future round

Action:
1. Train and evaluate.
2. Export visuals with tag:
   - `python export_results_visuals.py --tag <round_tag>`
3. Record metrics snapshot:
   - `python record_experiment_results.py --tag <round_tag>`
4. Append rationale in docs.

Expected impact:
- Full reproducibility.
- Clean audit trail across all model changes.

## Suggested immediate next round

If the goal is to beat current test macro-F1 while preserving robustness:
- Start with seed-repeat analysis of `nn_medium` and `nn_medium_rs=59`.
- Promote only if CV mean and variance stay competitive against `opt_round3_medium_tuned`.

