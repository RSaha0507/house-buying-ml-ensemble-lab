# Round 4 Optimization Trials

- File: `optimization_trials_round4_2026-02-16.csv`
- Families used for stacking: ['nn_medium', 'nn_deep', 'nn_small', 'knn']

## round4_ensemble_variants

| variant | test_accuracy | test_f1_macro |
| --- | --- | --- |
| ens_three_nns|kept=36 | 0.8733 | 0.8744 |
| ens_small_deep|kept=24 | 0.8700 | 0.8711 |
| ens_medium_deep|kept=24 | 0.8700 | 0.8709 |

## round4_seed_sensitivity

| variant | test_accuracy | test_f1_macro |
| --- | --- | --- |
| nn_medium_rs=59 | 0.8833 | 0.8845 |
| nn_medium_rs=105 | 0.8800 | 0.8816 |
| nn_medium_rs=45 | 0.8767 | 0.8779 |
| nn_medium_rs=120 | 0.8500 | 0.8514 |

## round4_single_baseline

| variant | test_accuracy | test_f1_macro |
| --- | --- | --- |
| nn_medium | 0.8767 | 0.8779 |
| nn_small | 0.8667 | 0.8679 |
| nn_deep | 0.8467 | 0.8484 |
| knn | 0.7867 | 0.7886 |
| kmeanspp | 0.6267 | 0.6220 |

## round4_stacking

| variant | test_accuracy | test_f1_macro |
| --- | --- | --- |
| stack_top4_C=0.2 | 0.8567 | 0.8575 |
| stack_top4_C=0.6 | 0.8500 | 0.8506 |
| stack_top4_C=1.0 | 0.8500 | 0.8506 |
| stack_top4_C=2.0 | 0.8467 | 0.8471 |
