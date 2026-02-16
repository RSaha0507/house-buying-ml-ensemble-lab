# Concepts, Mathematics, Statistics, and Why Each Method Was Used

## 1) Problem formulation

- Task type: multi-class classification.
- Target classes: `no`, `neutral`, `yes`.
- Input features: mixed numeric + categorical.

Goal:
- Learn a function
$$
  \( f(x) \to \{0,1,2\} \)
$$
   that predicts buying feasibility.

## 2) Preprocessing

### Standardization (numeric features)

For each numeric feature:

$$
z = \frac{x - \mu}{\sigma}
$$

Why:
- Puts features on comparable scale.
- Stabilizes gradient-based training (MLPs).
- Prevents distance-based models (KNN, clustering) from being dominated by large-scale features.

### One-hot encoding (categorical features)

Category value is converted into binary indicator columns.

Why:
- Converts non-numeric categories into model-usable numeric vectors.
- Avoids imposing artificial ordering among categories.

## 3) Base model families

### Neural Networks (MLPClassifier)

Core form (layer-wise):

$$
h^{(l+1)} = \sigma(W^{(l)}h^{(l)} + b^{(l)})
$$

$$
\hat{y} = \text{softmax}(W^{(L)}h^{(L)} + b^{(L)})
$$

Why:
- Captures nonlinear interactions between affordability, credit, and debt signals.
- Can model soft class boundaries in `no/neutral/yes`.

Regularization controls used:
- `alpha` (L2 penalty), early stopping, validation fraction, architecture sizing.

### KNN

Prediction from nearest neighbors in transformed feature space (distance-weighted voting).

Why:
- Strong local-pattern baseline.
- Useful comparator for bias/variance behavior.

### KMeans++ / K-Medoids label models

- Unsupervised clustering first.
- Then cluster-level class probabilities estimated from class frequencies.

Why:
- Adds geometric/cluster-structure viewpoint.
- Increases diversity for ensembling.

## 4) Bagging and bootstrap sampling

Bootstrap:
- Sample with replacement from training data to create multiple training replicas.

Bagging prediction (soft voting):
$$
\hat{p}(y=c \mid x) = \frac{1}{\sum w_m}\sum_m w_m p_m(y=c \mid x)
$$

Why:
- Reduces variance.
- Makes prediction less sensitive to one specific sample split.

## 5) Ensemble weighting strategy

Used:
- CV quality gate (only keep learners with CV macro-F1 above threshold).
- Weight by squared CV macro-F1:
$$
w_m = \max(\text{F1}_{cv,m}^2,\ \epsilon)
$$

Why:
- Promotes reliably strong models.
- Penalizes weak/noisy learners.
- Balances diversity with quality control.

## 6) Metrics used and mathematical meaning

### Accuracy
$$
\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}
$$

### Precision, Recall, F1 (macro)

For each class \(k\):
$$
Precision_k = \frac{TP_k}{TP_k + FP_k}, \quad
Recall_k = \frac{TP_k}{TP_k + FN_k}
$$
$$
F1_k = \frac{2 \cdot Precision_k \cdot Recall_k}{Precision_k + Recall_k}
$$

Macro-F1:
$$
F1_{macro} = \frac{1}{K}\sum_{k=1}^{K}F1_k
$$

Why macro:
- Gives equal importance to all classes.
- Better for multi-class balance than plain accuracy.

### Balanced Accuracy
$$
\text{Balanced Accuracy} = \frac{1}{K}\sum_{k=1}^{K} Recall_k
$$

Why:
- Handles class-imbalance sensitivity (even if mild).

### Log Loss

Penalizes miscalibrated probabilities:
$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}\mathbb{1}(y_i=k)\log p_{ik}
$$

Why:
- Measures confidence quality, not just hard labels.

### ROC-AUC (OvR macro)

- One-vs-rest ROC for each class, then macro-average AUC.

Why:
- Captures ranking quality across decision thresholds.

## 7) Bias-variance perspective

- Overfitting signal: train metrics much higher than CV/test.
- Controls applied:
  - stronger regularization
  - early stopping
  - architecture tuning
  - bootstrap ensembling
  - larger and more varied dataset

Why:
- Aim is lower generalization gap, not only high train score.

## 8) Why the promoted profile was chosen

`opt_round3_medium_tuned` was kept as promoted profile because:
- It gave the best stable CV/test balance.
- Some candidates beat test slightly but lost CV robustness.
- Decision policy favored reliability over one-off peak test score.

