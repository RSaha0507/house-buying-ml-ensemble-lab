import argparse
import os
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.cluster._kmeans')

sns.set_theme(style='whitegrid')


class KMeansLabelModel:
    def __init__(self, n_clusters=9, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y, n_classes=3):
        self.model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=20, random_state=self.random_state)
        cluster_ids = self.model.fit_predict(X)
        self.classes_ = np.arange(n_classes)
        self.cluster_proba_ = np.zeros((self.n_clusters, n_classes), dtype=float)
        for c in range(self.n_clusters):
            idx = np.where(cluster_ids == c)[0]
            if len(idx) == 0:
                self.cluster_proba_[c] = np.ones(n_classes) / n_classes
            else:
                counts = np.bincount(y[idx], minlength=n_classes).astype(float)
                self.cluster_proba_[c] = (counts + 1.0) / (counts.sum() + n_classes)
        return self

    def predict_proba(self, X):
        return self.cluster_proba_[self.model.predict(X)]


class KMedoidsLabelModel:
    def __init__(self, n_clusters=9, random_state=42, max_iter=35):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y, n_classes=3):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        k = min(self.n_clusters, n_samples)
        medoid_idx = rng.choice(n_samples, size=k, replace=False)

        for _ in range(self.max_iter):
            d2 = ((X[:, None, :] - X[medoid_idx][None, :, :]) ** 2).sum(axis=2)
            assignments = d2.argmin(axis=1)
            changed = False
            new_medoid_idx = medoid_idx.copy()
            for c in range(k):
                pts = np.where(assignments == c)[0]
                if len(pts) == 0:
                    continue
                sub = X[pts]
                within = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2)
                best_local = pts[within.sum(axis=1).argmin()]
                if best_local != medoid_idx[c]:
                    changed = True
                    new_medoid_idx[c] = best_local
            medoid_idx = new_medoid_idx
            if not changed:
                break

        self.medoids_ = X[medoid_idx]
        self.classes_ = np.arange(n_classes)
        d2f = ((X[:, None, :] - self.medoids_[None, :, :]) ** 2).sum(axis=2)
        assignments = d2f.argmin(axis=1)

        self.cluster_proba_ = np.zeros((k, n_classes), dtype=float)
        for c in range(k):
            idx = np.where(assignments == c)[0]
            if len(idx) == 0:
                self.cluster_proba_[c] = np.ones(n_classes) / n_classes
            else:
                counts = np.bincount(y[idx], minlength=n_classes).astype(float)
                self.cluster_proba_[c] = (counts + 1.0) / (counts.sum() + n_classes)
        return self

    def predict_proba(self, X):
        d2 = ((X[:, None, :] - self.medoids_[None, :, :]) ** 2).sum(axis=2)
        return self.cluster_proba_[d2.argmin(axis=1)]


class SeedAveragedMLP:
    def __init__(self, base_params, seeds):
        self.base_params = dict(base_params)
        self.seeds = list(seeds)
        self.models = []
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y):
        self.models = []
        for seed in self.seeds:
            model = MLPClassifier(random_state=seed, **self.base_params)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict_proba(self, X):
        if not self.models:
            raise RuntimeError('SeedAveragedMLP must be fitted before prediction.')

        n_classes = len(self.classes_)
        avg = np.zeros((X.shape[0], n_classes), dtype=float)
        for model in self.models:
            proba = model.predict_proba(X)
            aligned = np.zeros((X.shape[0], n_classes), dtype=float)
            for col_idx, cls in enumerate(model.classes_):
                aligned[:, int(cls)] = proba[:, col_idx]
            avg += aligned
        return avg / len(self.models)


def build_models(seed):
    return {
        'nn_small': SeedAveragedMLP(
            base_params={
                'hidden_layer_sizes': (24,),
                'alpha': 0.01,
                'learning_rate_init': 0.0012,
                'max_iter': 2200,
                'early_stopping': True,
                'validation_fraction': 0.2,
                'n_iter_no_change': 25,
            },
            seeds=[seed + 1, seed + 11, seed + 21, seed + 31, seed + 41],
        ),
        'nn_medium': MLPClassifier(hidden_layer_sizes=(44, 22), alpha=0.02, learning_rate_init=0.0007, max_iter=2600, early_stopping=True, validation_fraction=0.22, n_iter_no_change=20, random_state=seed + 3),
        'nn_deep': MLPClassifier(hidden_layer_sizes=(56, 28, 14), alpha=0.012, learning_rate_init=0.001, max_iter=2800, early_stopping=True, validation_fraction=0.2, n_iter_no_change=25, random_state=seed + 4),
        'knn': KNeighborsClassifier(n_neighbors=11, weights='distance'),
        'kmeanspp': KMeansLabelModel(n_clusters=9, random_state=seed + 5),
        'kmedoids': KMedoidsLabelModel(n_clusters=9, random_state=seed + 6, max_iter=35),
    }


def fit_model(model, X, y, n_classes):
    if isinstance(model, (KMeansLabelModel, KMedoidsLabelModel)):
        model.fit(X, y, n_classes=n_classes)
    else:
        model.fit(X, y)


def predict_proba_full(model, X, n_classes):
    proba = model.predict_proba(X)
    if proba.shape[1] == n_classes:
        return proba
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for i, cls in enumerate(getattr(model, 'classes_', np.arange(proba.shape[1]))):
        out[:, int(cls)] = proba[:, i]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum


def eval_row(y_true, y_pred, y_proba, n_classes):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': p,
        'recall_macro': r,
        'f1_macro': f1,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_proba, labels=np.arange(n_classes)),
        'roc_auc_macro_ovr': roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr'),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Export experiment visualizations and optionally archive them.')
    parser.add_argument('--base-dir', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--tag', default='', help='Snapshot tag for history archival, e.g. opt_round1_adaptive_ensemble')
    return parser.parse_args()


def archive_figures(figs_dir, results_dir, tag):
    if not tag:
        return None
    history_fig_dir = results_dir / 'history' / tag / 'figures'
    history_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig_path in sorted(figs_dir.glob('*.png')):
        shutil.copy2(fig_path, history_fig_dir / fig_path.name)
    return history_fig_dir


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    csv_dir = base_dir / 'csv'
    results_dir = base_dir / 'results'
    figs_dir = results_dir / 'figures'
    results_dir.mkdir(exist_ok=True)
    figs_dir.mkdir(exist_ok=True)

    train_df = pd.read_csv(csv_dir / 'house_buy_train.csv')
    cv_df = pd.read_csv(csv_dir / 'house_buy_cv.csv')
    test_df = pd.read_csv(csv_dir / 'house_buy_test.csv')

    feature_cols = [
        'buyer_income_lpa', 'house_price_lakh', 'loan_eligibility', 'credit_score',
        'down_payment_percent', 'existing_emi_lpa', 'employment_years', 'dependents',
        'property_location_score', 'employment_type'
    ]
    numeric_cols = [
        'buyer_income_lpa', 'house_price_lakh', 'credit_score', 'down_payment_percent',
        'existing_emi_lpa', 'employment_years', 'dependents', 'property_location_score'
    ]
    cat_cols = ['loan_eligibility', 'employment_type']

    label_to_int = {'no': 0, 'neutral': 1, 'yes': 2}
    int_to_label = {v: k for k, v in label_to_int.items()}
    class_order = np.array([0, 1, 2])
    class_names = [int_to_label[i] for i in class_order]
    n_classes = len(class_order)

    X_train_raw = train_df[feature_cols]
    y_train = train_df['can_buy'].map(label_to_int).to_numpy()
    X_cv_raw = cv_df[feature_cols]
    y_cv = cv_df['can_buy'].map(label_to_int).to_numpy()
    X_test_raw = test_df[feature_cols]
    y_test = test_df['can_buy'].map(label_to_int).to_numpy()

    pre = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])

    X_train = pre.fit_transform(X_train_raw)
    X_cv = pre.transform(X_cv_raw)
    X_test = pre.transform(X_test_raw)
    X_train = X_train.toarray() if hasattr(X_train, 'toarray') else np.asarray(X_train)
    X_cv = X_cv.toarray() if hasattr(X_cv, 'toarray') else np.asarray(X_cv)
    X_test = X_test.toarray() if hasattr(X_test, 'toarray') else np.asarray(X_test)

    splits = {
        'train': (X_train, y_train),
        'cv': (X_cv, y_cv),
        'test': (X_test, y_test),
    }

    base_models = build_models(seed=42)
    metrics_rows = []
    predictions = {name: {} for name in base_models}
    probabilities = {name: {} for name in base_models}

    for model_name, model in base_models.items():
        fit_model(model, X_train, y_train, n_classes)
        for split_name, (X_split, y_split) in splits.items():
            proba = predict_proba_full(model, X_split, n_classes)
            pred = proba.argmax(axis=1)
            metrics = eval_row(y_split, pred, proba, n_classes)
            metrics.update({'model': model_name, 'split': split_name})
            metrics_rows.append(metrics)
            predictions[model_name][split_name] = pred
            probabilities[model_name][split_name] = proba

    base_metrics_df = pd.DataFrame(metrics_rows)

    n_bootstrap = 14
    ensemble_top_k = 2
    ensemble_cv_threshold = 0.78
    rng = np.random.default_rng(123)
    proba_sum = {k: np.zeros((len(v[1]), n_classes), dtype=float) for k, v in splits.items()}
    weight_total = 0.0
    weight_records = []
    kept_learners = 0

    cv_rank = (
        base_metrics_df[base_metrics_df['split'] == 'cv']
        .sort_values('f1_macro', ascending=False)
    )
    selected_families = cv_rank.head(ensemble_top_k)['model'].tolist()

    for b in range(n_bootstrap):
        idx = rng.integers(0, len(y_train), size=len(y_train))
        X_boot = X_train[idx]
        y_boot = y_train[idx]
        models_b = build_models(seed=1000 + b * 17)

        for model_name in selected_families:
            model = models_b[model_name]
            fit_model(model, X_boot, y_boot, n_classes)
            cv_proba = predict_proba_full(model, X_cv, n_classes)
            cv_pred = cv_proba.argmax(axis=1)
            cv_f1 = f1_score(y_cv, cv_pred, average='macro')
            if cv_f1 < ensemble_cv_threshold:
                continue
            w = max(float(cv_f1) ** 2, 0.02)
            weight_total += w
            kept_learners += 1
            weight_records.append({'bootstrap_iter': b, 'model': model_name, 'vote_weight': w})

            for split_name, (X_split, _) in splits.items():
                proba_sum[split_name] += w * predict_proba_full(model, X_split, n_classes)

    if kept_learners == 0:
        for b in range(n_bootstrap):
            idx = rng.integers(0, len(y_train), size=len(y_train))
            X_boot = X_train[idx]
            y_boot = y_train[idx]
            models_b = build_models(seed=2000 + b * 17)
            for model_name in selected_families:
                model = models_b[model_name]
                fit_model(model, X_boot, y_boot, n_classes)
                cv_proba = predict_proba_full(model, X_cv, n_classes)
                cv_pred = cv_proba.argmax(axis=1)
                cv_f1 = f1_score(y_cv, cv_pred, average='macro')
                w = max(float(cv_f1), 0.05)
                weight_total += w
                kept_learners += 1
                weight_records.append({'bootstrap_iter': b, 'model': model_name, 'vote_weight': w})
                for split_name, (X_split, _) in splits.items():
                    proba_sum[split_name] += w * predict_proba_full(model, X_split, n_classes)

    ens_proba = {k: v / weight_total for k, v in proba_sum.items()}
    ens_pred = {k: p.argmax(axis=1) for k, p in ens_proba.items()}

    ens_rows = []
    for split_name, (_, y_split) in splits.items():
        metrics = eval_row(y_split, ens_pred[split_name], ens_proba[split_name], n_classes)
        metrics.update({'model': 'bootstrap_ensemble', 'split': split_name})
        ens_rows.append(metrics)

    all_metrics_df = pd.concat([base_metrics_df, pd.DataFrame(ens_rows)], ignore_index=True)
    plot_order = list(base_models.keys()) + ['bootstrap_ensemble']

    # 1) Accuracy comparison
    plt.figure(figsize=(13, 6))
    sns.barplot(data=all_metrics_df, x='model', y='accuracy', hue='split', order=plot_order)
    plt.title('Accuracy by Model and Split')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(figs_dir / 'accuracy_by_model_split.png', dpi=160)
    plt.close()

    # 2) Macro-F1 comparison
    plt.figure(figsize=(13, 6))
    sns.barplot(data=all_metrics_df, x='model', y='f1_macro', hue='split', order=plot_order)
    plt.title('Macro-F1 by Model and Split')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(figs_dir / 'f1_by_model_split.png', dpi=160)
    plt.close()

    # 3) Confusion matrices
    best_single = (
        all_metrics_df[(all_metrics_df['split'] == 'cv') & (all_metrics_df['model'] != 'bootstrap_ensemble')]
        .sort_values('f1_macro', ascending=False)
        .iloc[0]['model']
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_single = confusion_matrix(y_test, predictions[best_single]['test'], labels=class_order)
    cm_ens = confusion_matrix(y_test, ens_pred['test'], labels=class_order)

    sns.heatmap(cm_single, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Best single ({best_single}) - Test')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Bootstrap ensemble - Test')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(figs_dir / 'confusion_matrix_best_single_vs_ensemble.png', dpi=160)
    plt.close()

    # 4) ROC curves for ensemble
    y_test_bin = label_binarize(y_test, classes=class_order)
    p_test = ens_proba['test']

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], p_test[:, i])
        auc_i = roc_auc_score(y_test_bin[:, i], p_test[:, i])
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_i:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.title('Bootstrap Ensemble ROC Curves (Test, OvR)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(figs_dir / 'roc_curves_ensemble_test.png', dpi=160)
    plt.close()

    # 5) Average vote weight by model type
    weight_df = pd.DataFrame(weight_records)
    mean_weight = weight_df.groupby('model', as_index=False)['vote_weight'].mean().sort_values('vote_weight', ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=mean_weight, x='model', y='vote_weight', hue='model', palette='viridis', legend=False)
    plt.title('Average Ensemble Vote Weight by Model Type')
    plt.xlabel('Model')
    plt.ylabel('Average Vote Weight')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figs_dir / 'ensemble_vote_weights.png', dpi=160)
    plt.close()

    print('Ensemble families used for bagging:', selected_families)
    print('Kept ensemble learners:', kept_learners)
    print('Saved figures to:', figs_dir)
    for p in sorted(figs_dir.glob('*.png')):
        print('-', p.name)
    archived = archive_figures(figs_dir, results_dir, args.tag)
    if archived is not None:
        print('Archived figures to:', archived)


if __name__ == '__main__':
    main()
