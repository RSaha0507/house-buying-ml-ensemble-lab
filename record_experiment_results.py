from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")


class KMeansLabelModel:
    def __init__(self, n_clusters: int = 9, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int = 3):
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=20,
            random_state=self.random_state,
        )
        cluster_ids = self.model.fit_predict(X)
        self.classes_ = np.arange(n_classes)
        self.cluster_proba_ = np.zeros((self.n_clusters, n_classes), dtype=float)
        for cluster_id in range(self.n_clusters):
            idx = np.where(cluster_ids == cluster_id)[0]
            if len(idx) == 0:
                self.cluster_proba_[cluster_id] = np.ones(n_classes) / n_classes
            else:
                counts = np.bincount(y[idx], minlength=n_classes).astype(float)
                self.cluster_proba_[cluster_id] = (counts + 1.0) / (counts.sum() + n_classes)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        nearest = self.model.predict(X)
        return self.cluster_proba_[nearest]


class KMedoidsLabelModel:
    def __init__(self, n_clusters: int = 9, random_state: int = 42, max_iter: int = 35):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int = 3):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        k = min(self.n_clusters, n_samples)
        medoid_idx = rng.choice(n_samples, size=k, replace=False)

        for _ in range(self.max_iter):
            d2 = ((X[:, None, :] - X[medoid_idx][None, :, :]) ** 2).sum(axis=2)
            assignments = d2.argmin(axis=1)
            changed = False
            new_medoid_idx = medoid_idx.copy()

            for cluster_id in range(k):
                pts = np.where(assignments == cluster_id)[0]
                if len(pts) == 0:
                    continue
                cluster_pts = X[pts]
                within = ((cluster_pts[:, None, :] - cluster_pts[None, :, :]) ** 2).sum(axis=2)
                best_local = pts[within.sum(axis=1).argmin()]
                if best_local != medoid_idx[cluster_id]:
                    new_medoid_idx[cluster_id] = best_local
                    changed = True

            medoid_idx = new_medoid_idx
            if not changed:
                break

        self.medoids_ = X[medoid_idx]
        self.classes_ = np.arange(n_classes)
        d2 = ((X[:, None, :] - self.medoids_[None, :, :]) ** 2).sum(axis=2)
        assignments = d2.argmin(axis=1)

        self.cluster_proba_ = np.zeros((k, n_classes), dtype=float)
        for cluster_id in range(k):
            idx = np.where(assignments == cluster_id)[0]
            if len(idx) == 0:
                self.cluster_proba_[cluster_id] = np.ones(n_classes) / n_classes
            else:
                counts = np.bincount(y[idx], minlength=n_classes).astype(float)
                self.cluster_proba_[cluster_id] = (counts + 1.0) / (counts.sum() + n_classes)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[:, None, :] - self.medoids_[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        return self.cluster_proba_[nearest]


class SeedAveragedMLP:
    def __init__(self, base_params: dict, seeds: list[int]):
        self.base_params = dict(base_params)
        self.seeds = list(seeds)
        self.models: list[MLPClassifier] = []
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.models = []
        for seed in self.seeds:
            model = MLPClassifier(random_state=seed, **self.base_params)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("SeedAveragedMLP must be fitted before prediction.")

        n_classes = len(self.classes_)
        avg = np.zeros((X.shape[0], n_classes), dtype=float)
        for model in self.models:
            proba = model.predict_proba(X)
            aligned = np.zeros((X.shape[0], n_classes), dtype=float)
            for col_idx, cls in enumerate(model.classes_):
                aligned[:, int(cls)] = proba[:, col_idx]
            avg += aligned
        return avg / len(self.models)


def build_models(seed: int):
    return {
        "nn_small": SeedAveragedMLP(
            base_params={
                "hidden_layer_sizes": (24,),
                "alpha": 0.01,
                "learning_rate_init": 0.0012,
                "max_iter": 2200,
                "early_stopping": True,
                "validation_fraction": 0.2,
                "n_iter_no_change": 25,
            },
            seeds=[seed + 1, seed + 11, seed + 21, seed + 31, seed + 41],
        ),
        "nn_medium": MLPClassifier(
            hidden_layer_sizes=(44, 22),
            alpha=0.02,
            learning_rate_init=0.0007,
            max_iter=2600,
            early_stopping=True,
            validation_fraction=0.22,
            n_iter_no_change=20,
            random_state=seed + 3,
        ),
        "nn_deep": MLPClassifier(
            hidden_layer_sizes=(56, 28, 14),
            alpha=0.012,
            learning_rate_init=0.001,
            max_iter=2800,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=25,
            random_state=seed + 4,
        ),
        "knn": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "kmeanspp": KMeansLabelModel(n_clusters=9, random_state=seed + 5),
        "kmedoids": KMedoidsLabelModel(n_clusters=9, random_state=seed + 6, max_iter=35),
    }


def fit_model(model, X: np.ndarray, y: np.ndarray, n_classes: int):
    if isinstance(model, (KMeansLabelModel, KMedoidsLabelModel)):
        model.fit(X, y, n_classes=n_classes)
    else:
        model.fit(X, y)


def predict_proba_full(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == n_classes:
        return proba

    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    model_classes = getattr(model, "classes_", np.arange(proba.shape[1]))
    for col_idx, cls in enumerate(model_classes):
        out[:, int(cls)] = proba[:, col_idx]

    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_classes: int) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=np.arange(n_classes)),
        "roc_auc_macro_ovr": roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"),
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record experiment metrics and append to results log.")
    parser.add_argument("--base-dir", default=r"C:\Users\91960\house_pricing_nn")
    parser.add_argument("--tag", default="manual_run")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(base_dir / "house_buy_train.csv")
    cv_df = pd.read_csv(base_dir / "house_buy_cv.csv")
    test_df = pd.read_csv(base_dir / "house_buy_test.csv")

    feature_cols = [
        "buyer_income_lpa",
        "house_price_lakh",
        "loan_eligibility",
        "credit_score",
        "down_payment_percent",
        "existing_emi_lpa",
        "employment_years",
        "dependents",
        "property_location_score",
        "employment_type",
    ]
    numeric_cols = [
        "buyer_income_lpa",
        "house_price_lakh",
        "credit_score",
        "down_payment_percent",
        "existing_emi_lpa",
        "employment_years",
        "dependents",
        "property_location_score",
    ]
    cat_cols = ["loan_eligibility", "employment_type"]

    label_to_int = {"no": 0, "neutral": 1, "yes": 2}
    n_classes = 3

    X_train_raw = train_df[feature_cols]
    y_train = train_df["can_buy"].map(label_to_int).to_numpy()
    X_cv_raw = cv_df[feature_cols]
    y_cv = cv_df["can_buy"].map(label_to_int).to_numpy()
    X_test_raw = test_df[feature_cols]
    y_test = test_df["can_buy"].map(label_to_int).to_numpy()

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train = pre.fit_transform(X_train_raw)
    X_cv = pre.transform(X_cv_raw)
    X_test = pre.transform(X_test_raw)
    X_train = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
    X_cv = X_cv.toarray() if hasattr(X_cv, "toarray") else np.asarray(X_cv)
    X_test = X_test.toarray() if hasattr(X_test, "toarray") else np.asarray(X_test)

    splits = {
        "train": (X_train, y_train),
        "cv": (X_cv, y_cv),
        "test": (X_test, y_test),
    }

    rows = []
    base_models = build_models(seed=42)
    for model_name, model in base_models.items():
        fit_model(model, X_train, y_train, n_classes)
        for split_name, (X_split, y_split) in splits.items():
            split_proba = predict_proba_full(model, X_split, n_classes)
            split_pred = split_proba.argmax(axis=1)
            row = evaluate_metrics(y_split, split_pred, split_proba, n_classes)
            row["run_group"] = "single_models"
            row["model"] = model_name
            row["split"] = split_name
            rows.append(row)

    n_bootstrap = 14
    ensemble_top_k = 2
    ensemble_cv_threshold = 0.78
    rng = np.random.default_rng(123)
    proba_sums = {k: np.zeros((v[1].shape[0], n_classes), dtype=float) for k, v in splits.items()}
    weight_total = 0.0
    kept_learners = 0

    single_metrics_df = pd.DataFrame(rows)
    cv_rank = (
        single_metrics_df[single_metrics_df["split"] == "cv"]
        .sort_values("f1_macro", ascending=False)
    )
    selected_families = cv_rank.head(ensemble_top_k)["model"].tolist()

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, len(y_train), size=len(y_train))
        X_boot = X_train[sample_idx]
        y_boot = y_train[sample_idx]
        models_b = build_models(seed=1000 + b * 17)

        for model_name in selected_families:
            model = models_b[model_name]
            fit_model(model, X_boot, y_boot, n_classes)
            cv_proba = predict_proba_full(model, X_cv, n_classes)
            cv_pred = cv_proba.argmax(axis=1)
            cv_f1 = f1_score(y_cv, cv_pred, average="macro")
            if cv_f1 < ensemble_cv_threshold:
                continue
            weight = max(float(cv_f1) ** 2, 0.02)
            weight_total += weight
            kept_learners += 1
            for split_name, (X_split, _) in splits.items():
                proba_sums[split_name] += weight * predict_proba_full(model, X_split, n_classes)

    if kept_learners == 0:
        # Fallback: if quality gate is too strict, keep the same model families with linear weighting.
        for b in range(n_bootstrap):
            sample_idx = rng.integers(0, len(y_train), size=len(y_train))
            X_boot = X_train[sample_idx]
            y_boot = y_train[sample_idx]
            models_b = build_models(seed=2000 + b * 17)
            for model_name in selected_families:
                model = models_b[model_name]
                fit_model(model, X_boot, y_boot, n_classes)
                cv_proba = predict_proba_full(model, X_cv, n_classes)
                cv_pred = cv_proba.argmax(axis=1)
                cv_f1 = f1_score(y_cv, cv_pred, average="macro")
                weight = max(float(cv_f1), 0.05)
                weight_total += weight
                kept_learners += 1
                for split_name, (X_split, _) in splits.items():
                    proba_sums[split_name] += weight * predict_proba_full(model, X_split, n_classes)

    for split_name, (_, y_true) in splits.items():
        split_proba = proba_sums[split_name] / weight_total
        split_pred = split_proba.argmax(axis=1)
        row = evaluate_metrics(y_true, split_pred, split_proba, n_classes)
        row["run_group"] = "ensemble"
        row["model"] = "bootstrap_ensemble"
        row["split"] = split_name
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[
        [
            "run_group",
            "model",
            "split",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "balanced_accuracy",
            "log_loss",
            "roc_auc_macro_ovr",
        ]
    ]

    timestamp = datetime.now()
    stamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = results_dir / f"metrics_snapshot_{stamp}.csv"
    metrics_df.to_csv(csv_path, index=False)

    sorted_main = metrics_df.sort_values(["split", "f1_macro"], ascending=[True, False]).reset_index(drop=True)
    split_sizes = {
        "train": len(train_df),
        "cv": len(cv_df),
        "test": len(test_df),
    }
    train_class_counts = train_df["can_buy"].value_counts().to_dict()

    report_path = results_dir / "experiment_results_log.md"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8").rstrip() + "\n\n"
    else:
        report = "# House Buying Model Results Log\n\n"

    report += f"## Snapshot: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({args.tag})\n\n"
    report += f"- Dataset split sizes: train={split_sizes['train']}, cv={split_sizes['cv']}, test={split_sizes['test']}\n"
    report += f"- Train class distribution: {train_class_counts}\n"
    report += (
        f"- Ensemble setup: bootstrap={n_bootstrap}, selected_families={selected_families}, "
        f"cv_threshold={ensemble_cv_threshold}, weight=cv_f1^2, kept_learners={kept_learners}\n"
    )
    report += f"- Metrics CSV: `results/{csv_path.name}`\n"
    report += "- Figures:\n"
    report += "  - `results/figures/accuracy_by_model_split.png`\n"
    report += "  - `results/figures/f1_by_model_split.png`\n"
    report += "  - `results/figures/confusion_matrix_best_single_vs_ensemble.png`\n"
    report += "  - `results/figures/roc_curves_ensemble_test.png`\n"
    report += "  - `results/figures/ensemble_vote_weights.png`\n\n"
    if args.tag:
        report += f"- Archived figures: `results/history/{args.tag}/figures`\n\n"

    report += "### Accuracy and Macro-F1\n\n"
    report += markdown_table(
        sorted_main[["model", "split", "accuracy", "f1_macro"]],
        ["model", "split", "accuracy", "f1_macro"],
    )
    report += "\n\n### Full metric table\n\n"
    report += markdown_table(
        sorted_main,
        [
            "run_group",
            "model",
            "split",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "balanced_accuracy",
            "log_loss",
            "roc_auc_macro_ovr",
        ],
    )
    report += "\n"

    report_path.write_text(report, encoding="utf-8")
    print("Saved metrics snapshot:", csv_path)
    print("Updated report:", report_path)


if __name__ == "__main__":
    main()
