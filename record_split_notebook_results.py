from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from record_experiment_results import (
    build_models,
    evaluate_metrics,
    fit_model,
    markdown_table,
    predict_proba_full,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record experiment metrics for split model notebooks and append to results log."
    )
    parser.add_argument("--base-dir", default=r"C:\Users\91960\house_pricing_nn")
    parser.add_argument("--tag", default="split_notebook_runs")
    return parser.parse_args()


def load_data(base_dir: Path):
    csv_dir = base_dir / "csv"
    train_df = pd.read_csv(csv_dir / "house_buy_train.csv")
    cv_df = pd.read_csv(csv_dir / "house_buy_cv.csv")
    test_df = pd.read_csv(csv_dir / "house_buy_test.csv")
    return train_df, cv_df, test_df


def prepare_splits(train_df: pd.DataFrame, cv_df: pd.DataFrame, test_df: pd.DataFrame):
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
    return splits, n_classes


def evaluate_single_model_notebook(
    notebook_name: str,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    seed: int = 42,
) -> list[dict]:
    models = build_models(seed=seed)
    model = models[model_name]
    fit_model(model, X_train, y_train, n_classes)

    rows: list[dict] = []
    for split_name, (X_split, y_split) in splits.items():
        split_proba = predict_proba_full(model, X_split, n_classes)
        split_pred = split_proba.argmax(axis=1)
        row = evaluate_metrics(y_split, split_pred, split_proba, n_classes)
        row["run_group"] = "split_notebooks_single"
        row["notebook"] = notebook_name
        row["model"] = model_name
        row["split"] = split_name
        rows.append(row)
    return rows


def evaluate_bootstrap_notebook(
    notebook_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    n_classes: int,
) -> tuple[list[dict], dict]:
    base_models = build_models(seed=42)
    base_cv_rows = []
    for model_name, model in base_models.items():
        fit_model(model, X_train, y_train, n_classes)
        cv_proba = predict_proba_full(model, X_cv, n_classes)
        cv_pred = cv_proba.argmax(axis=1)
        row = evaluate_metrics(y_cv, cv_pred, cv_proba, n_classes)
        row["model"] = model_name
        base_cv_rows.append(row)

    base_cv_df = pd.DataFrame(base_cv_rows).sort_values("f1_macro", ascending=False)
    selected_families = base_cv_df.head(2)["model"].tolist()

    n_bootstrap = 14
    ensemble_cv_threshold = 0.78
    rng = np.random.default_rng(123)
    proba_sums = {k: np.zeros((v[1].shape[0], n_classes), dtype=float) for k, v in splits.items()}
    weight_total = 0.0
    kept_learners = 0

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
        raise RuntimeError("No ensemble learners passed the CV threshold for split notebook evaluation.")

    rows: list[dict] = []
    for split_name, (_, y_true) in splits.items():
        split_proba = proba_sums[split_name] / weight_total
        split_pred = split_proba.argmax(axis=1)
        row = evaluate_metrics(y_true, split_pred, split_proba, n_classes)
        row["run_group"] = "split_notebooks_ensemble"
        row["notebook"] = notebook_name
        row["model"] = "bootstrap_ensemble"
        row["split"] = split_name
        rows.append(row)

    ensemble_meta = {
        "n_bootstrap": n_bootstrap,
        "selected_families": selected_families,
        "cv_threshold": ensemble_cv_threshold,
        "kept_learners": kept_learners,
        "weight_rule": "cv_f1^2",
    }
    return rows, ensemble_meta


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    csv_dir = base_dir / "csv"
    results_dir = base_dir / "results"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df, cv_df, test_df = load_data(base_dir)
    splits, n_classes = prepare_splits(train_df, cv_df, test_df)
    X_train, y_train = splits["train"]
    X_cv, y_cv = splits["cv"]

    rows: list[dict] = []
    rows.extend(
        evaluate_single_model_notebook(
            notebook_name="notebooks/house_buying_knn_experiment.ipynb",
            model_name="knn",
            X_train=X_train,
            y_train=y_train,
            splits=splits,
            n_classes=n_classes,
        )
    )
    rows.extend(
        evaluate_single_model_notebook(
            notebook_name="notebooks/house_buying_kmedoids_experiment.ipynb",
            model_name="kmedoids",
            X_train=X_train,
            y_train=y_train,
            splits=splits,
            n_classes=n_classes,
        )
    )
    rows.extend(
        evaluate_single_model_notebook(
            notebook_name="notebooks/house_buying_kmeanspp_experiment.ipynb",
            model_name="kmeanspp",
            X_train=X_train,
            y_train=y_train,
            splits=splits,
            n_classes=n_classes,
        )
    )
    for nn_model in ["nn_small", "nn_medium", "nn_deep"]:
        rows.extend(
            evaluate_single_model_notebook(
                notebook_name="notebooks/house_buying_three_nns_experiment.ipynb",
                model_name=nn_model,
                X_train=X_train,
                y_train=y_train,
                splits=splits,
                n_classes=n_classes,
            )
        )

    ensemble_rows, ensemble_meta = evaluate_bootstrap_notebook(
        notebook_name="notebooks/house_buying_bootstrap_ensemble_experiment.ipynb",
        X_train=X_train,
        y_train=y_train,
        X_cv=X_cv,
        y_cv=y_cv,
        splits=splits,
        n_classes=n_classes,
    )
    rows.extend(ensemble_rows)

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[
        [
            "run_group",
            "notebook",
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
    csv_path = csv_dir / f"metrics_snapshot_split_notebooks_{stamp}.csv"
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
    report += "- Scope: split notebook experiment recording (5 model-specific notebooks)\n"
    report += f"- Dataset split sizes: train={split_sizes['train']}, cv={split_sizes['cv']}, test={split_sizes['test']}\n"
    report += f"- Train class distribution: {train_class_counts}\n"
    report += "- Notebooks:\n"
    report += "  - `notebooks/house_buying_knn_experiment.ipynb`\n"
    report += "  - `notebooks/house_buying_kmedoids_experiment.ipynb`\n"
    report += "  - `notebooks/house_buying_kmeanspp_experiment.ipynb`\n"
    report += "  - `notebooks/house_buying_bootstrap_ensemble_experiment.ipynb`\n"
    report += "  - `notebooks/house_buying_three_nns_experiment.ipynb`\n"
    report += (
        f"- Bootstrap ensemble setup: bootstrap={ensemble_meta['n_bootstrap']}, "
        f"selected_families={ensemble_meta['selected_families']}, "
        f"cv_threshold={ensemble_meta['cv_threshold']}, "
        f"weight={ensemble_meta['weight_rule']}, "
        f"kept_learners={ensemble_meta['kept_learners']}\n"
    )
    report += f"- Metrics CSV: `csv/{csv_path.name}`\n\n"

    report += "### Accuracy and Macro-F1\n\n"
    report += markdown_table(
        sorted_main[["notebook", "model", "split", "accuracy", "f1_macro"]],
        ["notebook", "model", "split", "accuracy", "f1_macro"],
    )
    report += "\n\n### Full metric table\n\n"
    report += markdown_table(
        sorted_main,
        [
            "run_group",
            "notebook",
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
    print("Saved split notebook metrics snapshot:", csv_path)
    print("Updated report:", report_path)


if __name__ == "__main__":
    main()
