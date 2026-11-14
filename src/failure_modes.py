"""Multi-label failure-mode training for the UCI AI4I 2020 dataset."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import FEATURES, FAILURE_MODES, TARGET, load_ai4i
from src.evaluation import choose_threshold_with_constraints, false_discovery_rate


def make_mode_pipeline():
    """Create an independently fitted binary classifier for one failure mode."""
    numeric = [column for column in FEATURES if column != "Type"]
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "type",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["Type"],
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        [("preprocessor", preprocessing), ("classifier", classifier)]
    )


def evaluate_mode(y_true, probabilities, threshold, min_recall):
    """Evaluate one mode and report whether its validation recall floor was met."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    has_both_classes = np.unique(y_true).size == 2
    return {
        "threshold": float(threshold),
        "support": int(y_true.sum()),
        "predicted_positives": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_discovery_rate": float(false_discovery_rate(y_true, predictions)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if has_both_classes
            else None
        ),
        "average_precision": (
            float(average_precision_score(y_true, probabilities))
            if y_true.sum() > 0
            else None
        ),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "validation_recall_floor": float(min_recall),
    }


def mode_is_available_for_alerting(selection):
    """Determine alerting eligibility from validation diagnostics only."""
    return selection.get("status") == "enabled"


def train_failure_modes(output_dir="artifacts", min_recall=0.30, max_fdr=0.50):
    """Fit five mode-specific models and tune each threshold on validation data."""
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in the interval (0, 1].")

    frame = load_ai4i()
    X = frame[FEATURES]
    overall_y = frame[TARGET]

    X_train, X_test, _, _ = train_test_split(
        X,
        overall_y,
        test_size=0.20,
        stratify=overall_y,
        random_state=42,
    )
    X_fit, X_val, _, _ = train_test_split(
        X_train,
        overall_y.loc[X_train.index],
        test_size=0.20,
        stratify=overall_y.loc[X_train.index],
        random_state=42,
    )

    models = {}
    thresholds = {}
    metrics = {}

    for mode in FAILURE_MODES:
        y = frame[mode]
        y_fit = y.loc[X_fit.index]
        y_val = y.loc[X_val.index]
        y_test = y.loc[X_test.index]

        if y_fit.nunique() < 2:
            raise ValueError(
                f"{mode} has only one class in the training split; "
                "cannot fit a binary classifier."
            )

        validation_model = make_mode_pipeline().fit(X_fit, y_fit)
        validation_probabilities = validation_model.predict_proba(X_val)[:, 1]
        threshold, selection = choose_threshold_with_constraints(
            y_val.to_numpy(),
            validation_probabilities,
            min_recall=min_recall,
            max_fdr=max_fdr,
        )

        test_probabilities = validation_model.predict_proba(X_test)[:, 1]
        # Unavailable modes remain score-only; they do not emit alerts.
        mode_metrics = evaluate_mode(
            y_test,
            test_probabilities,
            threshold if threshold is not None else 1.01,
            min_recall,
        )
        validation_threshold = threshold
        # Alerting eligibility is determined exclusively on validation data.
        # The held-out test split is used only to report final metrics.
        available_for_alerting = mode_is_available_for_alerting(selection)
        deployment_threshold = (
            validation_threshold if available_for_alerting else None
        )
        mode_metrics["validation_threshold"] = validation_threshold
        mode_metrics["threshold"] = deployment_threshold
        mode_metrics["status"] = (
            "enabled" if available_for_alerting
            else "unavailable_no_validation_threshold"
        )
        mode_metrics["available_for_alerting"] = available_for_alerting
        mode_metrics["max_validation_fdr"] = float(max_fdr)
        mode_metrics["max_test_fdr"] = float(max_fdr)
        mode_metrics["test_recall_floor_met"] = bool(
            mode_metrics["recall"] >= min_recall
        )
        mode_metrics["test_fdr_ceiling_met"] = bool(
            mode_metrics["false_discovery_rate"] <= max_fdr
        )
        mode_metrics["test_operating_constraints_met"] = (
            mode_metrics["test_recall_floor_met"]
            and mode_metrics["test_fdr_ceiling_met"]
        )
        mode_metrics.update(selection)
        # Restore deployment-facing status after adding validation diagnostics.
        mode_metrics["status"] = (
            "enabled" if available_for_alerting
            else "unavailable_no_validation_threshold"
        )

        # Keep the exact estimator used for validation threshold selection
        # and test evaluation as the deployed estimator. Refitting on a larger
        # split here would change its score distribution and invalidate the
        # threshold/test metrics associated with this artifact.
        models[mode] = validation_model
        thresholds[mode] = (
            float(deployment_threshold)
            if deployment_threshold is not None
            else None
        )
        metrics[mode] = mode_metrics

    artifact = {
        "models": models,
        "thresholds": thresholds,
        "features": list(FEATURES),
        "failure_modes": list(FAILURE_MODES),
        "metrics": metrics,
        "dataset": "UCI AI4I 2020",
        "target": "Multi-label failure-mode indicators",
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination / "failure_modes_pipeline.joblib")
    (destination / "failure_modes_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    (destination / "failure_modes_schema.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "failure_modes": FAILURE_MODES,
                "thresholds": thresholds,
                "min_validation_recall": min_recall,
                "max_validation_fdr": max_fdr,
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2))
    print(f"Artifact: {destination / 'failure_modes_pipeline.joblib'}")
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--min-recall", type=float, default=0.30)
    parser.add_argument("--max-fdr", type=float, default=0.50)
    args = parser.parse_args()
    train_failure_modes(args.output_dir, args.min_recall, args.max_fdr)


if __name__ == "__main__":
    main()
