"""Metrics and validation-only threshold selection."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def false_discovery_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    predicted_positive = int(cm[0, 1] + cm[1, 1])
    return float(cm[0, 1] / predicted_positive) if predicted_positive else 0.0

def choose_threshold(y_true, probabilities, min_recall=0.50):
    y, p = np.asarray(y_true), np.asarray(probabilities)
    candidates = np.unique(np.r_[np.linspace(.01, .99, 99), p])
    valid = []
    for t in candidates:
        pred = (p >= t).astype(int)
        if pred.sum() and recall_score(y, pred, zero_division=0) >= min_recall:
            valid.append((false_discovery_rate(y, pred), -recall_score(y, pred), float(t)))
    if valid: return min(valid)[2]
    # Fallback: maximize F1 among thresholds that predict at least one positive.
    scored = [(f1_score(y, (p >= t).astype(int), zero_division=0), float(t))
              for t in candidates if (p >= t).sum()]
    return max(scored)[1] if scored else .5

def evaluate(y_true, probabilities, threshold):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "false_discovery_rate": false_discovery_rate(y_true, pred),
            "false_positive_rate": float(fp/(fp+tn)) if fp+tn else 0.0,
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "average_precision": float(average_precision_score(y_true, probabilities)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def choose_threshold_with_constraints(
    y_true, probabilities, min_recall=0.30, max_fdr=0.50
):
    """Choose a threshold satisfying both recall and FDR constraints.

    Returns (threshold, diagnostics). A None threshold means no defensible
    operating point was found; callers must not silently fall back to an
    all-positive or F1-only classifier.
    """
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in the interval (0, 1].")
    if not 0 <= max_fdr < 1:
        raise ValueError("max_fdr must be in the interval [0, 1).")

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and probabilities must be equal-length 1D arrays.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1].")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels.")
    if y.sum() == 0:
        return None, {
            "status": "unavailable_no_positive_validation_examples",
            "validation_recall": 0.0,
            "validation_fdr": None,
            "validation_precision": None,
            "validation_predicted_positives": 0,
            "validation_recall_floor_met": False,
            "validation_fdr_ceiling_met": False,
        }

    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 99), p])
    valid = []
    for threshold in candidates:
        if threshold <= 0:
            continue
        predicted = (p >= threshold).astype(int)
        predicted_count = int(predicted.sum())
        if not predicted_count:
            continue
        recall = float(recall_score(y, predicted, zero_division=0))
        fdr = float(false_discovery_rate(y, predicted))
        if recall >= min_recall and fdr <= max_fdr:
            valid.append((fdr, -recall, -float(precision_score(
                y, predicted, zero_division=0
            )), float(threshold), predicted_count))

    if not valid:
        # Report the best-recall operating point for diagnosis, but do not
        # enable a model that violates either operating constraint.
        diagnostic_candidates = []
        for threshold in candidates:
            if threshold <= 0:
                continue
            predicted = (p >= threshold).astype(int)
            if not predicted.sum():
                continue
            recall = float(recall_score(y, predicted, zero_division=0))
            fdr = float(false_discovery_rate(y, predicted))
            diagnostic_candidates.append((recall, -fdr, float(threshold), int(predicted.sum())))
        if diagnostic_candidates:
            recall, neg_fdr, _, predicted_count = max(diagnostic_candidates)
            fdr = -neg_fdr
        else:
            recall, fdr, predicted_count = 0.0, None, 0
        return None, {
            "status": "unavailable_no_threshold_meets_constraints",
            "validation_recall": recall,
            "validation_fdr": fdr,
            "validation_precision": (1.0 - fdr) if fdr is not None else None,
            "validation_predicted_positives": predicted_count,
            "validation_recall_floor_met": recall >= min_recall,
            "validation_fdr_ceiling_met": fdr is not None and fdr <= max_fdr,
        }

    fdr, neg_recall, neg_precision, threshold, predicted_count = min(valid)
    return threshold, {
        "status": "enabled",
        "validation_recall": -neg_recall,
        "validation_fdr": fdr,
        "validation_precision": -neg_precision,
        "validation_predicted_positives": predicted_count,
        "validation_recall_floor_met": True,
        "validation_fdr_ceiling_met": True,
    }
