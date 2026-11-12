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
