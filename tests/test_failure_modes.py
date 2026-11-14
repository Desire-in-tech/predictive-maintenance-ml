import numpy as np
import pytest

from src.failure_modes import evaluate_mode
from src.data_loader import FAILURE_MODES


def test_failure_mode_metrics_include_fdr_and_confusion_matrix():
    metrics = evaluate_mode(
        y_true=[0, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.7, 0.2],
        threshold=0.5,
        min_recall=0.3,
    )
    assert metrics["false_discovery_rate"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["confusion_matrix"] == {
        "tn": 1, "fp": 1, "fn": 1, "tp": 1
    }
    assert metrics["support"] == 2


def test_failure_mode_list_contains_all_five_ai4i_indicators():
    assert FAILURE_MODES == ["TWF", "HDF", "PWF", "OSF", "RNF"]


def test_evaluate_mode_handles_test_split_with_one_class():
    metrics = evaluate_mode(
        y_true=[0, 0, 0],
        probabilities=[0.1, 0.2, 0.3],
        threshold=0.5,
        min_recall=0.3,
    )
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] == 0.0


def test_training_rejects_invalid_recall_floor():
    from src.failure_modes import train_failure_modes

    with pytest.raises(ValueError, match="min_recall"):
        train_failure_modes(min_recall=0)
