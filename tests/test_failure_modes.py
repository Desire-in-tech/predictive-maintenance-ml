import numpy as np
import pytest

from src.failure_modes import evaluate_mode, mode_meets_operating_constraints
from src.evaluation import choose_threshold_with_constraints
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
    assert metrics["average_precision"] is None


def test_training_rejects_invalid_recall_floor():
    from src.failure_modes import train_failure_modes

    with pytest.raises(ValueError, match="min_recall"):
        train_failure_modes(min_recall=0)



def test_threshold_selector_rejects_all_positive_high_fdr_fallback():
    threshold, details = choose_threshold_with_constraints(
        y_true=[0, 0, 0, 1],
        probabilities=[0.1, 0.1, 0.1, 0.1],
        min_recall=0.30,
        max_fdr=0.50,
    )
    assert threshold is None
    assert details["status"] == "unavailable_no_threshold_meets_constraints"
    assert details["validation_fdr"] == 0.75
    assert details["validation_recall_floor_met"] is True
    assert details["validation_fdr_ceiling_met"] is False


def test_threshold_selector_returns_threshold_when_both_constraints_met():
    threshold, details = choose_threshold_with_constraints(
        y_true=[0, 0, 1, 1],
        probabilities=[0.1, 0.2, 0.8, 0.9],
        min_recall=0.50,
        max_fdr=0.25,
    )
    assert threshold is not None
    assert details["status"] == "enabled"
    assert details["validation_recall_floor_met"] is True
    assert details["validation_fdr_ceiling_met"] is True


def test_threshold_selector_rejects_invalid_fdr_ceiling():
    with pytest.raises(ValueError, match="max_fdr"):
        choose_threshold_with_constraints([0, 1], [0.1, 0.9], max_fdr=1.0)



def test_alerting_gate_requires_validation_and_test_constraints():
    passing_selection = {"status": "enabled"}
    passing_test = {"recall": 0.5, "false_discovery_rate": 0.2}
    failing_test_fdr = {"recall": 0.8, "false_discovery_rate": 0.8}
    failing_test_recall = {"recall": 0.1, "false_discovery_rate": 0.0}

    assert mode_meets_operating_constraints(
        passing_selection, passing_test, min_recall=0.3, max_fdr=0.5
    )
    assert not mode_meets_operating_constraints(
        passing_selection, failing_test_fdr, min_recall=0.3, max_fdr=0.5
    )
    assert not mode_meets_operating_constraints(
        passing_selection, failing_test_recall, min_recall=0.3, max_fdr=0.5
    )
    assert not mode_meets_operating_constraints(
        {"status": "unavailable_no_threshold_meets_constraints"},
        passing_test,
        min_recall=0.3,
        max_fdr=0.5,
    )
