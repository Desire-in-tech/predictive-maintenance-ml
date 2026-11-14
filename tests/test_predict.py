import numpy as np

from src import predict as predict_module


class FakePipeline:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        positive = np.full(len(frame), self.probability)
        return np.column_stack([1 - positive, positive])


def test_enhanced_inference_returns_overall_and_mode_predictions(monkeypatch):
    features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    binary_artifact = {
        "pipeline": FakePipeline(0.8),
        "threshold": 0.5,
        "features": features,
    }
    mode_probabilities = {
        "TWF": 0.7,
        "HDF": 0.2,
        "PWF": 0.1,
        "OSF": 0.6,
        "RNF": 0.05,
    }
    mode_artifact = {
        "models": {
            mode: FakePipeline(probability)
            for mode, probability in mode_probabilities.items()
        },
        "thresholds": {mode: 0.5 for mode in modes},
        "failure_modes": modes,
    }
    artifacts = {
        "binary.joblib": binary_artifact,
        "modes.joblib": mode_artifact,
    }
    monkeypatch.setattr(
        predict_module.joblib,
        "load",
        lambda path: artifacts[path],
    )

    model = predict_module.PredictiveMaintenanceModel(
        "binary.joblib",
        failure_modes_artifact_path="modes.joblib",
    )
    reading = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }

    result = model.predict_with_failure_modes([reading])[0]

    assert result["predicted_failure"] is True
    assert result["predicted_failure_modes"] == ["TWF", "OSF"]
    assert result["failure_modes"]["TWF"]["probability"] == 0.7
    assert result["failure_modes"]["HDF"]["predicted"] is False


def test_enhanced_inference_requires_mode_artifact(monkeypatch):
    artifact = {
        "pipeline": FakePipeline(0.2),
        "threshold": 0.5,
        "features": ["Type"],
    }
    monkeypatch.setattr(predict_module.joblib, "load", lambda path: artifact)
    model = predict_module.PredictiveMaintenanceModel("binary.joblib")

    try:
        model.predict_with_failure_modes([{"Type": "L"}])
    except RuntimeError as exc:
        assert "failure_modes_artifact_path" in str(exc)
    else:
        raise AssertionError("Expected a RuntimeError when mode artifact is absent")



def test_unavailable_failure_mode_does_not_emit_alert(monkeypatch):
    features = ["Type"]
    artifact = {
        "pipeline": FakePipeline(0.1),
        "threshold": 0.5,
        "features": features,
    }
    modes = {
        "models": {"RNF": FakePipeline(0.99)},
        "thresholds": {"RNF": None},
        "failure_modes": ["RNF"],
    }
    monkeypatch.setattr(
        predict_module.joblib,
        "load",
        lambda path: artifact if path == "binary.joblib" else modes,
    )
    model = predict_module.PredictiveMaintenanceModel(
        "binary.joblib", failure_modes_artifact_path="modes.joblib"
    )
    result = model.predict_with_failure_modes([{"Type": "M"}])[0]
    assert result["failure_modes"]["RNF"]["probability"] == 0.99
    assert result["failure_modes"]["RNF"]["threshold"] is None
    assert result["failure_modes"]["RNF"]["available_for_alerting"] is False
    assert result["failure_modes"]["RNF"]["predicted"] is False
    assert result["predicted_failure_modes"] == []
