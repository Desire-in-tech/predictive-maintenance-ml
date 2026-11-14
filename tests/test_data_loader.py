from types import SimpleNamespace

import pandas as pd
import pytest

from src import data_loader


def test_loader_preserves_all_failure_mode_targets(monkeypatch):
    features = pd.DataFrame(
        {
            "Type": ["L", "M"],
            "Air temperature": [298.1, 299.1],
            "Process temperature": [308.6, 309.6],
            "Rotational speed": [1500, 1600],
            "Torque": [40.0, 45.0],
            "Tool wear": [10, 20],
        }
    )
    targets = pd.DataFrame(
        {
            "Machine failure": [0, 1],
            "TWF": [0, 1],
            "HDF": [0, 0],
            "PWF": [0, 0],
            "OSF": [0, 1],
            "RNF": [0, 0],
        }
    )
    dataset = SimpleNamespace(
        data=SimpleNamespace(features=features, targets=targets)
    )
    monkeypatch.setattr(data_loader, "fetch_ucirepo", lambda id: dataset)

    frame = data_loader.load_ai4i()

    assert list(frame.columns) == [
        *data_loader.FEATURES,
        data_loader.TARGET,
        *data_loader.FAILURE_MODES,
    ]
    assert frame.loc[1, "Machine failure"] == 1
    assert frame.loc[1, "TWF"] == 1
    assert frame.loc[1, "OSF"] == 1


def test_loader_rejects_non_binary_failure_mode(monkeypatch):
    features = pd.DataFrame(
        {
            "Type": ["L"],
            "Air temperature": [298.1],
            "Process temperature": [308.6],
            "Rotational speed": [1500],
            "Torque": [40.0],
            "Tool wear": [10],
        }
    )
    targets = pd.DataFrame(
        {
            "Machine failure": [1],
            "TWF": [2],
            "HDF": [0],
            "PWF": [0],
            "OSF": [0],
            "RNF": [0],
        }
    )
    dataset = SimpleNamespace(
        data=SimpleNamespace(features=features, targets=targets)
    )
    monkeypatch.setattr(data_loader, "fetch_ucirepo", lambda id: dataset)

    with pytest.raises(ValueError, match="TWF must be binary"):
        data_loader.load_ai4i()
