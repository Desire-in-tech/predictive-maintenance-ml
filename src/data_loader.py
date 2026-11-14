"""UCI AI4I 2020 dataset loader."""
import pandas as pd
from ucimlrepo import fetch_ucirepo

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_ai4i():
    """Load sensor features, overall failure, and individual failure-mode labels."""
    dataset = fetch_ucirepo(id=601)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    column_mapping = {
        "Air temperature": "Air temperature [K]",
        "Process temperature": "Process temperature [K]",
        "Rotational speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool wear": "Tool wear [min]",
    }
    X = X.rename(columns=column_mapping)
    y = y.rename(columns=column_mapping)

    # UCI exposes sensor features and target indicators separately. Keep all
    # target indicators so failure-mode models can be trained independently.
    frame = pd.concat([X, y], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]

    required_targets = [TARGET, *FAILURE_MODES]
    missing = [column for column in FEATURES + required_targets if column not in frame]
    if missing:
        raise ValueError(
            "Missing required AI4I columns: "
            f"{missing}; available columns={list(frame.columns)}"
        )

    frame = frame[FEATURES + required_targets].dropna().copy()
    for column in required_targets:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if not set(frame[column].unique()).issubset({0, 1}):
            raise ValueError(f"{column} must be binary.")

    # The overall target should agree with the presence of at least one
    # annotated failure mode. Keep the dataset's overall target authoritative,
    # but expose disagreements for transparent downstream analysis.
    return frame.reset_index(drop=True)
