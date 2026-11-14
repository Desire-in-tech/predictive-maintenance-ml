"""UCI AI4I 2020 dataset loader."""
import pandas as pd
from ucimlrepo import fetch_ucirepo

FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET = "Machine failure"


def load_ai4i():
    dataset = fetch_ucirepo(id=601)
    X, y = dataset.data.features.copy(), dataset.data.targets.copy()

    # Map the UCI API's column names to our expected project column names.
    column_mapping = {
        "Air temperature": "Air temperature [K]",
        "Process temperature": "Process temperature [K]",
        "Rotational speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool wear": "Tool wear [min]",
    }
    X = X.rename(columns=column_mapping)

    if TARGET not in y.columns:
        if TARGET in X.columns:
            y[TARGET] = X.pop(TARGET)
        else:
            raise ValueError(
                f"Missing target {TARGET}; target columns={list(y.columns)}"
            )

    frame = pd.concat([X, y[[TARGET]]], axis=1)
    missing = [column for column in FEATURES + [TARGET] if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame[FEATURES + [TARGET]].dropna().copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Machine failure must be binary.")

    return frame.reset_index(drop=True)
