"""Sensor-to-failure correlation analysis for Task 9."""
from pathlib import Path

import pandas as pd

from src.data_loader import FEATURES, FAILURE_MODES, TARGET


SENSOR_FEATURES = [feature for feature in FEATURES if feature != "Type"]
OUTCOMES = [TARGET, *FAILURE_MODES]


def sensor_failure_correlations(frame):
    """Rank numeric sensor associations with overall and mode-specific labels.

    Pearson captures linear association; Spearman captures monotonic
    association and is less sensitive to strictly linear assumptions.
    Correlation is descriptive, not evidence of causation or temporal lead time.
    """
    required = SENSOR_FEATURES + OUTCOMES
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for correlation analysis: {missing}")

    rows = []
    for outcome in OUTCOMES:
        for sensor in SENSOR_FEATURES:
            pearson = frame[sensor].corr(frame[outcome], method="pearson")
            spearman = frame[sensor].corr(frame[outcome], method="spearman")
            rows.append(
                {
                    "outcome": outcome,
                    "sensor": sensor,
                    "pearson_correlation": float(pearson),
                    "spearman_correlation": float(spearman),
                    "absolute_pearson_correlation": abs(float(pearson)),
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["outcome", "absolute_pearson_correlation"],
        ascending=[True, False],
    ).reset_index(drop=True)


def save_sensor_failure_correlations(frame, output_path="artifacts/sensor_failure_correlations.csv"):
    result = sensor_failure_correlations(frame)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result
