import pandas as pd
import pytest

from src.analysis import sensor_failure_correlations
from src.data_loader import FAILURE_MODES, TARGET


def test_sensor_failure_correlations_ranks_each_sensor_for_each_outcome():
    frame = pd.DataFrame(
        {
            "Air temperature [K]": [1, 2, 3, 4, 5, 6],
            "Process temperature [K]": [6, 5, 4, 3, 2, 1],
            "Rotational speed [rpm]": [1, 3, 2, 6, 4, 5],
            "Torque [Nm]": [6, 4, 5, 2, 3, 1],
            "Tool wear [min]": [1, 2, 1, 2, 1, 2],
            TARGET: [0, 0, 0, 1, 1, 1],
            **{mode: [0, 0, 1, 0, 1, 1] for mode in FAILURE_MODES},
        }
    )
    result = sensor_failure_correlations(frame)
    assert set(result["outcome"]) == {TARGET, *FAILURE_MODES}
    assert len(result) == 5 * (1 + len(FAILURE_MODES))
    assert result.groupby("outcome").size().eq(5).all()
    assert "spearman_correlation" in result.columns


def test_sensor_failure_correlations_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        sensor_failure_correlations(pd.DataFrame())
