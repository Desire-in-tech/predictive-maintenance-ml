from frontend.utils import flatten_predictions, validate_api_base_url


def test_flatten_predictions_preserves_alert_eligibility():
    rows = flatten_predictions([{
        "failure_probability": 0.2,
        "predicted_failure": False,
        "threshold": 0.51,
        "predicted_failure_modes": ["HDF"],
        "failure_modes": {
            "HDF": {
                "probability": 0.8,
                "predicted": True,
                "threshold": 0.63,
                "available_for_alerting": True,
            },
            "TWF": {
                "probability": 0.7,
                "predicted": False,
                "threshold": None,
                "available_for_alerting": False,
            },
        },
    }])

    assert len(rows) == 1
    assert rows[0]["HDF alert enabled"] is True
    assert rows[0]["TWF alert enabled"] is False
    assert rows[0]["Predicted failure modes"] == "HDF"


def test_flatten_predictions_handles_empty_failure_modes():
    rows = flatten_predictions([{"failure_modes": {}, "predicted_failure_modes": []}])
    assert rows[0]["Predicted failure modes"] == "None"


def test_validate_api_base_url_normalizes_trailing_slashes():
    assert validate_api_base_url(" https://example.com/api/ ") == "https://example.com/api"


def test_validate_api_base_url_rejects_missing_scheme():
    try:
        validate_api_base_url("example.com")
    except ValueError as exc:
        assert "https://" in str(exc)
    else:
        raise AssertionError("Expected an invalid API URL to be rejected.")
