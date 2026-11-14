"""Small pure helpers for presenting prediction API responses."""

def flatten_predictions(predictions):
    """Convert nested API predictions into display-friendly table rows."""
    rows = []
    for index, prediction in enumerate(predictions, start=1):
        row = {
            "Reading": index,
            "Failure probability": prediction.get("failure_probability"),
            "Predicted failure": prediction.get("predicted_failure"),
            "Overall threshold": prediction.get("threshold"),
            "Predicted failure modes": ", ".join(prediction.get("predicted_failure_modes", [])) or "None",
        }
        for mode, details in prediction.get("failure_modes", {}).items():
            row[f"{mode} probability"] = details.get("probability")
            row[f"{mode} predicted"] = details.get("predicted")
            row[f"{mode} threshold"] = details.get("threshold")
            row[f"{mode} alert enabled"] = details.get("available_for_alerting")
        rows.append(row)
    return rows

def validate_api_base_url(value):
    """Normalize and validate an HTTP(S) API base URL."""
    url = (value or "").strip().rstrip("/")
    if not url.startswith(("https://", "http://")):
        raise ValueError("Enter an API URL beginning with https:// or http://.")
    return url
