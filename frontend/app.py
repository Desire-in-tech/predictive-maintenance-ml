"""Streamlit client for the Task 10 predictive-maintenance API."""
import json
import os

import pandas as pd
import requests
import streamlit as st

try:
    from frontend.utils import validate_api_base_url
except ModuleNotFoundError:
    # Streamlit Cloud may place frontend/ rather than the repository root on sys.path.
    from utils import validate_api_base_url

DEFAULT_READING = {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
}
DEFAULT_JSON = json.dumps({"readings": [DEFAULT_READING]}, indent=2)
REQUEST_TIMEOUT = (10, 180)

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("⚙️ Industrial Predictive Maintenance")
st.caption("Task 9 machine-failure and failure-mode models, served through the Task 10 API.")
st.info("Decision support only: model scores are not a substitute for machine inspection or an established maintenance and safety process.")

with st.sidebar:
    st.header("API connection")
    configured_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    api_url_input = st.text_input("API base URL", value=configured_url)
    st.caption("Set API_BASE_URL in the deployment environment to configure the default.")
    check_api = st.button("Check API readiness", use_container_width=True)

try:
    API_BASE_URL = validate_api_base_url(api_url_input)
except ValueError as exc:
    API_BASE_URL = None
    st.sidebar.error(str(exc))

if check_api and API_BASE_URL:
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=REQUEST_TIMEOUT)
        ready = requests.get(f"{API_BASE_URL}/ready", timeout=REQUEST_TIMEOUT)
        if health.ok and ready.ok:
            details = ready.json()
            revision = details.get("model_revision") or "unknown"
            st.sidebar.success(f"API ready · revision {revision[:12]}")
        else:
            st.sidebar.error(f"API not ready (health {health.status_code}, readiness {ready.status_code}). {ready.text[:500]}")
    except requests.RequestException as exc:
        st.sidebar.error(f"Could not reach API: {exc}")

input_tab, guide_tab = st.tabs(["Run predictions", "Input guide"])
with input_tab:
    input_method = st.radio("Choose input format", ["JSON batch", "CSV file"], horizontal=True)
    response_data = None

    if input_method == "JSON batch":
        json_text = st.text_area("Prediction request (JSON)", value=DEFAULT_JSON, height=280,
            help="Send an object containing a readings array. Each reading must have the six sensor fields.")
        submitted = st.button("Run JSON predictions", type="primary", disabled=not API_BASE_URL)
        if submitted:
            try:
                payload = json.loads(json_text)
                if not isinstance(payload, dict) or not isinstance(payload.get("readings"), list):
                    raise ValueError('The JSON root must be an object containing a "readings" array.')
                with st.spinner("Running model inference…"):
                    response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
                if response.ok:
                    response_data = response.json()
                else:
                    st.error(f"API returned HTTP {response.status_code}: {response.text[:1500]}")
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"Invalid JSON input: {exc}")
            except requests.RequestException as exc:
                st.error(f"Could not complete prediction request: {exc}")
    else:
        uploaded_file = st.file_uploader("Upload sensor readings CSV", type=["csv"])
        st.caption("The CSV must contain the exact six columns listed in the Input guide. Maximum 5 MiB and 5,000 rows.")
        submitted = st.button("Run CSV predictions", type="primary", disabled=not API_BASE_URL or uploaded_file is None)
        if submitted and uploaded_file is not None:
            try:
                with st.spinner("Uploading data and running model inference…"):
                    response = requests.post(
                        f"{API_BASE_URL}/predict/csv",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                        timeout=REQUEST_TIMEOUT,
                    )
                if response.ok:
                    response_data = response.json()
                else:
                    st.error(f"API returned HTTP {response.status_code}: {response.text[:1500]}")
            except requests.RequestException as exc:
                st.error(f"Could not complete prediction request: {exc}")

    if response_data is not None:
        predictions = response_data.get("predictions", [])
        st.success(f"Completed predictions for {response_data.get('count', len(predictions))} reading(s).")
        if predictions:
            overall_alerts = sum(bool(item.get("predicted_failure")) for item in predictions)
            mode_alerts = sum(len(item.get("predicted_failure_modes", [])) for item in predictions)
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Readings scored", len(predictions))
            metric_b.metric("Overall failure alerts", overall_alerts)
            metric_c.metric("Failure-mode alerts", mode_alerts)

            st.subheader("Overall machine-failure results")
            overall_rows = [
                {"Reading": index, "Failure probability": item.get("failure_probability"),
                 "Predicted failure": item.get("predicted_failure"), "Decision threshold": item.get("threshold")}
                for index, item in enumerate(predictions, start=1)
            ]
            st.dataframe(pd.DataFrame(overall_rows), use_container_width=True, hide_index=True)

            st.subheader("Failure-mode results")
            mode_rows = []
            for index, item in enumerate(predictions, start=1):
                for mode, details in item.get("failure_modes", {}).items():
                    mode_rows.append({
                        "Reading": index, "Failure mode": mode,
                        "Probability": details.get("probability"), "Alert": details.get("predicted"),
                        "Threshold": details.get("threshold"),
                        "Alert enabled": details.get("available_for_alerting"),
                    })
            if mode_rows:
                st.dataframe(pd.DataFrame(mode_rows), use_container_width=True, hide_index=True)
            else:
                st.warning("The API response did not contain failure-mode details.")

            with st.expander("View raw API response"):
                st.json(response_data)

with guide_tab:
    st.markdown("### Required sensor fields")
    st.code("Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]", language="text")
    st.markdown(
        "- **Type:** L, M, or H\n"
        "- **Numeric values:** finite JSON numbers (not numeric strings)\n"
        "- **JSON:** an object with a readings array containing 1–5,000 observations\n"
        "- **CSV:** exact six-column header, maximum 5 MiB and 5,000 rows\n"
        "- **Invalid input:** the API returns HTTP 400; an unavailable model returns HTTP 503"
    )
    st.markdown(
        "Failure-mode indicators are independent and can overlap. A mode with no "
        "validation-approved threshold remains score-only; its probability is shown, "
        "but it cannot raise an alert."
    )
    st.code(DEFAULT_JSON, language="json")

st.divider()
st.caption("Model: UCI AI4I 2020 Predictive Maintenance · Task 9 artifacts · Predictions should be reviewed by qualified maintenance personnel.")
