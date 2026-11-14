import pytest
from fastapi.testclient import TestClient

from api.main import app


class StubModel:
    def predict_with_failure_modes(self, readings):
        return [
            {
                "failure_probability": 0.12,
                "predicted_failure": False,
                "threshold": 0.5,
                "failure_modes": {},
                "predicted_failure_modes": [],
            }
            for _ in readings
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("api.main.load_model", lambda: StubModel())
    with TestClient(app) as test_client:
        yield test_client


def valid_reading():
    return {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }


def test_health_and_ready(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["status"] == "ready"


def test_json_prediction(client):
    response = client.post("/predict", json={"readings": [valid_reading()]})
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["predictions"][0]["predicted_failure"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"readings": [{**valid_reading(), "Torque [Nm]": "42.8"}]},
        {"readings": [{**valid_reading(), "Type": "X"}]},
        {"readings": [{**valid_reading(), "unknown": 1}]},
        {"readings": [{key: value for key, value in valid_reading().items() if key != "Type"}]},
        {"readings": []},
    ],
)
def test_invalid_json_inputs_return_400(client, payload):
    assert client.post("/predict", json=payload).status_code == 400


def test_malformed_json_returns_400(client):
    response = client.post(
        "/predict",
        content='{"readings": [',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_csv_prediction(client):
    columns = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]
    row = ["M", 298.1, 308.6, 1551, 42.8, 0]
    csv_text = ",".join(columns) + "\n" + ",".join(map(str, row)) + "\n"
    response = client.post(
        "/predict/csv",
        files={"file": ("readings.csv", csv_text, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_invalid_csv_returns_400(client):
    response = client.post(
        "/predict/csv",
        files={"file": ("readings.csv", "Type,wrong\nM,1\n", "text/csv")},
    )
    assert response.status_code == 400


def test_csv_wrong_numeric_type_returns_400(client):
    columns = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]
    row = ["M", "not-a-number", 308.6, 1551, 42.8, 0]
    csv_text = ",".join(columns) + "\n" + ",".join(map(str, row)) + "\n"
    response = client.post(
        "/predict/csv",
        files={"file": ("readings.csv", csv_text, "text/csv")},
    )
    assert response.status_code == 400


def test_unready_model_returns_503(client):
    app.state.model = None
    response = client.post("/predict", json={"readings": [valid_reading()]})
    assert response.status_code == 503
