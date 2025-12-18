import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_validation_error():
    response = client.post("/predict", json={"age": "stary", "workclass": "Private"})
    assert response.status_code == 422


@patch("src.api.main.PREDICTOR")
@patch("src.api.main.save_prediction")
def test_predict_success(mock_save, mock_predictor):
    mock_predictor.predict.return_value = MagicMock(iloc=[0])  # Zwróci 0
    mock_predictor.predict_proba.return_value = MagicMock(iloc=[[0.1, 0.9]])  # Prawd. 0.9
    mock_predictor.feature_metadata_in.get_features.return_value = ["age", "workclass"]

    payload = {
        "age": 30, "workclass": "Private", "fnlwgt": 12345, "education": "Bachelors",
        "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical",
        "relationship": "Not-in-family", "race": "White", "sex": "Male",
        "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert mock_save.called