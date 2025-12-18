from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, text
from src.api.main import app
from src.api.config import settings
client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_validation_error():
    response = client.post("/predict", json={"age": "old", "workclass": "Private"})
    assert response.status_code == 422


@patch("src.api.main.PREDICTOR")
def test_predict_integration_db_count(mock_predictor):
    mock_predictor.predict.return_value = MagicMock(iloc=[0])
    mock_predictor.predict_proba.return_value = MagicMock(iloc=[[0.1, 0.9]])
    mock_predictor.feature_metadata_in.get_features.return_value = ["age", "workclass"]

    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as conn:
        count_before = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()

    payload = {
        "age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors",
        "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical",
        "relationship": "Not-in-family", "race": "White", "sex": "Male",
        "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States",
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    with engine.connect() as conn:
        count_after = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
    assert count_after == count_before + 1
