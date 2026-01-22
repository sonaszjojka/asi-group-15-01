import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from src.api.main import app
from src.api.config import settings

client = TestClient(app)


def _is_db_available():
    """Check if database is available."""
    if not settings.DATABASE_URL:
        return False
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_validation_error():
    response = client.post("/predict", json={"age": "old", "workclass": "Private"})
    assert response.status_code == 422


@pytest.mark.skipif(not _is_db_available(), reason="Database not available")
@patch("src.api.main.PREDICTOR")
def test_predict_integration_db_count(mock_predictor):
    mock_predictor.predict.return_value = MagicMock(iloc=[0])
    mock_predictor.predict_proba.return_value = MagicMock(iloc=[[0.1, 0.9]])
    mock_predictor.feature_metadata_in.get_features.return_value = ["age", "workclass"]

    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as conn:
        count_before = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()

    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    with engine.connect() as conn:
        count_after = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
    assert count_after == count_before + 1
