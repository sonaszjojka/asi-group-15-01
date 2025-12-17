import os
import pandas as pd
import wandb
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from autogluon.tabular import TabularPredictor

from ..asi_group_15_01.pipelines.data_science.nodes import basic_clean

MODEL_PATH = os.getenv(
    "MODEL_PATH", "data/06_models/ag_production/ds_sub_fit/sub_fit_ho"
)
MODEL_VERSION = "local"
PREDICTOR: Optional[TabularPredictor] = None


def load_model():
    """
    Loads model from W&B artifact (Production) or falls back to local.
    """
    try:
        api = wandb.Api()
        artifact = api.artifact(
            "s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/ag_model:production",
            type="model",
        )  # TODO: move to env?
        model_path = artifact.download()
        model = TabularPredictor.load(model_path)
        return model, artifact.version
    except Exception as e:
        print(f"Failed to load from W&B: {e}")
        if os.path.exists(MODEL_PATH):
            return TabularPredictor.load(MODEL_PATH), "local"
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Prepare singleton.
    """
    global PREDICTOR, MODEL_VERSION
    PREDICTOR, version = load_model()
    if version:
        MODEL_VERSION = version
    yield


app = FastAPI(
    title="Income Prediction API",
    description="API for income prediction (<=50K or >50K) based on demographic data.",
    version="0.1.0",
    lifespan=lifespan,
)


# Pydantic scheme


class Features(BaseModel):
    """
    Input model data.
    """

    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        """
        Model example configuration.
        """

        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }


class Prediction(BaseModel):
    """
    API response model.
    """

    prediction: str
    probability: float
    model_version: str


# Endpoints


@app.get("/healthz")
def healthz():
    """
    Endpoint for Health Check.
    """

    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    """
    Endpoint for making predictions.
    """

    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model did not load.")

    input_data = payload.model_dump(by_alias=True)
    df = pd.DataFrame([input_data])

    if "income" not in df.columns:
        df["income"] = "<=50K"

    try:
        df = basic_clean(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in basic_clean: {e}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Data is not valid.")

    required_features = PREDICTOR.feature_metadata_in.get_features()

    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0

    df = df[required_features]

    y_pred = PREDICTOR.predict(df).iloc[0]
    y_proba = PREDICTOR.predict_proba(df).iloc[0][1]

    class_mapping = {0: "<=50K", 1: ">50K"}

    return {
        "prediction": class_mapping.get(int(y_pred), "Unknown"),
        "probability": float(y_proba),
        "model_version": MODEL_VERSION,
    }
