from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Income Prediction API",
    description="API for income prediction (<=50K or >50K) based on demographic data.",
    version="0.1.0",
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

    prediction: int
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

    return {"prediction": 0, "probability": 0.85, "model_version": "local-dev-stub"}
