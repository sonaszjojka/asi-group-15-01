from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str | None = None
    ARTIFACT_NAME: str | None = None
    DATABASE_URL: str = "sqlite:///predictions.db"
    WANDB_API_KEY: str | None = None
    MODEL_VERSION: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()

