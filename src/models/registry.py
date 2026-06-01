"""MLflow model registry helpers."""
from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "fraud-detector"
client = MlflowClient()


def register(run_id: str, stage: str = "Staging") -> str:
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    return result.version


def load_production_model():
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
