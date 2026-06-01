"""FastAPI prediction service."""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from loguru import logger

from src.models.registry import load_production_model
from src.features.engineering import build_feature_matrix

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info("Loading production model...")
    _model = load_production_model()
    logger.info("Model ready.")
    yield
    _model = None


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


class Transaction(BaseModel):
    transaction_id: str
    timestamp: str
    amount: float = Field(gt=0)
    merchant_id: str
    user_id: str


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold: float = 0.5


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(txn: Transaction):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([txn.model_dump()])
    features = build_feature_matrix(df)

    from src.models.train import FEATURE_COLS
    proba = float(_model.predict_proba(features[FEATURE_COLS])[:, 1][0])
    threshold = 0.5

    return PredictionResponse(
        transaction_id=txn.transaction_id,
        fraud_probability=proba,
        is_fraud=proba >= threshold,
        threshold=threshold,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(transactions: list[Transaction]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([t.model_dump() for t in transactions])
    features = build_feature_matrix(df)

    from src.models.train import FEATURE_COLS
    probas = _model.predict_proba(features[FEATURE_COLS])[:, 1]
    threshold = 0.5

    return [
        PredictionResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=float(p),
            is_fraud=float(p) >= threshold,
            threshold=threshold,
        )
        for txn, p in zip(transactions, probas)
    ]
