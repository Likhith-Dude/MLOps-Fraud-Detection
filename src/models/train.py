"""Model training with MLflow experiment tracking."""
from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from loguru import logger

from src.features.engineering import PCA_FEATURES

# ── Feature column sets ───────────────────────────────────────────────────────

CREDITCARD_FEATURE_COLS = PCA_FEATURES + ["log_amount", "hour_of_day", "is_night", "v_abs_sum", "v_max_abs"]

RAW_FEATURE_COLS = [
    "log_amount", "amount_deviation",
    "hour_of_day", "day_of_week", "is_weekend",
    "txn_count_1h", "txn_sum_1h",
    "txn_count_24h", "txn_sum_24h",
]

FEATURE_COLS = RAW_FEATURE_COLS  # default; overridden below for creditcard schema


def _detect_feature_cols(df: pd.DataFrame) -> list[str]:
    if "V1" in df.columns:
        return CREDITCARD_FEATURE_COLS
    return RAW_FEATURE_COLS


def _label_col(df: pd.DataFrame) -> str:
    return "Class" if "Class" in df.columns else "is_fraud"


def _find_best_threshold(y_true, proba) -> float:
    """Sweep thresholds and return the one that maximises F1."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


# ── Main training function ────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    experiment_name: str = "fraud-detection",
    model_params: dict | None = None,
) -> str:
    """Train XGBoost with SMOTE + scale_pos_weight, optimised for F1. Returns MLflow run_id."""
    mlflow.set_experiment(experiment_name)

    feature_cols = _detect_feature_cols(df)
    label_col    = _label_col(df)

    y = df[label_col].astype(int)
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos
    logger.info(f"Class distribution — legit: {neg:,}  fraud: {pos:,}  ratio: {spw:.1f}:1")

    params = {
        "n_estimators":    500,
        "max_depth":       6,
        "learning_rate":   0.05,
        "scale_pos_weight": spw,
        "eval_metric":     "aucpr",
        "random_state":    42,
        "n_jobs":          -1,
    }
    if model_params:
        params.update(model_params)

    X = df[feature_cols]
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # SMOTE only on training split — val set stays untouched (real distribution)
    k = min(5, pos - 1)  # k_neighbors must be < minority class size
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
    n_synth = pd.Series(y_train).value_counts()[1] - y_train_raw.sum()
    logger.info(
        f"SMOTE: added {n_synth:,} synthetic fraud samples  "
        f"({y_train_raw.sum()} → {pd.Series(y_train).value_counts()[1]} fraud in train)"
    )

    with mlflow.start_run() as run:
        mlflow.log_params({
            **params,
            "smote": True,
            "n_train_after_smote": len(X_train),
            "n_val": len(X_val),
            "fraud_rate_original": f"{pos/len(y)*100:.3f}%",
        })

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        proba     = model.predict_proba(X_val)[:, 1]
        threshold = _find_best_threshold(y_val, proba)
        y_pred    = (proba >= threshold).astype(int)

        metrics = {
            "val_roc_auc":       round(roc_auc_score(y_val, proba), 4),
            "val_avg_precision": round(average_precision_score(y_val, proba), 4),
            "val_precision":     round(precision_score(y_val, y_pred, zero_division=0), 4),
            "val_recall":        round(recall_score(y_val, y_pred, zero_division=0), 4),
            "val_f1":            round(f1_score(y_val, y_pred, zero_division=0), 4),
            "best_threshold":    threshold,
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        logger.info(
            f"\n{'─'*45}\n"
            f"  Run ID:       {run.info.run_id}\n"
            f"  ROC-AUC:      {metrics['val_roc_auc']}\n"
            f"  Avg Precision:{metrics['val_avg_precision']}\n"
            f"  Precision:    {metrics['val_precision']}\n"
            f"  Recall:       {metrics['val_recall']}\n"
            f"  F1:           {metrics['val_f1']}\n"
            f"  Threshold:    {threshold:.2f}\n"
            f"{'─'*45}"
        )
        print(f"\nClassification Report (threshold={threshold:.2f}):")
        print(classification_report(y_val, y_pred, target_names=["legit", "fraud"], digits=4))

        return run.info.run_id
