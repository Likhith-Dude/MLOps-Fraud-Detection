"""Feature engineering — supports both raw transaction schema and creditcard PCA schema."""
from __future__ import annotations

import pandas as pd
import numpy as np

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]


def _is_pca_schema(df: pd.DataFrame) -> bool:
    return "V1" in df.columns and "Class" in df.columns


# ── PCA schema (creditcard.csv) ───────────────────────────────────────────────

def add_creditcard_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour_of_day"] = (df["Time"] // 3600 % 24).astype(int)
    df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
    # V-feature summary statistics as meta-features
    v_cols = PCA_FEATURES
    df["v_abs_sum"] = df[v_cols].abs().sum(axis=1)
    df["v_max_abs"] = df[v_cols].abs().max(axis=1)
    return df


# ── Raw transaction schema ────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_velocity_features(df: pd.DataFrame, windows: list[str] = ["1h", "24h"]) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df = df.set_index("timestamp")
    for window in windows:
        df[f"txn_count_{window}"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda s: s.rolling(window).count())
        )
        df[f"txn_sum_{window}"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda s: s.rolling(window).sum())
        )
    return df.reset_index()


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    user_mean = df.groupby("user_id")["amount"].transform("mean")
    df["amount_deviation"] = (df["amount"] - user_mean) / (user_mean + 1e-9)
    return df


# ── Public entrypoint ─────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if _is_pca_schema(df):
        return add_creditcard_features(df)
    df = add_time_features(df)
    df = add_velocity_features(df)
    df = add_amount_features(df)
    return df
