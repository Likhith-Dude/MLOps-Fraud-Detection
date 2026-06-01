"""Unit tests for feature engineering."""
import pandas as pd
import numpy as np
import pytest
from src.features.engineering import add_time_features, add_amount_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3"],
        "timestamp": pd.to_datetime(["2024-01-15 09:30:00", "2024-01-20 22:00:00", "2024-01-21 14:00:00"]),
        "amount": [100.0, 5000.0, 25.0],
        "merchant_id": ["m1", "m2", "m1"],
        "user_id": ["u1", "u1", "u2"],
    })


def test_time_features(sample_df):
    result = add_time_features(sample_df)
    assert "hour_of_day" in result.columns
    assert "is_weekend" in result.columns
    assert result.loc[0, "hour_of_day"] == 9
    assert result.loc[1, "is_weekend"] == 1  # Saturday


def test_amount_features(sample_df):
    result = add_amount_features(sample_df)
    assert "log_amount" in result.columns
    assert "amount_deviation" in result.columns
    assert result["log_amount"].iloc[0] == pytest.approx(np.log1p(100.0))
