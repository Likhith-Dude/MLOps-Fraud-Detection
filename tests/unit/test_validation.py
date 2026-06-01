"""Unit tests for data validation."""
import pandas as pd
import pytest
import pandera as pa
from src.data.validation import validate


def test_valid_schema():
    df = pd.DataFrame({
        "transaction_id": ["t1", "t2"],
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "amount": [10.0, 20.0],
        "merchant_id": ["m1", "m2"],
        "user_id": ["u1", "u2"],
        "is_fraud": [0, 1],
    })
    validated = validate(df)
    assert len(validated) == 2


def test_negative_amount_fails():
    df = pd.DataFrame({
        "transaction_id": ["t1"],
        "timestamp": pd.to_datetime(["2024-01-01"]),
        "amount": [-5.0],
        "merchant_id": ["m1"],
        "user_id": ["u1"],
    })
    with pytest.raises(pa.errors.SchemaError):
        validate(df)
