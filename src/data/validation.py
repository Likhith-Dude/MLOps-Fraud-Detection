"""Schema validation and data quality checks using Pandera."""
from __future__ import annotations

import pandera as pa
import pandas as pd
from pandera.typing import Series


class TransactionSchema(pa.DataFrameModel):
    transaction_id: Series[str] = pa.Field(unique=True, nullable=False)
    timestamp: Series[pd.DatetimeTZDtype] = pa.Field(nullable=False)
    amount: Series[float] = pa.Field(ge=0.0, nullable=False)
    merchant_id: Series[str] = pa.Field(nullable=False)
    user_id: Series[str] = pa.Field(nullable=False)
    is_fraud: Series[int] = pa.Field(isin=[0, 1], nullable=True)

    class Config:
        coerce = True
        strict = False


def validate(df: pd.DataFrame) -> pd.DataFrame:
    return TransactionSchema.validate(df)
