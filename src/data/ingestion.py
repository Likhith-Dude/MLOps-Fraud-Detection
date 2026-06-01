"""Data ingestion: load raw transactions from various sources."""
from __future__ import annotations

import pandas as pd
from loguru import logger
from pathlib import Path


class TransactionIngester:
    """Loads raw transaction data from CSV, Parquet, or a database."""

    def __init__(self, raw_data_dir: str | Path = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)

    def from_csv(self, filename: str) -> pd.DataFrame:
        path = self.raw_data_dir / filename
        logger.info(f"Ingesting CSV: {path}")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        return self._validate(df)

    def from_parquet(self, filename: str) -> pd.DataFrame:
        path = self.raw_data_dir / filename
        logger.info(f"Ingesting Parquet: {path}")
        df = pd.read_parquet(path)
        return self._validate(df)

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"transaction_id", "timestamp", "amount", "merchant_id", "user_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logger.info(f"Loaded {len(df):,} transactions")
        return df
