"""CLI entrypoint to run individual pipeline stages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger


def run_ingest():
    import pandas as pd
    from pathlib import Path

    cc_path = Path("data/raw/creditcard.csv")
    raw_path = Path("data/raw/transactions_raw.parquet")

    if cc_path.exists():
        logger.info(f"Loading creditcard.csv schema from {cc_path}")
        df = pd.read_csv(cc_path)
        logger.info(f"Loaded {len(df):,} rows — fraud rate: {df['Class'].mean()*100:.3f}%")
    else:
        from src.data.ingestion import TransactionIngester
        ingester = TransactionIngester()
        df = ingester.from_parquet("transactions_raw.parquet")

    df.to_parquet("data/processed/transactions.parquet", index=False)
    logger.info("Ingestion complete.")


def run_features():
    import pandas as pd
    from src.features.engineering import build_feature_matrix
    df = pd.read_parquet("data/processed/transactions.parquet")
    features = build_feature_matrix(df)
    features.to_parquet("data/features/feature_matrix.parquet", index=False)
    logger.info("Feature engineering complete.")


def run_train():
    import pandas as pd
    from src.models.train import train
    df = pd.read_parquet("data/features/feature_matrix.parquet")
    run_id = train(df)
    logger.info(f"Training complete. MLflow run_id: {run_id}")


STAGES = {"ingest": run_ingest, "features": run_features, "train": run_train}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--stage", choices=list(STAGES), required=True)
    args = parser.parse_args()
    STAGES[args.stage]()
