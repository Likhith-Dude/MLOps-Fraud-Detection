"""
Generate a synthetic replica of the Kaggle creditcard.csv dataset.

Schema matches exactly:
  - Time: seconds elapsed since first transaction
  - V1-V28: PCA-transformed features (correlated gaussian blobs per class)
  - Amount: transaction amount (log-normal)
  - Class: 0 = legit, 1 = fraud  (0.172% fraud — same as real dataset)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from loguru import logger

N_TOTAL   = 284_807
FRAUD_RATE = 0.001727          # matches the real dataset exactly
N_FRAUD   = round(N_TOTAL * FRAUD_RATE)   # ~492
N_LEGIT   = N_TOTAL - N_FRAUD
RNG = np.random.default_rng(42)


def _pca_block(n: int, means: np.ndarray, scale: float) -> np.ndarray:
    """Sample 28 correlated features via a random rotation of Gaussians."""
    raw = RNG.normal(loc=means, scale=scale, size=(n, 28))
    # random orthogonal rotation to create inter-feature correlation
    Q, _ = np.linalg.qr(RNG.standard_normal((28, 28)))
    return raw @ Q


def generate() -> pd.DataFrame:
    logger.info(f"Generating {N_TOTAL:,} transactions ({N_FRAUD} fraud, {N_LEGIT:,} legit)...")

    # V1-V28: fraud cluster is offset from legit cluster
    legit_means  = RNG.uniform(-0.5,  0.5, 28)
    fraud_means  = RNG.uniform(-3.0,  3.0, 28)   # more extreme values

    V_legit = _pca_block(N_LEGIT, legit_means, scale=1.0)
    V_fraud = _pca_block(N_FRAUD, fraud_means, scale=1.8)  # higher variance too

    V = np.vstack([V_legit, V_fraud])

    # Amount: log-normal; fraud skews toward smaller (card-testing) and large amounts
    amount_legit = RNG.lognormal(mean=3.0, sigma=1.5, size=N_LEGIT).round(2)
    amount_fraud = np.concatenate([
        RNG.lognormal(mean=1.5, sigma=1.2, size=N_FRAUD // 2),   # small card-test txns
        RNG.lognormal(mean=5.0, sigma=1.0, size=N_FRAUD - N_FRAUD // 2),  # large hits
    ]).round(2)
    amounts = np.concatenate([amount_legit, amount_fraud])

    # Time: spread across 48 hours in seconds
    time_legit = np.sort(RNG.uniform(0, 172_800, N_LEGIT))
    time_fraud = RNG.uniform(0, 172_800, N_FRAUD)
    times = np.concatenate([time_legit, time_fraud]).round(0)

    labels = np.array([0] * N_LEGIT + [1] * N_FRAUD)

    cols = {f"V{i}": V[:, i-1].round(6) for i in range(1, 29)}
    df = pd.DataFrame({"Time": times, **cols, "Amount": amounts, "Class": labels})

    # Shuffle so fraud isn't all at the end
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out = Path("data/raw/creditcard.csv")
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df):,} rows → {out}  (fraud rate: {df.Class.mean()*100:.3f}%)")
    return df


if __name__ == "__main__":
    generate()
