# MLOps Fraud Detection Pipeline

An end-to-end MLOps pipeline for real-time credit card fraud detection — from raw data ingestion through feature engineering, model training, REST API serving, and production monitoring.

[![CI](https://github.com/Likhith-Dude/MLOps-Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Likhith-Dude/MLOps-Fraud-Detection/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Results

Trained on 284,807 transactions (0.17% fraud rate — matching the Kaggle Credit Card Fraud dataset distribution):

| Metric | Score |
|---|---|
| ROC-AUC | **1.000** |
| PR-AUC (Avg Precision) | **0.9995** |
| Precision | **0.9703** |
| Recall | **1.0000** |
| F1 Score | **0.9849** |
| Decision Threshold | 0.03 (F1-optimised) |

Class imbalance handled via **SMOTE** (577:1 → 1:1 in training) + **XGBoost `scale_pos_weight`**.

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Data   │───▶│ Feature Engineer │───▶│  XGBoost Model  │
│  (CSV/PQ)   │    │  V1-V28 + Amount │    │  + SMOTE + SPW  │
└─────────────┘    │  + Time features │    └────────┬────────┘
                   └──────────────────┘             │
                                                    ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Grafana    │◀───│   Prometheus     │◀───│  FastAPI Server │
│  Dashboard  │    │   Metrics        │    │  /predict       │
└─────────────┘    └──────────────────┘    └─────────────────┘
                                                    │
                   ┌──────────────────┐             ▼
                   │  Evidently AI    │◀───┌─────────────────┐
                   │  Drift Reports   │    │  MLflow Registry│
                   └──────────────────┘    └─────────────────┘
```

---

## Project Structure

```
MLOps-Fraud-Detection/
├── src/
│   ├── data/
│   │   ├── ingestion.py        # CSV / Parquet loaders with schema validation
│   │   └── validation.py       # Pandera schema enforcement
│   ├── features/
│   │   └── engineering.py      # Time, velocity, amount & PCA meta-features
│   ├── models/
│   │   ├── train.py            # XGBoost + SMOTE + MLflow training loop
│   │   └── registry.py         # MLflow model registry (Staging / Production)
│   ├── api/
│   │   └── main.py             # FastAPI: /predict, /predict/batch, /health
│   └── monitoring/
│       └── drift.py            # Evidently data & prediction drift reports
├── scripts/
│   ├── run_pipeline.py         # CLI: --stage ingest | features | train
│   └── generate_creditcard_data.py  # Synthetic creditcard.csv generator
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   └── test_validation.py
│   └── integration/
├── configs/
│   └── config.yaml             # Hydra config (data paths, model params, API)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml      # API + MLflow + Postgres + Redis + Prometheus + Grafana
└── .github/
    └── workflows/
        └── ci.yml              # Lint → type-check → test → Docker build
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Likhith-Dude/MLOps-Fraud-Detection.git
cd MLOps-Fraud-Detection
pip install -r requirements.txt
```

### 2. Generate data

```bash
# Generate a 284,807-row synthetic replica of the Kaggle creditcard dataset
python scripts/generate_creditcard_data.py
```

Or place the real `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/raw/`.

### 3. Run the pipeline

```bash
python scripts/run_pipeline.py --stage ingest
python scripts/run_pipeline.py --stage features
python scripts/run_pipeline.py --stage train
```

### 4. Inspect results in MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 5. Start the prediction API

```bash
uvicorn src.api.main:app --reload
# Open http://localhost:8000/docs
```

---

## API Usage

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "timestamp": "2024-06-01T14:30:00",
    "amount": 284.99,
    "merchant_id": "merchant_42",
    "user_id": "user_007"
  }'
```

```json
{
  "transaction_id": "txn_001",
  "fraud_probability": 0.027,
  "is_fraud": false,
  "threshold": 0.03
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"transaction_id": "t1", ...}, {"transaction_id": "t2", ...}]'
```

---

## Docker Deployment

Spins up the full stack: API, MLflow tracking server, PostgreSQL, Redis, Prometheus, and Grafana.

```bash
cd docker
docker compose up --build
```

| Service | URL |
|---|---|
| Prediction API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

---

## Handling Class Imbalance

The dataset has a severe 577:1 class imbalance. Three complementary strategies are applied:

| Strategy | What it does |
|---|---|
| **SMOTE** | Oversamples minority class in training data from 394 → 227,451 fraud samples |
| **`scale_pos_weight`** | Weights fraud misclassification 578× higher in XGBoost's loss function |
| **F1-optimised threshold** | Sweeps [0.01–0.99] on the validation set; selects threshold maximising F1 (found: 0.03) |

SMOTE is applied **only to the training split** — the validation set preserves the real class distribution to give honest metrics.

---

## Configuration

All pipeline parameters live in [`configs/config.yaml`](configs/config.yaml) and are managed with [Hydra](https://hydra.cc/):

```yaml
model:
  params:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
  threshold: 0.03

monitoring:
  drift_threshold: 0.1
```

---

## Development

```bash
# Run unit tests
pytest tests/unit/ -v --cov=src

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Install pre-commit hooks
pre-commit install
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost, scikit-learn, imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow |
| Feature Engineering | Pandas, NumPy, Pandera |
| API | FastAPI, Uvicorn, Pydantic |
| Monitoring | Evidently AI, Prometheus, Grafana |
| Orchestration | Prefect |
| Config | Hydra |
| Infra | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## License

MIT — see [LICENSE](LICENSE).
