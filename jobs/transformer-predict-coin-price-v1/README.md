# BTC Price Prediction Model v1.2

This job predicts the future 5-minute interval BTC price using a machine learning model with technical indicators and market features.

## Overview

- **Model Version**: v1.2 (with sklearn Pipeline architecture)
- **Prediction Target**: Future 5-minute interval BTC price (not current price)
- **Model Type**: Ridge Regression with feature scaling and outlier handling
- **Features**: 30 features including moving averages, RSI, VWAP, price ratios, and market indicators

## Prerequisites

- Python 3.11+
- Google Cloud SDK (for GCS access)
- Required packages (see `requirements.txt`):
  - `pandas`, `numpy`, `scikit-learn`, `joblib`
  - `google-cloud-storage`

## Environment Variables

Set these environment variables for GCS access and model paths:

```bash
# GCS Configuration
export BUCKET_NAME="thomasanalytics-data1"
export COIN_DATA_PATH="lake/coins/usd_market_cap_desc/"
export DEFAULT_MODEL_BUCKET_PATH="lake/models/btc_price_model/pkl/prod"
export DEFAULT_MODEL_BUCKET_PATH_V1_1="lake/models/btc_price_model/pkl/prod"  # Optional: v1.1 specific path
```

## Training and Saving Models

Use `builder.py` to train new models and save them locally or to GCS.

### Basic Training (Local Data)

```bash
# Train with local JSON files
python builder.py --data-paths snapshot1.json snapshot2.json snapshot3.json

# Save trained model locally
python builder.py --data-paths snapshot1.json snapshot2.json --save-model btc_model_v1_1.pkl
```

### Training with GCS Data

```bash
# Train with latest N snapshots from GCS (default: 10)
python builder.py --bucket-path gs://thomasanalytics-data1/lake/coins/usd_market_cap_desc/ --latest-n 10

# Train with 100 snapshots and save locally
python builder.py --bucket-path gs://thomasanalytics-data1/lake/coins/usd_market_cap_desc/ --latest-n 100 --save-model btc_model_v1_1_100x.pkl

# Train and save to GCS bucket automatically
python builder.py --bucket-path gs://thomasanalytics-data1/lake/coins/usd_market_cap_desc/ --latest-n 50 --save-model-to-bucket
```

### Command-Line Options for `builder.py`

| Option | Description | Example |
|-------|-------------|----------|
| `--bucket-path` | GCS path to directory containing snapshots | `gs://bucket/path/to/dir/` |
| `--latest-n` | Number of latest snapshots to use (default: 10) | `--latest-n 100` |
| `--save-model` | Save model to local file | `--save-model model.pkl` |
| `--save-model-to-bucket` | Save model to GCS (uses `DEFAULT_MODEL_BUCKET_PATH`) | `--save-model-to-bucket` |
| `--load-model` | Load existing model (skips training) | `--load-model model.pkl` |
| `--data-path` | Single local file or GCS file path | `--data-path snapshot.json` |
| `--data-paths` | Multiple local files | `--data-paths snap1.json snap2.json` |

### Model File Naming

When saving to GCS, models are automatically named:
- Format: `model_v1.2_{n_samples}samples_{timestamp}.pkl`
- Example: `model_v1.2_98samples_20260102_123456.pkl`

## Running Predictions

Use `runner.py` to make predictions with trained models.

### Using Default GCS Production Model

```bash
# Use latest model from default GCS prod path (recommended for production)
python runner.py
```

### Using Local Model File

```bash
# Use a local model file
python runner.py --model btc_model_v1_1_100x.pkl

# Or use environment variable
export MODEL_FILE=btc_model_v1_1_100x.pkl
python runner.py
```

### Using Model from GCS

```bash
# Use a specific model from GCS
python runner.py --model gs://thomasanalytics-data1/lake/models/btc_price_model/pkl/prod/model_v1.2_98samples_20260102_123456.pkl
```

### Command-Line Options for `runner.py`

| Option | Description | Example |
|-------|-------------|----------|
| `--model` or `--model-file` | Path to model file (local or GCS). If not provided, uses latest from default GCS path | `--model model.pkl` or `--model gs://bucket/path.pkl` |

### Output Format

The `runner.py` script outputs JSON with the following structure:

```json
{
  "predicted_price": 88577.07,
  "current_price": 88709.0,
  "model_version": "1.2",
  "prediction_target": "Future 5-minute interval BTC price",
  "note": "v1.1 predicts the NEXT 5-minute interval price (future price)",
  "current_timestamp": "2026-01-02 00:49:52",
  "predicted_timestamp": "2026-01-02 00:54:52",
  "prediction_time": "2026-01-02 00:54:52"
}
```

## Cloud Run Job Integration

The `main.py` script is the entry point for Cloud Run Jobs. It:

1. Imports `predict_btc_price` from `runner.py`
2. Calls it with `model_file=None` (uses default GCS prod model)
3. Writes the result to the `task_events` table in the database

### Running Locally (for testing)

```bash
# Set database environment variables
export DB_USER="your_user"
export DB_PASSWORD="your_password"
export DB_HOST="your_host"
export DB_NAME="your_database"
export BUCKET_NAME="thomasanalytics-data1"

# Run the job
python main.py
```

## Common Workflows

### Workflow 1: Train New Model with Latest Data

```bash
# Train with 100 latest snapshots and save to GCS
python builder.py \
  --bucket-path gs://thomasanalytics-data1/lake/coins/usd_market_cap_desc/ \
  --latest-n 100 \
  --save-model-to-bucket
```

### Workflow 2: Test Model Locally

```bash
# Train and save locally
python builder.py \
  --bucket-path gs://thomasanalytics-data1/lake/coins/usd_market_cap_desc/ \
  --latest-n 10 \
  --save-model btc_model_test.pkl

# Test the model
python runner.py --model btc_model_test.pkl
```

### Workflow 3: Production Prediction

```bash
# Use latest production model from GCS (automatic)
python runner.py
```

## Model Architecture

The v1.2 model uses:

- **Preprocessing Pipeline**: Missing value handling → Outlier detection → Feature scaling
- **Model**: Ridge Regression with cross-validation for hyperparameter tuning
- **Features**: 30 engineered features including:
  - Moving averages (10, 20, 50, 200 periods)
  - RSI (Relative Strength Index)
  - VWAP (Volume Weighted Average Price)
  - Price ratios (BTC/ETH, BTC/MA ratios)
  - Market indicators (dominance, volume, market cap)
  - Interaction terms

## Troubleshooting

### Model Loading Issues

If you see `NotFittedError` or pipeline fitting warnings:
- The model may have been saved with an older version
- Retrain and save a new model with the latest code
- The model will still work but may show warnings

### GCS Access Issues

Ensure you have:
1. Google Cloud SDK installed and authenticated: `gcloud auth login`
2. Application Default Credentials set: `gcloud auth application-default login`
3. Proper IAM permissions for the bucket

### Missing Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

## File Structure

```
transformer-predict-coin-price-v1/
├── builder.py              # Train and save models
├── runner.py               # Run predictions
├── main.py                 # Cloud Run Job entry point
├── btc_price_model_v1_1.py # Model implementation
├── Dockerfile              # Container definition
├── job.yaml                # Cloud Run Job configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Notes

- The model predicts the **future** 5-minute interval price, not the current price
- Historical snapshots are required for proper feature engineering (moving averages, etc.)
- Models are saved in v1.2 format with separate preprocessing transformers for robustness
- The default GCS production path is: `lake/models/btc_price_model/pkl/prod`
