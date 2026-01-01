"""
Optimized script to get latest coin data, load default model, and return prediction.
Minimal overhead for maximum performance.
"""
import json
import os
from typing import Optional
from google.cloud import storage
from btc_price_model import BTCPricePredictor

# Cache for GCS storage client
_storage_client = None

def get_gcs_client():
    """Get or create a cached GCS storage client."""
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

def get_latest_data_file(bucket_path: str, bucket_name: Optional[str] = None) -> str:
    """Get the latest JSON file from GCS bucket path."""
    # Parse GCS path
    if bucket_path.startswith("gs://"):
        parts = bucket_path.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
    else:
        prefix = bucket_path
        if bucket_name is None:
            bucket_name = os.getenv('BUCKET_NAME', 'thomasanalytics-data1')
    
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    
    # Get latest file
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    json_blobs = [blob for blob in blobs if blob.name.endswith('.json')]
    if not json_blobs:
        raise ValueError(f"No JSON files found in gs://{bucket_name}/{prefix}")
    
    json_blobs.sort(key=lambda x: x.time_created, reverse=True)
    latest_blob = json_blobs[0]
    
    return f"gs://{bucket_name}/{latest_blob.name}"

def get_latest_model_file(bucket_path: str, bucket_name: Optional[str] = None) -> str:
    """Get the latest model file (.pkl) from GCS bucket path."""
    # Parse GCS path
    if bucket_path.startswith("gs://"):
        parts = bucket_path.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
    else:
        prefix = bucket_path
        if bucket_name is None:
            bucket_name = os.getenv('BUCKET_NAME', 'thomasanalytics-data1')
    
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    
    # Get latest model
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    pkl_blobs = [blob for blob in blobs if blob.name.endswith('.pkl')]
    if not pkl_blobs:
        raise ValueError(f"No model files (.pkl) found in gs://{bucket_name}/{prefix}")
    
    pkl_blobs.sort(key=lambda x: x.time_created, reverse=True)
    latest_blob = pkl_blobs[0]
    
    return f"gs://{bucket_name}/{latest_blob.name}"

def load_data_from_gcs(gcs_path: str) -> list:
    """Load JSON data from GCS path."""
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]
    
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    file_content = blob.download_as_bytes()
    return json.loads(file_content.decode('utf-8'))

def predict_btc_price() -> dict:
    """
    Get latest coin data, load default model, and return prediction results.
    
    Returns:
        dict: Prediction results with predicted_price, actual_price, difference, error_percentage
    """
    # Get paths from environment variables or use defaults
    coin_data_path = os.getenv('COIN_DATA_PATH', 'lake/coins/usd_market_cap_desc/')
    model_path = os.getenv('DEFAULT_MODEL_BUCKET_PATH', 'lake/models/btc_price_model/pkl/prod')
    bucket_name = os.getenv('BUCKET_NAME', 'thomasanalytics-data1')
    
    # Get latest data file
    latest_data_file = get_latest_data_file(coin_data_path, bucket_name)
    
    # Get latest model file
    latest_model_file = get_latest_model_file(model_path, bucket_name)
    
    # Load model
    model = BTCPricePredictor.load_model(latest_model_file)
    
    # Load data
    snapshot_data = load_data_from_gcs(latest_data_file)
    
    # Get actual BTC price
    import pandas as pd
    df = pd.DataFrame(snapshot_data)
    btc_data = df[df['symbol'] == 'btc'].iloc[0] if len(df[df['symbol'] == 'btc']) > 0 else None
    if btc_data is None:
        raise ValueError("BTC data not found in snapshot")
    actual_price = float(btc_data['current_price'])
    
    # Make prediction
    predicted_price = model.predict(snapshot_data)
    
    # Return results
    return {
        "predicted_price": float(predicted_price),
        "actual_price": actual_price,
        "difference": abs(predicted_price - actual_price),
        "error_percentage": abs(predicted_price - actual_price) / actual_price * 100
    }

if __name__ == "__main__":
    result = predict_btc_price()
    print(json.dumps(result, indent=2))
