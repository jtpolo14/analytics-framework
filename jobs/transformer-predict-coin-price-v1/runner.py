"""
Optimized script to get latest coin data, load default model (v1.1), and return prediction.
Minimal overhead for maximum performance.
Uses v1.1 model which predicts future 5-minute interval BTC price.

Usage:
    python test.py                          # Use latest model from GCS
    python test.py --model model.pkl         # Use local model file
    python test.py --model gs://bucket/path # Use model from GCS path
    MODEL_FILE=model.pkl python test.py      # Use environment variable
"""
import json
import os
import sys
import argparse
from typing import Optional
from datetime import datetime, timedelta
from google.cloud import storage
from btc_price_model_v1_1 import BTCPricePredictor, extract_btc_from_snapshot

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

def predict_btc_price(model_file: Optional[str] = None) -> dict:
    """
    Get latest coin data, load model (v1.1), and return prediction results.
    Uses v1.1 model which predicts future 5-minute interval BTC price.
    
    Args:
        model_file: Optional path to model file (local .pkl or GCS gs:// path).
                   If None, loads latest model from default GCS path.
    
    Returns:
        dict: Prediction results with predicted_price, actual_price, difference, error_percentage
    """
    # Get paths from environment variables or use defaults
    coin_data_path = os.getenv('COIN_DATA_PATH', 'lake/coins/usd_market_cap_desc/')
    model_path = os.getenv('DEFAULT_MODEL_BUCKET_PATH', 'lake/models/btc_price_model/pkl/prod')
    bucket_name = os.getenv('BUCKET_NAME', 'thomasanalytics-data1')
    
    # Get latest data file
    latest_data_file = get_latest_data_file(coin_data_path, bucket_name)
    
    # Load model - use provided model_file, or get latest from GCS
    if model_file:
        # Use provided model file (local or GCS)
        print(f"Loading model from: {model_file}")
        model = BTCPricePredictor.load_model(model_file)
    else:
        # Get latest model file from GCS (prefer v1.1 models if available)
        # Try v1.1 path first, fallback to default
        model_path_v1_1 = os.getenv('DEFAULT_MODEL_BUCKET_PATH_V1_1', model_path)
        try:
            latest_model_file = get_latest_model_file(model_path_v1_1, bucket_name)
            print(f"Loading latest v1.1 model from: {latest_model_file}")
        except ValueError:
            # Fallback to default path if v1.1 path doesn't exist
            latest_model_file = get_latest_model_file(model_path, bucket_name)
            print(f"Loading latest model from: {latest_model_file}")
        
        # Load model
        model = BTCPricePredictor.load_model(latest_model_file)
    
    # Load data
    snapshot_data = load_data_from_gcs(latest_data_file)
    
    # Get current BTC price and timestamp
    btc_data = extract_btc_from_snapshot(snapshot_data)
    if btc_data is None:
        raise ValueError("BTC data not found in snapshot")
    current_price = float(btc_data.get('current_price', 0))
    
    # Extract timestamp and calculate predicted time (current + 5 minutes)
    last_updated = btc_data.get('last_updated', None)
    current_time_str = None
    predicted_time_str = None
    
    if last_updated:
        try:
            # Parse ISO format timestamp: "2025-12-27T10:46:21.358Z"
            # Remove milliseconds and Z if present
            time_str = last_updated.split('.')[0].replace('Z', '')
            current_time = datetime.fromisoformat(time_str)
            current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add 5 minutes for predicted time
            predicted_time = current_time + timedelta(minutes=5)
            predicted_time_str = predicted_time.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            # If parsing fails, use original string
            current_time_str = str(last_updated)
            predicted_time_str = None
    
    # For v1.1, we need historical data for technical indicators
    # Try to get previous snapshot for history (optional - model will work without it)
    btc_price_history = None
    btc_volume_history = None
    try:
        # Get second latest file for history
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        parts = coin_data_path.replace("gs://", "").split("/", 1) if coin_data_path.startswith("gs://") else ("", coin_data_path)
        prefix = parts[1] if len(parts) > 1 else coin_data_path
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        
        blobs = list(bucket.list_blobs(prefix=prefix))
        json_blobs = [blob for blob in blobs if blob.name.endswith('.json')]
        json_blobs.sort(key=lambda x: x.time_created, reverse=True)
        
        if len(json_blobs) >= 2:
            # Get second latest for history
            prev_data_file = f"gs://{bucket_name}/{json_blobs[1].name}"
            prev_snapshot = load_data_from_gcs(prev_data_file)
            prev_btc_data = extract_btc_from_snapshot(prev_snapshot)
            if prev_btc_data:
                btc_price_history = [prev_btc_data.get('current_price', 0)]
                btc_volume_history = [prev_btc_data.get('total_volume', 0)]
    except Exception:
        # If we can't get history, continue without it (model will use fallback values)
        pass
    
    # Make prediction (v1.1 predicts future price)
    predicted_price = model.predict(
        snapshot_data,
        btc_price_history=btc_price_history,
        btc_volume_history=btc_volume_history
    )
    
    # Get model info
    model_info = model.get_model_info()
    
    # Return results
    result = {
        "predicted_price": float(predicted_price),
        "current_price": current_price,
        "model_version": model_info.get('version', '1.0'),
        "prediction_target": model_info.get('prediction_target', 'Current BTC price'),
        "note": "v1.1 predicts the NEXT 5-minute interval price (future price)"
    }
    
    # Add timestamp information if available
    if current_time_str:
        result["current_timestamp"] = current_time_str
    if predicted_time_str:
        result["predicted_timestamp"] = predicted_time_str
        result["prediction_time"] = predicted_time_str  # Alias for convenience
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test BTC price prediction with v1.1 model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest model from GCS (default)
  python test.py
  
  # Use local model file
  python test.py --model btc_model_v1_1_5x_test.pkl
  
  # Use model from GCS path
  python test.py --model gs://bucket-name/path/to/model.pkl
  
  # Use environment variable
  MODEL_FILE=model.pkl python test.py
        """
    )
    parser.add_argument(
        '--model',
        '--model-file',
        type=str,
        dest='model_file',
        help='Path to model file (.pkl) - local file or GCS path (gs://). '
             'If not provided, loads latest model from default GCS path. '
             'Can also be set via MODEL_FILE environment variable.',
        default=None
    )
    
    args = parser.parse_args()
    
    # Check environment variable if not provided via command line
    model_file = args.model_file or os.getenv('MODEL_FILE')
    
    try:
        result = predict_btc_price(model_file=model_file)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
