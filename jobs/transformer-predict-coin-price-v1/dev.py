"""
Development runner script for cryptocurrency price prediction.

This script handles data loading and orchestration. The actual model logic
is in btc_price_model.py, allowing easy model swapping and testing.

DATA SOURCES:
============

The script supports multiple ways to load data:

1. LOCAL FILES:
   python dev.py --data-path coins.json
   python dev.py --data-paths snapshot1.json snapshot2.json snapshot3.json

2. GCS BUCKET:
   python dev.py --data-path gs://bucket-name/path/to/file.json
   python dev.py --data-path path/to/file.json --bucket-name my-bucket
   
   # Get latest N files from a bucket directory (default: 10)
   python dev.py --bucket-path gs://bucket-name/path/to/dir/
   python dev.py --bucket-path gs://bucket-name/path/to/dir/ --latest-n 5

3. ENVIRONMENT VARIABLES:
   export DATA_PATH=coins.json
   export DATA_PATHS=snapshot1.json,snapshot2.json,snapshot3.json
   python dev.py

4. HARDCODED DATA (default):
   Add snapshots to HISTORICAL_SNAPSHOTS list in the code

MODEL SAVE/LOAD:
================

Save a trained model locally:
   python dev.py --data-paths snap1.json snap2.json --save-model model.pkl

Save a trained model to GCS bucket:
   python dev.py --data-paths snap1.json snap2.json --save-model-to-bucket
   # Uses MODEL_BUCKET_PATH env var (default: lake/models/btc_price_model/pkl)
   # Filename will be: model_{n_samples}samples_{timestamp}.pkl
   # Example: model_9samples_20251227_143022.pkl

Load a saved model (skips training):
   python dev.py --load-model model.pkl --data-path new_snapshot.json

MODEL CUSTOMIZATION:
====================

To use a different model, modify the model import and instantiation in main():
   from btc_price_model import BTCPricePredictor
   from sklearn.ensemble import RandomForestRegressor
   
   model = BTCPricePredictor(model=RandomForestRegressor())
"""
import json
import os
import argparse
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Import the model class
from btc_price_model import BTCPricePredictor

# GCS imports (optional - only needed if using GCS)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not available. GCS functionality will be disabled.")


# Sample data frame - in production, this would come from an API or database
SAMPLE_DATA = [
    {"id": "bitcoin", "symbol": "btc", "current_price": 87387, "total_volume": 31377869299, "price_change_24h": -1320.77},
    {"id": "ethereum", "symbol": "eth", "current_price": 2922.48, "total_volume": 13751868665, "price_change_24h": -46.61},
    {"id": "solana", "symbol": "sol", "current_price": 122.71, "total_volume": 2894287297, "price_change_24h": -0.87}
]

# Full data frame with all features - SNAPSHOT 1 (2025-12-27)
# To add more snapshots, append them to HISTORICAL_SNAPSHOTS list below
FULL_DATA_SNAPSHOT_1 = [
    {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "image": "https://coin-images.coingecko.com/coins/images/1/large/bitcoin.png?1696501400", "current_price": 87387, "market_cap": 1744910741787, "market_cap_rank": 1, "fully_diluted_valuation": 1744910741787, "total_volume": 31377869299, "high_24h": 88966, "low_24h": 86674, "price_change_24h": -1320.779519833217, "price_change_percentage_24h": -1.48891, "market_cap_change_24h": -25955886647.253662, "market_cap_change_percentage_24h": -1.46572, "circulating_supply": 19967796.0, "total_supply": 19967796.0, "max_supply": 21000000.0, "ath": 126080, "ath_change_percentage": -30.67298, "ath_date": "2025-10-06T18:57:42.558Z", "atl": 67.81, "atl_change_percentage": 128802.40449, "atl_date": "2013-07-06T00:00:00.000Z", "roi": None, "last_updated": "2025-12-27T10:46:21.358Z", "price_change_percentage_1h_in_currency": -0.1767300677262785, "price_change_percentage_24h_in_currency": -1.488910610294202, "price_change_percentage_30d_in_currency": -4.574958723516482, "price_change_percentage_7d_in_currency": -0.9346742295377745},
    {"id": "ethereum", "symbol": "eth", "name": "Ethereum", "image": "https://coin-images.coingecko.com/coins/images/279/large/ethereum.png?1696501628", "current_price": 2922.48, "market_cap": 352547418866, "market_cap_rank": 2, "fully_diluted_valuation": 352547418866, "total_volume": 13751868665, "high_24h": 2983.07, "low_24h": 2901.84, "price_change_24h": -46.61893618882368, "price_change_percentage_24h": -1.57013, "market_cap_change_24h": -5749199363.53186, "market_cap_change_percentage_24h": -1.60459, "circulating_supply": 120694960.0883406, "total_supply": 120694960.0883406, "max_supply": None, "ath": 4946.05, "ath_change_percentage": -40.88438, "ath_date": "2025-08-24T19:21:03.333Z", "atl": 0.432979, "atl_change_percentage": 675195.59105, "atl_date": "2015-10-20T00:00:00.000Z", "roi": {"times": 43.71559942051733, "currency": "btc", "percentage": 4371.559942051733}, "last_updated": "2025-12-27T10:46:21.339Z", "price_change_percentage_1h_in_currency": -0.3506828551721171, "price_change_percentage_24h_in_currency": -1.5701349891386853, "price_change_percentage_30d_in_currency": -3.432313376704394, "price_change_percentage_7d_in_currency": -1.9900446567724945},
    {"id": "tether", "symbol": "usdt", "name": "Tether", "image": "https://coin-images.coingecko.com/coins/images/325/large/Tether.png?1696501661", "current_price": 0.999512, "market_cap": 186786011203, "market_cap_rank": 3, "fully_diluted_valuation": 192250233789, "total_volume": 50257709985, "high_24h": 0.999585, "low_24h": 0.999211, "price_change_24h": 2.8e-05, "price_change_percentage_24h": 0.0028, "market_cap_change_24h": -7123641.7059021, "market_cap_change_percentage_24h": -0.00381, "circulating_supply": 186878783369.7212, "total_supply": 192345719904.5362, "max_supply": None, "ath": 1.32, "ath_change_percentage": -24.45875, "ath_date": "2018-07-24T00:00:00.000Z", "atl": 0.572521, "atl_change_percentage": 74.57574, "atl_date": "2015-03-02T00:00:00.000Z", "roi": None, "last_updated": "2025-12-27T10:46:18.758Z", "price_change_percentage_1h_in_currency": -0.006294873166242567, "price_change_percentage_24h_in_currency": 0.0028019029494946866, "price_change_percentage_30d_in_currency": -0.04808181216314098, "price_change_percentage_7d_in_currency": 0.0001597677922126978},
    {"id": "binancecoin", "symbol": "bnb", "name": "BNB", "image": "https://coin-images.coingecko.com/coins/images/825/large/bnb-icon2_2x.png?1696501970", "current_price": 840.43, "market_cap": 115766120593, "market_cap_rank": 4, "fully_diluted_valuation": 115766120593, "total_volume": 940644720, "high_24h": 842.71, "low_24h": 823.36, "price_change_24h": 0.191344, "price_change_percentage_24h": 0.02277, "market_cap_change_24h": 37995903, "market_cap_change_percentage_24h": 0.03283, "circulating_supply": 137734766.61, "total_supply": 137734766.61, "max_supply": 200000000.0, "ath": 1369.99, "ath_change_percentage": -38.62185, "ath_date": "2025-10-13T08:41:24.131Z", "atl": 0.0398177, "atl_change_percentage": 2111713.56279, "atl_date": "2017-10-19T00:00:00.000Z", "roi": None, "last_updated": "2025-12-27T10:46:21.237Z", "price_change_percentage_1h_in_currency": 0.16981770693081225, "price_change_percentage_24h_in_currency": 0.022772697023168975, "price_change_percentage_30d_in_currency": -5.895820337782824, "price_change_percentage_7d_in_currency": -1.498188838869461},
    {"id": "ripple", "symbol": "xrp", "name": "XRP", "image": "https://coin-images.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png?1696501442", "current_price": 1.85, "market_cap": 111815866117, "market_cap_rank": 5, "fully_diluted_valuation": 184570723564, "total_volume": 1617503928, "high_24h": 1.88, "low_24h": 1.83, "price_change_24h": -0.030207345761184, "price_change_percentage_24h": -1.61018, "market_cap_change_24h": -1798581825.8467865, "market_cap_change_percentage_24h": -1.58306, "circulating_supply": 60572944636.0, "total_supply": 99985740916.0, "max_supply": 100000000000.0, "ath": 3.65, "ath_change_percentage": -49.31741, "ath_date": "2025-07-18T03:40:53.808Z", "atl": 0.00268621, "atl_change_percentage": 68698.82424, "atl_date": "2014-05-22T00:00:00.000Z", "roi": None, "last_updated": "2025-12-27T10:46:27.066Z", "price_change_percentage_1h_in_currency": -0.2477301688599141, "price_change_percentage_24h_in_currency": -1.6101836879376237, "price_change_percentage_30d_in_currency": -16.049499307536546, "price_change_percentage_7d_in_currency": -4.377525256823691},
    {"id": "usd-coin", "symbol": "usdc", "name": "USDC", "image": "https://coin-images.coingecko.com/coins/images/6319/large/usdc.png?1696506694", "current_price": 0.999987, "market_cap": 76491291897, "market_cap_rank": 6, "fully_diluted_valuation": 76505391132, "total_volume": 7147311256, "high_24h": 1.0, "low_24h": 0.999561, "price_change_24h": 0.00017917, "price_change_percentage_24h": 0.01792, "market_cap_change_24h": -54709892.205703735, "market_cap_change_percentage_24h": -0.07147, "circulating_supply": 76491523966.71927, "total_supply": 76505623244.37384, "max_supply": None, "ath": 1.17, "ath_change_percentage": -14.75174, "ath_date": "2019-05-08T00:40:28.300Z", "atl": 0.877647, "atl_change_percentage": 13.90822, "atl_date": "2023-03-11T08:02:13.981Z", "roi": None, "last_updated": "2025-12-27T10:46:06.485Z", "price_change_percentage_1h_in_currency": 0.008665975570901648, "price_change_percentage_24h_in_currency": 0.017920093653517995, "price_change_percentage_30d_in_currency": 0.018265935397445414, "price_change_percentage_7d_in_currency": 0.006793900604644746},
    {"id": "solana", "symbol": "sol", "name": "Solana", "image": "https://coin-images.coingecko.com/coins/images/4128/large/solana.png?1718769756", "current_price": 122.71, "market_cap": 69046850304, "market_cap_rank": 7, "fully_diluted_valuation": 75715342406, "total_volume": 2894287297, "high_24h": 124.87, "low_24h": 120.98, "price_change_24h": -0.8764056419238102, "price_change_percentage_24h": -0.70912, "market_cap_change_24h": -460260140.37809753, "market_cap_change_percentage_24h": -0.66218, "circulating_supply": 562657510.4532143, "total_supply": 616998543.3636216, "max_supply": None, "ath": 293.31, "ath_change_percentage": -58.15336, "ath_date": "2025-01-19T11:15:27.957Z", "atl": 0.500801, "atl_change_percentage": 24408.96814, "atl_date": "2020-05-11T19:35:23.449Z", "roi": None, "last_updated": "2025-12-27T10:46:06.318Z", "price_change_percentage_1h_in_currency": -0.25170616461276957, "price_change_percentage_24h_in_currency": -0.709116343693594, "price_change_percentage_30d_in_currency": -13.850580132787224, "price_change_percentage_7d_in_currency": -2.8151444042509643}
]

# Example: How to add a second snapshot
# FULL_DATA_SNAPSHOT_2 = [
#     {"id": "bitcoin", "symbol": "btc", "current_price": 88000, ...},  # New BTC price
#     {"id": "ethereum", "symbol": "eth", "current_price": 2950, ...},  # New ETH price
#     # ... rest of your updated data with new prices/values
# ]

# Historical snapshots - List of all data snapshots for training
# Each snapshot is a list of cryptocurrency data dictionaries from a specific time point
# To add a new snapshot:
#   1. Create a new variable like FULL_DATA_SNAPSHOT_2 with your data (see example above)
#   2. Add it to this list below
HISTORICAL_SNAPSHOTS = [
    FULL_DATA_SNAPSHOT_1,
    # Add more snapshots here as you collect them
    # Example: FULL_DATA_SNAPSHOT_2, FULL_DATA_SNAPSHOT_3, etc.
    # 
    # When you add FULL_DATA_SNAPSHOT_2, uncomment the line below:
    # FULL_DATA_SNAPSHOT_2,
]

# For backward compatibility
FULL_DATA = FULL_DATA_SNAPSHOT_1


def load_data_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load cryptocurrency data from a local JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of cryptocurrency data dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded data from local file: {file_path}")
        print(f"Found {len(data)} cryptocurrency records")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {str(e)}")


# Cache for GCS storage client (reuse across multiple file loads)
_gcs_storage_client = None

def get_gcs_client():
    """Get or create a cached GCS storage client."""
    global _gcs_storage_client
    if _gcs_storage_client is None:
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is not installed. Install it with: pip install google-cloud-storage")
        _gcs_storage_client = storage.Client()
    return _gcs_storage_client

def load_data_from_gcs(bucket_path: str, bucket_name: Optional[str] = None, storage_client=None) -> List[Dict[str, Any]]:
    """
    Load cryptocurrency data from a Google Cloud Storage bucket.
    
    Args:
        bucket_path: Path to the file in GCS (e.g., "path/to/file.json" or "gs://bucket-name/path/to/file.json")
        bucket_name: Optional bucket name. If not provided, will try to extract from bucket_path
                     or use BUCKET_NAME environment variable
        storage_client: Optional pre-created storage client (for performance optimization)
        
    Returns:
        List of cryptocurrency data dictionaries
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install it with: pip install google-cloud-storage")
    
    try:
        # Parse GCS path
        if bucket_path.startswith("gs://"):
            # Format: gs://bucket-name/path/to/file.json
            parts = bucket_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            file_path = parts[1] if len(parts) > 1 else ""
        else:
            # Format: path/to/file.json (bucket_name must be provided or from env)
            file_path = bucket_path
            if bucket_name is None:
                bucket_name = os.getenv('BUCKET_NAME')
                if bucket_name is None:
                    raise ValueError("bucket_name must be provided or BUCKET_NAME environment variable must be set")
        
        # Check if path ends with / (directory path)
        if file_path.endswith('/') or file_path == "":
            raise ValueError(
                f"Path appears to be a directory, not a file: gs://{bucket_name}/{file_path}\n"
                f"Use --bucket-path instead of --data-path to get latest files from a directory.\n"
                f"Example: python dev.py --bucket-path gs://{bucket_name}/{file_path}"
            )
        
        # Reuse provided client or get cached one
        if storage_client is None:
            storage_client = get_gcs_client()
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        # Check if blob exists
        if not blob.exists():
            raise FileNotFoundError(
                f"File not found: gs://{bucket_name}/{file_path}\n"
                f"If this is a directory, use --bucket-path instead of --data-path"
            )
        
        # Download and parse JSON
        file_content = blob.download_as_bytes()
        
        if len(file_content) == 0:
            raise ValueError(f"File is empty: gs://{bucket_name}/{file_path}")
        
        data = json.loads(file_content.decode('utf-8'))
        
        print(f"Loaded data from GCS: gs://{bucket_name}/{file_path}")
        print(f"Found {len(data)} cryptocurrency records")
        return data
        
    except Exception as e:
        raise RuntimeError(f"Error reading from GCS {bucket_path}: {str(e)}")


def load_data_from_path(data_path: str, bucket_name: Optional[str] = None, storage_client=None) -> List[Dict[str, Any]]:
    """
    Load cryptocurrency data from either a local file or GCS bucket path.
    Automatically detects the source type.
    
    Args:
        data_path: Path to the data file. Can be:
                  - Local file path: "/path/to/file.json" or "file.json"
                  - GCS path: "gs://bucket-name/path/to/file.json" or "path/to/file.json" (with bucket_name)
        bucket_name: Optional bucket name for GCS (if not in gs:// format)
        storage_client: Optional pre-created GCS storage client (for performance optimization)
        
    Returns:
        List of cryptocurrency data dictionaries
    """
    # Check if it's a GCS path
    if data_path.startswith("gs://") or bucket_name is not None:
        return load_data_from_gcs(data_path, bucket_name, storage_client)
    else:
        # Assume local file
        return load_data_from_file(data_path)


def get_latest_files_from_bucket(
    bucket_path: str,
    n_files: int = 10,
    bucket_name: Optional[str] = None,
    storage_client=None
) -> List[str]:
    """
    Get the latest N files from a GCS bucket path.
    
    Args:
        bucket_path: GCS bucket path (e.g., "gs://bucket-name/path/to/dir/" or "path/to/dir/")
        n_files: Number of latest files to retrieve (default: 10)
        bucket_name: Optional bucket name (if not in gs:// format)
        storage_client: Optional pre-created storage client (for performance optimization)
        
    Returns:
        List of full GCS paths to the latest files
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install it with: pip install google-cloud-storage")
    
    try:
        # Parse GCS path
        if bucket_path.startswith("gs://"):
            # Format: gs://bucket-name/path/to/dir/
            parts = bucket_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
        else:
            # Format: path/to/dir/ (bucket_name must be provided or from env)
            prefix = bucket_path
            if bucket_name is None:
                bucket_name = os.getenv('BUCKET_NAME')
                if bucket_name is None:
                    raise ValueError("bucket_name must be provided or BUCKET_NAME environment variable must be set")
        
        # Ensure prefix ends with / if it's a directory
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        
        print(f"Searching for latest {n_files} files in gs://{bucket_name}/{prefix}")
        
        # List files in the bucket
        list_start = time.time()
        # Reuse provided client or get cached one
        if storage_client is None:
            storage_client = get_gcs_client()
        bucket = storage_client.bucket(bucket_name)
        
        # List all blobs with the prefix
        blobs = list(bucket.list_blobs(prefix=prefix))
        list_duration = time.time() - list_start
        
        # Filter for JSON files and sort by time created (newest first)
        json_blobs = [blob for blob in blobs if blob.name.endswith('.json')]
        json_blobs.sort(key=lambda x: x.time_created, reverse=True)
        
        # Get the latest N files
        latest_blobs = json_blobs[:n_files]
        
        if len(latest_blobs) == 0:
            raise ValueError(f"No JSON files found in gs://{bucket_name}/{prefix}")
        
        # Reverse to get ascending chronological order (oldest first) for training
        # This ensures the model trains on historical data in the correct order
        latest_blobs.reverse()
        
        # Return full GCS paths in ascending chronological order (oldest to newest)
        file_paths = [f"gs://{bucket_name}/{blob.name}" for blob in latest_blobs]
        
        print(f"Found {len(latest_blobs)} file(s) (requested {n_files}) in {list_duration:.3f}s")
        print("Files in chronological order (oldest to newest):")
        for i, path in enumerate(file_paths, 1):
            print(f"  {i}. {path}")
        
        return file_paths
        
    except Exception as e:
        raise RuntimeError(f"Error getting latest files from GCS {bucket_path}: {str(e)}")


def get_latest_model_from_bucket(
    bucket_path: str,
    bucket_name: Optional[str] = None,
    storage_client=None
) -> str:
    """
    Get the latest model file (.pkl) from a GCS bucket path.
    
    Args:
        bucket_path: GCS bucket path (e.g., "gs://bucket-name/path/to/dir/" or "path/to/dir/")
        bucket_name: Optional bucket name (if not in gs:// format)
        storage_client: Optional pre-created storage client (for performance optimization)
        
    Returns:
        str: Full GCS path to the latest model file
    """
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is not installed. Install it with: pip install google-cloud-storage")
    
    try:
        # Parse GCS path
        if bucket_path.startswith("gs://"):
            # Format: gs://bucket-name/path/to/dir/
            parts = bucket_path.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
        else:
            # Format: path/to/dir/ (bucket_name must be provided or from env)
            prefix = bucket_path
            if bucket_name is None:
                bucket_name = os.getenv('BUCKET_NAME')
                if bucket_name is None:
                    raise ValueError("bucket_name must be provided or BUCKET_NAME environment variable must be set")
        
        # Ensure prefix ends with / if it's a directory
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        
        print(f"Searching for latest model file in gs://{bucket_name}/{prefix}")
        
        # List files in the bucket
        list_start = time.time()
        # Reuse provided client or get cached one
        if storage_client is None:
            storage_client = get_gcs_client()
        bucket = storage_client.bucket(bucket_name)
        
        # List all blobs with the prefix
        blobs = list(bucket.list_blobs(prefix=prefix))
        list_duration = time.time() - list_start
        
        # Filter for .pkl files and sort by time created (newest first)
        pkl_blobs = [blob for blob in blobs if blob.name.endswith('.pkl')]
        
        if len(pkl_blobs) == 0:
            raise ValueError(f"No model files (.pkl) found in gs://{bucket_name}/{prefix}")
        
        # Sort by time created (newest first) and get the latest one
        pkl_blobs.sort(key=lambda x: x.time_created, reverse=True)
        latest_blob = pkl_blobs[0]
        
        # Return full GCS path
        file_path = f"gs://{bucket_name}/{latest_blob.name}"
        
        print(f"Found latest model: {latest_blob.name} (created: {latest_blob.time_created}) in {list_duration:.3f}s")
        
        return file_path
        
    except Exception as e:
        raise RuntimeError(f"Error getting latest model from GCS {bucket_path}: {str(e)}")


def load_multiple_snapshots(data_paths: List[str], bucket_name: Optional[str] = None) -> tuple:
    """
    Load multiple data snapshots from files or GCS paths.
    Optimized to reuse GCS storage client for better performance.
    
    Args:
        data_paths: List of paths (can mix local files and GCS paths)
        bucket_name: Optional bucket name for GCS paths
        
    Returns:
        tuple: (snapshots, timing_info) where:
            - snapshots: List of snapshots, where each snapshot is a list of cryptocurrency data dictionaries
            - timing_info: Dictionary with timing and data statistics
    """
    start_time = time.time()
    snapshots = []
    total_records = 0
    load_times = []
    
    # Reuse GCS client for all GCS operations (performance optimization)
    storage_client = None
    if any(path.startswith("gs://") or bucket_name for path in data_paths):
        if GCS_AVAILABLE:
            storage_client = get_gcs_client()
    
    for i, path in enumerate(data_paths, 1):
        print(f"\nLoading snapshot {i}/{len(data_paths)}: {path}")
        load_start = time.time()
        try:
            snapshot = load_data_from_path(path, bucket_name, storage_client)
            load_duration = time.time() - load_start
            load_times.append(load_duration)
            snapshots.append(snapshot)
            total_records += len(snapshot)
            print(f"  ✓ Loaded in {load_duration:.3f}s ({len(snapshot)} records)")
        except Exception as e:
            load_duration = time.time() - load_start
            print(f"  ✗ Failed in {load_duration:.3f}s: {str(e)}")
            print(f"  Skipping this snapshot...")
            continue
    
    if len(snapshots) == 0:
        raise ValueError("No valid snapshots were loaded. Check your file paths.")
    
    total_duration = time.time() - start_time
    avg_load_time = sum(load_times) / len(load_times) if load_times else 0
    
    timing_info = {
        "data_loading": {
            "total_duration_seconds": total_duration,
            "snapshots_loaded": len(snapshots),
            "total_records": total_records,
            "avg_load_time_per_snapshot": avg_load_time,
            "total_files_attempted": len(data_paths)
        }
    }
    
    print(f"\n✓ Successfully loaded {len(snapshots)} snapshot(s) in {total_duration:.3f}s")
    print(f"  Total records: {total_records:,}")
    print(f"  Average load time: {avg_load_time:.3f}s per snapshot")
    
    return snapshots, timing_info


def get_btc_price(snapshot_data: List[Dict[str, Any]]) -> float:
    """
    Extract BTC price from a snapshot.
    
    Args:
        snapshot_data: List of cryptocurrency data dictionaries
        
    Returns:
        float: BTC current price
    """
    df = pd.DataFrame(snapshot_data)
    btc_data = df[df['symbol'] == 'btc'].iloc[0] if len(df[df['symbol'] == 'btc']) > 0 else None
    if btc_data is None:
        raise ValueError("BTC data not found in snapshot")
    return float(btc_data['current_price'])


# Model-related functions are now in btc_price_model.py
# Only data loading and orchestration functions remain here


def train_and_predict_with_model(
    model: BTCPricePredictor,
    historical_snapshots: List[List[Dict[str, Any]]],
    prediction_snapshot: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train model on historical snapshots and optionally predict on a new snapshot.
    
    Args:
        model: BTCPricePredictor model instance
        historical_snapshots: List of historical data snapshots for training
        prediction_snapshot: Optional new snapshot to make prediction on.
                            If None, uses the last snapshot for prediction.
        
    Returns:
        dict: Dictionary containing prediction results and model information
    """
    if len(historical_snapshots) == 0:
        raise ValueError("At least one historical snapshot is required for training")
    
    timing_info = {}
    
    # Step 1: Build training dataset
    print("\n=== Step 1: Building Training Dataset from Historical Snapshots ===")
    step_start = time.time()
    X_train, y_train = model.build_training_dataset(historical_snapshots)
    step_duration = time.time() - step_start
    timing_info["dataset_building"] = {
        "duration_seconds": step_duration,
        "n_samples": len(X_train),
        "n_features": X_train.shape[1] if len(X_train) > 0 else 0
    }
    print(f"  ⏱️  Completed in {step_duration:.3f}s")
    
    print(f"\nTraining dataset shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Features: {model.feature_columns}")
    
    # Step 2: Train model
    print("\n=== Step 2: Model Training ===")
    step_start = time.time()
    if len(X_train) == 1:
        print("Warning: Only one training sample. Model will memorize the data.")
        print("For better predictions, collect more historical snapshots.")
    
    training_metrics = model.train(X_train, y_train, verbose=True)
    step_duration = time.time() - step_start
    timing_info["model_training"] = {
        "duration_seconds": step_duration,
        "n_samples": len(X_train),
        "n_features": X_train.shape[1] if len(X_train) > 0 else 0
    }
    print(f"  ⏱️  Completed in {step_duration:.3f}s")
    
    # Step 3: Make prediction on new snapshot (or last snapshot if none provided)
    if prediction_snapshot is None:
        prediction_snapshot = historical_snapshots[-1]
        print("\n=== Step 3: Making Prediction on Last Snapshot ===")
    else:
        print("\n=== Step 3: Making Prediction on New Snapshot ===")
    
    step_start = time.time()
    # Get actual BTC price for comparison
    actual_btc_price = get_btc_price(prediction_snapshot)
    
    # Make prediction
    predicted_price = model.predict(prediction_snapshot)
    step_duration = time.time() - step_start
    timing_info["prediction"] = {
        "duration_seconds": step_duration,
        "prediction_records": len(prediction_snapshot)
    }
    print(f"  ⏱️  Completed in {step_duration:.3f}s")
    
    print(f"\n=== Step 4: Prediction Results ===")
    print(f"Predicted BTC Price: ${predicted_price:,.2f}")
    print(f"Actual BTC Price: ${actual_btc_price:,.2f}")
    print(f"Difference: ${abs(predicted_price - actual_btc_price):,.2f}")
    print(f"Error Percentage: {abs(predicted_price - actual_btc_price) / actual_btc_price * 100:.2f}%")
    
    # Get model info
    model_info = model.get_model_info()
    
    # Model coefficients
    if "coefficients" in model_info:
        print(f"\nModel Coefficients (Top 10 by absolute value):")
        coef_dict = model_info["coefficients"]
        sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        for col, coef in sorted_coefs[:10]:
            print(f"  {col}: {coef:.6f}")
        if "intercept" in model_info:
            print(f"  Intercept: {model_info['intercept']:.2f}")
    
    return {
        "predicted_price": predicted_price,
        "actual_price": actual_btc_price,
        "difference": abs(predicted_price - actual_btc_price),
        "error_percentage": abs(predicted_price - actual_btc_price) / actual_btc_price * 100,
        "model_info": model_info,
        "training_metrics": training_metrics,
        "timing_info": timing_info,
        "note": f"Model trained on {len(historical_snapshots)} historical snapshots."
    }


def predict_with_loaded_model(
    model: BTCPricePredictor,
    snapshot_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Make prediction using a pre-trained model (no training step).
    
    Args:
        model: Pre-trained BTCPricePredictor model instance
        snapshot_data: Single snapshot of cryptocurrency data
        
    Returns:
        dict: Dictionary containing prediction results
    """
    timing_info = {}
    
    print("\n=== Making Prediction with Loaded Model ===")
    
    # Get actual BTC price for comparison
    step_start = time.time()
    actual_btc_price = get_btc_price(snapshot_data)
    step_duration = time.time() - step_start
    
    # Make prediction
    step_start = time.time()
    predicted_price = model.predict(snapshot_data)
    pred_duration = time.time() - step_start
    timing_info["prediction"] = {
        "duration_seconds": pred_duration,
        "prediction_records": len(snapshot_data)
    }
    print(f"  ⏱️  Prediction completed in {pred_duration:.3f}s")
    
    print(f"\n=== Prediction Results ===")
    print(f"Predicted BTC Price: ${predicted_price:,.2f}")
    print(f"Actual BTC Price: ${actual_btc_price:,.2f}")
    print(f"Difference: ${abs(predicted_price - actual_btc_price):,.2f}")
    print(f"Error Percentage: {abs(predicted_price - actual_btc_price) / actual_btc_price * 100:.2f}%")
    
    # Get model info
    model_info = model.get_model_info()
    
    return {
        "predicted_price": predicted_price,
        "actual_price": actual_btc_price,
        "difference": abs(predicted_price - actual_btc_price),
        "error_percentage": abs(predicted_price - actual_btc_price) / actual_btc_price * 100,
        "model_info": model_info,
        "timing_info": timing_info,
        "note": "Prediction made with pre-trained model (no training performed)."
    }


def predict_single_snapshot(
    model: BTCPricePredictor,
    snapshot_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Predict BTC price using a single snapshot (for demonstration).
    Note: This requires the model to be pre-trained.
    
    Args:
        model: Pre-trained BTCPricePredictor model instance
        snapshot_data: Single snapshot of cryptocurrency data
        
    Returns:
        dict: Dictionary containing prediction results
    """
    timing_info = {}
    
    print("\n=== Single Snapshot Prediction Mode ===")
    print("Warning: Model must be pre-trained for this to work properly.")
    print("For proper training, provide multiple snapshots.")
    
    actual_btc_price = get_btc_price(snapshot_data)
    print(f"Current BTC Price: ${actual_btc_price:,.2f}")
    
    if not model.is_trained:
        print("Model not trained. Training on single snapshot (will memorize data)...")
        # Train on the single snapshot
        step_start = time.time()
        X, y = model.build_training_dataset([snapshot_data])
        build_duration = time.time() - step_start
        timing_info["dataset_building"] = {"duration_seconds": build_duration}
        
        step_start = time.time()
        model.train(X, y, verbose=True)
        train_duration = time.time() - step_start
        timing_info["model_training"] = {"duration_seconds": train_duration}
    
    step_start = time.time()
    predicted_price = model.predict(snapshot_data)
    pred_duration = time.time() - step_start
    timing_info["prediction"] = {"duration_seconds": pred_duration}
    
    print(f"\nPredicted BTC Price: ${predicted_price:,.2f}")
    print(f"Actual BTC Price: ${actual_btc_price:,.2f}")
    print(f"Difference: ${abs(predicted_price - actual_btc_price):,.2f}")
    
    return {
        "predicted_price": predicted_price,
        "actual_price": actual_btc_price,
        "difference": abs(predicted_price - actual_btc_price),
        "model_info": model.get_model_info(),
        "timing_info": timing_info,
        "note": "Model trained on single data point. For production use, historical data is required."
    }


def main():
    """
    Main execution function.
    Supports loading data from:
    - Command-line arguments: --data-path or --data-paths
    - Environment variables: DATA_PATH or DATA_PATHS (comma-separated)
    - Default: Uses hardcoded HISTORICAL_SNAPSHOTS
    
    Examples:
        python dev.py --data-path coins.json
        python dev.py --data-paths snapshot1.json snapshot2.json snapshot3.json
        python dev.py --data-path gs://bucket-name/path/to/file.json
        python dev.py --data-path path/to/file.json --bucket-name my-bucket
    """
    parser = argparse.ArgumentParser(
        description='Cryptocurrency Price Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single local file
  python dev.py --data-path coins.json
  
  # Multiple local files
  python dev.py --data-paths snapshot1.json snapshot2.json snapshot3.json
  
  # GCS file
  python dev.py --data-path gs://bucket-name/path/to/file.json
  
  # Get latest 10 files from GCS bucket path (default)
  python dev.py --bucket-path gs://bucket-name/path/to/dir/
  
  # Get latest N files from GCS bucket path
  python dev.py --bucket-path gs://bucket-name/path/to/dir/ --latest-n 5
  
  # GCS file with bucket name
  python dev.py --data-path path/to/file.json --bucket-name my-bucket
  
  # Using environment variables
  export DATA_PATH=coins.json
  python dev.py
        """
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to a single data file (local or GCS). Can be used multiple times.',
        action='append'
    )
    parser.add_argument(
        '--data-paths',
        type=str,
        nargs='+',
        help='Multiple data file paths (local or GCS) separated by spaces'
    )
    parser.add_argument(
        '--bucket-name',
        type=str,
        help='GCS bucket name (if not specified in gs:// path)',
        default=None
    )
    parser.add_argument(
        '--bucket-path',
        type=str,
        help='GCS bucket path to directory. Will automatically get the latest N files from this path.',
        default=None
    )
    parser.add_argument(
        '--latest-n',
        type=int,
        help='Number of latest files to retrieve from bucket-path (default: 10)',
        default=10
    )
    parser.add_argument(
        '--load-model',
        type=str,
        help='Path to a saved model file to load (skips training)',
        default=None
    )
    parser.add_argument(
        '--save-model',
        type=str,
        help='Path where to save the trained model (e.g., model.pkl or models/btc_model.joblib)',
        default=None
    )
    parser.add_argument(
        '--save-model-to-bucket',
        action='store_true',
        help='Save the trained model to GCS bucket. Uses MODEL_BUCKET_PATH env var or default path.',
        default=False
    )
    
    args = parser.parse_args()
    
    # Start overall timing
    overall_start = time.time()
    all_timing_info = {}
    
    print("=" * 60)
    print("Cryptocurrency Price Prediction Model")
    print("=" * 60)
    
    # Determine data paths from arguments, environment variables, or defaults
    data_paths = None
    
    # Check if bucket-path is specified (highest priority)
    if args.bucket_path:
        print(f"\nGetting latest {args.latest_n} files from bucket path: {args.bucket_path}")
        bucket_list_start = time.time()
        try:
            data_paths = get_latest_files_from_bucket(args.bucket_path, args.latest_n, args.bucket_name)
            bucket_list_duration = time.time() - bucket_list_start
            all_timing_info["bucket_file_listing"] = {
                "duration_seconds": bucket_list_duration,
                "files_found": len(data_paths)
            }
            print(f"\nRetrieved {len(data_paths)} file path(s) from bucket")
        except Exception as e:
            print(f"\nError getting latest files from bucket: {str(e)}")
            raise
    # Check command-line arguments
    elif args.data_paths:
        data_paths = args.data_paths
        print(f"\nUsing data paths from command-line arguments: {data_paths}")
    elif args.data_path:
        data_paths = args.data_path
        print(f"\nUsing data path(s) from command-line arguments: {data_paths}")
    else:
        # Check environment variables
        env_data_paths = os.getenv('DATA_PATHS')
        if env_data_paths:
            data_paths = [p.strip() for p in env_data_paths.split(',')]
            print(f"\nUsing data paths from DATA_PATHS environment variable: {data_paths}")
        else:
            env_data_path = os.getenv('DATA_PATH')
            if env_data_path:
                data_paths = [env_data_path]
                print(f"\nUsing data path from DATA_PATH environment variable: {data_paths}")
    
    # Load or create model instance
    model_loaded = False
    if args.load_model:
        print(f"\n=== Loading Model from {args.load_model} ===")
        load_start = time.time()
        try:
            model = BTCPricePredictor.load_model(args.load_model)
            load_duration = time.time() - load_start
            all_timing_info["model_loading"] = {
                "duration_seconds": load_duration,
                "filepath": args.load_model
            }
            print(f"  ⏱️  Model loaded in {load_duration:.3f}s")
            model_loaded = True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to creating a new model...")
            model = BTCPricePredictor(use_regularization=True)
    else:
        # No model specified - try to load latest model from default GCS path
        default_model_path = os.getenv('DEFAULT_MODEL_BUCKET_PATH', 'lake/models/btc_price_model/pkl/prod')
        bucket_name = os.getenv('BUCKET_NAME', 'thomasanalytics-data1')
        
        print(f"\n=== No model specified, attempting to load latest from default path ===")
        print(f"  Default path: gs://{bucket_name}/{default_model_path}")
        
        load_start = time.time()
        try:
            latest_model_path = get_latest_model_from_bucket(default_model_path, bucket_name)
            print(f"\n=== Loading Latest Model from {latest_model_path} ===")
            model = BTCPricePredictor.load_model(latest_model_path)
            load_duration = time.time() - load_start
            all_timing_info["model_loading"] = {
                "duration_seconds": load_duration,
                "filepath": latest_model_path,
                "source": "default_bucket_path"
            }
            print(f"  ⏱️  Model loaded in {load_duration:.3f}s")
            model_loaded = True
        except Exception as e:
            print(f"  ⚠️  Could not load model from default path: {str(e)}")
            print(f"  Creating a new model instead...")
            # Create model instance
            # To use a different model, change this line:
            # from sklearn.ensemble import RandomForestRegressor
            # model = BTCPricePredictor(model=RandomForestRegressor())
            # 
            # By default, uses Ridge regression (L2 regularization) to prevent overfitting
            # Set use_regularization=False to use plain LinearRegression
            model = BTCPricePredictor(use_regularization=True)
    
    # Load data from files/GCS if paths provided
    if data_paths:
        try:
            snapshots, load_timing = load_multiple_snapshots(data_paths, args.bucket_name)
            all_timing_info.update(load_timing)
            
            if model_loaded:
                # Model already loaded, just make predictions
                print(f"\nUsing loaded model for prediction on {len(snapshots)} snapshot(s)...")
                if len(snapshots) > 0:
                    # Use the last snapshot for prediction
                    prediction_snapshot = snapshots[-1]
                    result = predict_with_loaded_model(model, prediction_snapshot)
                    # Merge timing info
                    if "timing_info" in result:
                        all_timing_info.update(result["timing_info"])
                else:
                    raise ValueError("No snapshots provided for prediction")
            elif len(snapshots) > 1:
                print(f"\nUsing {len(snapshots)} loaded snapshots for training...")
                # Use all but the last snapshot for training, last one for prediction
                training_snapshots = snapshots[:-1]
                prediction_snapshot = snapshots[-1]
                result = train_and_predict_with_model(model, training_snapshots, prediction_snapshot)
                # Merge timing info
                if "timing_info" in result:
                    all_timing_info.update(result["timing_info"])
            elif len(snapshots) == 1:
                print("\nOnly one snapshot loaded. Using single-snapshot prediction mode...")
                print("To enable proper training, provide multiple data files.")
                result = predict_single_snapshot(model, snapshots[0])
                # Merge timing info
                if "timing_info" in result:
                    all_timing_info.update(result["timing_info"])
            else:
                raise ValueError("No valid snapshots were loaded")
                
        except Exception as e:
            print(f"\nError loading data from provided paths: {str(e)}")
            print("Falling back to default hardcoded data...")
            # Fall through to default behavior
            data_paths = None
    
    # Use default hardcoded data if no paths provided or loading failed
    if data_paths is None:
        print("\nUsing default hardcoded data snapshots...")
        if model_loaded:
            # Model already loaded, just make predictions
            if len(HISTORICAL_SNAPSHOTS) > 0:
                prediction_snapshot = HISTORICAL_SNAPSHOTS[-1]
                result = predict_with_loaded_model(model, prediction_snapshot)
                if "timing_info" in result:
                    all_timing_info.update(result["timing_info"])
            else:
                result = predict_with_loaded_model(model, FULL_DATA)
                if "timing_info" in result:
                    all_timing_info.update(result["timing_info"])
        # Check if we have multiple snapshots for training
        elif len(HISTORICAL_SNAPSHOTS) > 1:
            print(f"\nUsing {len(HISTORICAL_SNAPSHOTS)} historical snapshots for training...")
            # Use all but the last snapshot for training, last one for prediction
            training_snapshots = HISTORICAL_SNAPSHOTS[:-1]
            prediction_snapshot = HISTORICAL_SNAPSHOTS[-1]
            result = train_and_predict_with_model(model, training_snapshots, prediction_snapshot)
            # Merge timing info
            if "timing_info" in result:
                all_timing_info.update(result["timing_info"])
        elif len(HISTORICAL_SNAPSHOTS) == 1:
            print("\nOnly one snapshot available. Using single-snapshot prediction mode...")
            print("To enable proper training, add more snapshots to HISTORICAL_SNAPSHOTS list.")
            result = predict_single_snapshot(model, HISTORICAL_SNAPSHOTS[0])
            # Merge timing info
            if "timing_info" in result:
                all_timing_info.update(result["timing_info"])
        else:
            print("\nNo snapshots available. Using default data...")
            result = predict_single_snapshot(model, FULL_DATA)
            # Merge timing info
            if "timing_info" in result:
                all_timing_info.update(result["timing_info"])
    
    # Save model if requested
    if args.save_model and not model_loaded:
        print(f"\n=== Saving Model to {args.save_model} ===")
        save_start = time.time()
        try:
            model.save_model(args.save_model)
            save_duration = time.time() - save_start
            all_timing_info["model_saving"] = {
                "duration_seconds": save_duration,
                "filepath": args.save_model
            }
            print(f"  ⏱️  Model saved in {save_duration:.3f}s")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    elif args.save_model and model_loaded:
        print(f"\nNote: Model was loaded from file. Use --save-model without --load-model to save a newly trained model.")
    
    # Save model to GCS bucket if requested
    if args.save_model_to_bucket and not model_loaded:
        print(f"\n=== Saving Model to GCS Bucket ===")
        save_start = time.time()
        try:
            # Get bucket path from environment variable or use default
            model_bucket_path = os.getenv('MODEL_BUCKET_PATH', 'lake/models/btc_price_model/pkl')
            bucket_name = os.getenv('BUCKET_NAME')
            
            if bucket_name is None:
                # Try to extract from model_bucket_path if it's a gs:// path
                if model_bucket_path.startswith("gs://"):
                    parts = model_bucket_path.replace("gs://", "").split("/", 1)
                    bucket_name = parts[0]
                    model_bucket_path = parts[1] if len(parts) > 1 else ""
                else:
                    raise ValueError("BUCKET_NAME environment variable must be set or provide gs:// path in MODEL_BUCKET_PATH")
            
            gcs_path = model.save_model_to_gcs(model_bucket_path, bucket_name, filename_prefix="model")
            save_duration = time.time() - save_start
            all_timing_info["model_saving_to_gcs"] = {
                "duration_seconds": save_duration,
                "gcs_path": gcs_path,
                "bucket_path": model_bucket_path
            }
            print(f"  ⏱️  Model saved to GCS in {save_duration:.3f}s")
            print(f"  📦 GCS Path: {gcs_path}")
        except Exception as e:
            print(f"Error saving model to GCS bucket: {str(e)}")
    elif args.save_model_to_bucket and model_loaded:
        print(f"\nNote: Model was loaded from file. Use --save-model-to-bucket without --load-model to save a newly trained model.")
    
    # Calculate total duration
    overall_duration = time.time() - overall_start
    all_timing_info["total_execution"] = {
        "duration_seconds": overall_duration
    }
    
    # Add timing info to result
    result["timing_info"] = all_timing_info
    
    # Print timing and data summary
    print("\n" + "=" * 60)
    print("Execution Summary")
    print("=" * 60)
    
    # Data processing summary
    if "data_loading" in all_timing_info:
        dl = all_timing_info["data_loading"]
        print(f"\n📊 Data Processing:")
        print(f"   Snapshots loaded: {dl.get('snapshots_loaded', 0)}")
        print(f"   Total records: {dl.get('total_records', 0):,}")
        print(f"   Duration: {dl.get('total_duration_seconds', 0):.3f}s")
        print(f"   Avg per snapshot: {dl.get('avg_load_time_per_snapshot', 0):.3f}s")
    
    # Model training summary
    if "dataset_building" in all_timing_info:
        db = all_timing_info["dataset_building"]
        print(f"\n🔧 Dataset Building:")
        print(f"   Samples: {db.get('n_samples', 0)}")
        print(f"   Features: {db.get('n_features', 0)}")
        print(f"   Duration: {db.get('duration_seconds', 0):.3f}s")
    
    if "model_training" in all_timing_info:
        mt = all_timing_info["model_training"]
        print(f"\n🎯 Model Training:")
        print(f"   Samples: {mt.get('n_samples', 0)}")
        print(f"   Features: {mt.get('n_features', 0)}")
        print(f"   Duration: {mt.get('duration_seconds', 0):.3f}s")
    
    # Prediction summary
    if "prediction" in all_timing_info:
        pred = all_timing_info["prediction"]
        print(f"\n🔮 Prediction:")
        print(f"   Duration: {pred.get('duration_seconds', 0):.3f}s")
    
    # Overall summary
    print(f"\n⏱️  Total Execution Time: {overall_duration:.3f}s ({overall_duration/60:.2f} minutes)")
    
    # Calculate breakdown percentages
    if overall_duration > 0:
        print(f"\n📈 Time Breakdown:")
        for step_name, step_info in all_timing_info.items():
            if step_name != "total_execution" and isinstance(step_info, dict):
                step_duration = step_info.get("duration_seconds", 0)
                if step_duration > 0:
                    percentage = (step_duration / overall_duration) * 100
                    print(f"   {step_name.replace('_', ' ').title()}: {step_duration:.3f}s ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    # Convert result to JSON-serializable format
    def make_serializable(obj):
        """Convert numpy types and other non-serializable types to native Python types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    serializable_result = make_serializable(result)
    print(json.dumps(serializable_result, indent=2))
    
    return result


if __name__ == "__main__":
    main()
