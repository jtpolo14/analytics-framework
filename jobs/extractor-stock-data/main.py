"""
Fetch stock data and upload to GCP bucket
"""
import requests
from types import SimpleNamespace
from google.cloud import storage
from datetime import datetime
import os
import json
import pymysql

# Retrieve Job-defined env vars
JOB_EXECUTION_ID = os.getenv('CLOUD_RUN_EXECUTION')
JOB_TAG = 'job-extractor-stock-data'
TASK_INDEX = os.getenv("CLOUD_RUN_TASK_INDEX", 0)
TASK_ATTEMPT = os.getenv("CLOUD_RUN_TASK_ATTEMPT", 0)
BUCKET_NAME = os.getenv('BUCKET_NAME')
PATH_PREFIX = os.getenv('PATH_PREFIX')
API_KEY = os.getenv('FMP_API_KEY')  # Financial Modeling Prep API key

# Group database configuration
db_config = SimpleNamespace(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    name=os.getenv("DB_NAME"),
)

def get_stocks():
    """
    Fetch stock data from Financial Modeling Prep API for multiple symbols
    
    Returns:
        dict: Dictionary mapping ticker symbols to their stock data
    """
    if not API_KEY:
        raise ValueError("FMP_API_KEY environment variable is not set")
    
    API_SUFFIX = "&apikey=" + API_KEY
    STOCK_QUOTE_API_URL = "https://financialmodelingprep.com/stable/quote?symbol="
    
    # List of all stock symbols
    symbols = [
        "AAPL", "TSLA", "AMZN", "MSFT", "NVDA", "GOOGL", "META", "NFLX", "JPM", "V",
        "BAC", "PYPL", "DIS", "T", "PFE", "COST", "INTC", "KO", "TGT", "NKE",
        "SPY", "BA", "BABA", "XOM", "WMT", "GE", "CSCO", "VZ", "JNJ", "CVX",
        "PLTR", "SQ", "SHOP", "SBUX", "SOFI", "HOOD", "RBLX", "SNAP", "AMD", "UBER",
        "FDX", "ABBV", "ETSY", "MRNA", "LMT", "GM", "F", "LCID", "CCL", "DAL",
        "UAL", "AAL", "TSM", "SONY", "ET", "MRO", "COIN", "RIVN", "RIOT", "CPRX",
        "VWO", "SPYG", "NOK", "ROKU", "VIAC", "ATVI", "BIDU", "DOCU", "ZM", "PINS",
        "TLRY", "WBA", "MGM", "NIO", "C", "GS", "WFC", "ADBE", "PEP", "UNH",
        "CARR", "HCA", "TWTR", "BILI", "SIRI", "FUBO", "RKT"
    ]
    
    # Store all stock data
    all_stock_data = {}
    failed_symbols = []
    
    # Fetch data for each symbol
    for ticker in symbols:
        composed_url = STOCK_QUOTE_API_URL + ticker + API_SUFFIX
        print(f"Fetching data for {ticker}...API Key length: {len(API_KEY)}...API Key masked: {API_KEY[:2]}..{API_KEY[-2:]}")
        
        try:
            response = requests.get(composed_url, timeout=10)
            
            # Check for 401 Unauthorized - fail immediately if API key is invalid
            if response.status_code == 401:
                error_msg = f"401 Unauthorized - Invalid API key. Response: {response.text[:200]}"
                print(f"✗ {ticker}: {error_msg}")
                raise ValueError(f"API authentication failed (401). Please check FMP_API_KEY. First error from {ticker}: {error_msg}")
            
            # Check HTTP status code
            if response.status_code == 200:
                data = response.json()
                
                # Check for API error responses
                if isinstance(data, dict) and "Error Message" in data:
                    error_msg = data.get("Error Message", "Unknown error")
                    print(f"✗ {ticker}: API error - {error_msg}")
                    failed_symbols.append(ticker)
                    continue
                
                if data and len(data) > 0:
                    all_stock_data[ticker] = data[0]
                    price = data[0].get('price', 'N/A')
                    print(f"✓ {ticker}: ${price}")
                else:
                    print(f"✗ {ticker}: No data returned")
                    failed_symbols.append(ticker)
            else:
                print(f"✗ {ticker}: API request failed - {response.status_code}")
                failed_symbols.append(ticker)
        
        except Exception as e:
            print(f"✗ {ticker}: Error - {str(e)}")
            failed_symbols.append(ticker)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Successfully fetched {len(all_stock_data)} out of {len(symbols)} symbols")
    if failed_symbols:
        print(f"Failed symbols: {', '.join(failed_symbols)}")
    print(f"{'='*50}")
    
    if len(all_stock_data) == 0:
        raise ValueError("No stock data was successfully fetched from the API")
    
    return all_stock_data

def upload_to_gcs(bucket_name, data, blob_name=None, path_prefix=None):
    """
    Upload data to Google Cloud Storage bucket
    
    Args:
        bucket_name: Name of the GCS bucket
        data: String data to upload (JSON, XML, etc.)
        blob_name: Name for the file in GCS (optional, defaults to timestamped filename)
        path_prefix: Optional path prefix for generated filenames
        
    Returns:
        str: GCS path of uploaded file (gs://bucket/path)
    """
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Generate filename with timestamp if not provided
        if blob_name is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            blob_name = f"{path_prefix}/stocks_quote_{timestamp}.json"
        
        # Create blob and upload
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type='application/json')
        
        print(f"File uploaded successfully to gs://{bucket_name}/{blob_name}")
        return f"gs://{bucket_name}/{blob_name}"
        
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise

def main():
    """
    Main function to fetch stock data and upload to GCS
    """

    print(f"Starting Task #{TASK_INDEX}, Attempt #{TASK_ATTEMPT}...")
    new_id = None
    conn = None
    
    # If database variables are set, write to the database
    if all([db_config.user, db_config.password, db_config.host, db_config.name]):
        try:
            print(f"Connecting to database...")
            conn = pymysql.connect(
                unix_socket=db_config.host,
                user=db_config.user,
                password=db_config.password,
                database=db_config.name,
                autocommit=False,  # Disable autocommit for explicit transaction control
            )
 
            with conn.cursor() as cursor:
                # Insert a record for the started task
                cursor.execute("INSERT INTO tasks (task_tag) VALUES (%s)", (JOB_TAG))
                new_id = cursor.lastrowid
                conn.commit()  # Commit the transaction to make the new task visible to other connections
                print(f"Created task record with ID: {new_id}")

                # Log the execution ID
                event_notes = f"Cloud Run Execution ID: {JOB_EXECUTION_ID}"
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, JOB_TAG, event_notes))
                conn.commit()
 
                # --- Main Task Logic ---
    
                print("Fetching stock data...")
                stock_data = get_stocks()
                
                # Convert to JSON string for storage
                stock_data_json = json.dumps(stock_data)
                
                task_event_metadata = {"job_status": "ok",
                                       "data_size_bytes": len(stock_data_json),
                                       "stocks_count": len(stock_data) if isinstance(stock_data, dict) else 0}
                print(f"Data fetched successfully ({task_event_metadata['stocks_count']} stocks, {len(stock_data_json)} bytes)")
                # Log the data pull
                event_notes = f"Data fetched successfully: {task_event_metadata}"
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, JOB_TAG, event_notes))
                conn.commit()
                
                print(f"Uploading to GCS bucket: {BUCKET_NAME}...")
                gcs_path = upload_to_gcs(BUCKET_NAME, stock_data_json, path_prefix=PATH_PREFIX)
                
                print(f"✓ Complete! Data available at: {gcs_path}")
                # Log the upload to GCS
                event_notes = f"Upload to GCS complete! Data available at: {gcs_path}"
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, JOB_TAG, event_notes))
                conn.commit()
    
                # --- Success ---
                task_metadata = {"job_results": "success"}
                task_metadata_json = json.dumps(task_metadata)

                update_query = "UPDATE tasks SET task_status = 'completed', task_completed_at = CURRENT_TIMESTAMP, task_meta_data = %s WHERE task_id = %s;"
                cursor.execute(update_query, (task_metadata_json, new_id))
                conn.commit()
 
            print(f"Successfully completed Task #{TASK_INDEX}. DB record ID: {new_id}. Results: {task_metadata_json}")
 
        except Exception as e:
            print(f"An error occurred during task execution: {e}")
            # --- Failure ---
            if conn and new_id:
                try:
                    with conn.cursor() as cursor:
                        error_meta = {"error": str(e)}
                        error_meta_json = json.dumps(error_meta)
                        update_query = "UPDATE tasks SET task_status = 'failed', task_completed_at = CURRENT_TIMESTAMP, task_meta_data = %s WHERE task_id = %s;"
                        cursor.execute(update_query, (error_meta_json, new_id))
                        conn.commit()
                    print(f"Marked task {new_id} as 'failed' in the database.")
                except pymysql.MySQLError as db_err:
                    print(f"CRITICAL: Could not update task status to 'failed'. DB error: {db_err}")
            # Re-raise the original exception to allow the job to fail and retry
            raise
        finally:
            if conn:
                conn.close()
                print("Database connection closed.")
 
    else:
        missing = [k for k, v in vars(db_config).items() if not v]
        print("Invalid DB config:", missing)

    print(f"Completed Task #{TASK_INDEX}.")

if __name__ == "__main__":
    main()