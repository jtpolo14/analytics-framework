"""
Fetch option chain data and upload to GCP bucket
"""
import yfinance as yf
from types import SimpleNamespace
from google.cloud import storage
from datetime import datetime
import os
import json
import pymysql
import time

# Retrieve Job-defined env vars
JOB_EXECUTION_ID = os.getenv('CLOUD_RUN_EXECUTION')
JOB_TAG = 'job-extractor-stock-option-data'
TASK_INDEX = os.getenv("CLOUD_RUN_TASK_INDEX", 0)
TASK_ATTEMPT = os.getenv("CLOUD_RUN_TASK_ATTEMPT", 0)
BUCKET_NAME = os.getenv('BUCKET_NAME')
PATH_PREFIX = os.getenv('PATH_PREFIX')

# Group database configuration
db_config = SimpleNamespace(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    name=os.getenv("DB_NAME"),
)

def get_option_chains():
    """
    Fetch option chain data from Yahoo Finance for multiple symbols
    
    Returns:
        dict: Dictionary mapping ticker symbols to their option chain data
    """
    # Curated list of highly liquid stocks with active options markets
    symbols = [
        "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "NFLX", 
        "SPY", "QQQ", "JPM", "V", "BAC", "PYPL", "DIS", "COST", "INTC", 
        "AMD", "UBER", "PLTR"
    ]
    
    # Store all option chain data
    all_option_data = {}
    failed_symbols = []
    timestamp = datetime.utcnow().isoformat()
    
    # Fetch data for each symbol
    for ticker in symbols:
        print(f"Fetching option chain data for {ticker}...")
        
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Get option expiration dates
            expirations = stock.options
            
            if not expirations or len(expirations) == 0:
                print(f"[ERROR] {ticker}: No option expiration dates available")
                failed_symbols.append(ticker)
                continue
            
            # Store option chains for this symbol
            symbol_option_chains = {
                "symbol": ticker,
                "timestamp": timestamp,
                "option_chains": {}
            }
            
            total_calls = 0
            total_puts = 0
            
            # Fetch option chain for each expiration date
            for expiration in expirations:
                try:
                    opt_chain = stock.option_chain(expiration)
                    
                    # Convert DataFrames to dictionaries for JSON serialization
                    # Handle empty DataFrames and None values
                    if opt_chain.calls is not None and not opt_chain.calls.empty:
                        calls_data = opt_chain.calls.to_dict('records')
                    else:
                        calls_data = []
                    
                    if opt_chain.puts is not None and not opt_chain.puts.empty:
                        puts_data = opt_chain.puts.to_dict('records')
                    else:
                        puts_data = []
                    
                    # Convert numpy/pandas types to native Python types for JSON serialization
                    def convert_numpy_types(obj):
                        if isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(item) for item in obj]
                        elif hasattr(obj, 'item'):  # numpy scalar types
                            return obj.item()
                        elif hasattr(obj, 'tolist'):  # numpy arrays
                            return obj.tolist()
                        # Handle pandas NaN and other special types
                        try:
                            import pandas as pd
                            if pd.isna(obj):
                                return None
                        except (ImportError, AttributeError):
                            pass
                        # Handle other non-serializable types
                        try:
                            if str(type(obj)).startswith("<class 'numpy.") or str(type(obj)).startswith("<class 'pandas."):
                                return str(obj)
                        except:
                            pass
                        return obj
                    
                    calls_data = convert_numpy_types(calls_data)
                    puts_data = convert_numpy_types(puts_data)
                    
                    symbol_option_chains["option_chains"][expiration] = {
                        "calls": calls_data,
                        "puts": puts_data
                    }
                    
                    total_calls += len(calls_data)
                    total_puts += len(puts_data)
                    
                    print(f"  [OK] {ticker} {expiration}: {len(calls_data)} calls, {len(puts_data)} puts")
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as exp_error:
                    print(f"  [ERROR] {ticker} {expiration}: Error fetching expiration - {str(exp_error)}")
                    # Continue with other expirations even if one fails
                    continue
            
            if len(symbol_option_chains["option_chains"]) > 0:
                all_option_data[ticker] = symbol_option_chains
                print(f"[OK] {ticker}: {len(symbol_option_chains['option_chains'])} expirations, {total_calls} calls, {total_puts} puts")
            else:
                print(f"[ERROR] {ticker}: No option chain data retrieved")
                failed_symbols.append(ticker)
        
        except Exception as e:
            print(f"[ERROR] {ticker}: Error - {str(e)}")
            failed_symbols.append(ticker)
            # Add delay before next symbol to avoid rate limiting
            time.sleep(1)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Successfully fetched option chains for {len(all_option_data)} out of {len(symbols)} symbols")
    if failed_symbols:
        print(f"Failed symbols: {', '.join(failed_symbols)}")
    print(f"{'='*50}")
    
    if len(all_option_data) == 0:
        raise ValueError("No option chain data was successfully fetched from the API")
    
    return all_option_data

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
            blob_name = f"{path_prefix}/option_chains_{timestamp}.json"
        
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
    Main function to fetch option chain data and upload to GCS
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
    
                print("Fetching option chain data...")
                option_chain_data = get_option_chains()
                
                # Calculate metadata
                total_expirations = sum(len(symbol_data.get("option_chains", {})) for symbol_data in option_chain_data.values())
                total_contracts = 0
                for symbol_data in option_chain_data.values():
                    for exp_data in symbol_data.get("option_chains", {}).values():
                        total_contracts += len(exp_data.get("calls", [])) + len(exp_data.get("puts", []))
                
                # Convert to JSON string for storage
                option_chain_data_json = json.dumps(option_chain_data)
                
                task_event_metadata = {
                    "job_status": "ok",
                    "data_size_bytes": len(option_chain_data_json),
                    "symbols_count": len(option_chain_data),
                    "total_expirations": total_expirations,
                    "total_contracts": total_contracts
                }
                print(f"Data fetched successfully ({task_event_metadata['symbols_count']} symbols, {total_expirations} expirations, {total_contracts} contracts, {len(option_chain_data_json)} bytes)")
                # Log the data pull
                event_notes = f"Data fetched successfully: {task_event_metadata}"
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, JOB_TAG, event_notes))
                conn.commit()
                
                print(f"Uploading to GCS bucket: {BUCKET_NAME}...")
                gcs_path = upload_to_gcs(BUCKET_NAME, option_chain_data_json, path_prefix=PATH_PREFIX)
                
                print(f"[OK] Complete! Data available at: {gcs_path}")
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