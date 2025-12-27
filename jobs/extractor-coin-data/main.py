"""
Fetch top 100 cryptocurrency data and upload to GCP bucket
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
JOB_TAG = 'job-extractor-coin-data'
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

def get_coins():
    """Fetch top N cryptocurrencies by market cap"""
    return requests.get("https://api.coingecko.com/api/v3/coins/markets", 
                       params={"vs_currency": "usd", "order": "market_cap_desc", 
                              "per_page": 100, "page": 1, "sparkline": False,
                              "price_change_percentage": "1h,24h,7d,30d"}, 
                       timeout=10).json()

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
            blob_name = f"{path_prefix}/coins_usd_market_cap_desc_{timestamp}.json"
        
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
    Main function to fetch coin data and upload to GCS
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
    
                print("Fetching coin data...")
                coin_data = get_coins()
                
                # Convert to JSON string for storage
                coin_data_json = json.dumps(coin_data)
                
                task_event_metadata = {"job_status": "ok",
                                       "data_size_bytes": len(coin_data_json),
                                       "coins_count": len(coin_data)}
                print(f"Data fetched successfully ({len(coin_data)} coins, {len(coin_data_json)} bytes)")
                # Log the data pull
                event_notes = f"Data fetched successfully: {task_event_metadata}"
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, JOB_TAG, event_notes))
                conn.commit()
                
                print(f"Uploading to GCS bucket: {BUCKET_NAME}...")
                gcs_path = upload_to_gcs(BUCKET_NAME, coin_data_json, path_prefix=PATH_PREFIX)
                
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