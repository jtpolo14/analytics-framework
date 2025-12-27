"""
Fetch National Hurricane Center XML data and upload to GCP bucket
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
JOB_TAG = 'job-extractor-nhc-gtwo-v1'
TASK_INDEX = os.getenv("CLOUD_RUN_TASK_INDEX", 0)
TASK_ATTEMPT = os.getenv("CLOUD_RUN_TASK_ATTEMPT", 0)
TARGET_URL = os.getenv('URL_NHC_GTWO')
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

def fetch_nhc_data(url: str):
    """
    Fetch XML data from NHC
    
    Args:
        url: URL of the NHC XML feed
        
    Returns:
        XML content as string
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise

def upload_to_gcs(bucket_name, data, blob_name=None, path_prefix=None):
    """
    Upload data to Google Cloud Storage bucket
    
    Args:
        bucket_name: Name of the GCS bucket
        data: String data to upload
        blob_name: Name for the file in GCS (optional, defaults to timestamped filename)
        
    Returns:
        Public URL of uploaded file
    """
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Generate filename with timestamp if not provided
        if blob_name is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            blob_name = f"{path_prefix}/nhc_gtwo_{timestamp}.xml"
        
        # Create blob and upload
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type='application/xml')
        
        print(f"File uploaded successfully to gs://{bucket_name}/{blob_name}")
        return f"gs://{bucket_name}/{blob_name}"
        
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise

def main():
    """
    Main function to fetch NHC data and upload to GCS
    """

    # Optional: specify a custom blob name, or leave as None for timestamped name
    BLOB_NAME = None  # e.g., "nhc/current_outlook.xml"

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
    
                print("Fetching NHC data...")
                xml_data = fetch_nhc_data(TARGET_URL)
                
                task_event_metadata = {"job_status": "ok",
                                       "data_size_bytes": len(xml_data)}
                print(f"Data fetched successfully ({len(xml_data)} bytes)")
                # Log the data pull
                event_notes = f"Data fetched successfully: {task_event_metadata}"
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, JOB_TAG, event_notes))
                conn.commit()
                
                print(f"Uploading to GCS bucket: {BUCKET_NAME}...")
                gcs_path = upload_to_gcs(BUCKET_NAME, xml_data, path_prefix=PATH_PREFIX)
                
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