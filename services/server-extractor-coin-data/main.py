"""
Fetch top cryptocurrency data and upload to GCP bucket
"""
import requests
from types import SimpleNamespace
from google.cloud import storage
from datetime import datetime
import os
import json
import pymysql
import signal
import atexit

from flask import Flask, jsonify, request

app = Flask(__name__)

# Track endpoint hits
endpoint_hits = {}

# Rate limiting for request_market_cap_desc endpoint
last_market_cap_request_time = None
RATE_LIMIT_SECONDS = 10

# Retrieve Job-defined env vars
TASK_TAG = 'service-extractor-coin-data'
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

# Validate database configuration at startup - fail immediately if missing
required_db_vars = ["user", "password", "host", "name"]
missing_db_vars = [var for var in required_db_vars if not getattr(db_config, var)]
if missing_db_vars:
    error_msg = f"CRITICAL: Missing required database configuration: {', '.join(missing_db_vars)}"
    print(error_msg)
    raise ValueError(error_msg)

# Global database connection (managed outside main function)
db_connection = None

# Global task ID (set once on initialization and reused)
new_id = None

def get_db_connection():
    """
    Get or create database connection. Reconnects if connection is lost.
    
    Returns:
        pymysql.Connection: Database connection object
    """
    global db_connection
    
    # Check if we have a valid connection
    if db_connection is not None:
        try:
            # Test if connection is still alive
            db_connection.ping(reconnect=False)
            return db_connection
        except (pymysql.Error, AttributeError):
            # Connection is dead, reset it
            db_connection = None
    
    # Create new connection if we don't have one or it's dead
    # Config is already validated at startup, so we can proceed directly
    try:
        db_connection = pymysql.connect(
            unix_socket=db_config.host,
            user=db_config.user,
            password=db_config.password,
            database=db_config.name,
            autocommit=False,
        )
        print("Database connection established.")
        return db_connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def mark_task_completed():
    """Mark the task as completed in the database (only called once on shutdown)."""
    global new_id
    if new_id is None:
        return
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            task_metadata = {"job_results": "success"}
            # Add endpoint_hits stats to the task_metadata before serialization
            endpoint_hits = globals().get("endpoint_hits", None)
            if endpoint_hits is not None:
                task_metadata["endpoint_hits"] = endpoint_hits
            task_metadata_json = json.dumps(task_metadata)
            update_query = "UPDATE tasks SET task_status = 'completed', task_completed_at = CURRENT_TIMESTAMP, task_meta_data = %s WHERE task_id = %s;"
            cursor.execute(update_query, (task_metadata_json, new_id))
            conn.commit()
            print(f"Marked task {new_id} as 'completed' in the database.")
    except Exception as e:
        print(f"Error marking task as completed: {e}")

def close_db_connection():
    """Close the global database connection and mark task as completed."""
    # Mark task as completed before closing
    mark_task_completed()
    
    global db_connection
    if db_connection is not None:
        try:
            db_connection.close()
            print("Database connection closed.")
        except Exception as e:
            print(f"Error closing database connection: {e}")
        finally:
            db_connection = None

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

def init_task():
    """
    Initialize task record in database (runs once at startup).
    """
    global new_id
    
    if new_id is not None:
        print(f"Task already initialized with ID: {new_id}")
        return
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Insert a record for the started task
            cursor.execute("INSERT INTO tasks (task_tag) VALUES (%s)", (TASK_TAG))
            new_id = cursor.lastrowid
            conn.commit()
            print(f"Created task record with ID: {new_id}")

            # Log the execution ID
            event_notes = f"{TASK_TAG} up and running...."
            cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, TASK_TAG, event_notes))
            conn.commit()
    except Exception as e:
        print(f"Error initializing task: {e}")
        raise

def main():
    """
    Main function to fetch coin data and upload to GCS
    """
    global new_id
    
    print(f"Starting Task #{TASK_INDEX}, Attempt #{TASK_ATTEMPT}...")
    
    try:
        # Get database connection (reuses existing or creates new)
        conn = get_db_connection()
        
        with conn.cursor() as cursor:
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
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, TASK_TAG, event_notes))
                conn.commit()
                
                print(f"Uploading to GCS bucket: {BUCKET_NAME}...")
                gcs_path = upload_to_gcs(BUCKET_NAME, coin_data_json, path_prefix=PATH_PREFIX)
                
                print(f"Complete! Data available at: {gcs_path}")
                # Log the upload to GCS
                event_notes = f"Upload to GCS complete! Data available at: {gcs_path}"
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, TASK_TAG, event_notes))
                conn.commit()
 
        print(f"Successfully processed request for market cap descending route")

    except Exception as e:
        print(f"An error occurred during task execution: {e}")
        # --- Failure ---
        try:
            conn = get_db_connection()
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
        except Exception as db_err:
            print(f"CRITICAL: Could not connect to database to update task status. Error: {db_err}")
        # Don't re-raise for server context - just log the error
        raise

@app.before_request
def track_endpoint():
    """Track endpoint hits before each request."""
    endpoint = request.endpoint or request.path
    if endpoint:
        endpoint_hits[endpoint] = endpoint_hits.get(endpoint, 0) + 1

@app.route("/")
def index():
    """Stats route - returns endpoint hit counts."""
    return jsonify(endpoint_hits)

@app.route("/request/marketcap/desc")
def request_market_cap_desc():
    """Data request for market cap descending route."""
    global last_market_cap_request_time
    
    # Rate limiting: check if 10 seconds have passed since last request
    current_time = datetime.utcnow()
    if last_market_cap_request_time is not None:
        time_since_last_request = (current_time - last_market_cap_request_time).total_seconds()
        if time_since_last_request < RATE_LIMIT_SECONDS:
            remaining_time = RATE_LIMIT_SECONDS - time_since_last_request
            return jsonify({
                "error": "Rate limit exceeded",
                "message": f"Please wait {remaining_time:.1f} seconds before making another request",
                "retry_after_seconds": round(remaining_time, 1)
            }), 429  # HTTP 429 Too Many Requests
    
    # Update last request time
    last_market_cap_request_time = current_time
    
    main()
    return jsonify({"message": "Data request for market cap descending route completed."})

@app.teardown_appcontext
def close_db(error):
    """Close database connection when app context is torn down."""
    # Don't close the connection here - we want to keep it open for reuse
    # Only close on actual app shutdown
    pass

# Register shutdown handlers to mark task as completed on exit
def shutdown_handler(signum=None, frame=None):
    """Handle shutdown signals."""
    print("Shutdown signal received, marking task as completed...")
    close_db_connection()

# Register signal handlers
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)
# Register atexit handler as backup
atexit.register(shutdown_handler)

# Initialize task on startup
init_task()

if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    finally:
        close_db_connection()
    