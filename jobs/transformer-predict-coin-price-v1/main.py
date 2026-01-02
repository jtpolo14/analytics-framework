"""
BTC Price Prediction Job - Predicts future 5-minute interval BTC price
"""
from types import SimpleNamespace
from google.cloud import storage
from datetime import datetime
import os
import json
import pymysql
import re
import anthropic
import time
from runner import predict_btc_price

# Retrieve Job-defined env vars
JOB_EXECUTION_ID = os.getenv('CLOUD_RUN_EXECUTION')
JOB_TAG = 'job-transformer-predict-coin-price-v1'
TASK_INDEX = os.getenv("CLOUD_RUN_TASK_INDEX", 0)
TASK_ATTEMPT = os.getenv("CLOUD_RUN_TASK_ATTEMPT", 0)
BUCKET_NAME = os.getenv('BUCKET_NAME')
PATH_PREFIX = os.getenv('PATH_PREFIX')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Group database configuration
db_config = SimpleNamespace(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    name=os.getenv("DB_NAME"),
)


# JSON Schema for output validation
NHC_ECONOMIC_SIGNAL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "nhc_high_seas_economic_signal",
    "type": "object",
    "required": [
        "metadata", "physical_conditions", "persistence",
        "economic_impacts", "route_exposure", "delta_vs_prior"
    ],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["source", "region", "issue_time_utc", "forecast_horizons_hr", "has_warnings"],
            "properties": {
                "source": {"type": "string"},
                "region": {"type": "string"},
                "issue_time_utc": {"type": "string", "format": "date-time"},
                "forecast_horizons_hr": {"type": "array", "items": {"type": "integer"}},
                "has_warnings": {"type": "boolean"}
            }
        },
        "physical_conditions": {
            "type": "object",
            "required": ["max_wind_kt", "max_seas_m", "winds_above_25kt", "seas_above_4m", "geographic_coverage"],
            "properties": {
                "max_wind_kt": {"type": "number"},
                "max_seas_m": {"type": "number"},
                "winds_above_25kt": {"type": "boolean"},
                "seas_above_4m": {"type": "boolean"},
                "geographic_coverage": {
                    "type": "object",
                    "required": ["north_atlantic", "caribbean", "gulf_of_america"],
                    "properties": {
                        "north_atlantic": {"type": "boolean"},
                        "caribbean": {"type": "boolean"},
                        "gulf_of_america": {"type": "boolean"}
                    }
                }
            }
        },
        "persistence": {
            "type": "object",
            "required": ["hours_above_25kt", "hours_above_4m", "trend"],
            "properties": {
                "hours_above_25kt": {"type": ["integer", "null"]},
                "hours_above_4m": {"type": ["integer", "null"]},
                "trend": {"type": "string", "enum": ["improving", "stable", "deteriorating"]}
            }
        },
        "economic_impacts": {
            "type": "object",
            "required": [
                "shipping_disruption", "freight_cost_pressure",
                "energy_logistics_risk", "insurance_premium_pressure",
                "supply_chain_reliability"
            ],
            "properties": {
                "shipping_disruption": {"type": "string", "enum": ["none", "low", "moderate", "high"]},
                "freight_cost_pressure": {"type": "string", "enum": ["none", "low", "moderate", "high"]},
                "energy_logistics_risk": {"type": "string", "enum": ["none", "low", "moderate", "high"]},
                "insurance_premium_pressure": {"type": "string", "enum": ["none", "low", "moderate", "high"]},
                "supply_chain_reliability": {
                    "type": "string",
                    "enum": ["normal", "slightly_degraded", "degraded", "severely_degraded"]
                }
            }
        },
        "route_exposure": {
            "type": "object",
            "required": ["affected_routes", "affected_sectors"],
            "properties": {
                "affected_routes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "transatlantic_eastbound", "transatlantic_westbound",
                            "us_east_coast_imports", "caribbean_energy_exports",
                            "gulf_energy_exports", "south_america_exports"
                        ]
                    }
                },
                "affected_sectors": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "container_shipping", "bulk_commodities",
                            "energy_transport", "agricultural_exports", "insurance"
                        ]
                    }
                }
            }
        },
        "delta_vs_prior": {
            "type": "object",
            "required": ["max_wind_kt", "max_seas_m", "coverage_change", "economic_risk_change"],
            "properties": {
                "max_wind_kt": {"type": ["number", "null"]},
                "max_seas_m": {"type": ["number", "null"]},
                "coverage_change": {"type": "string", "enum": ["expanded", "unchanged", "contracted"]},
                "economic_risk_change": {"type": "string", "enum": ["increased", "unchanged", "decreased"]}
            }
        }
    }
}

SYSTEM_PROMPT = """You are an economic risk signal extractor for maritime and logistics intelligence.
Your job is to convert National Hurricane Center High Seas Forecasts into structured, 
machine-readable economic risk signals.

Rules:
- Extract only measurable signals and inferred economic impacts
- Use enums and numeric values where specified in the schema
- If information is missing, set the field to null
- Be conservative in economic impact classification
- Return ONLY valid JSON matching the schema provided
- Infer short-term (0-72h) economic impacts on shipping, energy logistics, insurance, and supply chain
- Compare conditions to normal seasonal conditions when classifying severity
"""


def extract_nhc_economic_signal(forecast_xml, api_key=None, model="claude-haiku-4-5-20251001"):
    """
    Extract economic risk signals from NHC High Seas Forecast
    
    Args:
        forecast_xml: Raw XML/text from NHC forecast
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        model: Claude model to use (haiku-4-5 recommended for speed/cost)
        
    Returns:
        tuple: (response_text, call_stats) where call_stats is a dict containing:
            - input_tokens: Number of input tokens used
            - output_tokens: Number of output tokens used
            - response_time_seconds: Time taken for the API call
            - model: Model used for the call
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""Given the following NHC High Seas Forecast, extract and return a JSON object
using this schema:

{json.dumps(NHC_ECONOMIC_SIGNAL_SCHEMA, indent=2)}

Forecast text:

{forecast_xml}
"""
    print(f"Executing call. Prompt size: {len(prompt)} characters")
    
    # Record start time
    start_time = time.time()
    
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Record end time and calculate response time
    end_time = time.time()
    response_time = end_time - start_time
    
    response_text = message.content[0].text
    
    # Extract usage statistics
    usage = message.usage
    call_stats = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "response_time_seconds": round(response_time, 3),
        "model": model
    }
    
    print(f"Call completed. Response size: {len(response_text)} characters. "
          f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out. "
          f"Time: {response_time:.3f}s")
    
    return response_text, call_stats


def read_gcs(file_name):
    """
    Read data from Google Cloud Storage bucket
    
    Args:
        file_name: Path to the file in the GCS bucket (e.g., "path/to/file.xml")
        
    Returns:
        tuple: (text_content, byte_count) - The text content of the file and its size in bytes
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Get the blob
        blob = bucket.blob(file_name)
        
        # Download the file content as bytes
        file_content = blob.download_as_bytes()
        
        # Get byte count
        byte_count = len(file_content)
        
        # Decode bytes to text
        text_content = file_content.decode('utf-8')
        
        print(f"File read successfully from gs://{BUCKET_NAME}/{file_name} ({byte_count} bytes)")
        return text_content, byte_count
        
    except Exception as e:
        print(f"Error reading from GCS: {e}")
        raise

def get_unprocessed_files(prefix=None):
    """
    Get all files in the bucket that match nhc_gtwo_{timestamp}.xml pattern
    and don't have a corresponding _analytics.json file.
    
    Args:
        prefix: Optional prefix to filter files (e.g., PATH_PREFIX). 
                If None, searches entire bucket.
        
    Returns:
        list: List of file paths (blob names) that need to be processed
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # List all blobs in the bucket (optionally filtered by prefix)
        if prefix:
            blobs = bucket.list_blobs(prefix=prefix)
        else:
            blobs = bucket.list_blobs()
        
        # Pattern to match nhc_gtwo_{timestamp}.xml files
        # Example: nhc_gtwo_20251227_061322.xml
        pattern = re.compile(r'nhc_gtwo_(\d{8}_\d{6})\.xml$')
        
        # Dictionary to track base files and their analytics counterparts
        base_files = {}  # {timestamp: full_path}
        analytics_files = set()  # Set of timestamps that have analytics files
        
        # Process all blobs
        for blob in blobs:
            # Extract just the filename (last part of path)
            file_name = blob.name.split('/')[-1]
            
            # Check if it's a base file (nhc_gtwo_{timestamp}.xml)
            base_match = pattern.match(file_name)
            if base_match:
                timestamp = base_match.group(1)
                base_files[timestamp] = blob.name
                continue
            
            # Check if it's an analytics file (nhc_gtwo_{timestamp}_analytics.json)
            analytics_pattern = re.compile(r'nhc_gtwo_(\d{8}_\d{6})_analytics\.json$')
            analytics_match = analytics_pattern.match(file_name)
            if analytics_match:
                timestamp = analytics_match.group(1)
                analytics_files.add(timestamp)
        
        # Find base files that don't have corresponding analytics files
        unprocessed = []
        for timestamp, file_path in base_files.items():
            if timestamp not in analytics_files:
                unprocessed.append(file_path)
        
        print(f"Found {len(unprocessed)} unprocessed files out of {len(base_files)} total base files")
        return unprocessed
        
    except Exception as e:
        print(f"Error getting unprocessed files from GCS: {e}")
        raise

def upload_to_gcs(bucket_name, data, blob_name=None, path_prefix=None):
    """
    Upload data to Google Cloud Storage bucket
    
    Args:
        bucket_name: Name of the GCS bucket
        data: String data to upload
        blob_name: Name for the file in GCS (optional, defaults to timestamped filename)
        path_prefix: Optional path prefix for generated filenames
        
    Returns:
        str: GCS path of uploaded file (gs://bucket/path)
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Generate filename with timestamp if not provided
        if blob_name is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            blob_name = f"{path_prefix}/nhc_gtwo_{timestamp}.json"
        
        # Create blob and upload
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type='application/json')
        
        print(f"File uploaded successfully to gs://{BUCKET_NAME}/{blob_name}")
        return f"gs://{BUCKET_NAME}/{blob_name}"
        
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise

def main():
    """
    Main function to run BTC price prediction using default GCS prod model
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
                print("Running BTC price prediction with default GCS prod model...")
                
                # Get prediction result using default GCS prod model (model_file=None)
                prediction_result = predict_btc_price(model_file=None)
                
                print(f"Prediction result: {json.dumps(prediction_result, indent=2)}")
                
                # Write result to task_event payload
                event_tag = "metric_btc_price_prediction_v1_1"
                task_event_metadata = {
                    "job_status": "ok",
                    "prediction_result": prediction_result
                }
                payload = json.dumps(task_event_metadata)
                cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, event_tag, payload) VALUES (%s, 'log', %s, %s, %s)", (new_id, JOB_TAG, event_tag, payload))
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