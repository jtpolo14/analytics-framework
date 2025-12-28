"""
Fetch oldest task and analyze for next step
"""
import requests
from types import SimpleNamespace
from datetime import datetime, timedelta
import os
import json
import pymysql
import signal
import atexit

import anthropic

# Retrieve Job-defined env vars
JOB_EXECUTION_ID = os.getenv('CLOUD_RUN_EXECUTION')
JOB_TAG = 'transformer-task-cleaner'
TASK_INDEX = os.getenv("CLOUD_RUN_TASK_INDEX", 0)
TASK_ATTEMPT = os.getenv("CLOUD_RUN_TASK_ATTEMPT", 0)
BUCKET_NAME = os.getenv('BUCKET_NAME')
PATH_PREFIX = os.getenv('PATH_PREFIX')

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
STALE_TASK_THRESHOLD_HOURS = float(os.getenv('STALE_TASK_THRESHOLD_HOURS', 1.0))

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

# Global task execution status (used by close_task to determine final status)
task_execution_status = None  # Can be 'success', 'failed', or None

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

def close_task():
    """Mark the task as completed or failed in the database (only called once on shutdown)."""
    global new_id, task_execution_status
    if new_id is None:
        return
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Determine status based on global task_execution_status
            if task_execution_status == 'failed':
                task_status = 'failed'
                task_metadata = {"job_results": "failed"}
                status_message = "failed"
            else:
                # Default to 'completed' if status is 'success' or None
                task_status = 'completed'
                task_metadata = {"job_results": "success"}
                status_message = "completed"
            
            task_metadata_json = json.dumps(task_metadata)
            update_query = "UPDATE tasks SET task_status = %s, task_completed_at = CURRENT_TIMESTAMP, task_meta_data = %s WHERE task_id = %s;"
            cursor.execute(update_query, (task_status, task_metadata_json, new_id))
            conn.commit()
            print(f"Marked task {new_id} as '{status_message}' in the database.")
    except Exception as e:
        print(f"Error marking task status: {e}")

def close_db_connection():
    """Close the global database connection and mark task as completed."""
    # Mark task as completed before closing
    close_task()
    
    global db_connection
    if db_connection is not None:
        try:
            db_connection.close()
            print("Database connection closed.")
        except Exception as e:
            print(f"Error closing database connection: {e}")
        finally:
            db_connection = None

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
            cursor.execute("INSERT INTO tasks (task_tag) VALUES (%s)", (JOB_TAG))
            new_id = cursor.lastrowid
            conn.commit()
            print(f"Created task record with ID: {new_id}")

            # Log the execution ID
            event_notes = f"Cloud Run Execution ID: {JOB_EXECUTION_ID}"
            cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, JOB_TAG, event_notes))
            conn.commit()
    except Exception as e:
        print(f"Error initializing task: {e}")
        raise

def get_stale_tasks():
    """Fetch oldest stale task that is not claimed by any worker"""

    query = "CALL GetStaleTasks(1);"
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchone()


def get_job_server_logs():
    """Fetch task initialization logs"""
    pass


SYSTEM_PROMPT = """You are a Task Recovery Agent responsible for identifying stale or abandoned tasks, determining their most likely state based on logs, and selecting the safest corrective action.

You do not execute destructive actions directly unless explicitly authorized.
You recommend or trigger the next step according to the rules below.

Rules:
- Explain all your reasoning in the context field.
- The confidence field must be a number between 0 and 1.
- Your output must be valid JSON that matches the output schema provided, no extra text or markdown.

"""

output_schema = {
    "action": "string",
    "confidence": "number",
    "context": "string",
    "review_count": "number"
}

def suggest_next_steps(task_data, api_key=None, model="claude-haiku-4-5-20251001"):
    """
    Make a recommendation for the next step based on the task data
    
    Args:
        task_data: task data related to the stale task
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        model: Claude model to use (haiku-4-5 recommended for speed/cost)
        
    Returns:
        dict: json data matching the output schema

    Example:
    {
    "action": "MARK_ABANDONED",
    "confidence": 0.92,
    "context": "No progress events after TASK_STARTED; claim expired 3 hours ago",
    "review_count": 3
    }
    
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    current_time = datetime.now().isoformat()

    def get_next_review_count(task_events_data):
        """Extract highest review_count and increment"""
        def extract_review_count_from_item(item):
            """Extract review_count from an item, handling payload JSON strings"""
            if isinstance(item, dict):
                # Check if item has review_count directly
                if "review_count" in item:
                    return item.get("review_count", 0)
                
                # Check if item has a payload field (which might be a JSON string)
                payload = item.get("payload")
                if payload:
                    # Parse JSON if payload is a string
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except (json.JSONDecodeError, TypeError):
                            return 0
                    
                    # Extract review_count from parsed payload
                    if isinstance(payload, dict):
                        return payload.get("review_count", 0)
            
            return 0
        
        # Parse JSON if it's a string
        if isinstance(task_events_data, str):
            try:
                task_events = json.loads(task_events_data)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, return 1 (first review)
                return 1
        else:
            task_events = task_events_data
        
        # Handle different data structures
        if isinstance(task_events, list):
            # If it's a list, iterate through items and extract review_count
            max_review = max(
                (extract_review_count_from_item(item) for item in task_events),
                default=0
            )
        elif isinstance(task_events, dict):
            # If it's a single dict, extract review_count
            max_review = extract_review_count_from_item(task_events)
        else:
            # Unknown format, default to 1
            return 1
        
        return max_review + 1
    
    next_review_count = get_next_review_count(task_data)
    
    prompt = f"""
    Current Time: {current_time}
    
    Given the following task and task_events, analyze the data and return the appropriate action, confidence, and context:

    Action Classification Guidelines:
    - The task event history has not been updated after {STALE_TASK_THRESHOLD_HOURS} hour(s)
    - The action field must be one of the following:
        - MARK_ABANDONED
            - there are no clear signs of human intervention required, classify as MARK_ABANDONED.
        - MARK_HITL
            - there are follow up human intervention required, provide a detailed context for the action and classify as MARK_HITL.

    Confidence Guidelines:
    - The confidence field must be a number between 0 and 1.

    Context Guidelines:
    - The context field must be a string that explains the reasoning for the action and confidence.
    - always include the task id in the context for easy reference.
    - Any event with "changed_by": "transformer-task-cleaner":
        - MUST NOT be used as evidence of task progress, failure, or blockage.
        - 

    Review Count Guidelines:
    - The review count field must be a number that indicates the number of times the task has been reviewed including the current review.
    - To determine the review_count, look at the last object in the Task Event History. If a review_count exists, add 1 to that value. If no review_count exists in the history, set the value to 1.
    - The review count should be incremented by 1 for each review.
    - Example: If the Task Event History contains a review with "review_count": 3, your output must be "review_count": 4.
    - System generated review count (confirm this is the correct review_count based on the Task Event History and include in your output):
        - review_count: {next_review_count}.

    Output Requirements:
    - Your output must be valid JSON, no extra text.
    - Before generating the JSON, identify the highest review_count currently in the Task Event History. Your final output must be max(existing_review_count) + 1
    - Your output must be in the following format:

    {json.dumps(output_schema, indent=2)}

    Task Event History:

    {task_data}
    """
    
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = message.content[0].text
    
    # Extract JSON from response (handle markdown code blocks)
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    return json.loads(response_text)


def main():
    """
    Main function to fetch and process stale tasks
    """
    global new_id, task_execution_status
    
    try:
        # Get database connection (reuses existing or creates new)
        conn = get_db_connection()
 
        with conn.cursor() as cursor:
                # --- Main Task Logic ---
                # Loop until no more stale tasks are found
                while True:
                    print("Fetching stale task data...")
                    stale_task_data = get_stale_tasks()
                    
                    # Break loop if no stale tasks found
                    if not stale_task_data:
                        print("No stale task data found")
                        task_event_metadata = {"job_status": "no_stale_tasks"}
                        event_notes = f"Data fetch result: {json.dumps(task_event_metadata)}"
                        cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes) VALUES (%s, 'log', %s, %s)", (new_id, JOB_TAG, event_notes))
                        conn.commit()
                        break
                    
                    # Process the stale task
                    print(f"Data fetched successfully for Task #{stale_task_data[0]}")
                    task_event_metadata = {"stale_task_data": stale_task_data}
                    
                    # claim the stale task, this will be logged in the og task event history
                    event_tag = "agent_generated_task_claim_init"
                    
                    payload = json.dumps({
                        "task_claimed": stale_task_data[0],
                        "task_claimed_by": new_id,
                        "task_claim_created_at": datetime.now().isoformat(),
                        "task_claim_expires_at": (datetime.now() + timedelta(hours=0.5)).isoformat(),
                        "task_claim_type": "stale_task_cleanup",
                        "task_claim_notes": "stale task claimed by transformer-task-cleaner"
                    })
                    cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, event_tag, payload) VALUES (%s, 'log', %s, %s, %s)", (new_id, JOB_TAG, event_tag, payload))
                    event_tag = "agent_generated_task_claim_exec"
                    cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, event_tag, payload) VALUES (%s, 'log', %s, %s, %s)", (stale_task_data[0], JOB_TAG, event_tag, payload))
                    
                    conn.commit()

                    # Get AI suggestion for next steps
                    task_notes = stale_task_data[2] if len(stale_task_data) > 2 else None
                    if task_notes:

                        #### BEGIN AGENTIC RECOMMENDATION ### 
                        next_steps = suggest_next_steps(task_notes, ANTHROPIC_API_KEY)
                        ####  END AGENTIC RECOMMENDATION ####

                        payload = json.dumps(next_steps)
                        event_tag = "agent_generated_next_steps"
                        event_notes = f"Agent generated next steps for task {stale_task_data[0]}"
                        cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, notes, event_tag, payload) VALUES (%s, 'log', %s, %s, %s, %s)", (stale_task_data[0], JOB_TAG, event_notes, event_tag, payload))
                        conn.commit()
                        

                        # Update the task status as 'abandoned' only if conditions are met
                        review_count = next_steps.get("review_count", 0)
                        action = next_steps.get("action", "")
                        confidence = next_steps.get("confidence", 0.0)
                        task_status_update_note = "Task status update not required"
                        if (review_count >= 3 and 
                            action == "MARK_ABANDONED" and 
                            confidence >= 0.85):
                            task_status_update_note = "Task status updated to 'abandoned'"
                            update_query = "UPDATE tasks SET task_status = 'abandoned' WHERE task_id = %s;"
                            cursor.execute(update_query, (stale_task_data[0],))
                            conn.commit()
                            print(f"Marked task {stale_task_data[0]} as 'abandoned' (review_count={review_count}, action={action}, confidence={confidence})")
                        else:
                            print(f"Skipping task status update. Conditions not met (review_count={review_count}, action={action}, confidence={confidence})")

                        event_tag = "agent_generated_task_claim_done"
                        payload = json.dumps({
                        "task_claimed": stale_task_data[0],
                        "task_claimed_by": new_id,
                        "task_claim_ended_at": datetime.now().isoformat(),
                        "task_claim_type": "stale_task_cleanup",
                        "task_claim_notes": " | ".join(["Task next steps generated ok", task_status_update_note])
                        })
                        cursor.execute("INSERT INTO task_events (task_id, event_type, changed_by, event_tag, payload) VALUES (%s, 'log', %s, %s, %s)", (new_id, JOB_TAG, event_tag, payload))
                        conn.commit()
        
        # Mark execution as successful if we reach here
        task_execution_status = 'success'
 
    except Exception as e:
        print(f"An error occurred during task execution: {e}")
        # Mark execution as failed
        task_execution_status = 'failed'
        # --- Failure ---
        try:
            conn = get_db_connection()
            if conn and new_id:
                try:
                    with conn.cursor() as cursor:
                        error_meta = {"error": str(e)}
                        error_meta_json = json.dumps(error_meta)
                        event_tag = "task_execution_error"
                        update_query =  "INSERT INTO task_events (task_id, event_type, changed_by, event_tag, payload) VALUES (%s, 'error', %s, %s, %s);"
                        cursor.execute(update_query, (new_id, JOB_TAG, event_tag, error_meta_json))
                        conn.commit()
                    print(f"Marked task {new_id} as 'failed' in the database.")
                except pymysql.MySQLError as db_err:
                    print(f"CRITICAL: Could not update task status to 'failed'. DB error: {db_err}")
        except Exception as db_err:
            print(f"CRITICAL: Could not connect to database to update task status. Error: {db_err}")
        # Re-raise the original exception to allow the job to fail and retry
        raise
    finally:
        # Don't close connection here - let shutdown handlers manage it
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
        main()
    finally:
        close_db_connection()