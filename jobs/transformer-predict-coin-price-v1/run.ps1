# PowerShell equivalent of run.sh
# NOTE: For local development, .venv should be used as the dev environment
# Activate with: .\.venv\Scripts\Activate.ps1
# Or ensure .venv is activated before running Python commands

param(
    [string]$ACTIONS = ""
)

$ErrorActionPreference = "Stop"

# Check and activate .venv for local development
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating .venv virtual environment..."
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: .venv not found. For local development, create with: python -m venv .venv"
}

# ---- Config ----
$PROJECT_ID = "dulcet-iterator-344116"
$REGION = "us-central1"
$JOB_NAME = "job-transformer-predict-coin-price-v1"
$IMAGE = "us-central1-docker.pkg.dev/dulcet-iterator-344116/cloud-run-source-deploy/job-transformer-predict-coin-price-v1:latest"
$JOB_YAML = "job.yaml"
$CLOUD_SQL_INSTANCE = "dulcet-iterator-344116:us-central1:de-course-dev"
$SCHEDULER_NAME = "$JOB_NAME-scheduler"
$SCHEDULE = "*/5 * * * *"  # Every 5 minutes
$SERVICE_ACCOUNT = "619238021876-compute@developer.gserviceaccount.com"
$JOB_URI = "https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/${JOB_NAME}:run"

if ([string]::IsNullOrWhiteSpace($ACTIONS)) {
    Write-Host "Usage: .\run.ps1 [i][r][u][e][s]"
    Write-Host "  i = build & push image"
    Write-Host "  r = replace job (from YAML)"
    Write-Host "  u = update job (Cloud SQL, secrets, etc.)"
    Write-Host "  e = execute job"
    Write-Host "  s = create scheduler job (every 5 minutes)"
    exit 1
}

# ---- Functions ----
function push_image {
    Write-Host "Building & pushing image..."
    gcloud builds submit --tag $IMAGE
}

function replace_job {
    Write-Host "Deploying Cloud Run job [part 1 - replace]..."
    gcloud run jobs replace $JOB_YAML --region $REGION --project $PROJECT_ID
}

function update_job {
    Write-Host "Deploying Cloud Run job  [part 2 - update]..."
    gcloud run jobs update $JOB_NAME --region=$REGION --project=$PROJECT_ID --add-cloudsql-instances=$CLOUD_SQL_INSTANCE --max-retries=0 --set-secrets="DB_PASSWORD=db-password:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest"
}

function execute_job {
    Write-Host "Executing Cloud Run job..."
    gcloud run jobs execute $JOB_NAME --region $REGION --project $PROJECT_ID
}

function create_scheduler {
    Write-Host "Creating Cloud Scheduler job..."
    Write-Host "  Scheduler name: $SCHEDULER_NAME"
    Write-Host "  Schedule: $SCHEDULE (Daily at 11PM)"
    Write-Host "  Job URI: $JOB_URI"
    Write-Host "  Service Account: $SERVICE_ACCOUNT"
    
    gcloud scheduler jobs create http $SCHEDULER_NAME `
        --schedule="$SCHEDULE" `
        --uri="$JOB_URI" `
        --oidc-service-account-email=$SERVICE_ACCOUNT `
        --location=$REGION `
        --project=$PROJECT_ID `
        --http-method=POST
}

# ---- Execution order (fixed, regardless of param order) ----
if ($ACTIONS -match "i") { push_image }
if ($ACTIONS -match "r") { replace_job }
if ($ACTIONS -match "u") { update_job }
if ($ACTIONS -match "e") { execute_job }
if ($ACTIONS -match "s") { create_scheduler }

Write-Host "Done"

