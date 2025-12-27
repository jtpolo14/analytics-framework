# PowerShell equivalent of run.sh
param(
    [string]$ACTIONS = ""
)

$ErrorActionPreference = "Stop"

# ---- Config ----
$PROJECT_ID = "dulcet-iterator-344116"
$REGION = "us-central1"
$JOB_NAME = "job-extractor-coin-data"
$IMAGE = "us-central1-docker.pkg.dev/dulcet-iterator-344116/cloud-run-source-deploy/job-extractor-coin-data:latest"
$JOB_YAML = "job.yaml"
$CLOUD_SQL_INSTANCE = "dulcet-iterator-344116:us-central1:de-course-dev"

if ([string]::IsNullOrWhiteSpace($ACTIONS)) {
    Write-Host "Usage: .\run.ps1 [i][r][u][e][c]"
    Write-Host "  i = build & push image"
    Write-Host "  c = create job"
    Write-Host "  r = replace job (from YAML)"
    Write-Host "  u = update job (Cloud SQL, secrets, etc.)"
    Write-Host "  e = execute job"
    
    exit 1
}

# ---- Functions ----
function push_image {
    Write-Host "Building & pushing image..."
    gcloud builds submit --tag $IMAGE
}

function create_job {
    Write-Host "Creating Cloud Run job..."
    gcloud run jobs deploy $JOB_NAME --image $IMAGE --region $REGION --project $PROJECT_ID
}

function replace_job {
    Write-Host "Deploying Cloud Run job [part 1 - replace]..."
    gcloud run jobs replace $JOB_YAML --region $REGION --project $PROJECT_ID
}

function update_job {
    Write-Host "Deploying Cloud Run job  [part 2 - update]..."
    gcloud run jobs update $JOB_NAME --region=$REGION --project=$PROJECT_ID --add-cloudsql-instances=$CLOUD_SQL_INSTANCE --max-retries=0 --set-secrets=DB_PASSWORD=db-password:latest
}

function execute_job {
    Write-Host "Executing Cloud Run job..."
    gcloud run jobs execute $JOB_NAME --region $REGION --project $PROJECT_ID
}

# ---- Execution order (fixed, regardless of param order) ----
if ($ACTIONS -match "i") { push_image }
if ($ACTIONS -match "c") { create_job }
if ($ACTIONS -match "r") { replace_job }
if ($ACTIONS -match "u") { update_job }
if ($ACTIONS -match "e") { execute_job }

Write-Host "Done"

