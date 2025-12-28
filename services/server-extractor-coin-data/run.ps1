# PowerShell equivalent of run.sh
param(
    [string]$ACTIONS = ""
)

$ErrorActionPreference = "Stop"

# ---- Config ----
$PROJECT_ID = "dulcet-iterator-344116"
$REGION = "us-central1"
$PROCESS_ID = "service-extractor-coin-data"
$IMAGE = "us-central1-docker.pkg.dev/dulcet-iterator-344116/cloud-run-source-deploy/service-extractor-coin-data:latest"
$CONFIG_YAML = "service.yaml"
$CLOUD_SQL_INSTANCE = "dulcet-iterator-344116:us-central1:de-course-dev"

if ([string]::IsNullOrWhiteSpace($ACTIONS)) {
    Write-Host "Usage: .\run.ps1 [i][c][r][u]"
    Write-Host "  i = build & push image"
    Write-Host "  c = create service"
    Write-Host "  r = replace service (from YAML)"
    Write-Host "  u = update service (Cloud SQL, secrets, etc.)"
    
    exit 1
}

# ---- Functions ----
function push_image {
    Write-Host "Building & pushing image..."
    gcloud builds submit --tag $IMAGE
}

function create_service {
    Write-Host "Creating Cloud Run service..."
    gcloud run deploy $PROCESS_ID --image $IMAGE --region $REGION --project $PROJECT_ID
}

function replace_service {
    Write-Host "Replacing Cloud Run service..."
    gcloud run services replace $CONFIG_YAML --region $REGION --project $PROJECT_ID
}

function update_service {
    Write-Host "Updating Cloud Run service..."
    gcloud run services update $PROCESS_ID --region=$REGION --project=$PROJECT_ID --add-cloudsql-instances=$CLOUD_SQL_INSTANCE --set-secrets=DB_PASSWORD=db-password:latest
}

# ---- Execution order (fixed, regardless of param order) ----
if ($ACTIONS -match "i") { push_image }
if ($ACTIONS -match "c") { create_service }
if ($ACTIONS -match "r") { replace_service }
if ($ACTIONS -match "u") { update_service }

Write-Host "Done"

