# Agentic Analytics Framework

## Key Features

- **GCP Hosted**: Fully hosted on Google Cloud Platform with Cloud Run services and jobs
- **MySQL Integration**: Database connectivity for task tracking, event logging, and data persistence
- **Daily Stock Market Feed**: Automated daily extraction of stock market data for 77+ major symbols via Financial Modeling Prep API, with comprehensive error handling and GCS storage
- **Backend Technologies**: 
  - Go and Python support for flexible backend development
  - RESTful API services for data extraction and processing
- **Frontend**: Google AI Studio integration for user interface and interactions https://ai.studio/apps/drive/1gV10EtNh60RRidtm36zYrtAzYiJ10Xkl
- **Brain**: Anthropic LLM integration for intelligent decision-making and processing
- **Task & Event Based Architecture**: 
  - Task-driven execution model with status tracking
  - Event logging system for audit trails and monitoring
- **Services**: Cloud Run services for long-running, request-driven workloads
- **Jobs**: Scheduled and on-demand batch processing jobs
- **Scheduler**: Automated job scheduling and execution management
- **Containerized**: Docker-based deployment for consistent environments
- **Multi-language Support**: PowerShell and shell scripts for cross-platform execution
- **DevOps Automation**: Template-based job creation script (`create_job.py`) for rapid job scaffolding with automatic placeholder replacement, configuration updates, and standardized file structure generation

## Framework Structure

```
analytics-framework/
├── jobs/
│   ├── extractor-coin-data/
│   │   ├── Dockerfile
│   │   ├── job.yaml
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── run.ps1
│   ├── extractor-stock-data/
│   │   ├── Dockerfile
│   │   ├── job.yaml
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── run.ps1
│   ├── extractor-v1/
│   │   ├── Dockerfile
│   │   ├── extractor.py
│   │   ├── job.yaml
│   │   ├── requirements.txt
│   │   └── run.ps1
│   ├── transformer-nhc-v1/
│   │   ├── dev/
│   │   │   ├── crypto_data.json
│   │   │   ├── get_token_data.py
│   │   │   └── test.py
│   │   ├── Dockerfile
│   │   ├── job.yaml
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── run.ps1
│   └── transformer-task-cleaner/
│       ├── Dockerfile
│       ├── job.yaml
│       ├── main.py
│       ├── requirements.txt
│       └── run.ps1
├── services/
│   └── server-extractor-coin-data/
│       ├── Dockerfile
│       ├── main.py
│       ├── requirements.txt
│       ├── run.ps1
│       └── service.yaml
└── README.md
```

## TODO: 
- add the matching shell and/or powershell scripts for execution on linux/win
- add Agentic-to-Human loop
- add scheduler config to the jobs run scripts 

## Creating New Jobs

All jobs follow a standard 5-file structure:
- `Dockerfile` - Container definition
- `job.yaml` - Cloud Run job configuration
- `main.py` (or custom name) - Main Python script
- `requirements.txt` - Python dependencies
- `run.ps1` - PowerShell deployment script

### Using the Job Creation Script

Use the `create_job.py` script to quickly create a new job from a template:

```bash
# Basic usage - creates a new job from the default template
python create_job.py my-new-job

# Use a different template job
python create_job.py my-new-job --template extractor-stock-data

# Customize the Python filename
python create_job.py my-new-job --python-file processor.py

# Full example with all options
python create_job.py transformer-custom --template transformer-nhc-v1 --python-file transformer.py --project-id my-project --region us-east1
```

The script will:
1. Create a new job folder in `jobs/`
2. Copy all template files
3. Replace placeholders (job names, image names, etc.) with your new job name
4. Update configuration files with your project settings

After creation, you'll need to:
1. Edit the Python file to implement your job logic
2. Update `job.yaml` with your environment variables
3. Update `requirements.txt` with your dependencies
4. Review `run.ps1` if you need custom deployment settings

## Standards

### Commit Message Guidelines

When creating commit messages, follow these steps:

1. **Check git status** to see what files have been modified:
   ```powershell
   git status
   ```

2. **Review the changes** using git diff:
   ```powershell
   git diff <file_path>
   ```

3. **Create a commit message** following conventional commit format:
   - Use a short, descriptive subject line (50 chars or less)
   - Start with a type prefix: `feat:`, `fix:`, `docs:`, `refactor:`, etc.
   - Add bullet points for detailed changes (optional but recommended)
   
   Example:
   ```
   feat: enhance stats endpoint with task metadata
   
   - Include task_id, endpoint_hits, and timestamp in stats response
   - Improve deployment script message clarity
   ```

4. **Commit the changes**:
   ```powershell
   git commit -a -m "feat: enhance stats endpoint with task metadata" -m "- Include task_id, endpoint_hits, and timestamp in stats response" -m "- Improve deployment script message clarity"
   ```
