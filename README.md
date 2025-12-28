# Agentic Analytics Framework

## Key Features

- **GCP Hosted**: Fully hosted on Google Cloud Platform with Cloud Run services and jobs
- **MySQL Integration**: Database connectivity for task tracking, event logging, and data persistence
- **Backend Technologies**: 
  - Go and Python support for flexible backend development
  - RESTful API services for data extraction and processing
- **Frontend**: Google AI Studio integration for user interface and interactions
- **Brain**: Anthropic LLM integration for intelligent decision-making and processing
- **Task & Event Based Architecture**: 
  - Task-driven execution model with status tracking
  - Event logging system for audit trails and monitoring
- **Services**: Cloud Run services for long-running, request-driven workloads
- **Jobs**: Scheduled and on-demand batch processing jobs
- **Scheduler**: Automated job scheduling and execution management
- **Containerized**: Docker-based deployment for consistent environments
- **Multi-language Support**: PowerShell and shell scripts for cross-platform execution

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