#!/usr/bin/env python3
"""
Script to create a new job from a template.
All jobs follow a standard 5-file structure:
- Dockerfile
- job.yaml
- main.py (or custom name)
- requirements.txt
- run.ps1
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def validate_job_name(job_name: str) -> bool:
    """Validate job name contains only lowercase letters, numbers, and hyphens."""
    return bool(re.match(r'^[a-z0-9-]+$', job_name))


def replace_placeholders(content: str, template_job: str, new_job: str, 
                        full_job_name: str, image_name: str, 
                        project_id: str, region: str) -> str:
    """Replace placeholders in file content."""
    # Replace job name references (with and without 'job-' prefix)
    template_full_job_name = f"job-{template_job}"
    content = content.replace(template_full_job_name, full_job_name)
    content = content.replace(template_job, new_job)
    
    # Replace image references (any region/project pattern)
    image_pattern = r'[a-z0-9-]+-docker\.pkg\.dev/[^/]+/cloud-run-source-deploy/[^:]+:latest'
    content = re.sub(image_pattern, image_name, content)
    
    # Replace PowerShell variables in run.ps1
    content = re.sub(r'\$PROJECT_ID = "[^"]+"', f'$PROJECT_ID = "{project_id}"', content)
    content = re.sub(r'\$REGION = "[^"]+"', f'$REGION = "{region}"', content)
    content = re.sub(r'\$JOB_NAME = "[^"]+"', f'$JOB_NAME = "{full_job_name}"', content)
    content = re.sub(r'\$IMAGE = "[^"]+"', f'$IMAGE = "{image_name}"', content)
    
    return content


def find_python_file(template_path: Path) -> str | None:
    """Find the Python file in the template directory."""
    if (template_path / "main.py").exists():
        return "main.py"
    
    # Find any .py file
    py_files = list(template_path.glob("*.py"))
    if py_files:
        return py_files[0].name
    
    return None


def create_job(job_name: str, template_job: str = "extractor-coin-data",
               python_filename: str = "main.py", project_id: str = "dulcet-iterator-344116",
               region: str = "us-central1"):
    """Create a new job from a template."""
    
    # Validate job name
    if not validate_job_name(job_name):
        print("Error: Job name can only contain lowercase letters, numbers, and hyphens", 
              file=sys.stderr)
        sys.exit(1)
    
    # Get script directory and paths
    script_dir = Path(__file__).parent
    jobs_dir = script_dir / "jobs"
    template_path = jobs_dir / template_job
    new_job_path = jobs_dir / job_name
    
    # Check if template exists
    if not template_path.exists():
        print(f"Error: Template job '{template_job}' not found at {template_path}", 
              file=sys.stderr)
        sys.exit(1)
    
    # Check if new job already exists
    if new_job_path.exists():
        print(f"Error: Job '{job_name}' already exists at {new_job_path}", 
              file=sys.stderr)
        sys.exit(1)
    
    print(f"Creating new job: {job_name}")
    print(f"Template: {template_job}")
    print(f"Destination: {new_job_path}")
    
    # Create new job directory
    new_job_path.mkdir(parents=True, exist_ok=True)
    
    # Job name with 'job-' prefix
    full_job_name = f"job-{job_name}"
    image_name = f"{region}-docker.pkg.dev/{project_id}/cloud-run-source-deploy/{full_job_name}:latest"
    
    # Files to copy and process
    files_to_copy = ["Dockerfile", "job.yaml", "requirements.txt", "run.ps1"]
    
    # Copy and process each file
    for filename in files_to_copy:
        source_file = template_path / filename
        dest_file = new_job_path / filename
        
        if source_file.exists():
            print(f"  Copying {filename}...")
            content = source_file.read_text(encoding='utf-8')
            
            # Replace placeholders
            content = replace_placeholders(content, template_job, job_name, 
                                         full_job_name, image_name, project_id, region)
            
            dest_file.write_text(content, encoding='utf-8')
        else:
            print(f"  Warning: {filename} not found in template")
    
    # Handle Python file
    template_python_file = find_python_file(template_path)
    
    if template_python_file:
        source_python_file = template_path / template_python_file
        dest_python_file = new_job_path / python_filename
        
        print(f"  Copying Python file ({template_python_file} -> {python_filename})...")
        content = source_python_file.read_text(encoding='utf-8')
        
        # Replace JOB_TAG in Python file
        content = re.sub(r"JOB_TAG = '[^']+'", f"JOB_TAG = '{full_job_name}'", content)
        content = re.sub(r'JOB_TAG = "[^"]+"', f'JOB_TAG = "{full_job_name}"', content)
        
        # Update Dockerfile CMD if Python filename is different
        if python_filename != "main.py":
            dockerfile_path = new_job_path / "Dockerfile"
            if dockerfile_path.exists():
                docker_content = dockerfile_path.read_text(encoding='utf-8')
                docker_content = re.sub(
                    r'CMD \["python3", "main\.py"\]',
                    f'CMD ["python3", "{python_filename}"]',
                    docker_content
                )
                dockerfile_path.write_text(docker_content, encoding='utf-8')
        
        dest_python_file.write_text(content, encoding='utf-8')
    else:
        print("  Warning: No Python file found in template")
    
    print(f"\nJob created successfully!")
    print(f"Location: {new_job_path}")
    print(f"\nNext steps:")
    print(f"  1. Edit {python_filename} to implement your job logic")
    print(f"  2. Update job.yaml with your environment variables")
    print(f"  3. Update requirements.txt with your dependencies")
    print(f"  4. Review and update run.ps1 if needed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create a new job from a template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s my-new-job
  %(prog)s my-new-job --template extractor-stock-data
  %(prog)s my-transformer --template transformer-nhc-v1 --python-file processor.py
  %(prog)s processor --python-file processor.py --project-id my-project --region us-east1
        """
    )
    
    parser.add_argument(
        "job_name",
        help="Name of the new job (lowercase letters, numbers, and hyphens only)"
    )
    
    parser.add_argument(
        "--template", "-t",
        default="extractor-coin-data",
        dest="template_job",
        help="Template job to copy from (default: extractor-coin-data)"
    )
    
    parser.add_argument(
        "--python-file", "-p",
        default="main.py",
        dest="python_filename",
        help="Name for the Python file (default: main.py)"
    )
    
    parser.add_argument(
        "--project-id",
        default="dulcet-iterator-344116",
        help="GCP Project ID (default: dulcet-iterator-344116)"
    )
    
    parser.add_argument(
        "--region", "-r",
        default="us-central1",
        help="GCP Region (default: us-central1)"
    )
    
    args = parser.parse_args()
    
    create_job(
        job_name=args.job_name,
        template_job=args.template_job,
        python_filename=args.python_filename,
        project_id=args.project_id,
        region=args.region
    )


if __name__ == "__main__":
    main()

