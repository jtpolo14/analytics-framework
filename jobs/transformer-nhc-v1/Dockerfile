FROM python:3.11-slim

# Prevent Python from buffering logs (important for Cloud Logging)
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Command Cloud Run Job will execute
CMD ["python3", "main.py"]
