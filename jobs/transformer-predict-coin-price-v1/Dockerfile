FROM python:3.11-slim

# Prevent Python from buffering logs (important for Cloud Logging)
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (includes runner.py, btc_price_model_v1_1.py, etc.)
COPY . .

# Command Cloud Run Job will execute
CMD ["python3", "main.py"]
