# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies
# aria2 is required for parallel_downloader.py
# curl, wget, procps for debugging/healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    aria2 \
    curl \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all scripts
COPY . .

# Set env vars
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python3", "parallel_downloader.py", "--help"]
