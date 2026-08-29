# Production Dockerfile for PROJECT ULTRON Backend Kernel
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration
COPY pyproject.toml README.md .env.example /app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application source code
COPY app /app/app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    ULTRON_ENV=production

# Expose port
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
