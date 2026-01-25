# backend/Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install OS packages required for many Python packages (and for PyTorch wheel)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
      libsndfile1 \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements early to leverage Docker cache
COPY backend/requirements.txt ./requirements.txt

# Upgrade pip & install core requirements
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU wheel explicitly (recommended for stable CPU behavior)
# If you want GPU support, replace this line with your preferred wheel install
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Copy application sources
COPY backend/ /app/

# Expose port used by uvicorn
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/ || exit 1

# Default command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
