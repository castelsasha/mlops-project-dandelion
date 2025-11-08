# -----------------------------------------------------------------------------
# API image: FastAPI + Uvicorn
# - Loads the latest model from MinIO on startup
# - Exposes port 8000
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Avoid Python writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Work directory inside the container
WORKDIR /app

# OS-level deps for Pillow (JPEG/PNG) and building wheels
# Keep the base image lean: install what we actually need only.
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy only the code needed by the API
# (No need to copy the Streamlit webapp here.)
COPY src /app/src

# Expose the FastAPI port
EXPOSE 8000

# Run the API with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]