# -----------------------------------------------------------------------------
# Streamlit WebApp image:
# - Calls the API via API_URL (default set by docker-compose)
# - Exposes port 8501
# -----------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# OS-level deps for Pillow (image handling) and wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Streamlit is already pinned in requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy only what the webapp needs
COPY webapp /app/webapp
COPY src /app/src

# API URL inside the Docker network (overridden by compose if needed)
ENV API_URL=http://api:8000/predict

# Expose Streamlit port
EXPOSE 8501

# Launch Streamlit
CMD ["streamlit","run","webapp/app.py","--server.port","8501","--server.address","0.0.0.0"]