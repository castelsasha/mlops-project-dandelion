FROM python:3.11-slim

WORKDIR /app

# deps systèmes pour Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libjpeg-dev zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir streamlit

COPY webapp /app/webapp
COPY src /app/src

ENV API_URL=http://api:8000/predict

EXPOSE 8501
CMD ["streamlit","run","webapp/app.py","--server.port","8501","--server.address","0.0.0.0"]
