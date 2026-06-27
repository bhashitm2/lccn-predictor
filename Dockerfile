# Python 3.13: matches the wheels this service is pinned to (numpy 2.5 / scipy 1.18
# require >=3.12). All deps — curl_cffi, beanie, pymongo, fastapi — ship 3.13 wheels.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY predictor ./predictor
COPY static ./static

EXPOSE 8000

# Bind the platform-injected $PORT (Render/Railway/Fly set it); default 8000 for
# local/compose. Shell form so ${PORT} is expanded.
CMD ["sh", "-c", "uvicorn predictor.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
