# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is required by XGBoost's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt gunicorn

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN pip install -e .

# Train a model at build time so the image ships ready to serve.
# (Comment out to mount a pre-trained model at runtime instead.)
RUN PYTHONPATH=src python scripts/run_pipeline.py --n 12000 --n-iter 8 || true

EXPOSE 5000
ENV PORT=5000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT} creditscore.app.flask_app:app"]
