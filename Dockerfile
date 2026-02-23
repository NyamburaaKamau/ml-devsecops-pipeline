# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: Production Image ────────────────────────────────────────────────
# Security: distroless-style minimal image
FROM python:3.11-slim AS production

# Security: create non-root user
RUN groupadd --gid 1001 mluser && \
    useradd --uid 1001 --gid mluser --shell /bin/sh --create-home mluser

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /root/.local /home/mluser/.local

# Copy app code
COPY examples/sample-model/serve.py .
COPY ml/ ./ml/

# Security: set ownership
RUN chown -R mluser:mluser /app

# Security: switch to non-root user
USER mluser

# Security: set PATH for local packages
ENV PATH=/home/mluser/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=ml/model/model.joblib
ENV MODEL_VERSION=1.0.0
ENV ENVIRONMENT=production

EXPOSE 8000

# Health check built into image
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Security: use exec form to avoid shell injection
CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]