# syntax=docker/dockerfile:1

# =============================================================================
# BS-Opt ML Worker - Multi-Stage Build
# Heavy ML dependencies (PyTorch, Ray) isolated in builder stage
# This is the "System Killer" - designed to be built in CI, not locally
# =============================================================================

# STAGE 1: Base - OS Dependencies
FROM python:3.10-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=300

# Runtime dependencies (OpenMP for parallel computing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# STAGE 2: Builder - Heavy Compilation Stage
# ⚠️ WARNING: This stage uses significant CPU/RAM
# Designed to run in GitHub Actions, not on developer laptops
FROM base AS builder

WORKDIR /app

# Install all build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gfortran \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 🚀 OPTIMIZATION: Copy requirements FIRST for layer caching
COPY requirements/base.txt requirements/ml.txt ./requirements/

# Install dependencies in stages to maximize cache hits
# Base dependencies first (faster, smaller)
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with CPU-only to reduce size (use CUDA image for GPU)
RUN pip install torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining ML dependencies (will use cached torch)
RUN pip install -r requirements/ml.txt --no-deps || \
    pip install -r requirements/ml.txt

# STAGE 3: Runtime - Lean Production Image
FROM base AS runtime

WORKDIR /app

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser -u 1000 mluser

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ML source code
COPY --chown=mluser:mluser src/ml/ ./src/ml/
COPY --chown=mluser:mluser src/pricing/ ./src/pricing/
COPY --chown=mluser:mluser src/data/ ./src/data/

# Create directories for models and data
RUN mkdir -p /app/models /app/data && chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Environment for Ray
ENV RAY_OBJECT_STORE_MEMORY=4000000000
ENV RAY_memory_monitor_refresh_ms=0

# Default command
CMD ["python", "src/ml/autonomous_pipeline.py"]
