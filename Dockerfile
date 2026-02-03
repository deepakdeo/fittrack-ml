# FitTrack ML - HAR Prediction API
# Multi-stage build for optimized image size

# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -e .

# Stage 2: Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /app/wheels

# Install packages from wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /app/wheels/* && \
    rm -rf /app/wheels

# Copy source code
COPY src/ /app/src/
COPY pyproject.toml /app/

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/mlruns && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "fittrack.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
