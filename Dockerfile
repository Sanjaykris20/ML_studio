# Multi-stage production build for Unified ML Studio
# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Backend + Static SPA
FROM python:3.11-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    ENVIRONMENT=production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt /app/backend/
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# Copy backend code
COPY backend/ /app/backend/

# Copy built frontend dist from stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Create storage directories
RUN mkdir -p /app/backend/artifacts/datasets /app/backend/artifacts/models

EXPOSE 8000

WORKDIR /app/backend
CMD ["python", "run_server.py"]
