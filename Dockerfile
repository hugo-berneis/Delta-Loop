FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir -e .

# Build frontend
COPY frontend ./frontend
RUN cd frontend && npm install && npm run build

# Copy source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "deltaloop.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
