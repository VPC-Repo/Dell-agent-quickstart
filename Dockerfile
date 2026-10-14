FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 LANGSMITH_TRACING=false
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock && useradd --create-home --uid 10001 builder
COPY pyproject.toml LICENSE alembic.ini ./
COPY migrations/ migrations/
COPY backend/ backend/
COPY xeon_runtime/ xeon_runtime/
COPY --from=frontend /build/dist/ frontend/dist/
RUN chmod -R a+rX /app && mkdir data && chown builder:builder data \
    && mkdir -p /data/fixtures /data/artifacts /data/telemetry && chown -R builder:builder /data
USER builder
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
