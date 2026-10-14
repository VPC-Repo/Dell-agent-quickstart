#!/usr/bin/env bash
# Local development server: the API with the embedded worker and the built
# frontend on http://127.0.0.1:8000, SQLite databases under data/, no token.
# Optional .env.dev settings override shell defaults (for example
# WAREHOUSE_URL, SANDBOX_MODE or a real gateway key).
set -euo pipefail
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-sqlite:///data/dev.db}"
export CHECKPOINT_URL="${CHECKPOINT_URL:-data/dev-checkpoints.db}"
export BUILDER_AUTH="${BUILDER_AUTH:-off}"
export EMBEDDED_WORKER="${EMBEDDED_WORKER:-1}"
export PLACEMENT_REGISTRY="${PLACEMENT_REGISTRY:-data/placement.json}"
export LANGSMITH_TRACING=false
[[ -f .env.dev ]] && set -a && source .env.dev && set +a
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port "${PORT:-8000}"
