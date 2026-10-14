"""Generate the demo fixtures and seed the warehouse ahead of the first run.

    python -m backend.seed_warehouse

Uses WAREHOUSE_URL (the toolkit's PostgreSQL + pgvector) when set, otherwise the
local SQLite warehouse; embeddings use EMBEDDING_MODEL through the gateway when
set. Idempotent: a seeded warehouse at the current version is left alone.
"""

from __future__ import annotations

import json
import sys

from xeon_runtime import fixtures
from xeon_runtime.enterprise_tools import ensure_warehouse, warehouse_backend


def main():
    root = fixtures.ensure()
    print(f"fixtures: {root} ({fixtures.VERSION})")
    result = ensure_warehouse()
    print(json.dumps(result, indent=1, default=str))
    return 0 if warehouse_backend()["kind"] else 1


if __name__ == "__main__":
    sys.exit(main())
