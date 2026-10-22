# Contributing

## Local development

Use Python 3.14 and Node.js 22 or newer. The runtime supports Python 3.12 or newer.
You can test the API, runtime and UI without downloading models or installing the toolkit.
When their URLs are unset, warehouse and cache tools use SQLite and vector search uses
a local hashed embedding. Each tool result identifies its backend.

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.lock -e .
.venv/bin/python -m pytest -q
(cd frontend && npm ci && npm run build && npm run test:demo)
bash deploy/dev-server.sh
```

The development server serves the built UI at [http://127.0.0.1:8000](http://127.0.0.1:8000).
It uses SQLite and an embedded worker. Model calls still require an OpenAI-compatible
gateway; set its key in your shell and register it with:

```bash
.venv/bin/python -m backend.bootstrap \
  --gateway http://127.0.0.1:<port>/v1 --key-env <KEY_VARIABLE> --install-catalog
```

Code execution defaults to Docker and requires the sandbox image:

```bash
docker build -t dell-agent-quickstart-sandbox:local deploy/sandbox
```

For trusted local development only, `SANDBOX_MODE=process bash deploy/dev-server.sh`
enables execution on your host. It can read host files and access the network. Tests
that need this mode select it explicitly.

## Checks

CI runs the backend tests, frontend build and tests, formatting and import checks,
shell syntax and Compose validation, image builds and a database startup check.
It also scans Git history for secrets and audits the locked dependencies.

Optional integration tests require Docker and a PostgreSQL server with pgvector:

```bash
QUICKSTART_TEST_DOCKER=1 .venv/bin/python -m pytest -q tests/test_enterprise_tools.py
QUICKSTART_TEST_PG=<postgresql-url> .venv/bin/python -m pytest -q tests/test_enterprise_tools.py
```

Formatting uses Ruff 0.17.0 and the frontend's pinned Prettier:

```bash
.venv/bin/pip install ruff==0.17.0
.venv/bin/ruff check backend xeon_runtime deploy tests --select F
.venv/bin/ruff format --check backend xeon_runtime deploy tests
(cd frontend && npm run format:check)
bash -n install.sh deploy/*.sh deploy/installer/entrypoint.sh
docker compose config -q
docker compose --profile runner --profile gpu config -q
```

## Changing agents and installation

Agent and worker definitions live in `backend/demo_catalog.py`. A definition change
produces a new catalog version, installed on the next `docker compose up`. Older demo
records are archived; user-created records and completed run snapshots are retained.
The API supports listing agents and starting manual runs. There is no visual editor,
publishing UI or schedule runner.

Keep `deploy/setup.sh` idempotent. A rerun must preserve credentials and avoid taking
over unrelated containers or GPUs. Add regression tests for changes to installation,
security boundaries or run behavior. Record the hardware and commands used for claims
about performance or full installation. Keep local credentials and machine notes out
of version control.
