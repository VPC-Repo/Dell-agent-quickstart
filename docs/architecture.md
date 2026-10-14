# Architecture

```
 browser ── ssh tunnel ──► api (FastAPI + built UI, :8766 on loopback) ◄──► runner-db (PostgreSQL 17)
                              │  polls /api/runs/{id} every second         ▲
                              ▼                                             │ runs, events
                           worker ── one process per run ──► LangGraph engine
                              │                                  │ model calls        │ tool calls
                              │                                  ▼                    ▼
                              │                          LiteLLM gateway      PostgreSQL + pgvector, Redis,
                              │                          (toolkit)            sandbox containers, CPU jobs
                              │                                  │
                              │                     vLLM CPU engine, vLLM CUDA engines and replicas
                              └─ samples CPU, memory, GPUs (gpu-telemetry sidecar), power (power-telemetry sidecar)
```

## Installer

`docker compose up` starts the only default service, `installer` (`deploy/installer/`), which runs
`deploy/setup.sh` against the host's Docker through the mounted socket, as a non-root user who owns the checkout.
The repository is mounted at the same absolute path inside the installer, so relative bind mounts in the compose
files it runs resolve on the host. It deploys the toolkit from its own repository at a pinned commit (`toolkit/`,
Docker network `docker_agentic-net`), adds GPU engines with `deploy/gpu_compose.py` (written to
`deploy/gpu/compose.yaml`), registers every engine in LiteLLM with `deploy/litellm_models.py` under its model name
and role alias, writes the placement registry with `deploy/placement.py` (`data/placement.json`), creates the runner's own
gateway key (no admin rights) and writes `.env` with `deploy/configure.py`, starts the runner services (profile `runner`, plus `gpu`), and runs
`python -m backend.bootstrap` and `python -m backend.seed_warehouse` inside the api container. GPUs are found
through a short-lived CUDA container; a GPU with memory in use by anything other than this stack's engines is
skipped.

`install.sh` is only for a server without Docker: it installs Docker and, with NVIDIA GPUs, the driver and the
NVIDIA Container Toolkit, then runs `docker compose up`.

## Runner (`compose.yaml`, project `dell-agent-quickstart`, profile `runner`)

| Service | Role |
|---|---|
| `api` | FastAPI app serving the UI and the API; joins the toolkit network |
| `worker` | Claims queued runs and runs each in its own process; samples resources every 0.5 s |
| `runner-db` | The runner's own PostgreSQL: agents, runs, events |
| `power-telemetry` | Reads Intel RAPL package energy (root-only on the host) and publishes watts to a shared volume |
| `gpu-telemetry` | Profile `gpu`: writes `nvidia-smi` readings to a shared volume every 0.5 s |

The api and worker mount the Docker socket to read container stats and to start sandbox containers.
The API exposes manual runs; it does not provide a visual editor, publishing or recurring schedules.
The initial database schema retains a schedules table and related columns for compatibility with existing
databases. No scheduler reads them.

## Code

| Path | What it is |
|---|---|
| `backend/main.py` | The API: health, system snapshot, catalog status and install, runs and cancel |
| `backend/automations.py` | Stored agent designs and manual run endpoints: list, read, run |
| `backend/demo_catalog.py` | The eight agents and their 20 workers; `install()` writes them, content-addressed, so a changed definition reinstalls cleanly |
| `backend/worker.py`, `deployments.py`, `service.py`, `store.py` | Run queue, run snapshots, database models and Alembic migrations (`migrations/`) |
| `backend/system.py` | Background sampler: CPU per socket, memory, GPUs, containers, vLLM metrics of registered engines |
| `backend/bootstrap.py`, `seed_warehouse.py` | Installer CLIs run inside the api container |
| `xeon_runtime/engine.py` | Compiles a run into a LangGraph graph and executes it; required-tool nudges, retries |
| `xeon_runtime/planning.py`, `readiness.py` | Builds each run's graph from the agent's plan and checks it; output checks |
| `xeon_runtime/providers.py`, `telemetry.py`, `placement.py` | Model adapters, model-call events, and the lookup that names the engine and GPU behind each call |
| `xeon_runtime/enterprise_tools.py`, `demo_tools.py`, `sandbox.py`, `fixtures.py` | The tools, the sharded CPU jobs, the sandbox and the seeded data |
| `frontend/src/` | The UI: `LiveDemo.tsx` (API polling), `demoScreens.tsx` (screens), `demoModel.ts` (run document to timeline), `SystemPanel.tsx` |

## Run events

The worker writes events to `runner-db`; the UI reads them through `GET /api/runs/{id}`.

| Group | Events |
|---|---|
| Lifecycle | `attempt`, `started`, `completed`, `failed`, `retry`, `execution_segment`, `cpu_segment` |
| Planning | `planning`, `plan_created`, `plan_blocked`, `readiness_blocked` |
| Workers | `node_started`, `node_completed`, `node_skipped` |
| Model | `model_request`, `model` (with `placement`: engine, container, device, GPUs), `model_error` |
| Tools | `tool_started`, `tool_completed` (with `work_evidence`), `tool_error` |
| System | `resources` (CPU, memory, per-GPU utilization and memory, power, engines) |

Worker events carry `task_id` and `task_name`.

## Placement

LiteLLM returns the upstream engine it routed a call to in the `x-litellm-model-api-base` header. The runtime
matches it against `data/placement.json`, which lists every engine container (replicas included) with its device
and GPUs, and attaches the result to the call's `model` event.
