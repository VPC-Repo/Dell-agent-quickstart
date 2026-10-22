# Quick start

One Linux server with Docker, reached over SSH as `<host>`. GPUs are optional.

## 1. Install

```bash
ssh <host>
git clone https://github.com/VPC-Repo/Dell-agent-quickstart.git
cd Dell-agent-quickstart
docker compose up
```

Allow an hour or more for the first model downloads. Restrict access to toolkit ports before installing;
see [SECURITY.md](../SECURITY.md). No Docker yet? Run `./install.sh` instead (see appendix A).

## 2. Start again after a reboot

From the repository directory:

```bash
docker compose up
```

The installer checks everything, starts whatever is stopped and exits; it skips downloads that are done.

## 3. Tunnel

On your laptop:

```bash
ssh -L 8766:127.0.0.1:8766 <host>
```

Open http://127.0.0.1:8766.

## 4. Run

Pick an agent, press **Run the agent**, watch the execution page. **System** shows the machine, **About**
explains what you are looking at.

## 5. Stop the GPU engines when the GPUs are needed elsewhere

```bash
docker compose -f deploy/gpu/compose.yaml stop
```

Agents fail while the GPU engines are stopped, because the gateway still routes their roles to them. Start them
again with `docker compose -f deploy/gpu/compose.yaml start`; they take 2 to 10 minutes to become healthy.

## 6. Remove everything

This deletes run history, databases and model caches. From the repository directory:

```bash
docker compose --profile runner --profile gpu down --volumes
[ ! -f deploy/gpu/compose.yaml ] || docker compose -f deploy/gpu/compose.yaml down --volumes
bash toolkit/docker/deploy.sh --down
docker volume ls --filter name=docker_vllm-model-cache-
# Remove only the listed cache volumes that belong to this installation:
# docker volume rm <volume-name>
```

The first line removes the runner and its database, the second the GPU engines and their model cache, the third
the toolkit with its databases, and the last command lists the CPU engines' model cache volumes for manual removal. The images stay; `docker image ls`
lists them. Then delete the repository directory.

---

## Appendix

**A. What the installer does.** `docker compose up` starts the `installer` container (`deploy/setup.sh`), which
drives the host's Docker through its socket. It checks memory and disk. Deploys the Intel enterprise agent toolkit (LiteLLM gateway, PostgreSQL + pgvector, Redis, Langfuse,
a CPU vLLM engine and the embedding model). Builds the runner and sandbox images and downloads each GPU model into
the engines' cache before any GPU engine starts, retrying until the download completes (for gpt-oss also its
tokenizer files, which come from OpenAI's storage, not Hugging Face); the GPU engines then start offline from the
cache, and a rerun skips models already downloaded (a GPU model is skipped when fewer than 80 GiB would stay free
under Docker's data root). Puts one engine per model on the free GPUs that have
at least 80 GB, then replicas on the GPUs left over, all registered in LiteLLM so the gateway balances them; when
GPUs come free after an install without them, the GPU engines replace the CPU model behind each role. Starts
the runner, seeds the demo warehouse and installs the eight agents. GPUs with other processes on them are skipped,
never touched. Settings come from the environment, for example `USE_GPU=no docker compose up` to ignore GPUs, or
`HF_TOKEN` for gated models. On a bare Ubuntu server, `./install.sh` installs Docker (one `sudo`, for packages)
and, with NVIDIA GPUs, the driver and container runtime (a new driver asks for a reboot and a rerun), then runs
`docker compose up`; `./install.sh --status` prints what is running. One-line version for a bare server, which asks the same questions on the terminal:
`curl -fsSL https://raw.githubusercontent.com/VPC-Repo/Dell-agent-quickstart/main/deploy/bootstrap.sh | bash`
(append `-s -- --yes` to accept them all). On a server that already runs containers, `./install.sh` does not restart
Docker to load the NVIDIA runtime; it tells you to restart Docker when that is acceptable and to run it again.

**B. Watch for, when starting.** Three compose projects run: `docker` (the toolkit), `gpu` (the GPU engines,
`deploy/gpu/compose.yaml`) and `dell-agent-quickstart` (the runner). `docker compose down` in the repository stops
only the runner. GPU engines take 2 to 10 minutes to become healthy from cached weights
(`docker ps --format "{{.Names}} {{.Status}}" | grep vllm-gpu`). The first request to a freshly created engine takes
about 30 seconds while vLLM compiles its kernels, so the first run after a restart is slower.

**C. Sign-in.** Off by default (`BUILDER_AUTH=off` in `.env`); the UI listens on loopback only, hence the tunnel.
To share the port, set `BUILDER_AUTH=token` (the token is `BUILDER_API_TOKEN` in `.env`) and `BUILDER_BIND`, then
`docker compose up`. See [SECURITY.md](../SECURITY.md) first.

**D. The UI, in more detail.** The landing page shows this server and the eight agents with the duration of their
last run here. The execution page follows a run every second. "What this agent uses" shows what the agent itself
takes from the server: the cores its own work keeps busy (its processes, its sandbox, the database and cache it
queries) and, below, when each GPU or the CPU engine is working on its requests, stacked by model. A line under the
heading says what is happening now; **Task and activity** opens a side panel with the goal and the full history.
When the run finishes, one line sums it up: core-seconds, output tokens and, on a GPU host, CPUs for every GPU
(the info button explains the estimate). Below that, a timeline per agent with model calls shaded by model (the
legend names them) and tool calls, and the answer at the end. Expanding a row or a call shows timings, the engine and GPU that served the
call, and the tool's evidence. Model-heavy agents can take substantially longer on CPU inference. If the
landing page says the agents are not installed, press **Install the agents**.

**E. Update.** `git pull`, then `docker compose up --build`. The installer rebuilds the runner and reinstalls
the agents when their definitions changed.

**F. If something is off.**
- A GPU engine restarts in a loop and its log says free memory is less than desired: another process holds that
  GPU. Stop this stack's GPU engines, release the conflicting workload only if you own it, and restart
  the engines. Do not assume rerunning the installer resolves a shared GPU.
- The API answers 500 and `runner-db` restarts: Docker's data root is full. Free space; the runner recovers.
- No GPUs in System after the NVIDIA driver was reloaded: the telemetry sidecar restarts itself when `nvidia-smi`
  fails; otherwise `docker compose --profile gpu restart gpu-telemetry`.
- A run fails with a gateway 404 after an engine restart: check engine health and gateway routes, then retry.
  The runtime retries one failed call; a persistent 404 needs investigation.
- Logs: `docker compose logs installer`, `docker compose --profile runner logs -f api worker`; the gateway:
  `docker logs agentic-litellm`.
