# Dell Agent Quickstart

Run eight example AI agents on one Ubuntu server and watch their model calls, tool calls,
CPU load and GPU load in a browser. The agents work on fictional business data using a
warehouse, search, a cache and isolated code execution.

The installer deploys the [Intel AI for Enterprise Agent Toolkit](https://github.com/intel/enterprise-agent-toolkit)
at a pinned commit: LiteLLM, PostgreSQL with pgvector, Redis, Langfuse and vLLM on CPU.
On servers with suitable NVIDIA GPUs, it adds vLLM GPU engines and replicas. Without
GPUs, the CPU engine serves all three model roles.

## Requirements

| Requirement | Details |
| --- | --- |
| Server | Linux x86_64; Ubuntu 22.04, 24.04 or 26.04. Installations have been tested on 22.04 and 26.04. |
| Docker | Docker Engine and Compose 2.24 or newer. `install.sh` can install them on a bare Ubuntu server. |
| Memory | 192 GB minimum; 256 GB recommended. The measured stack footprint is about 137 GiB. The installer stops below 150 GiB of reported host RAM. |
| Disk | At least 150 GiB free under Docker's data root for CPU deployment; about 400 GiB recommended for all three GPU models and a reserve. The installer skips a new GPU model if its estimated 65 GiB download would leave less than 80 GiB free. |
| GPUs | Optional NVIDIA GPUs with at least 80 GB each and the NVIDIA Container Toolkit. Smaller GPUs and GPUs with other workloads are skipped. |
| Access | A non-root user who owns the checkout and can run Docker; SSH access from your laptop. |

The toolkit publishes several service ports on all host interfaces. Use a trusted
network or restrict inbound access **before installation**; see [SECURITY.md](SECURITY.md).
The quickstart UI and GPU engine ports bind to loopback by default.

## Install

On the server:

```bash
git clone https://github.com/VPC-Repo/Dell-agent-quickstart.git
cd Dell-agent-quickstart
docker compose up
```

On a bare Ubuntu server without Docker, run `./install.sh` instead. It installs Docker
and, when needed, the NVIDIA driver and container runtime, then starts the same installer.
A driver installation may require a reboot.

The first installation downloads images and model weights and can take an hour or more,
depending on the connection and hardware. Rerun `docker compose up` after an interruption.
The installer preserves generated credentials and resumes completed downloads. It waits
for the CPU engine; GPU engines may still be loading when installation finishes.

From your laptop:

```bash
ssh -L 8766:127.0.0.1:8766 <server>
```

Open [http://127.0.0.1:8766](http://127.0.0.1:8766), choose an agent and select
**Run the agent**. **System** shows the host and inference engines. Expand a worker
or call to inspect its timings and execution evidence.

## Documentation

- [Quick start](docs/QUICKSTART.md): installation, restarting, removal and troubleshooting.
- [Example agents](docs/agents.md): workers, tools and model roles.
- [Architecture](docs/architecture.md): installer, runner, UI and execution events.
- [Contributing](CONTRIBUTING.md): local development and checks.
- [Security](SECURITY.md): access, credentials and code execution.

## License

Source code is licensed under the [MIT License](LICENSE). Logos, product images and
fonts have separate terms; see [NOTICE](NOTICE). Downloaded software and models retain
their own licenses.
