# Security

This stack is intended for a server you control on a trusted network. It is not a
multi-user security boundary.

## Network access

The quickstart UI binds to `127.0.0.1:8766` by default. Use an SSH tunnel to reach it.
`BUILDER_AUTH=off` disables authentication for clients that can reach the API.

The pinned toolkit publishes LiteLLM (4000), PostgreSQL (5432), Redis (6379), Langfuse
(3002), the MinIO console (9001) and CPU inference engines (2080 and up) on all host
interfaces. Restrict inbound access to those ports before running the installer. Check
Docker's published ports and your network firewall; Docker port publishing can bypass
some host firewall rules. GPU inference ports (2180 and up) bind to loopback.

To expose the UI beyond loopback, set `BUILDER_AUTH=token`, keep a non-empty
`BUILDER_API_TOKEN`, and set `BUILDER_BIND` in `.env`. Restart with `docker compose up`.
Token mode refuses requests when its token is missing. Use TLS at a reverse proxy or
keep using SSH: the API does not provide HTTPS itself. All token holders have the same
access, including starting and canceling runs and inspecting their results.

## Host access and code execution

The installer, API and worker mount the Docker socket. Access to it is equivalent to
root access on the host. Run this stack only where you trust its code and operators.

Agent-written code runs in throwaway Docker containers with no network, dropped
capabilities, no privilege escalation, and CPU, memory, process and log limits. A
missing image or Docker failure fails the call. Build the sandbox image before local
use; installation builds it automatically.

`SANDBOX_MODE=process` is an explicit development option. Its resource limits do not
prevent access to host files or the network. Do not use it for untrusted code.

## Credentials and retained data

The installer generates local credentials and writes `.env` with owner-only permissions.
It preserves non-empty credentials on reruns and replaces blank example values. The
runner's `GATEWAY_KEY` can call models but cannot administer the gateway. The master key
and toolkit service passwords stay in `toolkit/docker/`.

If a model requires `HF_TOKEN`, supply it in the environment. The installer writes it
to the ignored toolkit configuration with owner-only permissions. Do not commit `.env`,
`toolkit/`, `data/`, private keys or production data. Do not share resolved Compose
configuration or deployment logs without checking them for credentials.

Run events and artifacts can contain prompts, outputs and tool results. Treat them as
sensitive if you replace the demo data. Completed database runs and events expire after
30 days by default (`RETENTION_DAYS=0` retains them); artifact files and model checkpoints
are not removed by that policy.

## Reporting

Report vulnerabilities through the repository's GitHub private vulnerability reporting
feature when enabled. Otherwise, contact the repository owner privately. Do not put
credentials or exploit details in a public issue.
