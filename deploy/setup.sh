#!/usr/bin/env bash
# The installation, run by the `installer` service of compose.yaml (`docker compose up`). It drives the host's
# Docker through the mounted socket, so everything it starts are ordinary containers on the host:
#   1. checks memory and disk, finds the GPUs (through a short-lived CUDA container) and which are free;
#   2. deploys the Intel enterprise agent toolkit at a pinned commit with the toolkit's own Docker deployment:
#      vLLM on CPU (Qwen3-30B-A3B and the embedding model bge-base-en-v1.5), LiteLLM, PostgreSQL + pgvector,
#      Redis, Langfuse;
#   3. on GPU hosts starts vLLM CUDA engines (Qwen3-30B-A3B, gpt-oss-120b, Qwen3-Coder-30B-A3B) on free GPUs
#      with enough memory, then replicas on the GPUs left over, registered in LiteLLM under the role aliases
#      general, coder and reasoning so the gateway balances the replicas;
#   4. writes the placement registry and .env, starts the runner (profile `runner`), seeds the warehouse and
#      installs the eight agents.
# Idempotent: run it again (docker compose up) after an interruption or a reboot.
# Settings, from the environment of `docker compose up`: USE_GPU=no ignores GPUs, MIN_GPU_GB (default 80),
# HF_TOKEN for gated models, WAIT_MODELS_MINUTES (default 90).
set -euo pipefail
umask 077

TOOLKIT_REPO="https://github.com/intel/enterprise-agent-toolkit.git"
TOOLKIT_COMMIT="f47defae58ef009b80b8493264ae54d6f4270d3c"
CPU_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# Served by the toolkit next to the chat model, for agents that need vector search through the gateway.
EMBED_MODEL="BAAI/bge-base-en-v1.5"
GPU_MODELS=("Qwen/Qwen3-30B-A3B-Instruct-2507" "openai/gpt-oss-120b" "Qwen/Qwen3-Coder-30B-A3B-Instruct")
declare -A ROLE_OF_GPU_MODEL=(["Qwen/Qwen3-30B-A3B-Instruct-2507"]=general ["openai/gpt-oss-120b"]=reasoning ["Qwen/Qwen3-Coder-30B-A3B-Instruct"]=coder)
CUDA_IMAGE="nvidia/cuda:12.6.3-base-ubuntu24.04"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLKIT_DIR="$ROOT/toolkit"
LITELLM_URL="http://127.0.0.1:4000"
USE_GPU="${USE_GPU:-auto}"
MIN_GPU_GB="${MIN_GPU_GB:-80}"
WAIT_MODELS_MINUTES="${WAIT_MODELS_MINUTES:-90}"

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; NC=$'\033[0m'
step() { echo; echo "${BOLD}== $*${NC}"; }
ok() { echo "${GREEN}✔${NC} $*"; }
warn() { echo "${YELLOW}⚠${NC} $*"; }
die() { echo "${RED}✘${NC} $*" >&2; exit 1; }
env_value() { grep -E "^$2=" "$1" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' ; }
runner() { docker compose --project-directory "$ROOT" -f "$ROOT/compose.yaml" "$@"; }

# ---------------------------------------------------------------- 1. The host
step "Checking the host"
docker info >/dev/null 2>&1 || die "Cannot reach Docker through /var/run/docker.sock."
MEM_GB=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
# Measured on a PowerEdge XE7740: the CPU chat engine holds about 122 GiB (the toolkit's 60 GiB cache plus weights and runtime),
# the embedding engine 2.3 GiB and the rest of the stack about 13 GiB, 137 GiB in all. MemTotal reads below the
# installed size, so a 192 GB server passes the first check and a 256 GB server the second.
if [[ "$MEM_GB" -lt 150 ]]; then
  die "${MEM_GB} GiB of RAM: the stack uses about 137 GiB (192 GB minimum, 256 GB recommended)."
elif [[ "$MEM_GB" -lt 240 ]]; then
  warn "${MEM_GB} GiB of RAM: 256 GB is recommended; the stack itself uses about 137 GiB."
else
  ok "${MEM_GB} GB of RAM"
fi
# Images and weights live under Docker's data root, a host path; measure it from a small container.
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
FREE_GB=$(docker run --rm -v "$DOCKER_ROOT:/r:ro" alpine:3.20 df -Pk /r | awk 'NR==2 {print int($4/1024/1024)}')
if [[ "$FREE_GB" -lt 150 ]]; then
  warn "Only ${FREE_GB} GB free under ${DOCKER_ROOT} (Docker's data root). The CPU model needs about 60 GB, each GPU model about 65 GB, the images about 55 GB."
else
  ok "${FREE_GB} GB free under ${DOCKER_ROOT}"
fi

# The toolkit publishes 4000 (LiteLLM), 5432 (PostgreSQL), 6379 (Redis), 3002 (Langfuse), 2080 and 2081 (the chat and
# embedding engines, numbered from 2080 in model order); the runner 8766.
for port in 4000 5432 6379 3002 2080 2081 8766; do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}$" \
     && ! docker ps --format '{{.Names}} {{.Ports}}' | grep -qE "(agentic|quickstart).*:${port}->"; then
    warn "Port ${port} is in use by something other than this stack; the deployment may fail to bind it."
  fi
done

# ---------------------------------------------------------------- 2. GPUs
FREE_GPUS=()
if [[ "$USE_GPU" != "no" ]]; then
  step "Looking for NVIDIA GPUs"
  if GPU_CSV="$(docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    # GPUs our own engines hold (a rerun) are ours to reuse; any other GPU with memory in use belongs to someone else.
    OURS=" $(for c in $(docker ps --filter label=com.docker.compose.project=gpu --filter "label=com.docker.compose.project.working_dir=$ROOT/deploy/gpu" --format '{{.Names}}'); do docker inspect -f '{{range .HostConfig.DeviceRequests}}{{join .DeviceIDs " "}}{{end}}' "$c"; done | tr '\n' ' ') "
    while IFS=, read -r index name total used; do
      index="${index// /}"; total="${total// /}"; used="${used// /}"
      gb=$(( total * 1024 * 1024 / 1000 / 1000 / 1000 ))
      if [[ "$OURS" != *" $index "* && "$used" -gt 2048 ]]; then
        warn "GPU $index (${name# }) is in use by another workload; skipping it."
      elif [[ "$gb" -lt "$MIN_GPU_GB" ]]; then
        warn "GPU $index (${name# }) has ${gb} GB; the GPU models need ${MIN_GPU_GB} GB. Skipping it."
      else
        FREE_GPUS+=("$index")
      fi
    done <<<"$GPU_CSV"
    ok "GPUs for the engines: ${FREE_GPUS[*]:-none}"
  else
    ok "No GPUs visible to containers; CPU inference only. (For GPUs, install the NVIDIA driver and the NVIDIA Container Toolkit.)"
  fi
fi

# ---------------------------------------------------------------- 3. The toolkit
step "Intel enterprise agent toolkit (pinned to ${TOOLKIT_COMMIT:0:12})"
[[ -d "$TOOLKIT_DIR/.git" ]] || git clone -q "$TOOLKIT_REPO" "$TOOLKIT_DIR"
git -C "$TOOLKIT_DIR" fetch -q origin "$TOOLKIT_COMMIT" 2>/dev/null || git -C "$TOOLKIT_DIR" fetch -q origin
git -C "$TOOLKIT_DIR" checkout -q "$TOOLKIT_COMMIT"
cat > "$TOOLKIT_DIR/docker/config.cfg" <<EOF
hugging_face_token=${HF_TOKEN:-}
models=${CPU_MODEL},${EMBED_MODEL}
deploy_docker_fresh=off
deploy_agenticai_plugin=off
deploy_observability=on
http_proxy=${http_proxy:-}
https_proxy=${https_proxy:-}
no_proxy=${no_proxy:-}
EOF
chmod 600 "$TOOLKIT_DIR/docker/config.cfg"   # it can hold HF_TOKEN
echo "Deploying the toolkit's Docker stack. The first model download takes a while."
# The toolkit asks for confirmation; this is the confirmation.
(cd "$TOOLKIT_DIR" && ./deploy-agentic-stack.sh --platform=docker < <(yes))
TOOLKIT_ENV="$TOOLKIT_DIR/docker/.env"
[[ -f "$TOOLKIT_ENV" ]] || die "The toolkit did not produce $TOOLKIT_ENV"
LITELLM_MASTER_KEY="$(env_value "$TOOLKIT_ENV" LITELLM_MASTER_KEY)"
PGVECTOR_PASSWORD="$(env_value "$TOOLKIT_ENV" PGVECTOR_PASSWORD)"
REDIS_PASSWORD="$(env_value "$TOOLKIT_ENV" REDIS_PASSWORD)"
[[ -n "$LITELLM_MASTER_KEY" ]] || die "LITELLM_MASTER_KEY missing from $TOOLKIT_ENV"
export LITELLM_MASTER_KEY
for _ in $(seq 1 60); do curl -fsS "$LITELLM_URL/health/liveliness" >/dev/null 2>&1 && break; sleep 5; done
curl -fsS "$LITELLM_URL/health/liveliness" >/dev/null 2>&1 || die "LiteLLM is not answering at $LITELLM_URL"
ok "LiteLLM is up"
# The toolkit names its engine containers from docker/model-configs.yaml; read the name from what it generated.
CPU_CONTAINER="$(awk -v model="$CPU_MODEL" '
  /container_name:/ {container=$2}
  /--served-model-name/ {getline; gsub(/[" ]/, "", $0); sub(/^-/, "", $0); if ($0 == model && container != "") {print container; exit}}
' "$TOOLKIT_DIR/docker/docker-compose.vllm.yml" 2>/dev/null)"
[[ -n "$CPU_CONTAINER" ]] || CPU_CONTAINER="$(docker ps --format '{{.Names}}' | grep -E '^agentic-vllm-' | head -1)"
[[ -n "$CPU_CONTAINER" ]] || die "Cannot find the toolkit's vLLM container for $CPU_MODEL"
echo "Waiting up to ${WAIT_MODELS_MINUTES} min for ${CPU_CONTAINER} to load ${CPU_MODEL} ..."
deadline=$(( $(date +%s) + WAIT_MODELS_MINUTES * 60 ))
until docker inspect --format '{{.State.Health.Status}}' "$CPU_CONTAINER" 2>/dev/null | grep -qx healthy; do
  [[ $(date +%s) -lt $deadline ]] || die "CPU engine is not healthy after ${WAIT_MODELS_MINUTES} min. Check: docker logs -f ${CPU_CONTAINER}; then rerun docker compose up."
  sleep 20; printf '.'
done; echo
ok "CPU engine: ${CPU_CONTAINER}"
# The runner gets a gateway key of its own that can call models but not change the gateway.
GATEWAY_KEY="$(GATEWAY_KEY="$(env_value "$ROOT/.env" GATEWAY_KEY)" python3 "$ROOT/deploy/litellm_models.py" --base "$LITELLM_URL" \
  client-key --alias agent-quickstart-runner --current-env GATEWAY_KEY)" || die "Could not create the runner's gateway key"
export GATEWAY_KEY
# Keys travel in a header file, never on a command line where other processes could read them.
AUTH_HEADER="$(mktemp)"; chmod 600 "$AUTH_HEADER"; trap 'rm -f "$AUTH_HEADER"' EXIT
printf 'Authorization: Bearer %s\n' "$GATEWAY_KEY" > "$AUTH_HEADER"
# Check that the gateway serves the embedding model (about a minute to load on a PowerEdge XE7740). The demo agents do not use it.
embed_ok=""
for _ in $(seq 1 60); do
  curl -fsS "$LITELLM_URL/v1/embeddings" -H @"$AUTH_HEADER" -H "Content-Type: application/json" \
    -d "{\"model\": \"${EMBED_MODEL}\", \"input\": \"ping\"}" >/dev/null 2>&1 && { embed_ok=1; break; }
  sleep 10
done
if [[ -n "$embed_ok" ]]; then ok "Embedding model: ${EMBED_MODEL}"; else warn "The gateway does not serve ${EMBED_MODEL} yet; check: docker ps | grep bge"; fi

# ---------------------------------------------------------------- 4. GPU engines
# Recheck after the CPU model and toolkit images have consumed disk space.
FREE_GB=$(docker run --rm -v "$DOCKER_ROOT:/r:ro" alpine:3.20 df -Pk /r | awk 'NR==2 {print int($4/1024/1024)}')
PLACEMENT_ARGS=(--model "${CPU_MODEL}|vllm-cpu|cpu||${CPU_CONTAINER}")
ALIAS_ARGS=()
ASSIGN=()
if [[ ${#FREE_GPUS[@]} -gt 0 ]]; then
  step "vLLM CUDA engines on GPUs ${FREE_GPUS[*]}"
  # A new model is a 60 to 65 GB download; start one only while the disk keeps 80 GB free, because a full disk
  # takes the databases down with it. A model whose container exists already has its weights cached.
  NEW_GB=0; i=0
  for model in "${GPU_MODELS[@]}"; do
    [[ $i -lt ${#FREE_GPUS[@]} ]] || { warn "Not enough GPUs for ${model}; skipped."; continue; }
    container="vllm-gpu-$(echo "$model" | tr '[:upper:]' '[:lower:]' | sed 's#/#-#g; s/[^a-z0-9-]/-/g; s/^-*//; s/-*$//')"
    if ! docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
      if [[ $(( FREE_GB - NEW_GB - 65 )) -lt 80 ]]; then
        warn "Only $(( FREE_GB - NEW_GB - 65 )) GiB would be left under ${DOCKER_ROOT} after downloading ${model}; skipped."
        continue
      fi
      NEW_GB=$((NEW_GB + 65))
    fi
    ASSIGN+=(--assign "${model}=${FREE_GPUS[$i]}"); i=$((i+1))
  done
  # The GPUs left over take replicas, round-robin; they share the weights through the cache volume.
  ASSIGNED=(); for a in "${ASSIGN[@]}"; do [[ "$a" == --assign ]] || ASSIGNED+=("${a%%=*}"); done
  r=0
  while [[ ${#ASSIGNED[@]} -gt 0 && $i -lt ${#FREE_GPUS[@]} ]]; do
    ASSIGN+=(--assign "${ASSIGNED[$(( r % ${#ASSIGNED[@]} ))]}=${FREE_GPUS[$i]}"); i=$((i+1)); r=$((r+1))
  done
  [[ $r -gt 0 ]] && ok "Replicas on the other GPUs: $r"
fi
if [[ ${#ASSIGN[@]} -gt 0 ]]; then
  mapfile -t GPU_PLACEMENTS < <(python3 "$ROOT/deploy/gpu_compose.py" --out "$ROOT/deploy/gpu/compose.yaml" "${ASSIGN[@]}")
  # Download every model into the engines' cache volume before any engine starts. A first download from inside a
  # starting engine often stalls, and the engine never comes up. snapshot_download resumes where it stopped, so a
  # retry continues the download, and the engines then start offline from the cache. The download runs in the
  # model's own engine service, so it uses the same volume, token and image.
  # gpt-oss also loads its tokenizer vocabulary (tiktoken files) from OpenAI's public storage, not from Hugging Face.
  # It goes into the same cache, where TIKTOKEN_ENCODINGS_BASE points the engine (deploy/gpu_compose.py).
  FETCH='
import os, sys, urllib.request
from huggingface_hub import snapshot_download
# A marker written after a complete download lets a rerun skip models that are already cached without asking the Hub.
marker = "/root/.cache/huggingface/quickstart-complete/" + sys.argv[1].replace("/", "--")
if not os.path.exists(marker):
    snapshot_download(sys.argv[1], ignore_patterns=["original/*"])
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").close()
if "gpt-oss" in sys.argv[1]:
    folder = "/root/.cache/huggingface/tiktoken"
    os.makedirs(folder, exist_ok=True)
    for name in ("o200k_base.tiktoken", "cl100k_base.tiktoken"):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/encodings/" + name, path + ".part")
            os.replace(path + ".part", path)
'
  for model in $(for a in "${ASSIGN[@]}"; do [[ "$a" == --assign ]] || echo "${a%%=*}"; done | sort -u); do
    service="vllm-gpu-$(echo "$model" | tr '[:upper:]' '[:lower:]' | sed 's#/#-#g; s/[^a-z0-9-]/-/g; s/^-*//; s/-*$//')"
    downloaded=""
    for attempt in 1 2 3 4 5 6; do
      echo "Downloading ${model} into the model cache (attempt ${attempt} of 6) ..."
      if HF_TOKEN="${HF_TOKEN:-}" docker compose -f "$ROOT/deploy/gpu/compose.yaml" run --rm --no-deps -T \
           -e HF_HUB_OFFLINE=0 --entrypoint python3 "$service" -c \
           "$FETCH" \
           "$model"; then
        downloaded=1; break
      fi
      sleep 30
    done
    [[ -n "$downloaded" ]] || die "Could not download ${model} after 6 attempts. Check the network, then run docker compose up again to resume."
    ok "${model} is in the model cache"
  done
  HF_TOKEN="${HF_TOKEN:-}" docker compose -f "$ROOT/deploy/gpu/compose.yaml" up -d
  # The first engine of a role replaces what the role pointed at before (the CPU model, after an install without
  # free GPUs); the role's other engines are replicas next to it.
  declare -A ROLE_SEEN=()
  for line in "${GPU_PLACEMENTS[@]}"; do
    IFS='|' read -r model _engine _device _gpus container <<<"$line"
    PLACEMENT_ARGS+=(--model "$line")
    python3 "$ROOT/deploy/litellm_models.py" --base "$LITELLM_URL" add --keep-others --name "$model" --upstream "openai/${model}" --api-base "http://${container}:8000/v1"
    role="${ROLE_OF_GPU_MODEL[$model]}"
    python3 "$ROOT/deploy/litellm_models.py" --base "$LITELLM_URL" add ${ROLE_SEEN[$role]:+--keep-others} --name "$role" --upstream "openai/${model}" --api-base "http://${container}:8000/v1"
    ROLE_SEEN[$role]=1
    [[ " ${ALIAS_ARGS[*]} " == *" ${role}=${model} "* ]] || ALIAS_ARGS+=(--alias "${role}=${model}")
  done
  ok "GPU engines started; they load the cached weights in the background (2 to 10 minutes)."
fi
# Roles without a GPU engine use the CPU model.
for role in general coder reasoning; do
  [[ " ${ALIAS_ARGS[*]} " == *" ${role}="* ]] && continue
  python3 "$ROOT/deploy/litellm_models.py" --base "$LITELLM_URL" add --name "$role" --upstream "openai/${CPU_MODEL}" --api-base "http://${CPU_CONTAINER}:8000/v1"
  ALIAS_ARGS+=(--alias "${role}=${CPU_MODEL}")
done

# ---------------------------------------------------------------- 5. The runner and the agents
step "Agent runner"
mkdir -p "$ROOT/data"
python3 "$ROOT/deploy/placement.py" --out "$ROOT/data/placement.json" --host "${HOST_NAME:-$(hostname -s)}" --gateway "http://agentic-litellm:4000/v1" "${PLACEMENT_ARGS[@]}" "${ALIAS_ARGS[@]}"
export WAREHOUSE_URL="postgresql://agentuser:${PGVECTOR_PASSWORD}@agentic-postgres:5432/agentdb"
export REDIS_URL="redis://admin:${REDIS_PASSWORD}@agentic-redis:6379/0"
# Secrets come from the environment; the master key stays out of the runner (an earlier install wrote it).
python3 "$ROOT/deploy/configure.py" \
  --set-from-env GATEWAY_KEY --set-from-env WAREHOUSE_URL --set-from-env REDIS_URL --unset LITELLM_MASTER_KEY \
  --set "GATEWAY_URL=http://agentic-litellm:4000/v1" \
  --set "DOCKER_GID=$(stat -c %g /var/run/docker.sock)" --set "HOST_NAME=${HOST_NAME:-$(hostname -s)}"
docker build -q -t dell-agent-quickstart-sandbox:local "$ROOT/deploy/sandbox" >/dev/null \
  || die "The sandbox image did not build. Agents that run code need it; see: docker build deploy/sandbox"
ok "Sandbox image built"
SERVICES=(runner-db api worker power-telemetry)
PROFILES=(--profile runner)
if [[ ${#ASSIGN[@]} -gt 0 ]]; then SERVICES+=(gpu-telemetry); PROFILES+=(--profile gpu); fi
runner "${PROFILES[@]}" up -d --build "${SERVICES[@]}"
PORT="$(env_value "$ROOT/.env" BUILDER_PORT)"
for _ in $(seq 1 60); do curl -fsS "http://127.0.0.1:${PORT:-8766}/api/health" >/dev/null 2>&1 && break; sleep 3; done
curl -fsS "http://127.0.0.1:${PORT:-8766}/api/health" >/dev/null 2>&1 || die "The runner is not answering on port ${PORT:-8766}; see: docker compose logs api"
ok "Runner is up"
runner exec -T api python -m backend.bootstrap --gateway http://agentic-litellm:4000/v1 --key-env GATEWAY_KEY \
  --model general --role coder=coder --role reasoning=reasoning --install-catalog
runner exec -T api python -m backend.seed_warehouse | tail -3

step "Done"
echo "${BOLD}Demo UI:${NC} http://127.0.0.1:${PORT:-8766} on this server (loopback only)."
echo "From your laptop: ssh -L ${PORT:-8766}:127.0.0.1:${PORT:-8766} ${HOST_USER:-$(whoami)}@${HOST_NAME:-$(hostname -s)}  then open http://127.0.0.1:${PORT:-8766}"
warn "The toolkit publishes LiteLLM (4000), PostgreSQL (5432), Redis (6379), Langfuse (3002), the MinIO console (9001) and its CPU engines (2080 and up) on every network interface. Firewall them if this server is reachable from networks you do not trust (SECURITY.md)."
