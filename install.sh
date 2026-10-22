#!/usr/bin/env bash
# Prepare a bare Ubuntu server (22.04, 24.04 or 26.04, x86_64) and start the installation.
#
#   ./install.sh            # install Docker and, with NVIDIA GPUs, the driver and container runtime; then docker compose up
#   ./install.sh --yes      # the same, answering yes to every question
#   ./install.sh --no-gpu   # ignore GPUs even if present
#   ./install.sh --status   # show what is running and the URL
#
# On a server that already has Docker (and, for GPUs, the NVIDIA Container Toolkit) you do not need this script:
# run `docker compose up` in this directory. The installation itself runs in the `installer` container
# (deploy/setup.sh). This script only does what a container cannot: install Docker and the NVIDIA runtime.
# It never enters credentials for you and never stops a process it did not start.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_ARGS=("$@")
USE_GPU=auto; STATUS_ONLY=0; ASSUME_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gpu) USE_GPU=no ;;
    --status) STATUS_ONLY=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; NC=$'\033[0m'
step() { echo; echo "${BOLD}== $*${NC}"; }
ok() { echo "${GREEN}✔${NC} $*"; }
warn() { echo "${YELLOW}⚠${NC} $*"; }
die() { echo "${RED}✘${NC} $*" >&2; exit 1; }
# Questions are read from the terminal, so they also work when the script arrives through a pipe
# (deploy/bootstrap.sh); without a terminal the answer is no.
confirm() {
  [[ $ASSUME_YES -eq 1 ]] && return 0
  local reply
  { read -r -p "$1 [y/N] " reply < /dev/tty; } 2>/dev/null || return 1
  [[ "$reply" =~ ^[Yy] ]]
}
have() { command -v "$1" >/dev/null 2>&1; }

if [[ $STATUS_ONLY -eq 1 ]]; then
  docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'agentic|vllm|quickstart|NAMES' || true
  port="$(grep -E '^BUILDER_PORT=' "$ROOT/.env" 2>/dev/null | cut -d= -f2)"
  echo; echo "${BOLD}Demo UI:${NC} http://127.0.0.1:${port:-8766} (loopback; from a laptop: ssh -L ${port:-8766}:127.0.0.1:${port:-8766} $(whoami)@$(hostname -s))"
  exit 0
fi

# ---------------------------------------------------------------- 1. OS and Docker
step "Checking the host"
[[ "$(id -u)" -ne 0 ]] || die "Run as a normal user with sudo, not as root."
[[ "$(uname -m)" == "x86_64" ]] || die "x86_64 only."
if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  case "${ID}-${VERSION_ID}" in
    ubuntu-22.04|ubuntu-24.04|ubuntu-26.04) ok "Ubuntu ${VERSION_ID}" ;;
    ubuntu-*) warn "Ubuntu ${VERSION_ID} is untested; 22.04, 24.04 and 26.04 are supported. Continuing." ;;
    *) warn "${PRETTY_NAME:-unknown OS} is untested; this script targets Ubuntu." ;;
  esac
fi
MISSING=()
have curl || MISSING+=(curl ca-certificates); have gpg || MISSING+=(gnupg); have lsb_release || MISSING+=(lsb-release)
have lspci || MISSING+=(pciutils)
if [[ ${#MISSING[@]} -gt 0 ]]; then
  step "Installing basics: ${MISSING[*]}"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING[@]}"
fi
if ! have docker; then
  step "Installing Docker Engine"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl gnupg lsb-release
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker
  # A fresh daemon has nothing to move yet: put its data root on the largest filesystem when / is small,
  # because images and model weights (100 to 300 GB) go there.
  ROOT_FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  BIG_MOUNT="$(df -BG --output=avail,target -x tmpfs -x devtmpfs -x squashfs -x overlay 2>/dev/null | tail -n +2 | sort -rn | head -1)"
  BIG_FREE_GB="$(echo "$BIG_MOUNT" | awk '{print $1}' | tr -dc '0-9')"; BIG_TARGET="$(echo "$BIG_MOUNT" | awk '{print $2}')"
  if [[ "${ROOT_FREE_GB:-0}" -lt 150 && -n "$BIG_TARGET" && "$BIG_TARGET" != "/" && "${BIG_FREE_GB:-0}" -gt "${ROOT_FREE_GB:-0}" ]]; then
    if confirm "/ has ${ROOT_FREE_GB} GB free; use ${BIG_TARGET}/docker (${BIG_FREE_GB} GB free) as Docker's data root?"; then
      sudo mkdir -p "${BIG_TARGET}/docker"
      python3 - "$BIG_TARGET/docker" <<'PY' | sudo tee /etc/docker/daemon.json >/dev/null
import json, os, sys
path = "/etc/docker/daemon.json"
conf = json.load(open(path)) if os.path.isfile(path) else {}
conf["data-root"] = sys.argv[1]
print(json.dumps(conf, indent=2))
PY
      sudo systemctl restart docker
      ok "Docker data root: ${BIG_TARGET}/docker"
    fi
  fi
  # Behind a proxy, the daemon (pulls) needs it too; the shell's variables are not enough.
  if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}${HTTP_PROXY:-${http_proxy:-}}" ]]; then
    sudo mkdir -p /etc/systemd/system/docker.service.d
    printf '[Service]\nEnvironment="HTTP_PROXY=%s"\nEnvironment="HTTPS_PROXY=%s"\nEnvironment="NO_PROXY=%s"\n' \
      "${HTTP_PROXY:-${http_proxy:-}}" "${HTTPS_PROXY:-${https_proxy:-}}" "${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}" \
      | sudo tee /etc/systemd/system/docker.service.d/proxy.conf >/dev/null
    sudo systemctl daemon-reload && sudo systemctl restart docker
    ok "Docker daemon proxy configured from your environment"
  fi
  sudo usermod -aG docker "$USER"
  if ! docker info >/dev/null 2>&1; then
    # Continue in this session with the new group instead of asking for a logout.
    ok "Added $USER to the docker group; continuing with that group active."
    exec sg docker -c "cd '$ROOT' && ./install.sh ${ORIGINAL_ARGS[*]:-}"
  fi
fi
docker info >/dev/null 2>&1 || die "Docker is installed but not usable by $USER. Run 'sudo usermod -aG docker $USER', log in again, and rerun."
# env_file entries marked optional need Compose 2.24 or newer.
COMPOSE_VERSION="$(docker compose version --short 2>/dev/null | sed 's/^v//')"
if [[ -z "$COMPOSE_VERSION" ]] || ! printf '2.24.0\n%s\n' "$COMPOSE_VERSION" | sort -V -C; then
  step "Installing the Docker Compose plugin"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-compose-plugin
  COMPOSE_VERSION="$(docker compose version --short 2>/dev/null | sed 's/^v//')"
  printf '2.24.0\n%s\n' "$COMPOSE_VERSION" | sort -V -C || die "Docker Compose 2.24 or newer is required (found ${COMPOSE_VERSION:-none})."
fi
ok "Docker $(docker version --format '{{.Server.Version}}') with Compose ${COMPOSE_VERSION}"

# ---------------------------------------------------------------- 2. NVIDIA driver and container runtime
step "NVIDIA GPUs"
# lspci (pciutils) is not on every server image; a working nvidia-smi is proof enough.
if [[ "$USE_GPU" != "no" ]] && { have nvidia-smi || lspci 2>/dev/null | grep -qi 'nvidia'; }; then
  if ! have nvidia-smi; then
    echo "An NVIDIA GPU is present but no driver is installed."
    if confirm "Install the recommended NVIDIA driver with ubuntu-drivers (a reboot follows)?"; then
      sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ubuntu-drivers-common
      sudo ubuntu-drivers install --gpgpu || sudo ubuntu-drivers install
      warn "Driver installed. Reboot now, then rerun ./install.sh to continue."
      exit 0
    fi
    warn "Continuing without GPUs."
  else
    ok "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | sed 's/^ *//' | paste -sd ';' -) · driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    if ! have nvidia-ctk || ! docker info 2>/dev/null | grep -qi 'nvidia'; then
      step "Installing the NVIDIA Container Toolkit"
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
      curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
      sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit
      sudo nvidia-ctk runtime configure --runtime=docker
      # Docker loads the runtime when it restarts, and a restart stops every running container.
      running="$(docker ps -q | wc -l | tr -d ' ')"
      if [[ "$running" -gt 0 ]]; then
        warn "The NVIDIA runtime is installed, but Docker must restart to load it, which stops the ${running} running containers."
        warn "When that is acceptable: sudo systemctl restart docker, then run ./install.sh again."
        exit 0
      fi
      sudo systemctl restart docker
      ok "NVIDIA container runtime configured"
    fi
  fi
else
  ok "No NVIDIA GPU used; CPU inference only"
fi

# ---------------------------------------------------------------- 3. The installation
step "Starting the installation (docker compose up)"
cd "$ROOT"
USE_GPU="$USE_GPU" docker compose up --build
# `docker compose up` exits 0 even when the installer fails; exit with the installer's own status.
exit "$(docker inspect -f '{{.State.ExitCode}}' "$(docker compose ps -aq installer)")"
