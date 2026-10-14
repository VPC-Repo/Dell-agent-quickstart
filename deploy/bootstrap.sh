#!/usr/bin/env bash
# One line from a bare Ubuntu server to a running demo:
#   curl -fsSL https://raw.githubusercontent.com/VPC-Repo/Dell-agent-quickstart/main/deploy/bootstrap.sh | bash
# Installs git and curl (sudo, once) if missing, clones the repository into your home directory and runs
# install.sh, which asks before installing an NVIDIA driver or moving Docker's data root.
# Pass installer options after "bash -s --", for example: ... | bash -s -- --no-gpu, or --yes to accept every question.
# REPO_URL and DEST override the source and the target directory.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/VPC-Repo/Dell-agent-quickstart.git}"
DEST="${DEST:-$HOME/Dell-agent-quickstart}"
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  echo "Installing git and curl (sudo, once)..."
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git curl ca-certificates
fi
if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" pull -q --ff-only \
    || { echo "Could not update $DEST (local changes?). Update or remove it, then run this again." >&2; exit 1; }
else
  git clone -q "$REPO_URL" "$DEST"
fi
cd "$DEST"
exec ./install.sh "$@"
