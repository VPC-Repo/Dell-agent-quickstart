#!/usr/bin/env bash
# Runs deploy/setup.sh as a non-root user (the toolkit's deployment refuses root) who owns the repository
# checkout, so the files it writes (toolkit/, .env, data/) belong to the host user, and who is in the group of
# the host's Docker socket.
set -euo pipefail
REPO="${REPO_DIR:?REPO_DIR is set by compose.yaml}"
[[ -S /var/run/docker.sock ]] || { echo "The Docker socket is not mounted." >&2; exit 1; }
uid="$(stat -c %u "$REPO")"; gid="$(stat -c %g "$REPO")"; sock_gid="$(stat -c %g /var/run/docker.sock)"
[[ "$uid" -ne 0 ]] || { echo "Clone the repository as a non-root user; the toolkit cannot write to a root-owned checkout." >&2; exit 1; }
getent group "$gid" >/dev/null || groupadd -g "$gid" installer
getent group "$sock_gid" >/dev/null || groupadd -g "$sock_gid" dockerhost
id -u installer >/dev/null 2>&1 || useradd -m -u "$uid" -g "$gid" -s /bin/bash installer
usermod -aG "$(getent group "$sock_gid" | cut -d: -f1)" installer
# Sudo inside this container only, in case the toolkit's script calls it for a step it then skips.
echo "installer ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/installer
cd "$REPO"
exec gosu installer env HOME=/home/installer USER=installer bash "$REPO/deploy/setup.sh"
