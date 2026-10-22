"""Code execution in a throwaway Docker container per call.

Docker mode (the default) creates a container from
SANDBOX_IMAGE with networking disabled, all capabilities dropped, a CPU and
memory limit and a PID limit, uploads the work files as a tar archive, runs one
command, collects stdout/stderr and removes the container. This is the
single-host pattern; the toolkit's own sandbox is Kubernetes-only.

SANDBOX_MODE=process explicitly enables local execution for trusted development
and tests. Resource limits do not isolate it from host files or the network.

Every result carries a `sandbox` record (kind, image, container id, limits,
wall time, exit code, CPU seconds) that the runtime attaches to the tool event,
so the UI can show where and how long the code ran. The sandbox's CPU is not
the caller's: a container is read from its own cgroup (`cpu.stat`) as the
command ends, a process from the resource usage of its children.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

import httpx

DEFAULT_IMAGE = "dell-agent-quickstart-sandbox:local"
OUTPUT_LIMIT = 24000
CPU_FILE = "/tmp/.sandbox-cpu"
# The command runs as "$@" and its exit status is kept; the container's own cgroup (v2, or v1's cpuacct)
# is copied to a file the caller reads from the stopped container. Nothing is written to stdout or stderr.
CGROUP_WRAPPER = (
    '"$@"; rc=$?; { cat /sys/fs/cgroup/cpu.stat || cat /sys/fs/cgroup/cpuacct/cpuacct.usage; } > '
    + CPU_FILE
    + " 2>/dev/null; exit $rc"
)
# Process mode: the same command as a child of a small launcher that writes its children's CPU time.
RUSAGE_WRAPPER = (
    "import resource, subprocess, sys\n"
    "code = subprocess.call(sys.argv[2:])\n"
    "usage = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
    "open(sys.argv[1], 'w').write(repr(usage.ru_utime + usage.ru_stime))\n"
    "sys.exit(code)\n"
)


class SandboxUnavailable(RuntimeError):
    pass


def mode() -> str:
    configured = os.getenv("SANDBOX_MODE", "docker")
    if configured in ("docker", "process"):
        return configured
    raise SandboxUnavailable("SANDBOX_MODE must be docker or process")


def docker_socket() -> str:
    return os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")


def image() -> str:
    return os.getenv("SANDBOX_IMAGE", DEFAULT_IMAGE)


def _tar(files: dict[str, bytes], root="work") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        directory = tarfile.TarInfo(root)
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o777
        archive.addfile(directory)
        seen = set()
        for name, content in files.items():
            parts = Path(name).parts
            for depth in range(1, len(parts)):
                sub = root + "/" + "/".join(parts[:depth])
                if sub not in seen:
                    seen.add(sub)
                    info = tarfile.TarInfo(sub)
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o777
                    archive.addfile(info)
            info = tarfile.TarInfo(root + "/" + name)
            info.size = len(content)
            info.mode = 0o666
            info.mtime = int(time.time())
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _demux(stream: bytes) -> tuple[str, str]:
    """Split Docker's multiplexed log stream into stdout and stderr."""
    out, err = [], []
    view = memoryview(stream)
    position = 0
    while position + 8 <= len(view):
        kind = view[position]
        size = int.from_bytes(view[position + 4 : position + 8], "big")
        chunk = bytes(view[position + 8 : position + 8 + size])
        (err if kind == 2 else out).append(chunk)
        position += 8 + size
    if position == 0 and stream:  # a TTY stream is not multiplexed
        out.append(bytes(stream))
    return (
        b"".join(out).decode("utf-8", "replace"),
        b"".join(err).decode("utf-8", "replace"),
    )


def parse_cpu(raw: str) -> float | None:
    """CPU seconds from a cgroup v2 cpu.stat (usage_usec) or a cgroup v1 cpuacct.usage (nanoseconds)."""
    for line in raw.splitlines():
        key, _, value = line.partition(" ")
        if key == "usage_usec" and value.strip().isdigit():
            return int(value) / 1e6
    raw = raw.strip()
    return int(raw) / 1e9 if raw.isdigit() else None


def _read_cpu_file(client, cid) -> float | None:
    try:
        response = client.get(f"/containers/{cid}/archive", params={"path": CPU_FILE})
        if response.status_code != 200:
            return None
        with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
            member = archive.next()
            handle = archive.extractfile(member) if member else None
            return parse_cpu(handle.read().decode()) if handle else None
    except (httpx.HTTPError, tarfile.TarError, ValueError):
        return None


def container_cpu_seconds(name: str) -> float | None:
    """A running container's total CPU seconds from the Docker API, or None (no socket, no such container)."""
    try:
        if mode() != "docker" or not name:
            return None
    except SandboxUnavailable:
        return None
    try:
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=docker_socket()),
            base_url="http://docker",
            timeout=5.0,
        ) as client:
            params = {"stream": "false", "one-shot": "true"}
            response = client.get(f"/containers/{name}/stats", params=params)
            if response.status_code == 404:
                # A Compose service name (runner-db) rather than a container name: find it by its label.
                listed = client.get(
                    "/containers/json",
                    params={
                        "filters": json.dumps(
                            {"label": [f"com.docker.compose.service={name}"]}
                        )
                    },
                )
                matches = listed.json() if listed.status_code == 200 else []
                if len(matches) != 1:
                    return None
                response = client.get(
                    f"/containers/{matches[0]['Id']}/stats", params=params
                )
            if response.status_code != 200:
                return None
            usage = (
                response.json()
                .get("cpu_stats", {})
                .get("cpu_usage", {})
                .get("total_usage")
            )
            return usage / 1e9 if usage else None
    except (httpx.HTTPError, ValueError):
        return None


def _clip(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    return (
        text[: OUTPUT_LIMIT // 2]
        + f"\n... [{len(text) - OUTPUT_LIMIT} characters omitted] ...\n"
        + text[-OUTPUT_LIMIT // 2 :]
    )


def run_docker(
    command: list[str],
    files: dict[str, bytes],
    timeout: int,
    cpus: float | None,
    memory_mb: int,
    workdir: str,
    pids_limit: int = 256,
) -> dict:
    started = time.perf_counter()
    chosen = image()
    with httpx.Client(
        transport=httpx.HTTPTransport(uds=docker_socket()),
        base_url="http://docker",
        timeout=httpx.Timeout(timeout + 15, connect=5.0),
    ) as client:
        spec = {
            "Image": chosen,
            "Cmd": ["sh", "-c", CGROUP_WRAPPER, "sandbox", *command],
            "WorkingDir": workdir,
            "Env": ["PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1", "HOME=/tmp"],
            "NetworkDisabled": True,
            "HostConfig": {
                "NetworkMode": "none",
                # No CPU limit (cpus=None) lets a pool sized to the machine use every core.
                "NanoCpus": int(cpus * 1e9) if cpus else 0,
                "Memory": memory_mb * 1024 * 1024,
                "MemorySwap": memory_mb * 1024 * 1024,
                "PidsLimit": pids_limit,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "AutoRemove": False,
                "LogConfig": {
                    "Type": "local",
                    "Config": {"max-size": "1m", "max-file": "1", "compress": "false"},
                },
            },
            "Labels": {"dell-agent-quickstart.sandbox": "1"},
        }
        response = client.post(
            "/containers/create",
            params={"name": "sandbox-" + uuid.uuid4().hex[:12]},
            json=spec,
        )
        if response.status_code >= 300:
            raise SandboxUnavailable(
                f"Docker could not create the sandbox container (HTTP {response.status_code})"
            )
        cid = response.json()["Id"]
        timed_out = False
        try:
            archive = client.put(
                f"/containers/{cid}/archive",
                params={"path": str(Path(workdir).parent)},
                content=_tar(files, Path(workdir).name),
                headers={"Content-Type": "application/x-tar"},
            )
            archive.raise_for_status()
            client.post(f"/containers/{cid}/start").raise_for_status()
            try:
                waited = client.post(
                    f"/containers/{cid}/wait",
                    params={"condition": "not-running"},
                    timeout=httpx.Timeout(timeout + 2, connect=5.0),
                )
                waited.raise_for_status()
                exit_code = waited.json().get("StatusCode", -1)
            except httpx.TimeoutException:
                timed_out = True
                client.post(f"/containers/{cid}/kill")
                exit_code = -9
            logs = client.get(
                f"/containers/{cid}/logs", params={"stdout": "true", "stderr": "true"}
            )
            logs.raise_for_status()
            stdout, stderr = _demux(logs.content)
            cpu_seconds = None if timed_out else _read_cpu_file(client, cid)
        finally:
            try:
                client.delete(
                    f"/containers/{cid}", params={"force": "true", "v": "true"}
                )
            except httpx.HTTPError:
                pass
    return {
        "stdout": _clip(stdout),
        "stderr": _clip(stderr),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "sandbox": {
            "kind": "docker",
            "image": chosen,
            "container_id": cid[:12],
            "network": "none",
            "cpu_limit": cpus or "none (every core)",
            "memory_limit_mb": memory_mb,
            "pids_limit": pids_limit,
            "capabilities": "all dropped",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "timeout_seconds": timeout,
            "cpu_seconds": None if cpu_seconds is None else round(cpu_seconds, 4),
            "cpu_source": "container cgroup (cpu.stat)"
            if cpu_seconds is not None
            else "not measured (timed out or no cgroup file)",
        },
    }


def run_process(
    command: list[str],
    files: dict[str, bytes],
    timeout: int,
    cpus: float | None,
    memory_mb: int,
    workdir: str,
) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sandbox-") as tmp:
        root = Path(tmp) / Path(workdir).name
        root.mkdir()
        for name, content in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        argv = [
            sys.executable if command[0] in ("python", "python3") else command[0],
            *command[1:],
        ]
        if command[0] in ("python", "python3") and "-I" not in argv:
            argv.insert(1, "-I")
        if shutil.which(argv[0]) is None and not Path(argv[0]).exists():
            raise SandboxUnavailable(
                f"{command[0]} is not available for the process sandbox"
            )
        cpu_file = Path(tmp) / ".sandbox-cpu"
        argv = [sys.executable, "-I", "-c", RUSAGE_WRAPPER, str(cpu_file), *argv]

        def limits():
            try:
                import resource

                budget = int(timeout * (cpus or os.cpu_count() or 1))
                resource.setrlimit(resource.RLIMIT_CPU, (budget + 5, budget + 10))
                if sys.platform.startswith("linux"):
                    size = memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (size, size))
            except (ImportError, ValueError, OSError):
                pass

        timed_out = False
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmp,
            "TMPDIR": tmp,
            "PYTHONDONTWRITEBYTECODE": "1",
            "LANG": "C.UTF-8",
        }
        # Its own session, so a timeout kills the command together with the launcher.
        process = subprocess.Popen(
            argv,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=limits if os.name == "posix" else None,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                process.kill()
            stdout, stderr = process.communicate()
            exit_code = -9
        try:
            cpu_seconds = float(cpu_file.read_text())
        except (OSError, ValueError):
            cpu_seconds = None
    return {
        "stdout": _clip(stdout.decode("utf-8", "replace")),
        "stderr": _clip(stderr.decode("utf-8", "replace")),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "sandbox": {
            "kind": "process",
            "image": None,
            "container_id": None,
            "network": "host (not isolated)",
            "cpu_limit": cpus or "none (every core)",
            "memory_limit_mb": memory_mb if sys.platform.startswith("linux") else None,
            "pids_limit": None,
            "capabilities": "inherited",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "timeout_seconds": timeout,
            "cpu_seconds": None if cpu_seconds is None else round(cpu_seconds, 4),
            "cpu_source": "child process resource usage"
            if cpu_seconds is not None
            else "not measured (timed out)",
        },
    }


def run(
    command: list[str],
    files: dict[str, bytes] | None = None,
    *,
    timeout: int = 60,
    cpus: float | None = 2.0,
    memory_mb: int = 2048,
    workdir: str = "/work",
    pids_limit: int = 256,
) -> dict:
    files = files or {}
    for name in files:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Sandbox files must use relative paths without traversal")
    if mode() == "docker":
        try:
            return run_docker(
                command, files, timeout, cpus, memory_mb, workdir, pids_limit
            )
        except (httpx.HTTPError, SandboxUnavailable, ValueError, KeyError) as exc:
            raise SandboxUnavailable(
                f"Docker sandbox failed: {type(exc).__name__}"
            ) from None
    return run_process(command, files, timeout, cpus, memory_mb, workdir)


def run_python(code: str, files: dict[str, bytes] | None = None, **limits) -> dict:
    files = dict(files or {})
    files["main.py"] = code.encode()
    return run(["python3", "main.py"], files, **limits)
