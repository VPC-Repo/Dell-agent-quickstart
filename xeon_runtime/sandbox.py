"""Code execution in a throwaway Docker container per call.

Docker mode (the default) creates a container from
SANDBOX_IMAGE with networking disabled, all capabilities dropped, a CPU and
memory limit and a PID limit, uploads the work files as a tar archive, runs one
command, collects stdout/stderr and removes the container. This is the
single-host pattern; the toolkit's own sandbox is Kubernetes-only.

SANDBOX_MODE=process explicitly enables local execution for trusted development
and tests. Resource limits do not isolate it from host files or the network.

Every result carries a `sandbox` record (kind, image, container id, limits,
wall time, exit code) that the runtime attaches to the tool event, so the UI
can show where and how long the code ran.
"""

from __future__ import annotations

import io
import os
import shutil
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
    cpus: float,
    memory_mb: int,
    workdir: str,
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
            "Cmd": command,
            "WorkingDir": workdir,
            "Env": ["PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1", "HOME=/tmp"],
            "NetworkDisabled": True,
            "HostConfig": {
                "NetworkMode": "none",
                "NanoCpus": int(cpus * 1e9),
                "Memory": memory_mb * 1024 * 1024,
                "MemorySwap": memory_mb * 1024 * 1024,
                "PidsLimit": 256,
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
            "cpu_limit": cpus,
            "memory_limit_mb": memory_mb,
            "pids_limit": 256,
            "capabilities": "all dropped",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "timeout_seconds": timeout,
        },
    }


def run_process(
    command: list[str],
    files: dict[str, bytes],
    timeout: int,
    cpus: float,
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

        def limits():
            try:
                import resource

                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (int(timeout * cpus) + 5, int(timeout * cpus) + 10),
                )
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
        try:
            completed = subprocess.run(
                argv,
                cwd=root,
                env=env,
                capture_output=True,
                timeout=timeout,
                preexec_fn=limits if os.name == "posix" else None,
            )
            stdout, stderr, exit_code = (
                completed.stdout,
                completed.stderr,
                completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout, stderr, exit_code = exc.stdout or b"", exc.stderr or b"", -9
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
            "cpu_limit": cpus,
            "memory_limit_mb": memory_mb if sys.platform.startswith("linux") else None,
            "pids_limit": None,
            "capabilities": "inherited",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "timeout_seconds": timeout,
        },
    }


def run(
    command: list[str],
    files: dict[str, bytes] | None = None,
    *,
    timeout: int = 60,
    cpus: float = 2.0,
    memory_mb: int = 2048,
    workdir: str = "/work",
) -> dict:
    files = files or {}
    for name in files:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Sandbox files must use relative paths without traversal")
    if mode() == "docker":
        try:
            return run_docker(command, files, timeout, cpus, memory_mb, workdir)
        except (httpx.HTTPError, SandboxUnavailable, ValueError, KeyError) as exc:
            raise SandboxUnavailable(
                f"Docker sandbox failed: {type(exc).__name__}"
            ) from None
    return run_process(command, files, timeout, cpus, memory_mb, workdir)


def run_python(code: str, files: dict[str, bytes] | None = None, **limits) -> dict:
    files = dict(files or {})
    files["main.py"] = code.encode()
    return run(["python3", "main.py"], files, **limits)
