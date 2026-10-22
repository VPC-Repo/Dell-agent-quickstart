"""Host, GPU, container and inference-engine load for the system panel.

One background sampler in the API process keeps a recent snapshot; `/api/system`
returns it. Sources, each optional and reported as such:

- CPU per socket and memory: psutil plus the Linux CPU topology in sysfs. Inside
  a container these are host-wide figures, which is what the panel wants.
- GPUs: `nvidia-smi` when it is on the PATH, otherwise a CSV written by the
  `gpu-telemetry` sidecar (compose profile `gpu`) at GPU_TELEMETRY_FILE.
- Containers: the Docker Engine API over the socket at DOCKER_SOCKET, one
  `stats` sample per container, read in parallel.
- Engines: the vLLM Prometheus metrics of every engine in the placement
  registry (requests running and waiting, KV-cache use, token rates).

Everything here is best-effort display data; a failing source shows as
unavailable and never affects runs.
"""

from __future__ import annotations

import csv
import io
import logging
import glob
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psutil

from xeon_runtime import placement

log = logging.getLogger(__name__)

GPU_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
]
ENGINE_METRICS = {
    "vllm:num_requests_running": "requests_running",
    "vllm:num_requests_waiting": "requests_waiting",
    "vllm:kv_cache_usage_perc": "kv_cache",
    "vllm:gpu_cache_usage_perc": "kv_cache",
    "vllm:cpu_cache_usage_perc": "kv_cache",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:request_success_total": "requests_total",
}
CONTAINER_ROLES = [
    (r"vllm", "engine"),
    (r"litellm|gateway", "gateway"),
    (r"postgres", "database"),
    (r"redis", "cache"),
    (r"langfuse|clickhouse|minio", "observability"),
    (r"flowise", "builder"),
    (r"sandbox", "sandbox"),
    (r"-api-|_api_|-api$|_api$|/api", "runner"),
    (r"worker", "runner"),
]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if hasattr(os, "uname") and os.uname().sysname == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def socket_map() -> dict[int, list[int]]:
    """Logical CPU ids per physical socket, from sysfs; one socket elsewhere."""
    sockets: dict[int, list[int]] = {}
    root = Path("/sys/devices/system/cpu")
    for path in sorted(root.glob("cpu[0-9]*/topology/physical_package_id")):
        try:
            cpu = int(path.parts[-3][3:])
            sockets.setdefault(int(path.read_text().strip()), []).append(cpu)
        except (OSError, ValueError):
            continue
    if not sockets:
        sockets = {0: list(range(psutil.cpu_count(logical=True) or 1))}
    return sockets


def core_map() -> list[list[int]]:
    """Logical CPUs grouped by physical core (SMT siblings), from sysfs.

    Utilization is judged per core: when one sibling thread is busy the core is
    busy, whatever its twin shows, so a core's utilization is the higher of its
    siblings. Without sysfs every logical CPU is its own core."""
    groups: dict[str, list[int]] = {}
    root = Path("/sys/devices/system/cpu")
    for path in sorted(root.glob("cpu[0-9]*/topology/thread_siblings_list")):
        try:
            cpu = int(path.parts[-3][3:])
            key = path.read_text().strip()
            groups.setdefault(key, []).append(cpu)
        except (OSError, ValueError):
            continue
    if not groups:
        return [[c] for c in range(psutil.cpu_count(logical=True) or 1)]
    return sorted(groups.values(), key=lambda g: g[0])


def core_utilization(per_thread: list[float], cores: list[list[int]]) -> float:
    """Mean over physical cores of the busier sibling thread."""
    values = [
        max((per_thread[c] for c in group if c < len(per_thread)), default=0.0)
        for group in cores
    ]
    values = [
        v for v, group in zip(values, cores) if any(c < len(per_thread) for c in group)
    ]
    return round(sum(values) / len(values), 1) if values else 0.0


def parse_gpu_csv(text: str) -> list[dict]:
    rows = []
    for record in csv.reader(io.StringIO(text.strip())):
        if len(record) < len(GPU_FIELDS):
            continue
        values = [v.strip() for v in record]

        def number(v):
            try:
                return float(v)
            except ValueError:
                return None

        rows.append(
            {
                "index": int(values[0]) if values[0].isdigit() else values[0],
                "name": values[1],
                "utilization_percent": number(values[2]),
                "memory_used_mb": number(values[3]),
                "memory_total_mb": number(values[4]),
                "power_w": number(values[5]),
                "temperature_c": number(values[6]),
            }
        )
    return rows


def read_gpus() -> tuple[list[dict], str]:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(GPU_FIELDS),
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if out.returncode == 0:
                gpus = parse_gpu_csv(out.stdout)
                apps = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=4,
                )
                for gpu in gpus:
                    gpu["processes"] = None
                if apps.returncode == 0:
                    # Per-GPU process counts need the UUID map; keep the total.
                    total = len([l for l in apps.stdout.splitlines() if l.strip()])
                    for gpu in gpus:
                        gpu["processes_host_total"] = total
                return gpus, "nvidia-smi"
        except (OSError, subprocess.SubprocessError):
            pass
    path = Path(os.getenv("GPU_TELEMETRY_FILE", "/data/telemetry/gpu.csv"))
    try:
        if time.time() - path.stat().st_mtime < 15:
            return parse_gpu_csv(path.read_text()), "telemetry-file"
        return [], "stale-telemetry-file"
    except OSError:
        return [], "unavailable"


def container_role(name: str, image: str) -> str:
    text = f"{name} {image}".lower()
    for pattern, role in CONTAINER_ROLES:
        if re.search(pattern, text):
            return role
    return "other"


class DockerStats:
    """Container list and one-shot stats over the Docker Engine socket."""

    def __init__(self, socket_path=None):
        self.socket_path = socket_path or os.getenv(
            "DOCKER_SOCKET", "/var/run/docker.sock"
        )
        self.pool = ThreadPoolExecutor(max_workers=12)

    def available(self):
        return Path(self.socket_path).exists()

    def client(self):
        return httpx.Client(
            transport=httpx.HTTPTransport(uds=self.socket_path),
            base_url="http://docker",
            timeout=httpx.Timeout(6.0, connect=2.0),
        )

    def sample(self) -> list[dict]:
        with self.client() as client:
            listing = client.get("/containers/json").json()

            def one(item):
                cid = item["Id"]
                name = (item.get("Names") or ["/" + cid[:12]])[0].lstrip("/")
                record = {
                    "id": cid[:12],
                    "name": name,
                    "image": item.get("Image", ""),
                    "state": item.get("State", ""),
                    "status": item.get("Status", ""),
                    "role": container_role(name, item.get("Image", "")),
                    "cpu_percent": None,
                    "memory_bytes": None,
                    "memory_limit_bytes": None,
                }
                try:
                    stats = client.get(
                        f"/containers/{cid}/stats", params={"stream": "false"}
                    ).json()
                    cpu = stats.get("cpu_stats", {})
                    pre = stats.get("precpu_stats", {})
                    cpu_delta = cpu.get("cpu_usage", {}).get(
                        "total_usage", 0
                    ) - pre.get("cpu_usage", {}).get("total_usage", 0)
                    system_delta = cpu.get("system_cpu_usage", 0) - pre.get(
                        "system_cpu_usage", 0
                    )
                    online = (
                        cpu.get("online_cpus")
                        or len(cpu.get("cpu_usage", {}).get("percpu_usage") or [])
                        or 1
                    )
                    if system_delta > 0 and cpu_delta >= 0:
                        record["cpu_percent"] = round(
                            cpu_delta / system_delta * online * 100, 1
                        )
                    memory = stats.get("memory_stats", {})
                    usage = memory.get("usage")
                    if usage is not None:
                        cache = memory.get("stats", {}).get("inactive_file", 0)
                        record["memory_bytes"] = max(0, usage - cache)
                        record["memory_limit_bytes"] = memory.get("limit")
                except (httpx.HTTPError, ValueError, KeyError):
                    pass
                return record

            return sorted(
                self.pool.map(one, listing),
                key=lambda r: (-(r["cpu_percent"] or 0), r["name"]),
            )


def parse_prometheus(text: str) -> list[tuple[str, dict, float]]:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if not match:
            continue
        labels = {}
        if match.group(2):
            for pair in re.finditer(r'(\w+)="((?:[^"\\]|\\.)*)"', match.group(2)):
                labels[pair.group(1)] = pair.group(2)
        try:
            out.append((match.group(1), labels, float(match.group(3))))
        except ValueError:
            continue
    return out


class SystemSampler:
    def __init__(self, interval=2.0, container_interval=4.0):
        self.interval = interval
        self.container_interval = container_interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.docker = DockerStats()
        self.sockets = socket_map()
        self.cores = core_map()
        self.cpu_model = cpu_model()
        self.hostname = os.getenv("HOST_NAME") or socket.gethostname()
        self.snapshot_data: dict = {"generated": None, "ready": False}
        self.containers: list[dict] = []
        self.containers_at = 0.0
        self.containers_source = "unavailable"
        self.engine_last: dict[str, tuple[float, float, float]] = {}
        self.engine_pool = ThreadPoolExecutor(max_workers=8)
        self.threads: list[threading.Thread] = []
        psutil.cpu_percent(percpu=True)

    # -- sources -------------------------------------------------------------
    def sample_cpu(self):
        per_thread = psutil.cpu_percent(percpu=True)
        sockets = []
        for sid, cpus in sorted(self.sockets.items()):
            members = set(cpus)
            groups = [g for g in self.cores if members & set(g)]
            sockets.append(
                {
                    "id": sid,
                    "threads": len(cpus),
                    "physical_cores": len(groups),
                    "percent": core_utilization(per_thread, groups),
                }
            )
        try:
            load = list(os.getloadavg())
        except (OSError, AttributeError):
            load = []
        return {
            # SMT-aware: one number for the whole machine (see core_utilization).
            "percent": core_utilization(per_thread, self.cores),
            "thread_percent": round(sum(per_thread) / len(per_thread), 1)
            if per_thread
            else 0.0,
            "per_core": [round(v) for v in per_thread],
            "physical_cores": len(self.cores),
            "sockets": sockets,
            "load": [round(v, 2) for v in load],
        }

    def sample_power(self):
        """CPU package power in watts from Intel RAPL (/sys/class/powercap), or None when unreadable.

        Best effort: containers usually lack the powercap tree; the UI greys the Power tab out then."""
        total_uj = 0.0
        found = False
        for domain in glob.glob("/sys/class/powercap/intel-rapl:*"):
            try:
                if not Path(domain, "name").read_text().strip().startswith("package"):
                    continue
                total_uj += float(Path(domain, "energy_uj").read_text().strip())
                found = True
            except (OSError, ValueError):
                continue
        if not found:
            return self._power_from_sidecar()
        now = time.monotonic()
        last = getattr(self, "_rapl_last", None)
        self._rapl_last = (now, total_uj)
        if not last or now - last[0] <= 0 or total_uj < last[1]:
            return None  # first sample or counter wrap
        return round((total_uj - last[1]) / 1e6 / (now - last[0]), 1)

    @staticmethod
    def _power_from_sidecar():
        """Watts published by the `power-telemetry` sidecar (root, reads RAPL), if fresh."""
        path = Path(os.getenv("POWER_TELEMETRY_FILE", "/data/telemetry/power.csv"))
        try:
            if time.time() - path.stat().st_mtime > 15:
                return None
            stamp, watts = path.read_text().strip().split(",")
            return round(float(watts), 1)
        except (OSError, ValueError):
            return None

    def sample_memory(self):
        vm = psutil.virtual_memory()
        return {
            "total_bytes": vm.total,
            "used_bytes": vm.total - vm.available,
            "available_bytes": vm.available,
            "percent": vm.percent,
        }

    def sample_engines(self):
        engines = placement.engines()
        now = time.monotonic()

        def one(engine):
            url = engine.get("metrics_url") or (
                re.sub(r"/v1/?$", "", engine["api_base"]) + "/metrics"
                if engine.get("api_base")
                else ""
            )
            record = {
                "model": engine.get("model"),
                "engine": engine.get("engine"),
                "engine_label": placement.ENGINE_LABELS.get(
                    engine.get("engine", ""), engine.get("engine")
                ),
                "host": engine.get("host"),
                "device": engine.get("device"),
                "device_detail": engine.get("device_detail", ""),
                "gpus": engine.get("gpus", []),
                "container": engine.get("container", ""),
                "reachable": None,
                "requests_running": None,
                "requests_waiting": None,
                "kv_cache_percent": None,
                "prompt_tokens_per_s": None,
                "generation_tokens_per_s": None,
            }
            if not url:
                return record
            try:
                with httpx.Client(timeout=httpx.Timeout(3.0, connect=1.5)) as client:
                    response = client.get(url)
                record["reachable"] = response.status_code == 200
                if response.status_code != 200:
                    return record
            except httpx.HTTPError:
                record["reachable"] = False
                return record
            totals = {"prompt_tokens_total": 0.0, "generation_tokens_total": 0.0}
            for name, labels, value in parse_prometheus(response.text):
                key = ENGINE_METRICS.get(name)
                if not key:
                    continue
                if key in totals:
                    totals[key] += value
                elif key == "kv_cache":
                    record["kv_cache_percent"] = round(value * 100, 1)
                elif key in ("requests_running", "requests_waiting"):
                    record[key] = (record[key] or 0) + value
            previous = self.engine_last.get(url)
            if previous:
                dt = now - previous[0]
                if dt > 0:
                    record["prompt_tokens_per_s"] = round(
                        max(0.0, totals["prompt_tokens_total"] - previous[1]) / dt, 1
                    )
                    record["generation_tokens_per_s"] = round(
                        max(0.0, totals["generation_tokens_total"] - previous[2]) / dt,
                        1,
                    )
            self.engine_last[url] = (
                now,
                totals["prompt_tokens_total"],
                totals["generation_tokens_total"],
            )
            return record

        if not engines:
            return []
        return list(self.engine_pool.map(one, engines))

    def sample_runs(self):
        try:
            from sqlalchemy import func, select

            from . import store

            with store.Session() as db:
                counts = dict(
                    db.execute(
                        select(store.Run.status, func.count()).group_by(
                            store.Run.status
                        )
                    ).all()
                )
            return {
                "running": counts.get("running", 0),
                "queued": counts.get("queued", 0),
                "waiting": counts.get("waiting", 0),
            }
        except Exception:
            return {"running": None, "queued": None, "waiting": None}

    # -- loops ----------------------------------------------------------------
    def container_loop(self):
        while not self.stop.is_set():
            if self.docker.available():
                try:
                    sample = self.docker.sample()
                    with self.lock:
                        self.containers, self.containers_at = sample, time.time()
                        self.containers_source = "docker-socket"
                except Exception as exc:  # display data; keep going
                    with self.lock:
                        self.containers_source = "error: " + type(exc).__name__
            else:
                with self.lock:
                    self.containers_source = "unavailable"
            self.stop.wait(self.container_interval)

    def sample(self) -> dict:
        gpus, gpu_source = read_gpus()
        with self.lock:
            containers, containers_at, containers_source = (
                list(self.containers),
                self.containers_at,
                self.containers_source,
            )
        return {
            "generated": time.time(),
            "ready": True,
            "interval_seconds": self.interval,
            "host": {
                "name": self.hostname,
                "cpu_model": self.cpu_model,
                "logical_cpus": psutil.cpu_count(logical=True),
                "physical_cores": len(self.cores),
                "smt": (psutil.cpu_count(logical=True) or 1) > len(self.cores),
                "sockets": len(self.sockets),
                "boot_time": psutil.boot_time(),
            },
            "cpu": self.sample_cpu(),
            "memory": self.sample_memory(),
            "gpus": gpus,
            "containers": containers,
            "engines": self.sample_engines(),
            "runs": self.sample_runs(),
            "sources": {
                "gpu": gpu_source,
                "containers": containers_source,
                "containers_age_seconds": round(time.time() - containers_at, 1)
                if containers_at
                else None,
                "engines": "placement-registry"
                if placement.engines()
                else "no-registry",
            },
        }

    def loop(self):
        while not self.stop.is_set():
            try:
                data = self.sample()
                with self.lock:
                    self.snapshot_data = data
            except Exception:
                log.exception("System sample failed; retrying")
            self.stop.wait(self.interval)

    def start(self):
        for target in (self.loop, self.container_loop):
            thread = threading.Thread(target=target, daemon=True, name=target.__name__)
            thread.start()
            self.threads.append(thread)
        return self

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.snapshot_data)


SAMPLER: SystemSampler | None = None


def current() -> dict:
    """Latest snapshot; samples once inline until the background loop has run."""
    global SAMPLER
    if SAMPLER is None:
        SAMPLER = SystemSampler()
    data = SAMPLER.snapshot()
    if not data.get("ready"):
        data = SAMPLER.sample()
        with SAMPLER.lock:
            SAMPLER.snapshot_data = data
    return data


def start():
    global SAMPLER
    if SAMPLER is None:
        SAMPLER = SystemSampler()
    if not SAMPLER.threads:
        SAMPLER.start()
    return SAMPLER


def close():
    if SAMPLER is not None:
        SAMPLER.close()
