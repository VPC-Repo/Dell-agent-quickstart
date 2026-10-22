"""The system panel's data: parsers and a snapshot without Docker or GPUs."""

from fastapi.testclient import TestClient

from backend import system
from backend.main import app


def test_gpu_csv_and_prometheus_parsers():
    gpus = system.parse_gpu_csv(
        "0, NVIDIA RTX PRO 6000 Blackwell, 87, 61234, 97887, 412.5, 61\n"
        "1, NVIDIA RTX PRO 6000 Blackwell, 0, 12, 97887, [N/A], 33\n"
    )
    assert gpus[0]["utilization_percent"] == 87 and gpus[0]["memory_total_mb"] == 97887
    assert gpus[1]["power_w"] is None and gpus[1]["index"] == 1
    metrics = system.parse_prometheus(
        "# HELP vllm:num_requests_running x\n"
        'vllm:num_requests_running{model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"} 3.0\n'
        'vllm:kv_cache_usage_perc{model_name="m"} 0.42\n'
        "garbage line\n"
        'vllm:generation_tokens_total{model_name="m"} 12345\n'
    )
    assert metrics[0] == (
        "vllm:num_requests_running",
        {"model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
        3.0,
    )
    assert len(metrics) == 3
    assert (
        system.container_role("agentic-vllm-qwen", "vllm/vllm-openai-cpu") == "engine"
    )
    assert system.container_role("agentic-litellm", "litellm") == "gateway"
    assert system.container_role("quickstart-worker-1", "quickstart:local") == "runner"


def test_core_utilization_takes_the_busier_smt_sibling():
    # Two cores, each with two threads: core 0 = threads 0,2; core 1 = threads 1,3.
    cores = [[0, 2], [1, 3]]
    assert (
        system.core_utilization([100, 0, 0, 0], cores) == 50.0
    )  # one core fully busy on one thread
    assert (
        system.core_utilization([100, 0, 100, 0], cores) == 50.0
    )  # both threads of core 0 busy: still one core
    assert system.core_utilization([50, 50, 50, 50], cores) == 50.0
    assert system.core_utilization([], cores) == 0.0


def test_snapshot_reports_sources_and_per_socket_cpu(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCKER_SOCKET", str(tmp_path / "missing.sock"))
    monkeypatch.setenv("GPU_TELEMETRY_FILE", str(tmp_path / "gpu.csv"))
    monkeypatch.setattr(system.shutil, "which", lambda name: None)
    monkeypatch.setenv("PLACEMENT_REGISTRY", str(tmp_path / "none.json"))
    sampler = system.SystemSampler()
    data = sampler.sample()
    assert (
        data["ready"] and data["cpu"]["sockets"] and data["memory"]["total_bytes"] > 0
    )
    assert sum(s["threads"] for s in data["cpu"]["sockets"]) == len(
        data["cpu"]["per_core"]
    )
    assert (
        sum(s["physical_cores"] for s in data["cpu"]["sockets"])
        == data["host"]["physical_cores"]
    )
    assert data["sources"]["gpu"] == "unavailable" and data["gpus"] == []
    assert data["sources"]["engines"] == "no-registry" and data["engines"] == []
    (tmp_path / "gpu.csv").write_text("0, Test GPU, 12, 100, 1000, 50, 40\n")
    data = sampler.sample()
    assert (
        data["sources"]["gpu"] == "telemetry-file"
        and data["gpus"][0]["name"] == "Test GPU"
    )


def test_system_endpoint_serves_a_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "app.db"))
    monkeypatch.setenv("EMBEDDED_WORKER", "0")
    monkeypatch.setenv("SYSTEM_SAMPLER", "0")
    monkeypatch.setenv("BUILDER_API_TOKEN", "t")
    monkeypatch.setenv("DOCKER_SOCKET", str(tmp_path / "missing.sock"))
    monkeypatch.setattr(system, "SAMPLER", None)
    with TestClient(app, headers={"Authorization": "Bearer t"}) as client:
        data = client.get("/api/system").json()
        assert data["host"]["logical_cpus"] >= 1 and "runs" in data
        assert (
            client.get("/api/system", headers={"Authorization": ""}).status_code == 401
        )


def test_worker_resources_events_carry_sockets_gpus_and_engines(tmp_path, monkeypatch):
    """The live run charts and the post-run summary read these fields."""
    from backend import store, worker

    monkeypatch.setenv("DOCKER_SOCKET", str(tmp_path / "missing.sock"))
    monkeypatch.setenv("GPU_TELEMETRY_FILE", str(tmp_path / "gpu.csv"))
    (tmp_path / "gpu.csv").write_text("0, Test GPU, 55, 2048, 8192, 120, 50\n")
    monkeypatch.setattr(system.shutil, "which", lambda name: None)
    monkeypatch.setenv("PLACEMENT_REGISTRY", str(tmp_path / "none.json"))
    store.initialize("sqlite:///" + str(tmp_path / "w.db"))
    sampler = system.SystemSampler()
    gpus, source = system.read_gpus()
    payload = {
        "sockets": [s["percent"] for s in sampler.sample_cpu()["sockets"]],
        "gpus": gpus,
        "gpu_source": source,
        "engines": sampler.sample_engines(),
    }
    assert (
        payload["sockets"]
        and payload["gpus"][0]["utilization_percent"] == 55
        and payload["engines"] == []
    )
    code = worker.supervise_run.__code__
    assert "sampler" in code.co_varnames + code.co_cellvars
    assert worker.RESOURCE_SAMPLE_SECONDS == 0.5  # short agents need sub-second samples


def test_power_falls_back_to_the_sidecar_file(monkeypatch, tmp_path):
    import time as _time

    monkeypatch.setenv("POWER_TELEMETRY_FILE", str(tmp_path / "power.csv"))
    monkeypatch.setattr(system.glob, "glob", lambda pattern: [])  # no RAPL here
    sampler = system.SystemSampler.__new__(system.SystemSampler)
    assert sampler.sample_power() is None
    (tmp_path / "power.csv").write_text(f"{_time.time():.3f},412.6\n")
    assert sampler.sample_power() == 412.6
    old = _time.time() - 60
    import os as _os

    _os.utime(tmp_path / "power.csv", (old, old))
    assert sampler.sample_power() is None
