"""Where an agent's CPU goes: the sandbox's CPU, fixed-size jobs and the containers a run uses."""

import json

import pytest

from backend import demo_catalog, worker
from xeon_runtime import analysis_jobs, sandbox
from xeon_runtime.demo_tools import (
    REGISTRY as DEMO_TOOLS,
    logs_job,
    measured,
    transactions_job,
)
from xeon_runtime.tool_catalog import SANDBOX_TOOLS, TOOL_INFO


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FIXTURE_DIR", str(tmp_path / "fixtures"))
    monkeypatch.setenv("SANDBOX_MODE", "process")
    return tmp_path


def test_process_sandbox_reports_the_commands_cpu_not_the_callers(local):
    busy = sandbox.run_python(
        "s = 0\nfor i in range(3_000_000):\n    s += i\nprint(s)", timeout=30
    )
    assert busy["exit_code"] == 0 and busy["stdout"].strip() == str(
        sum(range(3_000_000))
    )
    assert (
        busy["sandbox"]["cpu_seconds"] > 0.05
        and busy["sandbox"]["cpu_source"] == "child process resource usage"
    )
    failing = sandbox.run(["python3", "-c", "import sys; print('x'); sys.exit(3)"], {})
    assert failing["exit_code"] == 3 and failing["stdout"] == "x\n"
    stuck = sandbox.run_python(
        "import time\nwhile True:\n    time.sleep(0.1)", timeout=1
    )
    assert stuck["timed_out"] and stuck["sandbox"]["cpu_seconds"] is None


def test_cgroup_cpu_files_parse_for_v2_and_v1():
    assert (
        sandbox.parse_cpu("usage_usec 2500000\nuser_usec 2000000\nsystem_usec 500000\n")
        == 2.5
    )
    assert sandbox.parse_cpu("1500000000\n") == 1.5
    assert sandbox.parse_cpu("") is None


def test_analysis_job_counts_the_sandbox_cpu(local, monkeypatch):
    script = (
        analysis_jobs.PRELUDE
        + "print(json.dumps({'cpus': CPUS, 'total': sum(range(2_000_000))}))"
    )
    monkeypatch.setitem(
        analysis_jobs.JOBS, "simulate_credit_risk", (script, 32, 60, "a small job")
    )
    result = json.loads(analysis_jobs.REGISTRY["simulate_credit_risk"].invoke({}))
    evidence = result.pop("work_evidence")
    # no CPU limit: the job sees every CPU the runner may use
    assert result == {"cpus": analysis_jobs.host_cpus(), "total": sum(range(2_000_000))}
    assert evidence["units"]["cpu_limit"] == "none (every core)"
    assert (
        evidence["sandbox_cpu_seconds"] > 0
        and evidence["cpu_seconds"] >= evidence["sandbox_cpu_seconds"]
    )


def test_tools_are_catalogued_and_jobs_say_where_their_cpu_ran():
    for name in [*analysis_jobs.REGISTRY, *DEMO_TOOLS]:
        assert name in TOOL_INFO
    assert set(analysis_jobs.REGISTRY) <= SANDBOX_TOOLS
    one = json.loads(
        measured("reconcile_transactions", lambda: transactions_job(2000, 1))
    )["work_evidence"]
    assert one["in_run_process"] is True
    two = json.loads(measured("analyze_service_day", lambda: logs_job(4000, 2)))[
        "work_evidence"
    ]
    assert two["in_run_process"] is False
    # fixed chunks on any pool size give the same findings
    assert (
        logs_job(8000, 8, workers=2)[0]["services"]
        == logs_job(8000, 8, workers=4)[0]["services"]
    )
    assert logs_job(8000, 8, workers=2)[1]["processes"] == 2
    assert {t for a in demo_catalog.AGENTS for t in a["tools"]} <= set(TOOL_INFO)


def test_a_run_accounts_the_containers_it_uses(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "WAREHOUSE_URL", "postgresql://u:p@agentic-postgres:5432/agentdb"
    )
    monkeypatch.setenv("REDIS_URL", "redis://u:p@agentic-redis:6379/0")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///data/builder.db")
    registry = tmp_path / "placement.json"
    registry.write_text(json.dumps({"models": {"m": {"container": "vllm-gpu-m"}}}))
    monkeypatch.setenv("PLACEMENT_REGISTRY", str(registry))
    snapshot = {
        "providers": {
            "p": {"endpoint": "http://agentic-litellm:4000/v1"},
            "local": {"endpoint": "http://127.0.0.1:9/v1"},
        }
    }
    assert worker.service_containers(snapshot) == {
        "agentic-postgres": "warehouse",
        "agentic-redis": "cache",
        "agentic-litellm": "model gateway",
        "vllm-gpu-m": "model engine",
    }
    # without a Docker socket nothing is read, and a run carries on
    monkeypatch.setenv("SANDBOX_MODE", "process")
    assert worker.service_cpu(worker.service_containers(snapshot)) == {}
