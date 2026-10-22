import json

import pytest
from fastapi.testclient import TestClient

from backend import automations, demo_catalog, store
from backend.main import app
from xeon_runtime.demo_tools import (
    logs_job,
    measured,
    transactions_job,
)
from xeon_runtime.engine import tools_for
from xeon_runtime.planning import WorkerDefinition, worker_blueprint
from xeon_runtime.readiness import assess_output, inspect_goal
from xeon_runtime.specs import Blueprint, validate_blueprint


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "demo.db"))
    monkeypatch.setenv("EMBEDDED_WORKER", "0")
    monkeypatch.setenv("TOOL_CONNECTION_DIR", str(tmp_path / "connections"))
    monkeypatch.setenv("BUILDER_API_TOKEN", "demo-test")
    with TestClient(app, headers={"Authorization": "Bearer demo-test"}) as client:
        yield client


def test_processing_is_reproducible_and_finds_injected_exceptions():
    finance, _, _ = transactions_job(2000)
    assert finance["exceptions"] == 3
    assert finance["unsettled_usd"] == "75.00"
    assert finance == transactions_job(2000)[0]
    logs, _, _ = logs_job(4000)
    assert logs["services"][0]["error_percent"] == 18
    assert all(row["errors"] == 0 for row in logs["services"][1:])
    evidence = json.loads(measured("analyze_service_logs", lambda: logs_job(4000)))[
        "work_evidence"
    ]
    assert evidence["cpu_seconds"] >= 0 and evidence["units"]["events"] == 4000
    # one shard runs in the run process; its CPU is already counted there
    assert evidence["in_run_process"] is True


def test_output_checks_ignore_case():
    assert (
        assess_output(
            "completed", "Fleet-API spike on WH-470", [], ["fleet-api", "WH-470"], []
        )["status"]
        == "checks_passed"
    )
    # Typographic hyphens and quotes (gpt-oss writes "fleet‑api" with U+2011) still satisfy the check.
    assert (
        assess_output(
            "completed",
            "The fleet‑api service degraded; runbook “fleet‑api 429 storm”",
            [],
            ["fleet-api", '"fleet-api 429 storm"'],
            [],
        )["status"]
        == "checks_passed"
    )


def test_words_do_not_substitute_for_tool_completion():
    assert (
        assess_output(
            "completed", "FIN-470", [], ["FIN-470"], ["reconcile_transactions"]
        )["status"]
        == "incomplete"
    )
    start = {
        "kind": "tool_started",
        "payload": {"name": "reconcile_transactions", "request_id": "one"},
    }
    assert (
        assess_output(
            "completed", "FIN-470", [start], ["FIN-470"], ["reconcile_transactions"]
        )["status"]
        == "incomplete"
    )
    finish = {"kind": "tool_completed", "payload": {"request_id": "one"}}
    assert (
        assess_output(
            "completed",
            "FIN-470",
            [start, finish],
            ["FIN-470"],
            ["reconcile_transactions"],
        )["status"]
        == "checks_passed"
    )


def test_catalog_installs_the_eight_agents_idempotently(client):
    with store.Session.begin() as db:
        provider = db.get(store.Document, "provider:demo")
        provider.payload = {
            **provider.payload,
            "adapter": "openai",
            "endpoint": "http://localhost:9999/v1",
            "model": "test",
        }
    ids = demo_catalog.install("demo", archive_legacy=True)
    # Mechanism, not a pinned gallery size: one live agent per catalog entry.
    assert len(ids) == len(set(ids)) == len(demo_catalog.AGENTS)
    agents = client.get("/api/automations").json()
    assert {a["id"] for a in agents} == set(ids)
    from backend.planning import catalog as worker_catalog

    with store.Session() as db:
        workers = [w for w in worker_catalog(db) if w["ref"].startswith("demo-")]
    assert len(workers) == len(demo_catalog.WORKERS)
    from backend.tools import registered

    with store.Session() as db:
        extra = registered(db)
    for w in workers:
        definition = WorkerDefinition.model_validate(
            {k: v for k, v in w.items() if k not in ["ref", "contract"]}
        )
        bp = Blueprint.model_validate(
            worker_blueprint({**definition.model_dump(), "ref": w["ref"]}, "demo")
        )
        assert validate_blueprint(bp, tools=extra) == []
        assert {t.name for t in tools_for(bp, extra)} == set(w["tools"])
    # Staged fan-out: six panelists start together, the three reviewers wait for all of them, the writer for the reviewers.
    panel = next(a for a in agents if a["demo"]["slug"] == "gpu-panel")
    by_name = {t["name"]: t for t in panel["agents"]}
    panelists = [t["id"] for t in panel["agents"] if t["name"].endswith("panelist")]
    assert len(panelists) == 6 and all(
        by_name[n]["depends_on"] == [] for n in by_name if n.endswith("panelist")
    )
    assert set(by_name["Risk reviewer"]["depends_on"]) == set(panelists)
    assert set(by_name["Decision brief writer"]["depends_on"]) == {
        by_name[n]["id"] for n in by_name if n.endswith("reviewer")
    }
    for a in agents:
        assert a["readiness"]["status"] == "ready"
        assert a["editable"]
        with store.Session() as db:
            snapshot = automations.draft_snapshot(
                db, automations.metadata(db, a["id"]).payload
            )
            assert set(snapshot["required_tools"]) == set(
                a["demo"].get("required_tools") or a["demo"]["tools"]
            )
            assert len(snapshot["coordination"]["workers"]) == len(
                a["coordination"]["worker_refs"]
            )
    assert set(demo_catalog.install("demo")) == set(ids)
    assert (
        inspect_goal("Analyze the fictional event collection", ["demo_logs"])["status"]
        == "ready"
    )


def test_cpu_jobs_shard_across_processes(monkeypatch):
    """Sharded runs keep the seeded patterns and report every process; one shard equals the old single-core job."""
    from xeon_runtime.demo_tools import _ranges, shard_count

    assert _ranges(4000, 4) == [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
    assert all((b - a) % 40 == 0 for a, b in _ranges(100000, 7)[:-1])
    finance, units, cpu = transactions_job(4000, shards=2)
    assert finance["shards"] == 2 and units["processes"] == 2 and cpu >= 0
    assert finance["exceptions"] == 5 and finance["unsettled_usd"] == "125.00"
    logs, _, _ = logs_job(8000, shards=2)
    assert logs["services"][0]["error_percent"] == 18 and all(
        r["errors"] == 0 for r in logs["services"][1:]
    )
    monkeypatch.setenv("CPU_JOB_SHARDS", "3")
    assert shard_count() == 3


def _connect_demo_provider():
    with store.Session.begin() as db:
        provider = db.get(store.Document, "provider:demo")
        provider.payload = {
            **provider.payload,
            "adapter": "openai",
            "endpoint": "http://localhost:9999/v1",
            "model": "test",
        }


def test_agents_are_defined_in_the_catalog_and_not_created_over_the_api(client):
    assert client.post("/api/automations", json={}).status_code == 405
    assert sorted(a["slug"] for a in demo_catalog.AGENTS) == sorted(
        demo_catalog.FEATURED
    )


def test_catalog_upgrade_preserves_user_records_and_archives_only_old_demo(client):
    user = automations.create()
    with store.Session.begin() as db:
        provider = db.get(store.Document, "provider:demo")
        provider.payload = {
            **provider.payload,
            "adapter": "openai",
            "endpoint": "http://localhost:9999/v1",
            "model": "test",
        }
        old = automations.metadata(db, user["id"]).payload
        old_demo = {
            **old,
            "id": "old-demo",
            "blueprint_id": "old-demo",
            "bindings": {},
            "demo": {"version": "old", "slug": "old"},
            "schedule_id": "demo-schedule",
        }
        automations.put(db, "automation:old-demo", "automation", old_demo)
        automations.put(db, "draft:old-demo", "draft", {"id": "old-demo"})
        automations.put(db, "worker:custom@1", "worker", {"ref": "custom@1"})
        automations.put(db, "worker:demo-old@1", "worker", {"ref": "demo-old@1"})
        automations.put(db, "published:custom@1", "published", {"id": "custom"})
        automations.put(db, "published:old-demo@1", "published", {"id": "old-demo"})
        db.add(store.Schedule(id="user-schedule", paused=0))
        db.add(store.Schedule(id="demo-schedule", paused=0))
    ids = demo_catalog.install("demo", archive_legacy=True)
    with store.Session() as db:
        assert not db.get(store.Document, "automation:" + user["id"]).payload.get(
            "archived"
        )
        assert db.get(store.Document, "draft:" + user["blueprint_id"]).kind == "draft"
        assert not db.get(store.Document, "worker:custom@1").payload.get("archived")
        assert db.get(store.Schedule, "user-schedule").paused == 0
        assert db.get(store.Document, "automation:old-demo").payload["archived"]
        assert db.get(store.Document, "draft:old-demo").kind == "archived_draft"
        assert db.get(store.Document, "worker:demo-old@1").payload["archived"]
        assert db.get(store.Schedule, "demo-schedule").paused == 1
        assert db.get(store.Document, "catalog:demo").payload["hidden_published"] == [
            "published:old-demo@1"
        ]
    assert set(demo_catalog.install("demo", archive_legacy=True)) == set(ids)


def test_catalog_rebinds_to_the_selected_provider_without_duplicate_agents(client):
    from backend.bootstrap import save_provider

    save_provider("first", gateway="http://localhost:9998/v1")
    save_provider("second", gateway="http://localhost:9999/v1")
    ids = demo_catalog.install("first")
    assert set(demo_catalog.install("second")) == set(ids)
    for item in client.get("/api/automations").json():
        assert item["provider_id"] == "second"
