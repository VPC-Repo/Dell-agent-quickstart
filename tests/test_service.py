import briefing_fixture
import pytest
from fastapi.testclient import TestClient

from backend import service
from backend.main import app
from backend.worker import Worker


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "app.db"))
    monkeypatch.setenv("CHECKPOINT_URL", str(tmp_path / "cp.db"))
    monkeypatch.setenv("EMBEDDED_WORKER", "0")
    monkeypatch.setenv("BUILDER_API_TOKEN", "test-token")
    with TestClient(app, headers={"Authorization": "Bearer test-token"}) as c:
        briefing_fixture.install()
        yield c


def test_health_reports_database_and_worker_liveness(client):
    first = client.get("/api/health").json()
    assert first["status"] == "ok" and first["database"] is True
    assert first["worker_seconds_since_tick"] is None
    worker = Worker(1)
    worker.heartbeat()
    worker.pool.shutdown()
    age = client.get("/api/health").json()["worker_seconds_since_tick"]
    assert age is not None and age < 5


def test_missing_token_serves_only_local_clients(monkeypatch):
    from fastapi import HTTPException

    from backend.main import authorize

    class Req:
        def __init__(self, host):
            self.client = (
                type("Client", (), {"host": host})() if host is not None else None
            )

    monkeypatch.delenv("BUILDER_API_TOKEN", raising=False)
    monkeypatch.delenv("BUILDER_AUTH", raising=False)
    authorize(Req("127.0.0.1"), "")
    with pytest.raises(HTTPException):
        authorize(Req(None), "")
    with pytest.raises(HTTPException):
        authorize(Req("203.0.113.9"), "")
    monkeypatch.setenv("BUILDER_API_TOKEN", "secret")
    authorize(Req("203.0.113.9"), "Bearer secret")
    with pytest.raises(HTTPException):
        authorize(Req("203.0.113.9"), "Bearer wrong")
    with pytest.raises(HTTPException):
        authorize(Req("127.0.0.1"), "")


def test_auth_can_be_turned_off_explicitly(monkeypatch):
    from backend.main import authorize

    class Req:
        client = type("Client", (), {"host": "172.18.0.1"})()

    monkeypatch.setenv("BUILDER_API_TOKEN", "secret")
    monkeypatch.setenv("BUILDER_AUTH", "off")
    authorize(Req(), "")
    authorize(Req(), "Bearer wrong")
    monkeypatch.setenv("BUILDER_AUTH", "on")
    with pytest.raises(Exception):
        authorize(Req(), "")


def test_health_reports_no_login_when_auth_is_off(client, monkeypatch):
    monkeypatch.setenv("BUILDER_AUTH", "off")
    assert client.get("/api/health").json()["auth_required"] is False


def enqueue_seed(query):
    """Queue a run of the seeded daily-briefing agent (mock provider), the way an agent's Run button does."""
    from backend.deployments import enqueue

    return enqueue(
        service.snapshot_for_ref("daily-briefing@1"), query, "daily-briefing@1"
    )


def test_worker_runs_a_queued_run_and_the_api_reports_and_cancels_it(client):
    run_id = enqueue_seed("Prepare daily briefing")["id"]
    worker = Worker(1)
    worker.tick()
    for f in worker.futures:
        f.result(timeout=60)
    worker.pool.shutdown()
    result = client.get("/api/runs/" + run_id).json()
    assert result["status"] == "completed", result
    assert result["mode"] == "mock" and "Daily briefing" in result["output"]
    assert result["server_time"] >= result["finished"]
    assert any(r["id"] == run_id for r in client.get("/api/runs?limit=5").json())
    other = enqueue_seed("Cancel me")["id"]
    assert client.post("/api/runs/" + other + "/cancel").json()["status"] == "canceled"
    assert (
        client.post("/api/runs/" + other + "/cancel").status_code == 422
    )  # already finished


def test_token_mode_without_a_token_denies_even_loopback(client, monkeypatch):
    monkeypatch.setenv("BUILDER_AUTH", "token")
    monkeypatch.delenv("BUILDER_API_TOKEN", raising=False)
    assert client.get("/api/health").json()["auth_required"] is True
    response = client.get("/api/catalog")
    assert response.status_code == 503
    assert "requires BUILDER_API_TOKEN" in response.json()["detail"]
