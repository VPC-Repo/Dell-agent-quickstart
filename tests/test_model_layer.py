"""Roles pick a model per worker; placement says where each call ran."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from sqlalchemy import select
from fastapi.testclient import TestClient

from backend import automations, bootstrap, demo_catalog, store
from backend.main import app
from xeon_runtime import placement
from xeon_runtime.engine import run_agent
from xeon_runtime.planning import WorkerDefinition, worker_blueprint
from xeon_runtime.providers import resolve_role
from xeon_runtime.specs import Blueprint, Provider


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "app.db"))
    monkeypatch.setenv("CHECKPOINT_URL", str(tmp_path / "cp.db"))
    monkeypatch.setenv("EMBEDDED_WORKER", "0")
    monkeypatch.setenv("TOOL_CONNECTION_DIR", str(tmp_path / "connections"))
    monkeypatch.setenv("BUILDER_API_TOKEN", "test-token")
    monkeypatch.setenv("PLACEMENT_REGISTRY", str(tmp_path / "placement.json"))
    with TestClient(app, headers={"Authorization": "Bearer test-token"}) as c:
        yield c


def test_roles_resolve_to_models_and_fall_back_to_the_default():
    p = Provider(
        id="gw",
        name="Gateway",
        endpoint="http://gateway:4000/v1",
        model="general",
        roles={"coder": "coder", "reasoning": "reasoning"},
    )
    assert resolve_role(p, "coder").model == "coder"
    assert resolve_role(p, "reasoning").model == "reasoning"
    assert resolve_role(p, "general").model == "general"
    assert resolve_role(p, None) is p
    # A single-model endpoint serves every role.
    single = Provider(id="one", name="One", endpoint="http://x/v1", model="only")
    assert resolve_role(single, "coder").model == "only"
    with pytest.raises(ValueError):
        Provider(
            id="bad",
            name="Bad",
            endpoint="http://x/v1",
            model="m",
            roles={"Coder": "x"},
        )
    with pytest.raises(ValueError):
        Provider(
            id="bad",
            name="Bad",
            endpoint="http://x/v1",
            model="m",
            roles={"coder": " "},
        )


def test_worker_blueprints_carry_the_role_and_validate():
    w = WorkerDefinition(name="Fixer", instructions="Fix it", role="coder").model_dump()
    bp = Blueprint.model_validate(worker_blueprint({**w, "ref": "fixer@1"}, "gw"))
    model = next(n for n in bp.nodes if n.kind == "model")
    assert model.config == {"provider_id": "gw", "role": "coder"}
    plain = Blueprint.model_validate(
        worker_blueprint(
            {**WorkerDefinition(name="W", instructions="x").model_dump(), "ref": "w@1"},
            "gw",
        )
    )
    assert next(n for n in plain.nodes if n.kind == "model").config["role"] == "general"


def test_placement_registry_matches_aliases_and_upstreams(tmp_path, monkeypatch):
    path = tmp_path / "placement.json"
    monkeypatch.setenv("PLACEMENT_REGISTRY", str(path))
    assert (
        placement.lookup("anything", endpoint="http://gw:4000/v1")["registered"]
        is False
    )
    path.write_text(
        json.dumps(
            {
                "host": "cpu-host",
                "gateway": "http://gw:4000/v1",
                "models": {
                    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
                        "engine": "vllm-cpu",
                        "device": "cpu",
                        "device_detail": "2 x Xeon 6787P",
                        "api_base": "http://agentic-vllm-qwen:8000/v1",
                        "container": "agentic-vllm-qwen",
                    },
                    "gpt-oss-120b": {
                        "engine": "vllm-cuda",
                        "device": "cuda",
                        "gpus": [4, 5],
                        "api_base": "http://vllm-gpu-oss:8000/v1",
                    },
                },
                "aliases": {"general": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
            }
        )
    )
    by_alias = placement.lookup("general", endpoint="http://gw:4000/v1")
    assert by_alias["engine"] == "vllm-cpu" and by_alias["host"] == "cpu-host"
    assert by_alias["alias"] == "general" and by_alias["gateway"] is True
    by_upstream = placement.lookup("reasoning", upstream="http://vllm-gpu-oss:8000/v1/")
    assert by_upstream["engine"] == "vllm-cuda" and by_upstream["gpus"] == [4, 5]
    # The alias resolves to the first entry, but when the gateway names the upstream engine that wins: with
    # replicas of one model behind an alias, this is what tells the GPUs apart.
    routed = placement.lookup("general", upstream="http://vllm-gpu-oss:8000/v1")
    assert routed["gpus"] == [4, 5] and routed["alias"] == "general"
    unknown = placement.lookup("mystery", upstream="http://elsewhere:9000/v1")
    assert unknown["registered"] is False and unknown["host"] == "elsewhere"
    assert placement.lookup("demo", adapter="mock")["engine"] == "mock"
    assert {e["model"] for e in placement.engines()} == {
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "gpt-oss-120b",
    }
    # Edits are picked up without a restart.
    path.write_text(json.dumps({"models": {}}))
    import os

    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
    assert placement.lookup("general")["registered"] is False


def test_model_events_record_gateway_upstream_and_placement(tmp_path, monkeypatch):
    registry = tmp_path / "placement.json"
    monkeypatch.setenv("PLACEMENT_REGISTRY", str(registry))
    seen = []

    class Gateway(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            payload = json.dumps(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "def add(a, b): return a + b",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 9,
                        "total_tokens": 21,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("x-litellm-call-id", "call-abc")
            self.send_header("x-litellm-model-id", "model-123")
            self.send_header(
                "x-litellm-model-api-base", "http://vllm-gpu-coder:8000/v1"
            )
            self.send_header("x-litellm-timing-llm-api-ms", "812.5")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        registry.write_text(
            json.dumps(
                {
                    "host": "gpu-host",
                    "models": {
                        "Qwen/Qwen3-Coder-30B-A3B-Instruct": {
                            "engine": "vllm-cuda",
                            "device": "cuda",
                            "gpus": [2],
                            "api_base": "http://vllm-gpu-coder:8000/v1",
                        }
                    },
                    "aliases": {"coder": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
                }
            )
        )
        providers = {
            "gw": Provider(
                id="gw",
                name="Gateway",
                endpoint=f"http://127.0.0.1:{server.server_port}/v1",
                model="general",
                roles={"coder": "coder"},
                streaming=False,
            ).model_dump()
        }
        bp = Blueprint.model_validate(
            worker_blueprint(
                {
                    **WorkerDefinition(
                        name="Coder", instructions="Write code", role="coder"
                    ).model_dump(),
                    "ref": "coder@1",
                },
                "gw",
            )
        )
        events = []
        text = run_agent(
            bp, providers, "add two numbers", lambda k, p: events.append((k, p))
        )
        assert "def add" in text
        assert seen[0]["model"] == "coder"  # the role's alias reached the gateway
        request = next(p for k, p in events if k == "model_request")
        assert (
            request["model"] == "coder"
            and request["placement"]["engine"] == "vllm-cuda"
        )
        model = next(p for k, p in events if k == "model")
        assert model["model"] == "coder"
        assert model["served_model"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        assert model["upstream"] == "http://vllm-gpu-coder:8000/v1"
        assert (
            model["gateway_call_id"] == "call-abc"
            and model["gateway_model_id"] == "model-123"
        )
        assert model["gateway_llm_ms"] == 812.5
        assert model["placement"]["engine"] == "vllm-cuda" and model["placement"][
            "gpus"
        ] == [2]
        assert model["placement"]["host"] == "gpu-host" and model["usage_available"]
        assert "modeled" not in model
    finally:
        server.shutdown()


def test_bootstrap_saves_the_gateway_provider_and_installs_the_catalog(client):
    bootstrap.main(
        [
            "--gateway",
            "http://agentic-litellm:4000/v1",
            "--role",
            "coder=coder",
            "--role",
            "reasoning=reasoning",
            "--install-catalog",
        ]
    )
    status = client.get("/api/catalog").json()
    assert status["installed"] == status["available"] == len(demo_catalog.AGENTS)
    assert status["providers"][0]["id"] == "intel-toolkit"
    assert status["providers"][0]["roles"] == {
        "coder": "coder",
        "reasoning": "reasoning",
    }
    # Re-running updates the provider in place and adds no duplicate agents.
    bootstrap.main(["--gateway", "http://agentic-litellm:4000/v1", "--model", "Qwen/X"])
    with store.Session() as db:
        gateway = [
            d
            for d in db.scalars(
                select(store.Document).where(store.Document.kind == "provider")
            )
            if d.payload["id"] == "intel-toolkit"
        ]
    assert len(gateway) == 1 and gateway[0].payload["model"] == "Qwen/X"
    assert client.get("/api/catalog").json()["installed"] == len(demo_catalog.AGENTS)
    # Roles reach the worker blueprints the guided editor assembles.
    agents = client.get("/api/automations").json()
    roles = {t["role"] for a in agents for t in a["agents"]}
    assert {"general", "coder", "reasoning"} >= roles and "general" in roles
    with store.Session() as db:
        for a in agents:
            snapshot = automations.draft_snapshot(
                db, automations.metadata(db, a["id"]).payload
            )
            for agent in snapshot["agents"].values():
                model = next(n for n in agent["nodes"] if n["kind"] == "model")
                assert model["config"]["role"] in ("general", "coder", "reasoning")
    installed = client.post("/api/catalog/install", json={}).json()
    assert installed["provider_id"] == "intel-toolkit" and len(
        installed["installed"]
    ) == len(demo_catalog.AGENTS)


def test_catalog_install_needs_a_connected_provider(client):
    response = client.post("/api/catalog/install", json={})
    assert response.status_code == 422 and "Connect a model service" in response.text
