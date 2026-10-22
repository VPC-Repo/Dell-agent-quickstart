"""Registered tools: specs, executors, the registry, pinning and recovery."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from backend import store
from backend.main import app
from xeon_runtime.specs import Blueprint


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "app.db"))
    monkeypatch.setenv("CHECKPOINT_URL", str(tmp_path / "cp.db"))
    monkeypatch.setenv("EMBEDDED_WORKER", "0")
    monkeypatch.setenv("BUILDER_API_TOKEN", "test-token")
    with TestClient(app, headers={"Authorization": "Bearer test-token"}) as c:
        yield c


def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def spec_body(**over):
    base = dict(
        id="echo_api",
        label="Echo API",
        description="POST arguments to the echo endpoint and return the response.",
        capability="external_data",
        kind="http_api",
        endpoint="http://127.0.0.1:9",  # unreachable placeholder for pure-API tests
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )
    return {**base, **over}


def test_artifact_workspace_is_shared_within_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    from xeon_runtime.artifacts import list_artifacts, read_artifact, save_artifact

    first = {"configurable": {"thread_id": "run1:taskA"}}
    second = {"configurable": {"thread_id": "run1:taskB"}}
    other_run = {"configurable": {"thread_id": "run2"}}
    save_artifact.invoke({"name": "notes.md", "content": "hello"}, config=first)
    data = json.loads(read_artifact.invoke({"name": "notes.md"}, config=second))
    assert data["content"] == "hello"
    names = [x["name"] for x in json.loads(list_artifacts.invoke({}, config=second))]
    assert names == ["notes.md"]
    assert json.loads(list_artifacts.invoke({}, config=other_run)) == []
    with pytest.raises(ValueError):
        save_artifact.invoke({"name": "../escape", "content": "x"}, config=first)
    with pytest.raises(ValueError):
        read_artifact.invoke({"name": "missing.txt"}, config=first)


def mcp_server():
    class Mcp(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("id") is None:  # notification
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            method = body["method"]
            if method == "initialize":
                result = {"protocolVersion": "2025-03-26", "capabilities": {}}
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "add",
                            "description": "Add two numbers.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "number"},
                                    "b": {"type": "number"},
                                },
                                "required": ["a", "b"],
                            },
                            "annotations": {"readOnlyHint": True},
                        }
                    ]
                }
            elif method == "tools/call":
                arguments = body["params"]["arguments"]
                result = {
                    "content": [
                        {"type": "text", "text": str(arguments["a"] + arguments["b"])}
                    ]
                }
            else:
                result = {}
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": body["id"], "result": result}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if method == "initialize":
                self.send_header("mcp-session-id", "session-1")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    return serve(Mcp)


def test_side_effecting_tools_disable_automatic_recovery(client, monkeypatch):
    from backend import worker as worker_module

    assert worker_module.retry_safe({"tools": {}})
    assert worker_module.retry_safe(
        {"tools": {"x": {"side_effects": True, "idempotent": True}}}
    )
    assert not worker_module.retry_safe(
        {"tools": {"x": {"side_effects": True, "idempotent": False}}}
    )
    monkeypatch.setattr(worker_module, "supervise_run", lambda *a, **k: None)
    with store.Session.begin() as db:
        for id, tools in (
            ("r-effects", {"x": {"side_effects": True, "idempotent": False}}),
            ("r-safe", {"x": {"side_effects": True, "idempotent": True}}),
        ):
            db.add(
                store.Run(
                    id=id,
                    status="running",
                    lease_until=time.time() - 10,
                    snapshot={
                        "execution": {"policy": {"max_attempts": 3}},
                        "tools": tools,
                    },
                    query="q",
                    blueprint_ref="draft",
                )
            )
    w = worker_module.Worker(1)
    w.tick()
    w.pool.shutdown()
    with store.Session() as db:
        effects = db.get(store.Run, "r-effects")
        safe = db.get(store.Run, "r-safe")
        assert effects.status == "failed"
        assert "side-effecting" in effects.error
        assert safe.status in ("queued", "running")


def test_conversation_memory_survives_blueprint_edits(tmp_path):
    from briefing_fixture import seed_agents
    from xeon_runtime.memory import MemoryStore

    memory = MemoryStore("sqlite:///" + str(tmp_path / "m.db"))
    bp = Blueprint.model_validate(seed_agents()[0])
    with memory.conversation(bp, "user-1", 5) as turns:
        turns.append({"query": "q", "output": "o", "run_id": "r1"})
    bp.nodes[0].position = {"x": 999.0, "y": 5.0}
    with memory.conversation(bp, "user-1", 5) as turns:
        assert turns and turns[0]["output"] == "o"
