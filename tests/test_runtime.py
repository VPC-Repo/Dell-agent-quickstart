import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from briefing_fixture import seed_blueprints
from xeon_runtime.engine import (
    checkpoint_store,
    compile_agent,
    compile_workflow,
    run_agent,
)
from xeon_runtime.providers import JsonAPIModel, create_model
from xeon_runtime.specs import Blueprint, Provider, validate_blueprint


@pytest.fixture
def snapshot():
    seeds = seed_blueprints()
    return {
        "blueprint": seeds[-1],
        "agents": {a["id"] + "@1": a for a in seeds[:-1]},
        "providers": {
            "demo": Provider(id="demo", name="Demo", adapter="mock").model_dump()
        },
    }


def test_deep_agent_executes_real_tool(snapshot):
    events = []
    output = run_agent(
        Blueprint.model_validate(snapshot["agents"]["research@1"]),
        snapshot["providers"],
        "Prepare a briefing",
        lambda k, v: events.append((k, v)),
    )
    assert any(k == "tool" and "search_knowledge" in v["tools"] for k, v in events)
    assert "brief-001" in output and "Demonstration data" in output


def test_workflow_join_and_persistent_approval(snapshot, tmp_path):
    bp = Blueprint.model_validate(snapshot["blueprint"])
    # Split research and analysis, join at approval, then write the result.
    bp.nodes.insert(
        -2,
        {
            "id": "approve",
            "kind": "approval",
            "label": "Review findings",
            "position": {"x": 0, "y": 0},
            "config": {},
        },
    )
    data = bp.model_dump(warnings=False)
    data["edges"] = [
        {"id": str(i), "source": a, "target": b}
        for i, (a, b) in enumerate(
            [
                ("start", "research"),
                ("start", "analysis"),
                ("research", "approve"),
                ("analysis", "approve"),
                ("approve", "writer"),
                ("writer", "result"),
            ]
        )
    ]
    bp = Blueprint.model_validate(data)
    events = []
    cfg = {"configurable": {"thread_id": "resume-check"}}
    with checkpoint_store(str(tmp_path / "cp.db")) as cp:
        graph = compile_workflow(
            bp,
            snapshot["agents"],
            snapshot["providers"],
            lambda k, v: events.append((k, v)),
            cp,
        )
        result = graph.invoke({"query": "Briefing", "outputs": {}, "final": ""}, cfg)
        assert result["__interrupt__"]
        completed = [v["node_id"] for k, v in events if k == "node_completed"]
        assert (
            completed.count("research") == 1
            and completed.count("analysis") == 1
            and "writer" not in completed
        )
    # A fresh graph and connection resume the existing checkpoint.
    with checkpoint_store(str(tmp_path / "cp.db")) as cp:
        graph = compile_workflow(
            bp,
            snapshot["agents"],
            snapshot["providers"],
            lambda k, v: events.append((k, v)),
            cp,
        )
        result = graph.invoke(Command(resume=True), cfg)
        assert "Daily briefing" in result["final"]
    assert (
        sum(k == "node_completed" and v["node_id"] == "research" for k, v in events)
        == 1
    )


def test_graph_rejects_cycle_and_dangling_capabilities(snapshot):
    bp = Blueprint.model_validate(snapshot["blueprint"])
    bp.edges.append(type(bp.edges[0])(id="cycle", source="writer", target="research"))
    assert any("cycle" in e for e in validate_blueprint(bp, snapshot["agents"]))
    agent = Blueprint.model_validate(snapshot["agents"]["research@1"])
    agent.edges = []
    assert any("attach" in e for e in validate_blueprint(agent))


def test_provider_switch_and_custom_request_mapping(monkeypatch):
    captured = []
    monkeypatch.setenv("PRIVATE_GATEWAY_TOKEN", "secret-not-for-export")

    def handler(req):
        captured.append(
            (str(req.url), json.loads(req.content), req.headers["x-access"])
        )
        return httpx.Response(200, json={"result": {"answer": "from " + req.url.host}})

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    p = Provider(
        id="api",
        name="custom",
        adapter="http_json",
        endpoint="https://one.example/generate",
        model="chosen-model",
        headers_env={"x-access": "PRIVATE_GATEWAY_TOKEN"},
        supports_tools=False,
        request_template={"deployment": "$model", "conversation": "$messages"},
        response_path="result.answer",
        tool_calls_path="",
    )
    a = create_model(p).invoke([HumanMessage(content="hello")])
    b = create_model(
        p.model_copy(update={"endpoint": "https://two.example/generate"})
    ).invoke([HumanMessage(content="hello")])
    assert a.content == "from one.example" and b.content == "from two.example"
    assert captured[0][1]["conversation"][0]["content"] == "hello"
    assert captured[0][2] == "secret-not-for-export"
    assert "secret-not-for-export" not in p.model_dump_json()


def test_custom_api_tool_response(monkeypatch):
    response = {
        "result": {
            "text": None,
            "calls": [
                {
                    "id": "a",
                    "function": {"name": "lookup", "arguments": '{"query":"x"}'},
                }
            ],
        }
    }
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response)),
            **kwargs,
        ),
    )
    p = Provider(
        id="api",
        name="tools",
        adapter="http_json",
        endpoint="https://api.example/generate",
        response_path="result.text",
        tool_calls_path="result.calls",
    )
    out = JsonAPIModel(profile=p).invoke("lookup")
    assert out.tool_calls[0]["args"] == {"query": "x"}


def test_bad_provider_and_secret_hygiene():
    with pytest.raises(ValueError):
        Provider(
            id="bad", name="bad", endpoint="https://secret:password@example.com/v1"
        )
    with pytest.raises(ValueError):
        Provider(
            id="bad",
            name="bad",
            endpoint="https://example.com",
            api_key_env="literal-secret",
        )


def test_agent_recovery_preserves_input_and_handles_empty_checkpoint(
    snapshot, tmp_path
):
    bp = Blueprint.model_validate(snapshot["agents"]["research@1"])
    cfg = {"configurable": {"thread_id": "recovered-agent"}}
    with checkpoint_store(str(tmp_path / "recovery.db")) as cp:
        first = run_agent(
            bp, snapshot["providers"], "Briefing", lambda *a: None, cfg, cp, resume=True
        )
        second = run_agent(
            bp, snapshot["providers"], "Briefing", lambda *a: None, cfg, cp, resume=True
        )
        state = compile_agent(bp, snapshot["providers"], cp).get_state(cfg)
        assert sum(isinstance(m, HumanMessage) for m in state.values["messages"]) == 1
        assert first == second


def test_conditional_branch_skips_unchosen_agent_and_joins(snapshot, tmp_path):
    data = snapshot["blueprint"]
    data["nodes"].insert(
        1,
        dict(
            id="decision",
            kind="condition",
            label="Choose route",
            config={
                "source": "$input",
                "path": "route",
                "operator": "equals",
                "value": "research",
            },
            position={"x": 0, "y": 0},
        ),
    )
    data["edges"] = [
        dict(id=str(i), source=a, target=b, source_handle=h)
        for i, (a, b, h) in enumerate(
            [
                ("start", "decision", None),
                ("decision", "research", "true"),
                ("decision", "analysis", "false"),
                ("research", "writer", None),
                ("analysis", "writer", None),
                ("writer", "result", None),
            ]
        )
    ]
    events = []
    bp = Blueprint.model_validate(data)
    assert not validate_blueprint(bp, snapshot["agents"], snapshot["providers"])
    with checkpoint_store(str(tmp_path / "branch.db")) as cp:
        graph = compile_workflow(
            bp,
            snapshot["agents"],
            snapshot["providers"],
            lambda k, v: events.append((k, v)),
            cp,
        )
        result = graph.invoke(
            {"query": '{"route":"research"}', "outputs": {}, "final": ""},
            {"configurable": {"thread_id": "branch"}},
        )
    assert "Daily briefing" in result["final"]
    assert result["skipped"]["analysis"] is True
    assert sum(k == "node_started" and v["node_id"] == "writer" for k, v in events) == 1
    assert not any(
        k == "node_started" and v["node_id"] == "analysis" for k, v in events
    )


def test_contracts_reject_bad_fields_and_remote_refs(snapshot):
    from xeon_runtime.contracts import mapped_input

    assert mapped_input(
        {"count": {"source": "analysis", "path": "tickets_closed"}},
        {"analysis": '{"tickets_closed":84}'},
        "x",
    ) == {"count": 84}
    with pytest.raises(ValueError):
        mapped_input({"count": {"source": "missing", "path": "x"}}, {}, "x")
    bp = snapshot["agents"]["writer@1"]
    out = next(n for n in bp["nodes"] if n["kind"] == "output")
    out["config"] = {
        "format": "json",
        "schema": {
            "type": "object",
            "properties": {"tickets_closed": {"type": "integer", "minimum": 100}},
            "required": ["tickets_closed"],
        },
    }
    with pytest.raises(ValueError, match="Output"):
        run_agent(
            Blueprint.model_validate(bp),
            snapshot["providers"],
            "briefing",
            lambda *a: None,
        )
    out["config"]["schema"] = {"$ref": "https://untrusted.example/schema"}
    with pytest.raises(ValueError):
        Blueprint.model_validate(bp)


def test_conversation_memory_isolation_and_recovery(snapshot, tmp_path, monkeypatch):
    from xeon_runtime.memory import MemoryStore
    from xeon_runtime.providers import DemoModel

    seen = []
    original = DemoModel._generate

    def track(self, messages, **kwargs):
        seen.append([m.content for m in messages if isinstance(m, HumanMessage)])
        return original(self, messages, **kwargs)

    monkeypatch.setattr(DemoModel, "_generate", track)
    data = snapshot["agents"]["writer@1"]
    body = next(n["id"] for n in data["nodes"] if n["kind"] == "agent")
    data["nodes"].append(
        dict(
            id="memory",
            kind="memory",
            label="Conversation",
            config={"scope": "conversation", "max_turns": 2},
            position={"x": 0, "y": 0},
        )
    )
    data["edges"].append(dict(id="memory-edge", source="memory", target=body))
    bp = Blueprint.model_validate(data)
    memory = MemoryStore("sqlite:///" + str(tmp_path / "memory.db"))

    def invoke(key, run, query):
        return run_agent(
            bp,
            snapshot["providers"],
            query,
            lambda *a: None,
            {"configurable": {"thread_id": run, "conversation_key": key}},
            memory_store=memory,
        )

    invoke("alice", "one", "first request")
    invoke("alice", "two", "second request")
    assert seen[-1] == ["first request", "second request"]
    count = len(seen)
    invoke("alice", "two", "second request")
    assert len(seen) == count
    invoke("bob", "three", "independent request")
    assert seen[-1] == ["independent request"]


def test_custom_usage_mapping_preserves_unknown(monkeypatch):
    original = httpx.Client
    response = {"answer": "done", "stats": {"in": 7, "out": 3}}
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response)),
            **kw,
        ),
    )
    p = Provider(
        id="api",
        name="custom",
        adapter="http_json",
        endpoint="https://example.com/generate",
        supports_tools=False,
        response_path="answer",
        usage_path="stats",
        input_tokens_field="in",
        output_tokens_field="out",
    )
    assert JsonAPIModel(profile=p).invoke("hello").usage_metadata["total_tokens"] == 10
    response.pop("stats")
    assert JsonAPIModel(profile=p).invoke("hello").usage_metadata is None


def test_grounding_policy_is_named_and_switchable(snapshot):
    from xeon_runtime.policies import GROUNDING

    def system_prompts(config):
        agent = json.loads(json.dumps(snapshot["agents"]["analysis@1"]))
        body = next(n for n in agent["nodes"] if n["kind"] == "agent")
        body["config"].update(config)
        events = []
        run_agent(
            Blueprint.model_validate(agent),
            snapshot["providers"],
            "Analyze the metrics",
            lambda k, v: events.append((k, v)),
            {"configurable": {"capture_trace": True}},
        )
        return [
            m["content"]
            for k, v in events
            if k == "model_input"
            for batch in v.get("messages", [])
            for m in batch
            if m["role"] == "system"
        ]

    sentence = GROUNDING.split(".")[0]
    assert any(sentence in p for p in system_prompts({}))
    assert not any(sentence in p for p in system_prompts({"grounding": False}))
