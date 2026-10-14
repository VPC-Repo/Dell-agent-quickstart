from __future__ import annotations

import time
import json
import re
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from openai import APIError, APIStatusError
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .contracts import as_data, field, mapped_input, predicate, validate_data
from .demo_tools import REGISTRY as DEMO_TOOLS
from .enterprise_tools import REGISTRY as ENTERPRISE_TOOLS
from .providers import create_model, resolve_role
from .specs import Blueprint, Provider, validate_blueprint
from .telemetry import RunObserver


def merge_outputs(a: dict, b: dict):
    return {**a, **b}


class WorkflowState(TypedDict):
    query: str
    outputs: Annotated[dict[str, str], merge_outputs]
    final: str
    skipped: Annotated[dict[str, bool], merge_outputs]
    decisions: Annotated[dict[str, bool], merge_outputs]


def character_limit(instruction):
    limits = []
    for match in re.finditer(
        r"\b(under|less than|at most|up to|no more than|maximum(?: of)?)\s+(\d+)\s*[- ]?characters?\b",
        instruction,
        re.I,
    ):
        limits.append(
            max(
                0,
                int(match[2])
                - (1 if match[1].lower() in ("under", "less than") else 0),
            )
        )
    return min(limits) if limits else None


def tools_for(bp: Blueprint, extra_tools: dict | None = None, max_message_chars=None):
    corpus = json.loads(Path(__file__).with_name("corpus.json").read_text())

    @tool
    def search_knowledge(query: str) -> str:
        """Search the bundled demonstration corpus; return source IDs and source text."""
        words = set(query.lower().split())
        ranked = sorted(
            corpus["documents"],
            key=lambda d: sum(w in d["text"].lower() for w in words),
            reverse=True,
        )
        return json.dumps(ranked[:3])

    @tool
    def analyze_metrics() -> str:
        """Calculate the sample operations metrics. These are synthetic fixtures."""
        m = corpus["metrics"]
        return json.dumps(
            {
                **m,
                "backlog_change": m["backlog_current"] - m["backlog_previous"],
                "backlog_change_percent": round(
                    (m["backlog_current"] / m["backlog_previous"] - 1) * 100, 1
                ),
                "synthetic": True,
            }
        )

    selected = {n.config.get("tool_id") for n in bp.nodes if n.kind == "tool"}
    if any(n.kind == "knowledge" for n in bp.nodes):
        selected.add("search_knowledge")
    from .artifacts import ARTIFACT_TOOLS

    registry = {
        "search_knowledge": search_knowledge,
        "analyze_metrics": analyze_metrics,
        **DEMO_TOOLS,
        **ENTERPRISE_TOOLS,
        **{t.name: t for t in ARTIFACT_TOOLS},
    }
    unknown = sorted(selected - set(registry))
    if unknown:
        raise ValueError(
            "These tools are not installed or pinned in this snapshot: "
            + ", ".join(unknown)
        )
    return [registry[name] for name in sorted(selected)]


def model_profile(bp: Blueprint, providers: dict) -> Provider:
    """The agent's provider with its model node's role resolved to a model name."""
    node = next(n for n in bp.nodes if n.kind == "model")
    profile = Provider.model_validate(providers[node.config["provider_id"]])
    return resolve_role(profile, node.config.get("role"))


def compile_agent(
    bp: Blueprint,
    providers: dict,
    checkpointer=None,
    extra_tools=None,
    max_message_chars=None,
):
    errors = validate_blueprint(bp, providers=providers, tools=extra_tools)
    if errors:
        raise ValueError("; ".join(errors))
    body = next(n for n in bp.nodes if n.kind == "agent")
    model = create_model(model_profile(bp, providers))
    prompt = next(n.config["text"] for n in bp.nodes if n.kind == "instructions")
    prompt += (
        "\nExecution date (UTC): "
        + datetime.now(timezone.utc).date().isoformat()
        + ". If a report date is needed, use this execution date. Event dates must come from sources."
    )
    if body.config.get("grounding", True):
        from .policies import GROUNDING

        prompt += "\n" + GROUNDING
    output = next(n for n in bp.nodes if n.kind == "output")
    if output.config.get("format") == "json":
        prompt += "\nReturn valid JSON only. No markdown fences."
    schemas = [
        n.config["schema"]
        for n in bp.nodes
        if n.kind in ("output", "validation") and n.config.get("schema")
    ]
    for schema in schemas:
        prompt += "\nReturn JSON satisfying this schema: " + json.dumps(schema)
    params = dict(
        model=model,
        system_prompt=prompt,
        tools=tools_for(bp, extra_tools, max_message_chars),
        checkpointer=checkpointer,
    )
    if body.config.get("runtime", "langchain") == "deepagents":
        from deepagents import create_deep_agent

        subagents = [
            {
                "name": n.id,
                "description": n.label,
                "system_prompt": n.config["instructions"],
                "model": model,
                "tools": tools_for(bp, extra_tools, max_message_chars),
            }
            for n in bp.nodes
            if n.kind == "subagent"
        ]
        return create_deep_agent(**params, subagents=subagents)
    return create_agent(**params)


def run_agent(
    bp: Blueprint,
    providers: dict,
    query: str,
    emit,
    config=None,
    checkpointer=None,
    resume=False,
    memory_store=None,
    extra_tools=None,
    required_tools=(),
    max_message_chars=None,
):
    if max_message_chars is not None:
        selected = {n.config.get("tool_id") for n in bp.nodes if n.kind == "tool"}
        destinations = {
            name
            for name, spec in (extra_tools or {}).items()
            if name in selected
            and spec.get("kind") == "connector"
            and (spec.get("connection") or {}).get("family")
            in ("telegram", "email", "slack", "discord")
        }
        required_tools = set(required_tools) | destinations
    graph = compile_agent(bp, providers, checkpointer, extra_tools, max_message_chars)
    body = next(n for n in bp.nodes if n.kind == "agent")
    cfg = {**(config or {}), "recursion_limit": body.config.get("max_steps", 30)}
    if body.config.get("input_schema"):
        validate_data(as_data(query), body.config["input_schema"], "Input")
    profile = model_profile(bp, providers)
    policy = cfg.get("configurable", {})
    observer = RunObserver(
        emit,
        profile,
        policy.get("token_budget", 0),
        policy.get("prior_tokens", 0),
        capture_trace=policy.get("capture_trace", False),
    )
    callbacks = cfg.get("callbacks")
    if callbacks is not None and not isinstance(callbacks, list):
        # LangGraph and Deep Agents pass a CallbackManager to nested graphs.
        # Preserve its tracing context without mutating the caller's manager.
        callbacks = callbacks.copy()
        callbacks.add_handler(observer, inherit=True)
        cfg["callbacks"] = callbacks
    else:
        cfg["callbacks"] = [*(callbacks or []), observer]
    memory = next(
        (
            n
            for n in bp.nodes
            if n.kind == "memory" and n.config.get("scope") == "conversation"
        ),
        None,
    )
    if memory and (not memory_store or not policy.get("conversation_key")):
        raise ValueError(
            "Conversation memory requires a conversation key and memory store"
        )
    context = (
        memory_store.conversation(
            bp, policy["conversation_key"], memory.config.get("max_turns", 5)
        )
        if memory
        else nullcontext([])
    )
    with context as turns:
        invocation = policy.get("thread_id")
        if memory and invocation:
            cached = next((t for t in turns if t.get("run_id") == invocation), None)
            if cached:
                emit("memory_recovered", {"agent_id": bp.id})
                return cached["output"]
        messages = []
        for turn in turns:
            messages.extend(
                [
                    {"role": "user", "content": turn["query"]},
                    {"role": "assistant", "content": turn["output"]},
                ]
            )
        messages.append({"role": "user", "content": query})
        text = _stream_agent(
            graph, bp, messages, cfg, emit, resume, required_tools, max_message_chars
        )
        if memory:
            turns.append({"query": query, "output": text, "run_id": invocation})
        return text


CORRECTION_ROUNDS = 2


def _stream_agent(
    graph, bp, messages, cfg, emit, resume, required_tools=(), max_message_chars=None
):
    last = None
    # Stream completed steps rather than inventing token-level progress for generic APIs.
    has_checkpoint = resume and bool(graph.get_state(cfg).values)
    inp = None if has_checkpoint else {"messages": messages}
    stalls, previous, out_of_steps, api_retries = 0, None, False, 0
    for attempt in range(CORRECTION_ROUNDS + len(required_tools) + 1):
        try:
            for state in graph.stream(inp, cfg, stream_mode="values"):
                last = state
                messages = state.get("messages", [])
                if messages:
                    msg = messages[-1]
                    if getattr(msg, "tool_calls", None):
                        emit("tool", {"tools": [c["name"] for c in msg.tool_calls]})
        except APIError as exc:
            # An engine or gateway failure in the middle of a response (vLLM's gpt-oss parser choked on a tool-call
            # header once in a hundred calls). A client error is the request's fault and is not retried; anything
            # else gets one more attempt from the last good state. The telemetry already recorded the model_error.
            status = getattr(exc, "status_code", None)
            # 429 is back-pressure; 404 from the gateway has been a stale route to a restarted engine ("model does not
            # exist" from the wrong container), which a moment later resolves. Other 4xx are the request's fault.
            if (
                isinstance(exc, APIStatusError)
                and status
                and 400 <= status < 500
                and status not in (404, 429)
            ) or api_retries >= 1:
                raise
            api_retries += 1
            if status in (404, 429):
                time.sleep(2)
            inp = (
                inp
                if last is None
                else (
                    {"messages": []}
                    if getattr(graph, "checkpointer", None) is not None
                    else {"messages": last["messages"]}
                )
            )
            continue
        except GraphRecursionError:
            # The step budget ran out mid-investigation (Qwen3-Coder spent ten queries on the churn cohort).
            # Rather than failing the run, ask once for the answer from what was collected, with no more tools.
            if out_of_steps or not last or not last.get("messages"):
                raise
            out_of_steps = True
            emit(
                "tool_error",
                {
                    "tool": "step budget",
                    "message": "The step used all its tool steps; asked to answer from the results collected so far",
                    "detail": "Step limit reached before an answer; the worker was asked to write its answer from the results it already has",
                },
            )
            correction = {
                "role": "user",
                "content": "You have used all your tool steps. Do not call any more tools. Write your final answer now from the results you already have, and say plainly what you could not verify.",
            }
            inp = (
                {"messages": [correction]}
                if getattr(graph, "checkpointer", None) is not None
                else {"messages": [*last["messages"], correction]}
            )
            cfg = {**cfg, "recursion_limit": 4}
            continue
        if not last:
            raise ValueError("Agent returned no state")
        # Check persisted tool results too, so a resumed step does not send again
        # merely because its successful call happened before the checkpoint.
        completed_tools = {
            getattr(m, "name", None)
            for m in last["messages"]
            if getattr(m, "type", "") == "tool"
            and getattr(m, "status", "success") == "success"
        }
        missing = sorted(set(required_tools) - completed_tools)
        if not missing:
            break
        # A correction that changed nothing counts against the model; one that got a tool called does not
        # (gpt-oss-120b needed two nudges for the first tool and one more for the second).
        stalls += previous is not None and len(missing) >= len(previous)
        previous = missing
        if stalls > CORRECTION_ROUNDS or attempt == CORRECTION_ROUNDS + len(
            required_tools
        ):
            raise ValueError(
                "Step did not complete its required tool actions: " + ", ".join(missing)
            )
        # The model answered without a tool its step depends on (gpt-oss-120b wrote an incident analysis
        # after only describing the warehouse). Say so and let it continue; the slip is surfaced as a
        # recovered error like a failed tool call.
        names = ", ".join(missing)
        order = " first, then ".join(missing) if len(missing) > 1 else missing[0]
        emit(
            "tool_error",
            {
                "tool": names,
                "message": "The step answered without calling "
                + names
                + "; asked to continue with the tool",
                "detail": "Answer given before "
                + names
                + " was called; the worker was asked to call it and answer from its results",
            },
        )
        correction = {
            "role": "user",
            "content": f"Your answer is not accepted yet: your task requires calling {names} and you have not. Do not answer again from memory. Call {order}, each with the data you need, then write your final answer from the results.",
        }
        # With a checkpointer the state already holds the conversation; without one, resend it with the correction.
        inp = (
            {"messages": [correction]}
            if getattr(graph, "checkpointer", None) is not None
            else {"messages": [*last["messages"], correction]}
        )
    value = last["messages"][-1].content
    text = value if isinstance(value, str) else json.dumps(value)
    delivered = False
    for message in last["messages"]:
        if (
            getattr(message, "type", "") != "tool"
            or getattr(message, "status", "success") != "success"
        ):
            continue
        try:
            result = json.loads(message.content).get("result", {})
            delivered = delivered or bool(
                result.get("saved_to_demo_outbox")
                or result.get("delivered")
                or result.get("accepted_by_smtp")
            )
        except (ValueError, TypeError, AttributeError):
            pass
    if (
        max_message_chars is not None
        and not delivered
        and len(text) > max_message_chars
    ):
        raise ValueError(
            f"Step output has {len(text)} characters; maximum allowed is {max_message_chars}."
        )
    if any(n.kind == "output" and n.config.get("format") == "json" for n in bp.nodes):
        json.loads(
            text
        )  # Enforce the selected contract, not just the prompt instruction.
    for n in bp.nodes:
        if n.kind in ("output", "validation") and n.config.get("schema"):
            validate_data(as_data(text), n.config["schema"])
    emit("validated", {"agent_id": bp.id, "valid": True})
    return text


def compile_workflow(
    bp: Blueprint,
    agents: dict,
    providers: dict,
    emit,
    checkpointer,
    memory_store=None,
    runtime_config=None,
    extra_tools=None,
):
    errors = validate_blueprint(bp, agents, providers, tools=extra_tools)
    if errors:
        raise ValueError("; ".join(errors))
    graph = StateGraph(WorkflowState)
    predecessors = {
        n.id: [e.source for e in bp.edges if e.target == n.id] for n in bp.nodes
    }
    for node in bp.nodes:

        def execute(state, node=node):
            incoming = [e for e in bp.edges if e.target == node.id]
            active = [
                e.source
                for e in incoming
                if not state.get("skipped", {}).get(e.source, False)
                and (
                    e.source_handle is None
                    or state.get("decisions", {}).get(e.source)
                    is (e.source_handle == "true")
                )
            ]
            if incoming and not active:
                emit("node_skipped", {"node_id": node.id, "label": node.label})
                return {"skipped": {node.id: True}}
            emit("node_started", {"node_id": node.id, "label": node.label})
            inputs = [state["outputs"].get(i, "") for i in active]
            text = "\n\n".join(inputs) or state["query"]
            if node.kind == "agent_ref":
                agent = Blueprint.model_validate(agents[node.config["agent_ref"]])
                instruction = node.config.get(
                    "task", "Complete your task using the supplied context."
                )
                mapping = node.config.get("input_map", {})
                agent_query = (
                    json.dumps(mapped_input(mapping, state["outputs"], state["query"]))
                    if mapping
                    else f"Objective: {state['query']}\nTask: {instruction}\nContext:\n{text}"
                )
                text = run_agent(
                    agent,
                    providers,
                    agent_query,
                    lambda kind, payload: emit(
                        kind, {**payload, "task_id": node.id, "task_name": node.label}
                    ),
                    config={
                        **(runtime_config or {}),
                        "configurable": {
                            **(runtime_config or {}).get("configurable", {}),
                            "thread_id": str(
                                (runtime_config or {})
                                .get("configurable", {})
                                .get("thread_id", "")
                            )
                            + ":"
                            + node.id,
                        },
                    }
                    if runtime_config
                    else None,
                    memory_store=memory_store,
                    extra_tools=extra_tools,
                    required_tools=node.config.get("required_tools", []),
                    max_message_chars=character_limit(instruction),
                )
            elif node.kind == "condition":
                source = node.config["source"]
                value = (
                    state["query"]
                    if source == "$input"
                    else state["outputs"].get(source)
                )
                try:
                    value = field(value, node.config.get("path", ""))
                except ValueError:
                    if node.config.get("operator") == "exists":
                        value = None
                    else:
                        raise
                decision = predicate(
                    value,
                    node.config.get("operator", "equals"),
                    node.config.get("value"),
                )
                emit(
                    "condition",
                    {"node_id": node.id, "selected_branch": str(decision).lower()},
                )
                emit("node_completed", {"node_id": node.id, "label": node.label})
                return {
                    "outputs": {node.id: text},
                    "decisions": {node.id: decision},
                    "skipped": {node.id: False},
                }
            elif node.kind == "approval":
                approved = interrupt(
                    {"node_id": node.id, "label": node.label, "preview": text[:6000]}
                )
                if approved is not True:
                    raise ValueError("Workflow approval was declined")
            result = {"outputs": {node.id: text}, "skipped": {node.id: False}}
            if node.kind == "result":
                result["final"] = text
            emit("node_completed", {"node_id": node.id, "label": node.label})
            return result

        graph.add_node(node.id, execute)
    for node in bp.nodes:
        parents = predecessors[node.id]
        if node.kind == "trigger":
            graph.add_edge(START, node.id)
        elif parents:
            graph.add_edge(
                parents, node.id
            )  # Barrier: wait for every branch before a join.
        if node.kind == "result":
            graph.add_edge(node.id, END)
    return graph.compile(checkpointer=checkpointer)


@contextmanager
def checkpoint_store(url: str):
    if url.startswith("postgres"):
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(url) as saver:
            saver.setup()
            yield saver
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(url) as saver:
            yield saver
