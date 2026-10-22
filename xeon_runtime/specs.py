from __future__ import annotations

import math
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import check_schema
from .tool_catalog import TOOLS


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provider(SpecModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    name: str
    adapter: Literal["mock", "openai", "http_json"] = "openai"
    endpoint: str = ""
    model: str = ""
    api_key_env: str = ""
    headers_env: dict[str, str] = Field(default_factory=dict)
    supports_tools: bool = True
    timeout_seconds: int = Field(default=90, ge=1, le=600)
    streaming: bool = True
    # Templates are data only. Whole-value placeholders preserve JSON types.
    request_template: dict = Field(
        default_factory=lambda: {
            "model": "$model",
            "messages": "$messages",
            "tools": "$tools",
        }
    )
    finish_reason_path: str = "choices.0.finish_reason"
    response_path: str = "choices.0.message.content"
    tool_calls_path: str = "choices.0.message.tool_calls"
    usage_path: str = "usage"
    input_tokens_field: str = "prompt_tokens"
    output_tokens_field: str = "completion_tokens"
    tool_call_format: Literal["structured", "qwen_xml"] = "structured"
    # Role name -> model name at this endpoint (for example a LiteLLM alias).
    # A role that is not listed falls back to `model`, so a single-model
    # endpoint serves every role.
    roles: dict[str, str] = Field(default_factory=dict)

    def model_for(self, role: str | None) -> str:
        return self.roles.get(role or "", "") or self.model

    @model_validator(mode="after")
    def endpoint_and_secrets(self):
        for role, name in self.roles.items():
            if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", role) or not name.strip():
                raise ValueError(
                    "Roles map a short lowercase role name to a non-empty model name"
                )
        if self.adapter != "mock":
            u = urlsplit(self.endpoint)
            if (
                u.scheme not in ("http", "https")
                or not u.hostname
                or u.username
                or u.password
                or u.query
                or u.fragment
            ):
                raise ValueError(
                    "Endpoint must be an HTTP(S) URL without credentials, query, or fragment"
                )
            if (
                self.adapter == "http_json"
                and self.supports_tools
                and not self.tool_calls_path
                and self.tool_call_format == "structured"
            ):
                raise ValueError("Tool-capable JSON APIs need a tool_calls_path")
        if self.tool_call_format != "structured" and self.adapter != "http_json":
            raise ValueError("Text tool-call parsing requires the HTTP JSON adapter")
        for name in [self.api_key_env, *self.headers_env.values()]:
            if name and not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                raise ValueError(
                    "Credentials must reference an uppercase environment variable name"
                )
        return self


class Node(SpecModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    kind: str
    label: str
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0})
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def configuration_types(self):
        if self.id in ("__start__", "__end__"):
            raise ValueError("Component ID is reserved by the workflow runtime")
        if set(self.position) != {"x", "y"} or not all(
            math.isfinite(v) for v in self.position.values()
        ):
            raise ValueError(
                "Component position must contain finite x and y coordinates"
            )
        fields = {
            "agent": {
                "runtime": str,
                "max_steps": int,
                "input_schema": dict,
                # Named grounding policy; on unless explicitly disabled.
                "grounding": bool,
            },
            "instructions": {"text": str},
            "model": {"provider_id": str, "role": str},
            "knowledge": {"corpus": str},
            "tool": {"tool_id": str},
            "output": {"format": str, "schema": dict},
            "validation": {"schema": dict},
            "memory": {"scope": str, "max_turns": int},
            "subagent": {"instructions": str},
            "agent_ref": {
                "agent_ref": str,
                "task": str,
                "input_map": dict,
                "required_tools": list,
            },
            "condition": {
                "source": str,
                "path": str,
                "operator": str,
                "value": (str, int, float, bool, list, dict, type(None)),
            },
            "join": {},
            "trigger": {},
            "approval": {},
            "result": {},
        }.get(self.kind)
        if fields is not None:
            for key, value in self.config.items():
                allowed = fields.get(key)
                if allowed is None or (
                    type(value) not in allowed
                    if isinstance(allowed, tuple)
                    else type(value) is not allowed
                ):
                    raise ValueError(
                        f"{self.label}: unsupported configuration field or type: {key}"
                    )
        if "required_tools" in self.config and (
            len(self.config["required_tools"]) > 6
            or any(
                not isinstance(t, str) or not t for t in self.config["required_tools"]
            )
        ):
            raise ValueError("Required tools must be a list of up to six tool IDs")
        for key in ("schema", "input_schema"):
            if key in self.config:
                check_schema(self.config[key])
        if self.kind == "memory" and (
            self.config.get("scope", "run") not in ("run", "conversation")
            or not 1 <= self.config.get("max_turns", 5) <= 20
        ):
            raise ValueError(
                "Memory scope must be run or conversation, with 1–20 turns"
            )
        if self.kind == "condition" and self.config.get("operator", "equals") not in (
            "equals",
            "not_equals",
            "contains",
            "exists",
            "gt",
            "gte",
            "lt",
            "lte",
        ):
            raise ValueError("Choose a supported condition operator")
        for name, binding in self.config.get("input_map", {}).items():
            if (
                not isinstance(binding, dict)
                or set(binding) - {"source", "path"}
                or not isinstance(binding.get("source"), str)
                or not isinstance(binding.get("path", ""), str)
            ):
                raise ValueError(
                    f"Input mapping {name} needs source and optional field path"
                )
        return self


class Edge(SpecModel):
    id: str
    source: str
    target: str
    source_handle: Literal["true", "false"] | None = None


class Blueprint(SpecModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    kind: Literal["agent", "workflow"]
    nodes: list[Node] = Field(default_factory=list, max_length=100)
    edges: list[Edge] = Field(default_factory=list, max_length=300)


AGENT_KINDS = {
    "agent",
    "instructions",
    "model",
    "knowledge",
    "tool",
    "output",
    "subagent",
    "memory",
    "validation",
}
WORKFLOW_KINDS = {"trigger", "agent_ref", "approval", "result", "condition", "join"}


def validate_blueprint(
    bp: Blueprint,
    agents: dict | None = None,
    providers: dict | None = None,
    tools: dict | None = None,
) -> list[str]:
    """Validate a blueprint; `tools` extends the built-in catalog with the
    registered tool specs pinned in the execution snapshot."""
    errors = []
    ids = [n.id for n in bp.nodes]
    if len(ids) != len(set(ids)):
        errors.append("Node IDs must be unique")
    if len({e.id for e in bp.edges}) != len(bp.edges):
        errors.append("Connection IDs must be unique")
    if len({(e.source, e.target) for e in bp.edges}) != len(bp.edges):
        errors.append("Duplicate connections are not allowed")
    kinds = AGENT_KINDS if bp.kind == "agent" else WORKFLOW_KINDS
    for n in bp.nodes:
        if n.kind not in kinds:
            errors.append(f"{n.label}: unsupported component {n.kind}")
    incoming = {n: [] for n in ids}
    outgoing = {n: [] for n in ids}
    for e in bp.edges:
        if e.source not in ids or e.target not in ids:
            errors.append("A connection references a missing component")
        else:
            incoming[e.target].append(e.source)
            outgoing[e.source].append(e.target)
            source = next(n for n in bp.nodes if n.id == e.source)
            if (source.kind == "condition") != (e.source_handle is not None):
                errors.append("Only condition connections use true/false branches")
    if bp.kind == "agent":
        bodies = [n for n in bp.nodes if n.kind == "agent"]
        if len(bodies) != 1:
            errors.append("An agent needs exactly one agent body")
            return errors
        body = bodies[0]
        if outgoing[body.id]:
            errors.append(
                "Capabilities connect into the agent body; the body has no outgoing attachments"
            )
        if body.config.get("runtime", "langchain") not in ("langchain", "deepagents"):
            errors.append("Select LangChain or Deep Agents as the runtime")
        for n in bp.nodes:
            if n.id != body.id and (outgoing[n.id] != [body.id] or incoming[n.id]):
                errors.append(
                    f"{n.label}: attach this capability directly to the agent body"
                )
            if (
                n.kind == "tool"
                and n.config.get("tool_id") not in TOOLS
                and n.config.get("tool_id") not in (tools or {})
            ):
                errors.append(f"{n.label}: select an available tool")
            if n.kind == "knowledge" and n.config.get("corpus", "bundled") != "bundled":
                errors.append(
                    f"{n.label}: only the bundled corpus is available in this release"
                )
            if n.kind == "instructions" and not n.config.get("text", "").strip():
                errors.append("Instructions cannot be empty")
            if n.kind == "output" and n.config.get("format", "text") not in (
                "text",
                "json",
            ):
                errors.append("Output format must be text or json")
            if n.kind == "subagent":
                if body.config.get("runtime") != "deepagents":
                    errors.append("Specialist subagents require Deep Agents")
                if not n.config.get("instructions"):
                    errors.append(f"{n.label}: add specialist instructions")
            if n.kind == "model" and providers is not None:
                provider = providers.get(n.config.get("provider_id"))
                if not provider:
                    errors.append(f"{n.label}: provider profile is missing")
                elif not Provider.model_validate(provider).supports_tools and (
                    body.config.get("runtime") == "deepagents"
                    or any(x.kind in ("tool", "knowledge") for x in bp.nodes)
                ):
                    errors.append(
                        "This agent needs a provider that supports tool calling"
                    )
        for kind in ("instructions", "model", "output"):
            if sum(n.kind == kind for n in bp.nodes) != 1:
                errors.append(f"Attach exactly one {kind} component")
        for kind in ("memory", "validation"):
            if sum(n.kind == kind for n in bp.nodes) > 1:
                errors.append(f"Attach at most one {kind} component")
        limit = body.config.get("max_steps", 30)
        if not isinstance(limit, int) or not 2 <= limit <= 200:
            errors.append("Agent step limit must be between 2 and 200")
    else:
        starts = [n.id for n in bp.nodes if n.kind == "trigger"]
        ends = [n.id for n in bp.nodes if n.kind == "result"]
        if len(starts) != 1 or len(ends) != 1:
            errors.append("A workflow needs one trigger and one result")
        if not any(n.kind == "agent_ref" for n in bp.nodes):
            errors.append("Add at least one published agent")
        for n in bp.nodes:
            if n.kind == "condition":
                branches = [e.source_handle for e in bp.edges if e.source == n.id]
                if sorted(x or "" for x in branches) != ["false", "true"]:
                    errors.append(
                        f"{n.label}: connect exactly one true and one false branch"
                    )
                if not n.config.get("source"):
                    errors.append(f"{n.label}: choose the condition source")
            if (
                n.kind == "agent_ref"
                and agents is not None
                and n.config.get("agent_ref") not in agents
            ):
                errors.append(f"{n.label}: select a published agent version")
            if n.kind != "trigger" and not incoming[n.id]:
                errors.append(f"{n.label}: connect an input")
            if n.kind != "result" and not outgoing[n.id]:
                errors.append(f"{n.label}: connect an output")
            if (
                n.kind == "trigger"
                and incoming[n.id]
                or n.kind == "result"
                and outgoing[n.id]
            ):
                errors.append(f"{n.label}: connection direction is invalid")
        seen, active = set(), set()

        def walk(n):
            if n in active:
                errors.append("Workflow connections must not contain a cycle")
                return
            if n in seen:
                return
            seen.add(n)
            active.add(n)
            for nxt in outgoing[n]:
                walk(nxt)
            active.remove(n)

        for n in starts:
            walk(n)
        if set(ids) != seen:
            errors.append("Every component must be reachable from the trigger")

        def ancestors(node, visited=None):
            visited = set() if visited is None else visited
            for parent in incoming[node]:
                if parent not in visited:
                    visited.add(parent)
                    ancestors(parent, visited)
            return visited

        for n in bp.nodes:
            sources = [b["source"] for b in n.config.get("input_map", {}).values()]
            if n.kind == "condition":
                sources.append(n.config.get("source", ""))
            for source in sources:
                if source != "$input" and source not in ancestors(n.id):
                    errors.append(
                        f"{n.label}: mapped source must be an upstream component or $input"
                    )
    return list(dict.fromkeys(errors))
