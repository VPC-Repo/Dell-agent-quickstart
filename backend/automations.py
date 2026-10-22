"""Stored agent designs, catalog installation helpers and manual run endpoints."""

from __future__ import annotations

import copy
import time
import uuid
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select

from xeon_runtime.planning import ConnectionNeed, required_step_contract
from xeon_runtime.specs import Blueprint, validate_blueprint
from xeon_runtime.tool_catalog import TOOL_INFO, ToolName
from xeon_runtime.version import runtime_digest

from . import seeds, store
from .deployments import digest, enqueue

router = APIRouter(prefix="/api/automations")


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repeat: Literal["manual"] = "manual"
    time: str = Field(default="08:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    timezone: str = "America/Chicago"

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(
                "Choose a valid IANA timezone, such as America/Chicago"
            ) from None
        return value


class AgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    instructions: str = Field(max_length=50000)
    task: str = Field(
        default="Complete your job using the supplied context.", max_length=50000
    )
    tools: list[ToolName] = Field(default_factory=list, max_length=6)
    output: Literal["text", "json"] = "text"
    depends_on: list[str] | None = None
    position: dict[str, float] | None = None
    repeat_count: int = Field(default=1, ge=1, le=5)
    worker_ref: str = ""
    worker_name: str = ""
    expected_output: str = Field(default="", max_length=600)
    resource_notes: str = Field(default="", max_length=600)
    required: bool = False
    requires: list[str] = Field(default_factory=list, max_length=8)
    uses_tools: list[ToolName] = Field(default_factory=list, max_length=6)
    connection_needs: list[ConnectionNeed] = Field(default_factory=list, max_length=6)
    max_steps: int = Field(default=30, ge=4, le=60)
    # None keeps the library worker's role; the provider's role map picks the engine.
    role: Literal["general", "coder", "reasoning"] | None = None

    @field_validator("connection_needs")
    @classmethod
    def one_need_per_family(cls, value):
        # A step requires a family once; binding one connection satisfies it.
        seen = {}
        for need in value:
            seen.setdefault(need.family, need)
        return list(seen.values())


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    name: str = Field(min_length=1, max_length=120)
    goal: str = Field(max_length=100000)
    provider_id: str
    requirements: list[str] = Field(default_factory=list, max_length=8)
    output_checks: list[str] = Field(default_factory=list, max_length=60)
    expected_result: str = Field(default="", max_length=4000)
    agents: list[AgentInput] = Field(min_length=1, max_length=12)
    schedule: Settings = Field(default_factory=Settings)
    execution_mode: Literal["adaptive", "authored"] | None = None
    worker_refs: list[str] | None = Field(default=None, min_length=1, max_length=100)
    # Catalog presentation and verification settings, available to every agent
    # (seeded demo agents carry defaults in their demo block). None keeps the
    # stored value.
    deadline_seconds: int | None = Field(default=None, ge=1, le=3600)
    workload_note: str | None = Field(default=None, max_length=200)


class Revision(BaseModel):
    revision: str


class RunInput(Revision):
    model_config = ConfigDict(extra="forbid")


def put(db, key, kind, payload):
    doc = db.get(store.Document, key)
    if doc:
        doc.payload, doc.updated = copy.deepcopy(payload), time.time()
    else:
        db.add(store.Document(key=key, kind=kind, payload=copy.deepcopy(payload)))


def metadata(db, id, lock=False):
    query = select(store.Document).where(store.Document.key == "automation:" + id)
    doc = db.scalar(query.with_for_update() if lock else query)
    if not doc:
        raise HTTPException(404, "Automation not found")
    migrated = store.upgraded("automation", doc.payload)
    if migrated is not doc.payload:
        doc.payload = migrated
    if lock:
        keys = {"draft:" + doc.payload["blueprint_id"]}
        keys.update("draft:" + b["id"] for b in doc.payload["bindings"].values())
        # Lock the worker definitions before reading the design.
        list(
            db.scalars(
                select(store.Document)
                .where(store.Document.key.in_(keys))
                .order_by(store.Document.key)
                .with_for_update()
            )
        )
    return doc


def linear_order(bp):
    """Topological order of a simple worker DAG; advanced nodes remain lossless."""
    if any(n["kind"] not in ("trigger", "agent_ref", "result") for n in bp["nodes"]):
        return None
    if (
        sum(n["kind"] == "trigger" for n in bp["nodes"]) != 1
        or sum(n["kind"] == "result" for n in bp["nodes"]) != 1
    ):
        return None
    if any(
        n["id"] != n["kind"].replace("trigger", "start")
        for n in bp["nodes"]
        if n["kind"] in ("trigger", "result")
    ):
        return None
    ids = {n["id"] for n in bp["nodes"]}
    if len(ids) != len(bp["nodes"]) or any(
        e.get("source_handle") or e["source"] not in ids or e["target"] not in ids
        for e in bp["edges"]
    ):
        return None
    done, ordered = set(), []
    while len(done) < len(ids):
        ready = [
            n
            for n in bp["nodes"]
            if n["id"] not in done
            and all(e["source"] in done for e in bp["edges"] if e["target"] == n["id"])
        ]
        if not ready:
            return None
        for n in ready:
            incoming = [e for e in bp["edges"] if e["target"] == n["id"]]
            outgoing = [e for e in bp["edges"] if e["source"] == n["id"]]
            if (
                (n["kind"] == "trigger" and incoming)
                or (n["kind"] != "trigger" and not incoming)
                or (n["kind"] == "result" and outgoing)
                or (n["kind"] != "result" and not outgoing)
            ):
                return None
            done.add(n["id"])
            ordered.append(n)
    return ordered


def assemble(db, meta):
    from .tools import registered as registered_tools

    available_tools = registered_tools(db)
    draft = db.get(store.Document, "draft:" + meta["blueprint_id"])
    if not draft:
        raise ValueError("The automation's design is missing")
    bp = copy.deepcopy(draft.payload)
    order = linear_order(bp)
    explicit = meta.get("explicit_dependencies") or any(
        sum(e["source"] == n["id"] for e in bp["edges"]) > 1
        or sum(e["target"] == n["id"] for e in bp["edges"]) > 1
        for n in bp["nodes"]
    )
    reason = "" if order else "This graph cannot be edited by the catalog installer."
    models, agents, raw = set(), [], {}
    for node in order or bp["nodes"]:
        if node["kind"] != "agent_ref":
            continue
        binding = meta["bindings"].get(node["id"])
        if not binding or node["config"].get("agent_ref") != binding["ref"]:
            reason = "A worker reference differs from its stored binding. Repair the catalog definition."
            continue
        doc = db.get(store.Document, "draft:" + binding["id"])
        if not doc:
            reason = "A worker definition is missing. Reinstall the catalog."
            continue
        a = copy.deepcopy(doc.payload)
        raw[node["id"]] = a
        types = {n["kind"] for n in a["nodes"]}
        if (
            types - {"agent", "instructions", "model", "output", "tool"}
            or node["config"].get("input_map")
            or any(
                n["config"].get("schema") or n["config"].get("input_schema")
                for n in a["nodes"]
            )
        ):
            reason = (
                "This worker uses data mappings unsupported by the catalog installer."
            )
        for kind in ("agent", "instructions", "model", "output"):
            if sum(n["kind"] == kind for n in a["nodes"]) != 1:
                reason = (
                    "A worker must have one body, instructions, model and output node."
                )
        by_kind = {n["kind"]: n["config"] for n in a["nodes"]}
        if not 4 <= by_kind.get("agent", {}).get("max_steps", 30) <= 60:
            reason = (
                "The catalog installer supports worker execution limits from 4 to 60."
            )
        model = by_kind.get("model", {}).get("provider_id", "")
        models.add(model)
        agents.append(
            {
                "id": node["id"],
                "position": node["position"],
                "name": node["label"],
                "instructions": by_kind.get("instructions", {}).get("text", ""),
                "task": node["config"].get("task", ""),
                "tools": [
                    n["config"].get("tool_id")
                    for n in a["nodes"]
                    if n["kind"] == "tool"
                ],
                "output": by_kind.get("output", {}).get("format", "text"),
                "max_steps": by_kind.get("agent", {}).get("max_steps", 30),
                "role": by_kind.get("model", {}).get("role", "general"),
                "repeat_count": binding.get("repeat_count", 1),
                "expected_output": binding.get("expected_output", ""),
                "resource_notes": binding.get("resource_notes", ""),
                "required": binding.get("required", False),
                "requires": binding.get("requires", []),
                "uses_tools": binding.get("uses_tools", []),
                "connection_needs": binding.get("connection_needs", []),
                "worker_ref": binding.get("worker_ref", ""),
                "worker_name": binding.get(
                    "worker_name", a["name"].replace(" agent", " worker")
                ),
                "depends_on": [
                    e["source"]
                    for e in bp["edges"]
                    if e["target"] == node["id"] and e["source"] != "start"
                ]
                if explicit
                else None,
            }
        )
        # Empty instructions are a valid editing state; malformed topology is not.
        topology_errors = validate_blueprint(
            Blueprint.model_validate(a), tools=available_tools
        )
        if any(e != "Instructions cannot be empty" for e in topology_errors):
            reason = "A worker definition fails blueprint validation."
    if len(models) > 1:
        reason = "The catalog installer requires one model service per agent."
    if len(agents) > 12:
        reason = "The catalog installer supports at most 12 workers per agent."
    profiles = {
        p: d.payload if (d := db.get(store.Document, "provider:" + p)) else None
        for p in models
    }
    signature = digest(
        {
            "blueprint": bp,
            "agents": raw,
            "providers": profiles,
            "goal": meta["goal"],
            "schedule": meta["schedule"],
            "expected_result": meta.get("expected_result", ""),
            "requirements": meta.get("requirements", []),
            "output_checks": meta.get("output_checks", []),
            "runtime": runtime_digest(),
            "coordination": meta.get("coordination"),
            "execution_mode": meta.get(
                "execution_mode", "adaptive" if meta.get("coordination") else "authored"
            ),
            "step_contracts": {
                k: {
                    f: v.get(f)
                    for f in (
                        "required",
                        "requires",
                        "uses_tools",
                        "connection_needs",
                        "expected_output",
                        "resource_notes",
                    )
                }
                for k, v in meta["bindings"].items()
            },
            "repeat_counts": {
                k: v.get("repeat_count", 1) for k, v in meta["bindings"].items()
            },
        }
    )
    public = {
        "id": meta["id"],
        "demo": meta.get("demo"),
        "archived": meta.get("archived", False),
        "blueprint_id": bp["id"],
        "name": bp["name"],
        "goal": meta["goal"],
        "expected_result": meta.get("expected_result", ""),
        "requirements": meta.get("requirements", []),
        "output_checks": meta.get("output_checks", []),
        "deadline_seconds": meta.get("deadline_seconds")
        or (meta.get("demo") or {}).get("deadline"),
        "workload_note": meta.get("workload_note")
        or (meta.get("demo") or {}).get("workload", ""),
        "provider_id": next(iter(models), "demo"),
        "agents": agents,
        "schedule": meta["schedule"],
        "revision": signature,
        "editable": not reason,
        "coordination": meta.get("coordination"),
        "execution_mode": meta.get(
            "execution_mode", "adaptive" if meta.get("coordination") else "authored"
        ),
        "reason": reason,
        "updated": draft.updated,
    }
    from xeon_runtime.readiness import inspect_goal

    from .tools import registered

    public["readiness"] = inspect_goal(
        meta["goal"], meta.get("requirements", []), registered(db)
    )
    pending_connections = [
        {"step": a["name"], **n} for a in agents for n in a.get("connection_needs", [])
    ]
    if pending_connections:
        public["readiness"].update(
            status="blocked",
            connection_needs=pending_connections,
            message="Configure the connections for: "
            + ", ".join(dict.fromkeys(n["step"] for n in pending_connections)),
        )
    return public, bp, raw


def check_revision(item, revision):
    if item["revision"] != revision:
        raise HTTPException(
            409,
            "This automation changed in another view. Reload it before continuing; your local edits have not been applied.",
        )


def require_editable(item):
    if not item["editable"]:
        raise HTTPException(409, item["reason"])


def write_design(db, meta, req):
    from xeon_runtime.readiness import CAPABILITIES

    if set(req.requirements) - CAPABILITIES.keys() or any(
        not x.strip() or len(x) > 300 for x in req.output_checks
    ):
        raise ValueError(
            "Choose supported requirement labels and short nonempty output checks"
        )
    from .tools import catalog_info

    tool_labels = {k: v["label"] for k, v in catalog_info(db).items()}
    unknown_tools = sorted({t for a in req.agents for t in a.tools} - set(tool_labels))
    if unknown_tools:
        raise ValueError("Unknown tools: " + ", ".join(unknown_tools))
    current, bp, raw = assemble(db, meta)
    check_revision(current, req.revision)
    require_editable(current)
    if len({a.id for a in req.agents}) != len(req.agents) or any(
        a.id in ("start", "result") for a in req.agents
    ):
        raise ValueError("Each agent must have a unique ID")
    if req.worker_refs is not None or req.execution_mode is not None:
        from .planning import catalog

        available = {w["ref"]: w for w in catalog(db)}
        refs = req.worker_refs or meta.get("coordination", {}).get("worker_refs", [])
        if not refs or len(set(refs)) != len(refs) or set(refs) - available.keys():
            raise ValueError(
                "Choose available, distinct workers for the approved roster"
            )
        for task in req.agents:
            worker = available.get(task.worker_ref)
            if not worker or task.worker_ref not in refs:
                raise ValueError("Every plan task must use an approved worker")
            if any(
                getattr(task, key) != worker.get(key, default)
                for key, default in (
                    ("instructions", ""),
                    ("tools", []),
                    ("output", "text"),
                    ("max_steps", 30),
                )
            ) or (task.role is not None and task.role != worker.get("role", "general")):
                raise ValueError(
                    "Save edited worker settings as a new library worker first"
                )
        meta["coordination"] = {
            "worker_refs": refs,
            "max_tasks": meta.get("coordination", {}).get("max_tasks", 12),
        }
        meta["execution_mode"] = req.execution_mode or meta.get(
            "execution_mode", "adaptive"
        )
    explicit = any(a.depends_on is not None for a in req.agents)
    if explicit:
        from xeon_runtime.planning import task_edges

        if any(a.depends_on is None for a in req.agents):
            raise ValueError("Specify dependencies for every task")
        edges = task_edges(req.agents)
    provider = db.get(store.Document, "provider:" + req.provider_id)
    if not provider:
        raise ValueError("Choose an available model service")
    from .planning import catalog as worker_catalog

    library_roles = {w["ref"]: w.get("role", "general") for w in worker_catalog(db)}
    levels = {}
    rows = {}
    if explicit:
        for _ in req.agents:
            for task in req.agents:
                if all(parent in levels for parent in task.depends_on):
                    levels[task.id] = (
                        max([-1] + [levels[parent] for parent in task.depends_on]) + 1
                    )
    bindings = {}
    nodes = [
        {
            "id": "start",
            "kind": "trigger",
            "label": "Start",
            "position": {"x": 30, "y": 160},
            "config": {},
        }
    ]
    for i, value in enumerate(req.agents):
        binding = meta["bindings"].get(value.id)
        a = copy.deepcopy(raw.get(value.id))
        if a is None:
            aid = f"{meta['id']}-{uuid.uuid4().hex[:10]}"
            a = seeds.agent(aid, value.name, "", value.instructions)
            binding = {"id": aid, "ref": aid + "@1"}
        a["name"] = value.name
        body = next(n for n in a["nodes"] if n["kind"] == "agent")
        body["label"] = value.name
        body["config"]["max_steps"] = value.max_steps
        binding = {
            **binding,
            "repeat_count": value.repeat_count,
            "worker_ref": value.worker_ref,
            "worker_name": value.worker_name or value.name,
            "expected_output": value.expected_output,
            "resource_notes": value.resource_notes,
            "required": value.required,
            "requires": value.requires,
            "uses_tools": value.uses_tools,
            "connection_needs": [n.model_dump() for n in value.connection_needs],
        }
        for n in a["nodes"]:
            if n["kind"] == "instructions":
                n["config"]["text"] = value.instructions
            if n["kind"] == "model":
                n["config"]["provider_id"] = req.provider_id
                n["config"]["role"] = value.role or library_roles.get(
                    value.worker_ref, n["config"].get("role", "general")
                )
            if n["kind"] == "output":
                n["config"]["format"] = value.output
        existing_tools = {
            n["config"].get("tool_id"): n for n in a["nodes"] if n["kind"] == "tool"
        }
        removed = {n["id"] for n in existing_tools.values()}
        a["nodes"] = [n for n in a["nodes"] if n["id"] not in removed]
        a["edges"] = [
            e
            for e in a["edges"]
            if e["source"] not in removed and e["target"] not in removed
        ]
        for j, tool in enumerate(dict.fromkeys(value.tools)):
            nid = (
                existing_tools.get(tool, {}).get("id")
                or "guided-tool-" + uuid.uuid4().hex[:8]
            )
            a["nodes"].append(
                {
                    "id": nid,
                    "kind": "tool",
                    "label": tool_labels.get(tool, tool),
                    "position": existing_tools.get(tool, {}).get(
                        "position", {"x": 420 + j * 250, "y": 420}
                    ),
                    "config": {"tool_id": tool},
                }
            )
            a["edges"].append(
                {"id": nid + "-edge", "source": nid, "target": body["id"]}
            )
        a = Blueprint.model_validate(a).model_dump()
        if value.id not in raw:
            initial = copy.deepcopy(a)
            if validate_blueprint(
                Blueprint.model_validate(initial),
                providers={req.provider_id: provider.payload},
            ):
                initial = seeds.agent(
                    a["id"],
                    value.name,
                    "",
                    "Complete the user's task accurately using the supplied context.",
                )
                next(n for n in initial["nodes"] if n["kind"] == "model")["config"][
                    "provider_id"
                ] = req.provider_id
            put(db, "published:" + binding["ref"], "published", initial)
        put(db, "draft:" + a["id"], "draft", a)
        bindings[value.id] = binding
        level = levels.get(value.id, i)
        row = rows.get(level, 0)
        rows[level] = row + 1
        nodes.append(
            {
                "id": value.id,
                "kind": "agent_ref",
                "label": value.name,
                "position": value.position
                or {"x": 310 + level * 290, "y": 160 + row * 145},
                "config": {"agent_ref": binding["ref"], "task": value.task},
            }
        )
    nodes.append(
        {
            "id": "result",
            "kind": "result",
            "label": "Final answer",
            "position": {"x": 310 + len(req.agents) * 280, "y": 160},
            "config": {},
        }
    )
    if req.name == "Untitled agent" and req.goal.strip():
        req.name = req.goal.strip().split("\n")[0][:64].rstrip(" .")
    bp.update(
        name=req.name,
        nodes=nodes,
        edges=edges
        if explicit
        else [
            {"id": f"e{i}", "source": nodes[i]["id"], "target": nodes[i + 1]["id"]}
            for i in range(len(nodes) - 1)
        ],
    )
    put(db, "draft:" + bp["id"], "draft", Blueprint.model_validate(bp).model_dump())
    meta.update(
        bindings=bindings,
        goal=req.goal,
        schedule=req.schedule.model_dump(),
        expected_result=req.expected_result,
        explicit_dependencies=explicit,
    )
    meta["requirements"] = req.requirements
    meta["output_checks"] = req.output_checks
    if req.deadline_seconds is not None:
        meta["deadline_seconds"] = req.deadline_seconds
    if req.workload_note is not None:
        meta["workload_note"] = req.workload_note
    put(db, "automation:" + meta["id"], "automation", meta)
    db.flush()


def create(timezone="America/Chicago"):
    """A new agent with one placeholder worker. The catalog installer
    (backend/demo_catalog.py) creates every agent this way, then writes its
    real design; agents are defined in the catalog, not over the API."""
    settings = Settings(timezone=timezone)
    id = "auto-" + uuid.uuid4().hex[:12]
    with store.Session.begin() as db:
        profiles = list(
            db.scalars(select(store.Document).where(store.Document.kind == "provider"))
        )
        profiles.sort(
            key=lambda d: (
                d.payload["id"] != "intel-toolkit",
                d.payload["adapter"] == "mock",
            )
        )
        profile = next(
            (
                d.payload["id"]
                for d in profiles
                if d.payload.get("supports_tools", True)
            ),
            "demo",
        )
        bp = {
            "schema_version": 1,
            "id": id,
            "kind": "workflow",
            "name": "Untitled agent",
            "description": "",
            "nodes": [
                {
                    "id": "start",
                    "kind": "trigger",
                    "label": "Start",
                    "position": {"x": 0, "y": 0},
                    "config": {},
                },
                {
                    "id": "result",
                    "kind": "result",
                    "label": "Result",
                    "position": {"x": 280, "y": 0},
                    "config": {},
                },
            ],
            "edges": [{"id": "e0", "source": "start", "target": "result"}],
        }
        meta = {
            "id": id,
            "blueprint_id": id,
            "bindings": {},
            "goal": "",
            "schedule": settings.model_dump(),
        }
        put(db, "draft:" + id, "draft", bp)
        put(db, "automation:" + id, "automation", meta)
        db.flush()
        current, _, _ = assemble(db, meta)
        placeholder = AgentInput(
            id="custom",
            name="General worker",
            worker_name="General worker",
            instructions="Complete the user's task accurately. State what you know and what you cannot verify.",
            tools=[],
        )
        request = Edit(
            revision=current["revision"],
            name=bp["name"],
            goal="",
            provider_id=profile,
            agents=[placeholder],
            schedule=settings,
        )
        write_design(db, meta, request)
        return assemble(db, meta)[0]


@router.get("")
def list_automations():
    with store.Session() as db:
        # Most recently updated first; seeded agents get no special placement.
        return [
            assemble(db, d.payload)[0]
            for d in db.scalars(
                select(store.Document)
                .where(store.Document.kind == "automation")
                .order_by(store.Document.updated.desc())
            )
            if not d.payload.get("archived")
        ]


@router.get("/{id}")
def get_automation(id: str):
    with store.Session() as db:
        return assemble(db, metadata(db, id).payload)[0]


def draft_snapshot(db, meta):
    from .tools import registered

    item, bp, raw = assemble(db, meta)
    require_editable(item)
    if not item["goal"].strip():
        raise ValueError("Add a goal before running this automation")
    providers = {
        d.payload["id"]: d.payload
        for d in db.scalars(
            select(store.Document).where(store.Document.kind == "provider")
        )
    }
    available_tools = registered(db)
    agents = {}
    for n in bp["nodes"]:
        if n["kind"] != "agent_ref":
            continue
        a = raw[n["id"]]
        errors = validate_blueprint(
            Blueprint.model_validate(a), providers=providers, tools=available_tools
        )
        if errors:
            raise ValueError(a["name"] + ": " + "; ".join(errors))
        ref = a["id"] + "@draft"
        n["config"]["agent_ref"] = ref
        agents[ref] = a
    errors = validate_blueprint(
        Blueprint.model_validate(bp),
        agents=agents,
        providers=providers,
        tools=available_tools,
    )
    if errors:
        raise ValueError("; ".join(errors))
    used = {
        n["config"]["provider_id"]
        for a in agents.values()
        for n in a["nodes"]
        if n["kind"] == "model"
    }
    extra = {}
    if meta.get("coordination"):
        from .planning import catalog

        refs = meta["coordination"]["worker_refs"]
        extra["coordination"] = {
            "workers": [w for w in catalog(db) if w["ref"] in refs],
            "provider_id": item["provider_id"],
            "max_tasks": meta["coordination"]["max_tasks"],
        }
    # Pin the registered tool specs this agent can reach - its tool nodes
    # plus the approved roster's worker tools - so runs and deployments keep
    # the definitions they were prepared with.
    used_tools = {
        n["config"].get("tool_id")
        for a in agents.values()
        for n in a["nodes"]
        if n["kind"] == "tool"
    }
    for w in extra.get("coordination", {}).get("workers", []):
        used_tools.update(w.get("tools", []))
    used_tools.discard(None)
    unknown_tools = sorted(
        t for t in used_tools if t not in TOOL_INFO and t not in available_tools
    )
    if unknown_tools:
        raise ValueError(
            "These tools are not installed or registered: " + ", ".join(unknown_tools)
        )
    extra["tools"] = {
        t: available_tools[t] for t in sorted(used_tools) if t in available_tools
    }
    if any(a.get("connection_needs") for a in item["agents"]):
        raise ValueError("Configure the declared worker connections before running.")
    if meta.get("coordination") and meta.get("execution_mode") != "authored":
        extra["required_steps"] = required_step_contract(item["agents"])
        extra["repeat_counts"] = {
            t["id"]: next(
                a["repeat_count"] for a in item["agents"] if a["id"] == t["id"]
            )
            for t in extra["required_steps"]
        }
    if meta.get("execution_mode") == "authored" and meta.get("coordination"):
        extra["authored_plan"] = {
            "summary": "Authored plan",
            "expected_result": meta.get("expected_result")
            or "Complete the stated goal.",
            "tasks": [
                {
                    "id": a["id"],
                    "name": a["name"],
                    "task": a["task"],
                    "expected_output": a.get("expected_output", ""),
                    "resource_notes": a.get("resource_notes", ""),
                    "requires": a.get("requires", []),
                    "uses_tools": a.get("uses_tools", []),
                    "worker_ref": a["worker_ref"],
                    "depends_on": a["depends_on"]
                    if a["depends_on"] is not None
                    else ([item["agents"][i - 1]["id"]] if i else []),
                }
                for i, a in enumerate(item["agents"])
            ],
            "missing_workers": [],
            "gaps": [],
        }
        extra["repeat_counts"] = {a["id"]: a["repeat_count"] for a in item["agents"]}
    return {
        **extra,
        "requirements": meta.get("requirements", []),
        # Authored plans pin their tools, so completion is verifiable for every
        # agent, not just seeded ones. Adaptive plans may legitimately use a
        # subset of the roster, so only declared demo tools are required there.
        # A demo spec may name the tools a correct run must use (a cache hit legitimately skips the search).
        "required_tools": meta["demo"]["required_tools"]
        if "required_tools" in meta.get("demo", {})
        else sorted({t for a in item["agents"] for t in a["tools"]})
        if item["execution_mode"] == "authored"
        else meta.get("demo", {}).get("tools", []),
        "output_checks": meta.get("output_checks", []),
        "goal": execution_goal(meta),
        "blueprint": bp,
        "agents": agents,
        "providers": {k: v for k, v in providers.items() if k in used},
    }


def execution_goal(meta):
    criteria = meta.get("expected_result", "")
    return meta["goal"] + ("\nExpected result: " + criteria if criteria else "")


@router.post("/{id}/run")
def run(id: str, req: RunInput):
    with store.Session() as db:
        meta = metadata(db, id).payload
        item, _, _ = assemble(db, meta)
        check_revision(item, req.revision)
        snapshot = draft_snapshot(db, meta)
        ref, goal = id + "@draft", execution_goal(meta)
    return enqueue(snapshot, goal, ref, execution={"automation_id": id})
