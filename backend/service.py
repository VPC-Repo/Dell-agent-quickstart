from __future__ import annotations

import uuid


from xeon_runtime.specs import Blueprint, validate_blueprint

from . import store


def inventory(include_archived=False):
    with store.Session() as db:
        config = db.get(store.Document, "catalog:demo")
        hidden = (
            set(config.payload.get("hidden_published", []))
            if config and not include_archived
            else set()
        )
    return {
        "drafts": store.documents("draft"),
        "published": [
            d for d in store.documents("published") if d["key"] not in hidden
        ],
        "providers": store.documents("provider"),
    }


def resolved(bp: Blueprint):
    from .tools import registered

    data = inventory(include_archived=True)
    agents = {
        d["key"].removeprefix("published:"): d["payload"]
        for d in data["published"]
        if d["payload"]["kind"] == "agent"
    }
    providers = {d["payload"]["id"]: d["payload"] for d in data["providers"]}
    with store.Session() as db:
        available_tools = registered(db)
    errors = validate_blueprint(bp, agents, providers, tools=available_tools)
    if bp.kind == "workflow":
        selected = {
            n.config.get("agent_ref") for n in bp.nodes if n.kind == "agent_ref"
        }
        agents = {k: v for k, v in agents.items() if k in selected}
        for agent in agents.values():
            errors.extend(
                validate_blueprint(
                    Blueprint.model_validate(agent),
                    providers=providers,
                    tools=available_tools,
                )
            )
    else:
        agents = {}
    used = set()
    used_tools = set()
    for b in [bp.model_dump(), *agents.values()]:
        used.update(
            n["config"].get("provider_id") for n in b["nodes"] if n["kind"] == "model"
        )
        used_tools.update(
            n["config"].get("tool_id") for n in b["nodes"] if n["kind"] == "tool"
        )
    used_tools.discard(None)
    providers = {k: v for k, v in providers.items() if k in used}
    return {
        "blueprint": bp.model_dump(),
        "agents": agents,
        "providers": providers,
        # Registered tool specs are pinned like providers: the run keeps the
        # definitions it was prepared with.
        "tools": {
            t: available_tools[t] for t in sorted(used_tools) if t in available_tools
        },
    }, list(dict.fromkeys(errors))


def snapshot_for_ref(ref):
    with store.Session() as db:
        doc = db.get(store.Document, "published:" + ref)
        if not doc:
            raise ValueError("Published version not found")
        bp = Blueprint.model_validate(doc.payload)
    snapshot, errors = resolved(bp)
    if errors:
        raise ValueError("; ".join(errors))
    return snapshot


def make_run(snapshot, query, ref="draft", **kwargs):
    modes = {v["adapter"] == "mock" for v in snapshot["providers"].values()}
    mode = "mock" if modes == {True} else "mixed" if len(modes) > 1 else "live"
    return store.Run(
        id=uuid.uuid4().hex,
        snapshot=snapshot,
        query=query,
        blueprint_ref=ref,
        name=snapshot["blueprint"]["name"],
        mode=mode,
        status="queued",
        **kwargs,
    )
