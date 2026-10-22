"""A three-worker daily-briefing workflow on the mock provider, for tests only.

It runs without a model service (the built-in search_knowledge and analyze_metrics tools and the mock adapter), so
the runtime, worker and API tests exercise a complete run. It is not part of the shipped catalog.
"""

from backend import store
from backend.seeds import agent


def seed_agents():
    return [
        agent(
            "research",
            "Research agent",
            "Find the facts. Keep the sources.",
            "Research the supplied daily briefing objective. Search the knowledge corpus, cite source IDs, distinguish sample data from measurements, and summarize findings.",
            "search_knowledge",
            True,
        ),
        agent(
            "analysis",
            "Analysis agent",
            "Turn operational data into findings.",
            "Analyze the provided context and sample metrics. Use analyze_metrics. Explain changes and risks. Label synthetic data clearly.",
            "analyze_metrics",
        ),
        agent(
            "writer",
            "Briefing writer",
            "A clear report with accountable next steps.",
            "Write a concise daily briefing using the supplied research and analysis. Include a summary, changes, risks and next actions. Preserve source IDs. Do not invent facts.",
        ),
    ]


def seed_blueprints():
    workers = seed_agents()
    nodes = [
        {
            "id": "start",
            "kind": "trigger",
            "label": "Daily briefing",
            "position": {"x": 30, "y": 160},
            "config": {},
        },
        {
            "id": "research",
            "kind": "agent_ref",
            "label": "Research",
            "position": {"x": 310, "y": 160},
            "config": {
                "agent_ref": "research@1",
                "task": "Gather the latest operational facts and sources.",
            },
        },
        {
            "id": "analysis",
            "kind": "agent_ref",
            "label": "Analyze",
            "position": {"x": 590, "y": 160},
            "config": {
                "agent_ref": "analysis@1",
                "task": "Analyze changes, metrics and risks.",
            },
        },
        {
            "id": "writer",
            "kind": "agent_ref",
            "label": "Write briefing",
            "position": {"x": 870, "y": 160},
            "config": {
                "agent_ref": "writer@1",
                "task": "Produce a sourced briefing with next actions.",
            },
        },
        {
            "id": "result",
            "kind": "result",
            "label": "Briefing ready",
            "position": {"x": 1150, "y": 160},
            "config": {},
        },
    ]
    flow = {
        "schema_version": 1,
        "id": "daily-briefing",
        "kind": "workflow",
        "name": "Daily briefing",
        "description": "Research → analysis → a sourced daily report.",
        "nodes": nodes,
        "edges": [
            {"id": f"e{i}", "source": nodes[i]["id"], "target": nodes[i + 1]["id"]}
            for i in range(len(nodes) - 1)
        ],
    }
    return workers + [flow]


def install():
    """Store the briefing blueprints as drafts and published@1. Idempotent."""

    with store.Session.begin() as db:
        for bp in seed_blueprints():
            key = "draft:" + bp["id"]
            if db.get(store.Document, key) is None:
                db.add(store.Document(key=key, kind="draft", payload=bp))
                db.add(
                    store.Document(
                        key="published:" + bp["id"] + "@1",
                        kind="published",
                        payload=bp,
                    )
                )
