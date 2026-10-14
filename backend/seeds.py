"""Blueprint for one worker agent: body, instructions, model, output and an optional tool."""

from xeon_runtime.specs import Blueprint
from xeon_runtime.tool_catalog import TOOL_INFO


def agent(id, name, description, prompt, tool_id=None, deep=False):
    nodes = [
        {
            "id": "body",
            "kind": "agent",
            "label": name,
            "position": {"x": 420, "y": 190},
            "config": {
                "runtime": "deepagents" if deep else "langchain",
                "max_steps": 30,
            },
        },
        {
            "id": "instructions",
            "kind": "instructions",
            "label": "Instructions",
            "position": {"x": 60, "y": 60},
            "config": {"text": prompt},
        },
        {
            "id": "model",
            "kind": "model",
            "label": "Inference API",
            "position": {"x": 60, "y": 225},
            "config": {"provider_id": "demo"},
        },
        {
            "id": "output",
            "kind": "output",
            "label": "Briefing output",
            "position": {"x": 60, "y": 390},
            "config": {"format": "text"},
        },
    ]
    if tool_id:
        nodes.append(
            {
                "id": "tool",
                "kind": "tool",
                "label": TOOL_INFO[tool_id]["label"],
                "position": {"x": 420, "y": 420},
                "config": {"tool_id": tool_id},
            }
        )
    return Blueprint.model_validate(
        {
            "id": id,
            "kind": "agent",
            "name": name,
            "description": description,
            "nodes": nodes,
            "edges": [
                {"id": f"{n['id']}-body", "source": n["id"], "target": "body"}
                for n in nodes
                if n["id"] != "body"
            ],
        }
    ).model_dump()
