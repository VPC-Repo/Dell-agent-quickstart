"""Tool metadata for snapshots and validators: the built-in tools of xeon_runtime.tool_catalog."""

from __future__ import annotations


from sqlalchemy import select

from xeon_runtime.tool_catalog import TOOL_INFO

from . import store


def registered(db):
    """Registered tool specs by id."""
    return {
        d.payload["id"]: d.payload
        for d in db.scalars(select(store.Document).where(store.Document.kind == "tool"))
        if not d.payload.get("archived")
    }


def catalog_info(db):
    """Built-in and registered tool metadata for pickers and validators."""
    info = {k: dict(v) for k, v in TOOL_INFO.items()}
    for spec in registered(db).values():
        info[spec["id"]] = {
            "label": spec["label"],
            "capability": spec["capability"],
            "category": spec.get("category", "custom"),
            "description": spec["description"],
            "registered": True,
            "kind": spec["kind"],
            "connection": spec.get("connection"),
            "side_effects": spec.get("side_effects", False),
        }
    return info
