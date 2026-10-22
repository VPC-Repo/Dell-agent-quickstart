"""The worker library: every installed worker definition (backend/demo_catalog.py)."""

from sqlalchemy import select

from . import store


def catalog(db):
    return [
        d.payload
        for d in db.scalars(
            select(store.Document).where(store.Document.kind == "worker")
        )
        if not d.payload.get("archived")
    ]
