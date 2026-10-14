"""Explicit, serialized conversation memory, separate from run checkpoints."""

import hashlib
from contextlib import contextmanager

from sqlalchemy import (
    JSON,
    Column,
    MetaData,
    String,
    Table,
    create_engine,
    select,
    text,
)


class MemoryStore:
    def __init__(self, url="sqlite:///data/memory.db"):
        self.engine = create_engine(
            url,
            connect_args={"timeout": 600} if url.startswith("sqlite") else {},
            pool_pre_ping=True,
        )
        metadata = MetaData()
        self.table = Table(
            "agent_conversations",
            metadata,
            Column("id", String, primary_key=True),
            Column("turns", JSON, nullable=False),
        )
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "postgresql":
                connection.execute(text("SELECT pg_advisory_xact_lock(721864901)"))
            metadata.create_all(connection)

    @contextmanager
    def conversation(self, blueprint, key, max_turns):
        # Key on the blueprint's stable identity, never its serialized content:
        # editing instructions or a node's layout must not
        # silently orphan the conversation history.
        digest = hashlib.sha256((blueprint.id + "|" + key).encode()).hexdigest()
        with self.engine.connect() as db:
            if self.engine.dialect.name == "sqlite":
                db.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                # A transaction-scoped lock also serializes creation of a new conversation.
                lock = int(digest[:15], 16)
                db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock})
            row = db.execute(
                select(self.table.c.turns).where(self.table.c.id == digest)
            ).first()
            turns = list(row[0] if row else [])
            try:
                yield turns
                value = turns[-max_turns:]
                if row:
                    db.execute(
                        self.table.update()
                        .where(self.table.c.id == digest)
                        .values(turns=value)
                    )
                else:
                    db.execute(self.table.insert().values(id=digest, turns=value))
                db.commit()
            except BaseException:
                db.rollback()
                raise
