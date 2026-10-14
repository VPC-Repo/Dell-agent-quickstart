from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    key = Column(String, primary_key=True)
    kind = Column(String, nullable=False, index=True)
    payload = Column(JSON, nullable=False)
    updated = Column(Float, default=time.time)


class Run(Base):
    __tablename__ = "runs"
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    status = Column(String, default="queued", index=True)
    blueprint_ref = Column(String)
    name = Column(String)
    query = Column(Text)
    snapshot = Column(JSON)
    mode = Column(String)
    created = Column(Float, default=time.time)
    started = Column(Float, nullable=True)
    finished = Column(Float, nullable=True)
    output = Column(Text, default="")
    error = Column(Text, default="")
    approval = Column(JSON, nullable=True)
    resume = Column(Integer, default=0)
    lease_until = Column(Float, default=0)
    lease_owner = Column(String, default="")
    schedule_id = Column(String, nullable=True, index=True)
    occurrence = Column(String, unique=True, nullable=True)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (Index("ix_events_kind_created", "kind", "created"),)
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    created = Column(Float, default=time.time)
    kind = Column(String)
    payload = Column(JSON)


class Schedule(Base):
    __tablename__ = "schedules"
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    blueprint_ref = Column(String)
    name = Column(String)
    query = Column(Text)
    cron = Column(String)
    timezone = Column(String)
    overlap = Column(String)
    max_parallel = Column(Integer, default=1)
    paused = Column(Integer, default=0)
    next_fire = Column(Float, index=True)
    last_action = Column(String, default="")


# Head of migrations/versions. A test keeps this equal to the Alembic script
# head; the constant lets run processes skip importing Alembic entirely when
# the database is already current.
SCHEMA_REVISION = "0001"


def _alembic_config(engine):
    from alembic.config import Config

    root = Path(__file__).resolve().parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    config.attributes["engine"] = engine
    return config


def _apply_schema(engine):
    from sqlalchemy import inspect, text

    if "alembic_version" in inspect(engine).get_table_names():
        with engine.connect() as conn:
            current = conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar()
        if current != SCHEMA_REVISION:
            from alembic import command

            command.upgrade(_alembic_config(engine), "head")
        return
    # Fresh database, or one created before Alembic adoption: both carry the
    # initial-revision schema after create_all, so stamp instead of replaying.
    Base.metadata.create_all(engine)
    for table in Base.metadata.tables.values():
        for index in table.indexes:
            index.create(engine, checkfirst=True)
    from alembic import command

    try:
        command.stamp(_alembic_config(engine), "head")
    except Exception:
        # Concurrent processes race to stamp; accept whichever write landed.
        with engine.connect() as conn:
            current = conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar()
        if current != SCHEMA_REVISION:
            raise


def configure_database(url=None):
    Path("data").mkdir(exist_ok=True)
    url = url or os.getenv("DATABASE_URL", "sqlite:///data/builder.db")
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 30}
        if url.startswith("sqlite")
        else {},
        pool_pre_ping=True,
    )
    _apply_schema(engine)
    return sessionmaker(engine, expire_on_commit=False)


Session = None


def initialize(url=None):
    global Session
    Session = configure_database(url)
    from xeon_runtime.specs import Provider

    # Infrastructure only: a mock provider for tests and offline checks. The agents install
    # through the catalog (backend/demo_catalog.py), never unconditionally at startup.
    with Session.begin() as db:
        if db.get(Document, "provider:demo") is None:
            db.add(
                Document(
                    key="provider:demo",
                    kind="provider",
                    payload=Provider(
                        id="demo", name="Demo • no API calls", adapter="mock"
                    ).model_dump(),
                )
            )


# Document payloads carry an implicit schema version: a missing
# "payload_version" field means 1. To change a kind's stored shape, raise its
# entry in PAYLOAD_VERSIONS and register one step function per version in
# PAYLOAD_MIGRATIONS; readers that load through documents()/load() upgrade old
# rows at read time, so existing databases never break. Kinds validated by
# strict runtime specs (draft, published, provider) must keep the field out of
# their payloads until the spec models accept it.
PAYLOAD_VERSIONS: dict[str, int] = {}
PAYLOAD_MIGRATIONS: dict[tuple[str, int], object] = {}


def upgraded(kind, payload):
    current = PAYLOAD_VERSIONS.get(kind, 1)
    version = payload.get("payload_version", 1)
    while version < current:
        payload = PAYLOAD_MIGRATIONS[(kind, version)](dict(payload))
        version += 1
        payload["payload_version"] = version
    return payload


def load(db, key):
    doc = db.get(Document, key)
    return upgraded(doc.kind, doc.payload) if doc else None


def documents(kind):
    with Session() as db:
        return [
            {"key": d.key, "updated": d.updated, "payload": upgraded(kind, d.payload)}
            for d in db.scalars(select(Document).where(Document.kind == kind))
            if not d.payload.get("archived")
        ]


def serialize(obj):
    return {
        c.name: getattr(obj, c.name)
        for c in obj.__table__.columns
        if c.name not in ("snapshot", "lease_owner")
    }


def event(run_id, kind, payload):
    with Session.begin() as db:
        db.add(Event(run_id=run_id, kind=kind, payload=payload))
