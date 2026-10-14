import time

import pytest
from sqlalchemy import create_engine, text

from backend import store


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "app.db"))
    store.initialize()
    return tmp_path / "app.db"


def stamped_revision(path):
    engine = create_engine("sqlite:///" + str(path))
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def test_fresh_database_is_stamped_at_head(fresh):
    assert stamped_revision(fresh) == store.SCHEMA_REVISION


def test_schema_revision_matches_migration_head():
    from pathlib import Path

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(store.__file__).resolve().parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    assert ScriptDirectory.from_config(config).get_current_head() == (
        store.SCHEMA_REVISION
    )


def test_pre_alembic_database_is_adopted_without_data_loss(tmp_path):
    # A database created by plain create_all (before Alembic) has the same
    # schema as the initial revision; startup stamps it and keeps its rows.
    url = "sqlite:///" + str(tmp_path / "old.db")
    engine = create_engine(url)
    store.Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO documents (key, kind, payload, updated)"
                " VALUES ('draft:x', 'draft', '{}', 1)"
            )
        )
    session = store.configure_database(url)
    assert stamped_revision(tmp_path / "old.db") == store.SCHEMA_REVISION
    with session() as db:
        assert db.get(store.Document, "draft:x") is not None


def test_payload_migrations_upgrade_documents_at_read_time(fresh, monkeypatch):
    def rename(p):
        p["renamed"] = p.pop("legacy")
        return p

    monkeypatch.setitem(store.PAYLOAD_VERSIONS, "unittest_kind", 3)
    monkeypatch.setitem(store.PAYLOAD_MIGRATIONS, ("unittest_kind", 1), rename)
    monkeypatch.setitem(
        store.PAYLOAD_MIGRATIONS,
        ("unittest_kind", 2),
        lambda p: {**p, "added": True},
    )
    with store.Session.begin() as db:
        db.add(
            store.Document(
                key="unittest:1", kind="unittest_kind", payload={"legacy": 7}
            )
        )
        db.add(
            store.Document(
                key="unittest:2",
                kind="unittest_kind",
                payload={"renamed": 8, "payload_version": 2},
            )
        )
    rows = {d["key"]: d["payload"] for d in store.documents("unittest_kind")}
    assert rows["unittest:1"] == {"renamed": 7, "added": True, "payload_version": 3}
    assert rows["unittest:2"] == {"renamed": 8, "added": True, "payload_version": 3}
    with store.Session() as db:
        assert store.load(db, "unittest:1")["payload_version"] == 3
        # The stored row is untouched until something writes it back.
        assert db.get(store.Document, "unittest:1").payload == {"legacy": 7}


def test_current_version_payloads_pass_through_unchanged(fresh):
    with store.Session.begin() as db:
        db.add(store.Document(key="plain:1", kind="plain_kind", payload={"a": 1}))
        db.add(
            store.Document(
                key="heartbeat:x",
                kind="heartbeat",
                payload={"ts": time.time(), "owner": "t"},
            )
        )
    payload = store.documents("plain_kind")[0]["payload"]
    assert payload == {"a": 1} and "payload_version" not in payload
