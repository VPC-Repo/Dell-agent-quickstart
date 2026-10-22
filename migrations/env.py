import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import store  # noqa: E402

target_metadata = store.Base.metadata


def database_url():
    return (
        context.config.get_main_option("sqlalchemy.url")
        or os.getenv("DATABASE_URL", "sqlite:///data/builder.db")
    )


def run_migrations_offline():
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    # The application passes its already-configured engine; the CLI builds one.
    engine = context.config.attributes.get("engine") or create_engine(database_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
