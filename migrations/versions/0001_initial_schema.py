"""Initial schema: documents, runs, events, schedules.

Matches backend/store.py at adoption time. Databases created before Alembic
carry this exact schema already, so startup stamps them at this revision
instead of replaying it.

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "documents",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("updated", sa.Float()),
    )
    op.create_index("ix_documents_kind", "documents", ["kind"])
    op.create_table(
        "runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("status", sa.String()),
        sa.Column("blueprint_ref", sa.String()),
        sa.Column("name", sa.String()),
        sa.Column("query", sa.Text()),
        sa.Column("snapshot", sa.JSON()),
        sa.Column("mode", sa.String()),
        sa.Column("created", sa.Float()),
        sa.Column("started", sa.Float(), nullable=True),
        sa.Column("finished", sa.Float(), nullable=True),
        sa.Column("output", sa.Text()),
        sa.Column("error", sa.Text()),
        sa.Column("approval", sa.JSON(), nullable=True),
        sa.Column("resume", sa.Integer()),
        sa.Column("lease_until", sa.Float()),
        sa.Column("lease_owner", sa.String()),
        sa.Column("schedule_id", sa.String(), nullable=True),
        sa.Column("occurrence", sa.String(), nullable=True, unique=True),
    )
    op.create_index("ix_runs_status", "runs", ["status"])
    op.create_index("ix_runs_schedule_id", "runs", ["schedule_id"])
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String()),
        sa.Column("created", sa.Float()),
        sa.Column("kind", sa.String()),
        sa.Column("payload", sa.JSON()),
    )
    op.create_index("ix_events_run_id", "events", ["run_id"])
    op.create_index("ix_events_kind_created", "events", ["kind", "created"])
    op.create_table(
        "schedules",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("blueprint_ref", sa.String()),
        sa.Column("name", sa.String()),
        sa.Column("query", sa.Text()),
        sa.Column("cron", sa.String()),
        sa.Column("timezone", sa.String()),
        sa.Column("overlap", sa.String()),
        sa.Column("max_parallel", sa.Integer()),
        sa.Column("paused", sa.Integer()),
        sa.Column("next_fire", sa.Float()),
        sa.Column("last_action", sa.String()),
    )
    op.create_index("ix_schedules_next_fire", "schedules", ["next_fire"])


def downgrade():
    op.drop_table("schedules")
    op.drop_table("events")
    op.drop_table("runs")
    op.drop_table("documents")
