"""Run-scoped file workspace shared by an agent's workers.

Steps exchange references to named artifacts instead of inlining large
content into prompts: one worker saves a file, later workers read it by
name, and the whole workspace lives under one directory per run so
operators can inspect or collect results.

Names are single path segments; traversal is rejected. Text only, bounded
size. The workspace root comes from ARTIFACT_DIR (default data/artifacts).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
BYTE_LIMIT = 2_000_000
READ_LIMIT = 24000  # characters returned into model context


def workspace(config) -> Path:
    thread = str(((config or {}).get("configurable") or {}).get("thread_id", ""))
    # Workflow sub-agents run as "<run_id>:<node_id>"; share one run workspace.
    run_id = re.sub(r"[^A-Za-z0-9_-]", "_", thread.split(":")[0]) or "scratch"
    root = Path(os.getenv("ARTIFACT_DIR", "data/artifacts")) / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _target(config, name):
    if not NAME.fullmatch(name or ""):
        raise ValueError(
            "Artifact names are single file names: letters, digits, dot, dash, underscore"
        )
    return workspace(config) / name


@tool
def save_artifact(name: str, content: str, config: RunnableConfig) -> str:
    """Save named text content to this run's shared artifact workspace so later steps can read it by name."""
    if len(content.encode()) > BYTE_LIMIT:
        raise ValueError("Artifact exceeds the 2 MB text limit")
    path = _target(config, name)
    path.write_text(content)
    return json.dumps({"saved": name, "bytes": path.stat().st_size})


@tool
def read_artifact(name: str, config: RunnableConfig) -> str:
    """Read a named text artifact from this run's shared workspace."""
    path = _target(config, name)
    if not path.is_file():
        raise ValueError("No artifact named " + name + " exists in this run")
    text = path.read_text()
    return json.dumps(
        {
            "name": name,
            "bytes": path.stat().st_size,
            "truncated": len(text) > READ_LIMIT,
            "content": text[:READ_LIMIT],
        }
    )


@tool
def list_artifacts(config: RunnableConfig) -> str:
    """List the artifacts saved in this run's shared workspace."""
    root = workspace(config)
    return json.dumps(
        [
            {"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(root.iterdir())
            if p.is_file()
        ]
    )


ARTIFACT_TOOLS = [save_artifact, read_artifact, list_artifacts]
