from __future__ import annotations

import hmac
import os
import threading
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "false"

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import store, system
from .automations import router as automation_router
from .worker import Worker


@asynccontextmanager
async def lifespan(app):
    store.initialize()
    worker = Worker()
    thread = None
    if os.getenv("EMBEDDED_WORKER", "1") == "1":
        thread = threading.Thread(target=worker.loop, daemon=True)
        thread.start()
    if os.getenv("SYSTEM_SAMPLER", "1") == "1":
        system.start()
    yield
    worker.stop.set()
    system.close()
    if thread:
        thread.join(timeout=3)


app = FastAPI(title="Dell Agent Quickstart", version="0.1.0", lifespan=lifespan)


def auth_disabled():
    """BUILDER_AUTH=off: an operator-chosen open deployment, for a listener
    that is already private (bound to the host's loopback and reached
    through an SSH tunnel). Every client is served without a token."""
    return os.getenv("BUILDER_AUTH", "").strip().lower() == "off"


def authorize(request: Request, authorization: str = Header(default="")):
    if auth_disabled():
        return
    expected = os.getenv("BUILDER_API_TOKEN", "")
    if expected:
        if not hmac.compare_digest(authorization, "Bearer " + expected):
            raise HTTPException(401, "Enter the access token")
        return
    if os.getenv("BUILDER_AUTH", "").strip().lower() == "token":
        raise HTTPException(503, "BUILDER_AUTH=token requires BUILDER_API_TOKEN")
    # Without an explicit mode or token, permit loopback development clients.
    host = request.client.host if request.client else None
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(
            401, "Set BUILDER_API_TOKEN before serving non-local clients"
        )


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    database, worker_age = True, None
    try:
        with store.Session() as db:
            beat = db.get(store.Document, "runtime:worker-heartbeat")
            if beat:
                worker_age = round(max(0.0, time.time() - beat.payload.get("ts", 0)), 1)
    except Exception:
        database = False
    return {
        "status": "ok" if database else "degraded",
        "auth_required": not auth_disabled()
        and (
            bool(os.getenv("BUILDER_API_TOKEN"))
            or os.getenv("BUILDER_AUTH", "").strip().lower() == "token"
        ),
        "database": database,
        "worker_seconds_since_tick": worker_age,
    }


@app.get("/api/system", dependencies=[Depends(authorize)])
def system_snapshot():
    """Host CPU per socket, memory, GPUs, containers and inference engines: the
    latest sample from the background sampler (see backend/system.py)."""
    data = system.current()
    return data


@app.get("/api/catalog", dependencies=[Depends(authorize)])
def catalog_status():
    """What the gallery needs before showing Run buttons: whether the example
    agents are installed and which connected (non-mock) providers exist."""
    from .demo_catalog import AGENTS, VERSION

    with store.Session() as db:
        installed = [
            d.payload["id"]
            for d in db.scalars(
                select(store.Document).where(store.Document.kind == "automation")
            )
            if d.payload.get("demo") and not d.payload.get("archived")
        ]
        providers = [
            {
                "id": d.payload["id"],
                "name": d.payload["name"],
                "roles": d.payload.get("roles", {}),
            }
            for d in db.scalars(
                select(store.Document).where(store.Document.kind == "provider")
            )
            if d.payload.get("adapter") != "mock"
        ]
    return {
        "version": VERSION,
        "available": len(AGENTS),
        "installed": len(installed),
        "providers": providers,
    }


class CatalogInstall(BaseModel):
    provider_id: str = Field(default="", max_length=80)


@app.post("/api/catalog/install", dependencies=[Depends(authorize)])
def catalog_install(req: CatalogInstall):
    from .demo_catalog import install

    provider_id = req.provider_id
    if not provider_id:
        with store.Session() as db:
            connected = sorted(
                (
                    d.payload["id"]
                    for d in db.scalars(
                        select(store.Document).where(store.Document.kind == "provider")
                    )
                    if d.payload.get("adapter") != "mock"
                ),
                key=lambda id: id != "intel-toolkit",
            )
        if not connected:
            raise ValueError(
                "Connect a model service before installing the example agents"
            )
        provider_id = connected[0]
    return {"installed": install(provider_id), "provider_id": provider_id}


@app.get("/api/runs", dependencies=[Depends(authorize)])
def runs(
    limit: int = Query(default=100, ge=1, le=200),
    before: float | None = None,
    status: str | None = Query(default=None, max_length=40),
):
    # Newest first; page by passing the last row's `created` as `before`.
    query = select(store.Run).order_by(store.Run.created.desc()).limit(limit)
    if before is not None:
        query = query.where(store.Run.created < before)
    if status:
        query = query.where(store.Run.status == status)
    with store.Session() as db:
        return [store.serialize(r) for r in db.scalars(query)]


@app.get("/api/runs/{id}", dependencies=[Depends(authorize)])
def get_run(id: str):
    with store.Session() as db:
        run = db.get(store.Run, id)
        if not run:
            raise HTTPException(404, "Run not found")
        events = [
            store.serialize(e)
            for e in db.scalars(
                select(store.Event)
                .where(store.Event.run_id == id)
                .order_by(store.Event.id)
            )
        ]
        from xeon_runtime.readiness import assess_output

        outcome = assess_output(
            run.status,
            run.output or "",
            events,
            run.snapshot.get("output_checks", []),
            run.snapshot.get("required_tools", []),
        )
        # server_time lets the UI measure live elapsed time on the server's clock, not the browser's.
        return {
            **store.serialize(run),
            "events": events,
            "outcome": outcome,
            "server_time": time.time(),
        }


@app.post("/api/runs/{id}/{action}", dependencies=[Depends(authorize)])
def run_action(id: str, action: str):
    with store.Session.begin() as db:
        run = db.get(store.Run, id)
        if not run:
            raise HTTPException(404, "Run not found")
        if action == "cancel" and run.status in ("queued", "running", "waiting"):
            run.status, run.finished = "canceled", time.time()
        else:
            raise ValueError("Action is unavailable for this run state")
    return {"status": run.status}


app.include_router(automation_router, dependencies=[Depends(authorize)])

if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
