from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import psutil
from langgraph.types import Command
from sqlalchemy import delete, select, update

from xeon_runtime.engine import checkpoint_store, compile_workflow, run_agent
from xeon_runtime.memory import MemoryStore
from xeon_runtime.specs import Blueprint

RESOURCE_SAMPLE_SECONDS = float(os.getenv("RESOURCE_SAMPLE_SECONDS", "0.5"))

from . import store


def retry_safe(snapshot):
    """Automatic retry is safe only when re-running cannot repeat a side effect.

    Built-in tools are read-only fixtures. Registered tools declare their
    effects; a side-effecting, non-idempotent tool disables automatic recovery
    for runs that pin it.
    """
    return all(
        not (t.get("side_effects") and not t.get("idempotent", True))
        for t in (snapshot.get("tools") or {}).values()
    )


def execute_run(run_id, owner):
    with store.Session() as db:
        run = db.get(store.Run, run_id)
        snapshot, query, resume = run.snapshot, run.query, run.resume
    execution = snapshot.get("execution", {})
    policy = execution.get("policy", {})
    alive = threading.Event()

    def heartbeat():
        while not alive.wait(5):
            with store.Session.begin() as db:
                db.execute(
                    update(store.Run)
                    .where(
                        store.Run.id == run_id,
                        store.Run.lease_owner == owner,
                        store.Run.status == "running",
                    )
                    .values(lease_until=time.time() + 60)
                )

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()

    def emit(kind, payload):
        with store.Session() as db:
            current = db.get(store.Run, run_id)
            if current.status == "canceled" or current.lease_owner != owner:
                raise ValueError("Run canceled or worker lease superseded")
            budget = policy.get("token_budget", 0)
            if budget and kind in ("model_request", "model"):
                models = list(
                    db.scalars(
                        select(store.Event.payload).where(
                            store.Event.run_id == run_id, store.Event.kind == "model"
                        )
                    )
                )
                if any(not m.get("usage_available") for m in models):
                    raise ValueError(
                        "Token budget requires usage reporting from this provider"
                    )
                total = sum(m.get("total_tokens", 0) for m in models)
                if kind == "model_request" and total >= budget:
                    raise ValueError(
                        "Token budget reached before the next model request"
                    )
        store.event(run_id, kind, payload)
        if budget and kind == "model":
            if not payload.get("usage_available"):
                raise ValueError(
                    "Token budget requires usage reporting from this provider"
                )
            if total + payload["total_tokens"] > budget:
                raise ValueError(
                    "Token budget exceeded by the completed model response"
                )

    cpu_start = time.process_time()
    try:
        from xeon_runtime.version import runtime_digest

        identity = runtime_digest()
        if execution.get("runtime_sha256") and execution["runtime_sha256"] != identity:
            raise ValueError("The runtime differs from this snapshot. Queue a new run.")
        from xeon_runtime.planning import prepare_goal

        snapshot = prepare_goal(snapshot, query, emit)
        with store.Session.begin() as db:
            changed = db.execute(
                update(store.Run)
                .where(
                    store.Run.id == run_id,
                    store.Run.lease_owner == owner,
                    store.Run.status == "running",
                )
                .values(snapshot=snapshot)
            )
            if not changed.rowcount:
                raise ValueError("Run canceled or worker lease superseded")
        bp = Blueprint.model_validate(snapshot["blueprint"])
        emit(
            "started",
            {"name": bp.name, "resume": bool(resume), "runtime_sha256": identity},
        )
        url = os.getenv("CHECKPOINT_URL", "data/checkpoints.db")
        with checkpoint_store(url) as cp:
            cfg = {
                "configurable": {
                    "thread_id": run_id,
                    "capture_trace": execution.get("capture_trace", False),
                    "conversation_key": execution.get("conversation_key", ""),
                },
                "recursion_limit": 200,
            }
            dburl = store.Session.kw["bind"].url.render_as_string(hide_password=False)
            memory_url = os.getenv(
                "MEMORY_URL", dburl + ".memory" if dburl.startswith("sqlite") else dburl
            )
            memory = MemoryStore(memory_url)
            if bp.kind == "workflow":
                graph = compile_workflow(
                    bp,
                    snapshot["agents"],
                    snapshot["providers"],
                    emit,
                    cp,
                    memory,
                    cfg,
                    extra_tools=snapshot.get("tools"),
                )
                recovering = resume == 2 and bool(graph.get_state(cfg).values)
                inp = (
                    Command(resume=True)
                    if resume == 1
                    else None
                    if recovering
                    else {"query": query, "outputs": {}, "final": ""}
                )
                result = graph.invoke(inp, cfg)
                if result.get("__interrupt__"):
                    with store.Session.begin() as db:
                        db.execute(
                            update(store.Run)
                            .where(
                                store.Run.id == run_id,
                                store.Run.status == "running",
                                store.Run.lease_owner == owner,
                            )
                            .values(
                                status="waiting",
                                approval=result["__interrupt__"][0].value,
                                lease_until=0,
                            )
                        )
                    store.event(run_id, "waiting", {"message": "Approval required"})
                    return
                output = result.get("final", "")
            else:
                tested_task = (
                    snapshot.get("execution", {}).get("worker_test", {}).get("task")
                )
                step_emit = (
                    (
                        lambda kind, payload: emit(
                            kind,
                            {
                                **payload,
                                "task_id": tested_task["id"],
                                "task_name": tested_task["name"],
                            },
                        )
                    )
                    if tested_task
                    else emit
                )
                output = run_agent(
                    bp,
                    snapshot["providers"],
                    query,
                    step_emit,
                    cfg,
                    cp,
                    resume=resume == 2,
                    memory_store=memory,
                    extra_tools=snapshot.get("tools"),
                )
        with store.Session.begin() as db:
            budget = policy.get("token_budget", 0)
            if budget:
                observations = list(
                    db.scalars(
                        select(store.Event.payload).where(
                            store.Event.run_id == run_id, store.Event.kind == "model"
                        )
                    )
                )
                if any(not m.get("usage_available") for m in observations):
                    raise ValueError(
                        "Token budget requires usage reporting from this provider"
                    )
                if sum(m["total_tokens"] for m in observations) > budget:
                    raise ValueError(
                        "Token budget exceeded by completed parallel model responses"
                    )
            changed = db.execute(
                update(store.Run)
                .where(
                    store.Run.id == run_id,
                    store.Run.status == "running",
                    store.Run.lease_owner == owner,
                )
                .values(
                    status="completed",
                    output=output,
                    finished=time.time(),
                    lease_until=0,
                    approval=None,
                )
            )
        if changed.rowcount:
            store.event(run_id, "completed", {"message": "Output saved"})
    except Exception as exc:
        # Provider errors can embed credentials and prompt data. Keep public errors bounded.
        message = "Run failed. Check provider configuration, connectivity, output format and step limit."
        from langgraph.errors import GraphRecursionError

        if isinstance(exc, GraphRecursionError):
            message = "The agent reached its execution-step limit before producing a result. Review its tool requests and step instructions in Agent Creator."
        if isinstance(exc, ValueError) and str(exc).startswith(
            (
                "Set the server environment",
                "Inference API",
                "Response mapping",
                "Workflow approval",
                "Token budget",
                "Output",
                "Input",
                "Conversation",
                "Deployment runtime",
                "Planning",
            )
        ):
            message = str(exc)
        with store.Session.begin() as db:
            db.execute(
                update(store.Run)
                .where(
                    store.Run.id == run_id,
                    store.Run.status == "running",
                    store.Run.lease_owner == owner,
                )
                .values(
                    status="failed", error=message, finished=time.time(), lease_until=0
                )
            )
        store.event(
            run_id,
            "failed",
            {
                "message": message,
                "exception_type": type(exc).__name__,
                "retryable": str(exc).startswith(
                    (
                        "Inference API connection",
                        "Inference API returned HTTP 429",
                        "Inference API returned HTTP 5",
                    )
                ),
            },
        )
    finally:
        store.event(
            run_id,
            "cpu_segment",
            {
                "core_ms": round((time.process_time() - cpu_start) * 1000, 3),
                "scope": "builder run process; excludes external inference",
            },
        )
        alive.set()
        t.join(timeout=1)


def child_run(run_id, owner, database_url):
    # Spawn avoids inheriting pooled PostgreSQL connections or active HTTP threads.
    store.Session = store.configure_database(database_url)
    execute_run(run_id, owner)


def supervise_run(run_id, owner):
    with store.Session() as db:
        run = db.get(store.Run, run_id)
        policy = run.snapshot.get("execution", {}).get("policy", {})
        attempts = sum(
            1
            for _ in db.scalars(
                select(store.Event.id).where(
                    store.Event.run_id == run_id, store.Event.kind == "attempt"
                )
            )
        )
    maximum = policy.get("max_attempts", 1)
    # Approval continuation is part of the same attempt.
    if run.resume != 1:
        attempts += 1
        store.event(run_id, "attempt", {"number": attempts, "maximum": maximum})
    started = time.monotonic()
    timeout = policy.get("timeout_seconds", 600)
    if run.resume == 1:
        with store.Session() as db:
            last_attempt = (
                db.scalar(
                    select(store.Event.id)
                    .where(store.Event.run_id == run_id, store.Event.kind == "attempt")
                    .order_by(store.Event.id.desc())
                    .limit(1)
                )
                or 0
            )
            segments = list(
                db.scalars(
                    select(store.Event.payload).where(
                        store.Event.run_id == run_id,
                        store.Event.kind == "execution_segment",
                        store.Event.id > last_attempt,
                    )
                )
            )
        timeout -= sum(s["seconds"] for s in segments)
    url = store.Session.kw["bind"].url.render_as_string(hide_password=False)
    process = multiprocessing.get_context("spawn").Process(
        target=child_run, args=(run_id, owner, url)
    )
    process.start()
    last_sample = 0.0
    last_engines = 0.0
    engines: list = []
    from . import system as host_system

    sampler = host_system.SystemSampler()  # per-socket CPU, GPUs, engine metrics

    def sample_resources():
        # CPU, memory and GPUs every RESOURCE_SAMPLE_SECONDS (0.5 s: some agents finish in under ten seconds and a
        # model call can take 200 ms); the engines' Prometheus metrics are an HTTP call each, so every 2 s.
        nonlocal last_engines, engines
        try:
            proc = psutil.Process(process.pid)
            times = proc.cpu_times()
            cpu = sampler.sample_cpu()
            memory = sampler.sample_memory()
            gpus, gpu_source = host_system.read_gpus()
            if time.monotonic() - last_engines >= 2:
                engines = sampler.sample_engines()
                last_engines = time.monotonic()
            store.event(
                run_id,
                "resources",
                {
                    "scope": "run process",
                    "pid": process.pid,
                    "rss_bytes": proc.memory_info().rss,
                    "cpu_seconds": times.user + times.system,
                    "host_cpu_percent": cpu["percent"],
                    "host_memory_percent": memory["percent"],
                    "host_memory_used_bytes": memory["used_bytes"],
                    "host_memory_total_bytes": memory["total_bytes"],
                    "host_power_w": sampler.sample_power(),  # CPU package watts, None without RAPL
                    "sockets": [s["percent"] for s in cpu["sockets"]],
                    "gpus": [
                        {
                            "index": g["index"],
                            "utilization_percent": g["utilization_percent"],
                            "memory_used_mb": g["memory_used_mb"],
                            "memory_total_mb": g["memory_total_mb"],
                            "power_w": g["power_w"],
                        }
                        for g in gpus
                    ],
                    "gpu_source": gpu_source,
                    "engines": [
                        {
                            "model": e["model"],
                            "engine": e["engine"],
                            "device": e["device"],
                            "gpus": e["gpus"],
                            "reachable": e["reachable"],
                            "requests_running": e["requests_running"],
                            "requests_waiting": e["requests_waiting"],
                            "kv_cache_percent": e["kv_cache_percent"],
                            "generation_tokens_per_s": e["generation_tokens_per_s"],
                            "prompt_tokens_per_s": e["prompt_tokens_per_s"],
                        }
                        for e in engines
                    ],
                    "host_scope": "shared OS context; includes other workloads",
                },
            )
        except Exception:
            logging.exception("Resource sample failed; continuing")

    while process.is_alive():
        process.join(timeout=0.25)
        with store.Session() as db:
            current = db.get(store.Run, run_id)
            lost = current.lease_owner != owner or current.status == "canceled"
        expired = time.monotonic() - started > timeout
        if lost or expired:
            process.terminate()
            process.join(3)
            if process.is_alive():
                process.kill()
                process.join(3)
            if expired and not lost:
                with store.Session.begin() as db:
                    db.execute(
                        update(store.Run)
                        .where(
                            store.Run.id == run_id,
                            store.Run.lease_owner == owner,
                            store.Run.status == "running",
                        )
                        .values(
                            status="failed",
                            error="Run time limit exceeded",
                            finished=time.time(),
                            lease_until=0,
                        )
                    )
                store.event(
                    run_id,
                    "failed",
                    {"message": "Run time limit exceeded", "retryable": False},
                )
            return
        if time.monotonic() - last_sample >= RESOURCE_SAMPLE_SECONDS:
            sample_resources()
            last_sample = time.monotonic()
    process.join()
    sample_resources()  # one last measurement so the curves end on a sample, not a held value
    store.event(
        run_id, "execution_segment", {"seconds": round(time.monotonic() - started, 3)}
    )
    with store.Session.begin() as db:
        current = db.get(store.Run, run_id)
        if current.lease_owner != owner or current.status in (
            "completed",
            "waiting",
            "canceled",
        ):
            return
        failures = list(
            db.scalars(
                select(store.Event)
                .where(store.Event.run_id == run_id, store.Event.kind == "failed")
                .order_by(store.Event.id.desc())
                .limit(1)
            )
        )
        retryable = current.status == "running" or bool(
            failures and failures[0].payload.get("retryable")
        )
        if retryable and attempts < maximum and retry_safe(current.snapshot):
            current.status, current.resume = "queued", 2
            current.lease_until = time.time() + 2**attempts
            current.finished, current.error = None, ""
            db.add(
                store.Event(
                    run_id=run_id,
                    kind="retry",
                    payload={
                        "next_attempt": attempts + 1,
                        "delay_seconds": 2**attempts,
                    },
                )
            )
        elif current.status == "running":
            current.status, current.finished, current.lease_until = (
                "failed",
                time.time(),
                0,
            )
            current.error = "Worker process exited before saving a result"
            db.add(
                store.Event(
                    run_id=run_id, kind="failed", payload={"message": current.error}
                )
            )


class Worker:
    def __init__(self, concurrency=None):
        concurrency = (
            concurrency
            if concurrency is not None
            else int(os.getenv("WORKER_CONCURRENCY", "3"))
        )
        if not 1 <= concurrency <= 64:
            raise ValueError("Worker concurrency must be between 1 and 64")
        self.stop = threading.Event()
        self.concurrency = concurrency
        self.owner = uuid.uuid4().hex
        self.pool = ThreadPoolExecutor(max_workers=concurrency)
        self.futures = []
        self._heartbeat = 0.0
        self.last_maintenance = 0.0

    def heartbeat(self):
        # Lets /api/health report worker liveness; any live worker refreshes it.
        now = time.time()
        if now - self._heartbeat < 10:
            return
        self._heartbeat = now
        with store.Session.begin() as db:
            doc = db.get(store.Document, "runtime:worker-heartbeat")
            payload = {"ts": now, "owner": self.owner}
            if doc:
                doc.payload, doc.updated = payload, now
            else:
                db.add(
                    store.Document(
                        key="runtime:worker-heartbeat",
                        kind="heartbeat",
                        payload=payload,
                    )
                )

    def maintain(self):
        """Retention, once a minute: finished runs and their events age out after RETENTION_DAYS (0 keeps
        them)."""
        if time.time() - self.last_maintenance < 60:
            return
        self.last_maintenance = time.time()
        days = float(os.getenv("RETENTION_DAYS", "30"))
        if days <= 0:
            return
        cutoff = time.time() - days * 86400
        with store.Session.begin() as db:
            stale = (
                select(store.Run.id)
                .where(store.Run.finished.is_not(None))
                .where(store.Run.finished < cutoff)
            )
            db.execute(delete(store.Event).where(store.Event.run_id.in_(stale)))
            db.execute(
                delete(store.Run)
                .where(store.Run.finished.is_not(None))
                .where(store.Run.finished < cutoff)
            )

    def tick(self):
        self.heartbeat()
        self.futures = [f for f in self.futures if not f.done()]
        self.maintain()
        with store.Session.begin() as db:
            for abandoned in db.scalars(
                select(store.Run).where(
                    store.Run.status == "running", store.Run.lease_until < time.time()
                )
            ):
                attempts = len(
                    list(
                        db.scalars(
                            select(store.Event.id).where(
                                store.Event.run_id == abandoned.id,
                                store.Event.kind == "attempt",
                            )
                        )
                    )
                )
                maximum = (
                    abandoned.snapshot.get("execution", {})
                    .get("policy", {})
                    .get("max_attempts", 3)
                )
                if attempts < maximum and retry_safe(abandoned.snapshot):
                    abandoned.status, abandoned.resume, abandoned.lease_until = (
                        "queued",
                        2,
                        0,
                    )
                else:
                    abandoned.status, abandoned.finished, abandoned.error = (
                        "failed",
                        time.time(),
                        "Worker recovery attempt limit reached"
                        if attempts >= maximum
                        else "Automatic recovery is disabled: this run pins a side-effecting, non-idempotent tool",
                    )
            pending = list(
                db.scalars(
                    select(store.Run)
                    .where(
                        store.Run.status == "queued",
                        store.Run.lease_until <= time.time(),
                    )
                    .order_by(store.Run.created)
                    .limit(100)
                )
            )
            for run in pending:
                if len(self.futures) >= self.concurrency:
                    break
                claim_owner = self.owner + ":" + uuid.uuid4().hex
                claim = db.execute(
                    update(store.Run)
                    .where(store.Run.id == run.id, store.Run.status == "queued")
                    .values(
                        status="running",
                        started=run.started or time.time(),
                        lease_until=time.time() + 60,
                        lease_owner=claim_owner,
                    )
                )
                if claim.rowcount:
                    # Commit before the worker opens its own session.
                    db.commit()
                    self.futures.append(
                        self.pool.submit(supervise_run, run.id, claim_owner)
                    )
                    break  # one claim per tick keeps the transaction boundary explicit

    def loop(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception:
                logging.exception("Worker tick failed; retrying")
            self.stop.wait(0.5)
        self.pool.shutdown(wait=True)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    os.environ["LANGSMITH_TRACING"] = "false"
    store.initialize()
    worker = Worker()
    try:
        worker.loop()
    except KeyboardInterrupt:
        worker.stop.set()
