"""The CPU jobs behind the demo tools: seeded records, processed for real, in one process or a pool of processes.

Standard library only, on purpose: a pool's worker processes import this module to run a chunk, and a heavy
import (the tools module imports LangChain) would cost every worker about a second of CPU before its first chunk.
The tools that call these jobs are in demo_tools.py.
"""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
import random
import sys
import threading
import time
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal


# Jobs within one process (one agent run) take turns: each already fills every core.
_CPU_SLOT = threading.Lock()

# Records per shard, sized for roughly fifteen to twenty seconds on one Xeon core.
PER_SHARD = {"transactions": 1500000, "logs": 1500000}
# Shard boundaries stay multiples of this so the seeded patterns (four services
# round-robin) never straddle a shard.
ALIGN = 40


def shard_count() -> int:
    """One shard per CPU the process may use; `CPU_JOB_SHARDS` overrides (tests, small hosts)."""
    override = os.getenv("CPU_JOB_SHARDS")
    if override:
        return max(1, int(override))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _ranges(count: int, shards: int) -> list[tuple[int, int]]:
    shards = max(1, min(shards, max(1, count // ALIGN)))
    step = -(-count // shards)
    step += (-step) % ALIGN
    return [(start, min(count, start + step)) for start in range(0, count, step)]


def _run_shard(args):
    fn, shard, start, stop = args
    cpu = time.process_time()
    partial = fn(shard, start, stop)
    partial["_cpu_seconds"] = time.process_time() - cpu
    # The worker process's whole CPU so far, its start-up included: a pool counts each process once, by its last task.
    partial["_pid"], partial["_process_cpu"] = os.getpid(), time.process_time()
    return partial


def _context():
    # forkserver: children are forked from a server that has already imported this module, without the threads of
    # the calling process. Spawn where forkserver is unavailable.
    if sys.platform.startswith("linux"):
        context = multiprocessing.get_context("forkserver")
        context.set_forkserver_preload([__name__])
        return context
    return multiprocessing.get_context("spawn")


def _pool(fn, tasks, workers):
    """Run the tasks on a pool; partials and the CPU of every pool process, start-up included."""
    # ProcessPoolExecutor raises BrokenProcessPool if a child dies instead of respawning it forever.
    with ProcessPoolExecutor(
        max_workers=min(len(tasks), workers or len(tasks)), mp_context=_context()
    ) as pool:
        partials = list(pool.map(_run_shard, tasks))
    last = {}
    for p in partials:
        last[p["_pid"]] = max(last.get(p["_pid"], 0.0), p["_process_cpu"])
    for p in partials:
        p.pop("_cpu_seconds"), p.pop("_pid"), p.pop("_process_cpu")
    return partials, sum(last.values())


def _pool_in_helper(fn, count, shards, workers):
    """The pool, run from a helper process whose main module is this one.

    Every pool process re-imports its parent's main module before its first chunk. In a run that parent is the
    runner's worker (`python -m backend.worker`: LangGraph, SQLAlchemy, FastAPI), about a second of CPU per process:
    on 344 CPUs that was 300 core-seconds or more for a job of 25. The helper's main module is this
    standard-library one, so its pool processes start in milliseconds. The CPU is the helper's own (the resource
    usage of this process's children) plus every pool process's, start-up included, as the pool measured it (the
    pool processes are the forkserver's children, which the helper does not wait for)."""
    import json
    import resource
    import subprocess

    root = str(Path(__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    done = subprocess.run(
        [sys.executable, "-m", __name__],
        input=json.dumps(
            {"fn": fn.__name__, "count": count, "shards": shards, "workers": workers}
        ),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError("CPU job helper failed: " + done.stderr[-800:])
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    helper = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    result = json.loads(done.stdout)
    return result["partials"], helper + result["pool_cpu"]


def run_sharded(
    fn, count: int, shards: int, workers: int | None = None
) -> tuple[list[dict], float]:
    """Run fn(shard, start, stop) over `count` records in `shards` chunks, one process per chunk or a pool of
    `workers` processes; partials and CPU seconds."""
    tasks = [(fn, i, a, b) for i, (a, b) in enumerate(_ranges(count, shards))]
    if len(tasks) == 1 or workers == 1:
        partials = [_run_shard(t) for t in tasks]
    elif os.name == "posix":
        return _pool_in_helper(fn, count, shards, workers)
    else:
        return _pool(fn, tasks, workers)
    for p in partials:
        p.pop("_pid"), p.pop("_process_cpu")
    return partials, sum(p.pop("_cpu_seconds") for p in partials)


def measured(name, fn):
    wall = time.perf_counter()
    with _CPU_SLOT:
        waited = time.perf_counter() - wall
        result, units, cpu_seconds = fn()
    result["work_evidence"] = {
        "tool": name,
        "data": "demo data, reproducible on every install",
        "units": units,
        "cpu_seconds": round(cpu_seconds, 4),
        "wall_seconds": round(time.perf_counter() - wall, 4),
        "processing_slot_wait_seconds": round(waited, 4),
        "scope": "All shard processes of this tool call, including fixture generation; excludes inference and coordination",
        # One shard runs inside the run process (its CPU is already in the run's cpu_segment); more run as child processes.
        "in_run_process": units.get("processes") == 1,
    }
    return json.dumps(result)


# -- transactions ----------------------------------------------------------------
def _transactions_shard(shard, start, stop):
    rng = random.Random(470 + shard)
    exceptions, totals = Counter(), defaultdict(Decimal)
    examples = []
    for i in range(start, stop):
        # Parse each pair exactly as an exported record would be parsed.
        amount = rng.randrange(500, 500000)
        delta = 2500 if i % 997 == 0 else 0
        row = json.loads(
            json.dumps(
                {
                    "invoice": f"INV-{i:09d}",
                    "account": f"ACCT-{i % 240:03d}",
                    "due": f"{amount / 100:.2f}",
                    "paid": f"{(amount - delta) / 100:.2f}",
                }
            )
        )
        balance = Decimal(row["due"]) - Decimal(row["paid"])
        totals[row["account"]] += balance
        if balance:
            exceptions[row["account"]] += 1
            if len(examples) < 3:
                examples.append(
                    {"invoice": row["invoice"], "difference_usd": str(balance)}
                )
    return {
        "exceptions": dict(exceptions),
        "totals": {k: str(v) for k, v in totals.items()},
        "examples": examples,
    }


def transactions_job(count=3000000, shards=1):
    partials, cpu = run_sharded(_transactions_shard, count, shards)
    exceptions, totals = Counter(), defaultdict(Decimal)
    for p in partials:
        exceptions.update(p["exceptions"])
        for k, v in p["totals"].items():
            totals[k] += Decimal(v)
    return (
        {
            "invoice_pairs": count,
            "shards": len(partials),
            "exceptions": sum(exceptions.values()),
            "unsettled_usd": str(sum(totals.values())),
            "top_accounts": [
                {"account": k, "exceptions": v} for k, v in exceptions.most_common(3)
            ],
            "examples": partials[0]["examples"],
            "finding": "Every flagged invoice is underpaid by 25.00 USD; investigate settlement adjustments.",
        },
        {"invoice_pairs": count, "records": count * 2, "processes": len(partials)},
        cpu,
    )


# -- logs ----------------------------------------------------------------------------
SERVICES = ["checkout", "catalog", "identity", "fulfillment"]


def _logs_shard(shard, start, stop):
    rng = random.Random(470 + shard)
    stats = {s: {"events": 0, "errors": 0, "latency_ms": Counter()} for s in SERVICES}
    for i in range(start, stop):
        service = SERVICES[i % 4]
        incident = service == "checkout" and (i // 4) % 100 < 18
        record = json.loads(
            json.dumps(
                {
                    "service": service,
                    "status": 503 if incident else 200,
                    "latency_ms": round(
                        rng.lognormvariate(4.2 if not incident else 6.1, 0.35), 2
                    ),
                    "minute": i % 1440,
                }
            )
        )
        s = stats[record["service"]]
        s["events"] += 1
        s["errors"] += record["status"] >= 500
        # A millisecond histogram keeps every shard's percentile input small.
        s["latency_ms"][int(record["latency_ms"])] += 1
    return {
        name: {
            "events": s["events"],
            "errors": s["errors"],
            "latency_ms": dict(s["latency_ms"]),
        }
        for name, s in stats.items()
    }


def _percentile(histogram: Counter, q: float) -> int:
    total = sum(histogram.values())
    target = min(total - 1, int(q * total))
    seen = 0
    for value in sorted(histogram):
        seen += histogram[value]
        if seen > target:
            return value
    return 0


def logs_job(count=3000000, shards=1, workers=None):
    partials, cpu = run_sharded(_logs_shard, count, shards, workers)
    rows = []
    for name in SERVICES:
        events = sum(p[name]["events"] for p in partials)
        errors = sum(p[name]["errors"] for p in partials)
        histogram = Counter()
        for p in partials:
            histogram.update({int(k): v for k, v in p[name]["latency_ms"].items()})
        rows.append(
            {
                "service": name,
                "events": events,
                "errors": errors,
                "error_percent": round(100 * errors / events, 2) if events else 0,
                "p95_ms": _percentile(histogram, 0.95),
            }
        )
    return (
        {
            "events": count,
            "shards": len(partials),
            "services": rows,
            "finding": "Checkout has the injected 503 incident; other services provide a comparison baseline.",
        },
        {
            "events": count,
            "services": len(SERVICES),
            "processes": min(len(partials), workers or len(partials)),
        },
        cpu,
    )


if __name__ == "__main__":
    # The helper (see _pool_in_helper): the job spec on stdin, the partials as JSON on stdout.
    import json

    spec = json.load(sys.stdin)
    shard_fn = globals()[spec["fn"]]
    work = [
        (shard_fn, i, a, b)
        for i, (a, b) in enumerate(_ranges(spec["count"], spec["shards"]))
    ]
    partials, pool_cpu = _pool(shard_fn, work, spec["workers"])
    json.dump({"partials": partials, "pool_cpu": pool_cpu}, sys.stdout)
