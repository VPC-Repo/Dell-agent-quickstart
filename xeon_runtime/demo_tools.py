"""Real bounded CPU work on reproducible fictional enterprise data, on every core.

No shell, network, external writes, arbitrary code, artificial waits or busy loops.
Dataset generation is part of the measured work. Each job is split into one
shard per CPU (`CPU_JOB_SHARDS` overrides) and the shards run as separate
processes, so a heavy agent occupies the whole machine rather than one core.
Work per shard is fixed by this version, not controlled by a model; the total
therefore scales with the machine and the result reports how much was done.
Same seed, shard count and size give the same findings.
"""

import hashlib
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
import random
import re
import sys
import threading
import time
import zlib
from collections import Counter, defaultdict
from decimal import Decimal

from langchain_core.tools import tool

# Jobs within one process (one agent run) take turns: each already fills every core.
_CPU_SLOT = threading.Lock()

# Records per shard, sized for roughly fifteen to twenty seconds on one Xeon core.
PER_SHARD = {
    "documents": 40000,
    "transactions": 1500000,
    "logs": 1500000,
    "demand": 40000,
}
# Shard boundaries stay multiples of this so the seeded patterns (every tenth
# document a duplicate, four services round-robin) never straddle a shard.
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
    return partial


def _context():
    # forkserver: children inherit an already-imported module without the
    # threads of the worker process; spawn where forkserver is unavailable.
    if sys.platform.startswith("linux"):
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context("spawn")


def run_sharded(fn, count: int, shards: int) -> tuple[list[dict], float]:
    """Run fn(shard, start, stop) over `count` records in `shards` processes; partials and CPU seconds."""
    tasks = [(fn, i, a, b) for i, (a, b) in enumerate(_ranges(count, shards))]
    if len(tasks) == 1:
        partials = [_run_shard(tasks[0])]
    else:
        # ProcessPoolExecutor raises BrokenProcessPool if a child dies instead of respawning it forever.
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=_context()) as pool:
            partials = list(pool.map(_run_shard, tasks))
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
    }
    return json.dumps(result)


# -- documents -----------------------------------------------------------------
def _documents_shard(shard, start, stop):
    rng = random.Random(470 + shard)
    index, fingerprints = defaultdict(Counter), set()
    unique = duplicates = raw_bytes = compressed_bytes = 0
    departments = ["fulfillment", "payments", "identity", "inventory"]
    vocabulary = "service deployment latency retry timeout queue backlog capacity incident recovery audit customer order invoice settlement policy approval retention ownership escalation monitoring".split()
    batch = []
    body = ""
    for i in range(start, stop):
        # Every tenth document is an exact duplicate of its predecessor.
        if i % 10 != 9 or not body:
            department = departments[i % 4]
            words = [rng.choice(vocabulary) for _ in range(180)]
            body = (
                f"<article><h1>{department} service note {i}</h1><p>"
                + " ".join(words)
                + "</p></article>"
            )
        text = " ".join(re.sub(r"<[^>]+>", " ", body).lower().split())
        encoded = text.encode()
        fingerprint = hashlib.sha256(encoded).hexdigest()
        duplicates += fingerprint in fingerprints
        if fingerprint not in fingerprints:
            unique += 1
            fingerprints.add(fingerprint)
            for term, frequency in Counter(re.findall(r"[a-z]+", text)).items():
                index[term][departments[i % 4]] += frequency
        raw_bytes += len(encoded)
        batch.append(encoded)
        if len(batch) == 200 or i == stop - 1:
            compressed_bytes += len(zlib.compress(b"\n".join(batch), level=9))
            batch.clear()
    return {
        "unique": unique,
        "duplicates": duplicates,
        "raw": raw_bytes,
        "compressed": compressed_bytes,
        "index": {t: dict(c) for t, c in index.items()},
    }


def documents_job(count=60000, shards=1):
    partials, cpu = run_sharded(_documents_shard, count, shards)
    index: dict[str, int] = Counter()
    for p in partials:
        for term, depts in p["index"].items():
            index[term] += sum(depts.values())
    return (
        {
            "collection": "Fictional service knowledge base",
            "documents": count,
            "shards": len(partials),
            "unique_documents": sum(p["unique"] for p in partials),
            "duplicates": sum(p["duplicates"] for p in partials),
            "indexed_terms": len(index),
            "uncompressed_bytes": sum(p["raw"] for p in partials),
            "compressed_bytes": sum(p["compressed"] for p in partials),
            "findings": "Deduplication removes repeated service notes before indexing.",
            "top_incident_terms": {
                w: index[w] for w in ["incident", "timeout", "recovery"]
            },
        },
        {
            "documents": count,
            "input_bytes": sum(p["raw"] for p in partials),
            "processes": len(partials),
        },
        cpu,
    )


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


def logs_job(count=3000000, shards=1):
    partials, cpu = run_sharded(_logs_shard, count, shards)
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
        {"events": count, "services": len(SERVICES), "processes": len(partials)},
        cpu,
    )


# -- demand -------------------------------------------------------------------------
POLICIES = ["lean", "balanced", "buffered"]


def _demand_shard(shard, start, stop):
    rng = random.Random(470 + shard)
    result = {p: {"stockouts": 0, "inventory_days": 0} for p in POLICIES}
    # Common random demand paths compare the same demand under each policy.
    for scenario in range(start, stop):
        product = scenario % 24
        mean = 14 + product
        stock = [mean * days for days in [5, 8, 12]]
        pending = [0, 0, 0]
        due = [-1, -1, -1]
        for day in range(90):
            demand = max(
                0, round(rng.gauss(mean * (1 + 0.15 * math.sin(day / 7)), mean * 0.35))
            )
            for j, policy in enumerate(POLICIES):
                if due[j] == day:
                    stock[j] += pending[j]
                    pending[j] = 0
                lost = max(0, demand - stock[j])
                stock[j] = max(0, stock[j] - demand)
                result[policy]["stockouts"] += lost
                result[policy]["inventory_days"] += stock[j]
                if stock[j] < mean * [3, 5, 7][j] and not pending[j]:
                    pending[j] = mean * [5, 8, 12][j]
                    due[j] = day + 4
    return result


def demand_job(scenarios=60000, shards=1):
    partials, cpu = run_sharded(_demand_shard, scenarios, shards)
    result = {
        p: {
            "stockouts": sum(x[p]["stockouts"] for x in partials),
            "inventory_days": sum(x[p]["inventory_days"] for x in partials),
        }
        for p in POLICIES
    }
    return (
        {
            "scenarios": scenarios,
            "shards": len(partials),
            "products": 24,
            "days_per_scenario": 90,
            "policies": result,
            "inventory_comparisons": {
                "buffered_divided_by_balanced": round(
                    result["buffered"]["inventory_days"]
                    / result["balanced"]["inventory_days"],
                    3,
                ),
                "buffered_divided_by_lean": round(
                    result["buffered"]["inventory_days"]
                    / result["lean"]["inventory_days"],
                    3,
                ),
            },
            "decision_limit": "No cost model or service-level target is supplied. A preferred policy is conditional on business priorities, not mathematically optimal.",
            "finding": "Compare unfilled demand with inventory held; these are seeded scenarios, not a production forecast.",
        },
        {
            "scenarios": scenarios,
            "product_days": scenarios * 90,
            "policy_evaluations": scenarios * 90 * 3,
            "processes": len(partials),
        },
        cpu,
    )


def _full(job, key):
    shards = shard_count()
    return lambda: job(PER_SHARD[key] * shards, shards)


@tool
def process_documents() -> str:
    """Parse, deduplicate, index and compress seeded service documents on every CPU core at once (40,000 documents per core). Fictional data; real CPU work that fills the machine for about fifteen seconds."""
    return measured("process_documents", _full(documents_job, "documents"))


@tool
def reconcile_transactions() -> str:
    """Reconcile seeded invoice/payment pairs with exact decimals on every CPU core at once (1.5 million pairs per core). Fictional data; real CPU work that fills the machine for about fifteen seconds."""
    return measured("reconcile_transactions", _full(transactions_job, "transactions"))


@tool
def analyze_service_logs() -> str:
    """Parse seeded JSON service events and compute error rates and p95 latency on every CPU core at once (1.5 million events per core). Fictional data; real CPU work that fills the machine for about fifteen seconds."""
    return measured("analyze_service_logs", _full(logs_job, "logs"))


@tool
def forecast_demand() -> str:
    """Compare three inventory policies over seeded 90-day demand scenarios on every CPU core at once (40,000 scenarios per core). Fictional data; real CPU work that fills the machine for about fifteen seconds."""
    return measured("forecast_demand", _full(demand_job, "demand"))


REGISTRY = {
    t.name: t
    for t in [
        process_documents,
        reconcile_transactions,
        analyze_service_logs,
        forecast_demand,
    ]
}
