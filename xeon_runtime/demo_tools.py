"""Real bounded CPU work on reproducible fictional enterprise data, on every core.

No shell, network, external writes, arbitrary code, artificial waits or busy loops.
Dataset generation is part of the measured work. Each job is split into one
shard per CPU (`CPU_JOB_SHARDS` overrides) and the shards run as separate
processes, so a heavy agent occupies the whole machine rather than one core.
Work per shard is fixed by this version, not controlled by a model; the total
therefore scales with the machine and the result reports how much was done.
Same seed, shard count and size give the same findings.
"""

from langchain_core.tools import tool

from .cpu_jobs import (  # noqa: F401  (re-exported for the tests and callers)
    ALIGN,
    PER_SHARD,
    SERVICES,
    _ranges,
    logs_job,
    measured,
    run_sharded,
    shard_count,
    transactions_job,
)


def _full(job, key):
    shards = shard_count()
    return lambda: job(PER_SHARD[key] * shards, shards)


@tool
def reconcile_transactions() -> str:
    """Reconcile seeded invoice/payment pairs with exact decimals on every CPU core at once (1.5 million pairs per core). Fictional data; real CPU work that fills the machine for about fifteen seconds."""
    return measured("reconcile_transactions", _full(transactions_job, "transactions"))


@tool
def analyze_service_logs() -> str:
    """Parse seeded JSON service events and compute error rates and p95 latency on every CPU core at once (1.5 million events per core). Fictional data; real CPU work that fills the machine for about fifteen seconds."""
    return measured("analyze_service_logs", _full(logs_job, "logs"))


# A fixed job: the same work whatever the machine (a full-machine job above grows with the core count), in fixed
# chunks on a pool of one process per CPU, so a bigger machine finishes sooner and every machine reports the same
# findings. Sized to sit between a light agent and a heavy one.
FIXED_CHUNKS = 512


def _fixed(job, count):
    return lambda: job(count, FIXED_CHUNKS, shard_count())


@tool
def analyze_service_day() -> str:
    """Parse one day of seeded JSON service events (3 million) and compute error rates and p95 latency, on every CPU core. Fictional data; real CPU work."""
    return measured("analyze_service_day", _fixed(logs_job, 2 * PER_SHARD["logs"]))


REGISTRY = {
    t.name: t
    for t in [reconcile_transactions, analyze_service_logs, analyze_service_day]
}
