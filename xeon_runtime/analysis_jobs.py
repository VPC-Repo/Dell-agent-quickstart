"""Fixed analysis jobs that run in the code sandbox: the work a team hands to a script, not to a model.

Each tool runs one script from this module in a throwaway sandbox container (see sandbox.py) on data the script
generates from a fixed seed. The work is fixed and split into fixed chunks, and the script runs them on a pool sized
to the machine (the mainstream `Pool()`; the container has no CPU limit), so a bigger machine finishes sooner while
every install measures the same work and reports the same findings. The container's CPU is read from its cgroup and
reported as the tool's CPU. Only the libraries in the sandbox image are used (numpy and the standard library); no
network, no model, no arguments the model could vary. Sized between a light agent (milliseconds of tool work) and
a heavy one (work that grows with the machine).
"""

from __future__ import annotations

import json
import os
import time

from langchain_core.tools import tool

from . import sandbox

# The pool is the parallelism: one process per CPU the container may use, each with single-threaded numpy.
PRELUDE = """
import os, sys, json
CPUS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
"""


CREDIT_RISK = (
    PRELUDE
    + """
import multiprocessing as mp
import numpy as np
from statistics import NormalDist
LOANS, RHO, CHUNKS, PER_CHUNK = 4000, 0.18, 1600, 250  # 400,000 scenarios in fixed chunks
base = np.random.default_rng(470)
pd_ = base.uniform(0.005, 0.06, LOANS); ead = base.lognormal(11, 0.8, LOANS); lgd = base.uniform(0.25, 0.65, LOANS)
threshold = np.array([NormalDist().inv_cdf(p) for p in pd_])
exposure = (ead * lgd).astype(np.float32)
def chunk(index):
    rng = np.random.default_rng(470 + index)
    z = rng.standard_normal((PER_CHUNK, 1)); e = rng.standard_normal((PER_CHUNK, LOANS))
    defaults = np.sqrt(RHO) * z + np.sqrt(1 - RHO) * e < threshold
    return defaults.astype(np.float32) @ exposure
with mp.get_context("fork").Pool(CPUS) as pool:
    losses = np.concatenate(pool.map(chunk, range(CHUNKS)))
losses.sort()
var99 = losses[int(0.99 * len(losses))]; var999 = losses[int(0.999 * len(losses))]
print(json.dumps({
    "loans": LOANS, "scenarios": int(losses.size), "processes": CPUS,
    "expected_loss_musd": round(float(losses.mean()) / 1e6, 2),
    "var_99_musd": round(float(var99) / 1e6, 2), "var_99_9_musd": round(float(var999) / 1e6, 2),
    "expected_shortfall_99_musd": round(float(losses[losses >= var99].mean()) / 1e6, 2),
}))
"""
)


CUSTOMER_DEDUPE = (
    PRELUDE
    + """
import difflib, multiprocessing as mp, random
from collections import defaultdict
rng = random.Random(470)
first = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jo", "Kofi", "Lena", "Mia", "Noor", "Omar", "Pia"]
last = ["Garcia", "Smith", "Nguyen", "Patel", "Kim", "Okafor", "Rossi", "Muller", "Silva", "Cohen", "Tanaka", "Haddad"]
streets = ["Oak", "Maple", "Pine", "Cedar", "Elm", "Birch", "Lake", "Hill"]
records = []
for i in range(24000):
    name = f"{rng.choice(first)} {rng.choice(last)}"; addr = f"{rng.randrange(1, 999)} {rng.choice(streets)} St"
    records.append((i, name, addr, f"{rng.randrange(10000, 99999)}"))
    if i % 8 == 0:  # a near-duplicate with a typo, as two systems would hold the same customer
        typo = name.replace("a", "e", 1) if "a" in name else name + "e"
        records.append((100000 + i, typo, addr.replace("St", "Street"), records[-1][3]))
blocks = defaultdict(list)
for r in records:
    blocks[(r[3][:1], r[1][0])].append(r)  # coarse blocking: postcode region and initial
def match(block):
    found = 0
    for a in range(len(block)):
        for b in range(a + 1, len(block)):
            x, y = block[a], block[b]
            score = 0.6 * difflib.SequenceMatcher(None, x[1], y[1]).ratio() + 0.4 * difflib.SequenceMatcher(None, x[2], y[2]).ratio()
            found += score > 0.86 and x[3] == y[3]
    return found, len(block) * (len(block) - 1) // 2
with mp.get_context("fork").Pool(CPUS) as pool:
    results = pool.map(match, list(blocks.values()), chunksize=1)
print(json.dumps({"records": len(records), "blocks": len(blocks), "pairs_compared": sum(r[1] for r in results), "duplicates_found": sum(r[0] for r in results),
                  "duplicates_expected": sum(1 for r in records if r[0] >= 100000), "processes": CPUS}))
"""
)

JOBS = {
    # name: (script, memory per pool process in MB, timeout_s, label for the evidence)
    "simulate_credit_risk": (
        CREDIT_RISK,
        96,
        300,
        "400,000 correlated-default portfolio scenarios, numpy",
    ),
    "dedupe_customer_records": (
        CUSTOMER_DEDUPE,
        48,
        300,
        "fuzzy matching of 27,000 customer records with difflib",
    ),
}


def host_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def run_job(name: str) -> str:
    script, per_process_mb, timeout, label = JOBS[name]
    started, cpu = time.perf_counter(), time.thread_time()
    # No CPU limit: the pool takes every core. Memory and the process limit grow with the pool so it fits.
    cpus = host_cpus()
    result = sandbox.run_python(
        script,
        timeout=timeout,
        cpus=None,
        memory_mb=1024 + per_process_mb * cpus,
        pids_limit=256 + 2 * cpus,
    )
    if result["exit_code"] != 0:
        raise ValueError(
            f"{name} failed in the sandbox: "
            + (result["stderr"] or result["stdout"])[-800:]
        )
    findings = json.loads(result["stdout"].strip().splitlines()[-1])
    caller = time.thread_time() - cpu
    in_sandbox = result["sandbox"].get("cpu_seconds")
    findings["work_evidence"] = {
        "tool": name,
        "data": "demo data generated in the sandbox from a fixed seed, reproducible on every install",
        "backend": {"kind": "sandbox", **result["sandbox"]},
        "units": {
            "job": label,
            "cpu_limit": "none (every core)",
            "processes": findings.get("processes"),
            "sandbox": result["sandbox"]["kind"],
            "wall_seconds": result["sandbox"]["wall_seconds"],
        },
        "cpu_seconds": round(caller + (in_sandbox or 0), 4),
        "caller_cpu_seconds": round(caller, 4),
        "sandbox_cpu_seconds": in_sandbox,
        "wall_seconds": round(time.perf_counter() - started, 4),
        "sandbox": result["sandbox"],
        "scope": "The sandbox container's CPU (its cgroup) plus the calling thread; excludes inference and coordination",
    }
    return json.dumps(findings)


@tool
def simulate_credit_risk() -> str:
    """Simulate 400,000 correlated-default scenarios for a 4,000-loan seeded portfolio with numpy in the sandbox, on every core; report expected loss, VaR and expected shortfall."""
    return run_job("simulate_credit_risk")


@tool
def dedupe_customer_records() -> str:
    """Find duplicate customers across two seeded systems (27,000 records) by fuzzy name and address matching in the sandbox, on every core."""
    return run_job("dedupe_customer_records")


REGISTRY = {t.name: t for t in [simulate_credit_risk, dedupe_customer_records]}
