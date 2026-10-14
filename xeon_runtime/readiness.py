"""Explicit capability contracts checked against declared sources.

Declarations are the contract: a goal's prose is task data and is never parsed
for capabilities. Feasibility is enforced where it is checkable — declared
sources against installed capabilities here, and planner-declared task
requirements and tool use against worker contracts in task_readiness.
"""

from .tool_catalog import SANDBOX_TOOLS, TOOL_INFO

CAPABILITIES = {
    "provided_context": "Supplied text",
    "sample_documents": "Sample documents",
    "sample_metrics": "Sample metrics",
    "demo_documents": "Sample document collection",
    "demo_transactions": "Sample invoices & payments",
    "demo_logs": "Sample service events",
    "demo_demand": "Sample demand scenarios",
    "warehouse": "Data warehouse (PostgreSQL or SQLite)",
    "knowledge_base": "Knowledge base (full-text and vector search)",
    "files": "Demo files (source, pages, articles)",
    "web_pages": "Demo web pages",
    "pdf_reports": "Demo PDF reports",
    "code_sandbox": "Code sandbox",
    "source_code": "Source tree and tests",
    "cache": "Shared cache (Redis)",
    "web": "Website access",
    "code_execution": "Code execution",
    "file_artifacts": "Files and mock-ups",
    "email": "Email delivery",
    "sql": "SQL database",
    "github": "GitHub repository",
    "telegram": "Telegram delivery",
    "discord": "Discord delivery",
    "slack": "Slack delivery",
    "external_data": "External business data",
}
TOOL_CAPABILITIES = {t: info["capability"] for t, info in TOOL_INFO.items()}


def capability_map(extra_tools=None):
    """Tool id → capability across built-ins and registered tool specs."""
    merged = dict(TOOL_CAPABILITIES)
    for spec in (extra_tools or {}).values():
        data = spec if isinstance(spec, dict) else spec.model_dump()
        merged[data["id"]] = data["capability"]
    return merged


def supported_capabilities(extra_tools=None):
    return {"provided_context", *capability_map(extra_tools).values()}


def worker_contract(worker, extra_tools=None):
    capabilities = capability_map(extra_tools)
    return {
        "inputs": ["Goal", "Task assignment", "Predecessor outputs"],
        "capabilities": [
            "provided_context",
            *[capabilities.get(t, t) for t in worker.get("tools", [])],
        ],
        "outputs": [
            "Structured JSON" if worker.get("output") == "json" else "Written answer"
        ],
        "sandbox": "Throwaway container per call · no network · CPU and memory limits"
        if any(t in SANDBOX_TOOLS for t in worker.get("tools", []))
        else "Not provisioned",
        "step_limit": worker.get("max_steps", 30),
    }


def inspect_goal(goal, declared=(), extra_tools=None):
    """Check declared sources against installed capabilities.

    The goal text is untrusted task data and is deliberately not scanned for
    capability keywords: phrasing must never grant or deny access. The
    declaration is the contract; tool binding limits what workers can do.
    Registered tools extend the supported set with their capabilities.
    """
    supported = supported_capabilities(extra_tools)
    required = set(declared)
    missing = sorted(required - supported)
    status = "blocked" if missing else "ready" if required else "needs_review"
    return {
        "status": status,
        "requirements": sorted(required),
        "missing": missing,
        "checks": [
            {
                "capability": c,
                "label": CAPABILITIES.get(c, c),
                "available": c in supported,
                "evidence": "Declared source",
            }
            for c in sorted(required)
        ],
        "message": "Required capabilities are unavailable"
        if missing
        else "Source contract checked; output still needs verification"
        if required
        else "Choose the sources this goal uses before running. Goal suitability has not been verified.",
    }


TYPOGRAPHIC = str.maketrans(
    {c: "-" for c in "‐‑‒–—―−"}
    | {c: "'" for c in "‘’"}
    | {c: '"' for c in "“”"}
    | {" ": " ", " ": " "}
)


def plain_text(text):
    """Lower-case with typographic dashes, quotes and spaces folded to ASCII: gpt-oss writes "fleet‑api" with a
    non-breaking hyphen, which must still satisfy a check spelled fleet-api."""
    return str(text).translate(TYPOGRAPHIC).lower()


def assess_output(status, output, events, required=(), required_tools=()):
    if status in ("queued", "running", "waiting"):
        return {"status": "pending", "label": "In progress", "checks": []}
    if status != "completed":
        blocked = any(
            e["kind"] in ("readiness_blocked", "plan_blocked") for e in events
        )
        return {
            "status": "blocked" if blocked else "incomplete",
            "label": "Blocked" if blocked else "Incomplete",
            "checks": [],
        }
    truncated = any(
        e["kind"] == "model"
        and e["payload"].get("finish_reason") in ("length", "max_tokens", "MAX_TOKENS")
        for e in events
    )
    plain = plain_text(output)
    checks = [{"text": s, "passed": plain_text(s) in plain} for s in required]
    completed = {
        e["payload"].get("request_id") for e in events if e["kind"] == "tool_completed"
    }
    actual_tools = {
        e["payload"].get("name")
        for e in events
        if e["kind"] == "tool_started" and e["payload"].get("request_id") in completed
    }
    checks += [
        {
            "text": "Tool completed: " + TOOL_INFO.get(t, {}).get("label", t),
            "passed": t in actual_tools,
        }
        for t in required_tools
    ]
    if truncated or not output.strip() or any(not c["passed"] for c in checks):
        return {
            "status": "incomplete",
            "label": "Incomplete",
            "checks": checks,
            "truncated": truncated,
        }
    return {
        "status": "checks_passed" if checks else "needs_review",
        "label": "Output checks passed" if checks else "Finished · needs review",
        "checks": checks,
        "note": "These checks do not establish semantic correctness.",
    }


def task_readiness(task, workers, tasks, extra_tools=None):
    """Direct operations need this worker's tools; source findings may arrive upstream."""
    by_ref = {w["ref"]: w for w in workers}
    by_id = {t["id"]: t for t in tasks}
    direct = set(
        worker_contract(by_ref[task["worker_ref"]], extra_tools)["capabilities"]
    )
    inherited = set()
    seen = set()

    def visit(id):
        if id in seen or id not in by_id:
            return
        seen.add(id)
        predecessor = by_id[id]
        if predecessor["worker_ref"] in by_ref:
            inherited.update(
                worker_contract(by_ref[predecessor["worker_ref"]], extra_tools)[
                    "capabilities"
                ]
            )
        for dependency in predecessor.get("depends_on", []):
            visit(dependency)

    for dependency in task.get("depends_on", []):
        visit(dependency)
    declared = set(task.get("requires", []))
    readiness = inspect_goal(task["task"], declared, extra_tools)
    unsupported = set(readiness["missing"]) | (
        set(readiness["requirements"]) - direct - inherited
    )
    unsupported.update(
        "tool:" + tool
        for tool in set(task.get("uses_tools", []))
        - set(by_ref[task["worker_ref"]].get("tools", []))
    )
    return {
        **readiness,
        "status": "blocked" if unsupported else "ready",
        "missing": sorted(unsupported),
        "task_id": task["id"],
        "assignment": task["task"],
        "task_name": task.get("name", task["id"]),
        "worker_ref": task["worker_ref"],
        "depends_on": task.get("depends_on", []),
        "direct_capabilities": sorted(direct),
        "inherited_capabilities": sorted(inherited),
        "message": "This assignment requires capabilities outside the worker and its supplied predecessor findings"
        if unsupported
        else "Worker operations and supplied predecessor findings checked",
    }
