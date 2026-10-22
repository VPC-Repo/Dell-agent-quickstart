"""Eight example agents and their content-addressed worker definitions."""

import argparse
import hashlib
import json
import time
import copy

from sqlalchemy import select

from . import automations as agents
from . import store
from .planning import catalog

# VERSION is content-derived; see below AGENTS. Any change to a worker or agent spec reinstalls the catalog.
COMMON = (
    " Write for a business reader in plain language: one sentence with the outcome, two to four short bullets with the key"
    " numbers, then next actions. Everyone knows this is demo data, so never write fictional, synthetic, seed, dataset or"
    " version names, process counts, tool names, source or document IDs, or execution times. Use only observed tool results"
    " or predecessor findings and keep quantities exact. Keep your answer under 130 words. Do not claim external actions occurred."
)


def worker(
    id,
    name,
    description,
    instruction,
    tools=(),
    role="general",
    max_steps=8,
    version=1,
    required=(),
    common=COMMON,
):
    body = {
        "name": name,
        "description": description,
        "instructions": instruction + common,
        "tools": list(tools),
        "output": "text",
        "max_steps": max_steps,
        "role": role,
        # The tool a correct step cannot skip: the runtime tells a model that answers without it to
        # continue with the tool (seen with gpt-oss-120b writing an incident analysis after only
        # describing the warehouse), and the outcome counts the slip as a recovered error.
        "required_tools": list(required),
    }
    # Content-addressed ref: any wording change (including the shared COMMON text) yields a new
    # worker version, so a reinstall never collides with an older worker stored under the same key.
    digest = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:6]
    return {"ref": f"demo-{id}@{version}.{digest}", **body}


WORKERS = [
    worker(
        "finance",
        "Transaction reconciliation specialist",
        "Match seeded invoice/payment pairs on every CPU core and flag exceptions.",
        "Call reconcile_transactions exactly once; it fills every core for about fifteen seconds. Report pairs processed, exception count, unsettled amount, account examples.",
        ["reconcile_transactions"],
        required=["reconcile_transactions"],
        version=2,
    ),
    worker(
        "logs",
        "Service incident analyst",
        "Parse seeded service events on every CPU core to identify errors and latency changes.",
        "Call analyze_service_logs exactly once; it fills every core for about fifteen seconds. Compare service error rates and p95 latency and name the affected service.",
        ["analyze_service_logs"],
        required=["analyze_service_logs"],
        version=2,
    ),
    worker(
        "review",
        "Evidence reviewer",
        "Check the proposed claims against the findings already collected.",
        "Review the predecessor findings. List supported claims and any uncertainties. Preserve all source IDs. Do not repeat data collection or invent a new source.",
    ),
    worker(
        "writer",
        "Decision brief writer",
        "Turn worker findings into a concise report with next actions.",
        "Write the final requested report, under 180 words, for a business reader. Use the predecessor findings. Open with the outcome in one sentence, then the specific numbers, risks and next actions. Do not add unsupported claims or a generic daily briefing title unless that is the goal. Copy every figure exactly as the findings give it, including percentages and totals, and never recalculate a figure or derive a new one.",
    ),
]

AGENTS = []
SCHEMA_NOTE = " The warehouse tables are customers, orders, payments, tickets and service_events(event_id, at, service, endpoint, status, latency_ms, region); call describe_warehouse first and use only the columns it lists, compare timestamps as text (at >= '2026-09-22 09:00'), bucket by hour with substr(CAST(at AS TEXT), 1, 13), never alias a column as window, user or end, and ask one question per query: several small SELECT statements, never one UNION of unrelated results. Results are cut at 200 rows, so aggregate by one dimension first (per service, per month), then drill into the value that stands out; order by the measure and limit when a result could be long. Status values are delivered, shipped, pending, cancelled and returned for orders and open, waiting_on_customer, escalated and resolved for tickets. A query that returned no rows must be changed before it is run again."

# --- Agents on the toolkit's stack: PostgreSQL + pgvector, Redis, the code sandbox
# and the seeded demo files. Roles pick the model: coder for code and SQL,
# reasoning for analysis and planning, general otherwise.
WORKERS += [
    worker(
        "bug-fixer",
        "Bug fixer",
        "Reproduce a failing test in the sandbox, fix the code and rerun the tests.",
        "Work in this order, one tool call per step, and do not write any prose until step 5: (1) run_tests to see the failure; (2) read_fixture_file for the failing test; (3) read_fixture_file for the module under test; (4) write_source_file with the corrected full module, then run_tests again, repeating the write and the run at most twice; (5) only then report the root cause, the change and the final pass and fail counts from the tool output. A fix exists only once write_source_file and run_tests have both been called; if the suite still fails, say so.",
        ["read_fixture_file", "write_source_file", "run_tests"],
        role="coder",
        max_steps=18,
        required=["write_source_file", "run_tests"],
    ),
    worker(
        "incident-analyst",
        "Incident analyst",
        "Locate an incident in the service events and find the matching runbook.",
        "Work through these steps in order and do not write any conclusion before step 4 has returned results. "
        "Step 1: call describe_warehouse once. "
        "Step 2: call query_warehouse on service_events (columns: at, service, status, latency_ms) for the error rate (status >= 400) and the average latency_ms by service. "
        "Step 3: call query_warehouse again for the worst service by hour, using substr(CAST(at AS TEXT), 1, 13) AS hour_bucket, and narrow to the hours where the error rate or latency stands out. "
        "Step 4: call search_documents for the runbook that matches that service and symptom. "
        "Then report the service, the hours, the figures and the runbook document ID. The data does contain an incident; an answer that none was found means a step was skipped."
        + SCHEMA_NOTE,
        ["describe_warehouse", "query_warehouse", "search_documents"],
        max_steps=18,
        version=4,
        required=["query_warehouse", "search_documents"],
    ),
    worker(
        "faq-responder",
        "FAQ responder with cache",
        "Answer a customer question, reusing a cached answer when one exists.",
        "Derive a cache key from the question (letters, digits and dashes). Call cache_get first. If found, use the cached answer and say it was cached. Otherwise call search_documents and read the best document with the snippet, compose the answer, and store it with cache_set. Name the document the answer came from.",
        ["cache_get", "cache_set", "search_documents"],
        max_steps=10,
        required=["cache_get"],
    ),
    worker(
        "root-cause",
        "Root cause analyst",
        "Reason from the collected evidence to the most likely cause.",
        "Using only the predecessor findings, rank the plausible causes, state the evidence for and against each, and name what would confirm the top cause. Do not collect new data.",
        role="reasoning",
    ),
]

# Panelists keep a GPU engine generating: the same rules as COMMON, but a long section instead of a 130-word answer.
PANEL = (
    " Write a detailed assessment of about 900 words for a business reader, in six sections: summary, findings, the evidence"
    " with specific figures and names from what you read, risks, recommendations with an owner and a first step each, and open"
    " questions. Cover every topic in your task fully; do not stop early or summarize in bullets. Everyone knows this is demo data, so never write fictional,"
    " synthetic, seed, dataset or version names, process counts, tool names, source or document IDs, or execution times. Use"
    " only observed tool results and keep quantities exact. Do not claim external actions occurred."
)
WORKERS += [
    worker(
        "panel-reliability",
        "Reliability panelist",
        "Assess service reliability from the runbooks and incident notes.",
        "Call search_documents twice (incidents and throttling; runbooks) and read_document for any document whose snippet is not enough.",
        # The reasoning engine answers from the snippets and ignores a nudge to read; requiring the search keeps the step honest without a false error.
        ["search_documents", "read_document"],
        role="reasoning",
        max_steps=12,
        required=["search_documents"],
        common=PANEL,
    ),
    worker(
        "panel-policy",
        "Policy panelist",
        "Assess the customer-facing policies: refunds, warranty, data retention, support plans.",
        "Call search_documents for the refund, warranty, data retention and support plan policies (two searches), then read_document for the three most relevant documents.",
        ["search_documents", "read_document"],
        max_steps=12,
        required=["read_document"],
        common=PANEL,
    ),
    worker(
        "panel-code",
        "Code panelist",
        "Review the orderflow source for defects, error handling and maintainability.",
        "Call list_fixture_files for src, then read_fixture_file for every module and its tests (up to six files). Quote file paths and line-level details; propose changes but do not claim to have run or changed anything.",
        ["list_fixture_files", "read_fixture_file"],
        role="coder",
        max_steps=16,
        required=["read_fixture_file"],
        common=PANEL,
    ),
    worker(
        "panel-data",
        "Data panelist",
        "Assess revenue, order flow and returns from the warehouse.",
        "Call describe_warehouse once, then up to four query_warehouse statements: revenue by month in 2026, orders by status, returns by region, the top five products by revenue."
        + SCHEMA_NOTE,
        ["describe_warehouse", "query_warehouse"],
        role="coder",
        max_steps=14,
        required=["query_warehouse"],
        common=PANEL,
    ),
    worker(
        "panel-support",
        "Support panelist",
        "Assess the support queue from the tickets.",
        "Call describe_warehouse once, then up to four query_warehouse statements on tickets: counts by topic and priority, counts by status, the five oldest open tickets, monthly volume in 2026."
        + SCHEMA_NOTE,
        ["describe_warehouse", "query_warehouse"],
        max_steps=18,
        required=["query_warehouse"],
        common=PANEL,
    ),
    worker(
        "panel-strategy",
        "Strategy panelist",
        "Assess the product direction from release notes, blog posts and supplier notes.",
        "Call semantic_search twice (product direction and releases; supply and partners) and read_document for any document whose snippet is not enough.",
        ["semantic_search", "read_document"],
        role="reasoning",
        max_steps=12,
        required=["semantic_search"],
        common=PANEL,
    ),
]
# Second stage of the panel: three reviewers, one per engine role, read every section at once (long prompts, long answers).
REVIEW = (
    " Write about 700 words for a business reader: for each panel section you were given, what holds up, what is missing or"
    " contradicted by another section, and what you would change; then the five recommendations you would keep, each with an"
    " owner and a first step. Everyone knows this is demo data, so never write fictional, synthetic, seed, dataset or version"
    " names, process counts, tool names, source or document IDs, or execution times. Use only the predecessor findings. Do not"
    " claim external actions occurred."
)
WORKERS += [
    worker(
        "panel-review-business",
        "Business reviewer",
        "Review every panel section from the business owner's seat.",
        "Review all the panel sections you were given for business impact and priority.",
        common=REVIEW,
    ),
    worker(
        "panel-review-technical",
        "Technical reviewer",
        "Review the panel sections for technical accuracy and feasibility.",
        "Review all the panel sections you were given for technical accuracy, feasibility and the cost of each recommendation.",
        role="coder",
        common=REVIEW,
    ),
    worker(
        "panel-review-risk",
        "Risk reviewer",
        "Review the panel sections for risks, contradictions and missing evidence.",
        "Review all the panel sections you were given for risks, contradictions between sections and claims without evidence.",
        role="reasoning",
        common=REVIEW,
    ),
]


def _agent(
    slug,
    name,
    description,
    goal,
    workers,
    fan_in,
    requirements,
    checks,
    tools,
    load,
    workload,
    deadline,
    category,
):
    return dict(
        slug=slug,
        name=name,
        description=description,
        goal=goal,
        workers=workers,
        fan_in=fan_in,
        requirements=requirements,
        checks=checks,
        tools=tools,
        load=load,
        workload=workload,
        deadline=deadline,
        category=category,
    )


AGENTS += [
    _agent(
        "bug-fix",
        "Fix the failing test",
        "Runs a failing test suite, reads the code, writes a fix and runs the tests again.",
        "The orderflow test suite has a failing test. Reproduce it, fix the code so the whole suite passes, and report the root cause, the change and the final test counts as reported by the tool.",
        ["bug-fixer", "writer"],
        ["writer"],
        ["source_code", "code_sandbox"],
        ["pass"],
        ["read_fixture_file", "write_source_file", "run_tests"],
        "Medium",
        "pytest runs in throwaway sandbox containers",
        480,
        "Code",
    ),
    _agent(
        "incident-rca",
        "Incident root cause analysis",
        "Finds the outage in a week of service events, matches it to the runbook and explains the likely cause.",
        "Find the incident in the demo service events between 2026-09-20 and 2026-09-24: which service, which hours, error rate and latency versus normal. Match it to the runbook and explain the most likely root cause and the mitigation.",
        ["incident-analyst", "root-cause", "writer"],
        ["root-cause", "writer"],
        ["warehouse", "knowledge_base"],
        ["fleet-api"],
        ["describe_warehouse", "query_warehouse", "search_documents"],
        "Medium",
        "SQL over 60,000 events plus reasoning",
        420,
        "Operations",
    ),
    _agent(
        "faq-cache",
        "Cached FAQ responder",
        "Answers a customer's question from the knowledge base and remembers the answer for next time.",
        "Answer this customer question using the knowledge base: 'What is your refund window and how do refunds work for Premier support plans?' Check the cache first, store the answer for next time.",
        ["faq-responder"],
        [],
        ["knowledge_base", "cache"],
        ["refund"],
        ["cache_get", "cache_set", "search_documents"],
        "Light",
        "Redis cache lookup and one knowledge base search",
        240,
        "Assistants",
    ),
    # GPU heavy: six specialists start together and each generates about 500 words, so the general, coder and reasoning
    # engines are all busy at once. The gallery shows this tier only on a host with GPUs.
    _agent(
        "gpu-panel",
        "Expert panel review",
        "Six specialists each assess one part of the business, three reviewers challenge them, and a brief sums it up.",
        "Convene an expert panel on the demo business: reliability, customer policies, code quality, data, support and product direction. Each specialist writes a detailed assessment from the knowledge base, the source code or the warehouse; three reviewers then critique every section; then combine the reviews into one brief with the three most important actions.",
        [
            "panel-reliability",
            "panel-policy",
            "panel-code",
            "panel-data",
            "panel-support",
            "panel-strategy",
            "panel-review-business",
            "panel-review-technical",
            "panel-review-risk",
            "writer",
        ],
        ["writer"],
        ["knowledge_base", "warehouse", "source_code"],
        ["action"],
        [
            "search_documents",
            "read_document",
            "semantic_search",
            "list_fixture_files",
            "read_fixture_file",
            "describe_warehouse",
            "query_warehouse",
        ],
        "GPU heavy",
        "Six specialists, then three reviewers, generating at once across the general, coder and reasoning engines",
        600,
        "Operations",
    )
    | {
        "stages": [
            [
                "panel-reliability",
                "panel-policy",
                "panel-code",
                "panel-data",
                "panel-support",
                "panel-strategy",
            ],
            ["panel-review-business", "panel-review-technical", "panel-review-risk"],
            ["writer"],
        ]
    },
    # Full machine: two sharded CPU jobs fill every core while the six panelists keep every GPU engine generating,
    # then the three reviewers and the brief. Twelve workers, the most an agent may have. GPU hosts only.
    _agent(
        "full-machine",
        "Full business review",
        "Analyzes the service outage and reconciles every payment while a panel of six specialists reviews the business, then reviewers and a brief.",
        "Review the whole demo business at once: analyze the service events for the incident and reconcile every invoice and payment (both computed across every core), while an expert panel assesses reliability, customer policies, code quality, data, support and product direction from the knowledge base, the source code and the warehouse. Three reviewers then critique everything; then one brief with the three most important actions.",
        [
            "logs",
            "panel-reliability",
            "panel-policy",
            "panel-code",
            "panel-data",
            "panel-support",
            "panel-strategy",
            "finance",
            "panel-review-business",
            "panel-review-technical",
            "panel-review-risk",
            "writer",
        ],
        ["writer"],
        [
            "demo_logs",
            "demo_transactions",
            "knowledge_base",
            "warehouse",
            "source_code",
        ],
        ["action"],
        [
            "analyze_service_logs",
            "reconcile_transactions",
            "search_documents",
            "read_document",
            "semantic_search",
            "list_fixture_files",
            "read_fixture_file",
            "describe_warehouse",
            "query_warehouse",
        ],
        "Full machine",
        "Two sharded CPU jobs and six long generations at once, then three reviewers",
        600,
        "Operations",
    )
    # A run executes its CPU jobs one at a time, so one job goes with the panelists and the other with the reviewers:
    # every core and every engine stay busy through both stages.
    | {
        "stages": [
            [
                "logs",
                "panel-reliability",
                "panel-policy",
                "panel-code",
                "panel-data",
                "panel-support",
                "panel-strategy",
            ],
            [
                "finance",
                "panel-review-business",
                "panel-review-technical",
                "panel-review-risk",
            ],
            ["writer"],
        ]
    },
]


# Fixed CPU jobs: the same work whatever the machine, in fixed chunks on a pool of one process per CPU (in the run's
# job processes, or a fixed script in the sandbox container), so a bigger machine finishes sooner and their CPU per
# run is a property of the agent. Each worker calls its job once.
def job_worker(id, name, description, tool, report, role="general"):
    return worker(
        id,
        name,
        description,
        f"Call {tool} exactly once. {report}",
        [tool],
        role=role,
        required=[tool],
    )


WORKERS += [
    job_worker(
        "service-day",
        "Service day analyst",
        "Analyze one day of service events.",
        "analyze_service_day",
        "Compare service error rates and p95 latency and name the affected service.",
    ),
    job_worker(
        "risk-modeler",
        "Credit risk modeler",
        "Simulate the loan portfolio's credit losses in the sandbox.",
        "simulate_credit_risk",
        "Report loans and scenarios, expected loss, VaR at 99% and 99.9% and expected shortfall, in millions of USD.",
    ),
    job_worker(
        "record-matcher",
        "Customer record matcher",
        "Find duplicate customers across two systems in the sandbox.",
        "dedupe_customer_records",
        "Report records compared, pairs compared, duplicates found versus expected and the match rate.",
    ),
]

AGENTS += [
    _agent(
        "service-health-day",
        "Daily service health report",
        "Reads a day of service events and reports which service degraded.",
        "Analyze one day of demo service events. Name the affected service, compare its error rate and p95 latency with the others, and recommend next actions.",
        ["service-day", "writer"],
        ["writer"],
        ["demo_logs"],
        ["checkout"],
        ["analyze_service_day"],
        "Medium",
        "3 million JSON service events on every core",
        300,
        "Operations",
    ),
    _agent(
        "credit-risk",
        "Portfolio credit risk",
        "Simulates the loan book's losses and checks the capital buffer.",
        "Simulate the demo loan portfolio's credit losses and review them: expected loss, VaR at 99% and 99.9%, expected shortfall, and what they mean for the capital buffer.",
        ["risk-modeler", "review", "writer"],
        ["review", "writer"],
        ["code_sandbox"],
        ["loss"],
        ["simulate_credit_risk"],
        "Medium",
        "400,000 Monte Carlo scenarios with numpy in the sandbox, on every core",
        420,
        "Data & analysis",
    ),
    _agent(
        "customer-dedupe",
        "Customer master cleanup",
        "Finds the same customer held twice across two systems.",
        "Find duplicate customer records across the two demo systems: records and pairs compared, duplicates found versus expected, and the merge plan.",
        ["record-matcher", "writer"],
        ["writer"],
        ["code_sandbox"],
        ["duplicate"],
        ["dedupe_customer_records"],
        "Medium",
        "2.5 million fuzzy record comparisons in the sandbox, on every core",
        420,
        "Data & analysis",
    ),
]

# The eight agents, each working the machine in a different way (the tiers from Light to Full machine). All eight run
# on every host; the two GPU tiers run on CPU inference too, more slowly.
FEATURED = [
    "faq-cache",
    "incident-rca",
    "bug-fix",
    "service-health-day",
    "credit-risk",
    "customer-dedupe",
    "gpu-panel",
    "full-machine",
]
assert sorted(a["slug"] for a in AGENTS) == sorted(FEATURED), (
    "the demo screen shows exactly these eight agents"
)


# A cache hit answers without searching or writing, so only the lookup is required for a correct run.
next(a for a in AGENTS if a["slug"] == "faq-cache")["required_tools"] = ["cache_get"]

# Any change to a worker or an agent spec yields a new catalog version, so `install(archive_legacy=True)`
# replaces every agent; nothing keeps pointing at a superseded worker.
VERSION = (
    "enterprise-demo-"
    + hashlib.sha1(
        json.dumps([WORKERS, AGENTS], sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
)

WORKER_REFS = {w["ref"].split("@")[0].removeprefix("demo-"): w["ref"] for w in WORKERS}


def install(provider_id="intel-toolkit", archive_legacy=False):
    """Explicit administrator operation, idempotent; not an automatic startup seed."""
    with store.Session() as db:
        provider = db.get(store.Document, "provider:" + provider_id)
        if not provider or provider.payload["adapter"] == "mock":
            raise ValueError(
                "Choose a connected model before installing the demo catalog"
            )
        if db.scalar(
            select(store.Run.id).where(
                store.Run.status.in_(["queued", "running", "waiting"])
            )
        ):
            raise ValueError("Wait for current runs before changing the demo catalog")
    with store.Session.begin() as db:
        for w in WORKERS:
            key = "worker:" + w["ref"]
            existing = db.get(store.Document, key)
            if existing:
                current = {k: v for k, v in existing.payload.items() if k != "archived"}
                if current != w:
                    raise ValueError(
                        "Demo worker version already exists with different content"
                    )
            else:
                agents.put(db, key, "worker", w)
        agents.put(
            db,
            "catalog:demo",
            "catalog_config",
            {"version": VERSION},
        )
    installed = []
    for spec in AGENTS:
        with store.Session() as db:
            old = next(
                (
                    d
                    for d in db.scalars(
                        select(store.Document).where(
                            store.Document.kind == "automation"
                        )
                    )
                    if d.payload.get("demo", {}).get("slug") == spec["slug"]
                    and d.payload["demo"].get("version") == VERSION
                    and not d.payload.get("archived")
                ),
                None,
            )
            if old:
                # Rebind the current design when an operator selects another gateway.
                # Completed run snapshots retain their original provider definitions.
                with store.Session.begin() as write_db:
                    for binding in old.payload["bindings"].values():
                        draft = write_db.get(store.Document, "draft:" + binding["id"])
                        payload = copy.deepcopy(draft.payload)
                        for node in payload["nodes"]:
                            if node["kind"] == "model":
                                node["config"]["provider_id"] = provider_id
                        if payload != draft.payload:
                            draft.payload, draft.updated = payload, time.time()
                installed.append(old.payload["id"])
                continue
        a = agents.create()
        with store.Session.begin() as db:
            meta = copy.deepcopy(agents.metadata(db, a["id"], True).payload)
            refs = [WORKER_REFS[w] for w in spec["workers"]]
            available = {w["ref"]: w for w in catalog(db)}
            tasks = []
            # Aggregation is declared per template, never inferred from worker naming, so user-authored plans can
            # express the same shape: fan_in tasks depend on every task before them; with explicit stages, each
            # task depends on the whole previous stage (parallel groups one after another).
            stage_of = {
                slug: n
                for n, stage in enumerate(spec.get("stages") or [])
                for slug in stage
            }
            for i, (slug, ref) in enumerate(zip(spec["workers"], refs)):
                w = available[ref]
                if stage_of:
                    deps = [
                        tasks[j].id
                        for j, earlier in enumerate(spec["workers"][:i])
                        if stage_of[earlier] == stage_of[slug] - 1
                    ]
                else:
                    deps = [t.id for t in tasks] if slug in spec["fan_in"] else []
                tasks.append(
                    agents.AgentInput(
                        id="task" + str(i + 1),
                        name=w["name"],
                        worker_name=w["name"],
                        worker_ref=ref,
                        instructions=w["instructions"],
                        tools=w["tools"],
                        max_steps=w["max_steps"],
                        role=w.get("role", "general"),
                        uses_tools=w.get("required_tools", []),
                        task=w["description"],
                        depends_on=deps,
                    )
                )
            meta["coordination"] = {"worker_refs": refs, "max_tasks": len(refs)}
            meta["demo"] = {"version": VERSION, **spec}
            agents.put(db, "automation:" + a["id"], "automation", meta)
            db.flush()
            item = agents.assemble(db, meta)[0]
            edit = agents.Edit(
                revision=item["revision"],
                name=spec["name"],
                goal=spec["goal"],
                provider_id=provider_id,
                expected_result="A plain-language brief under 180 words for a business reader: the outcome first, the key numbers, then next actions. No internal IDs, seeds, versions or tool names.",
                requirements=spec["requirements"],
                output_checks=spec["checks"],
                agents=tasks,
                schedule=agents.Settings(repeat="manual"),
                execution_mode="authored",
            )
            agents.write_design(db, meta, edit)
        installed.append(a["id"])
    if archive_legacy:
        _archive_previous_catalog(installed)

    return installed


def _archive_previous_catalog(installed):
    """Archive older demo generations and their records; preserve user designs."""
    with store.Session.begin() as db:
        old_agents = [
            d
            for d in db.scalars(
                select(store.Document).where(store.Document.kind == "automation")
            )
            if d.payload.get("demo") and d.payload.get("id") not in installed
        ]
        old_blueprints = {
            id
            for d in old_agents
            for id in [
                d.payload["blueprint_id"],
                *(b["id"] for b in d.payload["bindings"].values()),
            ]
        }
        old_deployments = {d.payload.get("deployment_id") for d in old_agents}
        old_schedules = {d.payload.get("schedule_id") for d in old_agents}
        current_workers = {w["ref"] for w in WORKERS}
        for d in db.scalars(select(store.Document)):
            owned = (
                d in old_agents
                or (d.kind == "draft" and d.payload.get("id") in old_blueprints)
                or (d.kind == "deployment" and d.payload.get("id") in old_deployments)
                or (
                    d.kind == "worker"
                    and d.payload.get("ref", "").startswith("demo-")
                    and d.payload.get("ref") not in current_workers
                )
            )
            if owned:
                if d.kind == "draft":
                    d.kind = "archived_draft"
                else:
                    d.payload = {
                        **d.payload,
                        "archived": True,
                        "archive_reason": "Superseded demo catalog generation",
                    }
        hidden = [
            d.key
            for d in db.scalars(
                select(store.Document).where(store.Document.kind == "published")
            )
            if d.payload.get("id") in old_blueprints
        ]
        config = db.get(store.Document, "catalog:demo")
        config.payload = {**config.payload, "hidden_published": hidden}
        for sched in db.scalars(select(store.Schedule)):
            if sched.id in old_schedules:
                sched.paused = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="intel-toolkit")
    parser.add_argument("--archive-legacy", action="store_true")
    args = parser.parse_args()
    store.initialize()
    print(install(args.provider, args.archive_legacy))
