"""The built-in tool catalog shared by the planner, validator and executor.

Tool identity is an open string namespace carried in execution snapshots; the eight demo agents use the
built-in tools listed here. Existence is checked where a catalog is available, never by a closed type.
"""

from typing import Annotated

from pydantic import Field

# Shared identifier shape for built-in and registered tools.
ToolName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")]

TOOL_INFO = {
    "search_knowledge": {
        "label": "Search briefing sources",
        "capability": "sample_documents",
        "category": "knowledge",
        "description": "Search the bundled operational briefing documents and return citations.",
    },
    "analyze_metrics": {
        "label": "Calculate briefing metrics",
        "capability": "sample_metrics",
        "category": "knowledge",
        "description": "Calculate changes in the bundled operational metrics.",
    },
    "reconcile_transactions": {
        "label": "Reconcile transactions",
        "capability": "demo_transactions",
        "category": "finance",
        "description": "Generate and reconcile 1,500,000 seeded invoice/payment pairs per available CPU with exact decimal amounts; identify mismatches and summarize exceptions.",
    },
    "analyze_service_logs": {
        "label": "Analyze service logs",
        "capability": "demo_logs",
        "category": "operations",
        "description": "Parse 1,500,000 seeded JSON service events per available CPU, aggregate latency and errors by service and detect an injected incident.",
    },
    # A scoped job: the same work on a fixed number of processes, whatever the machine.
    "analyze_service_day": {
        "label": "Analyze one day of service logs",
        "capability": "demo_logs",
        "category": "operations",
        "description": "Parse 3 million seeded JSON service events on every core; error rates and p95 latency by service.",
    },
    # Analysis jobs: fixed scripts in the sandbox container on seeded data (see analysis_jobs.py).
    "simulate_credit_risk": {
        "label": "Simulate portfolio credit risk (sandbox)",
        "capability": "code_sandbox",
        "category": "finance",
        "description": "Monte Carlo of 400,000 correlated-default scenarios for a seeded 4,000-loan portfolio with numpy in the sandbox, on every core; expected loss, VaR, expected shortfall.",
    },
    "dedupe_customer_records": {
        "label": "Match duplicate customers (sandbox)",
        "capability": "code_sandbox",
        "category": "data",
        "description": "Fuzzy-match 27,000 seeded customer records from two systems by name and address in the sandbox, on every core.",
    },
    "save_artifact": {
        "label": "Save file artifact",
        "capability": "file_artifacts",
        "category": "files",
        "description": "Save named text content to the run's shared artifact workspace so later steps can read it by name instead of inlining it.",
    },
    "read_artifact": {
        "label": "Read file artifact",
        "capability": "file_artifacts",
        "category": "files",
        "description": "Read a named text artifact from the run's shared workspace.",
    },
    "list_artifacts": {
        "label": "List file artifacts",
        "capability": "file_artifacts",
        "category": "files",
        "description": "List the artifacts saved so far in the run's shared workspace.",
    },
}
ENTERPRISE_TOOLS = {
    "describe_warehouse": (
        "Describe the warehouse",
        "warehouse",
        "data",
        "List the tables, columns and row counts of the demo data warehouse (customers, orders, payments, tickets, service_events).",
    ),
    "query_warehouse": (
        "Query the warehouse (SQL)",
        "warehouse",
        "data",
        "Run one read-only SELECT against the demo data warehouse (PostgreSQL from the Intel toolkit, or SQLite) and return up to 200 rows. Tables: customers, orders, payments, tickets, service_events(event_id, at, service, endpoint, status, latency_ms, region).",
    ),
    "search_documents": (
        "Full-text search",
        "knowledge_base",
        "documents",
        "Full-text search over the knowledge base (policies, runbooks, notes and ingested documents); returns document IDs, titles and snippets.",
    ),
    "semantic_search": (
        "Semantic search (vectors)",
        "knowledge_base",
        "documents",
        "Vector similarity search over the knowledge base using embeddings stored in pgvector (or a local index); returns the closest documents.",
    ),
    "read_document": (
        "Read a document",
        "knowledge_base",
        "documents",
        "Read the full text of one knowledge base document by its ID (for example KB-0105).",
    ),
    "list_fixture_files": (
        "List files",
        "files",
        "files",
        "List the files in one demo folder: docs, reports, web, src or warehouse.",
    ),
    "read_fixture_file": (
        "Read a file",
        "files",
        "files",
        "Read a text file from the demo folders (source code, web pages, knowledge base articles). Binary files are rejected.",
    ),
    "write_source_file": (
        "Write a source file",
        "source_code",
        "code",
        "Write or replace one file of the orderflow source tree for this run (the change is kept with the run, not applied to the original).",
    ),
    "run_tests": (
        "Run the test suite (sandbox)",
        "source_code",
        "code",
        "Run the orderflow test suite with pytest in the sandbox, including any source files written during this run; returns pass and fail counts and the output tail.",
    ),
    "cache_get": (
        "Read the cache",
        "cache",
        "cache",
        "Read a value from the shared Redis cache (the toolkit's Redis, or a local table when Redis is not configured).",
    ),
    "cache_set": (
        "Write the cache",
        "cache",
        "cache",
        "Store a value in the shared Redis cache with a one-day expiry.",
    ),
}
for _name, (_label, _capability, _category, _description) in ENTERPRISE_TOOLS.items():
    TOOL_INFO[_name] = {
        "label": _label,
        "capability": _capability,
        "category": _category,
        "description": _description,
    }
# Tools whose work happens in a throwaway sandbox container (or a limited
# local process explicitly enabled for development); the UI shows these as sandbox jobs.
SANDBOX_TOOLS = {"run_tests", "simulate_credit_risk", "dedupe_customer_records"}
TOOLS = {key: value["description"] for key, value in TOOL_INFO.items()}
