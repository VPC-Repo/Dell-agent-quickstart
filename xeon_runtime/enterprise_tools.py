"""Tools that use the stack the toolkit installs: PostgreSQL with pgvector,
Redis, a code sandbox and the local demo files.

Mainstream by design: a new database connection per call, plain SQL, the
library defaults. Each tool returns JSON with a `work_evidence` record
(backend, host, units, timings) so the live view can say where the work ran.

Backends degrade explicitly, never silently:
- WAREHOUSE_URL unset -> a local SQLite warehouse (labelled as such);
- EMBEDDING_MODEL unset -> a local hashed embedding (labelled as such);
- REDIS_URL unset -> a cache table in the warehouse (labelled as such);
- Docker failure -> a failed tool call; local process execution requires an explicit
  SANDBOX_MODE=process setting for trusted development.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from sqlalchemy import create_engine, inspect, text

from . import fixtures, sandbox
from .artifacts import workspace

WAREHOUSE_VERSION = fixtures.VERSION + "+warehouse-v1"
ROW_LIMIT = 200
STOPWORDS = set(
    "a an and are as at be by do does for from how i in is it its of on or our that the this to was we what when where which who why will with you your".split()
)
TEXT_LIMIT = 24000
HASH_DIMS = 256


# --- evidence ---------------------------------------------------------------------
def _evidence(tool_name, started, cpu_started, backend, units, **extra):
    caller = time.thread_time() - cpu_started
    # Code that ran in the sandbox used the sandbox's CPU, not this thread's; both count as the tool's work.
    in_sandbox = (extra.get("sandbox") or {}).get("cpu_seconds")
    cpu = {"cpu_seconds": round(caller + (in_sandbox or 0), 4)}
    if extra.get("sandbox"):
        cpu |= {
            "caller_cpu_seconds": round(caller, 4),
            "sandbox_cpu_seconds": in_sandbox,
        }
    return {
        "tool": tool_name,
        "data": "demo data, reproducible on every install",
        "backend": backend,
        "units": units,
        **cpu,
        "wall_seconds": round(time.perf_counter() - started, 4),
        **extra,
    }


def _result(tool_name, started, cpu_started, backend, units, result, **extra):
    return json.dumps(
        {
            "result": result,
            "work_evidence": _evidence(
                tool_name, started, cpu_started, backend, units, **extra
            ),
        },
        default=str,
    )


def _host(url: str) -> str:
    try:
        return urlsplit(url).hostname or "local"
    except ValueError:
        return "local"


# --- warehouse ---------------------------------------------------------------------
def warehouse_url() -> str:
    url = os.getenv("WAREHOUSE_URL", "")
    if url.startswith(("postgresql://", "postgres://")):
        url = "postgresql+psycopg://" + url.split("://", 1)[1]
    if not url:
        Path("data").mkdir(exist_ok=True)
        url = "sqlite:///data/warehouse.sqlite3"
    return url


def warehouse_backend(url=None) -> dict:
    url = url or warehouse_url()
    if url.startswith("sqlite"):
        return {
            "kind": "sqlite",
            "host": "runner",
            "label": "SQLite (WAREHOUSE_URL not set)",
        }
    return {
        "kind": "postgresql",
        "host": _host(url),
        "label": "PostgreSQL + pgvector (toolkit)",
    }


def _engine(url=None):
    url = url or warehouse_url()
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"timeout": 30}
        if url.startswith("sqlite")
        else {"connect_timeout": 10},
    )


TABLES = {
    "customers": "customer_id TEXT PRIMARY KEY, name TEXT, region TEXT, segment TEXT, since DATE, credit_limit INTEGER",
    "orders": "order_id TEXT PRIMARY KEY, customer_id TEXT, product_id TEXT, product_name TEXT, quantity INTEGER, amount_usd NUMERIC(12,2), status TEXT, ordered_on DATE, region TEXT",
    "payments": "payment_id TEXT PRIMARY KEY, order_id TEXT, paid_usd NUMERIC(12,2), paid_on DATE, method TEXT",
    "tickets": "ticket_id TEXT PRIMARY KEY, customer_id TEXT, topic TEXT, priority TEXT, subject TEXT, opened_at TIMESTAMP, status TEXT, hours_to_resolve REAL, csat INTEGER",
    "service_events": "event_id INTEGER PRIMARY KEY, at TIMESTAMP, service TEXT, endpoint TEXT, status INTEGER, latency_ms REAL, region TEXT",
}
NUMERIC = {"quantity", "credit_limit", "status", "csat", "event_id"}
SCHEMA_HINT = (
    "Tables: customers(customer_id, name, region, segment, since, credit_limit); "
    "orders(order_id, customer_id, product_id, product_name, quantity, amount_usd, status, ordered_on, region); "
    "payments(payment_id, order_id, paid_usd, paid_on, method); "
    "tickets(ticket_id, customer_id, topic, priority, subject, opened_at, status, hours_to_resolve, csat); "
    "service_events(event_id, at, service, endpoint, status, latency_ms, region); "
    "documents(doc_id, title, kind, collection, body). "
    "Timestamps are opened_at and at: compare as text (at >= '2026-09-22 09:00') and bucket by hour with substr(CAST(at AS TEXT), 1, 13); never call substr on the raw column. ROUND needs a numeric: ROUND(CAST(x AS NUMERIC), 2). Filter on a computed column with an outer query or HAVING, not WHERE. "
    "Do not use reserved words as aliases (window, user, end, order, group): use hour_bucket, evt, etc."
)
REAL = {"amount_usd", "paid_usd", "hours_to_resolve", "latency_ms"}


def _coerce(table, row):
    out = {}
    for key, value in row.items():
        if value == "":
            out[key] = None
        elif (
            key in NUMERIC
            and not (table == "tickets" and key == "status")
            and not (table == "orders" and key == "status")
        ):
            out[key] = int(value)
        elif key in REAL:
            out[key] = float(value)
        else:
            out[key] = value
    return out


def ensure_warehouse(url=None, embed_fn=None) -> dict:
    """Create and seed the warehouse once; safe to call from many processes."""
    url = url or warehouse_url()
    engine = _engine(url)
    dialect = engine.dialect.name
    try:
        with engine.begin() as conn:
            if dialect == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(470470)"))
            else:
                conn.exec_driver_sql(
                    "BEGIN IMMEDIATE"
                ) if conn.connection.driver_connection.in_transaction is False else None
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS demo_meta (key TEXT PRIMARY KEY, value TEXT)"
                )
            )
            current = conn.execute(
                text("SELECT value FROM demo_meta WHERE key = 'version'")
            ).scalar()
            if current == WAREHOUSE_VERSION:
                return {"seeded": False, "version": current, **warehouse_backend(url)}
            root = fixtures.ensure()
            counts = {}
            for table, columns in TABLES.items():
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                conn.execute(text(f"CREATE TABLE {table} ({columns})"))
                with (root / "warehouse" / f"{table}.csv").open() as handle:
                    rows = [_coerce(table, r) for r in csv.DictReader(handle)]
                names = list(rows[0].keys())
                statement = text(
                    f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join(':' + n for n in names)})"
                )
                for offset in range(0, len(rows), 5000):
                    conn.execute(statement, rows[offset : offset + 5000])
                counts[table] = len(rows)
            for name, table in (
                ("ix_orders_customer", "orders(customer_id)"),
                ("ix_orders_status", "orders(status)"),
                ("ix_events_service", "service_events(service, at)"),
                ("ix_tickets_topic", "tickets(topic, opened_at)"),
            ):
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {table}"))
            _create_document_tables(conn, dialect)
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS cache_entries (key TEXT PRIMARY KEY, value TEXT, expires REAL)"
                )
            )
            documents = [
                {
                    "doc_id": item["doc_id"],
                    "title": item["title"],
                    "kind": item["kind"],
                    "collection": "knowledge_base",
                    "body": (root / "docs" / item["path"]).read_text(),
                }
                for item in json.loads((root / "docs" / "index.json").read_text())
            ]
            counts["documents"] = len(documents)
            embedding = _upsert_documents(conn, dialect, documents, embed_fn)
            conn.execute(text("DELETE FROM demo_meta"))
            conn.execute(
                text(
                    "INSERT INTO demo_meta (key, value) VALUES ('version', :v), ('embedding', :e), ('seeded_at', :t)"
                ),
                {"v": WAREHOUSE_VERSION, "e": embedding, "t": str(time.time())},
            )
        return {
            "seeded": True,
            "version": WAREHOUSE_VERSION,
            "counts": counts,
            "embedding": embedding,
            **warehouse_backend(url),
        }
    finally:
        engine.dispose()


def _create_document_tables(conn, dialect):
    conn.execute(text("DROP TABLE IF EXISTS documents"))
    conn.execute(text("DROP TABLE IF EXISTS doc_embeddings"))
    if dialect == "postgresql":
        conn.execute(
            text(
                "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT, kind TEXT, collection TEXT, body TEXT, "
                "tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body, ''))) STORED)"
            )
        )
        conn.execute(text("CREATE INDEX ix_documents_tsv ON documents USING GIN (tsv)"))
        has_vector = conn.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        if not has_vector:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                has_vector = True
            except Exception:
                has_vector = False
        column = (
            "embedding vector(%d)" % embedding_dims()
            if has_vector
            else "embedding TEXT"
        )
        conn.execute(
            text(
                f"CREATE TABLE doc_embeddings (doc_id TEXT PRIMARY KEY, model TEXT, dims INTEGER, {column})"
            )
        )
    else:
        conn.execute(
            text(
                "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT, kind TEXT, collection TEXT, body TEXT)"
            )
        )
        conn.execute(text("DROP TABLE IF EXISTS documents_fts"))
        conn.execute(
            text(
                "CREATE VIRTUAL TABLE documents_fts USING fts5(doc_id UNINDEXED, title, body)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE doc_embeddings (doc_id TEXT PRIMARY KEY, model TEXT, dims INTEGER, embedding TEXT)"
            )
        )


def _vector_column(conn) -> bool:
    row = conn.execute(
        text(
            "SELECT udt_name FROM information_schema.columns WHERE table_name = 'doc_embeddings' AND column_name = 'embedding'"
        )
    ).scalar()
    return row == "vector"


def _upsert_documents(conn, dialect, documents, embed_fn=None) -> str:
    embed_fn = embed_fn or embed
    ids = [d["doc_id"] for d in documents]
    if dialect == "postgresql":
        conn.execute(
            text("DELETE FROM documents WHERE doc_id = ANY(:ids)"), {"ids": ids}
        )
        conn.execute(
            text("DELETE FROM doc_embeddings WHERE doc_id = ANY(:ids)"), {"ids": ids}
        )
    else:
        for doc_id in ids:
            conn.execute(
                text("DELETE FROM documents WHERE doc_id = :id"), {"id": doc_id}
            )
            conn.execute(
                text("DELETE FROM documents_fts WHERE doc_id = :id"), {"id": doc_id}
            )
            conn.execute(
                text("DELETE FROM doc_embeddings WHERE doc_id = :id"), {"id": doc_id}
            )
    conn.execute(
        text(
            "INSERT INTO documents (doc_id, title, kind, collection, body) VALUES (:doc_id, :title, :kind, :collection, :body)"
        ),
        documents,
    )
    if dialect != "postgresql":
        conn.execute(
            text(
                "INSERT INTO documents_fts (doc_id, title, body) VALUES (:doc_id, :title, :body)"
            ),
            [
                {"doc_id": d["doc_id"], "title": d["title"], "body": d["body"]}
                for d in documents
            ],
        )
    model, vectors = embed_fn([d["title"] + "\n" + d["body"][:4000] for d in documents])
    use_vector = dialect == "postgresql" and _vector_column(conn)
    rows = [
        {
            "doc_id": d["doc_id"],
            "model": model,
            "dims": len(v),
            "embedding": "[" + ",".join(f"{x:.6f}" for x in v) + "]",
        }
        for d, v in zip(documents, vectors)
    ]
    cast = "CAST(:embedding AS vector)" if use_vector else ":embedding"
    conn.execute(
        text(
            f"INSERT INTO doc_embeddings (doc_id, model, dims, embedding) VALUES (:doc_id, :model, :dims, {cast})"
        ),
        rows,
    )
    return model


# --- embeddings ----------------------------------------------------------------------
def embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "")


def embedding_dims() -> int:
    return int(os.getenv("EMBEDDING_DIMS", "0") or 0) or (
        768 if embedding_model() else HASH_DIMS
    )


def _hash_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for value in texts:
        vector = [0.0] * HASH_DIMS
        for token in re.findall(r"[a-z0-9]+", value.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            index = int.from_bytes(digest, "big") % HASH_DIMS
            vector[index] += 1.0 if digest[0] % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        out.append([x / norm for x in vector])
    return out


def embed(texts: list[str]) -> tuple[str, list[list[float]]]:
    """(model label, vectors). Uses the gateway's embedding model when configured."""
    model = embedding_model()
    if not model:
        return "local-hash-%d" % HASH_DIMS, _hash_embed(texts)
    base = os.getenv("EMBEDDING_API_BASE") or os.getenv(
        "GATEWAY_URL", "http://agentic-litellm:4000/v1"
    )
    key = (
        os.getenv(os.getenv("EMBEDDING_API_KEY_ENV", "GATEWAY_KEY"), "")
        or "not-required"
    )
    vectors: list[list[float]] = []
    with httpx.Client(timeout=120) as client:
        for offset in range(0, len(texts), 64):
            response = client.post(
                base.rstrip("/") + "/embeddings",
                headers={"Authorization": "Bearer " + key},
                json={
                    "model": model,
                    "input": texts[offset : offset + 64],
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
    return model, vectors


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# --- the tools -----------------------------------------------------------------------
@tool
def describe_warehouse() -> str:
    """List the demo warehouse tables with their columns and row counts."""
    started, cpu = time.perf_counter(), time.thread_time()
    ensure_warehouse()
    engine = _engine()
    try:
        inspector = inspect(engine)
        tables = {}
        with engine.connect() as conn:
            for name in TABLES:
                tables[name] = {
                    "columns": [
                        f"{c['name']} {c['type']}" for c in inspector.get_columns(name)
                    ],
                    "rows": conn.execute(text(f"SELECT count(*) FROM {name}")).scalar(),
                }
            tables["documents"] = {
                "columns": ["doc_id", "title", "kind", "collection", "body"],
                "rows": conn.execute(text("SELECT count(*) FROM documents")).scalar(),
            }
        backend = warehouse_backend()
        return _result(
            "describe_warehouse",
            started,
            cpu,
            backend,
            {"tables": len(tables)},
            {
                "tables": tables,
                "dialect": engine.dialect.name,
                "note": "Demo data, reproducible on every install.",
            },
        )
    finally:
        engine.dispose()


def clean_sql(sql: str) -> tuple[str, bool]:
    """Normalize a model-written statement: drop code fences and `--` or block comments outside string
    literals, trailing semicolons and whitespace. Returns (statement, has_second_statement).

    The model writes SQL the way people do (a comment line first, a fence, `;` then a note); none of that
    is a second statement, and rejecting it only burns the worker's steps."""
    text = sql.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    out: list[str] = []
    i, n, quote, extra = 0, len(text), "", False
    while i < n:
        c = text[i]
        if quote:
            out.append(c)
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            out.append(c)
        elif text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        elif c == ";":
            # A semicolon followed only by whitespace or comments is a terminator, not a second statement.
            rest = re.sub(r"(--[^\n]*|/\*.*?\*/|\s)", "", text[i + 1 :], flags=re.S)
            if rest:
                extra = True
            i += 1
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out).strip().rstrip(";").strip(), extra


def _split_args(text: str) -> list[str]:
    """Top-level comma split of a call's argument text (parentheses and quotes respected)."""
    args, depth, quote, cur = [], 0, None, []
    for c in text:
        if quote:
            cur.append(c)
            if c == quote:
                quote = None
            continue
        if c in ("'", '"'):
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        cur.append(c)
    if cur:
        args.append("".join(cur).strip())
    return args


def _rewrite_calls(statement: str, name: str, rewrite) -> str:
    """Replace every `name(args)` call (identifier boundary, nesting and quotes respected) by rewrite(args) or leave it."""
    out, i, lower, n = [], 0, statement.lower(), len(name) + 1
    while True:
        j = lower.find(name + "(", i)
        if j < 0:
            out.append(statement[i:])
            return "".join(out)
        if j and (statement[j - 1].isalnum() or statement[j - 1] == "_"):
            out.append(statement[i : j + n])
            i = j + n
            continue
        depth, k, quote = 0, j + n, None
        while k < len(statement):
            c = statement[k]
            if quote:
                if c == quote:
                    quote = None
            elif c in ("'", '"'):
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    break
                depth -= 1
            k += 1
        if k >= len(statement):
            out.append(statement[i:])
            return "".join(out)
        replacement = rewrite(_split_args(statement[j + n : k]))
        out.append(
            statement[i:j]
            + (replacement if replacement is not None else statement[j : k + 1])
        )
        i = k + 1


_STRFTIME = [
    ("%Y", "YYYY"),
    ("%m", "MM"),
    ("%d", "DD"),
    ("%H", "HH24"),
    ("%M", "MI"),
    ("%S", "SS"),
    ("%j", "DDD"),
    ("%W", "IW"),
]


def _is_cast_or_literal(expr: str) -> bool:
    e = expr.lower()
    return e.startswith("cast(") or e.startswith("'") or e.startswith("to_char(")


def portable_sql(statement: str, dialect: str) -> str:
    """Make the model's SQL run on PostgreSQL. Models write the portable-looking SQLite idioms they know best;
    these are the ones that failed in recorded runs, each fixed here so the worker does not spend a retry:

    - ROUND(x, n) needs a NUMERIC: wrap x in a CAST;
    - substr(date_column, ...) needs text: wrap the first argument in CAST(... AS TEXT);
    - strftime('%Y-%m', col) does not exist: to_char(CAST(col AS TIMESTAMP), 'YYYY-MM').
    Other dialects are left alone."""
    if dialect != "postgresql":
        return statement

    def round_(args):
        if len(args) >= 2 and not args[0].lower().startswith("cast("):
            return (
                "ROUND(" + ", ".join([f"CAST({args[0]} AS NUMERIC)", *args[1:]]) + ")"
            )
        return "ROUND(" + ", ".join(args) + ")"

    def substr_(args):
        if args and not _is_cast_or_literal(args[0]):
            return "substr(" + ", ".join([f"CAST({args[0]} AS TEXT)", *args[1:]]) + ")"
        return None

    def strftime_(args):
        if len(args) == 2 and args[0].startswith("'") and args[0].endswith("'"):
            fmt = args[0][1:-1]
            for a, b in _STRFTIME:
                fmt = fmt.replace(a, b)
            return f"to_char(CAST({args[1]} AS TIMESTAMP), '{fmt}')"
        return None

    statement = _rewrite_calls(statement, "round", round_)
    statement = _rewrite_calls(statement, "substr", substr_)
    statement = _rewrite_calls(statement, "substring", substr_)
    return _rewrite_calls(statement, "strftime", strftime_)


@tool
def query_warehouse(sql: str) -> str:
    """Run one read-only SELECT (or WITH) statement against the demo warehouse and return up to 200 rows.
    Tables: customers(customer_id, name, region, segment, since, credit_limit); orders(order_id, customer_id,
    product_id, product_name, quantity, amount_usd, status, ordered_on, region); payments(payment_id, order_id,
    paid_usd, paid_on, method); tickets(ticket_id, customer_id, topic, priority, subject, opened_at, status,
    hours_to_resolve, csat); service_events(event_id, at, service, endpoint, status, latency_ms, region).
    Use standard SQL; compare timestamps as text (at >= '2026-09-22 09:00'), bucket hours with
    substr(CAST(at AS TEXT), 1, 13), round with ROUND(CAST(x AS NUMERIC), 2), and never alias as window/user/end."""
    started, cpu = time.perf_counter(), time.thread_time()
    statement, extra = clean_sql(sql)
    if not re.match(r"(?is)^\s*(select|with)\b", statement) or extra:
        raise ValueError(
            f"Submit one SELECT or WITH statement (no second statement{', no semicolons' if extra else ''}). "
            f"Received: {statement[:160]!r}. " + SCHEMA_HINT
        )
    ensure_warehouse()
    engine = _engine()
    try:
        try:
            with engine.begin() as conn:
                if engine.dialect.name == "postgresql":
                    conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                    conn.execute(
                        text("SELECT set_config('statement_timeout', '30000', true)")
                    )
                else:
                    conn.exec_driver_sql("PRAGMA query_only=ON")
                result = conn.execute(
                    text(portable_sql(statement, engine.dialect.name))
                )
                columns = list(result.keys())
                rows = result.fetchmany(ROW_LIMIT + 1)
        except Exception as exc:
            # The database's own message, then the schema, so the model can correct the query.
            detail = str(getattr(exc, "orig", exc)).split("\n[SQL")[0].strip()
            raise ValueError(f"SQL failed: {detail[:400]}. {SCHEMA_HINT}") from None
        backend = warehouse_backend()
        return _result(
            "query_warehouse",
            started,
            cpu,
            backend,
            {"rows_returned": min(len(rows), ROW_LIMIT)},
            {
                "columns": columns,
                "rows": [list(r) for r in rows[:ROW_LIMIT]],
                "truncated": len(rows) > ROW_LIMIT,
                "rows_total": len(rows),
                "sql": statement[:2000],
                # A model that reads a silently cut result concludes "nothing unusual" (gpt-oss-120b on the incident
                # analysis: 702 service-hour rows, the incident beyond row 200). Say so where it reads the rows.
                **(
                    {
                        "warning": f"Only the first {ROW_LIMIT} of {len(rows)} rows are shown, in no particular order, so the rows you need may be missing. Aggregate by fewer columns, filter, or ORDER BY the measure DESC with a LIMIT, then query again."
                    }
                    if len(rows) > ROW_LIMIT
                    else {}
                ),
                # Qwen3-Coder reran an unchanged zero-row query three times in a row; say what to change instead.
                **(
                    {
                        "note": "No rows matched. Running the same statement again returns nothing again: check the literal values you filtered on (SELECT DISTINCT <column> FROM <table>) and the date comparisons, then change the query."
                    }
                    if not rows
                    else {}
                ),
            },
        )
    finally:
        engine.dispose()


@tool
def search_documents(query: str, limit: int = 5) -> str:
    """Full-text search over the knowledge base; returns matching document IDs, titles and snippets."""
    started, cpu = time.perf_counter(), time.thread_time()
    limit = max(1, min(int(limit), 20))
    ensure_warehouse()
    engine = _engine()
    try:
        with engine.connect() as conn:
            terms = [
                t
                for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", query)
                if t.lower() not in STOPWORDS
            ]
            if engine.dialect.name == "postgresql":
                # Any-term match, ranked: a question rarely contains every word of the answer.
                rows = conn.execute(
                    text(
                        "SELECT doc_id, title, kind, collection, ts_rank(tsv, q) AS score, "
                        "ts_headline('english', body, q, 'MaxWords=40, MinWords=20') AS snippet "
                        "FROM documents, to_tsquery('english', :q) q WHERE tsv @@ q ORDER BY score DESC LIMIT :n"
                    ),
                    {
                        "q": " | ".join(t.replace("-", "") for t in terms) or "empty",
                        "n": limit,
                    },
                ).fetchall()
                method = "PostgreSQL tsvector (GIN)"
            else:
                match = " OR ".join('"' + t + '"' for t in terms) or '""'
                rows = conn.execute(
                    text(
                        "SELECT d.doc_id, d.title, d.kind, d.collection, -bm25(documents_fts) AS score, "
                        "snippet(documents_fts, 2, '[', ']', '…', 24) AS snippet FROM documents_fts "
                        "JOIN documents d ON d.doc_id = documents_fts.doc_id WHERE documents_fts MATCH :q ORDER BY bm25(documents_fts) LIMIT :n"
                    ),
                    {"q": match, "n": limit},
                ).fetchall()
                method = "SQLite FTS5 (BM25)"
        hits = [
            {
                "doc_id": r[0],
                "title": r[1],
                "kind": r[2],
                "collection": r[3],
                "score": round(float(r[4] or 0), 4),
                "snippet": r[5],
            }
            for r in rows
        ]
        backend = {**warehouse_backend(), "method": method}
        return _result(
            "search_documents",
            started,
            cpu,
            backend,
            {"hits": len(hits)},
            {"query": query, "hits": hits},
        )
    finally:
        engine.dispose()


@tool
def semantic_search(query: str, limit: int = 5) -> str:
    """Vector similarity search over the knowledge base embeddings; returns the closest documents."""
    started, cpu = time.perf_counter(), time.thread_time()
    limit = max(1, min(int(limit), 20))
    ensure_warehouse()
    model, (vector,) = embed([query])
    engine = _engine()
    try:
        with engine.connect() as conn:
            literal = "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
            if engine.dialect.name == "postgresql" and _vector_column(conn):
                rows = conn.execute(
                    text(
                        "SELECT e.doc_id, d.title, d.kind, d.collection, 1 - (e.embedding <=> CAST(:v AS vector)) AS score "
                        "FROM doc_embeddings e JOIN documents d ON d.doc_id = e.doc_id ORDER BY e.embedding <=> CAST(:v AS vector) LIMIT :n"
                    ),
                    {"v": literal, "n": limit},
                ).fetchall()
                method = "pgvector cosine (<=>)"
                scored = [(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]
            else:
                rows = conn.execute(
                    text(
                        "SELECT e.doc_id, d.title, d.kind, d.collection, e.embedding FROM doc_embeddings e JOIN documents d ON d.doc_id = e.doc_id"
                    )
                ).fetchall()
                scored = sorted(
                    (
                        (r[0], r[1], r[2], r[3], _cosine(vector, json.loads(r[4])))
                        for r in rows
                    ),
                    key=lambda r: -r[4],
                )[:limit]
                method = "brute-force cosine over stored vectors"
        hits = [
            {
                "doc_id": r[0],
                "title": r[1],
                "kind": r[2],
                "collection": r[3],
                "score": round(r[4], 4),
            }
            for r in scored
        ]
        backend = {**warehouse_backend(), "method": method, "embedding_model": model}
        return _result(
            "semantic_search",
            started,
            cpu,
            backend,
            {"hits": len(hits), "dims": len(vector)},
            {"query": query, "hits": hits},
        )
    finally:
        engine.dispose()


@tool
def read_document(doc_id: str) -> str:
    """Read one knowledge base document by ID (for example KB-0105)."""
    started, cpu = time.perf_counter(), time.thread_time()
    ensure_warehouse()
    engine = _engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT doc_id, title, kind, collection, body FROM documents WHERE doc_id = :id"
                ),
                {"id": doc_id.strip()},
            ).fetchone()
        if not row:
            raise ValueError(
                f"No document {doc_id!r}; use search_documents to find IDs"
            )
        body = row[4]
        return _result(
            "read_document",
            started,
            cpu,
            warehouse_backend(),
            {"characters": len(body)},
            {
                "doc_id": row[0],
                "title": row[1],
                "kind": row[2],
                "collection": row[3],
                "text": body[:TEXT_LIMIT],
                "truncated": len(body) > TEXT_LIMIT,
            },
        )
    finally:
        engine.dispose()


FOLDERS = {"docs", "reports", "web", "src", "warehouse"}
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".html",
    ".json",
    ".csv",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
}


def _fixture_path(path: str) -> Path:
    """Resolve a path inside the demo folders. Models often omit the folder
    (orderflow/pricing.py, status.html): try each folder before refusing."""
    root = fixtures.ensure().resolve()
    clean = path.strip().lstrip("/").removeprefix("fixtures/").removeprefix("./")
    candidates = [clean] + [
        f"{folder}/{clean}" for folder in ("src", "web", "docs", "reports", "warehouse")
    ]
    for candidate in candidates:
        target = (root / candidate).resolve()
        if (
            root in target.parents
            and len(target.parts) > len(root.parts)
            and target.parts[len(root.parts)] in FOLDERS
        ):
            if target.exists() or candidate == clean:
                return target
    raise ValueError(
        "Paths are relative to the demo folders: docs, reports, web, src, warehouse (for example src/orderflow/pricing.py)"
    )


@tool
def list_fixture_files(folder: str) -> str:
    """List the files in one demo folder: docs, reports, web, src or warehouse."""
    started, cpu = time.perf_counter(), time.thread_time()
    if folder.strip("/") not in FOLDERS:
        raise ValueError("Choose one of: " + ", ".join(sorted(FOLDERS)))
    root = _fixture_path(folder.strip("/"))
    files = [
        {"path": str(p.relative_to(root.parent)), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]
    return _result(
        "list_fixture_files",
        started,
        cpu,
        {"kind": "files", "host": "runner"},
        {"files": len(files)},
        {"folder": folder, "files": files[:400], "truncated": len(files) > 400},
    )


@tool
def read_fixture_file(path: str) -> str:
    """Read a text file from the demo folders, for example src/orderflow/pricing.py or web/status.html."""
    started, cpu = time.perf_counter(), time.thread_time()
    target = _fixture_path(path)
    if not target.is_file():
        raise ValueError(f"No file at {path!r}; use list_fixture_files first")
    if target.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError("Only text files can be read here")
    content = target.read_text(errors="replace")
    return _result(
        "read_fixture_file",
        started,
        cpu,
        {"kind": "files", "host": "runner"},
        {"characters": len(content)},
        {
            "path": path,
            "text": content[:TEXT_LIMIT],
            "truncated": len(content) > TEXT_LIMIT,
        },
    )


def _sandbox_units(result):
    return {
        "sandbox": result["sandbox"]["kind"],
        "wall_seconds": result["sandbox"]["wall_seconds"],
        "exit_code": result["exit_code"],
        "timed_out": result["timed_out"],
    }


def _overlay_dir(config) -> Path:
    path = workspace(config) / "src-overlay"
    path.mkdir(parents=True, exist_ok=True)
    return path


@tool
def write_source_file(path: str, content: str, config: RunnableConfig) -> str:
    """Write or replace one file of the orderflow source tree for this run, for example orderflow/pricing.py. The original files are not modified; run_tests uses your version."""
    started, cpu = time.perf_counter(), time.thread_time()
    clean = Path(path.strip().removeprefix("src/"))
    if (
        clean.is_absolute()
        or ".." in clean.parts
        or not clean.parts
        or clean.suffix not in TEXT_SUFFIXES
    ):
        raise ValueError(
            "Give a relative text-file path inside the source tree, such as orderflow/pricing.py"
        )
    if len(content) > 200000:
        raise ValueError("Files are limited to 200,000 characters")
    target = _overlay_dir(config) / clean
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return _result(
        "write_source_file",
        started,
        cpu,
        {"kind": "run workspace", "host": "runner"},
        {"files": 1, "characters": len(content)},
        {"path": str(clean), "written": True},
    )


@tool
def run_tests(config: RunnableConfig) -> str:
    """Run the orderflow test suite with pytest in the sandbox, using any files written with write_source_file in this run. Returns pass and fail counts and the output tail."""
    started, cpu = time.perf_counter(), time.thread_time()
    files = fixtures.tree(fixtures.ensure() / "src")
    overlay = _overlay_dir(config)
    changed = []
    for path in sorted(overlay.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(overlay))
            files[relative] = path.read_bytes()
            changed.append(relative)
    result = sandbox.run(
        ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        files,
        timeout=120,
        cpus=2.0,
        memory_mb=1024,
    )
    output = result["stdout"] + ("\n" + result["stderr"] if result["stderr"] else "")
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", output))
    failed = sum(int(n) for n in re.findall(r"(\d+) failed", output))
    errors = sum(int(n) for n in re.findall(r"(\d+) error", output))
    return _result(
        "run_tests",
        started,
        cpu,
        {"kind": "sandbox", **result["sandbox"]},
        {
            "tests_passed": passed,
            "tests_failed": failed,
            "errors": errors,
            "files_changed": len(changed),
            **_sandbox_units(result),
        },
        {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "exit_code": result["exit_code"],
            "changed_files": changed,
            "output_tail": output[-6000:],
        },
        sandbox=result["sandbox"],
    )


# --- cache ---------------------------------------------------------------------------
def _redis():
    url = os.getenv("REDIS_URL", "")
    if not url:
        return None
    import redis

    return redis.Redis.from_url(
        url, socket_connect_timeout=3, socket_timeout=5, decode_responses=True
    )


def _cache_key(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9:._-]{1,120}", key or ""):
        raise ValueError(
            "Cache keys are 1-120 letters, digits, colon, dot, dash or underscore"
        )
    return "quickstart:" + key


@tool
def cache_get(key: str) -> str:
    """Read a value from the shared cache (Redis). Returns found=false when the key is absent."""
    started, cpu = time.perf_counter(), time.thread_time()
    full = _cache_key(key)
    client = _redis()
    if client is not None:
        value = client.get(full)
        backend = {"kind": "redis", "host": _host(os.getenv("REDIS_URL", ""))}
    else:
        ensure_warehouse()
        engine = _engine()
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT value, expires FROM cache_entries WHERE key = :k"),
                    {"k": full},
                ).fetchone()
        finally:
            engine.dispose()
        value = row[0] if row and row[1] > time.time() else None
        backend = {**warehouse_backend(), "kind": "table (REDIS_URL not set)"}
    return _result(
        "cache_get",
        started,
        cpu,
        backend,
        {"keys": 1, "hit": int(value is not None)},
        {"key": key, "found": value is not None, "value": value},
    )


@tool
def cache_set(key: str, value: str) -> str:
    """Store a value in the shared cache (Redis) for one day."""
    started, cpu = time.perf_counter(), time.thread_time()
    full = _cache_key(key)
    if len(value) > 20000:
        raise ValueError("Cached values are limited to 20,000 characters")
    client = _redis()
    if client is not None:
        client.set(full, value, ex=86400)
        backend = {"kind": "redis", "host": _host(os.getenv("REDIS_URL", ""))}
    else:
        ensure_warehouse()
        engine = _engine()
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM cache_entries WHERE key = :k"), {"k": full}
                )
                conn.execute(
                    text(
                        "INSERT INTO cache_entries (key, value, expires) VALUES (:k, :v, :e)"
                    ),
                    {"k": full, "v": value, "e": time.time() + 86400},
                )
        finally:
            engine.dispose()
        backend = {**warehouse_backend(), "kind": "table (REDIS_URL not set)"}
    return _result(
        "cache_set",
        started,
        cpu,
        backend,
        {"keys": 1, "characters": len(value)},
        {"key": key, "stored": True, "ttl_seconds": 86400},
    )


# A failing call comes back to the model as an error message (the worker may
# correct its request) instead of ending the run.
REGISTRY = {
    t.name: t
    for t in [
        describe_warehouse,
        query_warehouse,
        search_documents,
        semantic_search,
        read_document,
        list_fixture_files,
        read_fixture_file,
        write_source_file,
        run_tests,
        cache_get,
        cache_set,
    ]
}


def _guarded(fn):
    """Turn a tool failure into a ToolException so LangChain returns it to the
    model as an error message instead of ending the run."""
    from functools import wraps

    from langchain_core.tools import ToolException

    @wraps(fn)
    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolException:
            raise
        except sandbox.SandboxUnavailable as exc:
            raise ToolException("Sandbox unavailable: " + str(exc)) from None
        except (ValueError, LookupError, OSError, httpx.HTTPError) as exc:
            raise ToolException(str(exc)[:600] or type(exc).__name__) from None
        except Exception as exc:  # database and driver errors carry useful text
            raise ToolException(f"{type(exc).__name__}: {str(exc)[:600]}") from None

    return call


for _tool in REGISTRY.values():
    _tool.func = _guarded(_tool.func)
    _tool.handle_tool_error = True
