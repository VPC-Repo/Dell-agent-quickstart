"""Fixtures, warehouse, search, sandbox and cache tools on the local backends."""

import json
import os

import pytest

from xeon_runtime import enterprise_tools as et
from xeon_runtime import fixtures, sandbox
from xeon_runtime.engine import tools_for
from xeon_runtime.planning import WorkerDefinition, worker_blueprint
from xeon_runtime.readiness import inspect_goal, worker_contract
from xeon_runtime.specs import Blueprint
from xeon_runtime.tool_catalog import ENTERPRISE_TOOLS, SANDBOX_TOOLS, TOOL_INFO


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FIXTURE_DIR", str(tmp_path / "fixtures"))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SANDBOX_MODE", "process")
    monkeypatch.delenv("WAREHOUSE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    return tmp_path


def call(tool, **kwargs):
    config = kwargs.pop("config", None)
    return json.loads(
        tool.invoke(kwargs, config=config) if config else tool.invoke(kwargs)
    )


def test_fixtures_are_deterministic_and_carry_the_planted_facts(local):
    root = fixtures.ensure()
    first = fixtures.checksum(root / "warehouse")
    (root / "VERSION").unlink()
    fixtures.ensure()
    assert fixtures.checksum(root / "warehouse") == first
    assert (
        (root / "reports" / "Q2-2026-operations-review.pdf")
        .read_bytes()
        .startswith(b"%PDF-1.4")
    )
    assert len(json.loads((root / "docs" / "index.json").read_text())) == 400
    assert "45 days" in (root / "docs" / "KB-0101.md").read_text()


def test_catalog_lists_every_enterprise_tool_and_the_engine_can_bind_them(local):
    assert set(ENTERPRISE_TOOLS) == set(et.REGISTRY)
    assert SANDBOX_TOOLS <= set(et.REGISTRY)
    assert not any(name.startswith("reference_") for name in TOOL_INFO)
    for name in et.REGISTRY:
        w = {
            **WorkerDefinition(name="W", instructions="x", tools=[name]).model_dump(),
            "ref": "w@1",
        }
        bp = Blueprint.model_validate(worker_blueprint(w, "demo"))
        assert [t.name for t in tools_for(bp)] == [name]
        assert inspect_goal("", [TOOL_INFO[name]["capability"]])["status"] == "ready"
    contract = worker_contract({"tools": ["run_python"]})
    assert "container" in contract["sandbox"]
    assert (
        worker_contract({"tools": ["query_warehouse"]})["sandbox"] == "Not provisioned"
    )


def test_warehouse_seeds_once_and_answers_read_only_sql(local):
    seeded = et.ensure_warehouse()
    assert (
        seeded["seeded"]
        and seeded["counts"]["orders"] == 50000
        and seeded["kind"] == "sqlite"
    )
    assert et.ensure_warehouse()["seeded"] is False
    described = call(et.describe_warehouse)
    assert described["result"]["tables"]["service_events"]["rows"] == 60000
    assert (
        "source_id" not in described
        and described["work_evidence"]["backend"]["kind"] == "sqlite"
    )
    # Planted churn: 40 customers ordered early in 2026, none after June, and each holds an open ticket.
    churn = call(
        et.query_warehouse,
        sql="SELECT count(*) AS n FROM (SELECT o.customer_id FROM orders o GROUP BY o.customer_id HAVING max(o.ordered_on) <= '2026-06-30' AND min(o.ordered_on) < '2026-04-01') c JOIN (SELECT DISTINCT customer_id FROM tickets WHERE status <> 'resolved') t ON t.customer_id = c.customer_id",
    )["result"]["rows"]
    assert churn[0][0] == 40
    rows = call(
        et.query_warehouse,
        sql="SELECT status, count(*) AS n FROM orders GROUP BY status ORDER BY n DESC",
    )
    assert rows["result"]["rows"][0][0] == "delivered" and rows["result"][
        "columns"
    ] == ["status", "n"]
    short = call(
        et.query_warehouse,
        sql="SELECT count(*), round(sum(o.amount_usd - p.paid_usd), 2) FROM orders o JOIN payments p ON p.order_id = o.order_id WHERE p.paid_usd < o.amount_usd",
    )
    count, total = short["result"]["rows"][0]
    assert (
        count == 113 and total == count * 25
    )  # the planted 25.00 shortfall on every 400th paid order
    # A long result says what was cut, so the model does not read 200 of 702 service-hours as "no incident".
    long = call(
        et.query_warehouse,
        sql="SELECT service, substr(CAST(at AS TEXT), 1, 13) AS h, count(*) FROM service_events GROUP BY 1, 2",
    )["result"]
    assert (
        long["truncated"]
        and long["rows_total"] > 200
        and len(long["rows"]) == 200
        and "200 of" in long["warning"]
    )
    assert "warning" not in rows["result"]
    with pytest.raises(ValueError):
        call(et.query_warehouse, sql="DELETE FROM orders")
    with pytest.raises(Exception):
        call(et.query_warehouse, sql="SELECT 1; DROP TABLE orders")


def test_full_text_vector_and_ingestion(local):
    hits = call(et.search_documents, query="How long is the refund window?", limit=3)[
        "result"
    ]["hits"]
    assert hits and hits[0]["doc_id"] == "KB-0101"
    semantic = call(
        et.semantic_search, query="fleet api 429 storm export throttling", limit=3
    )
    assert semantic["result"]["hits"][0]["doc_id"] == "KB-0107"
    assert semantic["work_evidence"]["backend"]["embedding_model"] == "local-hash-256"
    doc = call(et.read_document, doc_id="KB-0107")["result"]
    assert "throttling" in doc["text"]
    indexed = call(
        et.index_document,
        doc_id="ING-001",
        title="Q2 review",
        text_content="Payment exceptions short by 25.00 USD with the processor.",
    )
    assert indexed["result"]["indexed"]
    assert (
        call(et.search_documents, query="payment exceptions processor", limit=1)[
            "result"
        ]["hits"][0]["doc_id"]
        == "ING-001"
    )
    with pytest.raises(ValueError):
        call(et.read_document, doc_id="KB-9999")


def test_files_html_pdf_and_sandbox_tools(local):
    files = call(et.list_fixture_files, folder="src")["result"]["files"]
    assert any(f["path"] == "src/orderflow/pricing.py" for f in files)
    with pytest.raises(ValueError):
        call(et.read_fixture_file, path="../../etc/passwd")
    with pytest.raises(ValueError):
        call(et.read_fixture_file, path="reports/Q1-2026-operations-review.pdf")
    page = call(et.extract_html, path="status.html")["result"]
    assert (
        page["title"] == "Service status"
        and "fleet-api" in page["text"]
        and "Cookie notice" not in page["text"]
    )
    pdf = call(et.extract_pdf_text, name="fleet-api-capacity-plan.pdf")
    assert pdf["result"]["pages"] == 1 and "600 requests" in pdf["result"]["text"]
    assert pdf["work_evidence"]["sandbox"]["kind"] == "process"
    run = call(
        et.run_python,
        code="import csv\nprint(sum(1 for _ in open('fixtures/warehouse/orders.csv')) - 1)",
    )
    assert (
        run["result"]["stdout"].strip() == "50000" and run["result"]["exit_code"] == 0
    )
    assert run["work_evidence"]["units"]["sandbox"] == "process"


def test_fix_loop_runs_tests_against_the_run_overlay(local):
    config = {"configurable": {"thread_id": "run-1:fixer"}}
    before = call(et.run_tests, config=config)["result"]
    assert (
        before["failed"] == 1
        and before["passed"] == 6
        and before["changed_files"] == []
    )
    source = fixtures.SOURCE["orderflow/pricing.py"].replace(
        "Decimal(str(round(float(amount * factor), 2)))", "(amount * factor)"
    )
    written = call(
        et.write_source_file,
        path="src/orderflow/pricing.py",
        content=source,
        config=config,
    )
    assert written["result"]["path"] == "orderflow/pricing.py"
    after = call(et.run_tests, config=config)["result"]
    assert (
        after["failed"] == 0
        and after["passed"] == 7
        and after["changed_files"] == ["orderflow/pricing.py"]
    )
    # Another run sees the original tree.
    other = call(et.run_tests, config={"configurable": {"thread_id": "run-2:fixer"}})[
        "result"
    ]
    assert other["failed"] == 1
    with pytest.raises(ValueError):
        call(et.write_source_file, path="../escape.py", content="x", config=config)


def test_cache_falls_back_to_a_table_and_reports_it(local):
    missing = call(et.cache_get, key="faq:refund")
    assert (
        missing["result"]["found"] is False
        and "REDIS_URL" in missing["work_evidence"]["backend"]["kind"]
    )
    call(et.cache_set, key="faq:refund", value="45 days")
    assert call(et.cache_get, key="faq:refund")["result"]["value"] == "45 days"
    with pytest.raises(ValueError):
        call(et.cache_get, key="bad key!")


def test_process_sandbox_enforces_timeout_and_reports_limits(local):
    result = sandbox.run_python(
        "import time\nwhile True:\n    time.sleep(0.1)", timeout=2
    )
    assert result["timed_out"] and result["exit_code"] == -9
    assert (
        result["sandbox"]["kind"] == "process"
        and result["sandbox"]["timeout_seconds"] == 2
    )
    ok = sandbox.run(["python3", "-c", "print(2 + 2)"], {})
    assert ok["stdout"].strip() == "4" and ok["exit_code"] == 0


@pytest.mark.skipif(
    not os.getenv("QUICKSTART_TEST_DOCKER"),
    reason="set QUICKSTART_TEST_DOCKER=1 with a Docker socket",
)
def test_docker_sandbox_isolates_network(local, monkeypatch):
    monkeypatch.setenv("SANDBOX_MODE", "docker")
    result = sandbox.run_python(
        "import urllib.request\n"
        "try:\n    urllib.request.urlopen('http://example.com', timeout=3)\n"
        "except Exception as e:\n    print('blocked', type(e).__name__)",
        timeout=30,
    )
    assert result["sandbox"]["kind"] == "docker" and result["stdout"].startswith(
        "blocked"
    )


@pytest.mark.skipif(
    not os.getenv("QUICKSTART_TEST_PG"),
    reason="set QUICKSTART_TEST_PG to a pgvector PostgreSQL URL",
)
def test_postgres_warehouse_uses_tsvector_and_pgvector(local, monkeypatch):
    monkeypatch.setenv("WAREHOUSE_URL", os.environ["QUICKSTART_TEST_PG"])
    seeded = et.ensure_warehouse()
    assert seeded["kind"] == "postgresql"
    fts = call(et.search_documents, query="refund window", limit=2)
    assert (
        fts["result"]["hits"][0]["doc_id"] == "KB-0101"
        and "tsvector" in fts["work_evidence"]["backend"]["method"]
    )
    vec = call(et.semantic_search, query="fleet api 429 storm", limit=2)
    assert (
        vec["result"]["hits"][0]["doc_id"] == "KB-0107"
        and "pgvector" in vec["work_evidence"]["backend"]["method"]
    )


def test_folderless_paths_resolve_and_tool_errors_return_to_the_model(local):
    assert (
        "apply_discount"
        in call(et.read_fixture_file, path="orderflow/pricing.py")["result"]["text"]
    )
    assert (
        call(et.read_fixture_file, path="./status.html")["result"]["path"]
        == "./status.html"
    )
    assert call(et.list_fixture_files, folder="src")["result"]["files"]
    # An unknown path is an error message for the model, not an exception for the run.
    message = et.read_fixture_file.invoke(
        {
            "name": "read_fixture_file",
            "args": {"path": "nope/missing.py"},
            "id": "c1",
            "type": "tool_call",
        }
    )
    assert message.status == "error" and "demo folders" in str(message.content)
    with pytest.raises(ValueError):
        call(et.read_fixture_file, path="../../etc/passwd")


def test_portable_sql_casts_round_arguments_for_postgresql():
    from xeon_runtime.enterprise_tools import portable_sql

    sql = "SELECT ROUND(AVG(hours_to_resolve), 2) AS h, round(sum(o.amount_usd - p.paid_usd), 2), ROUND(x) FROM t"
    assert (
        portable_sql(sql, "postgresql")
        == "SELECT ROUND(CAST(AVG(hours_to_resolve) AS NUMERIC), 2) AS h, ROUND(CAST(sum(o.amount_usd - p.paid_usd) AS NUMERIC), 2), ROUND(x) FROM t"
    )
    assert (
        portable_sql(
            "SELECT ROUND(CAST(a AS NUMERIC), 1), around(1, 2) FROM t", "postgresql"
        )
        == "SELECT ROUND(CAST(a AS NUMERIC), 1), around(1, 2) FROM t"
    )
    assert portable_sql(sql, "sqlite") == sql
    # SQLite idioms the coder model wrote against PostgreSQL: strftime and substr on a date.
    assert (
        portable_sql(
            "SELECT strftime('%Y-%m', ordered_on) AS month, SUM(amount_usd) FROM orders GROUP BY 1",
            "postgresql",
        )
        == "SELECT to_char(CAST(ordered_on AS TIMESTAMP), 'YYYY-MM') AS month, SUM(amount_usd) FROM orders GROUP BY 1"
    )
    assert (
        portable_sql(
            "SELECT substr(at, 1, 13) AS hour_bucket, count(*) FROM service_events GROUP BY 1",
            "postgresql",
        )
        == "SELECT substr(CAST(at AS TEXT), 1, 13) AS hour_bucket, count(*) FROM service_events GROUP BY 1"
    )
    assert (
        portable_sql(
            "SELECT substr(CAST(at AS TEXT), 1, 13), substr('a,b', 1, 1), subs(x) FROM t",
            "postgresql",
        )
        == "SELECT substr(CAST(at AS TEXT), 1, 13), substr('a,b', 1, 1), subs(x) FROM t"
    )


def test_clean_sql_accepts_human_shaped_statements_and_rejects_a_second_one():
    from xeon_runtime.enterprise_tools import clean_sql

    assert clean_sql("-- churned customers\nSELECT 1;\n") == ("SELECT 1", False)
    assert clean_sql("```sql\nSELECT 1; -- done\n```") == ("SELECT 1", False)
    assert clean_sql("/* note */ WITH x AS (SELECT 1) SELECT * FROM x;") == (
        "WITH x AS (SELECT 1) SELECT * FROM x",
        False,
    )
    assert clean_sql("SELECT ';' AS semi, \"a;b\" FROM t") == (
        "SELECT ';' AS semi, \"a;b\" FROM t",
        False,
    )
    assert clean_sql("SELECT 1; DELETE FROM orders")[1] is True
    assert clean_sql("SELECT 1;\nSELECT 2")[1] is True


@pytest.mark.skipif(
    not os.getenv("QUICKSTART_TEST_DOCKER"),
    reason="set QUICKSTART_TEST_DOCKER=1 with a Docker socket",
)
def test_docker_tools_extract_pdf_and_run_the_fixture_tests(local, monkeypatch):
    monkeypatch.setenv("SANDBOX_MODE", "docker")
    pdf = call(et.extract_pdf_text, name="Q2-2026-operations-review.pdf")
    assert pdf["work_evidence"]["sandbox"]["kind"] == "docker"
    assert pdf["result"]["pages"] >= 1 and pdf["result"]["text"]
    suite = call(et.run_tests, config={"configurable": {"thread_id": "docker-check"}})
    assert suite["result"]["passed"] > 0
    assert (
        suite["result"]["failed"] > 0
    )  # the coding example starts with a deliberate bug
    assert suite["work_evidence"]["sandbox"]["kind"] == "docker"
    result = sandbox.run_python("import numpy, pandas; print('ready')")
    assert result["exit_code"] == 0 and result["stdout"].strip() == "ready"
    timed = sandbox.run_python("import time; time.sleep(30)", timeout=1)
    assert timed["timed_out"] and timed["exit_code"] == -9
