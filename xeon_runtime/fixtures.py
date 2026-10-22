"""Deterministic enterprise fixtures the example agents work on.

Everything is generated from seed 470 on first use into FIXTURE_DIR (default
data/fixtures) and stamped with VERSION, so every host has identical data and
reruns produce identical findings. Fictional throughout: Northwind-style
customers, orders and payments, support tickets, service events, a knowledge
base of policies and runbooks, a few PDF reports, some web pages and a small
Python package with a failing test.

Generation is real work but small (a few seconds); nothing is downloaded.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import random
import textwrap
import zlib
from datetime import date, datetime, timedelta
from pathlib import Path

VERSION = "fixtures-v2"
# Planted churn: these customers (by index) place no orders after CHURN_CUTOFF and hold open tickets.
CHURN_CUTOFF = date(2026, 6, 30)


def churned(index: int) -> bool:
    return index % 50 == 7


SEED = 470

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America"]
SEGMENTS = ["enterprise", "mid-market", "smb"]
PRODUCTS = [
    ("PRD-100", "Edge Gateway 4", 1890.00),
    ("PRD-110", "Rack Sensor Kit", 349.00),
    ("PRD-120", "Fleet Controller License", 1200.00),
    ("PRD-130", "Cold Chain Monitor", 615.00),
    ("PRD-140", "Field Tablet R2", 980.00),
    ("PRD-150", "Support Plan · Premier", 2400.00),
    ("PRD-160", "Installation Service", 750.00),
    ("PRD-170", "Spare Battery Pack", 88.00),
]
COMPANIES = [
    "Northwind Traders",
    "Fabrikam Logistics",
    "Contoso Retail",
    "Tailspin Freight",
    "Wingtip Foods",
    "Alpine Grocers",
    "Litware Manufacturing",
    "Adatum Pharma",
    "Coho Vineyard",
    "Proseware Clinics",
    "Woodgrove Bank",
    "Margie's Travel",
    "Blue Yonder Airlines",
    "Lucerne Publishing",
    "Trey Research",
    "Humongous Insurance",
]
TICKET_TOPICS = [
    ("billing", "Invoice {inv} shows a duplicate charge for order {order}.", "high"),
    ("billing", "Refund for order {order} has not arrived after 20 days.", "medium"),
    ("shipping", "Order {order} tracking has not updated in a week.", "medium"),
    ("shipping", "Order {order} arrived with a damaged Edge Gateway unit.", "high"),
    (
        "device",
        "Cold Chain Monitor {serial} reports temperature -40 constantly.",
        "high",
    ),
    (
        "device",
        "Field Tablet {serial} will not hold a charge after firmware 4.2.",
        "medium",
    ),
    ("account", "Need to add two users to the Fleet Controller license.", "low"),
    ("account", "Password reset email never arrives for our admin account.", "medium"),
    ("api", "Fleet API returns 429 during the nightly export since Monday.", "high"),
    ("api", "Webhook retries fire twice for the same shipment event.", "medium"),
]
SERVICES = [
    "checkout",
    "catalog",
    "identity",
    "fulfillment",
    "fleet-api",
    "notifications",
]


def fixture_dir() -> Path:
    return Path(os.getenv("FIXTURE_DIR", "data/fixtures"))


def ensure() -> Path:
    """Generate the fixtures once (any process may win the race) and return the directory."""
    root = fixture_dir()
    stamp = root / "VERSION"
    if stamp.is_file() and stamp.read_text().strip() == VERSION:
        return root
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if stamp.is_file() and stamp.read_text().strip() == VERSION:
                return root
            generate(root)
            stamp.write_text(VERSION)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return root


def generate(root: Path):
    rng = random.Random(SEED)
    _warehouse(root / "warehouse", rng)
    _knowledge_base(root / "docs", rng)
    _reports(root / "reports")
    _web(root / "web")
    _source(root / "src")
    (root / "README.md").write_text(
        textwrap.dedent(
            f"""\
            # Demo fixtures ({VERSION})

            Demo data for the example agents, generated the same way on every install.

            - warehouse/: customers, orders, payments, tickets, service_events (CSV)
            - docs/: knowledge base articles (Markdown) with index.json
            - reports/: quarterly PDF reports
            - web/: static HTML pages
            - src/: the orderflow Python package with its tests
            """
        )
    )


# --- warehouse ------------------------------------------------------------------
def _warehouse(folder: Path, rng: random.Random):
    folder.mkdir(parents=True, exist_ok=True)
    start = date(2026, 1, 1)
    customers = []
    for i in range(1, 2001):
        company = COMPANIES[i % len(COMPANIES)]
        customers.append(
            {
                "customer_id": f"C{i:05d}",
                "name": company
                if i <= len(COMPANIES)
                else f"{company} · site {i // len(COMPANIES)}",
                "region": REGIONS[i % 4],
                "segment": SEGMENTS[(i * 7) % 3],
                "since": (start - timedelta(days=rng.randint(30, 2000))).isoformat(),
                "credit_limit": rng.choice([10000, 25000, 50000, 100000, 250000]),
            }
        )
    _write_csv(folder / "customers.csv", customers)

    orders, payments = [], []
    statuses = (
        ["delivered"] * 70
        + ["shipped"] * 12
        + ["pending"] * 10
        + ["cancelled"] * 5
        + ["returned"] * 3
    )
    for i in range(1, 50001):
        product = PRODUCTS[(i * 13) % len(PRODUCTS)]
        quantity = 1 + (i * 7) % 6
        day = start + timedelta(days=(i * 37) % 270)
        index = (i * 31) % len(customers)
        original = customers[index]
        # Planted churn: a churned customer's later orders go to the next customer instead.
        if churned(index) and day > CHURN_CUTOFF:
            index = (index + 1) % len(customers)
        customer = customers[index]
        # A planted anomaly: PRD-130 orders in Asia Pacific are returned 6x more often.
        # (Checked on the original customer so the random sequence, and every other planted figure, is unchanged.)
        status = rng.choice(statuses)
        if (
            product[0] == "PRD-130"
            and original["region"] == "Asia Pacific"
            and rng.random() < 0.18
        ):
            status = "returned"
        amount = round(
            product[2]
            * quantity
            * (0.9 if customer["segment"] == "enterprise" else 1.0),
            2,
        )
        orders.append(
            {
                "order_id": f"O{i:06d}",
                "customer_id": customer["customer_id"],
                "product_id": product[0],
                "product_name": product[1],
                "quantity": quantity,
                "amount_usd": f"{amount:.2f}",
                "status": status,
                "ordered_on": day.isoformat(),
                "region": customer["region"],
            }
        )
        if status in ("delivered", "shipped", "returned"):
            # 1 in 400 payments is short by exactly 25.00 (a settlement fee leak).
            short = 25.00 if i % 400 == 0 else 0.0
            payments.append(
                {
                    "payment_id": f"P{i:06d}",
                    "order_id": f"O{i:06d}",
                    "paid_usd": f"{amount - short:.2f}",
                    "paid_on": (day + timedelta(days=rng.randint(1, 30))).isoformat(),
                    "method": rng.choice(["card", "wire", "ach", "card", "card"]),
                }
            )
    _write_csv(folder / "orders.csv", orders)
    _write_csv(folder / "payments.csv", payments)

    tickets = []
    for i in range(1, 3001):
        topic, template, priority = TICKET_TOPICS[(i * 3) % len(TICKET_TOPICS)]
        order = f"O{rng.randint(1, 50000):06d}"
        opened = datetime(2026, 1, 3) + timedelta(hours=(i * 19) % (24 * 260))
        # Planted spike: fleet-api 429 tickets cluster in September 2026.
        if topic == "api" and i % 2 == 0:
            opened = datetime(2026, 9, 14) + timedelta(hours=(i * 5) % 240)
        resolved = rng.random() < 0.82
        index = (i * 17) % len(customers)
        if churned(index):
            # Churned customers complained in July to September and their tickets are still open.
            opened = datetime(2026, 7, 2) + timedelta(hours=(i * 23) % (24 * 80))
            resolved = False
        tickets.append(
            {
                "ticket_id": f"T{i:05d}",
                "customer_id": customers[index]["customer_id"],
                "topic": topic,
                "priority": priority,
                "subject": template.format(
                    inv=f"INV-{i:05d}",
                    order=order,
                    serial=f"SN{rng.randint(100000, 999999)}",
                ),
                "opened_at": opened.isoformat(timespec="minutes"),
                "status": "resolved"
                if resolved
                else rng.choice(["open", "waiting_on_customer", "escalated"]),
                "hours_to_resolve": round(rng.lognormvariate(2.6, 0.7), 1)
                if resolved
                else "",
                "csat": rng.choice([5, 5, 4, 4, 3, 2, 1]) if resolved else "",
            }
        )
    _write_csv(folder / "tickets.csv", tickets)

    events = []
    base = datetime(2026, 9, 20, 0, 0)
    for i in range(60000):
        service = SERVICES[i % len(SERVICES)]
        at = base + timedelta(seconds=i * 7)
        # Planted incident: fleet-api errors and latency spike 09:00-11:00 on 2026-09-22.
        incident = service == "fleet-api" and datetime(2026, 9, 22, 9) <= at < datetime(
            2026, 9, 22, 11
        )
        status = (
            429
            if incident and rng.random() < 0.35
            else (500 if rng.random() < 0.004 else 200)
        )
        latency = rng.lognormvariate(5.6 if incident else 4.3, 0.4)
        events.append(
            {
                "event_id": i + 1,
                "at": at.isoformat(timespec="seconds"),
                "service": service,
                "endpoint": rng.choice(
                    ["/v1/list", "/v1/get", "/v1/update", "/v1/export"]
                ),
                "status": status,
                "latency_ms": round(latency, 1),
                "region": REGIONS[i % 4],
            }
        )
    _write_csv(folder / "service_events.csv", events)


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --- knowledge base ---------------------------------------------------------------
POLICIES = [
    (
        "Refund policy",
        "Customers may request a refund within 45 days of delivery. Refunds for Support Plan · Premier are prorated monthly. Damaged-on-arrival claims must include photos and are approved within 2 business days.",
    ),
    (
        "Warranty policy",
        "Hardware carries a 24-month limited warranty. Cold Chain Monitors used below -30 °C are covered only with the arctic firmware profile enabled. Batteries are consumables with a 6-month warranty.",
    ),
    (
        "Data retention policy",
        "Fleet telemetry is retained for 13 months. Support ticket transcripts are retained for 3 years. Deleted accounts are purged after 30 days.",
    ),
    (
        "Escalation policy",
        "High-priority tickets are acknowledged within 1 hour and escalated to engineering if unresolved after 8 hours. Security incidents page the on-call lead immediately.",
    ),
    (
        "API rate limits",
        "The Fleet API allows 600 requests per minute per tenant and 50 concurrent exports. Nightly exports should use the bulk endpoint /v1/export with pagination to avoid 429 responses.",
    ),
    (
        "Firmware release notes 4.2",
        "Firmware 4.2 for Field Tablet R2 improves Wi-Fi roaming. Known issue: battery calibration drifts after upgrade; run the calibration cycle from Settings → Power. Fixed in 4.2.1.",
    ),
    (
        "Incident runbook: fleet-api 429 storm",
        "Symptoms: elevated 429 responses and p95 latency above 500 ms on fleet-api. Cause is usually a tenant export loop. Mitigation: enable per-tenant export throttling, confirm the bulk endpoint is used, and notify the tenant.",
    ),
    (
        "Onboarding runbook",
        "New tenants receive a Fleet Controller License with 5 seats. Additional seats are added from the admin console under Organization → Seats; changes take effect within 15 minutes.",
    ),
    (
        "Cold chain calibration guide",
        "A constant -40 reading indicates a disconnected probe. Reseat the probe connector and run a two-point calibration at 0 °C and 20 °C. Replace the probe if the reading persists.",
    ),
    (
        "Shipping and tracking guide",
        "Tracking updates arrive from carriers every 6 to 24 hours. If tracking is silent for 5 business days, open a carrier trace and offer a replacement shipment.",
    ),
]
DEPARTMENTS = ["Support", "Engineering", "Finance", "Sales", "Operations", "Security"]
TOPICS = [
    "deployment",
    "monitoring",
    "billing",
    "onboarding",
    "hardware",
    "compliance",
    "logistics",
    "performance",
]


def _knowledge_base(folder: Path, rng: random.Random):
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    doc = 100
    for title, body in POLICIES:
        doc += 1
        _article(
            folder,
            index,
            doc,
            title,
            "Policy" if "policy" in title.lower() else "Runbook",
            body,
        )
    words = "service deployment latency retry timeout queue backlog capacity incident recovery audit customer order invoice settlement approval retention ownership escalation monitoring rollout calibration firmware tenant export throttle".split()
    for i in range(390):
        doc += 1
        department = DEPARTMENTS[i % len(DEPARTMENTS)]
        topic = TOPICS[(i * 5) % len(TOPICS)]
        title = f"{department} note {i + 1}: {topic}"
        sentences = []
        for _ in range(rng.randint(5, 9)):
            sentences.append(
                " ".join(
                    rng.choice(words) for _ in range(rng.randint(8, 16))
                ).capitalize()
                + "."
            )
        body = f"{department} guidance on {topic}. " + " ".join(sentences)
        if i % 37 == 0:
            body += " See the Incident runbook: fleet-api 429 storm for export throttling steps."
        _article(folder, index, doc, title, "Note", body)
    (folder / "index.json").write_text(json.dumps(index, indent=1))


def _article(folder, index, number, title, kind, body):
    doc_id = f"KB-{number:04d}"
    text = f"# {title}\n\nDocument {doc_id} · {kind}\n\n{body}\n"
    (folder / f"{doc_id}.md").write_text(text)
    index.append(
        {
            "doc_id": doc_id,
            "title": title,
            "kind": kind,
            "path": f"{doc_id}.md",
            "chars": len(text),
        }
    )


# --- reports (minimal PDFs) -----------------------------------------------------------
REPORTS = {
    "Q1-2026-operations-review.pdf": [
        "Q1 2026 Operations Review (fictional)",
        "Orders processed: 16,480. On-time delivery: 93.1 percent.",
        "Returns rose to 4.2 percent, concentrated in Cold Chain Monitors shipped to Asia Pacific.",
        "Support backlog fell from 210 to 168 open tickets.",
        "Action: launch the arctic firmware profile and update the calibration guide.",
    ],
    "Q2-2026-operations-review.pdf": [
        "Q2 2026 Operations Review (fictional)",
        "Orders processed: 17,905. On-time delivery: 94.6 percent.",
        "Fleet API adoption doubled; nightly exports now account for 61 percent of API traffic.",
        "Payment exceptions: 1 in 400 card settlements short by 25.00 USD, under investigation with the processor.",
        "Action: publish bulk export guidance and add per-tenant throttling.",
    ],
    "Q3-2026-security-summary.pdf": [
        "Q3 2026 Security Summary (fictional)",
        "No confirmed breaches. 3 phishing simulations run; click rate 6 percent.",
        "Password reset email deliverability issue traced to a DMARC alignment change on 2026-08-30.",
        "Action: rotate the transactional email signing key and monitor bounce rates.",
    ],
    "supplier-agreement-summary.pdf": [
        "Supplier Agreement Summary (fictional)",
        "Battery packs are sourced from two suppliers with a 6-month defect warranty.",
        "Lead time is 8 weeks; safety stock target is 12 weeks of demand.",
        "Penalty clause: 2 percent per week of late delivery, capped at 10 percent.",
    ],
    "fleet-api-capacity-plan.pdf": [
        "Fleet API Capacity Plan (fictional)",
        "Current limit: 600 requests per minute per tenant; 50 concurrent exports.",
        "Observed peak on 2026-09-22 between 09:00 and 11:00: 4.1x normal request rate, 35 percent 429 responses.",
        "Recommendation: move nightly exports to the bulk endpoint and shard by tenant.",
    ],
}


def _pdf(lines: list[str]) -> bytes:
    """A minimal valid single-page PDF with Helvetica text, no external tools."""

    def escape(text):
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_lines = ["BT", "/F1 12 Tf", "50 750 Td", "16 TL"]
    for line in lines:
        content_lines.append(f"({escape(line)}) Tj T*")
    content_lines.append("ET")
    content = "\n".join(content_lines).encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _reports(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for name, lines in REPORTS.items():
        (folder / name).write_bytes(_pdf(lines))


# --- web pages ----------------------------------------------------------------------
PAGES = {
    "index.html": (
        "Contoso Fleet · Home",
        "Contoso Fleet builds edge gateways and cold-chain monitors for logistics fleets. Products ship from Austin and Rotterdam. Our Fleet API lets tenants export telemetry nightly.",
    ),
    "status.html": (
        "Service status",
        "All systems operational. Incident history: 2026-09-22 09:00–11:00 UTC fleet-api elevated 429 responses for tenants running unthrottled exports; resolved by enabling per-tenant throttling.",
    ),
    "pricing.html": (
        "Pricing",
        "Edge Gateway 4: 1,890 USD. Rack Sensor Kit: 349 USD. Fleet Controller License: 1,200 USD per year for 5 seats. Support Plan · Premier: 2,400 USD per year with 1-hour response for high-priority tickets.",
    ),
    "blog-firmware-42.html": (
        "Firmware 4.2 for Field Tablet R2",
        "Firmware 4.2 improves Wi-Fi roaming and adds offline forms. Some tablets report shorter battery life after upgrading; a calibration cycle resolves it and 4.2.1 fixes the drift.",
    ),
    "docs-api.html": (
        "Fleet API documentation",
        "Authenticate with a tenant token. Limits: 600 requests per minute, 50 concurrent exports. Use POST /v1/export with a cursor for bulk data. Webhooks retry with exponential backoff; deliveries are idempotent by event id.",
    ),
    "careers.html": (
        "Careers",
        "We are hiring field engineers in Rotterdam and support specialists in Austin.",
    ),
}


def _web(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    nav = "".join(
        f'<li><a href="{name}">{title.split(" · ")[0]}</a></li>'
        for name, (title, _) in PAGES.items()
    )
    for name, (title, body) in PAGES.items():
        html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<script>window.analytics=1;</script><style>nav{{display:flex}}</style></head>
<body><header><nav><ul>{nav}</ul></nav><div class="banner">Cookie notice: this fictional site uses no cookies.</div></header>
<main><h1>{title}</h1><p>{body}</p><p>Contact: support@contoso-fleet.example</p></main>
<footer><p>© 2026 Contoso Fleet (fictional). <a href="privacy.html">Privacy</a></p></footer></body></html>"""
        (folder / name).write_text(html)


# --- source tree --------------------------------------------------------------------
SOURCE = {
    "README.md": """# orderflow

A small order-processing library used by the demo agents.

- `orderflow/pricing.py`: discounts and totals
- `orderflow/retry.py`: retry policy for carrier API calls
- `orderflow/inventory.py`: reorder point calculation

Run the tests with `python -m pytest -q`.
""",
    "orderflow/__init__.py": "",
    "orderflow/pricing.py": '''from decimal import Decimal, ROUND_HALF_UP

TWO_PLACES = Decimal("0.01")


def apply_discount(amount, percent):
    """Return `amount` reduced by `percent`, rounded to cents."""
    amount = Decimal(str(amount))
    factor = (Decimal(100) - Decimal(str(percent))) / Decimal(100)
    # BUG: rounding through float uses binary half-even, not half-up cents.
    return Decimal(str(round(float(amount * factor), 2))).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def order_total(lines, discount_percent=0):
    """lines: iterable of (unit_price, quantity). Returns the discounted total."""
    subtotal = sum(Decimal(str(price)) * quantity for price, quantity in lines)
    return apply_discount(subtotal, discount_percent)
''',
    "orderflow/retry.py": '''import time


class RetryPolicy:
    """Exponential backoff for carrier API calls.

    attempts: total tries including the first; base: seconds; cap: max sleep.
    """

    def __init__(self, attempts=5, base=0.5, cap=30.0):
        self.attempts = attempts
        self.base = base
        self.cap = cap

    def delays(self):
        for n in range(self.attempts - 1):
            yield min(self.cap, self.base * 2 ** n)

    def call(self, fn, sleep=time.sleep):
        last = None
        for delay in [*self.delays(), None]:
            try:
                return fn()
            except Exception as exc:  # carrier errors are broad
                last = exc
                if delay is None:
                    raise
                sleep(delay)
        raise last
''',
    "orderflow/inventory.py": '''def reorder_point(daily_demand, lead_time_days, safety_days=3):
    """Units on hand at which to reorder."""
    if daily_demand < 0 or lead_time_days < 0:
        raise ValueError("demand and lead time must be non-negative")
    return round(daily_demand * (lead_time_days + safety_days))
''',
    "tests/__init__.py": "",
    "tests/test_pricing.py": """from decimal import Decimal

from orderflow.pricing import apply_discount, order_total


def test_discount_is_exact_to_the_cent():
    # 1.15 * 0.9 = 1.035 -> 1.04 with half-up rounding; float gives 1.0349999
    assert apply_discount("1.15", 10) == Decimal("1.04")


def test_large_order_total():
    assert order_total([("1890.00", 3), ("88.00", 10)], 10) == Decimal("5895.00")


def test_no_discount_is_identity():
    assert apply_discount("349.00", 0) == Decimal("349.00")
""",
    "tests/test_retry.py": """import pytest

from orderflow.retry import RetryPolicy


def test_delays_double_and_cap():
    assert list(RetryPolicy(attempts=5, base=1, cap=5).delays()) == [1, 2, 4, 5]


def test_call_retries_then_raises():
    calls = []

    def flaky():
        calls.append(1)
        raise RuntimeError("carrier down")

    with pytest.raises(RuntimeError):
        RetryPolicy(attempts=3, base=0).call(flaky, sleep=lambda s: None)
    assert len(calls) == 3
""",
    "tests/test_inventory.py": """import pytest

from orderflow.inventory import reorder_point


def test_reorder_point():
    assert reorder_point(14, 5) == 112


def test_rejects_negative():
    with pytest.raises(ValueError):
        reorder_point(-1, 5)
""",
}


def _source(folder: Path):
    for name, content in SOURCE.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def tree(folder: Path) -> dict[str, bytes]:
    """All files under a fixture folder as {relative path: bytes}."""
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def checksum(folder: Path) -> str:
    value = 0
    for name, content in tree(folder).items():
        value = zlib.crc32(name.encode() + content, value)
    return f"{value:08x}"
