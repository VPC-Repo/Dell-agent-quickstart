# The eight agents

`docker compose up` installs eight example agents (or press **Install the agents** in the UI). Each is a small team of
workers: every worker is a LangGraph agent with its own instructions, tools and model role. Each agent's flow is
fixed in `backend/demo_catalog.py`: a run starts the workers without dependencies together, passes their findings
to the workers that depend on them, and ends with a writer that turns them into a short brief for a business reader. The data is fictional and seeded; the processing is real.

| Agent | Tier | What it does | Workers (model role) | Tools |
|---|---|---|---|---|
| Finance exception review | CPU heavy | Matches every invoice to its payment, flags the exceptions and explains what to settle first. | Transaction reconciliation specialist (general); Evidence reviewer (general); Decision brief writer (general) | Reconcile transactions |
| Ask the warehouse | Light | Turns a plain-English question into SQL, runs it on the warehouse and reports the figures. | SQL developer (coder); Decision brief writer (general) | Describe the warehouse, Query the warehouse (SQL) |
| Fix the failing test | Medium | Runs a failing test suite, reads the code, writes a fix and runs the tests again. | Bug fixer (coder); Decision brief writer (general) | Read a file, Run the test suite (sandbox), Write a source file |
| Document ingestion pipeline | Medium | Reads two PDF reports, adds them to the knowledge base and checks that search can find them. | Document ingestion specialist (general); Knowledge base researcher (general); Decision brief writer (general) | Extract PDF text (sandbox), Full-text search, Index a document, List files, Read a document |
| Incident root cause analysis | Medium | Finds the outage in a week of service events, matches it to the runbook and explains the likely cause. | Incident analyst (general); Root cause analyst (reasoning); Decision brief writer (general) | Describe the warehouse, Full-text search, Query the warehouse (SQL) |
| Cached FAQ responder | Light | Answers a customer's question from the knowledge base and remembers the answer for next time. | FAQ responder with cache (general) | Full-text search, Read the cache, Write the cache |
| Expert panel review | GPU heavy | Six specialists each assess one part of the business, three reviewers challenge them, and a brief sums it up. | Reliability panelist (reasoning); Policy panelist (general); Code panelist (coder); Data panelist (coder); Support panelist (general); Strategy panelist (reasoning); Business reviewer (general); Technical reviewer (coder); Risk reviewer (reasoning); Decision brief writer (general) | Describe the warehouse, Full-text search, List files, Query the warehouse (SQL), Read a document, Read a file, Semantic search (vectors) |
| Full business review | Full machine | Analyzes the service outage and reconciles every payment while a panel of six specialists reviews the business, then reviewers and a brief. | Service incident analyst (general); Reliability panelist (reasoning); Policy panelist (general); Code panelist (coder); Data panelist (coder); Support panelist (general); Strategy panelist (reasoning); Transaction reconciliation specialist (general); Business reviewer (general); Technical reviewer (coder); Risk reviewer (reasoning); Decision brief writer (general) | Analyze service logs, Describe the warehouse, Full-text search, List files, Query the warehouse (SQL), Read a document, Read a file, Reconcile transactions, Semantic search (vectors) |

## Tiers

| Tier | Meaning |
|---|---|
| Light | One or two workers, a few model and tool calls; seconds on GPUs |
| Medium | Several workers, sandbox jobs or reasoning; under a minute on GPUs |
| CPU heavy | Sharded CPU jobs that run one process per core and fill the machine |
| GPU heavy | Six workers generating at once, then three reviewers: the inference engines are the bottleneck |
| Full machine | Both: sharded CPU jobs beside six long generations |

All eight support CPU inference on a server meeting the requirements. Model-heavy agents can take
substantially longer on CPU inference; timings depend on hardware, model load and concurrent work.

## Model roles

Workers ask for a role, not a model. The installer points the roles at engines through LiteLLM aliases:

| Role | GPU host | CPU host |
|---|---|---|
| general | Qwen3-30B-A3B-Instruct-2507 (with replicas) | Qwen3-30B-A3B-Instruct-2507 on vLLM CPU |
| reasoning | gpt-oss-120b (with replicas) | the same CPU engine |
| coder | Qwen3-Coder-30B-A3B-Instruct (with replicas) | the same CPU engine |

Every model call records the engine, container and GPU the gateway routed it to, so the UI can show which
replica answered.

## What the tools run on

| Tool group | Backend |
|---|---|
| Warehouse SQL, full-text search | The toolkit's PostgreSQL with pgvector (`agentdb`): a seeded warehouse of customers, orders, payments, tickets and 60,000 service events |
| Vector search | pgvector cosine over a local hashed embedding (labelled as such in each result), so search results are the same on every server. The bge-base-en-v1.5 model the installer deploys is for agents you build on the stack |
| Cache | The toolkit's Redis |
| Sandbox (Python, tests, PDF text) | A throwaway container per call from `dell-agent-quickstart-sandbox:local`: no network, capabilities dropped, CPU, memory and process limits |
| CPU jobs (reconcile transactions, analyze service logs) | In the run's process, sharded into one process per CPU so the job fills the machine |

Every tool result carries `work_evidence` (backend, host, units processed, CPU and wall seconds, sandbox limits),
shown when you expand a call.

## Checks

Each agent declares the facts a correct answer must contain and the tools a correct run must call. A run that
skips a required tool is told to continue with it (counted as a recovered error); the run's result says whether
its checks passed.
