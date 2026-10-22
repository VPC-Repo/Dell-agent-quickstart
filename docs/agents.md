# The eight agents

`docker compose up` installs eight example agents (or press **Install the agents** in the UI). Each is a small team of
workers: every worker is a LangGraph agent with its own instructions, tools and model role. Each agent's flow is
fixed in `backend/demo_catalog.py`: a run starts the workers without dependencies together, passes their findings
to the workers that depend on them, and ends with a writer that turns them into a short brief for a business reader. The data is fictional and seeded; the processing is real.

| Agent | Tier | What it does | Workers (model role) | Tools |
|---|---|---|---|---|
| Cached FAQ responder | Light | Answers a customer's question from the knowledge base and remembers the answer for next time. | FAQ responder with cache (general) | Full-text search, Read the cache, Write the cache |
| Incident root cause analysis | Medium | Finds the outage in a week of service events, matches it to the runbook and explains the likely cause. | Incident analyst (general); Root cause analyst (reasoning); Decision brief writer (general) | Describe the warehouse, Full-text search, Query the warehouse (SQL) |
| Fix the failing test | Medium | Runs a failing test suite, reads the code, writes a fix and runs the tests again. | Bug fixer (coder); Decision brief writer (general) | Read a file, Run the test suite (sandbox), Write a source file |
| Daily service health report | Medium | Reads a day of service events and reports which service degraded. | Service day analyst (general); Decision brief writer (general) | Analyze one day of service logs |
| Portfolio credit risk | Medium | Simulates the loan book's losses and checks the capital buffer. | Credit risk modeler (general); Evidence reviewer (general); Decision brief writer (general) | Simulate portfolio credit risk (sandbox) |
| Customer master cleanup | Medium | Finds the same customer held twice across two systems. | Customer record matcher (general); Decision brief writer (general) | Match duplicate customers (sandbox) |
| Expert panel review | GPU heavy | Six specialists each assess one part of the business, three reviewers challenge them, and a brief sums it up. | Reliability panelist (reasoning); Policy panelist (general); Code panelist (coder); Data panelist (coder); Support panelist (general); Strategy panelist (reasoning); Business reviewer (general); Technical reviewer (coder); Risk reviewer (reasoning); Decision brief writer (general) | Describe the warehouse, Full-text search, List files, Query the warehouse (SQL), Read a document, Read a file, Semantic search (vectors) |
| Full business review | Full machine | Analyzes the service outage and reconciles every payment while a panel of six specialists reviews the business, then reviewers and a brief. | Service incident analyst (general); Reliability panelist (reasoning); Policy panelist (general); Code panelist (coder); Data panelist (coder); Support panelist (general); Strategy panelist (reasoning); Transaction reconciliation specialist (general); Business reviewer (general); Technical reviewer (coder); Risk reviewer (reasoning); Decision brief writer (general) | Analyze service logs, Describe the warehouse, Full-text search, List files, Query the warehouse (SQL), Read a document, Read a file, Reconcile transactions, Semantic search (vectors) |

## Tiers

| Tier | Meaning |
|---|---|
| Light | One or two workers, a few model and tool calls; seconds on GPUs |
| Medium | Several workers, sandbox jobs, reasoning, or a fixed CPU job that takes every core for a few seconds; under a minute on GPUs |
| GPU heavy | Six workers generating at once, then three reviewers: the inference engines are the bottleneck |
| Full machine | Sharded CPU jobs that run one process per core and fill the machine, beside six long generations |

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
| Sandbox (tests, credit risk simulation, customer matching) | A throwaway container per call from `dell-agent-quickstart-sandbox:local`: no network, capabilities dropped, CPU, memory and process limits. The credit risk and customer matching jobs are fixed numpy and difflib scripts on seeded data; the container has no CPU limit and the script runs its chunks on a pool of one process per CPU |
| CPU jobs | Reconcile transactions and analyze service logs: sharded into one process per CPU so the job fills the machine. Analyze one day of service logs: a fixed 3 million events in 512 chunks on a pool of one process per CPU, so a bigger machine finishes sooner |

Every tool result carries `work_evidence` (backend, host, units processed, CPU and wall seconds, sandbox limits),
shown when you expand a call. A sandbox tool's CPU is the container's own, read from its cgroup as the command ends,
plus the calling thread's. Each run also records its own process's CPU and the CPU the warehouse, cache, runner
database, gateway and local engine containers used while it ran (the run's `cpu_segment` event).

## CPU per output token

Measured on 2026-10-22 on a Dell PowerEdge XE7740 (2 x Intel Xeon 6787P, 172 cores) with the models on three of its
RTX PRO 6000 GPUs (Qwen3-30B-A3B general, gpt-oss-120b reasoning, Qwen3-Coder-30B-A3B coder), one run per agent,
one at a time. The agent's CPU is its run process, its tool jobs (sandbox containers and job processes) and the
warehouse, cache and runner-database containers during the run; the model service is not in it. It is capped at
the machine's 172 cores at every instant, as the run screen draws it (CPU time counts every busy hardware thread,
and a core one thread keeps busy has little left for its second); only the full business review reaches the cap.

| Agent | Agent core-ms per output token | CPUs for every GPU |
|---|---:|---:|
| Expert panel review | 0.5 | 0.0058 |
| Incident root cause analysis | 0.9 | 0.0096 |
| Fix the failing test | 4.0 | 0.044 |
| Cached FAQ responder | 5.8 | 0.064 |
| Daily service health report | 137 | 1.5 |
| Portfolio credit risk | 236 | 2.6 |
| Customer master cleanup | 339 | 3.7 |
| Full business review | 984 | 11 |

"CPUs for every GPU" is arithmetic, not a measurement: core-ms per token x 950 output tokens a second per GPU /
(1000 x 86 cores per socket), the CPUs a GPU kept busy with runs like this one would need. The run screen shows
the same figure under its chart, with an info button that walks through it.

The three fixed jobs do the same work on every machine and finish in about 2 seconds on this one's 344 hardware
threads. Spread over every thread they cost more CPU time than on fewer cores: pinned to 64, 86 and 128 cores
(service day, customer dedupe, credit risk) they ran within 10% of their fastest time for up to 2.7 times less
CPU, so treat their figures as what the job costs when it takes the whole host. The whole-machine jobs give every
core the same work, so the full business review's figure grows with the core count. On the same GPU host the
inference engines' own host CPU was 0.1 to 10 core-ms per output token and the LiteLLM gateway's 0.5 to 1.4. One
run each: treat the light agents' figures as an order of magnitude.

## Checks

Each agent declares the facts a correct answer must contain and the tools a correct run must call. A run that
skips a required tool is told to continue with it (counted as a recovered error); the run's result says whether
its checks passed.
