/** Turn the API's run document into the timeline, calls, samples and activity feed the screens draw. */
type Payload = Record<string, any>;
export type RunEvent = {
  id: number;
  created: number;
  kind: string;
  payload: Payload;
};
export type RunRecord = {
  agent: {
    id: string;
    demo: { slug: string };
    agents: Payload[];
    goal: string;
  };
  run: {
    events: RunEvent[];
    started: number;
    finished: number;
    status: string;
    output: string;
    error?: string;
  };
};
export type Call = {
  id: string;
  kind: "model" | "tool";
  title: string;
  start: number;
  end: number | null;
  failed: boolean;
  evidence: string;
  model: string;
  /** Where the model call ran, from the gateway's routing: "GPU 4", "CPU" or "". */
  where: string;
  /** Which of the demo's models answered: "general", "reasoning", "coder" or "other"; "" for tool calls. */
  family: ModelFamily | "";
};
export type ModelFamily = "general" | "reasoning" | "coder" | "other";
/** The families by shade, lightest to darkest (matches the --model-* colours in demo.css). */
export const SHADE_ORDER: ModelFamily[] = [
  "coder",
  "general",
  "other",
  "reasoning",
];
/** The three demo models are told apart by a slight shade each; anything else gets the base shade. */
export function modelFamily(model: string): ModelFamily {
  if (/coder/i.test(model)) return "coder";
  if (/gpt-oss/i.test(model)) return "reasoning";
  if (/qwen|llama|mistral|gemma|instruct/i.test(model)) return "general";
  return model ? "other" : "general";
}
/** A short, readable model name for legends: Qwen3-30B, Qwen3-Coder-30B, gpt-oss-120b. */
export function modelLabel(model: string): string {
  const short = shortModel(model);
  const m = short.match(
    /^(Qwen\d*(?:\.\d+)?-Coder-\d+B|Qwen\d*(?:\.\d+)?-\d+B|gpt-oss-\d+b|Llama-[\d.]+-\d+B)/i,
  );
  return m ? m[1] : short.replace(/-Instruct.*$/i, "");
}
export type Task = {
  id: string;
  name: string;
  description: string;
  start: number | null;
  end: number | null;
  calls: Call[];
};
export type Sample = { t: number; cpu: number | null; gpu: number | null };
export type DemoRun = {
  duration: number;
  tasks: Task[];
  samples: Sample[];
  /** The models this run called, in order of first use, for the legend. */
  models: { family: ModelFamily; label: string }[];
  gpuCount: number;
  output: string;
  status: string;
  error: string;
  activity: Activity[];
};
export type Activity = { id: string; t: number; worker: string; text: string };
export const catalog = [
  {
    slug: "faq-cache",
    title: "Answer a refund policy question",
    description: "Look up the policy and check for a saved answer.",
    category: "Customer support",
    icon: "support",
  },
  {
    slug: "warehouse-question",
    title: "Find the top five customers by revenue",
    description: "Query sales records and calculate their share of revenue.",
    category: "Business data",
    icon: "data",
  },
  {
    slug: "pdf-ingestion",
    title: "Index and search two PDF reports",
    description: "Extract the text, index it, and check the results.",
    category: "Documents",
    icon: "documents",
  },
  {
    slug: "bug-fix",
    title: "Fix a bug and rerun the tests",
    description: "Find a failing test, edit the code, and verify the fix.",
    category: "Software",
    icon: "code",
  },
  {
    slug: "incident-rca",
    title: "Find the cause of a service outage",
    description: "Analyze logs and check the recovery runbook.",
    category: "Operations",
    icon: "activity",
  },
  {
    slug: "finance-close",
    title: "Match invoices to payments",
    description: "Reconcile accounts and flag discrepancies.",
    category: "Finance",
    icon: "finance",
  },
  {
    slug: "gpu-panel",
    title: "Review the business with a team of agents",
    description: "Gather specialist assessments and review findings.",
    category: "Business analysis",
    icon: "team",
  },
  {
    slug: "full-machine",
    title: "Investigate operations and reconcile accounts",
    description: "Run service, financial, and expert reviews together.",
    category: "Operations & finance",
    icon: "network",
  },
];
const toolNames: Record<string, string> = {
  cache_get: "Check for a saved answer",
  cache_set: "Save the answer",
  analyze_service_logs: "Analyze service events",
  reconcile_transactions: "Match invoices and payments",
  search_documents: "Search the knowledge base",
  semantic_search: "Find related documents",
  read_document: "Read a document",
  index_document: "Add text to the knowledge base",
  list_fixture_files: "Find source files",
  read_fixture_file: "Read a source file",
  write_source_file: "Update the source code",
  run_tests: "Run the tests",
  describe_warehouse: "Inspect the database tables",
  query_warehouse: "Query business data",
  extract_pdf_text: "Extract text from a PDF",
  process_documents: "Process documents",
};
export const friendlyTool = (name: string) =>
  toolNames[name] || name.replaceAll("_", " ");
export const cleanDescription = (s: string) =>
  s
    .replaceAll("seeded ", "")
    .replaceAll("demo business", "business")
    .replaceAll("demo ", "")
    .replaceAll("in the sandbox ", "")
    .replaceAll("in the isolated environment ", "")
    .replaceAll("worker findings", "the findings")
    .replaceAll("sandbox", "code environment");
export const shortName = (s: string) =>
  ({
    "Transaction reconciliation specialist": "Reconciliation specialist",
    "Service incident analyst": "Incident analyst",
    "Decision brief writer": "Brief writer",
    "Document ingestion specialist": "PDF ingestion specialist",
    "Knowledge base researcher": "Knowledge researcher",
    "FAQ responder with cache": "Refund policy assistant",
  })[s] || s;
export const seconds = (n: number) => n.toFixed(1) + " s";
export const shortModel = (model: string) => model.split("/").at(-1) || model;

export function adaptRun(record: RunRecord): DemoRun {
  const events = [...record.run.events].sort((a, b) => a.created - b.created);
  const origin = events[0]?.created ?? record.run.started;
  const relative = (t: number) => Math.max(0, t - origin);
  const duration = Math.max(
    0.1,
    relative(record.run.finished || events.at(-1)?.created || origin),
  );
  const plan =
    events.find((e) => e.kind === "plan_created")?.payload.tasks ||
    record.agent.agents;
  const tasks: Task[] = plan.map((w: Payload) => {
    const started = events.find(
      (e) => e.kind === "node_started" && e.payload.node_id === w.id,
    );
    const finished = events.find(
      (e) =>
        ["node_completed", "node_skipped"].includes(e.kind) &&
        e.payload.node_id === w.id,
    );
    const calls: Call[] = events
      .filter(
        (e) =>
          e.payload.task_id === w.id &&
          ["model_request", "tool_started"].includes(e.kind),
      )
      .map((e) => {
        const model = e.kind === "model_request";
        const done = events.find(
          (x) =>
            x.payload.request_id === e.payload.request_id &&
            (model
              ? ["model", "model_error"]
              : ["tool_completed", "tool_error"]
            ).includes(x.kind),
        );
        const units = done?.payload.work_evidence?.units || {};
        const evidence = Object.entries(units)
          .filter(
            ([k, v]) =>
              typeof v === "number" &&
              [
                "events",
                "rows",
                "transactions",
                "invoices",
                "payments",
                "documents",
                "services",
                "files",
                "matches",
                "tests_passed",
                "tests_failed",
                "files_changed",
              ].includes(k),
          )
          .map(
            ([k, v]) =>
              `${Number(v).toLocaleString()} ${
                Number(v) === 1
                  ? k
                      .replaceAll("_", " ")
                      .replace(/^matches$/, "match")
                      .replace(
                        /^(events|rows|transactions|invoices|payments|documents|services|files|tests)\b/,
                        (word) => word.slice(0, -1),
                      )
                  : k.replaceAll("_", " ")
              }`,
          )
          .join(" · ");
        const place = done?.payload.placement || e.payload.placement || {};
        const where = !model
          ? ""
          : place.gpus?.length
            ? `GPU ${place.gpus.join(", ")}`
            : place.device === "cpu"
              ? "CPU"
              : "";
        return {
          id: String(e.payload.request_id || e.id),
          kind: model ? "model" : "tool",
          title: model
            ? "Language model call"
            : friendlyTool(e.payload.name || "Tool call"),
          start: relative(e.created),
          end: done ? relative(done.created) : null,
          failed: !!done?.kind.endsWith("_error"),
          evidence,
          model: model
            ? done?.payload.placement?.model_id ||
              e.payload.placement?.model_id ||
              e.payload.model ||
              ""
            : "",
          where,
          family: model
            ? modelFamily(
                done?.payload.placement?.model_id ||
                  e.payload.placement?.model_id ||
                  e.payload.model ||
                  "",
              )
            : "",
        };
      });
    return {
      id: w.id,
      name: shortName(w.name),
      description: cleanDescription(w.task || ""),
      start: started ? relative(started.created) : null,
      end: finished ? relative(finished.created) : null,
      calls,
    };
  });
  let gpuCount = 0;
  const samples: Sample[] = events
    .filter((e) => e.kind === "resources")
    .map((e) => {
      const gpus: Payload[] = e.payload.gpus || [];
      gpuCount = Math.max(gpuCount, gpus.length);
      const values = gpus
        .map((g) => g.utilization_percent)
        .filter((v) => typeof v === "number" && Number.isFinite(v));
      return {
        t: relative(e.created),
        cpu:
          typeof e.payload.host_cpu_percent === "number"
            ? e.payload.host_cpu_percent
            : null,
        gpu:
          values.length === gpus.length && values.length
            ? values.reduce((a, b) => a + b, 0) / values.length
            : null,
      };
    });
  const activity: Activity[] = [
    { id: "begin", t: 0, worker: "", text: "Preparing the agent’s tasks." },
  ];
  for (const w of tasks) {
    if (w.start !== null)
      activity.push({
        id: w.id + "-start",
        t: w.start,
        worker: w.name,
        text: "Task started.",
      });
    for (const c of w.calls) {
      activity.push({
        id: c.id,
        t: c.start,
        worker: w.name,
        text:
          c.kind === "model" ? "Calling the language model." : c.title + ".",
      });
      if (c.end !== null && (c.kind === "tool" || c.failed || c.where))
        activity.push({
          id: c.id + "-end",
          t: c.end,
          worker: w.name,
          text:
            (c.failed
              ? c.title + " failed."
              : c.kind === "model"
                ? `Model answered: ${shortModel(c.model)}${c.where ? " on " + c.where : ""}.`
                : "Completed: " + c.title.toLowerCase() + ".") +
            (c.evidence ? " " + c.evidence + "." : ""),
        });
    }
    if (w.end !== null)
      activity.push({
        id: w.id + "-end",
        t: w.end,
        worker: w.name,
        text: "Task complete.",
      });
  }
  activity.push({
    id: "finished",
    t: duration,
    worker: "",
    text:
      record.run.status === "completed"
        ? "Finished. The answer is ready to view."
        : "Execution stopped before completing.",
  });
  return {
    duration,
    tasks,
    samples,
    models: tasks
      .flatMap((t) => t.calls)
      .filter((c) => c.kind === "model" && c.model)
      .reduce<{ family: ModelFamily; label: string }[]>((acc, c) => {
        const label = modelLabel(c.model);
        if (!acc.some((m) => m.label === label))
          acc.push({ family: c.family as ModelFamily, label });
        return acc;
      }, [])
      // Lightest shade first, so the legend reads left to right the way the bars grade.
      .sort(
        (a, b) => SHADE_ORDER.indexOf(a.family) - SHADE_ORDER.indexOf(b.family),
      ),
    gpuCount,
    output: record.run.output,
    status: record.run.status,
    error: record.run.error || "",
    activity: activity.sort((a, b) => a.t - b.t),
  };
}
/** Shape-preserving cubic curve: measured points are retained and never overshot. */
export function smoothCurve(points: [number, number][]) {
  if (!points.length) return { path: "", value: (_x: number) => 0 };
  if (points.length === 1)
    return {
      path: `M${points[0].join(",")}`,
      value: (_x: number) => points[0][1],
    };
  const slopes = points
    .slice(1)
    .map(
      (p, i) => (p[1] - points[i][1]) / Math.max(0.000001, p[0] - points[i][0]),
    );
  const m = points.map((_, i) =>
    i === 0
      ? slopes[0]
      : i === points.length - 1
        ? slopes.at(-1)!
        : (slopes[i - 1] + slopes[i]) / 2,
  );
  slopes.forEach((s, i) => {
    if (!s) {
      m[i] = m[i + 1] = 0;
      return;
    }
    if (m[i] * s < 0) m[i] = 0;
    if (m[i + 1] * s < 0) m[i + 1] = 0;
    const a = m[i] / s,
      b = m[i + 1] / s,
      length = Math.hypot(a, b);
    if (length > 3) {
      m[i] = ((3 * a) / length) * s;
      m[i + 1] = ((3 * b) / length) * s;
    }
  });
  let path = `M${points[0].join(",")}`;
  points.slice(1).forEach((b, i) => {
    const a = points[i],
      h = (b[0] - a[0]) / 3;
    path += ` C${a[0] + h},${a[1] + h * m[i]} ${b[0] - h},${b[1] - h * m[i + 1]} ${b[0]},${b[1]}`;
  });
  return {
    path,
    value: (x: number) => {
      let i = 0;
      while (i < points.length - 2 && points[i + 1][0] < x) i++;
      const a = points[i],
        b = points[i + 1],
        h = b[0] - a[0],
        t = Math.max(0, Math.min(1, (x - a[0]) / Math.max(0.000001, h)));
      return (
        (2 * t ** 3 - 3 * t ** 2 + 1) * a[1] +
        (t ** 3 - 2 * t ** 2 + t) * h * m[i] +
        (-2 * t ** 3 + 3 * t ** 2) * b[1] +
        (t ** 3 - t ** 2) * h * m[i + 1]
      );
    },
  };
}

/** Summarize only calls and measurements reached by playback, keeping missing readings distinct from zero. */
export function executionSummary(run: DemoRun, time: number) {
  const through = Math.min(run.duration, Math.max(0, time));
  const samples = run.samples.filter((sample) => sample.t <= through);
  const average = (key: "cpu" | "gpu") => {
    const values = samples
      .map((sample) => sample[key])
      .filter(
        (value): value is number => value !== null && Number.isFinite(value),
      );
    return values.length
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : null;
  };
  return {
    llmCalls: run.tasks.reduce(
      (count, task) =>
        count +
        task.calls.filter(
          (call) => call.kind === "model" && call.start <= through,
        ).length,
      0,
    ),
    cpuAverage: average("cpu"),
    gpuAverage: average("gpu"),
  };
}

/** "Intel Xeon", "AMD EPYC" or "" from a CPU model string such as "Intel(R) Xeon(R) 6787P". */
export function cpuFamily(model = ""): string {
  return /xeon/i.test(model)
    ? "Intel Xeon"
    : /epyc/i.test(model)
      ? "AMD EPYC"
      : /intel/i.test(model)
        ? "Intel"
        : /amd/i.test(model)
          ? "AMD"
          : "";
}
/** "NVIDIA", "AMD Instinct" or "" from a GPU name. */
export function gpuFamily(name = ""): string {
  return /nvidia|geforce|rtx|tesla|[ahlb]\d{2,3}\b/i.test(name)
    ? "NVIDIA"
    : /instinct|\bmi\d{3}/i.test(name)
      ? "AMD Instinct"
      : "";
}
