// The demo's screens: the top bar, the selection page (this server and the eight agents), the execution page
// (what the agent uses, the agent timeline, the answer) and the dialogs. LiveDemo.tsx feeds them the API's data.
import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  Activity as ActivityIcon,
  Info,
  ChevronDown,
  ChevronRight,
  Code2,
  Database,
  Files,
  MessageCircle,
  Network,
  Receipt,
  Users,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import dellLogo from "./assets/dell-technologies-blue.svg";
import intelLogo from "./assets/intel-logo-energy-blue.png";
import cpuServerImage from "./assets/server-cpu.png";
import gpuServerImage from "./assets/server-gpu.png";
import {
  GPU_TOKENS_PER_S,
  SHADE_ORDER,
  curveValue,
  executionSummary,
  modelLabel,
  seconds,
  smoothSteps,
  stepIntegral,
  stepValue,
  type Activity,
  type DemoRun,
  type Step,
  type Task,
} from "./demoModel";
import "./demo.css";

export const icons = {
  support: MessageCircle,
  data: Database,
  documents: Files,
  code: Code2,
  activity: ActivityIcon,
  finance: Receipt,
  team: Users,
  network: Network,
};
export const roundDuration = (n: number) => Math.round(n) + " s";
/** The server's display name: what the host calls itself. */
export const serverName = (host: string) => host;
export type HostCard = {
  slug: string;
  name: string;
  cores: number;
  gpus: number;
  /** "Intel Xeon", "AMD EPYC", ... or empty when unknown. */
  cpuName?: string;
  /** "NVIDIA", "AMD Instinct", ... or empty when unknown. */
  gpuName?: string;
};
export type CatalogItem = {
  slug: string;
  title: string;
  description: string;
  category: string;
  icon: keyof typeof icons;
  /** Seconds a run took on the selected server, when one has been measured. */
  duration: number | null;
  /** False when this server cannot run the agent (the GPU agents on a CPU-only server). */
  available: boolean;
};

/** A round number at or above `peak`, for a lane's scale: 1, 2, 5, 10, 20, 50 ... */
function niceMax(peak: number) {
  if (!(peak > 0)) return 1;
  const base = 10 ** Math.floor(Math.log10(peak));
  return (
    [1, 2, 5, 10].map((m) => m * base).find((v) => v >= peak * 1.05) ??
    10 * base
  );
}
const coresText = (v: number) =>
  v >= 10
    ? Math.round(v).toLocaleString()
    : v >= 1
      ? v.toFixed(1)
      : v.toFixed(2);
/** Cores in use; at the machine's core count (the cap, see adaptRecording) it reads "all 172". */
const coresInUse = (v: number, cores: number) =>
  cores && v >= cores ? `all ${cores.toLocaleString()}` : coresText(v);
/** CPUs (sockets) for every GPU, as in the CPU-per-token table: "1.5 CPUs for every GPU", "0.0057 CPUs for every GPU". */
const cpusPerGpuText = (sockets: number) => {
  const v =
    sockets >= 10
      ? Math.round(sockets).toLocaleString()
      : sockets >= 1
        ? sockets.toFixed(1)
        : String(Number(sockets.toPrecision(2)));
  return `${v} ${v === "1.0" ? "CPU" : "CPUs"} for every GPU`;
};

export function ResourceChart({
  run,
  time,
  compact = false,
  status,
  onDetails,
}: {
  run: DemoRun;
  time: number;
  compact?: boolean;
  /** The line under the heading: what the agent is doing now. */
  status?: React.ReactNode;
  /** Opens the task and activity panel; the button sits next to the heading. */
  onDetails?: () => void;
}) {
  const rowHeight = compact ? 48 : 54,
    scale = compact ? 0.3 : 0.36;
  const cores = run.footprint.physicalCores;
  // Smoothed: each step is blurred over a fraction of a second, keeping its area (core-seconds, request-seconds).
  const chart = useMemo(() => {
    const cpu = smoothSteps(run.cores, run.duration);
    const cpuPeak = Math.max(0, ...cpu.map((p) => p.value));
    const order = (l: (typeof run.engines)[number]) =>
      SHADE_ORDER.indexOf(l.family);
    const layers = [...run.engines]
      .sort((a, b) => order(a) - order(b))
      .map((l) => ({ lane: l, curve: smoothSteps(l.requests, run.duration) }));
    // Stacked: each layer sits on the ones below it.
    const stacked = layers.map((layer, i) => ({
      ...layer,
      top: layer.curve.map((p, k) => ({
        t: p.t,
        value: layers.slice(0, i + 1).reduce((n, l) => n + l.curve[k].value, 0),
      })),
      bottom: layer.curve.map((p, k) => ({
        t: p.t,
        value: layers.slice(0, i).reduce((n, l) => n + l.curve[k].value, 0),
      })),
    }));
    const gpuPeak = Math.max(
      0,
      ...(stacked.at(-1)?.top.map((p) => p.value) || []),
    );
    return {
      cpu,
      cpuMax: cores && cpuPeak >= cores * 0.95 ? cores : niceMax(cpuPeak),
      stacked,
      gpuMax: gpuPeak || 1,
    };
  }, [run, cores]);
  const rows = chart.stacked.length ? 2 : 1;
  const plotHeight = rowHeight * rows + 10;
  const ref = useRef<HTMLDivElement>(null),
    id = useId().replaceAll(":", "");
  const [width, setWidth] = useState(800),
    [left, setLeft] = useState(190),
    [rightInset, setRightInset] = useState(55);
  useEffect(() => {
    const el = ref.current!;
    const observer = new ResizeObserver(() => {
      setWidth(el.clientWidth);
      setRightInset(
        parseFloat(getComputedStyle(el).getPropertyValue("--status-width")) ||
          55,
      );
      setLeft(
        parseFloat(getComputedStyle(el).getPropertyValue("--lane-label")) ||
          190,
      );
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const [hover, setHover] = useState<number | null>(null);
  const right = width - rightInset,
    x = (t: number) => left + ((right - left) * t) / run.duration;
  const through = Math.min(time, run.duration);
  const summary = executionSummary(run, time);
  const coreSeconds = stepIntegral(run.cores, through),
    peakSoFar = Math.max(
      0,
      ...run.cores.filter((c) => c.t <= through).map((c) => c.value),
    );
  const lead = x(through);
  const line = (points: Step[], y: (v: number) => number) =>
    points
      .map(
        (p, k) =>
          `${k ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.value).toFixed(1)}`,
      )
      .join(" ");
  const cpuTop = 12,
    cpuBase = cpuTop + 100 * scale,
    cpuY = (v: number) =>
      cpuBase - (Math.min(v, chart.cpuMax) / chart.cpuMax) * 100 * scale,
    cpuPath = line(chart.cpu, cpuY),
    cpuNow = curveValue(chart.cpu, through);
  const gpuTop = rowHeight + 12,
    gpuBase = gpuTop + 100 * scale,
    gpuY = (v: number) => gpuBase - (v / chart.gpuMax) * 100 * scale;
  const allGpu = run.engines.length > 0 && run.engines.every((l) => l.gpu);
  return (
    <section className="demo-resources" aria-label="What this agent uses">
      <div className="demo-panel-heading">
        <div>
          <div className="demo-resources-title">
            <h2
              title={`Its own processes, sandbox jobs and services, and the ${allGpu ? "GPUs" : "model engine"} working on its requests`}
            >
              What this agent uses
            </h2>
            {onDetails && (
              <button className="demo-details-button" onClick={onDetails}>
                Task and activity
              </button>
            )}
          </div>
          {status}
        </div>
        <dl className="demo-run-summary" aria-label="Execution summary">
          <div title="Model calls started so far">
            <dt>Model calls</dt>
            <dd>{summary.llmCalls}</dd>
          </div>
          <div title="The most cores this agent has used at once so far">
            <dt>Peak cores</dt>
            <dd>{coresInUse(peakSoFar, cores)}</dd>
          </div>
          <div title="CPU time this agent has used so far, in core-seconds">
            <dt>CPU time</dt>
            <dd>
              {coreSeconds >= 10
                ? Math.round(coreSeconds)
                : coreSeconds.toFixed(1)}{" "}
              core-s
            </dd>
          </div>
        </dl>
      </div>
      <div
        ref={ref}
        className="demo-resource-plot"
        style={{ height: plotHeight }}
        onPointerLeave={() => setHover(null)}
      >
        <svg
          viewBox={`0 0 ${width} ${plotHeight}`}
          style={{ height: plotHeight }}
          role="img"
          aria-label="Cores this agent uses, and when each GPU is working on its requests, over elapsed time"
          onPointerMove={(e) => {
            const t =
              ((e.clientX -
                e.currentTarget.getBoundingClientRect().left -
                left) /
                (right - left)) *
              run.duration;
            setHover(t >= 0 && t <= time ? t : null);
          }}
        >
          <defs>
            <clipPath id={`${id}-clip`}>
              <rect
                x={left - 2}
                y="0"
                width={Math.max(0, lead - left + 2)}
                height={plotHeight}
              />
            </clipPath>
          </defs>
          <g className="demo-resource-cpu">
            <text className="demo-chart-title" x="2" y={cpuTop + 12}>
              CPU
            </text>
            <text x="2" y={cpuTop + 27}>
              cores in use
            </text>
            <text x={left + 4} y={cpuTop - 3} className="demo-scale-label">
              {coresText(chart.cpuMax)} {chart.cpuMax === 1 ? "core" : "cores"}
            </text>
            <line
              x1={left}
              x2={right}
              y1={cpuTop}
              y2={cpuTop}
              className="demo-gridline"
            />
            <line
              x1={left}
              x2={right}
              y1={cpuBase}
              y2={cpuBase}
              className="demo-baseline"
            />
            <g clipPath={`url(#${id}-clip)`}>
              <path
                className="demo-chart-fill"
                d={`${cpuPath} V${cpuBase} H${x(0)} Z`}
              />
              <path className="demo-chart-line" d={cpuPath} />
            </g>
            <circle
              cx={lead}
              cy={cpuY(cpuNow)}
              r="7"
              className="demo-lead-glow"
            />
            <circle
              cx={lead}
              cy={cpuY(cpuNow)}
              r="3.5"
              className="demo-lead-dot"
            />
            <rect
              x={lead + 8}
              y={cpuY(cpuNow) - 10}
              width="44"
              height="20"
              rx="10"
              className="demo-lead-pill"
            />
            <text
              x={lead + 30}
              y={cpuY(cpuNow) + 4}
              textAnchor="middle"
              className="demo-lead-value"
            >
              {coresInUse(cpuNow, cores)}
            </text>
          </g>
          {rows > 1 && (
            <g className="demo-resource-gpu">
              <text className="demo-chart-title" x="2" y={gpuTop + 12}>
                {allGpu ? "GPU" : "Model engine"}
              </text>
              <text x="2" y={gpuTop + 27}>
                working on its requests
              </text>
              <line
                x1={left}
                x2={right}
                y1={gpuBase}
                y2={gpuBase}
                className="demo-baseline"
              />
              <g clipPath={`url(#${id}-clip)`}>
                {chart.stacked.map((layer) => {
                  const top = line(layer.top, gpuY),
                    bottom = [...layer.bottom]
                      .reverse()
                      .map(
                        (p) =>
                          `L${x(p.t).toFixed(1)},${gpuY(p.value).toFixed(1)}`,
                      )
                      .join(" ");
                  return (
                    <g
                      key={layer.lane.key}
                      className={`demo-gpu-layer model-${layer.lane.family}`}
                    >
                      <path
                        className="demo-gpu-fill"
                        d={`${top} ${bottom} Z`}
                      />
                      <path className="demo-chart-line" d={top} />
                    </g>
                  );
                })}
              </g>
            </g>
          )}
        </svg>
        {hover !== null && (
          <div className="demo-chart-tooltip" role="tooltip">
            At {seconds(hover)} ·{" "}
            {coresInUse(curveValue(chart.cpu, hover), cores)} cores
            {chart.stacked
              .filter((l) => stepValue(l.lane.requests, hover) > 0)
              .map(
                (l) =>
                  ` · ${l.lane.place} working (${modelLabel(l.lane.model)})`,
              )
              .join("")}
          </div>
        )}
      </div>
      {chart.stacked.length > 0 && (
        <ul className="demo-gpu-legend" aria-label="Engines">
          {[...chart.stacked].reverse().map((l) => (
            <li key={l.lane.key}>
              <i className={`model-${l.lane.family}`} />
              {l.lane.place} · {modelLabel(l.lane.model)}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** A number for the explanation, to three significant figures (1,234 · 42.6 · 5.83 · 0.0646): each step's
 * arithmetic checks out to its last digit, and the steps say "≈" for that last digit. */
const stepNumber = (n: number) =>
  n >= 100 ? Math.round(n).toLocaleString() : String(Number(n.toPrecision(3)));

/** The info button next to "N CPUs for every GPU": the calculation in three steps, with this run's numbers. */
function RatioInfo({ run }: { run: DemoRun }) {
  const [open, setOpen] = useState(false),
    ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: Event) => {
      if (
        e instanceof KeyboardEvent
          ? e.key === "Escape"
          : !ref.current?.contains(e.target as Node)
      )
        setOpen(false);
    };
    addEventListener("pointerdown", close);
    addEventListener("keydown", close);
    return () => {
      removeEventListener("pointerdown", close);
      removeEventListener("keydown", close);
    };
  }, [open]);
  const f = run.footprint,
    perToken = (f.coreSeconds * 1000) / f.outputTokens,
    cores = (perToken * GPU_TOKENS_PER_S) / 1000;
  return (
    <span ref={ref} className="demo-info">
      <button
        className="demo-info-button"
        aria-label="How this is worked out"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <Info size={14} />
      </button>
      {open && (
        <span
          className="demo-info-pop"
          role="dialog"
          aria-label="How this is worked out"
        >
          <strong>How this is worked out</strong>
          <ol>
            <li>
              This run used{" "}
              {f.coreSeconds >= 10
                ? Math.round(f.coreSeconds).toLocaleString()
                : f.coreSeconds.toFixed(1)}{" "}
              core-seconds of CPU while the model wrote{" "}
              {f.outputTokens.toLocaleString()} tokens: about{" "}
              {stepNumber(perToken)} core-milliseconds for each token.
            </li>
            <li>
              A GPU writes about {GPU_TOKENS_PER_S.toLocaleString()} tokens a
              second. Keeping up with it takes {stepNumber(perToken)} ×{" "}
              {GPU_TOKENS_PER_S.toLocaleString()} ÷ 1,000 ≈ {stepNumber(cores)}{" "}
              cores.
            </li>
            <li>
              One CPU in this server has {f.coresPerSocket} cores, so that is{" "}
              {stepNumber(cores)} ÷ {f.coresPerSocket} ≈{" "}
              {cpusPerGpuText(f.socketsPerGpu!).replace(/ for every GPU$/, "")}{" "}
              for every GPU.
            </li>
          </ol>
          <small>
            An estimate: it scales this one run up to enough runs like it to
            keep a GPU fully busy, with CPU time growing in step.
          </small>
        </span>
      )}
    </span>
  );
}

/** What one run used, in one line: shown when the run has finished. */
export function FootprintPanel({ run }: { run: DemoRun }) {
  const f = run.footprint;
  return (
    <div className="demo-footprint" role="group" aria-label="What one run used">
      <strong>One run used</strong>
      <span>
        {f.coreSeconds >= 10
          ? Math.round(f.coreSeconds).toLocaleString()
          : f.coreSeconds.toFixed(1)}{" "}
        core-seconds
      </span>
      <span>{f.outputTokens.toLocaleString()} output tokens</span>
      {f.socketsPerGpu !== null && (
        <span className="demo-ratio">
          {cpusPerGpuText(f.socketsPerGpu)}
          <RatioInfo run={run} />
        </span>
      )}
    </div>
  );
}

/** The run's history so far (a live run's closing line is not yet true), and its latest line: what is happening
 * now, from the calls in progress, or the last thing that happened. */
function activityNow(run: DemoRun, time: number, active: boolean) {
  const history = run.activity.filter(
    (e) => e.t <= time && !(active && e.id === "finished"),
  );
  const current = run.tasks.flatMap((w) =>
    w.calls
      .filter(
        (c) =>
          c.start <= time &&
          (c.end === null ? active || time < run.duration : c.end > time),
      )
      .map((c) => ({ w, c })),
  );
  let now: Activity | undefined = history.at(-1);
  if (current.length) {
    const models = current.filter((x) => x.c.kind === "model").length,
      tools = current.length - models;
    now = {
      id: "now",
      t: time,
      worker: current.length === 1 ? current[0].w.name : "",
      text:
        current.length === 1
          ? current[0].c.title + " in progress."
          : [
              models
                ? `${models} model ${models === 1 ? "call" : "calls"}`
                : "",
              tools ? `${tools} tool ${tools === 1 ? "call" : "calls"}` : "",
            ]
              .filter(Boolean)
              .join(" and ") + " in progress.",
    };
  }
  return { history, now };
}

/** One line in the panel header: what the agent is doing now. */
export function StatusLine({
  run,
  time,
  active,
}: {
  run: DemoRun;
  time: number;
  active: boolean;
}) {
  const { now } = activityNow(run, time, active);
  return (
    <p className="demo-status-line" aria-live="polite">
      <span>
        <time>{seconds(Math.min(time, run.duration))}</time>
        {now?.worker && <strong>{now.worker} · </strong>}
        {now?.text || "Preparing the agent’s tasks."}
      </span>
    </p>
  );
}

/** A side panel the visitor opts into: the task and the run's full activity, while the run keeps playing. */
export function DetailsDrawer({
  description,
  goal,
  run,
  time,
  active,
  onClose,
}: {
  description: string;
  goal: string;
  run: DemoRun;
  time: number;
  active: boolean;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null),
    pinned = useRef(true);
  const { history } = activityNow(run, time, active);
  const lastId = history.at(-1)?.id;
  useEffect(() => {
    if (pinned.current && ref.current)
      ref.current.scrollTop = ref.current.scrollHeight;
  }, [lastId]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    addEventListener("keydown", key);
    return () => removeEventListener("keydown", key);
  }, [onClose]);
  return (
    <aside className="demo-drawer" aria-label="Task and activity">
      <header>
        <h2>Task and activity</h2>
        <button onClick={onClose} aria-label="Close">
          <X size={18} />
        </button>
      </header>
      <section className="demo-drawer-task">
        <h3>Task</h3>
        <p>{description}</p>
        {goal && goal !== description && (
          <p className="demo-drawer-goal">{goal}</p>
        )}
      </section>
      <section className="demo-drawer-activity">
        <h3>Activity</h3>
        <div
          ref={ref}
          className="demo-drawer-log"
          role="log"
          onScroll={(e) => {
            const el = e.currentTarget;
            pinned.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
        >
          {history.map((e) => (
            <div key={e.id} className="demo-event">
              <time>{seconds(e.t)}</time>
              <span>
                {e.worker && <strong>{e.worker} · </strong>}
                {e.text}
              </span>
            </div>
          ))}
        </div>
      </section>
    </aside>
  );
}

export function TaskRow({
  task,
  run,
  time,
  active,
  roomy,
  expanded,
  onToggle,
}: {
  task: Task;
  run: DemoRun;
  time: number;
  /** A live run still in progress: nothing has "stopped" yet. */
  active: boolean;
  roomy: boolean;
  expanded: boolean;
  onToggle: () => void;
}) {
  const started = task.start !== null && time >= task.start,
    complete = task.end !== null && time >= task.end;
  const stopped = !active && time >= run.duration && !complete;
  const calls = task.calls.filter((c) => c.start <= time),
    current = calls.filter((c) => c.end === null || c.end > time).at(-1);
  const status = complete
    ? "Complete"
    : stopped
      ? "Stopped"
      : started
        ? "Working"
        : "Waiting";
  const start = task.start ?? 0,
    end = Math.min(task.end ?? run.duration, time);
  return (
    <article
      className={`demo-worker ${roomy ? "roomy" : ""} ${expanded ? "expanded" : ""}`}
    >
      <button
        className="demo-worker-button"
        aria-expanded={expanded}
        aria-controls={`task-${task.id}`}
        onClick={onToggle}
        aria-label={`${expanded ? "Collapse" : "Expand"} ${task.name}`}
      >
        <span className="demo-worker-label">
          {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          <span className={`demo-status-dot ${status.toLowerCase()}`} />
          <strong>{task.name}</strong>
        </span>
        <span className="demo-worker-lane" aria-hidden="true">
          {started && (
            <span
              className="demo-worker-span"
              style={{
                left: `${(start / run.duration) * 100}%`,
                width: `${(Math.max(0, end - start) / run.duration) * 100}%`,
              }}
            />
          )}
          {calls.map((c) => (
            <span
              key={c.id}
              className={`demo-call ${c.kind} ${c.family ? `model-${c.family}` : ""} ${c.failed && c.end !== null && time >= c.end ? "failed" : ""}`}
              style={{
                left: `${(c.start / run.duration) * 100}%`,
                width: `${(Math.max(0, Math.min(c.end ?? run.duration, time) - c.start) / run.duration) * 100}%`,
              }}
            />
          ))}
        </span>
        <span className="demo-worker-status" aria-label={status}>
          {complete ? "" : status}
        </span>
        {roomy && (
          <span className="demo-worker-description">{task.description}</span>
        )}
        {roomy && (
          <span className="demo-worker-current">
            {complete
              ? `${calls.length} ${calls.length === 1 ? "call" : "calls"} completed`
              : current
                ? `${current.kind === "model" ? "Model" : "Tool"} · ${current.title}`
                : started
                  ? "Preparing the next step"
                  : "Waiting for inputs"}
          </span>
        )}
      </button>
      {expanded && (
        <div className="demo-task-detail" id={`task-${task.id}`}>
          <p>{task.description}</p>
          {!calls.length ? (
            <p>Waiting for inputs.</p>
          ) : (
            <ol>
              {calls.map((c) => {
                const ended = c.end !== null && time >= c.end;
                return (
                  <li key={c.id}>
                    <span className={`demo-kind ${c.kind}`}>{c.kind}</span>
                    <div>
                      <strong>{c.title}</strong>
                      <small>
                        {seconds(c.start)} →{" "}
                        {ended
                          ? seconds(c.end!)
                          : stopped
                            ? "stopped"
                            : "in progress"}
                        {ended && c.failed ? " · Failed" : ""}
                        {c.kind === "model" && c.model
                          ? ` · ${c.model.split("/").at(-1)}${c.where ? ` on ${c.where}` : ""}`
                          : ""}
                      </small>
                      {ended && c.evidence && <small>{c.evidence}</small>}
                    </div>
                    <span>
                      {ended
                        ? c.end! - c.start < 1
                          ? `${Math.round((c.end! - c.start) * 1000)} ms`
                          : `${(c.end! - c.start).toFixed(2)} s`
                        : "—"}
                    </span>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      )}
    </article>
  );
}

/** Small runs get visible calls; utilization never determines the layout. */
function CompactCalls({
  run,
  time,
  active,
  expanded,
  onToggle,
  showAll,
}: {
  run: DemoRun;
  time: number;
  active: boolean;
  expanded: string[];
  onToggle: (id: string) => void;
  showAll: boolean;
}) {
  const calls = run.tasks
    .flatMap((task) => task.calls.map((c) => ({ ...c, worker: task.name })))
    .filter((c) => c.start <= time)
    .sort((a, b) => a.start - b.start);
  const limit = 2;
  const visible = showAll ? calls : calls.slice(-limit);
  // A finished run has a known final call count. A running one reserves the full preview
  // capacity so new calls never borrow height from the worker timeline.
  const slots = active
    ? limit
    : Math.min(
        limit,
        run.tasks.reduce((n, task) => n + task.calls.length, 0),
      );
  const stopped = !active && time >= run.duration;
  return (
    <div
      className="demo-compact-calls"
      style={{ "--call-slots": Math.max(1, slots) } as React.CSSProperties}
    >
      {!calls.length && (
        <p className="demo-call-waiting">
          Calls will appear as the agent works.
        </p>
      )}
      <ol aria-label="Agent calls" className={showAll ? "all-calls" : ""}>
        {visible.map((c) => {
          const ended = c.end !== null && time >= c.end;
          const open = expanded.includes(c.id);
          const status = ended
            ? c.failed
              ? "Failed"
              : "Complete"
            : stopped
              ? "Stopped"
              : "Working";
          const duration = ended ? c.end! - c.start : null;
          return (
            <li key={c.id}>
              <button
                className="demo-compact-call"
                aria-expanded={open}
                aria-controls={`call-${c.id}`}
                onClick={() => onToggle(c.id)}
              >
                <span className="demo-call-number">
                  {open ? (
                    <ChevronDown size={13} />
                  ) : (
                    <ChevronRight size={13} />
                  )}
                </span>
                <span className={`demo-kind ${c.kind}`}>{c.kind}</span>
                <strong>
                  {run.tasks.length > 1 && (
                    <span className="demo-call-worker">{c.worker} · </span>
                  )}
                  {c.title}
                </strong>
                <span className="demo-call-duration">
                  {duration === null
                    ? "—"
                    : duration < 1
                      ? `${Math.round(duration * 1000)} ms`
                      : `${duration.toFixed(2)} s`}
                </span>
                <span className="demo-call-state" aria-label={status}>
                  {status === "Complete" ? "" : status}
                </span>
              </button>
              {open && (
                <div id={`call-${c.id}`} className="demo-compact-call-detail">
                  <p>
                    {run.tasks.length > 1 ? `${c.worker} · ` : ""}
                    {c.title}
                  </p>
                  <p>
                    Started {seconds(c.start)}
                    {ended ? ` · Finished ${seconds(c.end!)}` : ` · ${status}`}
                  </p>
                  {c.model && (
                    <p>
                      {c.model.split("/").at(-1)}
                      {c.where ? ` on ${c.where}` : ""}
                    </p>
                  )}
                  {ended && c.evidence && <p>{c.evidence}</p>}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export function Dialog({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current!;
    el.showModal();
    return () => el.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className="demo-dialog"
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <header>
        <h2>{title}</h2>
        <button onClick={onClose} aria-label="Close">
          <X size={18} />
        </button>
      </header>
      <div className="demo-dialog-content">{children}</div>
    </dialog>
  );
}

export type Screen = "selection" | "execution";

export function DemoHeader({
  screen,
  hasRun,
  onSelect,
  onExecution,
  onAbout,
  extra,
}: {
  screen: Screen;
  hasRun: boolean;
  onSelect: () => void;
  onExecution: () => void;
  onAbout: () => void;
  extra?: React.ReactNode;
}) {
  return (
    <header className="demo-header">
      <div className="demo-brand">
        <img src={dellLogo} alt="Dell Technologies" />
        <img src={intelLogo} alt="Intel" />
        <span>Agent Execution Demo</span>
      </div>
      <nav aria-label="Main navigation">
        <button
          aria-current={screen === "selection" ? "page" : undefined}
          onClick={onSelect}
        >
          Agents
        </button>
        <button
          aria-current={screen === "execution" ? "page" : undefined}
          disabled={!hasRun}
          onClick={onExecution}
        >
          Execution
        </button>
        <button onClick={onAbout}>About</button>
        {extra}
      </nav>
    </header>
  );
}

export function SelectionScreen({
  hosts,
  hostSlug,
  onHost,
  items,
  slug,
  onSlug,
  launchTitle,
  launchDetail,
  launchLabel,
  launchDisabled,
  onStart,
}: {
  hosts: HostCard[];
  hostSlug: string;
  onHost: (slug: string) => void;
  items: CatalogItem[];
  slug: string;
  onSlug: (slug: string) => void;
  launchTitle: string;
  launchDetail: string;
  launchLabel: string;
  launchDisabled?: boolean;
  onStart: () => void;
}) {
  return (
    <main className="demo-selection">
      <div className="demo-intro">
        <h1>See how agents get work done.</h1>
        <p>
          {hosts.length > 1 ? "Choose a server and a task" : "Choose a task"},
          then watch the calls and the server activity together.
        </p>
      </div>
      <section aria-label="Server">
        <div className="demo-section-heading">
          <h2>Server</h2>
        </div>
        <div
          className="demo-servers"
          style={
            {
              "--server-count": Math.max(2, hosts.length),
            } as React.CSSProperties
          }
        >
          {hosts.map((h) => (
            <button
              key={h.slug}
              className="demo-server demo-glass"
              aria-pressed={hostSlug === h.slug}
              onClick={() => onHost(h.slug)}
              disabled={hosts.length === 1}
            >
              <img src={h.gpus ? gpuServerImage : cpuServerImage} alt="" />
              <span className="demo-server-copy">
                <strong>{serverName(h.name)}</strong>
                <small>
                  {h.cores} {h.cpuName ? `${h.cpuName} ` : ""}CPU cores
                  <br />
                  {h.gpus
                    ? `${h.gpus} ${h.gpuName ? `${h.gpuName} ` : ""}GPUs · GPU inference`
                    : "CPU inference · No GPUs"}
                </small>
              </span>
              <span className="demo-server-check" aria-hidden="true">
                {hostSlug === h.slug ? "✓" : ""}
              </span>
            </button>
          ))}
        </div>
      </section>
      <section className="demo-agent-selection" aria-label="Choose a task">
        <div className="demo-section-heading">
          <h2>Task</h2>
        </div>
        <div className="demo-catalog">
          {items.map((c) => (
            <button
              key={c.slug}
              className="demo-agent demo-glass"
              aria-pressed={slug === c.slug}
              onClick={() => onSlug(c.slug)}
            >
              <span className="demo-agent-category">
                {c.category}
                <span aria-hidden="true">{slug === c.slug ? "✓" : ""}</span>
              </span>
              <h3>{c.title}</h3>
              <p>{c.description}</p>
              {!c.available && (
                <small className="demo-availability">
                  Not available on this server
                </small>
              )}
            </button>
          ))}
        </div>
      </section>
      <div className="demo-launch demo-glass">
        <p aria-label={`Selected task: ${launchTitle}`}>{launchDetail}</p>
        <button
          className="demo-primary"
          onClick={onStart}
          disabled={launchDisabled}
        >
          {launchLabel}
        </button>
      </div>
    </main>
  );
}

export function ExecutionScreen({
  title,
  description,
  subtitle,
  goal,
  run,
  time,
  active,
  statusLabel,
  onRestart,
  onAnswer,
  onLayoutChange,
}: {
  title: string;
  description: string;
  subtitle: string;
  goal: string;
  run: DemoRun;
  time: number;
  active: boolean;
  statusLabel: string;
  onRestart: () => void;
  /** Present once the run has an answer to show. */
  onAnswer?: () => void;
  onLayoutChange?: (expanded: boolean) => void;
}) {
  const [detailsOpen, setDetailsOpen] = useState(false),
    [expanded, setExpanded] = useState<string[]>([]),
    [callExpanded, setCallExpanded] = useState<string[]>([]),
    [showAllCalls, setShowAllCalls] = useState(false);
  const compact = run.tasks.length > 0 && run.tasks.length <= 3;
  const visibleCallCount = run.tasks.reduce(
    (n, task) => n + task.calls.filter((c) => c.start <= time).length,
    0,
  );
  const anyExpanded =
    expanded.length > 0 ||
    (compact && (callExpanded.length > 0 || showAllCalls));
  useEffect(() => onLayoutChange?.(anyExpanded), [anyExpanded]);
  return (
    <main
      className={`demo-execution ${compact ? "compact" : ""}`}
      style={
        {
          "--agent-count": Math.max(1, run.tasks.length),
        } as React.CSSProperties
      }
    >
      <div className="demo-run-heading">
        <div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        <div className="demo-run-actions">
          {statusLabel === "Stopped" && (
            <span className="demo-run-status">{statusLabel}</span>
          )}
          <button onClick={onRestart}>Restart</button>
        </div>
      </div>
      <div className="demo-execution-panel demo-glass">
        <ResourceChart
          run={run}
          time={time}
          compact
          status={<StatusLine run={run} time={time} active={active} />}
          onDetails={() => setDetailsOpen((v) => !v)}
        />
        {!active && time >= run.duration && <FootprintPanel run={run} />}
        <section className="demo-tasks" aria-label="Agent activity">
          <div className="demo-panel-heading">
            <h2>
              Agent activity{" "}
              <span>
                {run.tasks.length} {run.tasks.length === 1 ? "agent" : "agents"}
              </span>
            </h2>
            <span className="demo-elapsed">
              {!active && time >= run.duration
                ? `${run.status === "completed" ? "Completed" : "Stopped"} in ${seconds(run.duration)}`
                : `${seconds(time)}${active ? "" : ` / ${seconds(run.duration)}`}`}
            </span>
          </div>
          <div className="demo-timeline-axis" aria-hidden="true">
            <span>Agent</span>
            <div>
              <span>0 s</span>
              <span>{seconds(run.duration / 2)}</span>
              <span>{seconds(run.duration)}</span>
            </div>
            <span />
          </div>
          <div className="demo-task-list">
            {run.tasks.map((w) => (
              <TaskRow
                key={w.id}
                task={w}
                run={run}
                time={time}
                active={active}
                roomy={!compact && run.tasks.length <= 3}
                expanded={expanded.includes(w.id)}
                onToggle={() =>
                  setExpanded((v) =>
                    v.includes(w.id)
                      ? v.filter((id) => id !== w.id)
                      : [...v, w.id],
                  )
                }
              />
            ))}
          </div>
          <div className="demo-timeline-legend">
            <span>
              <i className="worker" /> Agent active
            </span>
            {run.models.length ? (
              run.models.map((m) => (
                <span key={m.label}>
                  <i className={`model model-${m.family}`} /> Model call (
                  {m.label})
                </span>
              ))
            ) : (
              <span>
                <i className="model" /> Model call
              </span>
            )}
            <span>
              <i className="tool" /> Tool call
            </span>
          </div>
          {compact && (
            <>
              <div className="demo-calls-heading">
                <h3>{showAllCalls ? "All calls" : "Recent calls"}</h3>
                <button
                  className="demo-show-calls"
                  onClick={() => setShowAllCalls((v) => !v)}
                  aria-expanded={showAllCalls}
                >
                  {showAllCalls
                    ? "Show recent calls"
                    : `All ${visibleCallCount} calls`}
                </button>
              </div>
              <CompactCalls
                run={run}
                time={time}
                active={active}
                expanded={callExpanded}
                onToggle={(id) =>
                  setCallExpanded((v) =>
                    v.includes(id) ? v.filter((x) => x !== id) : [...v, id],
                  )
                }
                showAll={showAllCalls}
              />
            </>
          )}
          <section className="demo-answer-preview" aria-label="Answer preview">
            <div>
              <h2>Answer</h2>
              {onAnswer && run.output ? (
                <div className="demo-answer-excerpt">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {run.output}
                  </ReactMarkdown>
                </div>
              ) : (
                <p>
                  {onAnswer
                    ? "This run did not produce an answer."
                    : "The answer will appear here when the agent finishes."}
                </p>
              )}
            </div>
            <button
              onClick={onAnswer}
              disabled={!onAnswer}
              aria-hidden={!onAnswer}
              style={{ visibility: onAnswer ? "visible" : "hidden" }}
            >
              View answer
            </button>
          </section>
        </section>
      </div>
      {detailsOpen && (
        <DetailsDrawer
          description={description}
          goal={goal}
          run={run}
          time={time}
          active={active}
          onClose={() => setDetailsOpen(false)}
        />
      )}
    </main>
  );
}

export function AnswerDialog({
  markdown,
  onClose,
}: {
  markdown: string;
  onClose: () => void;
}) {
  return (
    <Dialog title="Agent answer" onClose={onClose}>
      <div className="demo-answer">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {markdown || "This run did not produce an answer."}
        </ReactMarkdown>
      </div>
    </Dialog>
  );
}

export function AboutDialog({ onClose }: { onClose: () => void }) {
  return (
    <Dialog title="About this demo" onClose={onClose}>
      <div className="demo-about">
        <p className="demo-about-lead">
          Each task is done by one or more AI agents: answering a policy
          question, finding why a service failed, stress-testing a loan
          portfolio. Pick a task and watch what the agents do and what it takes
          from the server&rsquo;s CPUs and GPUs.
        </p>
        <section>
          <h3>What an agent does</h3>
          <p>
            An agent asks a language model what to do next, then does it with
            tools: it queries a data warehouse, reads documents, or runs code in
            a sandbox. Bigger tasks split the work across several agents, some
            working at the same time.
          </p>
        </section>
        <section>
          <h3>Reading the run screen</h3>
          <dl>
            <div>
              <dt>CPU</dt>
              <dd>
                Cores in use by the agent&rsquo;s own work: its processes, the
                sandbox it runs code in, and the database and cache it queries.
              </dd>
            </div>
            <div>
              <dt>GPU</dt>
              <dd>
                When each GPU is working on the agent&rsquo;s requests, one
                color per model. The band grows when several requests run at
                once.
              </dd>
            </div>
            <div>
              <dt>Agent activity</dt>
              <dd>
                One row per agent, with its model calls and tool calls as they
                happen, including the time spent waiting for answers.
              </dd>
            </div>
            <div>
              <dt>One run used</dt>
              <dd>
                The run&rsquo;s CPU time, the tokens the model wrote, and how
                many CPUs to pair with each GPU for this kind of work (taking a
                GPU to write about 950 tokens a second).
              </dd>
            </div>
          </dl>
        </section>
        <p className="demo-about-note">
          The agents run live on this server, with a real model and real tools.
          The business records are fictional.
        </p>
      </div>
    </Dialog>
  );
}
