// The demo's screens: the top bar, the selection page (this server and the eight agents), the execution page
// (activity feed, utilization strip, agent timeline, answer) and the dialogs. LiveDemo.tsx feeds them the API's data.
import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  Activity as ActivityIcon,
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
  executionSummary,
  seconds,
  smoothCurve,
  type Activity,
  type DemoRun,
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

export function ResourceChart({
  run,
  time,
  compact = false,
}: {
  run: DemoRun;
  time: number;
  compact?: boolean;
}) {
  const rowHeight = compact ? 36 : 42,
    plotHeight = rowHeight * (run.gpuCount > 0 ? 2 : 1) + 6,
    scale = compact ? 0.22 : 0.28;
  const summary = executionSummary(run, time);
  const percent = (value: number | null) =>
    value === null ? "—" : value.toFixed(1) + "%";
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
  const series = useMemo(
    () =>
      (["cpu", ...(run.gpuCount > 0 ? ["gpu" as const] : [])] as const).map(
        (key, index) => {
          const top = index * rowHeight + 7,
            base = top + 100 * scale;
          const data = run.samples.filter((s) => s[key] !== null);
          const points = data.map(
            (s) => [x(s.t), top + (100 - s[key]!) * scale] as [number, number],
          );
          return { key, top, base, data, points, curve: smoothCurve(points) };
        },
      ),
    [run, width, left, rightInset, compact],
  );
  const sample =
    hover === null ? null : run.samples.filter((s) => s.t <= hover).at(-1);
  return (
    <section className="demo-resources" aria-label="Server utilization">
      <div className="demo-panel-heading">
        <div>
          <h2>Server utilization</h2>
          <p className="demo-scale-note">Whole-server capacity · 0–100%</p>
        </div>
        <dl className="demo-run-summary" aria-label="Execution summary">
          <div title="Model calls started so far">
            <dt>Model calls</dt>
            <dd>{summary.llmCalls}</dd>
          </div>
          <div title="Average of the whole-server CPU samples so far">
            <dt>Avg CPU</dt>
            <dd>{percent(summary.cpuAverage)}</dd>
          </div>
          {run.gpuCount > 0 && (
            <div
              title={`Average of the GPU samples so far, across all ${run.gpuCount} GPUs`}
            >
              <dt>Avg GPU</dt>
              <dd>{percent(summary.gpuAverage)}</dd>
            </div>
          )}
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
          aria-label="CPU and GPU utilization over elapsed time. Curves interpolate between measured samples."
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
          {series.map(({ key, top, base, data, points, curve }) => {
            const available = !!points.length && time >= data[0].t,
              lead = x(Math.min(time, run.duration)),
              lastMeasured = points.at(-1)?.[0] ?? lead,
              y = curve.value(lead);
            return (
              <g key={key} className={`demo-resource-${key}`}>
                <text className="demo-chart-title" x="2" y={top + 12}>
                  {key === "cpu" ? "CPU" : "GPU"}
                </text>
                <text x="2" y={top + 27}>
                  {key === "gpu"
                    ? run.gpuCount
                      ? `Mean of ${run.gpuCount} GPUs`
                      : "Not installed"
                    : "All physical cores"}
                </text>
                <line
                  x1={left}
                  x2={right}
                  y1={base}
                  y2={base}
                  className="demo-baseline"
                />
                {available && (
                  <>
                    <defs>
                      <clipPath id={`${id}-${key}`}>
                        <rect
                          x={left - 2}
                          y="0"
                          width={Math.max(0, lead - left + 2)}
                          height="96"
                        />
                      </clipPath>
                    </defs>
                    <g clipPath={`url(#${id}-${key})`}>
                      <path
                        className="demo-chart-fill"
                        d={`${curve.path} L${points.at(-1)![0]},${base} L${points[0][0]},${base} Z`}
                      />
                      <path className="demo-chart-line" d={curve.path} />
                      {lead > lastMeasured && (
                        <path
                          className="demo-chart-line"
                          d={`M${lastMeasured},${y} H${lead}`}
                          strokeDasharray="3 3"
                        >
                          <title>
                            Last measured value held through the end of the
                            timeline; no additional samples.
                          </title>
                        </path>
                      )}
                    </g>
                    <circle cx={lead} cy={y} r="7" className="demo-lead-glow" />
                    <circle
                      cx={lead}
                      cy={y}
                      r="3.5"
                      className="demo-lead-dot"
                    />
                    <rect
                      x={lead + 8}
                      y={y - 10}
                      width="39"
                      height="20"
                      rx="10"
                      className="demo-lead-pill"
                    />
                    <text
                      x={lead + 27.5}
                      y={y + 4}
                      textAnchor="middle"
                      className="demo-lead-value"
                    >
                      {Math.round((base - y) / scale)}%
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
        {sample && (
          <div className="demo-chart-tooltip" role="tooltip">
            Measured at {seconds(sample.t)} · CPU{" "}
            {sample.cpu?.toFixed(1) ?? "—"}%
            {sample.gpu !== null ? ` · GPU ${sample.gpu.toFixed(1)}%` : ""}
          </div>
        )}
      </div>
    </section>
  );
}

export function ActivityFeed({
  run,
  time,
  active,
  expanded,
  onToggle,
}: {
  run: DemoRun;
  time: number;
  active: boolean;
  expanded: boolean;
  onToggle: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null),
    pinned = useRef(true);
  // While a live run is still going, its closing line ("Finished") is not yet true.
  const history = run.activity.filter(
    (e) => e.t <= time && !(active && e.id === "finished"),
  );
  let rows: Activity[] = history;
  if (!expanded) {
    const current = run.tasks.flatMap((w) =>
      w.calls
        .filter(
          (c) =>
            c.start <= time &&
            (c.end === null ? active || time < run.duration : c.end > time),
        )
        .map((c) => ({ w, c })),
    );
    const seen = new Set<string>();
    rows = [];
    for (const event of [...history].reverse()) {
      const key = event.worker || event.id;
      if (seen.has(key)) continue;
      seen.add(key);
      rows.unshift(event);
      if (rows.length === (current.length ? 2 : 3)) break;
    }
    if (run.tasks.length <= 3) rows = history.slice(-(current.length ? 2 : 3));
    if (current.length) {
      const models = current.filter((x) => x.c.kind === "model").length,
        tools = current.length - models;
      rows.push({
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
      });
    }
  }
  const complete = !active && time >= run.duration;
  if (!expanded) rows = complete ? history.slice(-1) : rows.slice(-2);
  const lastId = history.at(-1)?.id;
  useEffect(() => {
    if (expanded && pinned.current && ref.current)
      ref.current.scrollTop = ref.current.scrollHeight;
  }, [expanded, lastId]);
  return (
    <section
      className={`demo-activity ${expanded ? "expanded" : ""}`}
      aria-label={complete ? "Activity" : "Ongoing activity"}
    >
      <h2>{complete ? "Activity" : "Ongoing activity"}</h2>
      <div
        ref={ref}
        className="demo-activity-lines"
        role="log"
        aria-live={expanded ? "off" : "polite"}
        onScroll={(e) => {
          const el = e.currentTarget;
          pinned.current =
            el.scrollHeight - el.scrollTop - el.clientHeight < 24;
        }}
      >
        {rows.map((e) => (
          <div key={e.id} className="demo-event">
            <time>{e.id === "now" ? "Now" : seconds(e.t)}</time>
            <span>
              {e.worker && <strong>{e.worker} · </strong>}
              {e.text}
            </span>
          </div>
        ))}
      </div>
      <button
        onClick={() => {
          pinned.current = true;
          onToggle();
        }}
        aria-expanded={expanded}
      >
        {expanded ? "Hide history" : "Activity history"}{" "}
        <ChevronDown size={13} className={expanded ? "rotated" : ""} />
      </button>
    </section>
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
        <div className="demo-servers">
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
  const [activityExpanded, setActivityExpanded] = useState(false),
    [expanded, setExpanded] = useState<string[]>([]),
    [taskOpen, setTaskOpen] = useState(false),
    [callExpanded, setCallExpanded] = useState<string[]>([]),
    [showAllCalls, setShowAllCalls] = useState(false);
  const compact = run.tasks.length > 0 && run.tasks.length <= 3;
  const visibleCallCount = run.tasks.reduce(
    (n, task) => n + task.calls.filter((c) => c.start <= time).length,
    0,
  );
  const anyExpanded =
    activityExpanded ||
    expanded.length > 0 ||
    taskOpen ||
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
      <div className="demo-goal">
        <span>{description}</span>
        <button onClick={() => setTaskOpen((v) => !v)} aria-expanded={taskOpen}>
          {taskOpen ? "Hide description" : "Task description"}{" "}
          <ChevronDown size={13} className={taskOpen ? "rotated" : ""} />
        </button>
      </div>
      {taskOpen && <div className="demo-goal-detail">{goal}</div>}
      <ActivityFeed
        run={run}
        time={time}
        active={active}
        expanded={activityExpanded}
        onToggle={() => setActivityExpanded((v) => !v)}
      />
      <div className="demo-execution-panel demo-glass">
        <ResourceChart run={run} time={time} compact />
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
            <small title="Dashed segments hold the last measured utilization until the run ends.">
              Dashed line: last measured value
            </small>
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
      <p>
        This demo runs real agents on this server and measures it while they
        work. The business records are fictional.
      </p>
      <p>
        Every worker is an agent calling a language model and real tools: a data
        warehouse, a document library, a code sandbox. Nothing is pre-recorded
        here.
      </p>
      <p>
        CPU utilization covers all physical cores. GPU utilization is the mean
        across the installed GPUs. The curves interpolate between measurements;
        the call bars show elapsed time, including waits, rather than exclusive
        use of a processor.
      </p>
    </Dialog>
  );
}
