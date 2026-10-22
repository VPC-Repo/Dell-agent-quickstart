import { useEffect, useState } from "react";
import { ChevronRight, Gauge } from "lucide-react";
import { cpuSummary, machine } from "./machine";

type Item = Record<string, any>;
const gb = (n?: number | null) =>
  n == null ? "–" : (n / 2 ** 30).toFixed(n > 100 * 2 ** 30 ? 0 : 1) + " GB";

function Bar({
  value,
  label,
  detail,
  tone,
}: {
  value: number | null | undefined;
  label: string;
  detail?: string;
  tone?: string;
}) {
  const v = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div className={"qs-bar " + (tone || "")}>
      <div className="qs-bar-label">
        <span>{label}</span>
        <b>{value == null ? "–" : `${Math.round(value)}%`}</b>
      </div>
      <div className="qs-bar-track">
        <div className="qs-bar-fill" style={{ width: v + "%" }} />
      </div>
      {detail && <small>{detail}</small>}
    </div>
  );
}

export default function SystemPanel({ get, collapsed, onToggle }: any) {
  const [data, setData] = useState<Item | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let gone = false;
    const poll = async () => {
      try {
        const d = await get("/system");
        if (!gone) {
          setData(d);
          setError("");
        }
      } catch (e: any) {
        if (!gone) setError(e.message);
      }
    };
    void poll();
    const timer = setInterval(poll, collapsed ? 6000 : 2000);
    return () => {
      gone = true;
      clearInterval(timer);
    };
  }, [collapsed]);
  if (collapsed)
    return (
      <button
        className="qs-system-toggle"
        onClick={onToggle}
        title="Show system load"
      >
        <Gauge size={16} />
        <span>System</span>
        {data && <small>CPU {Math.round(data.cpu?.percent ?? 0)}%</small>}
      </button>
    );
  const engines: Item[] = data?.engines || [];
  const containers: Item[] = (data?.containers || []).slice(0, 10);
  const m = machine(data);
  return (
    <aside className="qs-system" aria-label="System load">
      <header>
        <Gauge size={16} />
        <strong>{data?.host?.name || "System"}</strong>
        <button
          className="qs-link"
          onClick={onToggle}
          aria-label="Hide system panel"
        >
          <ChevronRight size={16} />
        </button>
      </header>
      {error && <small className="qs-system-error">{error}</small>}
      {data && (
        <>
          <small className="qs-system-host">
            {data.host.cpu_model || "CPU"} · {m ? cpuSummary(m) : ""}
            {data.cpu.load?.length
              ? ` · load ${data.cpu.load.join(" / ")}`
              : ""}
          </small>
          <section>
            <h4>CPU</h4>
            <Bar
              value={data.cpu.percent}
              label="Utilization"
              detail={m ? `${m.cores} cores, SMT-aware` : ""}
            />
            {data.cpu.sockets.length > 1 && (
              <small className="qs-sockets">
                {data.cpu.sockets
                  .map(
                    (s: Item) =>
                      `socket ${s.id} ${Math.round(s.percent)}% (${s.physical_cores} cores, ${s.threads} threads)`,
                  )
                  .join(" · ")}
              </small>
            )}
            <div className="qs-cores" title="Per-core utilization">
              {data.cpu.per_core.map((v: number, i: number) => (
                <i key={i} style={{ opacity: 0.15 + (v / 100) * 0.85 }} />
              ))}
            </div>
          </section>
          <section>
            <h4>Memory</h4>
            <Bar
              value={data.memory.percent}
              label="Used"
              detail={`${gb(data.memory.used_bytes)} of ${gb(data.memory.total_bytes)}`}
              tone="mem"
            />
          </section>
          <section>
            <h4>GPUs</h4>
            {data.gpus.length ? (
              data.gpus.map((g: Item) => (
                <div key={g.index} className="qs-gpu">
                  <div className="qs-gpu-name">
                    <b>GPU {g.index}</b> <span>{g.name}</span>
                  </div>
                  <Bar
                    value={g.utilization_percent}
                    label="Utilization"
                    tone="gpu"
                  />
                  <Bar
                    value={
                      g.memory_total_mb
                        ? (g.memory_used_mb / g.memory_total_mb) * 100
                        : null
                    }
                    label="Memory"
                    detail={`${Math.round((g.memory_used_mb || 0) / 1024)} of ${Math.round((g.memory_total_mb || 0) / 1024)} GB${g.power_w != null ? ` · ${Math.round(g.power_w)} W` : ""}${g.temperature_c != null ? ` · ${g.temperature_c} °C` : ""}`}
                    tone="gpu"
                  />
                </div>
              ))
            ) : (
              <small>
                {data.sources.gpu === "unavailable"
                  ? "No NVIDIA GPU telemetry on this host"
                  : `GPU telemetry ${data.sources.gpu}`}
              </small>
            )}
          </section>
          <section>
            <h4>Inference engines</h4>
            {engines.length ? (
              <table className="qs-table">
                <tbody>
                  {engines.map((e) => (
                    <tr
                      key={e.model + e.container}
                      className={e.reachable === false ? "down" : ""}
                    >
                      <td>
                        <b>{e.model}</b>
                        <small>
                          {e.engine_label} ·{" "}
                          {e.device === "cuda"
                            ? `GPU ${e.gpus?.join(",")}`
                            : e.device}
                          {e.reachable === false ? " · unreachable" : ""}
                        </small>
                      </td>
                      <td>
                        <b>{e.requests_running ?? "–"}</b>
                        <small>running</small>
                      </td>
                      <td>
                        <b>{e.requests_waiting ?? "–"}</b>
                        <small>waiting</small>
                      </td>
                      <td>
                        <b>{e.kv_cache_percent ?? "–"}%</b>
                        <small>KV cache</small>
                      </td>
                      <td>
                        <b>{e.generation_tokens_per_s ?? "–"}</b>
                        <small>tok/s out</small>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <small>
                No engines in the placement registry (data/placement.json)
              </small>
            )}
          </section>
          <section>
            <h4>Containers</h4>
            {containers.length ? (
              <table className="qs-table">
                <tbody>
                  {containers.map((c) => (
                    <tr key={c.id}>
                      <td>
                        <b>{c.name}</b>
                        <small>{c.role}</small>
                      </td>
                      <td>
                        <b>{c.cpu_percent ?? "–"}%</b>
                        <small>CPU</small>
                      </td>
                      <td>
                        <b>
                          {c.memory_bytes == null ? "–" : gb(c.memory_bytes)}
                        </b>
                        <small>memory</small>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <small>
                {data.sources.containers === "unavailable"
                  ? "Docker socket not mounted"
                  : `Containers: ${data.sources.containers}`}
              </small>
            )}
          </section>
          <footer>
            <small>
              Runs: {data.runs.running ?? "–"} running ·{" "}
              {data.runs.queued ?? "–"} queued · sampled every{" "}
              {data.interval_seconds} s · GPU {data.sources.gpu} · containers{" "}
              {data.sources.containers}
            </small>
          </footer>
        </>
      )}
    </aside>
  );
}
