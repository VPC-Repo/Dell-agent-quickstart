// The demo: the agents run on this machine. The catalog
// comes from the API, a run is started with POST and polled every second, and the run's clock is the server's.
import { useEffect, useMemo, useRef, useState } from "react";
import { Gauge } from "lucide-react";
import {
  adaptRun,
  catalog,
  cleanDescription,
  cpuFamily,
  gpuFamily,
  type DemoRun,
  type RunRecord,
} from "./demoModel";
import {
  AboutDialog,
  AnswerDialog,
  DemoHeader,
  ExecutionScreen,
  SelectionScreen,
  roundDuration,
  serverName,
  type CatalogItem,
  type HostCard,
  type Screen,
} from "./demoScreens";
import { Drawer } from "./components";
import SystemPanel from "./SystemPanel";

type Item = Record<string, any>;
const ACTIVE = ["queued", "running", "waiting"];

export default function LiveDemo({
  api,
  get,
  ready,
  error,
  onConnect,
  authRequired,
}: any) {
  const [host, setHost] = useState<Item | null>(null);
  const [agents, setAgents] = useState<Item[]>([]);
  const [status, setStatus] = useState<Item | null>(null);
  const [history, setHistory] = useState<Item[]>([]);
  const [slug, setSlug] = useState(
    sessionStorage.getItem("demo-slug") || "faq-cache",
  );
  const [screen, setScreen] = useState<Screen>("selection");
  const [runId, setRunId] = useState(
    sessionStorage.getItem("quickstart-run") || "",
  );
  const [runDoc, setRunDoc] = useState<Item | null>(null);
  const [runKey, setRunKey] = useState(0);
  const [extra, setExtra] = useState(false);
  const [busy, setBusy] = useState("");
  const [failure, setFailure] = useState("");
  const [dialog, setDialog] = useState<"about" | "answer" | null>(null);
  const [systemOpen, setSystemOpen] = useState(false);
  const alive = useRef(true);

  useEffect(() => {
    document.title = "Agent Execution Demo";
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  // The machine: every 5 s, so the card and the GPU availability follow the host.
  useEffect(() => {
    if (!ready) return;
    let gone = false;
    const poll = () =>
      get("/system")
        .then((d: Item) => {
          if (!gone) setHost(d);
        })
        .catch((e: Error) => {
          if (!gone) setFailure(e.message);
        });
    poll();
    const timer = setInterval(poll, 5000);
    return () => {
      gone = true;
      clearInterval(timer);
    };
  }, [ready]);
  async function loadCatalog() {
    const [items, st, runs] = await Promise.all([
      get("/automations"),
      get("/catalog").catch(() => null),
      get("/runs?limit=200").catch(() => []),
    ]);
    if (!alive.current) return;
    setAgents(items.filter((a: Item) => a.demo));
    setStatus(st);
    setHistory(runs);
  }
  useEffect(() => {
    if (ready) loadCatalog().catch((e) => setFailure(e.message));
  }, [ready]);
  // Reopen a run that was in progress when the page was last left.
  useEffect(() => {
    if (runId && screen === "selection" && agents.length)
      setScreen("execution");
  }, [agents.length]);
  // Poll the open run every second until it finishes; keep polling a finished one a little for the answer.
  useEffect(() => {
    if (!runId || screen !== "execution") return;
    let gone = false;
    const poll = async () => {
      try {
        const d = await get(`/runs/${runId}`);
        if (gone) return;
        setRunDoc(d);
        setFailure("");
        if (!ACTIVE.includes(d.status)) {
          clearInterval(timer);
          loadCatalog().catch(() => {});
        }
      } catch (e: any) {
        if (!gone) setFailure(e.message);
      }
    };
    poll();
    const timer = setInterval(poll, 1000);
    return () => {
      gone = true;
      clearInterval(timer);
    };
  }, [runId, screen, runKey]);

  const gpus: number = host?.gpus?.length || 0;
  const hostCard: HostCard | null = host?.host
    ? {
        slug: "live",
        name: host.host.name || "this server",
        cores: host.host.physical_cores || host.host.logical_cpus || 0,
        gpus,
        cpuName: cpuFamily(host.host.cpu_model),
        gpuName: gpuFamily(host.gpus?.[0]?.name),
      }
    : null;
  const bySlug = useMemo(
    () => new Map(agents.filter((a) => a.demo).map((a) => [a.demo.slug, a])),
    [agents],
  );
  // A run names its agent in blueprint_ref ("<agent id>@<revision>"); the API lists runs newest first.
  const agentIdOf = (r: Item) => String(r.blueprint_ref || "").split("@")[0];
  const lastDuration = (agentId: string) => {
    const r = history.find(
      (x: Item) =>
        agentIdOf(x) === agentId &&
        x.status === "completed" &&
        x.started &&
        x.finished,
    );
    return r ? r.finished - r.started : null;
  };
  const items: CatalogItem[] = catalog.map((c) => {
    // Every installed agent runs on any host: the model-heavy ones are simply slower on CPU inference.
    const a = bySlug.get(c.slug);
    return {
      ...c,
      icon: c.icon as CatalogItem["icon"],
      duration: a ? lastDuration(a.id) : null,
      available: !!a,
    };
  });
  const spec = catalog.find((c) => c.slug === slug) || catalog[0];
  const agent = bySlug.get(spec.slug);
  const selected = items.find((i) => i.slug === spec.slug);

  async function start() {
    if (!agent || !selected?.available || busy) return;
    setBusy("Starting");
    setFailure("");
    try {
      const r = await (
        await api("/automations/" + agent.id + "/run", "POST", {
          revision: agent.revision,
        })
      ).json();
      sessionStorage.setItem("quickstart-run", r.id);
      sessionStorage.setItem("demo-slug", spec.slug);
      setRunDoc(null);
      setRunId(r.id);
      setRunKey((k) => k + 1);
      setScreen("execution");
    } catch (e: any) {
      setFailure(e.message);
    } finally {
      setBusy("");
    }
  }
  function toSelection() {
    setScreen("selection");
    sessionStorage.removeItem("quickstart-run");
    setRunId("");
    setRunDoc(null);
  }

  // The run as the screens draw it, from the API's run document as it stands now.
  const runAgent = runDoc
    ? agents.find((a) => a.id === agentIdOf(runDoc)) || agent
    : agent;
  const active = !!runDoc && ACTIVE.includes(runDoc.status);
  const adapted: DemoRun | null = useMemo(() => {
    if (!runDoc || !runAgent) return null;
    const rec = {
      agent: runAgent,
      run: {
        ...runDoc,
        events: runDoc.events || [],
        output: runDoc.output || "",
        finished: runDoc.finished || 0,
      },
    } as RunRecord;
    return adaptRun(rec);
  }, [runDoc, runAgent]);
  const origin = runDoc?.events?.[0]?.created ?? runDoc?.started ?? 0;
  const liveTime =
    active && runDoc?.server_time && origin
      ? Math.max(0, runDoc.server_time - origin)
      : adapted?.duration || 0;
  const run: DemoRun | null =
    adapted && active
      ? { ...adapted, duration: Math.max(adapted.duration, liveTime + 1) }
      : adapted;
  const liveSpec = catalog.find((c) => c.slug === runAgent?.demo?.slug) || spec;
  const statusLabel = !runDoc
    ? "Starting"
    : active
      ? "In progress"
      : runDoc.status === "completed"
        ? "Complete"
        : "Stopped";

  return (
    <div
      className={`execution-demo ${screen === "execution" && extra ? "has-expanded" : ""}`}
    >
      <DemoHeader
        screen={screen}
        hasRun={!!runId}
        onSelect={() => setScreen("selection")}
        onExecution={() => setScreen("execution")}
        onAbout={() => setDialog("about")}
        extra={
          <>
            <button
              onClick={() => setSystemOpen(true)}
              title="This machine: CPU, memory, GPUs, inference engines, containers"
            >
              <Gauge size={13} /> System
            </button>
            {authRequired && <button onClick={onConnect}>Sign in</button>}
          </>
        }
      />
      {error ? (
        <main className="demo-loading" role="alert">
          <h1>Unable to reach the runner</h1>
          <p>{error}</p>
          <button onClick={onConnect}>Connect to the server</button>
        </main>
      ) : !ready || !hostCard ? (
        <main className="demo-loading" role="status">
          {failure ? `Unable to load the demo: ${failure}` : "Connecting…"}
        </main>
      ) : !agents.length ? (
        <main className="demo-loading">
          <h1>The example agents are not installed yet</h1>
          <p>
            {status?.providers?.length
              ? `Install the ${status.available} agents using ${status.providers[0].name}.`
              : "Connect a model service first (install.sh does this), then install the agents."}
          </p>
          <button
            className="demo-primary"
            disabled={!!busy || !status?.providers?.length}
            onClick={async () => {
              setBusy("Installing");
              try {
                await api("/catalog/install", "POST", {});
                await loadCatalog();
              } catch (e: any) {
                setFailure(e.message);
              } finally {
                setBusy("");
              }
            }}
          >
            {busy ? "Installing…" : "Install the agents"}
          </button>
          {failure && <p role="alert">{failure}</p>}
        </main>
      ) : screen === "selection" ? (
        <SelectionScreen
          hosts={[hostCard]}
          hostSlug="live"
          onHost={() => {}}
          items={items}
          slug={spec.slug}
          onSlug={(s) => {
            setSlug(s);
            sessionStorage.setItem("demo-slug", s);
          }}
          launchTitle={spec.title}
          launchDetail={
            failure
              ? failure
              : !selected?.available
                ? "This agent is not installed on this server."
                : `${serverName(hostCard.name)} · ${(agent?.agents || []).length} ${(agent?.agents || []).length === 1 ? "agent" : "agents"}${selected?.duration ? ` · last run ${roundDuration(selected.duration)}` : ""}`
          }
          launchLabel={busy ? "Starting…" : "Run the agent"}
          launchDisabled={!!busy || !selected?.available}
          onStart={start}
        />
      ) : run && runAgent ? (
        <>
          {failure && <p role="alert">Unable to refresh the run: {failure}</p>}
          <ExecutionScreen
            key={runKey}
            title={liveSpec.title}
            description={liveSpec.description}
            subtitle={`${serverName(hostCard.name)} · ${hostCard.cores} CPU cores${gpus ? ` · ${gpus} GPUs` : " · CPU inference"}`}
            goal={cleanDescription(runAgent.goal || "")}
            run={run}
            time={liveTime}
            active={active}
            statusLabel={statusLabel}
            onRestart={start}
            onAnswer={
              !active && runDoc?.output ? () => setDialog("answer") : undefined
            }
            onLayoutChange={setExtra}
          />
        </>
      ) : (
        <main className="demo-loading" role="status">
          {failure
            ? `Unable to load the run: ${failure}`
            : "Starting the agent…"}
          <p>
            <button onClick={toSelection}>Back to the agents</button>
          </p>
        </main>
      )}
      {dialog === "about" && <AboutDialog onClose={() => setDialog(null)} />}
      {dialog === "answer" && (
        <AnswerDialog
          markdown={runDoc?.output || runDoc?.error || ""}
          onClose={() => setDialog(null)}
        />
      )}
      {systemOpen && (
        <Drawer title="This machine" onClose={() => setSystemOpen(false)} wide>
          <SystemPanel get={get} />
        </Drawer>
      )}
    </div>
  );
}
