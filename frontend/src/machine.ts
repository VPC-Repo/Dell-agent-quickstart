// The one machine object every screen renders from. Cores are physical cores;
// utilization is measured per core (SMT-aware), so "N cores" is the denominator
// everywhere a percentage is shown. Threads are shown alongside, never as CPUs.
type Item = Record<string, any>;

export type Machine = {
  cores: number;
  threads: number;
  sockets: number;
  smt: boolean;
  cpuModel: string;
  gpus: number;
  gpuNames: string;
  memoryBytes: number;
};

export function machine(host: Item | null | undefined): Machine | null {
  if (!host?.host) return null;
  const h = host.host;
  const threads = h.logical_cpus || 0;
  const cores = h.physical_cores || threads;
  return {
    cores,
    threads,
    sockets: h.sockets || 1,
    smt: threads > cores,
    cpuModel: (h.cpu_model || "CPU")
      .replace(/\(R\)|\(TM\)|Intel |CPU| @.*/g, "")
      .trim(),
    gpus: host.gpus?.length ?? 0,
    gpuNames: host.gpus?.length
      ? [...new Set<string>(host.gpus.map((g: Item) => g.name))].join(", ")
      : "",
    memoryBytes: host.memory?.total_bytes || 0,
  };
}

/** "172 cores · 344 threads · 2 sockets" (threads and sockets only when they add information). */
export function cpuSummary(m: Machine): string {
  const parts = [`${m.cores.toLocaleString()} cores`];
  if (m.smt) parts.push(`${m.threads.toLocaleString()} threads`);
  if (m.sockets > 1) parts.push(`${m.sockets} sockets`);
  return parts.join(" · ");
}

export const plural = (n: number, one: string, many = one + "s") =>
  `${n.toLocaleString()} ${n === 1 ? one : many}`;
