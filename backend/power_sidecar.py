"""Power telemetry sidecar: read Intel RAPL package energy as root and publish watts.

    python -m backend.power_sidecar /out/power.csv

The runner containers run unprivileged and the RAPL counters are root-only on the host, so
this sidecar (compose service `power-telemetry`, user 0) reads /sys/class/powercap every two
seconds and writes one line, `<unix time>,<watts>`, atomically. The sampler reads the file
when it is fresh (see backend/system.py). Without RAPL the file is never written and the UI
greys the Power tab out.
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path


def package_energy_uj() -> float | None:
    total, found = 0.0, False
    # The class symlinks do not resolve inside a container; the device tree (bind-mounted read-only) does.
    domains = (
        glob.glob(
            "/host/powercap/intel-rapl/intel-rapl:*"
        )  # compose bind-mounts the host tree here
        or glob.glob("/sys/devices/virtual/powercap/intel-rapl/intel-rapl:*")
        or glob.glob("/sys/class/powercap/intel-rapl:*")
    )
    for domain in domains:
        try:
            if not Path(domain, "name").read_text().strip().startswith("package"):
                continue
            total += float(Path(domain, "energy_uj").read_text().strip())
            found = True
        except (OSError, ValueError):
            continue
    return total if found else None


def main(argv=None) -> int:
    out = Path((argv or sys.argv[1:] or ["/out/power.csv"])[0])
    last = None
    while True:
        energy = package_energy_uj()
        now = time.monotonic()
        if energy is None:
            print(
                "power_sidecar: no readable RAPL package counters; retrying", flush=True
            )
            time.sleep(30)
            continue
        if last and now > last[0] and energy >= last[1]:
            watts = (energy - last[1]) / 1e6 / (now - last[0])
            tmp = out.with_suffix(".tmp")
            tmp.write_text(f"{time.time():.3f},{watts:.1f}\n")
            tmp.replace(out)
        last = (now, energy)
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
