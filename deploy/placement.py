#!/usr/bin/env python3
"""Write the placement registry the runtime attaches to every model call.

    placement.py --out data/placement.json --host cpu-host \
        --gateway http://agentic-litellm:4000/v1 \
        --model 'Qwen/Qwen3-30B-A3B-Instruct-2507|vllm-cpu|cpu||agentic-vllm-qwen3-30b-a3b-instruct-2507' \
        --model 'openai/gpt-oss-120b|vllm-cuda|cuda|4,5|vllm-gpu-gpt-oss-120b' \
        --alias general=Qwen/Qwen3-30B-A3B-Instruct-2507 --alias reasoning=openai/gpt-oss-120b

Each --model is `model id|engine|device|gpu indexes|container` (the api base is
http://<container>:8000/v1 unless a sixth field gives it). CPU and GPU details
are read from the host (lscpu, nvidia-smi) when available. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import socket
import subprocess
from pathlib import Path


def cpu_detail():
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    fields = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    sockets = fields.get("Socket(s)", "")
    model = (
        fields.get("Model name", "")
        .replace("(R)", "")
        .replace("(TM)", "")
        .replace("  ", " ")
        .strip()
    )
    threads = fields.get("CPU(s)", "")
    # Physical cores are the machine number everywhere in the UI; threads ride along.
    try:
        cores = int(sockets) * int(fields.get("Core(s) per socket", ""))
    except ValueError:
        cores = 0
    size = (
        f"({cores} cores, {threads} threads)"
        if cores and threads and str(cores) != threads
        else f"({threads} cores)"
        if threads
        else ""
    )
    return " ".join(
        x for x in [f"{sockets} x" if sockets else "", model, size] if x
    ).strip()


def gpu_names():
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    names = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            names[int(parts[0])] = f"{parts[1]} ({int(float(parts[2])) // 1024} GB)"
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="data/placement.json")
    parser.add_argument("--host", default=socket.gethostname())
    parser.add_argument("--gateway", default="http://agentic-litellm:4000/v1")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model id|engine|device|gpus|container[|api_base]",
    )
    parser.add_argument("--alias", action="append", default=[], help="alias=model id")
    parser.add_argument(
        "--remote",
        action="append",
        default=[],
        help="model id|api_base for remote endpoints",
    )
    args = parser.parse_args()
    cpu = cpu_detail()
    gpus = gpu_names()
    models = {}
    for spec in args.model:
        parts = spec.split("|")
        if len(parts) < 5:
            raise SystemExit(f"--model needs at least 5 fields: {spec!r}")
        model_id, engine, device, gpu_list, container = parts[:5]
        indexes = [int(g) for g in gpu_list.split(",") if g.strip()]
        api_base = parts[5] if len(parts) > 5 else f"http://{container}:8000/v1"
        detail = (
            cpu
            if device == "cpu"
            else ", ".join(dict.fromkeys(gpus.get(i, f"GPU {i}") for i in indexes))
            or "NVIDIA GPU"
        )
        # A second container serving the same model (a replica on another GPU) keeps its own entry, so a call's
        # placement can name the replica the gateway routed to; aliases point at the first entry.
        copies = sum(1 for k in models if k == model_id or k.startswith(model_id + "#"))
        key = model_id if not copies else f"{model_id}#{copies + 1}"
        models[key] = {
            "engine": engine,
            "host": args.host,
            "device": device,
            "device_detail": detail,
            "gpus": indexes,
            "model_id": model_id,
            "api_base": api_base,
            "container": container,
            "metrics_url": api_base.rsplit("/v1", 1)[0] + "/metrics",
        }
    for spec in args.remote:
        model_id, api_base = spec.split("|", 1)
        host = api_base.split("//", 1)[-1].split("/", 1)[0]
        models[model_id] = {
            "engine": "remote-api",
            "host": host,
            "device": "remote",
            "device_detail": "",
            "gpus": [],
            "model_id": model_id,
            "api_base": api_base,
            "container": "",
        }
    aliases = dict(a.split("=", 1) for a in args.alias)
    unknown = sorted(set(aliases.values()) - set(models))
    if unknown:
        raise SystemExit("Aliases point at unregistered models: " + ", ".join(unknown))
    registry = {
        "host": args.host,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "gateway": args.gateway,
        "cpu": cpu,
        "models": models,
        "aliases": aliases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=1) + "\n")
    print(f"wrote {out} with {len(models)} models and {len(aliases)} aliases")


if __name__ == "__main__":
    main()
