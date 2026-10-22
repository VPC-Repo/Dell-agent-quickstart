#!/usr/bin/env python3
"""Generate deploy/gpu/compose.yaml: one vLLM CUDA container per assignment on the
GPUs the installer found free; a model named more than once gets numbered replica
containers that the gateway load-balances.

    gpu_compose.py --out deploy/gpu/compose.yaml \
        --assign 'Qwen/Qwen3-30B-A3B-Instruct-2507=0' \
        --assign 'openai/gpt-oss-120b=1' --assign 'Qwen/Qwen3-Coder-30B-A3B-Instruct=2,3'

Prints the `model id|engine|device|gpus|container` lines placement.py expects.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

IMAGE = "vllm/vllm-openai:v0.11.0"
BASE_PORT = 2180
NETWORK = "docker_agentic-net"
# Tool-call parsers per model family (vLLM's own names).
PARSERS = [
    (r"gpt-oss", "openai"),
    (r"Qwen3-Coder", "qwen3_coder"),
    (r"Qwen3|Qwen2\.5", "hermes"),
    (r"Llama-3", "llama3_json"),
]


def served_name(model_id: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", model_id.lower().replace("/", "-")).strip("-")


def parser_for(model_id: str) -> str:
    for pattern, name in PARSERS:
        if re.search(pattern, model_id, re.I):
            return name
    return "hermes"


def service(
    model_id: str, gpus: list[int], port: int, hf_token_env: str, replica: int = 1
) -> tuple[str, str]:
    name = served_name(model_id)
    # The first container of a model keeps the plain name (existing installs keep their containers); replicas on
    # other GPUs are numbered. All share the weights through the cache volume.
    container = f"vllm-gpu-{name}" + (f"-{replica}" if replica > 1 else "")
    args = [
        "--model",
        model_id,
        "--served-model-name",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        str(len(gpus)),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        parser_for(model_id),
        # Some repositories ship a second copy of the weights in an `original/` folder (gpt-oss-120b: 61 GB,
        # Llama too). vLLM's default ignore pattern `original/**/*` does not match files directly under it, so
        # the download would fetch them; this one does, and the served weights are the top-level files.
        "--ignore-patterns",
        "original/*",
    ]
    if "gpt-oss" not in model_id:
        args += ["--max-model-len", "32768"]
    quoted = "\n".join(f'      - "{a}"' for a in args)
    # gpt-oss reads its tokenizer vocabulary from the cache too; the installer downloads it there.
    extra_env = (
        "\n      TIKTOKEN_ENCODINGS_BASE: /root/.cache/huggingface/tiktoken"
        if "gpt-oss" in model_id
        else ""
    )
    block = f"""
  {container}:
    image: {IMAGE}
    container_name: {container}
    hostname: {container}
    ipc: host
    environment:
      HF_TOKEN: ${{{hf_token_env}:-}}
      # The installer downloads the weights before any engine starts; the engine reads them from the cache only.
      HF_HUB_OFFLINE: "1"
      VLLM_NO_USAGE_STATS: "1"
      DO_NOT_TRACK: "1"{extra_env}
    command:
{quoted}
    # Loopback only: the gateway reaches the engine by container name on the toolkit network.
    ports:
      - "127.0.0.1:{port}:8000"
    volumes:
      - vllm-gpu-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [{", ".join(f'"{g}"' for g in gpus)}]
              capabilities: [gpu]
    restart: unless-stopped
    networks:
      - agentic-net
    healthcheck:
      test: ["CMD-SHELL", "python3 -c \\"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\\""]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 1800s
"""
    placement = f"{model_id}|vllm-cuda|cuda|{','.join(map(str, gpus))}|{container}"
    return block, placement


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="deploy/gpu/compose.yaml")
    parser.add_argument(
        "--assign", action="append", default=[], help="model id=gpu[,gpu]"
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args()
    if not args.assign:
        raise SystemExit("Give at least one --assign model=gpu")
    blocks, placements, used, copies = [], [], set(), {}
    for index, spec in enumerate(args.assign):
        model_id, gpu_list = spec.split("=", 1)
        model_id = model_id.strip()
        gpus = [int(g) for g in gpu_list.split(",") if g.strip()]
        if (
            not gpus
            or any(g < 0 for g in gpus)
            or len(gpus) != len(set(gpus))
            or used & set(gpus)
        ):
            raise SystemExit(f"Bad or overlapping GPU assignment: {spec!r}")
        used |= set(gpus)
        copies[model_id] = copies.get(model_id, 0) + 1
        block, placement = service(
            model_id, gpus, BASE_PORT + index, args.hf_token_env, copies[model_id]
        )
        blocks.append(block)
        placements.append(placement)
    text = (
        "# vLLM CUDA engines added by deploy/setup.sh next to the Intel toolkit's stack.\n"
        "# Generated by deploy/gpu_compose.py; regenerate rather than editing.\n"
        "services:"
        + "".join(blocks)
        + f"""
volumes:
  vllm-gpu-cache:
networks:
  agentic-net:
    external: true
    name: {NETWORK}
"""
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("\n".join(placements))


if __name__ == "__main__":
    main()
