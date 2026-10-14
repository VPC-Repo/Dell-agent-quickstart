"""Where a model call ran: engine, host and device for each gateway model name.

The installer writes a small JSON registry (see deploy/placement.py) that maps
every LiteLLM model name and role alias to the engine that serves it. The
runtime attaches that record to every `model` event so the live view can show
"vLLM on CPU · cpu-host · 2 x Xeon 6787P" or "vLLM CUDA · GPU 3" next to each call.

Registry shape (all keys optional except `models`):

    {
      "host": "cpu-host",
      "generated": "2026-09-28T20:00:00Z",
      "gateway": "http://agentic-litellm:4000/v1",
      "models": {
        "Qwen/Qwen3-30B-A3B-Instruct-2507": {
          "engine": "vllm-cpu", "host": "cpu-host", "device": "cpu",
          "device_detail": "2 x Intel Xeon 6787P (172 cores)", "gpus": [],
          "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
          "api_base": "http://agentic-vllm-qwen3-30b-a3b-instruct-2507:8000/v1",
          "container": "agentic-vllm-qwen3-30b-a3b-instruct-2507",
          "metrics_url": "http://agentic-vllm-qwen3-30b-a3b-instruct-2507:8000/metrics"
        }
      },
      "aliases": {"general": "Qwen/Qwen3-30B-A3B-Instruct-2507"}
    }

The registry is data only. A model that is not listed is reported as
"unregistered", never guessed from its name.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from urllib.parse import urlsplit

_lock = threading.Lock()
_cache: dict = {"path": None, "mtime": None, "data": None}

ENGINE_LABELS = {
    "vllm-cpu": "vLLM · Intel CPU",
    "vllm-cuda": "vLLM · NVIDIA CUDA",
    "remote-api": "Remote OpenAI-compatible API",
    "litellm": "LiteLLM gateway",
    "mock": "Local fixture",
    "unregistered": "Unregistered endpoint",
}


def registry_path() -> Path:
    return Path(os.getenv("PLACEMENT_REGISTRY", "data/placement.json"))


def load(path: Path | None = None) -> dict:
    """Return the registry, re-reading the file when it changes. Missing or
    invalid files yield an empty registry rather than an error: placement is
    display data and must never stop a run."""
    path = path or registry_path()
    with _lock:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            _cache.update(
                path=str(path), mtime=None, data={"models": {}, "aliases": {}}
            )
            return _cache["data"]
        if _cache["path"] == str(path) and _cache["mtime"] == mtime and _cache["data"]:
            return _cache["data"]
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
                raise ValueError("registry must be an object with a models map")
            data.setdefault("aliases", {})
        except (OSError, ValueError):
            data = {"models": {}, "aliases": {}}
        _cache.update(path=str(path), mtime=mtime, data=data)
        return data


def _host_of(url: str) -> str:
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def lookup(
    model: str, upstream: str = "", endpoint: str = "", adapter: str = "openai"
) -> dict:
    """Placement for one model call.

    `model` is the name the runtime asked for (a gateway alias or model ID),
    `upstream` the api base LiteLLM reported in `x-litellm-model-api-base`
    (when it did), `endpoint` the provider endpoint the runtime called.
    """
    if adapter == "mock":
        return {
            "engine": "mock",
            "engine_label": ENGINE_LABELS["mock"],
            "host": "runner",
            "device": "none",
            "model_id": model or "demo",
            "registered": True,
        }
    registry = load()
    models = registry.get("models", {})
    name = registry.get("aliases", {}).get(model, model)
    record = None
    if upstream:
        # The engine the gateway actually routed to comes first: with replicas of one model behind an alias, the
        # upstream api base is the only thing that tells them (and their GPUs) apart.
        record = next(
            (
                r
                for r in models.values()
                if r.get("api_base")
                and r["api_base"].rstrip("/") == upstream.rstrip("/")
            ),
            None,
        )
        if record is None:
            host = _host_of(upstream)
            record = next(
                (
                    r
                    for r in models.values()
                    if r.get("api_base") and _host_of(r["api_base"]) == host
                ),
                None,
            )
    if record is None:
        record = models.get(name) or models.get(model)
    if record is not None:
        placement = {
            "engine": record.get("engine", "unregistered"),
            "host": record.get("host", registry.get("host", "")),
            "device": record.get("device", ""),
            "device_detail": record.get("device_detail", ""),
            "gpus": list(record.get("gpus", [])),
            "model_id": record.get("model_id", name),
            "container": record.get("container", ""),
            "api_base": record.get("api_base", ""),
            "alias": model if name != model else "",
            "registered": True,
        }
    else:
        host = _host_of(upstream or endpoint)
        placement = {
            "engine": "unregistered",
            "host": host,
            "device": "",
            "device_detail": "",
            "gpus": [],
            "model_id": name,
            "container": "",
            "api_base": upstream,
            "alias": "",
            "registered": False,
        }
    placement["engine_label"] = ENGINE_LABELS.get(
        placement["engine"], placement["engine"]
    )
    placement["gateway"] = bool(upstream) or (
        _host_of(endpoint) == _host_of(registry.get("gateway", "")) and bool(endpoint)
    )
    return placement


def engines() -> list[dict]:
    """Every registered engine once, for the system panel's engine table."""
    seen, out = set(), []
    for name, record in load().get("models", {}).items():
        key = record.get("container") or record.get("api_base") or name
        if key in seen:
            continue
        seen.add(key)
        out.append({"model": record.get("model_id", name), **record})
    return out
