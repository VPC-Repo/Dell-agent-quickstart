import json
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler

from . import placement as placement_registry


def gateway_headers(metadata: dict) -> dict:
    """Lower-cased x-litellm-* headers from a message's response metadata."""
    headers = metadata.get("headers") or {}
    return {
        str(k).lower(): v
        for k, v in headers.items()
        if str(k).lower().startswith("x-litellm-")
    }


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RunObserver(BaseCallbackHandler):
    raise_error = True

    def __init__(self, emit, provider, budget=0, prior_tokens=0, capture_trace=False):
        self.emit, self.provider, self.budget = emit, provider, budget
        self.capture_trace = capture_trace
        self.tokens = prior_tokens
        self.started = {}
        self.first_token = {}
        self.requested_at = {}
        self.lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self.lock:
            if self.budget and self.tokens >= self.budget:
                raise ValueError("Token budget reached before the next model request")
            self.emit(
                "model_request",
                {
                    "request_id": str(run_id),
                    "provider_id": self.provider.id,
                    "model": self.provider.model,
                    "placement": placement_registry.lookup(
                        self.provider.model,
                        endpoint=self.provider.endpoint,
                        adapter=self.provider.adapter,
                    ),
                    "timestamp": time.time(),
                },
            )
            self.started[str(run_id)] = time.monotonic()
            self.requested_at[str(run_id)] = time.time()
            if self.capture_trace:
                self.emit(
                    "model_input",
                    {
                        "request_id": str(run_id),
                        "messages": [
                            [
                                {"role": m.type, "content": str(m.content)[:6000]}
                                for m in batch
                            ]
                            for batch in messages
                        ],
                    },
                )

    def on_llm_new_token(self, token, *, run_id, chunk=None, **kwargs):
        # Empty usage/role chunks are not a first model response. Tool argument
        # and reasoning chunks count, since these also make forward progress.
        message = getattr(chunk, "message", None)
        meaningful = (
            token
            or getattr(message, "tool_call_chunks", None)
            or (getattr(message, "additional_kwargs", {}) or {}).get(
                "reasoning_content"
            )
        )
        if meaningful:
            with self.lock:
                self.first_token.setdefault(str(run_id), time.monotonic())

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self.lock:
            # A callback-side budget check after a completed response is not an API error.
            if str(run_id) not in self.started:
                return
            now = time.monotonic()
            began = self.started.pop(str(run_id), now)
            first = self.first_token.pop(str(run_id), None)
            status = getattr(error, "status_code", None)
            self.emit(
                "model_error",
                {
                    "request_id": str(run_id),
                    "provider_id": self.provider.id,
                    "model": self.provider.model,
                    "timestamp": time.time(),
                    "started_at": self.requested_at.pop(str(run_id), None),
                    "latency_ms": max(0, now - began) * 1000,
                    "ttft_ms": (first - began) * 1000 if first is not None else None,
                    "error_kind": "rate_limited"
                    if status == 429
                    else "timeout"
                    if getattr(error, "timeout", False)
                    or "timeout" in type(error).__name__.lower()
                    else "model_error",
                },
            )

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self.lock:
            msg = response.generations[0][0].message
            if self.capture_trace:
                self.emit(
                    "model_output",
                    {
                        "request_id": str(run_id),
                        "content": str(msg.content)[:6000],
                        "tool_calls": json.dumps(
                            getattr(msg, "tool_calls", []), default=str
                        )[:6000],
                    },
                )
            usage = getattr(msg, "usage_metadata", None)
            observed = usage is not None
            total = usage.get("total_tokens", 0) if observed else None
            if observed:
                self.tokens += total
            now = time.monotonic()
            began = self.started.pop(str(run_id), now)
            first = self.first_token.pop(str(run_id), None)
            ttft = (first - began) * 1000 if first is not None else None
            output_tokens = usage.get("output_tokens") if observed else None
            decode = (
                (output_tokens - 1) / (now - first)
                if first is not None
                and now > first
                and output_tokens
                and output_tokens > 1
                else None
            )
            metadata = msg.response_metadata or {}
            headers = gateway_headers(metadata)
            upstream = headers.get("x-litellm-model-api-base", "") or ""
            placement = placement_registry.lookup(
                self.provider.model,
                upstream=upstream,
                endpoint=self.provider.endpoint,
                adapter=self.provider.adapter,
            )
            self.emit(
                "model",
                {
                    "finish_reason": metadata.get("finish_reason"),
                    "provider_id": self.provider.id,
                    "model": self.provider.model or "demo",
                    # The model name the engine reported, when it did.
                    "served_model": metadata.get("model_name")
                    or metadata.get("model")
                    or None,
                    "endpoint": self.provider.endpoint or "local fixture",
                    "upstream": upstream or None,
                    "placement": placement,
                    "request_id": str(run_id),
                    "gateway_call_id": metadata.get("gateway_call_id")
                    or headers.get("x-litellm-call-id"),
                    "gateway_model_id": headers.get("x-litellm-model-id"),
                    "gateway_llm_ms": _float(
                        headers.get("x-litellm-timing-llm-api-ms")
                    ),
                    "gateway_overhead_ms": _float(
                        headers.get("x-litellm-overhead-duration-ms")
                    ),
                    "timestamp": time.time(),
                    "started_at": self.requested_at.pop(str(run_id), None),
                    "latency_ms": round(max(0, now - began) * 1000, 2),
                    "ttft_ms": ttft,
                    "decode_tokens_per_second": decode,
                    "timing_source": "client",
                    "input_tokens": usage.get("input_tokens") if observed else None,
                    "output_tokens": output_tokens,
                    "total_tokens": total,
                    "usage_available": observed,
                },
            )
            if self.budget and not observed:
                raise ValueError(
                    "Token budget requires usage reporting from this provider"
                )
            if self.budget and self.tokens > self.budget:
                raise ValueError(
                    "Token budget exceeded by the completed model response"
                )

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.emit(
            "tool_started",
            {"request_id": str(run_id), "name": (serialized or {}).get("name", "Tool")},
        )
        if self.capture_trace:
            self.emit(
                "tool_input",
                {
                    "request_id": str(run_id),
                    "name": (serialized or {}).get("name", "Tool"),
                    "input": str(input_str)[:6000],
                },
            )

    def on_tool_end(self, output, *, run_id, **kwargs):
        payload = {"request_id": str(run_id)}
        if getattr(output, "status", None) == "error":
            self.emit(
                "tool_error",
                {
                    **payload,
                    "message": "Tool could not complete; the worker may correct its request",
                    "detail": str(getattr(output, "content", ""))[:300],
                },
            )
            return
        try:
            data = json.loads(getattr(output, "content", output))
            evidence = data.get("work_evidence") if isinstance(data, dict) else None
            if evidence:
                payload["work_evidence"] = evidence
            result = data.get("result", {}) if isinstance(data, dict) else {}
            if isinstance(result, dict) and result.get("saved_to_demo_outbox"):
                payload["delivery"] = {
                    key: result.get(key)
                    for key in (
                        "saved_to_demo_outbox",
                        "delivered_externally",
                        "connection_id",
                        "id",
                        "character_count",
                    )
                }
        except (TypeError, ValueError):
            pass
        self.emit("tool_completed", payload)
        if self.capture_trace:
            self.emit(
                "tool_output",
                {
                    "request_id": str(run_id),
                    "output": str(getattr(output, "content", output))[:6000],
                },
            )

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.emit("tool_error", {"request_id": str(run_id), "message": "Tool failed"})
