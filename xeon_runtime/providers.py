"""Endpoint adapters; inference placement is never encoded into agent definitions."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from .specs import Provider


class InferenceUnavailable(ValueError):
    """Sanitized inference transport failure, safe to show to the user."""

    def __init__(self, message, *, status_code=None, timeout=False):
        super().__init__(message)
        self.status_code = status_code
        self.timeout = timeout


def read_chat_stream(response, run_manager=None):
    """Assemble one OpenAI-compatible SSE response, including fragmented tool calls."""
    content, calls, usage, finish = [], {}, None, None
    completed = False
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            completed = True
            break
        event = json.loads(value)
        if event.get("error"):
            raise InferenceUnavailable(
                "The model stream reported an error. Check the model service."
            )
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            if text:
                content.append(text)
            if run_manager and (
                text or delta.get("tool_calls") or delta.get("reasoning_content")
            ):
                run_manager.on_llm_new_token(
                    text or delta.get("reasoning_content") or "[tool delta]"
                )
            for part in delta.get("tool_calls") or []:
                call = calls.setdefault(
                    part["index"], {"id": "", "function": {"name": "", "arguments": ""}}
                )
                if part.get("id"):
                    call["id"] += part["id"]
                for key in ("name", "arguments"):
                    call["function"][key] += (part.get("function") or {}).get(key) or ""
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if not completed and not finish:
        raise InferenceUnavailable(
            "The model stream ended before completing the response."
        )
    return {
        "choices": [
            {
                "message": {
                    "content": "".join(content),
                    "tool_calls": [calls[k] for k in sorted(calls)],
                },
                "finish_reason": finish,
            }
        ],
        "usage": usage,
    }


def read_path(value: Any, path: str, default=None):
    try:
        for part in path.split(".") if path else []:
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def render_template(value, variables):
    if isinstance(value, str) and value in variables:
        return variables[value]
    if isinstance(value, dict):
        return {k: render_template(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, variables) for v in value]
    return value


def env_secret(name):
    if not name:
        return None
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Set the server environment variable {name} before using this provider"
        )
    return value


class JsonAPIModel(BaseChatModel):
    profile: Provider
    bound_tools: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "configured-http-json"

    def bind_tools(self, tools, **kwargs):
        if tools and not self.profile.supports_tools:
            raise ValueError("Provider does not support tool calling")
        return self.model_copy(
            update={"bound_tools": [convert_to_openai_tool(t) for t in tools]}
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        p = self.profile
        headers = {k: env_secret(v) for k, v in p.headers_env.items()}
        if run_manager:
            headers.setdefault("X-Request-ID", str(run_manager.run_id))
        if p.api_key_env:
            headers.setdefault("Authorization", f"Bearer {env_secret(p.api_key_env)}")
        body = render_template(
            p.request_template,
            {
                "$messages": convert_to_openai_messages(messages),
                "$model": p.model,
                "$tools": self.bound_tools,
                "$stop": stop,
                "$prompt": "\n\n".join(
                    str(m.content) for m in messages if m.type != "system"
                ),
                "$system": "\n\n".join(
                    str(m.content) for m in messages if m.type == "system"
                ),
            },
        )
        # Provider response bodies can contain secrets; never echo HTTP errors to the UI.
        try:
            with httpx.Client(
                timeout=p.timeout_seconds, follow_redirects=False
            ) as client:
                streaming = (
                    p.streaming
                    and body.get("stream") is not False
                    and "messages" in body
                    and p.response_path == "choices.0.message.content"
                    and p.tool_calls_path == "choices.0.message.tool_calls"
                    and p.usage_path == "usage"
                    and p.finish_reason_path == "choices.0.finish_reason"
                )
                if streaming:
                    body = {
                        **body,
                        "stream": True,
                        "stream_options": {
                            "include_usage": True,
                            **body.get("stream_options", {}),
                        },
                    }
                    with client.stream(
                        "POST", p.endpoint, headers=headers, json=body
                    ) as response:
                        response.raise_for_status()
                        if "text/event-stream" in response.headers.get(
                            "content-type", ""
                        ):
                            data = read_chat_stream(response, run_manager)
                        else:
                            response.read()
                            data = response.json()
                else:
                    response = client.post(p.endpoint, headers=headers, json=body)
                    response.raise_for_status()
                    data = response.json()
        except httpx.HTTPStatusError as e:
            raise InferenceUnavailable(
                f"The model service returned HTTP {e.response.status_code}. Check the saved model connection and server logs.",
                status_code=e.response.status_code,
            ) from None
        except httpx.TimeoutException:
            raise InferenceUnavailable(
                "The model service timed out. It may still be loading or busy; try again when it is ready.",
                timeout=True,
            ) from None
        except httpx.HTTPError:
            raise InferenceUnavailable(
                "Cannot reach the model service. Start the inference server or update the saved model connection."
            ) from None
        except InferenceUnavailable:
            raise
        except ValueError:
            raise InferenceUnavailable(
                "The model service returned invalid JSON. Check that the saved endpoint is a chat completion API."
            ) from None
        content = read_path(data, p.response_path)
        calls = read_path(data, p.tool_calls_path, []) if p.tool_calls_path else []
        if (
            self.bound_tools
            and p.tool_call_format == "qwen_xml"
            and not calls
            and isinstance(content, str)
        ):
            # Explicit opt-in for Qwen/Hermes endpoints without server-side parsing.
            pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
            matches = list(re.finditer(pattern, content, re.DOTALL))
            remaining = re.sub(pattern, "", content, flags=re.DOTALL).strip()
            if "<tool_call>" in remaining or "</tool_call>" in remaining:
                raise ValueError("Response mapping found an incomplete text tool call")
            calls = []
            for match in matches:
                try:
                    record = json.loads(match.group(1))
                    if (
                        not isinstance(record, dict)
                        or not isinstance(record.get("name"), str)
                        or not isinstance(record.get("arguments", {}), dict)
                    ):
                        raise ValueError()
                except (ValueError, TypeError):
                    raise ValueError(
                        "Response mapping found an invalid text tool call"
                    ) from None
                calls.append({"id": "call-" + uuid.uuid4().hex, "function": record})
            content = remaining
        tool_calls = []
        for i, call in enumerate(calls or []):
            function = call.get("function", call)
            args = function.get("arguments", function.get("args", {}))
            tool_calls.append(
                {
                    "id": call.get("id", f"call-{i}"),
                    "name": function["name"],
                    "args": json.loads(args) if isinstance(args, str) else args,
                }
            )
        if content is None and not tool_calls:
            raise ValueError(
                "Response mapping did not locate message content or tool calls"
            )
        if content is not None and not isinstance(content, str):
            content = json.dumps(content)
        usage = read_path(data, p.usage_path) if p.usage_path else None
        metadata = None
        if isinstance(usage, dict):
            incoming = read_path(usage, p.input_tokens_field)
            outgoing = read_path(usage, p.output_tokens_field)
            if (
                type(incoming) is int
                and type(outgoing) is int
                and min(incoming, outgoing) >= 0
            ):
                metadata = {
                    "input_tokens": incoming,
                    "output_tokens": outgoing,
                    "total_tokens": incoming + outgoing,
                }
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content or "",
                        tool_calls=tool_calls,
                        usage_metadata=metadata,
                        response_metadata={
                            "headers": {
                                k: v
                                for k, v in response.headers.items()
                                if k.lower().startswith("x-litellm-")
                            },
                            "gateway_call_id": response.headers.get(
                                "x-litellm-call-id"
                            ),
                            "finish_reason": read_path(data, p.finish_reason_path)
                            if p.finish_reason_path
                            else None,
                        },
                    )
                )
            ]
        )


class DemoModel(BaseChatModel):
    """Deterministic fixture model that drives real LangChain/Deep Agents tool loops."""

    bound_tools: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "demo-fixture"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(
            update={"bound_tools": [convert_to_openai_tool(t) for t in tools]}
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        names = [t["function"]["name"] for t in self.bound_tools]
        if not any(isinstance(m, ToolMessage) for m in messages):
            selected = next(
                (n for n in ("search_knowledge", "analyze_metrics") if n in names), None
            )
            if selected:
                args = (
                    {"query": "service deployment briefing"}
                    if selected == "search_knowledge"
                    else {}
                )
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": selected,
                                        "args": args,
                                        "id": "fixture-tool-1",
                                    }
                                ],
                            )
                        )
                    ]
                )
        context = "\n".join(str(m.content) for m in messages)
        if "Return valid JSON" in context:
            content = json.dumps(
                {
                    "summary": "Synthetic daily briefing",
                    "tickets_closed": 84,
                    "backlog_change": -8,
                    "sources": ["brief-001", "brief-002"],
                    "mode": "mock",
                }
            )
        else:
            content = (
                "# Daily briefing\n\n**Demonstration data — not hardware measurements.**\n\n"
                "## Operations\n84 tickets closed; backlog decreased from 46 to 38 (17.4%). "
                "Two firmware cases need engineering follow-up. [brief-001]\n\n"
                "## Deployments\nThree deployments passed functional acceptance. Research source review remains open. "
                "A model endpoint change is proposed for staging only. [brief-002]\n\n"
                "## Next actions\n- Review the two firmware cases.\n- Complete research source review.\n"
                "- Prepare for the Sunday 02:00 UTC maintenance window. [brief-001]\n"
            )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


def resolve_role(profile: Provider, role: str | None) -> Provider:
    """The profile with `model` set to the role's model at this endpoint.

    Roles are how an agent asks for a general, coder or reasoning model without
    naming hardware: the provider's role map (for example LiteLLM aliases)
    decides which engine serves it. Unknown roles use the default model.
    """
    chosen = profile.model_for(role)
    if chosen == profile.model:
        return profile
    return profile.model_copy(update={"model": chosen})


def create_model(profile: Provider):
    if profile.adapter == "mock":
        return DemoModel()
    if profile.adapter == "http_json":
        return JsonAPIModel(profile=profile)
    from langchain_openai import ChatOpenAI

    headers = {k: env_secret(v) for k, v in profile.headers_env.items()}
    return ChatOpenAI(
        model=profile.model,
        base_url=profile.endpoint,
        api_key=env_secret(profile.api_key_env) or "not-required",
        default_headers=headers,
        timeout=profile.timeout_seconds,
        max_retries=0,
        streaming=profile.streaming,
        stream_usage=profile.streaming,
        # Gateways report the upstream engine in response headers
        # (LiteLLM: x-litellm-model-api-base, x-litellm-model-id); the
        # telemetry reads them from response_metadata["headers"].
        include_response_headers=True,
    )
