import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from xeon_runtime.providers import JsonAPIModel, create_model
from xeon_runtime.specs import Provider


def test_qwen_text_tool_round_trip(monkeypatch):
    @tool
    def check_status() -> str:
        """Return the demo status."""
        return "ready"

    requests = []

    def handler(req):
        body = json.loads(req.content)
        requests.append(body)
        content = (
            '<tool_call>{"name":"check_status","arguments":{}}</tool_call>'
            if len(requests) == 1
            else "Ready."
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content, "tool_calls": None}}]},
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    )
    profile = Provider(
        id="qwen",
        name="Qwen",
        adapter="http_json",
        endpoint="http://inference.example/v1/chat/completions",
        tool_call_format="qwen_xml",
    )
    model = create_model(profile).bind_tools([check_status])
    first = model.invoke("Check the status")
    assert first.content == "" and first.tool_calls[0]["name"] == "check_status"
    result = model.invoke(
        [
            HumanMessage(content="Check the status"),
            first,
            ToolMessage(content="ready", tool_call_id=first.tool_calls[0]["id"]),
        ]
    )
    assert result.content == "Ready."
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert (
        requests[1]["messages"][-2]["tool_calls"][0]["function"]["name"]
        == "check_status"
    )


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name":"x"}',
        '<tool_call>{"name":"x","arguments":"wrong"}</tool_call>',
    ],
)
def test_qwen_text_parser_rejects_malformed_calls(monkeypatch, content):
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"text": content})
            ),
            **kw,
        ),
    )
    profile = Provider(
        id="qwen",
        name="Qwen",
        adapter="http_json",
        endpoint="http://inference.example/generate",
        tool_call_format="qwen_xml",
        response_path="text",
    )
    model = JsonAPIModel(
        profile=profile,
        bound_tools=[
            {
                "type": "function",
                "function": {
                    "name": "x",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="Response mapping"):
        model.invoke("Check")
