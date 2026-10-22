import pytest
from langchain_core.messages import AIMessage, ToolMessage

from backend import seeds
from xeon_runtime.engine import _stream_agent
from xeon_runtime.specs import Blueprint


def test_required_tool_action_must_complete_including_resumed_results():
    bp = Blueprint.model_validate(seeds.agent("writer", "Writer", "", "Write"))

    class Graph:
        def stream(self, *args, **kwargs):
            yield {"messages": [AIMessage(content="Pretended to send")]}

    with pytest.raises(ValueError, match="required tool actions"):
        _stream_agent(Graph(), bp, [], {}, lambda *a: None, False, ["send_email"])

    class Completed:
        def stream(self, *args, **kwargs):
            yield {
                "messages": [
                    ToolMessage(content="Sent", name="send_email", tool_call_id="1"),
                    AIMessage(content="Finished"),
                ]
            }

    assert (
        _stream_agent(Completed(), bp, [], {}, lambda *a: None, False, ["send_email"])
        == "Finished"
    )


def test_step_out_of_tool_steps_answers_from_what_it_has():
    """Hitting the recursion limit asks once for an answer without tools instead of failing the run."""
    from langgraph.errors import GraphRecursionError

    bp = Blueprint.model_validate(seeds.agent("writer", "Writer", "", "Write"))
    calls = []

    class Graph:
        checkpointer = None

        def stream(self, inp, cfg, **kwargs):
            calls.append(cfg)
            if len(calls) == 1:
                yield {
                    "messages": [
                        ToolMessage(
                            content="10 rows", name="query_warehouse", tool_call_id="1"
                        )
                    ]
                }
                raise GraphRecursionError("limit")
            yield {
                "messages": [
                    *inp["messages"],
                    AIMessage(content="Ten accounts are at risk"),
                ]
            }

    events = []
    out = _stream_agent(
        Graph(),
        bp,
        [{"role": "user", "content": "Churn"}],
        {"recursion_limit": 20},
        lambda k, p: events.append((k, p)),
        False,
        ["query_warehouse"],
    )
    assert out == "Ten accounts are at risk" and calls[1]["recursion_limit"] == 4
    assert [k for k, _ in events] == [
        "tool_error",
        "validated",
    ] and "tool steps" in events[0][1]["message"]

    class Hopeless(Graph):
        def stream(self, inp, cfg, **kwargs):
            yield {"messages": [AIMessage(content="still looking")]}
            raise GraphRecursionError("limit")

    with pytest.raises(GraphRecursionError):
        _stream_agent(Hopeless(), bp, [], {}, lambda *a: None, False, [])


def test_mid_stream_api_error_is_retried_once_from_the_last_good_state():
    """vLLM failed mid-response once; the step retries from the messages it already has, then gives up."""
    import httpx
    from openai import APIError, APIStatusError

    bp = Blueprint.model_validate(seeds.agent("writer", "Writer", "", "Write"))
    calls = []

    class Flaky:
        checkpointer = None

        def stream(self, inp, cfg, **kwargs):
            calls.append(inp)
            if len(calls) == 1:
                yield {
                    "messages": [
                        ToolMessage(
                            content="hits", name="semantic_search", tool_call_id="1"
                        )
                    ]
                }
                raise APIError(
                    "unexpected tokens remaining in message header",
                    request=httpx.Request("POST", "http://gw"),
                    body=None,
                )
            yield {
                "messages": [*inp["messages"], AIMessage(content="Recovered answer")]
            }

    out = _stream_agent(
        Flaky(), bp, [{"role": "user", "content": "Go"}], {}, lambda *a: None, False, []
    )
    assert (
        out == "Recovered answer"
        and len(calls) == 2
        and calls[1]["messages"][0].name == "semantic_search"
    )

    class Broken(Flaky):
        def stream(self, inp, cfg, **kwargs):
            calls.append(inp)
            yield {"messages": [AIMessage(content="partial")]}
            raise APIError(
                "still broken", request=httpx.Request("POST", "http://gw"), body=None
            )

    with pytest.raises(APIError):
        _stream_agent(Broken(), bp, [], {}, lambda *a: None, False, [])

    class BadRequest(Flaky):
        def stream(self, inp, cfg, **kwargs):
            yield {"messages": [AIMessage(content="partial")]}
            raise APIStatusError(
                "bad request",
                response=httpx.Response(
                    400, request=httpx.Request("POST", "http://gw")
                ),
                body=None,
            )

    with pytest.raises(APIStatusError):  # the request's own fault: no retry
        _stream_agent(BadRequest(), bp, [], {}, lambda *a: None, False, [])

    class StaleRoute(Flaky):
        # The gateway answered "model does not exist" because a restarted engine's name briefly pointed at another
        # container; one retry after a pause gets through.
        def stream(self, inp, cfg, **kwargs):
            calls.append(inp)
            if len(calls) == 1:
                yield {"messages": [AIMessage(content="partial")]}
                raise APIStatusError(
                    "model does not exist",
                    response=httpx.Response(
                        404, request=httpx.Request("POST", "http://gw")
                    ),
                    body=None,
                )
            yield {"messages": [*inp["messages"], AIMessage(content="Routed answer")]}

    calls.clear()
    monkeypatch_sleep = __import__("xeon_runtime.engine", fromlist=["time"]).time
    real_sleep, monkeypatch_sleep.sleep = monkeypatch_sleep.sleep, lambda s: None
    try:
        assert (
            _stream_agent(
                StaleRoute(),
                bp,
                [{"role": "user", "content": "Go"}],
                {},
                lambda *a: None,
                False,
                [],
            )
            == "Routed answer"
            and len(calls) == 2
        )
    finally:
        monkeypatch_sleep.sleep = real_sleep


def test_step_that_skips_a_required_tool_is_told_to_continue_with_it():
    """A model that answers without the tool its step needs gets one correction and goes on; the slip is an error event."""
    bp = Blueprint.model_validate(seeds.agent("writer", "Writer", "", "Write"))
    calls = []

    class Graph:
        checkpointer = None

        def stream(self, inp, *args, **kwargs):
            calls.append(inp)
            if len(calls) == 1:
                yield {"messages": [AIMessage(content="No incident found")]}
            else:
                yield {
                    "messages": [
                        *inp["messages"],
                        ToolMessage(
                            content="rows", name="query_warehouse", tool_call_id="1"
                        ),
                        AIMessage(content="Incident at 09:00"),
                    ]
                }

    events = []
    out = _stream_agent(
        Graph(),
        bp,
        [{"role": "user", "content": "Find it"}],
        {},
        lambda k, p: events.append((k, p)),
        False,
        ["query_warehouse"],
    )
    assert out == "Incident at 09:00"
    assert len(calls) == 2 and "query_warehouse" in calls[1]["messages"][-1]["content"]
    assert [k for k, _ in events if k == "tool_error"] == ["tool_error"]
    assert "query_warehouse" in events[0][1]["message"]
