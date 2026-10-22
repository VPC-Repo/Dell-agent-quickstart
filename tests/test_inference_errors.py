import httpx
import pytest
from langchain_core.messages import HumanMessage
from xeon_runtime.providers import InferenceUnavailable, create_model
from xeon_runtime.specs import Provider


def test_connection_failure_is_actionable_and_does_not_leak_request(monkeypatch):
    def failed(*args, **kwargs):
        raise httpx.ConnectError("secret-host-and-token-do-not-display")

    monkeypatch.setattr(httpx.Client, "post", failed)
    profile = Provider(
        id="test",
        name="Test",
        adapter="http_json",
        endpoint="http://localhost:8000/v1/chat/completions",
        model="test",
        streaming=False,
    )
    with pytest.raises(
        InferenceUnavailable, match="Cannot reach the model service"
    ) as caught:
        create_model(profile).invoke([HumanMessage(content="Hello")])
    assert "secret-host" not in str(caught.value)
