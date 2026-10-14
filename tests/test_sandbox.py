"""Agent code must never move from Docker onto the host implicitly."""

import httpx
import pytest

from xeon_runtime import sandbox


def test_default_and_invalid_modes_never_execute_on_the_host(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.setenv("DOCKER_SOCKET", str(tmp_path / "missing.sock"))

    def host_execution(*args, **kwargs):
        pytest.fail("Agent code reached the host process runner")

    monkeypatch.setattr(sandbox, "run_process", host_execution)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run_python("print('must not execute')")
    monkeypatch.setenv("SANDBOX_MODE", "auto")
    with pytest.raises(sandbox.SandboxUnavailable, match="must be docker or process"):
        sandbox.run_python("print('must not execute')")


def test_docker_failure_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("SANDBOX_MODE", "docker")
    monkeypatch.setattr(
        sandbox,
        "run_docker",
        lambda *a: (_ for _ in ()).throw(httpx.ConnectError("down")),
    )
    monkeypatch.setattr(sandbox, "run_process", lambda *a: pytest.fail("Host fallback"))
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run_python("print('must not execute')")


@pytest.mark.parametrize(
    "name", ["../escape.py", "/tmp/escape.py", "a/../../escape.py", ""]
)
def test_files_cannot_escape_the_sandbox_workspace(monkeypatch, name):
    monkeypatch.setenv("SANDBOX_MODE", "process")
    with pytest.raises(ValueError, match="relative paths"):
        sandbox.run_python("print('must not execute')", {name: b"content"})
