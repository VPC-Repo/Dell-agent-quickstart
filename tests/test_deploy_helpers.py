"""The installer's host-side helpers: LiteLLM registration, placement, GPU compose."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"


def run(script, *args, env=None):
    return subprocess.run(
        [sys.executable, str(DEPLOY / script), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=60,
    )


def test_litellm_registration_is_idempotent_and_replaces_changed_upstreams():
    state = {"models": []}

    class LiteLLM(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer sk-master":
                return self._send(401, {"error": "bad key"})
            self._send(200, {"data": state["models"]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/model/new":
                state["models"].append(
                    {
                        **body,
                        "model_info": {
                            "id": f"id-{len(state['models'])}",
                            **body.get("model_info", {}),
                        },
                    }
                )
                return self._send(200, {"ok": True})
            if self.path == "/model/delete":
                state["models"] = [
                    m for m in state["models"] if m["model_info"]["id"] != body["id"]
                ]
                return self._send(200, {"ok": True})
            self._send(404, {})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), LiteLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        env = {"LITELLM_MASTER_KEY": "sk-master", "REMOTE_API_KEY": "sk-remote"}
        first = run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--name",
            "general",
            "--upstream",
            "openai/Qwen/Q",
            "--api-base",
            "http://cpu:8000/v1",
            env=env,
        )
        assert first.returncode == 0 and "registered" in first.stdout, first.stderr
        again = run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--name",
            "general",
            "--upstream",
            "openai/Qwen/Q",
            "--api-base",
            "http://cpu:8000/v1",
            env=env,
        )
        assert "already registered" in again.stdout and len(state["models"]) == 1
        moved = run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--name",
            "general",
            "--upstream",
            "openai/gpt-oss-120b",
            "--api-base",
            "http://gpu:8000/v1",
            env=env,
        )
        assert "replaced" in moved.stdout and len(state["models"]) == 1
        assert state["models"][0]["litellm_params"]["api_base"] == "http://gpu:8000/v1"
        replica = run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--keep-others",
            "--name",
            "general",
            "--upstream",
            "openai/gpt-oss-120b",
            "--api-base",
            "http://gpu-2:8000/v1",
            env=env,
        )
        assert (
            replica.returncode == 0
            and "registered" in replica.stdout
            and len(state["models"]) == 2
        )  # balanced alongside, not replaced
        same = run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--keep-others",
            "--name",
            "general",
            "--upstream",
            "openai/gpt-oss-120b",
            "--api-base",
            "http://gpu-2:8000/v1",
            env=env,
        )
        assert "already registered" in same.stdout and len(state["models"]) == 2
        # An install without free GPUs points the role at the CPU model; a later run with GPUs re-adds its first
        # engine without --keep-others, which must clear that fallback (and stale replicas) even though the engine
        # itself is already registered.
        run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--keep-others",
            "--name",
            "general",
            "--upstream",
            "openai/Qwen/Q",
            "--api-base",
            "http://cpu:8000/v1",
            env=env,
        )
        assert len(state["models"]) == 3
        reset = run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--name",
            "general",
            "--upstream",
            "openai/gpt-oss-120b",
            "--api-base",
            "http://gpu:8000/v1",
            env=env,
        )
        assert "already registered" in reset.stdout and [
            m["litellm_params"]["api_base"] for m in state["models"]
        ] == ["http://gpu:8000/v1"]
        run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--keep-others",
            "--name",
            "general",
            "--upstream",
            "openai/gpt-oss-120b",
            "--api-base",
            "http://gpu-2:8000/v1",
            env=env,
        )
        request_remove = run(
            "litellm_models.py", "--base", base, "remove", "--name", "general", env=env
        )
        assert request_remove.returncode == 0
        remote = run(
            "litellm_models.py",
            "--base",
            base,
            "add",
            "--name",
            "remote/x",
            "--upstream",
            "openai/x",
            "--api-base",
            "https://api.example/v1",
            "--api-key-env",
            "REMOTE_API_KEY",
            env=env,
        )
        assert (
            remote.returncode == 0
            and state["models"][1]["litellm_params"]["api_key"] == "sk-remote"
        )
        listed = run("litellm_models.py", "--base", base, "list", env=env)
        assert "general\topenai/gpt-oss-120b" in listed.stdout
        missing = run(
            "litellm_models.py", "--base", base, "list", env={"LITELLM_MASTER_KEY": ""}
        )
        assert missing.returncode != 0
        wrong = run(
            "litellm_models.py",
            "--base",
            base,
            "list",
            env={"LITELLM_MASTER_KEY": "nope"},
        )
        assert wrong.returncode != 0 and "HTTP 401" in wrong.stderr
    finally:
        server.shutdown()


def test_client_key_is_reused_while_valid_and_replaced_when_not():
    state = {"keys": {}, "n": 0}

    class LiteLLM(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # /v1/models with the client key itself
            token = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
            self._send(200, {"data": []}) if token in state["keys"] else self._send(
                401, {"error": "bad key"}
            )

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer sk-master":
                return self._send(401, {"error": "bad key"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/key/generate":
                state["n"] += 1
                key = f"sk-client-{state['n']}"
                state["keys"][key] = body["key_alias"]
                return self._send(200, {"key": key})
            if self.path == "/key/delete":
                state["keys"] = {
                    k: a
                    for k, a in state["keys"].items()
                    if a not in body["key_aliases"]
                }
                return self._send(200, {"deleted_keys": body["key_aliases"]})
            self._send(404, {})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), LiteLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        args = (
            "litellm_models.py",
            "--base",
            base,
            "client-key",
            "--alias",
            "runner",
            "--current-env",
            "GATEWAY_KEY",
        )
        first = run(*args, env={"LITELLM_MASTER_KEY": "sk-master", "GATEWAY_KEY": ""})
        assert first.returncode == 0 and first.stdout.strip() == "sk-client-1", (
            first.stderr
        )
        again = run(
            *args, env={"LITELLM_MASTER_KEY": "sk-master", "GATEWAY_KEY": "sk-client-1"}
        )
        assert (
            again.stdout.strip() == "sk-client-1" and state["n"] == 1
        )  # still valid: reused, nothing generated
        stale = run(
            *args, env={"LITELLM_MASTER_KEY": "sk-master", "GATEWAY_KEY": "sk-unknown"}
        )
        assert stale.stdout.strip() == "sk-client-2" and list(state["keys"]) == [
            "sk-client-2"
        ]  # the alias keeps one key
    finally:
        server.shutdown()


def test_configure_takes_secrets_from_the_environment_and_removes_keys(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LITELLM_MASTER_KEY=sk-master\nKEEP=1\n")
    result = run(
        "configure.py",
        "--set-from-env",
        "GATEWAY_KEY",
        "--unset",
        "LITELLM_MASTER_KEY",
        "--set",
        "HOST_NAME=h",
        env={"QUICKSTART_ENV_FILE": str(env_file), "GATEWAY_KEY": "sk-client"},
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert (
        values["GATEWAY_KEY"] == "sk-client"
        and values["KEEP"] == "1"
        and values["HOST_NAME"] == "h"
    )
    assert (
        "LITELLM_MASTER_KEY" not in values
        and oct(env_file.stat().st_mode & 0o777) == "0o600"
    )
    missing = run(
        "configure.py",
        "--set-from-env",
        "NOT_SET_ANYWHERE",
        env={"QUICKSTART_ENV_FILE": str(env_file)},
    )
    assert missing.returncode != 0


def test_gpu_compose_and_placement_registry(tmp_path):
    out = tmp_path / "gpu.yaml"
    result = run(
        "gpu_compose.py",
        "--out",
        str(out),
        "--assign",
        "Qwen/Qwen3-30B-A3B-Instruct-2507=0",
        "--assign",
        "openai/gpt-oss-120b=1,2",
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert (
        lines[1]
        == "openai/gpt-oss-120b|vllm-cuda|cuda|1,2|vllm-gpu-openai-gpt-oss-120b"
    )
    text = out.read_text()
    assert (
        '"--tensor-parallel-size"\n      - "2"' in text
        and '"--tool-call-parser"\n      - "openai"' in text
    )
    assert 'device_ids: ["1", "2"]' in text and "docker_agentic-net" in text
    assert (
        '"--ignore-patterns"\n      - "original/*"' in text
    )  # no second copy of the weights (gpt-oss-120b's original/ is 61 GB)
    overlap = run(
        "gpu_compose.py", "--out", str(out), "--assign", "a=0", "--assign", "b=0"
    )
    assert overlap.returncode != 0
    # Replicas: the same model on more GPUs gets numbered containers; the first keeps the plain name.
    replicas = run(
        "gpu_compose.py",
        "--out",
        str(tmp_path / "r.yaml"),
        "--assign",
        "openai/gpt-oss-120b=1",
        "--assign",
        "Qwen/Qwen3-30B-A3B-Instruct-2507=0",
        "--assign",
        "openai/gpt-oss-120b=4",
        "--assign",
        "openai/gpt-oss-120b=6",
    )
    assert replicas.returncode == 0, replicas.stderr
    names = [line.split("|")[-1] for line in replicas.stdout.strip().splitlines()]
    assert names == [
        "vllm-gpu-openai-gpt-oss-120b",
        "vllm-gpu-qwen-qwen3-30b-a3b-instruct-2507",
        "vllm-gpu-openai-gpt-oss-120b-2",
        "vllm-gpu-openai-gpt-oss-120b-3",
    ]
    assert (tmp_path / "r.yaml").read_text().count(
        "TIKTOKEN_ENCODINGS_BASE: /root/.cache/huggingface/tiktoken"
    ) == 3  # gpt-oss engines only
    assert (
        (tmp_path / "r.yaml").read_text().count("container_name:") == 4
        and '"127.0.0.1:2186:8000"' not in (tmp_path / "r.yaml").read_text()
        and '"127.0.0.1:2183:8000"' in (tmp_path / "r.yaml").read_text()
    )
    registry = tmp_path / "placement.json"
    placed = run(
        "placement.py",
        "--out",
        str(registry),
        "--host",
        "gpu-host",
        "--model",
        "Qwen/Qwen3-30B-A3B-Instruct-2507|vllm-cpu|cpu||agentic-vllm-qwen",
        *sum((["--model", line] for line in lines), []),
        "--alias",
        "reasoning=openai/gpt-oss-120b",
        "--alias",
        "general=Qwen/Qwen3-30B-A3B-Instruct-2507",
        "--remote",
        "remote/gpt-4o-mini|https://api.openai.com/v1",
    )
    assert placed.returncode == 0, placed.stderr
    data = json.loads(registry.read_text())
    assert data["aliases"]["reasoning"] == "openai/gpt-oss-120b"
    assert (
        data["models"]["openai/gpt-oss-120b"]["gpus"] == [1, 2]
        and data["models"]["openai/gpt-oss-120b"]["metrics_url"]
        == "http://vllm-gpu-openai-gpt-oss-120b:8000/metrics"
    )
    assert (
        data["models"]["remote/gpt-4o-mini"]["engine"] == "remote-api"
        and data["models"]["remote/gpt-4o-mini"]["host"] == "api.openai.com"
    )
    # A replica keeps its own registry entry; the alias still points at the first.
    twice = run(
        "placement.py",
        "--out",
        str(tmp_path / "p2.json"),
        "--host",
        "gpu-host",
        "--model",
        "openai/gpt-oss-120b|vllm-cuda|cuda|1|vllm-gpu-openai-gpt-oss-120b",
        "--model",
        "openai/gpt-oss-120b|vllm-cuda|cuda|4|vllm-gpu-openai-gpt-oss-120b-2",
        "--alias",
        "reasoning=openai/gpt-oss-120b",
    )
    assert twice.returncode == 0, twice.stderr
    two = json.loads((tmp_path / "p2.json").read_text())["models"]
    assert two["openai/gpt-oss-120b"]["gpus"] == [1] and two["openai/gpt-oss-120b#2"][
        "gpus"
    ] == [4]
    assert two["openai/gpt-oss-120b#2"]["model_id"] == "openai/gpt-oss-120b" and two[
        "openai/gpt-oss-120b#2"
    ]["container"].endswith("-2")
    from xeon_runtime import placement

    os.environ["PLACEMENT_REGISTRY"] = str(registry)
    try:
        assert placement.lookup("reasoning")["gpus"] == [1, 2]
    finally:
        del os.environ["PLACEMENT_REGISTRY"]
    bad = run("placement.py", "--out", str(registry), "--alias", "general=missing")
    assert bad.returncode != 0


def test_configure_fills_blank_example_credentials_and_keeps_them_on_rerun(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text((ROOT / ".env.example").read_text())
    env = {"QUICKSTART_ENV_FILE": str(env_file)}
    first = run("configure.py", env=env)
    assert first.returncode == 0, first.stderr
    values = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert len(values["BUILDER_API_TOKEN"]) == 64
    assert len(values["POSTGRES_PASSWORD"]) == 48
    second = run("configure.py", env=env)
    assert second.returncode == 0, second.stderr
    assert (
        dict(line.split("=", 1) for line in env_file.read_text().splitlines()) == values
    )
    assert values["BUILDER_API_TOKEN"] not in first.stdout + first.stderr


def test_configure_rejects_public_binding_without_auth_and_line_injection(tmp_path):
    env_file = tmp_path / ".env"
    env = {"QUICKSTART_ENV_FILE": str(env_file)}
    public = run("configure.py", "--set", "BUILDER_BIND=0.0.0.0", env=env)
    assert public.returncode != 0 and not env_file.exists()
    bad = run("configure.py", "--set", "GATEWAY_KEY=x\rBUILDER_AUTH=off", env=env)
    assert bad.returncode != 0 and not env_file.exists()
    good = run(
        "configure.py",
        "--set",
        "BUILDER_BIND=0.0.0.0",
        "--set",
        "BUILDER_AUTH=token",
        env=env,
    )
    assert good.returncode == 0, good.stderr


def test_gpu_compose_rejects_duplicate_and_negative_device_ids(tmp_path):
    for devices in ("0,0", "-1"):
        out = tmp_path / "invalid.yaml"
        result = run("gpu_compose.py", "--out", str(out), "--assign", f"a={devices}")
        assert result.returncode != 0
        assert "Bad or overlapping GPU assignment" in result.stderr
        assert not out.exists()
