#!/usr/bin/env python3
"""Register models and role aliases with the toolkit's LiteLLM through its API.

The toolkit's generated LiteLLM config is marked "do not edit", so GPU engines,
remote endpoints and the role aliases (general, coder, reasoning) are added with
POST /model/new, which LiteLLM persists in its database (STORE_MODEL_IN_DB).
Standard library only; runs on the host.

    litellm_models.py --base http://127.0.0.1:4000 --key-env LITELLM_MASTER_KEY \
        add --name general --upstream openai/Qwen/Qwen3-30B-A3B-Instruct-2507 \
            --api-base http://agentic-vllm-qwen3-30b-a3b-instruct-2507:8000/v1
    litellm_models.py ... add --name reasoning --upstream openai/openai/gpt-oss-120b \
            --api-base http://vllm-gpu-openai-gpt-oss-120b:8000/v1
    litellm_models.py ... add --name remote-gpt --upstream openai/gpt-4o-mini \
            --api-base https://api.openai.com/v1 --api-key-env REMOTE_API_KEY
    litellm_models.py ... list
    litellm_models.py ... client-key --alias agent-quickstart-runner --current-env GATEWAY_KEY

`add` is idempotent: an existing entry with the same name and upstream is left
alone; one with the same name and a different upstream is replaced, unless
--keep-others says it is a replica to be balanced alongside the others.

`client-key` prints a gateway key that can call every model but not change the
gateway (no admin rights), for the runner. The key in --current-env is reused
while the gateway accepts it; otherwise keys with the alias are replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def request(base, key, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            text = response.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise SystemExit(
            f"LiteLLM {method} {path} failed: HTTP {exc.code} {detail}"
        ) from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach LiteLLM at {base}: {exc.reason}") from None


def existing(base, key):
    data = request(base, key, "GET", "/model/info")
    rows = data.get("data", data) if isinstance(data, dict) else data
    return [r for r in rows if isinstance(r, dict)]


def add(args, base, key):
    params = {"model": args.upstream, "api_base": args.api_base}
    if args.api_key_env:
        value = os.getenv(args.api_key_env, "")
        if not value:
            raise SystemExit(
                f"Set {args.api_key_env} in the environment before registering a keyed endpoint"
            )
        params["api_key"] = value
    else:
        params["api_key"] = "not-required"
    info = {"mode": args.mode}
    registered = False
    for row in existing(base, key):
        if row.get("model_name") != args.name:
            continue
        current = row.get("litellm_params", {})
        if current.get("model") == args.upstream and current.get("api_base", "").rstrip(
            "/"
        ) == args.api_base.rstrip("/"):
            registered = True
            continue
        if args.keep_others:
            continue  # a replica: the other deployments of this name stay, LiteLLM balances across them
        model_id = (row.get("model_info") or {}).get("id")
        if model_id:
            try:
                request(base, key, "POST", "/model/delete", {"id": model_id})
                print(f"{args.name}: replaced previous entry")
            except (Exception, SystemExit) as exc:  # noqa: BLE001 -- config-file entries cannot be deleted through the API
                print(
                    f"{args.name}: could not remove the previous entry ({str(exc)[:120]}); adding the new one alongside it"
                )
    if registered:
        print(f"{args.name}: already registered -> {args.upstream} @ {args.api_base}")
        return
    body = {"model_name": args.name, "litellm_params": params, "model_info": info}
    request(base, key, "POST", "/model/new", body)
    print(f"{args.name}: registered -> {args.upstream} @ {args.api_base}")


def list_models(args, base, key):
    for row in existing(base, key):
        params = row.get("litellm_params", {})
        print(
            f"{row.get('model_name')}\t{params.get('model')}\t{params.get('api_base', '')}"
        )


def remove(args, base, key):
    for row in existing(base, key):
        if row.get("model_name") == args.name:
            model_id = (row.get("model_info") or {}).get("id")
            if model_id:
                request(base, key, "POST", "/model/delete", {"id": model_id})
                print(f"{args.name}: removed")
                return
    print(f"{args.name}: not found")


def client_key(args, base, key):
    current = os.getenv(args.current_env, "") if args.current_env else ""
    if current:
        try:
            request(
                base, current, "GET", "/v1/models"
            )  # checked with the key itself, so it never appears in a URL
            print(current)
            return
        except SystemExit:
            pass
    try:
        request(base, key, "POST", "/key/delete", {"key_aliases": [args.alias]})
    except SystemExit:
        pass  # no earlier key with this alias
    print(request(base, key, "POST", "/key/generate", {"key_alias": args.alias})["key"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:4000")
    parser.add_argument(
        "--key-env",
        default="LITELLM_MASTER_KEY",
        help="environment variable holding the master key",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("add")
    p.add_argument("--name", required=True, help="public model name or alias")
    p.add_argument(
        "--upstream",
        required=True,
        help="litellm model string, for example openai/<served model>",
    )
    p.add_argument("--api-base", required=True)
    p.add_argument(
        "--api-key-env",
        default="",
        help="host environment variable with the upstream key (remote APIs)",
    )
    p.add_argument("--mode", default="chat")
    p.add_argument(
        "--keep-others",
        action="store_true",
        help="add this deployment next to existing ones with the same name (a replica) instead of replacing them",
    )
    p.set_defaults(fn=add)
    sub.add_parser("list").set_defaults(fn=list_models)
    r = sub.add_parser("remove")
    r.add_argument("--name", required=True)
    r.set_defaults(fn=remove)
    k = sub.add_parser("client-key")
    k.add_argument("--alias", required=True)
    k.add_argument(
        "--current-env",
        default="",
        help="environment variable holding the key to reuse if still valid",
    )
    k.set_defaults(fn=client_key)
    args = parser.parse_args()
    key = os.getenv(args.key_env, "")
    if not key:
        sys.exit(f"Set {args.key_env} (the LiteLLM master key) in the environment")
    args.fn(args, args.base, key)


if __name__ == "__main__":
    main()
