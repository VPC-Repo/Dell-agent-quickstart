"""Point the runner at a model gateway and install the example agents.

The installer runs this once inside the api container after the toolkit's
LiteLLM is up:

    python -m backend.bootstrap --gateway http://agentic-litellm:4000/v1 \
        --key-env GATEWAY_KEY --model general \
        --role coder=coder --role reasoning=reasoning --install-catalog

It is idempotent: the provider document is created or updated in place, and
the catalog install skips agents that already exist. Nothing here starts a run.
"""

from __future__ import annotations

import argparse
import time

from xeon_runtime.specs import Provider

from . import store

DEFAULT_PROVIDER = "intel-toolkit"


def save_provider(
    provider_id=DEFAULT_PROVIDER,
    name="Local toolkit gateway",
    gateway="http://agentic-litellm:4000/v1",
    key_env="GATEWAY_KEY",
    model="general",
    roles=None,
    timeout_seconds=300,
):
    profile = Provider(
        id=provider_id,
        name=name,
        adapter="openai",
        endpoint=gateway,
        model=model,
        api_key_env=key_env,
        roles=dict(roles or {}),
        timeout_seconds=timeout_seconds,
        supports_tools=True,
    )
    with store.Session.begin() as db:
        doc = db.get(store.Document, "provider:" + provider_id)
        if doc:
            doc.payload, doc.updated = profile.model_dump(), time.time()
        else:
            db.add(
                store.Document(
                    key="provider:" + provider_id,
                    kind="provider",
                    payload=profile.model_dump(),
                )
            )
    return profile.model_dump()


def parse_roles(pairs):
    roles = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--role expects role=model, got {pair!r}")
        role, model = pair.split("=", 1)
        roles[role.strip()] = model.strip()
    return roles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider-id", default=DEFAULT_PROVIDER)
    parser.add_argument("--name", default="Local toolkit gateway")
    parser.add_argument("--gateway", default="http://agentic-litellm:4000/v1")
    parser.add_argument("--key-env", default="GATEWAY_KEY")
    parser.add_argument("--model", default="general", help="default model or alias")
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="role=model, for example coder=coder (repeatable)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--install-catalog", action="store_true")
    args = parser.parse_args(argv)
    store.initialize()
    profile = save_provider(
        args.provider_id,
        args.name,
        args.gateway,
        args.key_env,
        args.model,
        parse_roles(args.role),
        args.timeout_seconds,
    )
    print(
        f"provider {profile['id']}: {profile['endpoint']} model={profile['model']} "
        f"roles={profile['roles']}"
    )
    if args.install_catalog:
        from .demo_catalog import install

        ids = install(
            args.provider_id, archive_legacy=True
        )  # replace older generations
        print(f"catalog: {len(ids)} agents available")


if __name__ == "__main__":
    main()
