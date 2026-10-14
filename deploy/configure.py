"""Create or update the runner's .env without printing or clobbering secrets.

    python3 deploy/configure.py                 # create .env with fresh credentials
    python3 deploy/configure.py --set WAREHOUSE_URL=postgresql://... --set REDIS_URL=...
    WAREHOUSE_URL=... python3 deploy/configure.py --set-from-env WAREHOUSE_URL --unset OLD_KEY

The file is created with owner-only permissions. Existing keys are kept unless
--set, --set-from-env or --unset names them; BUILDER_API_TOKEN and POSTGRES_PASSWORD
are generated once. Pass secrets with --set-from-env, which keeps them off the
command line.
"""

import argparse
import os
import re
import secrets
from pathlib import Path

root = Path(__file__).resolve().parent.parent
# QUICKSTART_ENV_FILE points the tests at a scratch file.
path = Path(os.environ.get("QUICKSTART_ENV_FILE") or root / ".env")


def read():
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--set-from-env",
        action="append",
        default=[],
        metavar="KEY",
        help="take the value of KEY from the environment",
    )
    parser.add_argument("--unset", action="append", default=[], metavar="KEY")
    args = parser.parse_args()
    values = read()
    created = not path.exists()
    for key, size in (("BUILDER_API_TOKEN", 32), ("POSTGRES_PASSWORD", 24)):
        if not values.get(key, "").strip():
            values[key] = secrets.token_hex(size)
    values.setdefault("BUILDER_PORT", "8766")
    # No sign-in by default: the listener is loopback-only and reached over SSH.
    values.setdefault("BUILDER_AUTH", "off")
    values.setdefault("LANGSMITH_TRACING", "false")
    pairs = list(args.set)
    for name in args.set_from_env:
        if name not in os.environ:
            raise SystemExit(f"--set-from-env {name}: not set in the environment")
        pairs.append(f"{name}={os.environ[name]}")
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or "\n" in value
            or "\r" in value
        ):
            raise SystemExit(f"Bad key or value for {key!r}")
        values[key.strip()] = value
    for name in args.unset:
        values.pop(name, None)
    if values.get("BUILDER_BIND", "127.0.0.1") not in ("127.0.0.1", "::1", "localhost"):
        if (
            values.get("BUILDER_AUTH", "").lower() != "token"
            or not values.get("BUILDER_API_TOKEN", "").strip()
        ):
            raise SystemExit(
                "Binding beyond loopback requires BUILDER_AUTH=token and BUILDER_API_TOKEN"
            )
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("".join(f"{k}={v}\n" for k, v in values.items()))
    os.chmod(path, 0o600)
    print(("Created" if created else "Updated") + " .env with owner-only permissions.")


if __name__ == "__main__":
    main()
