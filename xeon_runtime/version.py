"""Content identity of the portable runtime and its locked dependencies."""

import hashlib
from pathlib import Path

VERSION = "0.8.0"


def runtime_digest():
    root = Path(__file__).parent
    files = sorted(p for p in root.iterdir() if p.suffix in (".py", ".json"))
    lock = root.parent / "requirements.lock"
    if not lock.exists():
        lock = root.parent / "requirements.txt"
    value = hashlib.sha256()
    for path in [*files, lock]:
        value.update((path.name if path != lock else "requirements").encode())
        value.update(path.read_bytes())
    return value.hexdigest()
