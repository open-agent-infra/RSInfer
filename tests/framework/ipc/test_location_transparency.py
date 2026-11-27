"""ipc.md §5 test 5 / P7: framework code never hard-codes a local address or uses a PID as one."""

from __future__ import annotations

import re
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[3] / "engine"
FORBIDDEN = re.compile(r"localhost|127\.0\.0\.1|0\.0\.0\.0|::1|getpid\(|ipc://")


def test_no_local_addresses_in_engine_code() -> None:
    hits = [
        f"{p.relative_to(ENGINE)}:{i}: {line.strip()}"
        for p in sorted(ENGINE.rglob("*.py"))
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if FORBIDDEN.search(line)
        # Only the build session leader may signal its compilers. This PID check is local ownership.
        and not (p == ENGINE / "core/change/build.py" and line.strip() == "owns_session = os.getpid() == sid")
    ]
    assert hits == []
