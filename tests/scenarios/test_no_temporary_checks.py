"""Temporary checks written into the engine's code to look inside a running engine are marked
``# PROBE`` (``.claude/skills/rsinfer/SKILL.md``, 「在引擎里做检查」): none may stay in the tree.

The skill keeps them in a copy of the tree (a second workspace applied with ``--source``), so the
repo never has them; this is the net for one written into the repo itself."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MARK = re.compile(r"#\s*PROBE\b")


def test_no_temporary_check_is_left_in_the_engine() -> None:
    found = [f"{path.relative_to(REPO)}:{n}: {line.strip()}"
             for path in sorted((REPO / "engine").rglob("*.py")) if "__pycache__" not in path.parts
             for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if MARK.search(line)]
    assert not found, "temporary checks (# PROBE) left in the engine:\n" + "\n".join(found)
