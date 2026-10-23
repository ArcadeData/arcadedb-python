"""No text describes the monorepo layout this package moved out of (#6).

The version comes from the engine jar (scripts/extract_version.py), not a
Maven pom, and there is no repo-root uv project wrapping this package: tests
run against the wheel installed into a virtual environment (README, Testing).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STALE = re.compile(r"Maven pom|from the pom|repo-root|docs/development/", re.IGNORECASE)
# Text in double quotes is a quotation (a docstring citing what the code used
# to say), not a claim about the layout.
QUOTED = re.compile(r'"[^"]*"')


def _texts():
    yield from (ROOT / "src").rglob("*.py")
    yield from (
        p for p in (ROOT / "tests").rglob("*.py") if p.name != Path(__file__).name
    )
    yield from (ROOT / "scripts").glob("*")
    yield from (
        ROOT / name for name in ("pyproject.toml", "README.md", "tests/README.md")
    )


def test_no_monorepo_wording():
    hits = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in _texts()
        if path.is_file()
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        )
        if STALE.search(QUOTED.sub("", line))
    ]
    assert hits == []
