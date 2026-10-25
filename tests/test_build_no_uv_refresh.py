"""A local build does not run ``uv lock`` or ``uv sync``.

scripts/build.sh used to refresh "the repo-root uv dev environment" after a
build, for the monorepo layout where a separate uv project wrapped this
package. Here the repository root is the package: its pyproject says
``[tool.uv] managed = false``, and ``uv lock`` refuses it. The block was guarded
by a string comparison of a physical and a logical path, so a checkout reached
through a symlink ran it and failed the build (#5); with a correct comparison it
never runs at all, so it is gone. Tests run against the wheel installed into a
virtual environment instead (README, Testing).
"""

from __future__ import annotations

import re
from pathlib import Path

BUILD_SCRIPTS = sorted(
    (Path(__file__).resolve().parents[1] / "scripts").glob("build*.sh")
)

UV_PROJECT_COMMAND = re.compile(r"^\s*[^#\n]*\buv\s+(lock|sync)\b", re.MULTILINE)


def test_build_scripts_do_not_lock_or_sync_a_uv_project():
    offenders = [
        f"{script.name}: {match.group(0).strip()}"
        for script in BUILD_SCRIPTS
        for match in UV_PROJECT_COMMAND.finditer(script.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_the_scan_sees_the_old_refresh():
    old = (
        "        (\n"
        '            cd "$REPO_ROOT" &&\n'
        "                uv lock --upgrade-package arcadedb &&\n"
        "                uv sync --reinstall-package arcadedb\n"
        "        )\n"
        "    # uv lock in a comment is not a command\n"
    )
    assert [m.group(1) for m in UV_PROJECT_COMMAND.finditer(old)] == ["lock", "sync"]
