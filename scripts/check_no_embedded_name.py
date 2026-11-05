#!/usr/bin/env python3
"""Fail when the old package name comes back.

The distributions are `arcadedb` (native wheel) and `arcadedb-jvm` (JVM wheel), and both import
as `arcadedb`. The names `arcadedb_embedded` and `arcadedb-embedded` belong to the old PyPI
packages, which get a deprecation shim from the humemai repository, not from this one. This script
scans every tracked (and every untracked, not ignored) text file and every path in the repository
and exits 1 for each place the old name appears outside the allowlist below.

The generic word "embedded" (embedded mode, embedded database) is fine and is not checked. Only the
package names are: `arcadedb_embedded`, `arcadedb-embedded` (this also catches `arcadedb-embedded-native`
and the old repository name `arcadedb-embedded-python`) and the title `ArcadeDB Embedded Python`.

What the allowlist lets through, and why:

* Whole files that are history or that must name the old package to do their job: the plan and spec
  documents under docs/superpowers/, a changelog, and this script.
* References to issues of the old repository, written `humemai/arcadedb-embedded-python#123`. They
  point at real threads, so they keep their text.
* The sentence "This package was published as arcadedb-embedded until 26.10.x; install arcadedb or arcadedb-jvm now."
  and lines that say "formerly arcadedb-embedded".
* The three steps of "Migrate from the old names" (docs/modules/ROOT/pages/which-package.adoc): they must name the old
  packages to tell a reader what to uninstall, what they used and which import to change. Each is allowed by one phrase of its
  step, not by the whole page.
* Any line that carries the marker `allow-old-name`, for the rare mention that needs it.

usage: check_no_embedded_name.py [--root DIR]

Exit 0 when clean, 1 when a violation is found. Output is ASCII only.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess  # nosec B404 - runs git to list files
import sys
from pathlib import Path

# The old names. `arcadedb-embedded` also matches inside `arcadedb-embedded-native` and
# `arcadedb-embedded-python`; those are old names too.
# Also glob/wildcard spellings that only match the old wheel name (`*embed*.whl`, `embedded*`): the rename missed
# `dist/*embed*.whl` in the examples workflow and six examples, and CI failed on it (#42, 2026-10-25).
OLD_NAME = re.compile(
    r"arcadedb[_-]embedded|ArcadeDB Embedded Python|\*embed(?:ded)?\*", re.IGNORECASE
)

# Files (fnmatch patterns on the path relative to the repository root) that may name the old package.
ALLOWED_PATHS: tuple[str, ...] = (
    "docs/superpowers/*",  # plans and specs of the move, history
    "CHANGELOG*",
    "HISTORY*",
    "NEWS*",
    "scripts/check_no_embedded_name.py",
    "tests/test_no_embedded_name.py",  # holds the old patterns as test cases
)

# Text that is removed from a line before it is scanned: issue references of the old repository.
ALLOWED_TOKENS = re.compile(r"humemai/arcadedb-embedded-python#\d+")

# A line that matches is allowed as a whole.
ALLOWED_LINES = re.compile(
    r"allow-old-name"
    r"|This package was published as arcadedb-embedded until 26\.10\.x; install arcadedb now\."
    r"|formerly `?arcadedb[_-]embedded"
    # The "Migrate from the old names" steps must name the old packages (one phrase per step):
    r"|uv remove arcadedb[_-]embedded"  # step 1, what to remove
    r"|if you used `arcadedb[_-]embedded"  # step 2, which new package replaces which old one
    r"|`import arcadedb_embedded` and `import arcadedb_native` both become",  # step 3, the import change
    re.IGNORECASE,
)

SKIP_DIRS = {
    ".git",
    "build",
    "dist",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
}
MAX_BYTES = 5_000_000


def is_allowed_path(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in ALLOWED_PATHS)


def line_hits(line: str) -> bool:
    """True when this line names the old package outside the allowlist."""
    if ALLOWED_LINES.search(line):
        return False
    return bool(OLD_NAME.search(ALLOWED_TOKENS.sub("", line)))


def list_files(root: Path) -> list[str]:
    """Repository files, relative, with forward slashes: tracked plus untracked-not-ignored."""
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed git command, no shell
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        ).stdout
        files = [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        files = [
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(root).parts)
        ]
    # `git ls-files` still lists a path that was deleted in the working tree and not yet staged.
    return sorted(p for p in set(files) if (root / p).is_file())


def find_violations(root: Path) -> list[tuple[str, int, str]]:
    """(path, line number, text) for every use of the old name outside the allowlist.

    A path that carries the old name is reported with line number 0.
    """
    found: list[tuple[str, int, str]] = []
    for rel in list_files(root):
        if is_allowed_path(rel):
            continue
        if OLD_NAME.search(rel):
            found.append((rel, 0, "the path itself names the old package"))
        path = root / rel
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if line_hits(line):
                found.append((rel, number, line.strip()[:160]))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when the old package name reappears."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: the parent of scripts/)",
    )
    args = parser.parse_args(argv)
    found = find_violations(args.root.resolve())
    for rel, number, text in found:
        where = rel if number == 0 else f"{rel}:{number}"
        print(f"{where}: {text}")
    if found:
        print(
            f"\n{len(found)} use(s) of the old package name. The package is arcadedb (import arcadedb); "
            "see ALLOWED_PATHS and ALLOWED_LINES in scripts/check_no_embedded_name.py for the few "
            "places that may name the old one.",
            file=sys.stderr,
        )
        return 1
    print("ok: no use of the old package name outside the allowlist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
