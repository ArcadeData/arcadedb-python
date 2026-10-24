#!/usr/bin/env python3
"""Rename this repository to the final package names. Re-runnable, deterministic, idempotent.

Final names (owner decision 2026-10-25): distribution `arcadedb` (JVM wheel) and `arcadedb-native`
(native wheel, when ported); the import name is `arcadedb` for BOTH. No `arcadedb_embedded` and no
`arcadedb-embedded` anywhere in this repository. There is no compatibility shim here: the old PyPI
packages get a deprecation shim from the humemai repository.

Why a script and not a branch full of edits: the rename touches about 130 files, and other pull
requests keep merging into them. A hand-made rename conflicts with each of those. This script is run
on a fresh checkout of main once the open stack has merged, so it never conflicts, and running it
again changes nothing.

usage (from a clean checkout of main, on the rename branch):

    python3 scripts/rename_to_arcadedb.py              # rewrite, format, check
    python3 scripts/rename_to_arcadedb.py --dry-run    # report only, change nothing
    python3 scripts/rename_to_arcadedb.py --no-format  # skip the pre-commit formatting pass

What it does, in order:

1. Moves `src/arcadedb_embedded` to `src/arcadedb` with `git mv` (file by file when both exist).
2. Applies a few exact-text edits to prose that stops being true: the release workflow comments, the
   package attributes in docs/antora.yml, the two statements that the native wheel installs beside the
   JVM wheel (it cannot: both import as `arcadedb`), and the rows of the package table for the placeholder
   names. Each edit reports "applied", "already", or "MISSING" (the text moved; fix by hand).
3. Rewrites every other use of the old name in every text file, line by line:
   `import arcadedb_embedded as arcadedb` becomes `import arcadedb`; `arcadedb-embedded-native` becomes
   `arcadedb-native`; the repository URL becomes https://github.com/ArcadeData/arcadedb-python;
   `ArcadeDB Embedded Python` becomes `ArcadeDB Python`; any other `arcadedb_embedded` or
   `arcadedb-embedded` becomes `arcadedb`. That covers imports, pyproject name, packages and
   package-data, build.sh, build-native.sh, Dockerfile.build, setup.py, .gitignore, CI workflow paths
   and names, wheel-content checks, the tests and conftest, the examples, README and the Antora pages.
4. Adds the docs line "This package was published as arcadedb-embedded until 26.10.x; install arcadedb
   now." to README.md and docs/modules/ROOT/pages/installation.adoc.
5. Wires the check in: a local pre-commit hook and tests/test_no_embedded_name.py, both running
   scripts/check_no_embedded_name.py.
6. Runs the repository's pre-commit hooks (black, isort, shfmt) on the Python and shell files it touched
   (twice, so the second pass proves them clean) unless --no-format, then the check.

What it never touches:

* Java package names (com.arcadedb...), environment variables (ARCADEDB_*), engine artifact names
  (arcadedb-engine, the arcadedata/arcadedb image), and every `arcadedb` that is not followed by
  `_embedded` or `-embedded`.
* History: files and lines on the allowlist of scripts/check_no_embedded_name.py (docs/superpowers/,
  changelogs, the "published as"/"formerly" lines, `humemai/arcadedb-embedded-python#123` issue references).
* The release workflow's guard logic: only two comments that describe the old state are edited.
* The generic word "embedded" (embedded mode, embedded database).

Output lists what changed by category (path-based) and by rule. Exit 0 when the tree is clean after the
run, 1 when the check still finds the old name, 2 when --strict and an exact-text edit is MISSING.
Output is ASCII only.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess  # nosec B404 - runs git and pre-commit
import sys
from collections import Counter
from pathlib import Path

OLD_PKG = "arcadedb_embedded"
NEW_PKG = "arcadedb"
OLD_DIR = "src/arcadedb_embedded"
NEW_DIR = "src/arcadedb"

DOCS_LINE = "This package was published as arcadedb-embedded until 26.10.x; install arcadedb now."


def _load_checker():
    path = Path(__file__).resolve().with_name("check_no_embedded_name.py")
    spec = importlib.util.spec_from_file_location("check_no_embedded_name", path)
    assert (
        spec is not None and spec.loader is not None
    )  # nosec B101 - import machinery contract
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()

# --- generic rules: (name, pattern, replacement), applied in this order to every line -------------
RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "repository URL",
        re.compile(r"github\.com/humemai/arcadedb-embedded-python"),
        "github.com/ArcadeData/arcadedb-python",
    ),
    ("repository name", re.compile(r"arcadedb-embedded-python"), "arcadedb-python"),
    (
        "native distribution name",
        re.compile(r"arcadedb-embedded-native"),
        "arcadedb-native",
    ),
    ("project title", re.compile(r"ArcadeDB Embedded Python"), "ArcadeDB Python"),
    (
        "import statement",
        re.compile(r"\bimport arcadedb_embedded as arcadedb(?![A-Za-z0-9_])"),
        "import arcadedb",
    ),
    ("distribution or package name", re.compile(r"arcadedb[_-]embedded"), "arcadedb"),
)

# --- exact-text edits: (path, label, old, new). "already" when new is present, "MISSING" otherwise ---
EDITS: tuple[tuple[str, str, str, str], ...] = (
    (
        ".github/workflows/release.yml",
        "release.yml header comment",
        "# Until the distribution is renamed (tracker #19), pyproject.toml still names it arcadedb-embedded.\n"
        "# The guard in the publish job refuses to publish anything that is not `arcadedb`, so this workflow\n"
        "# cannot publish arcadedb-embedded from here by accident. The rename is a separate change.\n",
        "# pyproject.toml names the distribution arcadedb (tracker #19).\n"
        "# The guard in the publish job refuses to publish anything that is not `arcadedb`, so this workflow\n"
        "# cannot publish another distribution from here by accident.\n",
    ),
    (
        ".github/workflows/release.yml",
        "release.yml guard comment",
        "      # GUARD: this workflow publishes the distribution `arcadedb` and nothing else. While the\n"
        "      # package is still named arcadedb-embedded (tracker #19, rename is a separate change), the\n"
        "      # job stops here, before anything reaches PyPI. Two checks, so neither can drift alone:\n",
        "      # GUARD: this workflow publishes the distribution `arcadedb` and nothing else. A wheel with\n"
        "      # another name stops the job here, before anything reaches PyPI. Two checks, so neither can drift alone:\n",
    ),
    (
        "docs/antora.yml",
        "antora package attributes",
        "    # Package names. After the rename of the distribution these are the only\n"
        "    # lines to change; every page takes the names from here.\n",
        "    # Package names. Every page takes them from here.\n",
    ),
    (
        "docs/antora.yml",
        "antora native import name",
        "    native-import-name: arcadedb_native\n"
        "    # Other names a reader meets on PyPI (the pages that compare packages use these).\n"
        "    short-pkg-name: arcadedb\n"
        "    short-native-pkg-name: arcadedb-native\n"
        "    jvm-pkg-name: arcadedb-jvm\n",
        "    native-import-name: arcadedb\n"
        "    # Other names a reader meets on PyPI (the pages that compare packages use these).\n"
        "    jvm-pkg-name: arcadedb-jvm\n",
    ),
    (
        "docs/modules/ROOT/pages/packages.adoc",
        "packages table: placeholder rows",
        "| `{short-pkg-name}`\n"
        "| Embedded, JVM build\n"
        "| The name `{pkg-name}` moves to in the middle of November. Until then a placeholder on PyPI that installs `{pkg-name}`.\n"
        "\n"
        "| `{short-native-pkg-name}`\n"
        "| Embedded, native build (preview)\n"
        "| The name `{native-pkg-name}` moves to at the same time. Until then an alias that installs it.\n"
        "\n"
        "| `{jvm-pkg-name}`\n"
        "| Embedded, JVM build\n"
        "| A reserved name for the JVM build. Until the move it installs `{pkg-name}`.\n",
        "| `{jvm-pkg-name}`\n"
        "| Embedded, JVM build\n"
        "| A reserved name for the JVM build. It installs `{pkg-name}`.\n",
    ),
    (
        "docs/modules/ROOT/pages/index.adoc",
        "index page: side by side",
        "The preview installs beside the default package. Its public API follows the one of `{import-name}`, so the pages in this component apply to both;",
        "Both packages import as `{import-name}`, so an environment holds one of them. The preview's public API follows the one of the default package, so the pages in this component apply to both;",
    ),
    (
        "docs/modules/ROOT/pages/installation.adoc",
        "installation page: side by side",
        "It installs beside `{pkg-name}`: the distribution names differ, and so do the import names (`{native-import-name}` instead of `{import-name}`).",
        "It imports as `{native-import-name}`, the same name as `{pkg-name}`, so install only one of the two in an environment.",
    ),
    # The docs line, after the generic rewrite has produced the install commands it follows.
    (
        "README.md",
        "README docs line",
        "```bash\nuv add arcadedb   # or: pip install arcadedb\n```\n",
        "```bash\nuv add arcadedb   # or: pip install arcadedb\n```\n\n"
        + DOCS_LINE
        + "\n",
    ),
    (
        "docs/modules/ROOT/pages/installation.adoc",
        "installation page docs line",
        '[source,shell,subs="attributes+"]\n----\npip install {pkg-name}\n----\n',
        '[source,shell,subs="attributes+"]\n----\npip install {pkg-name}\n----\n\n'
        + DOCS_LINE
        + "\n",
    ),
)

# Edits that must run after the generic rewrite (their old text is the rewritten form).
LATE_EDITS = {"README docs line", "installation page docs line"}

PRE_COMMIT_HOOK = """\
- repo: local
  hooks:
  - id: no-old-package-name
    name: the old package name does not come back
    entry: python3 scripts/check_no_embedded_name.py
    language: system
    pass_filenames: false
    always_run: true
"""

CHECK_TEST = '''\
"""The old package name must not come back.

Runs scripts/check_no_embedded_name.py over the repository. It needs no wheel. The allowlist (history,
the rename codemod, issue references) lives in that script.
"""

import subprocess  # nosec B404 - runs sys.executable on a script this repository owns
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECK = ROOT / "scripts" / "check_no_embedded_name.py"


def test_old_package_name_is_absent():
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-I", str(CHECK), "--root", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
'''


def category(rel: str) -> str:
    if rel in ("pyproject.toml", "setup.py"):
        return "packaging metadata"
    if rel.startswith(".github/"):
        return "CI workflows"
    if rel.startswith("src/"):
        return "package source"
    if rel.startswith("tests/") and rel.endswith(".py"):
        return "tests"
    if rel.startswith("examples/") and rel.endswith(".py"):
        return "examples"
    if rel.startswith("scripts/") or rel in (".gitignore", ".pre-commit-config.yaml"):
        return "build and verify scripts"
    return "docs"


class Report:
    def __init__(self) -> None:
        self.moves: list[str] = []
        self.files: Counter[str] = Counter()  # category -> files changed
        self.edits_made: Counter[str] = Counter()  # category -> replacements
        self.by_rule: Counter[str] = Counter()
        self.exact: list[tuple[str, str, str]] = []  # (status, path, label)
        self.wiring: list[str] = []
        self.touched: set[str] = (
            set()
        )  # paths written or moved, for the formatting pass
        self.notes: list[str] = []

    def show(self, dry: bool) -> None:
        mode = " (dry run, nothing written)" if dry else ""
        print(f"Rename to arcadedb{mode}\n")
        print("Moved paths:")
        for m in self.moves or ["  none (already moved)"]:
            print(m if m.startswith(" ") else f"  {m}")
        print("\nExact-text edits:")
        for status, path, label in self.exact:
            print(f"  {status:8} {path}: {label}")
        print("\nChanged by category (files / replacements):")
        cats = sorted(set(self.files) | set(self.edits_made))
        for cat in cats:
            print(f"  {cat:26} {self.files[cat]:4} / {self.edits_made[cat]}")
        print(
            f"  {'total':26} {sum(self.files.values()):4} / {sum(self.edits_made.values())}"
        )
        print("\nChanged by rule (replacements):")
        for rule, n in sorted(self.by_rule.items()):
            print(f"  {rule:36} {n}")
        if self.wiring:
            print("\nWiring:")
            for w in self.wiring:
                print(f"  {w}")
        for n in self.notes:
            print(f"\n{n}")


def run_git(root: Path, *args: str) -> str:
    return subprocess.run(  # nosec B603 B607 - fixed git argv, no shell
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def move_package(root: Path, report: Report, dry: bool) -> None:
    old, new = root / OLD_DIR, root / NEW_DIR
    if not old.exists():
        return
    tracked = [
        p for p in run_git(root, "ls-files", "-z", "--", OLD_DIR).split("\0") if p
    ]
    if not new.exists():
        report.moves.append(
            f"git mv {OLD_DIR} -> {NEW_DIR} ({len(tracked)} tracked files)"
        )
        if not dry:
            run_git(root, "mv", OLD_DIR, NEW_DIR)
        return
    # Both exist (a merge brought a new file into the old directory): move file by file.
    for rel in tracked:
        target = NEW_DIR + rel[len(OLD_DIR) :]
        report.moves.append(f"git mv {rel} -> {target}")
        if not dry:
            (root / target).parent.mkdir(parents=True, exist_ok=True)
            run_git(root, "mv", rel, target)
    if not dry and old.exists() and not any(old.rglob("*")):
        old.rmdir()


def text_files(root: Path) -> list[str]:
    """Files the rewrite may touch: every repository file that is not on the allowlist."""
    return [f for f in checker.list_files(root) if not checker.is_allowed_path(f)]


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data or len(data) > checker.MAX_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def apply_rules(line: str, rules_hit: Counter[str]) -> str:
    """Generic rewrite of one line; issue references of the old repository are left as written."""
    if checker.ALLOWED_LINES.search(line):
        return line
    pieces = re.split(f"({checker.ALLOWED_TOKENS.pattern})", line)
    for i in range(0, len(pieces), 2):  # even pieces are outside the issue references
        for name, pattern, repl in RULES:
            pieces[i], n = pattern.subn(repl, pieces[i])
            rules_hit[name] += n
    return "".join(pieces)


def exact_edit(rel: str, text: str, wanted: set[str], report: Report) -> str:
    for path, label, old, new in EDITS:
        if path != rel or label not in wanted:
            continue
        if "\r\n" in text:  # a Windows checkout with autocrlf
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        if (
            new in text
        ):  # first: the docs-line edits keep their old text in the new text
            report.exact.append(("already", path, label))
        elif old in text:
            text = text.replace(old, new, 1)
            report.exact.append(("applied", path, label))
        else:
            report.exact.append(("MISSING", path, label))
    return text


def rewrite_files(root: Path, report: Report, dry: bool) -> None:
    early = {label for _, label, _, _ in EDITS if label not in LATE_EDITS}
    paths = {p for p, _, _, _ in EDITS}
    for rel in text_files(root):
        src = root / rel
        text = read_text(src)
        if text is None:
            continue
        before = text
        rules_hit: Counter[str] = Counter()
        if rel in paths:
            text = exact_edit(rel, text, early, report)
        text = "".join(
            apply_rules(line, rules_hit) for line in text.splitlines(keepends=True)
        )
        if rel in paths:
            text = exact_edit(rel, text, LATE_EDITS, report)
        if text != before:
            cat = category(rel)
            report.files[cat] += 1
            report.edits_made[cat] += sum(rules_hit.values()) + sum(
                1 for s, p, _ in report.exact if p == rel and s == "applied"
            )
            report.by_rule.update(rules_hit)
            report.touched.add(rel)
            if not dry:
                src.write_bytes(text.encode("utf-8"))


def wire_check(root: Path, report: Report, dry: bool) -> None:
    hook_file = root / ".pre-commit-config.yaml"
    if hook_file.exists():
        text = hook_file.read_text(encoding="utf-8")
        if "check_no_embedded_name.py" in text:
            report.wiring.append("already  .pre-commit-config.yaml: check hook")
        else:
            report.wiring.append(
                "added    .pre-commit-config.yaml: local hook running the check"
            )
            if not dry:
                hook_file.write_text(
                    text.rstrip("\n") + "\n" + PRE_COMMIT_HOOK, encoding="utf-8"
                )
    test_file = root / "tests" / "test_no_embedded_name.py"
    if test_file.exists():
        report.wiring.append("already  tests/test_no_embedded_name.py")
    elif (root / "tests").is_dir():
        report.wiring.append("added    tests/test_no_embedded_name.py")
        report.touched.add("tests/test_no_embedded_name.py")
        if not dry:
            test_file.write_text(CHECK_TEST, encoding="utf-8")


def format_tree(root: Path, report: Report) -> bool:
    """Format the Python and shell files this run touched, with the hooks the repository already has.

    Only the touched files, and only .py and .sh: `pre-commit run --all-files` would also rewrite files
    this run did not change (docs/antora.yml is not clean under the YAML hook on main). The first pass
    autofixes; the second proves the files are clean.
    """
    files = sorted(
        f
        for f in report.touched
        if f.endswith((".py", ".sh"))
        and (root / f).is_file()
        and not checker.is_allowed_path(f)
    )
    if not files:
        report.notes.append("pre-commit: no Python or shell file to format")
        return True
    for cmd in (["pre-commit"], ["uvx", "pre-commit"]):
        try:
            for attempt in (1, 2):
                result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
                    [*cmd, "run", "--files", *files],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    report.notes.append(
                        f"pre-commit: {len(files)} file(s) clean after pass {attempt}"
                    )
                    return True
            report.notes.append(
                "pre-commit: still failing after two passes:\n" + result.stdout[-2000:]
            )
            return False
        except FileNotFoundError:
            continue
    report.notes.append(
        "pre-commit: not found (tried pre-commit and uvx pre-commit); formatting step skipped"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rename arcadedb-embedded to arcadedb (see the module docstring)."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    parser.add_argument(
        "--no-format", action="store_true", help="skip the pre-commit formatting pass"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 when an exact-text edit is MISSING",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report = Report()

    move_package(root, report, args.dry_run)
    rewrite_files(root, report, args.dry_run)
    wire_check(root, report, args.dry_run)
    formatted_ok = True
    if not args.dry_run and not args.no_format:
        formatted_ok = format_tree(root, report)
    report.show(args.dry_run)

    if args.dry_run:
        return 0
    leftovers = checker.find_violations(root)
    for rel, number, text in leftovers:
        print(f"LEFTOVER {rel}:{number}: {text}")
    missing = [x for x in report.exact if x[0] == "MISSING"]
    print(
        f"\ncheck: {len(leftovers)} leftover use(s) of the old name, {len(missing)} MISSING exact edit(s)"
    )
    if leftovers or not formatted_ok:
        return 1
    if missing and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
