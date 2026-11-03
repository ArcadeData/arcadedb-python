#!/usr/bin/env python3
"""Move the engine that dev, PR and nightly builds link to a new upstream release.

After upstream releases the engine NEW (A.B.C), the pins below follow it, and the dev version moves
to the next base NEXT_DEV (A.B.C.dev0 of the next base, because a devN of a released base is refused):

  .github/workflows/test-native.yml   DEV_ENGINE: "<engine>"
  .github/workflows/test-native.yml   DEFAULT_VERSION: "<dev version>"
  .github/workflows/test-native.yml   the default of the manual-run input `version`: "<dev version>"
  native/pom.xml                      <arcadedb.version><engine></arcadedb.version>
  native/scripts/build-image.sh       ENGINE=${ARCADEDB_ENGINE_VERSION:-<engine>}
  native/wheel/build_wheel.py         the default of --version: "<dev version>"

A release tag derives its engine from the tag and needs none of these. Each pin is one exact
anchor: a missing or repeated anchor stops the script before it writes anything.

native/wheel/THIRD_PARTY_NOTICES.md names the engine too, but it is generated from the build log
of the native image, so this script only reports it (see docs, "Release day"): the build fails
on that drift until the notices are regenerated.

usage:
    bump_engine.py --engine NEW --dev-version NEXT_DEV            # edit the four pins
    bump_engine.py --engine NEW --dev-version NEXT_DEV --check    # report drift, write nothing

NEXT_DEV must be a devN version that scripts/check_version.py would accept as the next version
(read from the git tags and PyPI of both packages, or from --published-file instead), and its
base must be higher than the base of NEW.

Exit codes: 0 no drift (or edited), 1 drift found by --check or a refused version, 2 an anchor
or a source could not be read. Output is ASCII only.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PACKAGES = ("arcadedb-jvm", "arcadedb")
OWN_REMOTE = "https://github.com/ArcadeData/arcadedb-python.git"

WORKFLOW = Path(".github/workflows/test-native.yml")
POM = Path("native/pom.xml")
BUILD_IMAGE = Path("native/scripts/build-image.sh")
BUILD_WHEEL = Path("native/wheel/build_wheel.py")
NOTICES = Path("native/wheel/THIRD_PARTY_NOTICES.md")

# (file, label, regex with one group around the value). One match each, exactly.
PINS = (
    (WORKFLOW, "DEV_ENGINE", r'^  DEV_ENGINE: "([^"]+)"[ \t]*$', "engine"),
    (WORKFLOW, "DEFAULT_VERSION", r'^  DEFAULT_VERSION: "([^"]+)"[ \t]*$', "dev"),
    (
        WORKFLOW,
        "manual-run version default",
        r'(?:Wheel version to build \(PEP 440: A\.B\.C\.devN, A\.B\.C or A\.B\.C\.postN\)"\n'
        r'        required: false\n        default: ")([^"]+)"',
        "dev",
    ),
    (
        POM,
        "arcadedb.version",
        r"<arcadedb\.version>([^<]+)</arcadedb\.version>",
        "engine",
    ),
    (
        BUILD_IMAGE,
        "ENGINE default",
        r"^ENGINE=\$\{ARCADEDB_ENGINE_VERSION:-([^}]+)\}[ \t]*$",
        "engine",
    ),
    (
        BUILD_WHEEL,
        "build_wheel.py --version default",
        r'^        "--version",\n        default="([^"]+)",',
        "dev",
    ),
)
NOTICES_RE = re.compile(r"\(engine ([0-9][^,)]*),")


class BumpError(Exception):
    """An anchor is missing or repeated, or a source could not be read."""


def _load_check_version():
    spec = importlib.util.spec_from_file_location(
        "check_version", SCRIPTS / "check_version.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # nosec B101 - spec of an existing file
    spec.loader.exec_module(module)
    return module


def validate_versions(engine: str, dev: str, published: list[str]) -> str | None:
    """None when the pair is acceptable, else the reason it is refused."""
    cv = _load_check_version()
    try:
        e = cv.parse_version(engine)
        d = cv.parse_version(dev)
    except cv.VersionError as err:
        return str(err)
    if e.kind != "":
        return f"--engine {engine} must be a release A.B.C (no devN or postN)"
    if d.kind != "dev":
        return f"--dev-version {dev} must be a devN version (A.B.C.devN)"

    def key(base: str) -> tuple[int, ...]:
        return tuple(int(x) for x in base.split("."))

    if key(d.base) <= key(e.base):
        return (
            f"--dev-version {dev}: its base must be higher than the engine {engine}, "
            "because a devN of a released base sorts before the release and is refused"
        )
    try:
        cv.check_next(dev, published)
    except cv.VersionError as err:
        return f"--dev-version refused by check_version.py: {err}"
    return None


def published_versions(files: list[str], remote: str) -> list[str]:
    cv = _load_check_version()
    published: list[str] = []
    try:
        if files:
            for path in files:
                published += cv.read_lines(path)
        else:
            for package in PACKAGES:
                published += cv.pypi_versions(package)
            published += cv.git_tag_versions(remote)
    except OSError as err:
        raise BumpError(str(err)) from err
    return published


def plan(root: Path, engine: str, dev: str) -> list[tuple[Path, str, str, str, str]]:
    """For each pin: (file, label, regex, current value, wanted value). Fails on a missing or repeated anchor."""
    rows = []
    for rel, label, pattern, kind in PINS:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except OSError as err:
            raise BumpError(f"cannot read {rel}: {err}") from err
        found = re.findall(pattern, text, re.M)
        if len(found) != 1:
            raise BumpError(
                f"{rel}: expected exactly one anchor for {label}, found {len(found)} "
                "(the file changed: update PINS in scripts/bump_engine.py)"
            )
        rows.append(
            (rel, label, pattern, found[0], engine if kind == "engine" else dev)
        )
    return rows


def apply(root: Path, rows) -> None:
    """Write every pin that differs. All anchors were counted by plan() before this runs."""
    by_file: dict[Path, str] = {}
    for rel, label, pattern, current, wanted in rows:
        text = by_file.get(rel) or (root / rel).read_text(encoding="utf-8")
        new, count = re.subn(
            pattern,
            lambda m, w=wanted: (
                m.group(0)[: m.start(1) - m.start(0)]
                + w
                + m.group(0)[m.end(1) - m.start(0) :]
            ),
            text,
            flags=re.M,
        )
        if count != 1:
            raise BumpError(f"{rel}: {label} anchor matched {count} times on write")
        by_file[rel] = new
    for rel, text in by_file.items():
        (root / rel).write_text(text, encoding="utf-8")


def notices_engine(root: Path) -> str | None:
    try:
        text = (root / NOTICES).read_text(encoding="utf-8")
    except OSError as err:
        raise BumpError(f"cannot read {NOTICES}: {err}") from err
    found = NOTICES_RE.findall(text)
    return found[0] if len(set(found)) == 1 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", required=True, help="the released engine A.B.C")
    parser.add_argument(
        "--dev-version",
        required=True,
        help="the next dev version, A.B.C.dev0 of the next base",
    )
    parser.add_argument(
        "--check", action="store_true", help="report drift, write nothing"
    )
    parser.add_argument("--root", default=str(SCRIPTS.parent), help="repository root")
    parser.add_argument(
        "--published-file",
        action="append",
        default=[],
        metavar="FILE",
        help="published versions, one per line, read instead of PyPI and git tags (tests, offline)",
    )
    parser.add_argument("--git-remote", default=OWN_REMOTE)
    args = parser.parse_args(argv)
    root = Path(args.root)

    try:
        published = published_versions(args.published_file, args.git_remote)
        reason = validate_versions(args.engine, args.dev_version, published)
        if reason:
            print(f"FAIL: {reason}")
            return 1
        rows = plan(root, args.engine, args.dev_version)
        notices = notices_engine(root)
    except BumpError as err:
        print(f"ERROR: {err}")
        return 2

    drift = 0
    for rel, label, _pattern, current, wanted in rows:
        if current == wanted:
            print(f"OK     {rel}  {label} = {wanted}")
        else:
            drift += 1
            print(f"DRIFT  {rel}  {label}: {current} -> {wanted}")
    notices_drift = notices != args.engine
    if notices_drift:
        print(
            f"TODO   {NOTICES}  names engine {notices or 'several'}, not {args.engine}: regenerate "
            "it with native/wheel/gen_notices.py from the first build log of the new engine "
            "(docs, Release day); this script does not edit it"
        )
    else:
        print(f"OK     {NOTICES}  names engine {args.engine}")

    if args.check:
        return 1 if (drift or notices_drift) else 0
    if drift:
        try:
            apply(root, rows)
        except BumpError as err:
            print(f"ERROR: {err}")
            return 2
        print(f"edited {drift} pin(s); review with git diff")
    else:
        print("nothing to edit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
