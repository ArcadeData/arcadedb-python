#!/usr/bin/env python3
"""Put the Python sources of this checkout over an installed wheel, so a Python-only change runs without building a wheel.

Run it in an environment that holds ONE of the two wheels; `uv run --with` makes such an environment:

  uv run --no-project --with "<arcadedb-jvm wheel file>[test]" python scripts/dev_overlay.py -m pytest tests/test_core.py
  uv run --no-project --with "<arcadedb wheel file>" --with pytest python scripts/dev_overlay.py -m pytest tests/test_core.py
  uv run --no-project --with "arcadedb-jvm[test]==<version>" python scripts/dev_overlay.py my_script.py
  uv run --no-project --with "arcadedb==<version>" python scripts/dev_overlay.py        # make the overlay, print how to use it

The overlay is `.dev-overlay/<distribution>/arcadedb` (`ARCADEDB_DEV_OVERLAY_DIR` names another directory): every `.py` file of
`src/arcadedb` of this checkout, and links to what only the wheel has (the jars and the JRE of `arcadedb-jvm`, the libraries of
`arcadedb`, the generated `_version.py` or `_build_info.py`). For the native wheel the sources go through `nativize_sources` of
`native/wheel/build_wheel.py`, the rewrite the wheel build applies. The arguments after the script name run as
`python <arguments>` with the overlay first on `PYTHONPATH`. The overlay is made again on every run, so an edit is picked up.

Valid for a change to Python files only. The rest is still the wheel's: the bridge jar (`src/java`), the native library and the
generated files (`native/`), and the engine. When it can tell, the script lists the files of that side that changed since the
wheel was built. CI builds the real wheels and tests those; this is the quick loop before a push.
"""

import json
import os
import re
import shutil
import subprocess  # nosec B404 - git and the interpreter of this environment, fixed argument lists
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "arcadedb"
# what a wheel is built from besides src/arcadedb (the binary side and the generated files): a change there needs a new wheel
BUILT_FROM = {"arcadedb-jvm": ["src/java"], "arcadedb": ["native"]}


def installed_distribution() -> metadata.Distribution:
    """The one installed distribution of the two; exits when none or both are installed."""
    found = []
    for name in BUILT_FROM:
        try:
            found.append(metadata.distribution(name))
        except metadata.PackageNotFoundError:
            pass
    if len(found) != 1:
        sys.exit(
            f"dev_overlay: {len(found)} of the distributions arcadedb and arcadedb-jvm are installed here; it needs exactly one. "
            'Run it as: uv run --no-project --with "<wheel file, or arcadedb-jvm==<version>>" python scripts/dev_overlay.py'
        )
    return found[0]


def wheel_file(dist: metadata.Distribution) -> str | None:
    """The wheel file a distribution was installed from, or None when it came from an index."""
    text = dist.read_text("direct_url.json")
    url = json.loads(text).get("url", "") if text else ""
    if not url.startswith("file:"):
        return None
    path = unquote(urlparse(url).path)
    # file:///C:/x.whl parses to /C:/x.whl
    return path[1:] if os.name == "nt" and path[2:3] == ":" else path


def link(src: Path, dst: Path) -> None:
    """A link to a file or directory of the wheel; a copy where links are not allowed (Windows without developer mode)."""
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def build(dist: metadata.Distribution) -> Path:
    """Make the overlay of the installed distribution again and return its directory (the one that goes on PYTHONPATH)."""
    name = dist.metadata["Name"]
    wheel_pkg = Path(str(dist.locate_file("arcadedb"))).resolve()
    if not (wheel_pkg / "__init__.py").is_file():
        sys.exit(f"dev_overlay: no arcadedb package at {wheel_pkg}")
    base = Path(os.environ.get("ARCADEDB_DEV_OVERLAY_DIR") or REPO / ".dev-overlay")
    root = base / name
    pkg = root / "arcadedb"
    if root.exists():
        shutil.rmtree(root)
    # 1. the sources of this checkout (not the generated _version.py of a local build: the wheel's own is linked below)
    shutil.copytree(
        SRC,
        pkg,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "_version.py", "jars", "jre"
        ),
    )
    if name == "arcadedb":
        sys.path.insert(0, str(REPO / "native" / "wheel"))
        from build_wheel import nativize_sources

        nativize_sources(pkg)
    # 2. everything only the wheel has, in the same place
    for src in sorted(wheel_pkg.rglob("*")):
        rel = src.relative_to(wheel_pkg)
        dst = pkg / rel
        if "__pycache__" in rel.parts or src.suffix == ".pyc":
            continue
        if dst.exists() or not dst.parent.is_dir() or dst.parent.is_symlink():
            continue  # a source of the checkout, or inside a directory that is already a link
        link(src, dst)
    return root


def git(*args: str) -> str | None:
    try:
        r = subprocess.run(  # nosec B603 B607 - fixed argument list
            ["git", "-C", str(REPO), *args], capture_output=True, text=True
        )
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def built_from(dist: metadata.Distribution, root: Path) -> str | None:
    """The commit the wheel was built from: recorded in a native wheel, the release tag for a wheel from an index."""
    info = root / "arcadedb" / "_build_info.py"
    if info.is_file():
        m = re.search(
            r'^SOURCE_COMMIT = "([0-9a-f]{40})"', info.read_text(encoding="utf-8"), re.M
        )
        if m:
            return m.group(1)
    if wheel_file(dist) is None:
        return dist.version  # a release tag is the bare version
    return None


def report_changes(dist: metadata.Distribution, root: Path) -> None:
    """Say whether the files the wheel is built from, other than src/arcadedb, changed since it was built."""
    paths = BUILT_FROM[dist.metadata["Name"]]
    ref = built_from(dist, root)
    if ref is None:
        print(
            f"dev_overlay: this wheel does not record the commit it was built from. The overlay is valid only if "
            f"{', '.join(paths)} did not change since."
        )
        return
    short = ref[:12] if len(ref) == 40 else ref  # a commit, or a release tag
    changed = git("diff", "--name-only", ref, "--", *paths)
    if changed is None:
        print(
            f"dev_overlay: git cannot compare this checkout with {short}, which built the wheel (fetch the commits and tags). "
            f"The overlay is valid only if {', '.join(paths)} did not change since."
        )
    elif changed:
        print(
            f"dev_overlay: WARNING: these files changed since the wheel was built ({short}), and the wheel does not have the "
            "change. If the import or a test fails here, take a wheel that CI built from main:"
        )
        for line in changed.splitlines():
            print(f"    {line}")
    else:
        print(
            f"dev_overlay: {', '.join(paths)} did not change since the wheel was built ({short})."
        )


def main() -> int:
    dist = installed_distribution()
    name = dist.metadata["Name"]
    root = build(dist)
    n = sum(1 for _ in SRC.rglob("*.py"))
    print(
        f"dev_overlay: {name} {dist.version}: {n} Python files of src/arcadedb over the wheel, in {root}"
    )
    report_changes(dist, root)
    args = sys.argv[1:]
    if args:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(root), env.get("PYTHONPATH")) if p
        )
        sys.stdout.flush()  # the lines above come before the output of the command
        return subprocess.call([sys.executable, *args], env=env)  # nosec B603
    wheel = wheel_file(dist)
    extra = "[test]" if name == "arcadedb-jvm" else ""
    spec = f"{wheel}{extra}" if wheel else f"{name}{extra}=={dist.version}"
    more = "" if extra else " --with pytest --with numpy"
    print(
        "Run a test file against it (the overlay is made again each time):\n"
        f'  uv run --no-project --with "{spec}"{more} python scripts/dev_overlay.py -m pytest tests/test_core.py'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
