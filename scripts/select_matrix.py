#!/usr/bin/env python3
"""Pick the platform x Python legs a CI workflow runs for the event that started it.

The wheels support four platforms and four Python versions.

- `jvm` (test-jvm.yml) runs the same 10 legs on every event: a pull request, a push to main, the nightly
  schedule, a manual dispatch and the release workflow (which calls the workflow as `workflow_call`). The JVM
  wheel is one file for each platform (py3-none-<platform>, built once by the `build` job), so a Python version
  is a test leg, not a build. Like the native package (test-native.yml), the Pythons are all four on
  linux/amd64 and the oldest and the newest on the other platforms: a bug that depends on the interpreter shows
  on the ends of the range, and one that depends on the platform shows on any version of it. The organisation
  shares one runner cap with ArcadeData/arcadedb, so every extra macOS or Windows leg delays upstream too.
- `examples` (test-python-examples.yml) runs the newest Python on every platform on a pull request, and every
  combination (16 legs) on every other event.

The selection fails open to the full matrix: an event name this script does not know runs everything, never less
(for `jvm`, whose legs do not depend on the event, that is the same 10 legs).

usage:
    select_matrix.py jvm [--event NAME]             # test-jvm.yml: the test legs
    select_matrix.py jvm [--event NAME] --wheels    # test-jvm.yml: the build legs, one for each platform of the test legs
    select_matrix.py examples [--event NAME]        # test-python-examples.yml
    select_matrix.py examples [--event NAME] --wheels   # test-python-examples.yml: the build legs, one for each platform

`--event` defaults to $GITHUB_EVENT_NAME. The output is one line of JSON for
`strategy.matrix`: {"include": [{"platform": ..., "runs-on": ..., "python-version": ...}, ...]}, and with
`--wheels` {"include": [{"platform": ..., "runs-on": ...}, ...]}.
The names of the jobs stay "Test arcadedb (<platform>, Python <version>)", so the history of a
check is comparable before and after the selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Temporarily limited to four platforms: macOS x86_64 (runner macos-15-intel) and Windows ARM64
# (windows-11-arm) are off. This is the platform column both workflows had before this script.
RUNNERS = {
    "linux/amd64": "ubuntu-24.04",
    "linux/arm64": "ubuntu-24.04-arm",
    "darwin/arm64": "macos-15",
    "windows/amd64": "windows-2025",
}

# Every Python the package supports. tests/test_select_matrix.py compares this with the
# classifiers of pyproject.toml, so adding a version there without adding it here fails.
PYTHONS = ("3.11", "3.12", "3.13", "3.14")
OLDEST = PYTHONS[0]
NEWEST = PYTHONS[-1]

# The legs of `jvm`, the same for every event: all Pythons on linux/amd64 (the cheap, fast runner and the one most
# contributors use), the oldest and the newest on the three other platforms (the shape of test-native.yml). The wheel
# is py3-none-<platform>, one file for each platform, so these are test legs against that file, not builds.
JVM_LEGS = {
    "linux/amd64": PYTHONS,
    "linux/arm64": (OLDEST, NEWEST),
    "darwin/arm64": (OLDEST, NEWEST),
    "windows/amd64": (OLDEST, NEWEST),
}

# The legs of a pull request, for the workflows whose legs depend on the event.
#
# examples: the newest Python only, on every platform. The examples are slow (torch,
# sentence-transformers), and they carry platform-specific code (`os.name == "nt"` branches in
# examples 07 to 12) and per-platform steps in the workflow (a CPU torch index on macOS, UTF-8
# on Windows), so all four platforms stay.
PULL_REQUEST_LEGS = {
    "examples": {platform: (NEWEST,) for platform in RUNNERS},
}

WORKFLOWS = ("jvm", "examples")


def _legs(per_platform: dict[str, tuple[str, ...]]) -> list[dict[str, str]]:
    return [
        {"platform": platform, "runs-on": RUNNERS[platform], "python-version": python}
        for platform in RUNNERS
        for python in per_platform.get(platform, ())
    ]


def select(workflow: str, event: str) -> dict[str, list[dict[str, str]]]:
    """The `strategy.matrix` of `workflow` ("jvm" or "examples") for `event`."""
    if workflow not in WORKFLOWS:
        raise ValueError(f"unknown workflow {workflow!r}; use one of {sorted(WORKFLOWS)}")
    if workflow == "jvm":
        return {"include": _legs(JVM_LEGS)}
    if event == "pull_request":
        return {"include": _legs(PULL_REQUEST_LEGS[workflow])}
    return {"include": _legs({platform: PYTHONS for platform in RUNNERS})}


def select_wheels(workflow: str, event: str) -> dict[str, list[dict[str, str]]]:
    """The build legs of `workflow` for `event`: one for each platform that has a test leg, whatever the Python."""
    runners: dict[str, str] = {}
    for leg in select(workflow, event)["include"]:
        runners.setdefault(leg["platform"], leg["runs-on"])
    return {"include": [{"platform": p, "runs-on": r} for p, r in runners.items()]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workflow", choices=sorted(WORKFLOWS))
    parser.add_argument(
        "--event",
        default=os.environ.get("GITHUB_EVENT_NAME", ""),
        help="the event that started the workflow (default: $GITHUB_EVENT_NAME)",
    )
    parser.add_argument(
        "--wheels",
        action="store_true",
        help="print the build legs (one for each platform) instead of the test legs",
    )
    args = parser.parse_args(argv)
    chosen = select_wheels if args.wheels else select
    print(json.dumps(chosen(args.workflow, args.event), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
