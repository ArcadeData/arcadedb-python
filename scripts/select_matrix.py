#!/usr/bin/env python3
"""Pick the platform x Python legs a CI workflow runs for the event that started it.

The wheels support four platforms and four Python versions.

CI has two tiers. A pull request runs the fast tier: linux/amd64 only and no example legs, so it does not wait for
the macOS and Windows runners the organisation shares with ArcadeData/arcadedb, nor for the examples (the slowest
jobs). Every other event runs the full tier, and a pull request that asks for it (the label `ci:full`, or a
Dependabot author; scripts/ci_tier.py decides and passes the event `pull_request_full`) runs the legs of a push to
main. A pull request that changes the examples or their runner gets the event `pull_request_examples`: the fast
tier and the example leg of linux/amd64.

- `jvm` (test-jvm.yml) runs 2 legs on a pull request (linux/amd64, the oldest and the newest Python), 10 legs on
  a push to main and the nightly schedule, and all 16 for a release (release-jvm.yml calls the workflow with
  `release: true`, which test-jvm.yml passes as the event `release`) and a manual dispatch. The JVM wheel is one
  file for each platform (py3-none-<platform>, built once by the `build` job), so a Python version is a test leg,
  not a build. The 10 legs are all four Pythons on linux/amd64 and the oldest and the newest on the other
  platforms: a bug that depends on the interpreter shows on the ends of the range, and one that depends on the
  platform shows on any version of it. What is published, and a run started by hand before a release, gets every
  combination, like test-native.yml.
- `examples` (test-python-examples.yml) runs no leg on a pull request (the matrix is empty and the workflow skips
  its jobs), the newest Python on linux/amd64 on a pull request that changes the examples, and every combination
  (16 legs) on every other event.

The selection fails open to the full matrix: an event name this script does not know runs everything, never less
(for `jvm` too: only a pull request, a push and the nightly schedule get fewer than 16 legs).

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

# Temporarily limited to four platforms: macOS x86_64 (runner macos-15-intel) is off. Windows ARM64 (runner
# windows-11-arm exists) is off because neither package can be built there: jpype1 1.7.1 ships no win_arm64 wheel
# (win_amd64 only) and GraalVM CE 25.4.4.1.1 has no Windows ARM64 build (windows-x64 only). This is the platform
# column both workflows had before this script.
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

# The legs of `jvm` on a push and the nightly run (and on a pull request that asks for the full tier): all Pythons on linux/amd64
# (the cheap, fast runner and the one most contributors use), the oldest and the newest on the three other platforms (the shape of
# test-native.yml). The wheel is py3-none-<platform>, one file for each platform, so these are test legs against that file, not builds.
JVM_LEGS = {
    "linux/amd64": PYTHONS,
    "linux/arm64": (OLDEST, NEWEST),
    "darwin/arm64": (OLDEST, NEWEST),
    "windows/amd64": (OLDEST, NEWEST),
}

# The events on which `jvm` runs only JVM_LEGS. Every other event (a release, a manual dispatch, an event this
# script does not know) runs every platform x Python combination, so the selection fails open.
JVM_REDUCED_EVENTS = ("push", "schedule")

# The fast tier: the legs of a pull request, linux/amd64 only (ArcadeData/arcadedb-python#442). The other
# platforms run after the merge (the push to main) and every night, and before the merge when the pull request
# carries the label `ci:full`.
#
# jvm: the oldest and the newest Python, the ends of the range.
# examples: no leg. The examples are the slowest jobs (about 20 to 30 minutes: they build the vectors of a dataset
# and run 30 scripts), so they run in the full tier. A pull request that changes the examples or their runner
# (the event `pull_request_examples`, scripts/ci_tier.py) runs them on the newest Python; their platform-specific
# code (`os.name == "nt"` branches in examples 07 to 12) and the per-platform steps of the workflow (a CPU torch
# index on macOS, UTF-8 on Windows) are covered by the full tier.
FAST_PLATFORM = "linux/amd64"
PULL_REQUEST_LEGS = {
    "jvm": {FAST_PLATFORM: (OLDEST, NEWEST)},
    "examples": {},
}
PULL_REQUEST_EXAMPLES = "pull_request_examples"
PULL_REQUEST_EXAMPLES_LEGS = {
    "jvm": PULL_REQUEST_LEGS["jvm"],
    "examples": {FAST_PLATFORM: (NEWEST,)},
}
# The event scripts/ci_tier.py passes for a pull request that asks for the full tier: the legs of a push to main.
PULL_REQUEST_FULL = "pull_request_full"

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
        raise ValueError(
            f"unknown workflow {workflow!r}; use one of {sorted(WORKFLOWS)}"
        )
    if event == PULL_REQUEST_FULL:
        event = "push"
    if event == "pull_request":
        return {"include": _legs(PULL_REQUEST_LEGS[workflow])}
    if event == PULL_REQUEST_EXAMPLES:
        return {"include": _legs(PULL_REQUEST_EXAMPLES_LEGS[workflow])}
    if workflow == "jvm" and event in JVM_REDUCED_EVENTS:
        return {"include": _legs(JVM_LEGS)}
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
