#!/usr/bin/env python3
"""Pick the platform x Python legs a CI workflow runs for the event that started it.

The wheels support four platforms and five Python versions, which is 20 legs per workflow. A
pull request does not need all of them: the organisation shares one runner cap with
ArcadeData/arcadedb, so every extra macOS or Windows leg on a pull request delays upstream too.

- `pull_request` runs the small matrix below.
- Every other event runs the full matrix: a push to main, the nightly schedule, a manual
  dispatch, and the release workflow (which calls the workflows as `workflow_call`; the event
  name seen there is the caller's, a tag push). The release needs one wheel per platform and
  Python version, and `release.yml` counts them.

The selection fails open to the full matrix: an event name this script does not know runs
everything, never less.

usage:
    select_matrix.py bindings [--event NAME]    # test-python-bindings.yml
    select_matrix.py examples [--event NAME]    # test-python-examples.yml

`--event` defaults to $GITHUB_EVENT_NAME. The output is one line of JSON for
`strategy.matrix`: {"include": [{"platform": ..., "runs-on": ..., "python-version": ...}, ...]}.
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
PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
OLDEST = PYTHONS[0]
NEWEST = PYTHONS[-1]

# The legs of a pull request, per workflow: platform -> Python versions.
#
# bindings: all Pythons on linux/amd64 (the cheap, fast runner and the one most contributors
# use), the oldest and the newest on the three other platforms. A bug that depends on the
# interpreter shows on the ends of the range, and one that depends on the platform shows on
# any version of it.
#
# examples: the newest Python only, on every platform. The examples are slow (torch,
# sentence-transformers), and they carry platform-specific code (`os.name == "nt"` branches in
# examples 07 to 12) and per-platform steps in the workflow (a CPU torch index on macOS, UTF-8
# on Windows), so all four platforms stay.
PULL_REQUEST_LEGS = {
    "bindings": {
        "linux/amd64": PYTHONS,
        "linux/arm64": (OLDEST, NEWEST),
        "darwin/arm64": (OLDEST, NEWEST),
        "windows/amd64": (OLDEST, NEWEST),
    },
    "examples": {platform: (NEWEST,) for platform in RUNNERS},
}


def _legs(per_platform: dict[str, tuple[str, ...]]) -> list[dict[str, str]]:
    return [
        {"platform": platform, "runs-on": RUNNERS[platform], "python-version": python}
        for platform in RUNNERS
        for python in per_platform.get(platform, ())
    ]


def select(workflow: str, event: str) -> dict[str, list[dict[str, str]]]:
    """The `strategy.matrix` of `workflow` ("bindings" or "examples") for `event`."""
    if workflow not in PULL_REQUEST_LEGS:
        raise ValueError(
            f"unknown workflow {workflow!r}; use one of {sorted(PULL_REQUEST_LEGS)}"
        )
    if event == "pull_request":
        return {"include": _legs(PULL_REQUEST_LEGS[workflow])}
    return {"include": _legs({platform: PYTHONS for platform in RUNNERS})}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workflow", choices=sorted(PULL_REQUEST_LEGS))
    parser.add_argument(
        "--event",
        default=os.environ.get("GITHUB_EVENT_NAME", ""),
        help="the event that started the workflow (default: $GITHUB_EVENT_NAME)",
    )
    args = parser.parse_args(argv)
    print(json.dumps(select(args.workflow, args.event), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
