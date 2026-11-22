#!/usr/bin/env python3
"""Name the event the leg planners get, so a pull request runs the fast tier unless it asks for the full one.

CI has two tiers (docs page development/ci.adoc):

- fast: a pull request. Linux x86-64 only, the legs of the event `pull_request` in scripts/select_matrix.py and
  scripts/select_native_matrix.py.
- full: a push to main, the nightly run, a manual run and a release, each with the legs of its own event. A pull request gets the
  legs of a push to main when it carries the label `ci:full` or when Dependabot opened it (nobody reads a Dependabot pull request
  before the hourly job merges it, so the full matrix is its only review).

usage: ci_tier.py EVENT        EVENT: the event that started the workflow

Prints EVENT unchanged unless it is `pull_request`. For a pull request it prints `pull_request` (fast) or `pull_request_full`
(the legs of a push to main), and says why on stderr. It reads:

    PR_AUTHOR           the login of the author (github.event.pull_request.user.login)
    PR_LABELS           the label names in the event, a JSON list (toJSON(github.event.pull_request.labels.*.name))
    PR_NUMBER, GITHUB_REPOSITORY, GH_TOKEN
                        to read the labels the pull request has now, through `gh api`

The labels are read again from the API because a re-run keeps the event of the first attempt: with the label added afterwards,
"Re-run all jobs" is enough to get the full matrix, no new commit is needed. The choice fails open: when the labels cannot be read,
the pull request runs the full tier, never less.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable

FULL_LABEL = "ci:full"
# Authors whose pull requests always run the full tier.
FULL_AUTHORS = ("dependabot[bot]",)

FAST_EVENT = "pull_request"
FULL_EVENT = "pull_request_full"


def live_labels(repo: str, number: str) -> list[str] | None:
    """The label names the pull request has now, or None when they cannot be read."""
    if not repo or not number:
        return None
    try:
        proc = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/{number}/labels?per_page=100",
                "--jq",
                ".[].name",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()


def payload_labels(text: str) -> list[str]:
    """The label names of the event from PR_LABELS; anything that is not a JSON list of strings counts as no label."""
    try:
        value = json.loads(text or "[]")
    except ValueError:
        return []
    if not isinstance(value, list):
        return []
    return [name for name in value if isinstance(name, str)]


def planner_event(
    event: str,
    author: str,
    labels: list[str],
    read_live: Callable[[], list[str] | None],
) -> tuple[str, str]:
    """(the event for the planners, why). `read_live` is called only when nothing else has already decided."""
    if event != FAST_EVENT:
        return (
            event,
            f"{event or 'no event'}: not a pull request, the legs of the event",
        )
    if author in FULL_AUTHORS:
        return FULL_EVENT, f"full tier: opened by {author}"
    if FULL_LABEL in labels:
        return FULL_EVENT, f"full tier: the label {FULL_LABEL} is in the event"
    live = read_live()
    if live is None:
        return (
            FULL_EVENT,
            "full tier: the labels of the pull request could not be read (fails open)",
        )
    if FULL_LABEL in live:
        return (
            FULL_EVENT,
            f"full tier: the label {FULL_LABEL} was added after the event",
        )
    return FAST_EVENT, f"fast tier: no label {FULL_LABEL}"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: ci_tier.py EVENT", file=sys.stderr)
        return 2
    env = os.environ
    event, why = planner_event(
        args[0],
        env.get("PR_AUTHOR", ""),
        payload_labels(env.get("PR_LABELS", "")),
        lambda: live_labels(env.get("GITHUB_REPOSITORY", ""), env.get("PR_NUMBER", "")),
    )
    print(why, file=sys.stderr)
    print(event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
