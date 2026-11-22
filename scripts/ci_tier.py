#!/usr/bin/env python3
"""Name the event the leg planners get, so a pull request runs the fast tier unless it asks for more.

CI has two tiers (docs page development/ci.adoc):

- fast: a pull request. Linux x86-64 only and no example legs, the legs of the event `pull_request` in scripts/select_matrix.py and
  scripts/select_native_matrix.py. A pull request that changes the examples or their runner (EXAMPLE_PATHS) also runs the example
  legs of Linux x86-64: the event `pull_request_examples`.
- full: a push to main, the nightly run, a manual run and a release, each with the legs of its own event. A pull request gets the
  legs of a push to main (the event `pull_request_full`) when it carries the label `ci:full` or when Dependabot opened it (nobody
  reads a Dependabot pull request before the hourly job merges it, so the full matrix is its only review).

usage: ci_tier.py EVENT        EVENT: the event that started the workflow

Prints EVENT unchanged unless it is `pull_request`. For a pull request it prints `pull_request`, `pull_request_examples` or
`pull_request_full`, and says why on stderr. It reads:

    PR_AUTHOR           the login of the author (github.event.pull_request.user.login)
    PR_LABELS           the label names in the event, a JSON list (toJSON(github.event.pull_request.labels.*.name))
    PR_NUMBER, GITHUB_REPOSITORY, GH_TOKEN
                        to read the labels the pull request has now and the files it changes, through `gh api`

The labels are read again from the API because a re-run keeps the event of the first attempt: with the label added afterwards,
"Re-run all jobs" is enough to get the full matrix, no new commit is needed. The choice fails open, never to less: when the labels
cannot be read the pull request runs the full tier, and when the changed files cannot be read it runs the example legs.
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
EXAMPLES_EVENT = "pull_request_examples"
FULL_EVENT = "pull_request_full"

# A pull request that changes one of these runs the example legs: a file under examples/ (the scripts, their data loader), or the
# runner both example jobs call and the list of examples the native wheel leaves out. examples/README.md is prose only.
EXAMPLE_DIRS = ("examples/",)
EXAMPLE_FILES = ("scripts/run_examples.sh", "scripts/native_examples_skip.txt")
NOT_EXAMPLES = ("examples/README.md",)


def _gh_lines(path: str, jq: str) -> list[str] | None:
    """The lines `gh api --paginate PATH --jq JQ` prints (one value on each line), or None when the call fails."""
    try:
        proc = subprocess.run(
            ["gh", "api", "--paginate", path, "--jq", jq],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return [line for line in proc.stdout.splitlines() if line]


def live_labels(repo: str, number: str) -> list[str] | None:
    """The label names the pull request has now, or None when they cannot be read."""
    if not repo or not number:
        return None
    return _gh_lines(f"repos/{repo}/issues/{number}/labels?per_page=100", ".[].name")


def changed_files(repo: str, number: str) -> list[str] | None:
    """The paths the pull request changes (for a rename: the new and the old path), or None when they cannot be read."""
    if not repo or not number:
        return None
    return _gh_lines(
        f"repos/{repo}/pulls/{number}/files?per_page=100",
        ".[] | .filename, (.previous_filename // empty)",
    )


def touches_examples(paths: list[str]) -> bool:
    """True when one of the paths is an example, its data loader or the runner of the example jobs."""
    return any(
        path not in NOT_EXAMPLES
        and (path in EXAMPLE_FILES or path.startswith(EXAMPLE_DIRS))
        for path in paths
    )


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
    read_files: Callable[[], list[str] | None],
) -> tuple[str, str]:
    """(the event for the planners, why). `read_live` and `read_files` are called only when nothing else has already decided."""
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
    files = read_files()
    if files is None:
        return (
            EXAMPLES_EVENT,
            "fast tier with the example legs: the changed files could not be read (fails open)",
        )
    if touches_examples(files):
        return (
            EXAMPLES_EVENT,
            "fast tier with the example legs: the pull request changes the examples or their runner",
        )
    return FAST_EVENT, f"fast tier: no label {FULL_LABEL}, no example changed"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: ci_tier.py EVENT", file=sys.stderr)
        return 2
    env = os.environ
    repo, number = env.get("GITHUB_REPOSITORY", ""), env.get("PR_NUMBER", "")
    event, why = planner_event(
        args[0],
        env.get("PR_AUTHOR", ""),
        payload_labels(env.get("PR_LABELS", "")),
        lambda: live_labels(repo, number),
        lambda: changed_files(repo, number),
    )
    print(why, file=sys.stderr)
    print(event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
