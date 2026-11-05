#!/usr/bin/env python3
"""Squash-merge open Dependabot pull requests that are safe to merge unattended.

A pull request is merged only when ALL of these hold:

* it is a single-package update whose title reads "... from X to Y" (grouped updates and
  anything else Dependabot titles differently stay open for a person);
* the update is a patch or minor one: the major component of X and Y is the same;
* the list of checks is not empty and every check is a success (a skipped or neutral check is
  fine; a failed, pending or missing one is not);
* GitHub reports the pull request as mergeable with merge state CLEAN.

The merge is pinned to the head commit that was inspected, so a branch Dependabot rebased in
the meantime fails the merge instead of landing unchecked code. With --dry-run nothing is merged.
Needs the `gh` CLI with GH_TOKEN set (contents: write, pull-requests: write).
"""

import argparse
import json
import re
import subprocess
import sys

TITLE_RE = re.compile(r"\bfrom (\S+) to (\S+)")
VERSION_RE = re.compile(r"^[v>=~^<]*(\d+)((?:\.\d+)*)")
OK_CONCLUSIONS = {"SUCCESS", "SKIPPED", "NEUTRAL"}


def update_kind(title):
    """Return "patch-or-minor", "major" or None when the title cannot be read."""
    found = TITLE_RE.search(title)
    if not found:
        return None
    old, new = (VERSION_RE.match(v) for v in found.groups())
    if not old or not new:
        return None
    return "patch-or-minor" if old.group(1) == new.group(1) else "major"


def checks_green(rollup):
    """True when the rollup is not empty and every entry succeeded or was skipped."""
    if not rollup:
        return False
    for check in rollup:
        if check.get("__typename") == "StatusContext":
            if check.get("state") != "SUCCESS":
                return False
        elif (
            check.get("status") != "COMPLETED"
            or check.get("conclusion") not in OK_CONCLUSIONS
        ):
            return False
    return True


def gh(*args):
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="report decisions, merge nothing"
    )
    args = parser.parse_args()

    fields = "number,title,headRefOid,mergeable,mergeStateStatus,statusCheckRollup"
    prs = json.loads(
        gh(
            "pr",
            "list",
            "--state",
            "open",
            "--base",
            "main",
            "--author",
            "app/dependabot",
            "--limit",
            "100",
            "--json",
            fields,
        )
    )
    failed = False
    for pr in sorted(prs, key=lambda p: p["number"]):
        number, title = pr["number"], pr["title"]
        kind = update_kind(title)
        if kind != "patch-or-minor":
            print(
                f"#{number}: left open ({kind or 'not a single-package update'}): {title}"
            )
        elif not checks_green(pr["statusCheckRollup"]):
            print(f"#{number}: left open (checks not all green yet): {title}")
        elif pr["mergeable"] != "MERGEABLE" or pr["mergeStateStatus"] != "CLEAN":
            print(
                f"#{number}: left open ({pr['mergeable']}, {pr['mergeStateStatus']}): {title}"
            )
        elif args.dry_run:
            print(f"#{number}: would merge: {title}")
        else:
            try:
                gh(
                    "pr",
                    "merge",
                    str(number),
                    "--squash",
                    "--delete-branch",
                    "--match-head-commit",
                    pr["headRefOid"],
                )
                print(f"#{number}: merged: {title}")
            except subprocess.CalledProcessError as err:
                failed = True
                print(f"#{number}: merge failed: {err.stderr.strip()}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
