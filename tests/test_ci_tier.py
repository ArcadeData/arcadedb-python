"""The tier of a CI run (scripts/ci_tier.py): a pull request runs the fast tier unless it asks for more."""

from __future__ import annotations

import importlib.util
import os
import subprocess  # nosec B404 - runs this repository's script with sys.executable
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci_tier.py"
spec = importlib.util.spec_from_file_location("ci_tier", SCRIPT)
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)


def _never():
    raise AssertionError("the API was asked")


@pytest.mark.parametrize(
    "event",
    ["push", "schedule", "workflow_dispatch", "release", "", "some_new_event"],
)
def test_every_event_but_a_pull_request_passes_through_unchanged(event):
    # the label and the author matter for a pull request only, and no other event costs an API call
    chosen, _why = ct.planner_event(event, "dependabot[bot]", ["ci:full"], _never)
    assert chosen == event


def test_a_pull_request_without_the_label_runs_the_fast_tier():
    chosen, why = ct.planner_event(
        "pull_request", "someone", ["bug"], lambda: ["bug", "native"]
    )
    assert chosen == "pull_request"
    assert "fast tier" in why


def test_the_label_in_the_event_asks_for_the_full_tier_without_an_api_call():
    chosen, _why = ct.planner_event(
        "pull_request", "someone", ["bug", "ci:full"], _never
    )
    assert chosen == "pull_request_full"


def test_the_label_added_after_the_event_asks_for_the_full_tier_on_a_rerun():
    # a re-run keeps the event of the first attempt, so the labels are read again from the API
    chosen, why = ct.planner_event(
        "pull_request", "someone", ["bug"], lambda: ["bug", "ci:full"]
    )
    assert chosen == "pull_request_full"
    assert "after the event" in why


def test_labels_that_cannot_be_read_fail_open_to_the_full_tier():
    chosen, why = ct.planner_event("pull_request", "someone", [], lambda: None)
    assert chosen == "pull_request_full"
    assert "fails open" in why


def test_a_dependabot_pull_request_runs_the_full_tier():
    # the hourly job merges it without a reader, so the full matrix is its only review
    chosen, _why = ct.planner_event("pull_request", "dependabot[bot]", [], _never)
    assert chosen == "pull_request_full"


def test_only_the_exact_label_counts():
    near = ["ci:fullish", "not ci:full", "CI:FULL", "ci"]
    chosen, _why = ct.planner_event("pull_request", "someone", near, lambda: near)
    assert chosen == "pull_request"


def test_a_change_to_the_examples_does_not_leave_the_fast_tier():
    # the examples are the slowest jobs: a pull request that changes them runs them on main, or before the merge with the label.
    # The tier does not read the changed files at all.
    assert not hasattr(ct, "changed_files")
    chosen, _why = ct.planner_event("pull_request", "someone", [], lambda: [])
    assert chosen == "pull_request"


@pytest.mark.parametrize(
    ("text", "labels"),
    [
        ('["bug","ci:full"]', ["bug", "ci:full"]),
        ("[]", []),
        ("", []),
        ("null", []),
        ("not json", []),
        ('{"ci:full": 1}', []),
        ('[1, "x", null]', ["x"]),
    ],
)
def test_the_labels_of_the_event_are_a_json_list_and_anything_else_is_no_label(
    text, labels
):
    assert ct.payload_labels(text) == labels


class _Proc:
    def __init__(self, returncode, stdout):
        self.returncode = returncode
        self.stdout = stdout


def test_the_live_labels_come_from_gh_api_one_value_on_each_line(
    monkeypatch,
):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return _Proc(0, "bug\nci:full\n")

    monkeypatch.setattr(ct.subprocess, "run", run)
    assert ct.live_labels("ArcadeData/arcadedb-python", "7") == ["bug", "ci:full"]
    assert [cmd[:3] for cmd in calls] == [["gh", "api", "--paginate"]]
    assert "repos/ArcadeData/arcadedb-python/issues/7/labels" in calls[0][3]


def test_the_api_answers_are_unknown_when_gh_fails_is_missing_or_has_nothing_to_ask(
    monkeypatch,
):
    def missing(cmd, **kwargs):
        raise FileNotFoundError("gh")

    def never(cmd, **kwargs):
        raise AssertionError("gh was called without a repository or a number")

    monkeypatch.setattr(ct.subprocess, "run", lambda cmd, **kwargs: _Proc(1, ""))
    assert ct.live_labels("ArcadeData/arcadedb-python", "7") is None
    monkeypatch.setattr(ct.subprocess, "run", missing)
    assert ct.live_labels("ArcadeData/arcadedb-python", "7") is None
    monkeypatch.setattr(ct.subprocess, "run", never)
    assert ct.live_labels("", "7") is None
    assert ct.live_labels("ArcadeData/arcadedb-python", "") is None


def test_no_label_is_an_answer_not_a_failure(monkeypatch):
    monkeypatch.setattr(ct.subprocess, "run", lambda cmd, **kwargs: _Proc(0, ""))
    assert ct.live_labels("ArcadeData/arcadedb-python", "7") == []


def _run(args, **env):
    clean = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PR_AUTHOR", "PR_LABELS", "PR_NUMBER", "GITHUB_REPOSITORY")
    }
    return subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-I", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**clean, **env},
    )


def test_the_command_line_prints_the_event_on_one_line_and_the_reason_on_stderr():
    # neither call reaches the API: the event is not a pull request, or the label is already in the event
    proc = _run(["push"])
    assert (proc.returncode, proc.stdout) == (0, "push\n")
    proc = _run(["pull_request"], PR_AUTHOR="someone", PR_LABELS='["ci:full"]')
    assert (proc.returncode, proc.stdout) == (0, "pull_request_full\n")
    assert "ci:full" in proc.stderr
    # no number and no repository: the labels cannot be read, so the pull request runs the full tier
    proc = _run(["pull_request"], PR_AUTHOR="someone", PR_LABELS="[]")
    assert (proc.returncode, proc.stdout) == (0, "pull_request_full\n")
    assert _run([]).returncode == 2


def test_both_planners_know_the_events_this_script_prints():
    # the names are written in three scripts; a rename in one of them would send a pull request to the wrong legs
    for name in ("select_matrix.py", "select_native_matrix.py"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert f'"{ct.FULL_EVENT}"' in text, name
    assert ct.FAST_EVENT == "pull_request"


def test_the_label_is_the_one_contributing_names():
    # CONTRIBUTING.md only: the test workflows ignore docs/**, so a test must not depend on a page there
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert f"`{ct.FULL_LABEL}`" in text
