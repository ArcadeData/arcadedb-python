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


def _no_files():
    return ["src/arcadedb/core.py", "tests/test_core.py"]


@pytest.mark.parametrize(
    "event",
    ["push", "schedule", "workflow_dispatch", "release", "", "some_new_event"],
)
def test_every_event_but_a_pull_request_passes_through_unchanged(event):
    # the label, the author and the changed files matter for a pull request only, and no other event costs an API call
    chosen, _why = ct.planner_event(
        event, "dependabot[bot]", ["ci:full"], _never, _never
    )
    assert chosen == event


def test_a_pull_request_without_the_label_runs_the_fast_tier():
    chosen, why = ct.planner_event(
        "pull_request", "someone", ["bug"], lambda: ["bug", "native"], _no_files
    )
    assert chosen == "pull_request"
    assert "fast tier" in why


def test_the_label_in_the_event_asks_for_the_full_tier_without_an_api_call():
    chosen, _why = ct.planner_event(
        "pull_request", "someone", ["bug", "ci:full"], _never, _never
    )
    assert chosen == "pull_request_full"


def test_the_label_added_after_the_event_asks_for_the_full_tier_on_a_rerun():
    # a re-run keeps the event of the first attempt, so the labels are read again from the API
    chosen, why = ct.planner_event(
        "pull_request", "someone", ["bug"], lambda: ["bug", "ci:full"], _never
    )
    assert chosen == "pull_request_full"
    assert "after the event" in why


def test_labels_that_cannot_be_read_fail_open_to_the_full_tier():
    chosen, why = ct.planner_event("pull_request", "someone", [], lambda: None, _never)
    assert chosen == "pull_request_full"
    assert "fails open" in why


def test_a_dependabot_pull_request_runs_the_full_tier():
    # the hourly job merges it without a reader, so the full matrix is its only review
    chosen, _why = ct.planner_event(
        "pull_request", "dependabot[bot]", [], _never, _never
    )
    assert chosen == "pull_request_full"


def test_only_the_exact_label_counts():
    near = ["ci:fullish", "not ci:full", "CI:FULL", "ci"]
    chosen, _why = ct.planner_event(
        "pull_request", "someone", near, lambda: near, _no_files
    )
    assert chosen == "pull_request"


@pytest.mark.parametrize(
    "path",
    [
        "examples/03_vector_search.py",
        "examples/download_data.py",
        "examples/scripts/anything.sh",
        "scripts/run_examples.sh",
        "scripts/native_examples_skip.txt",
    ],
)
def test_a_pull_request_that_changes_the_examples_or_their_runner_runs_the_example_legs(
    path,
):
    chosen, why = ct.planner_event(
        "pull_request", "someone", [], lambda: [], lambda: ["README.md", path]
    )
    assert chosen == "pull_request_examples"
    assert "example" in why


@pytest.mark.parametrize(
    "path",
    [
        "examples/README.md",
        "src/arcadedb/examples/x.py",
        "docs/examples/x.adoc",
        "scripts/run_examples.sh.bak",
        "tests/test_examples.py",
        "examples",
    ],
)
def test_other_paths_do_not_start_the_example_legs(path):
    chosen, _why = ct.planner_event(
        "pull_request", "someone", [], lambda: [], lambda: [path]
    )
    assert chosen == "pull_request"


def test_changed_files_that_cannot_be_read_fail_open_to_the_example_legs():
    chosen, why = ct.planner_event(
        "pull_request", "someone", [], lambda: [], lambda: None
    )
    assert chosen == "pull_request_examples"
    assert "fails open" in why


def test_the_full_tier_wins_over_the_example_legs():
    # the full tier has the example legs of every platform, so the changed files are not even read
    chosen, _why = ct.planner_event(
        "pull_request", "someone", ["ci:full"], _never, _never
    )
    assert chosen == "pull_request_full"


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


def test_the_live_labels_and_the_changed_files_come_from_gh_api_one_value_on_each_line(
    monkeypatch,
):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if "/labels" in cmd[3]:
            return _Proc(0, "bug\nci:full\n")
        return _Proc(0, "examples/new_name.py\nexamples/old_name.py\nREADME.md\n")

    monkeypatch.setattr(ct.subprocess, "run", run)
    assert ct.live_labels("ArcadeData/arcadedb-python", "7") == ["bug", "ci:full"]
    assert ct.changed_files("ArcadeData/arcadedb-python", "7") == [
        "examples/new_name.py",
        "examples/old_name.py",
        "README.md",
    ]
    assert [cmd[:3] for cmd in calls] == [["gh", "api", "--paginate"]] * 2
    assert "repos/ArcadeData/arcadedb-python/issues/7/labels" in calls[0][3]
    assert "repos/ArcadeData/arcadedb-python/pulls/7/files" in calls[1][3]
    # a renamed file counts under both names: moving a file out of examples/ changes the examples too
    assert "previous_filename" in calls[1][5]


def test_the_api_answers_are_unknown_when_gh_fails_is_missing_or_has_nothing_to_ask(
    monkeypatch,
):
    def missing(cmd, **kwargs):
        raise FileNotFoundError("gh")

    def never(cmd, **kwargs):
        raise AssertionError("gh was called without a repository or a number")

    for read in (ct.live_labels, ct.changed_files):
        monkeypatch.setattr(ct.subprocess, "run", lambda cmd, **kwargs: _Proc(1, ""))
        assert read("ArcadeData/arcadedb-python", "7") is None
        monkeypatch.setattr(ct.subprocess, "run", missing)
        assert read("ArcadeData/arcadedb-python", "7") is None
        monkeypatch.setattr(ct.subprocess, "run", never)
        assert read("", "7") is None
        assert read("ArcadeData/arcadedb-python", "") is None


def test_no_label_and_no_changed_file_are_answers_not_failures(monkeypatch):
    monkeypatch.setattr(ct.subprocess, "run", lambda cmd, **kwargs: _Proc(0, ""))
    assert ct.live_labels("ArcadeData/arcadedb-python", "7") == []
    assert ct.changed_files("ArcadeData/arcadedb-python", "7") == []


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
        for event in (ct.EXAMPLES_EVENT, ct.FULL_EVENT):
            assert f'"{event}"' in text, (name, event)
    assert ct.FAST_EVENT == "pull_request"


def test_the_label_is_the_one_contributing_names():
    # CONTRIBUTING.md only: the test workflows ignore docs/**, so a test must not depend on a page there
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert f"`{ct.FULL_LABEL}`" in text
