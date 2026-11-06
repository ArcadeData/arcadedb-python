"""Which runs of the test workflows cancel each other: superseded runs of a pull request, never a release or a dispatch."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    name: ROOT / ".github" / "workflows" / f"{name}.yml"
    for name in ("test-jvm", "test-python-examples", "test-native")
}


def _concurrency(text: str) -> tuple[str, str]:
    """The group and cancel-in-progress expressions of the top-level concurrency block, `${{ }}` stripped."""
    block = text.split("\nconcurrency:\n", 1)[1].split("\n\n", 1)[0]
    group_text, cancel_text = block.split("  cancel-in-progress:", 1)
    group = group_text.split("group:", 1)[1].replace(">-", "")
    group = " ".join(line.strip() for line in group.strip().splitlines())
    cancel = cancel_text.strip()
    for value in (group, cancel):
        assert value.startswith("${{") and value.endswith("}}"), value
    return group[3:-2].strip(), cancel[3:-2].strip()


def _evaluate(expression: str, context: dict[str, str]):
    """GitHub's expression language restricted to what the blocks use: && || == != strings, format() and github.* names.

    `&&` and `||` return an operand and bind like Python's `and` and `or`, so the translation keeps the meaning.
    """

    def fmt(template: str, *values: str) -> str:
        return template.format(*values)

    translated = re.sub(
        r"github\.([a-z_.]+)", lambda m: repr(context.get(m.group(1), "")), expression
    )
    translated = translated.replace("&&", " and ").replace("||", " or ")
    # only literals, operators and format() are left; nothing else may run
    assert re.fullmatch(r"[\s\w'\"(),.=!/{}\-]*", translated), translated
    return eval(translated, {"__builtins__": {}, "format": fmt})  # nosec B307


def _run(workflow: str, event: str, ref: str, run_id: str = "1", pr: str = ""):
    group, cancel = _concurrency(WORKFLOWS[workflow].read_text(encoding="utf-8"))
    context = {
        "event_name": event,
        "ref": ref,
        "ref_type": "tag" if ref.startswith("refs/tags/") else "branch",
        "run_id": run_id,
        "event.pull_request.number": pr,
    }
    return _evaluate(group, context), bool(_evaluate(cancel, context))


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_a_new_push_to_a_pull_request_cancels_the_old_run(workflow):
    old = _run(workflow, "pull_request", "refs/pull/7/merge", "1", "7")
    new = _run(workflow, "pull_request", "refs/pull/7/merge", "2", "7")
    other = _run(workflow, "pull_request", "refs/pull/8/merge", "3", "8")
    assert old == new and new[1] is True
    assert other[0] != new[0]


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_pushes_to_main_share_one_group_so_no_backlog_builds(workflow):
    first = _run(workflow, "push", "refs/heads/main", "1")
    second = _run(workflow, "push", "refs/heads/main", "2")
    assert first[0] == second[0]
    if workflow == "test-native":
        # the native workflow keeps its older rule: a newer push to main cancels the older run
        assert first[1] is True
    else:
        # JVM and examples: one running and one pending run; a running run is never cancelled
        assert first[1] is False


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_the_nightly_run_has_its_own_group_and_replaces_a_stuck_one(workflow):
    nightly = _run(workflow, "schedule", "refs/heads/main", "1")
    push = _run(workflow, "push", "refs/heads/main", "2")
    assert nightly[0] != push[0]
    assert nightly[1] is True
    assert nightly == _run(workflow, "schedule", "refs/heads/main", "3")


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_a_dispatch_has_a_group_of_its_own_and_is_never_cancelled(workflow):
    dispatch = _run(workflow, "workflow_dispatch", "refs/heads/main", "10")
    push = _run(workflow, "push", "refs/heads/main", "11")
    again = _run(workflow, "workflow_dispatch", "refs/heads/main", "12")
    assert dispatch[1] is False
    assert len({dispatch[0], push[0], again[0]}) == 3


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_a_release_run_is_never_cancelled_and_never_shares_the_group_of_main(workflow):
    # a workflow_call from release-*.yml: github.event_name is the caller's, a push, and the ref is the tag
    release = _run(workflow, "push", "refs/tags/26.11.1", "20")
    main = _run(workflow, "push", "refs/heads/main", "21")
    assert release[1] is False
    assert release[0] != main[0]
    # a newer push to main cannot replace it
    assert _run(workflow, "push", "refs/heads/main", "22")[0] != release[0]


def test_the_release_workflows_never_cancel_a_running_release():
    for name in ("release-jvm", "release-native"):
        text = (ROOT / ".github" / "workflows" / f"{name}.yml").read_text(
            encoding="utf-8"
        )
        block = text.split("\nconcurrency:\n", 1)[1].split("\n\n", 1)[0]
        assert "cancel-in-progress: false" in block
        assert "ref_name" in block
