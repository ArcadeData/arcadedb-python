"""Which platform x Python legs CI runs per event: a small matrix on pull requests, the full one everywhere else."""

from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess  # nosec B404 - test-controlled child processes, no untrusted input
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "select_matrix.py"
WORKFLOWS = {
    "jvm": ROOT / ".github" / "workflows" / "test-jvm.yml",
    "examples": ROOT / ".github" / "workflows" / "test-python-examples.yml",
}

# The full matrix of both workflows: 4 platforms x 4 Pythons. Written out here on purpose, not
# imported from the script, so the full matrix cannot shrink unnoticed.
PLATFORMS = {
    "linux/amd64": "ubuntu-24.04",
    "linux/arm64": "ubuntu-24.04-arm",
    "darwin/arm64": "macos-15",
    "windows/amd64": "windows-2025",
}
PYTHONS = ["3.11", "3.12", "3.13", "3.14"]
FULL = {(p, v) for p, v in itertools.product(PLATFORMS, PYTHONS)}
NON_PULL_REQUEST_EVENTS = [
    "push",
    "schedule",
    "workflow_dispatch",
    "workflow_call",
    "",
    "some_new_event",
]


def _load():
    spec = importlib.util.spec_from_file_location("select_matrix", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["select_matrix"] = module
    spec.loader.exec_module(module)
    return module


sm = _load()


def _pairs(matrix):
    legs = matrix["include"]
    pairs = [(leg["platform"], leg["python-version"]) for leg in legs]
    assert len(pairs) == len(set(pairs)), "a leg is listed twice"
    for leg in legs:
        assert set(leg) == {"platform", "runs-on", "python-version"}
        assert leg["runs-on"] == PLATFORMS[leg["platform"]]
    return set(pairs)


@pytest.mark.parametrize("workflow", ["jvm", "examples"])
@pytest.mark.parametrize("event", NON_PULL_REQUEST_EVENTS)
def test_every_event_but_pull_request_runs_the_full_matrix(workflow, event):
    matrix = sm.select(workflow, event)
    assert len(matrix["include"]) == 16
    assert _pairs(matrix) == FULL


def test_pull_request_jvm_matrix():
    pairs = _pairs(sm.select("jvm", "pull_request"))
    assert {v for p, v in pairs if p == "linux/amd64"} == set(PYTHONS)
    for platform in ("linux/arm64", "darwin/arm64", "windows/amd64"):
        assert {v for p, v in pairs if p == platform} == {"3.11", "3.14"}, platform
    assert len(pairs) == 10


def test_pull_request_examples_matrix():
    pairs = _pairs(sm.select("examples", "pull_request"))
    assert pairs == {(p, "3.14") for p in PLATFORMS}


@pytest.mark.parametrize("workflow", ["jvm", "examples"])
def test_pull_request_matrix_is_part_of_the_full_matrix(workflow):
    assert _pairs(sm.select(workflow, "pull_request")) < FULL


def test_unknown_workflow_is_refused():
    with pytest.raises(ValueError):
        sm.select("docs", "pull_request")


def test_pythons_match_the_classifiers_of_pyproject():
    # release-jvm.yml expects one wheel per classifier Python and platform from the full matrix; a
    # Python added to one place and not the other would fail the release, not a pull request.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    prefix = "Programming Language :: Python :: 3."
    classifiers = [
        c.rsplit(" ", 1)[1] for c in project["classifiers"] if c.startswith(prefix)
    ]
    assert classifiers, "no Python version classifiers found in pyproject.toml"
    assert list(sm.PYTHONS) == classifiers
    # The oldest Python of the matrix is the floor of requires-python, so dropping or adding the
    # oldest one cannot leave the install metadata saying something else.
    assert project["requires-python"] == f">={sm.PYTHONS[0]}"


def test_command_line_prints_one_line_of_matrix_json():
    for event, count in (("pull_request", 10), ("push", 16)):
        out = subprocess.run(  # nosec B603 - fixed argument list, no untrusted input
            [sys.executable, "-I", str(SCRIPT_PATH), "jvm", "--event", event],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert out.count("\n") == 1
        assert len(json.loads(out)["include"]) == count


@pytest.mark.parametrize("workflow", sorted(WORKFLOWS))
def test_workflow_takes_its_matrix_from_the_script(workflow):
    text = WORKFLOWS[workflow].read_text(encoding="utf-8")
    assert f"scripts/select_matrix.py {workflow}" in text
    assert "matrix: ${{ fromJSON(needs.download-jars.outputs.matrix) }}" in text
    # the lists are in the script; a copy left in the workflow would say one thing and run another
    assert (
        "python-version:\n" not in text.split("strategy:", 1)[1].split("steps:", 1)[0]
    )
