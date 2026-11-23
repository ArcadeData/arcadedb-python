"""The rerun of the known engine race (ArcadeData/arcadedb-python#409) is bounded, visible, counted and narrow.

tests/test_vector_read_your_writes.py runs each of its tests again, at most twice, when the body failed with one of the three messages of the race
in the commit-to-search path of engine 26.11.1 (ArcadeData/arcadedb#12087 is the fix). The mechanism is `run_with_known_race_rerun` in
tests/conftest.py. These are plain functions on fake bodies: no engine is started and no database is created. The tests of the gate's awk program
with a `flaky_race` count are in tests/test_parity_gate_awk.py, the file that already runs that program.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import json
import os
import subprocess  # nosec B404 - a test-controlled child process
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

import arcadedb
from tests import conftest
from tests.conftest import (
    KNOWN_RACE_ISSUE,
    KNOWN_RACE_PROPERTY,
    KNOWN_RACE_RERUNS,
    is_known_engine_race,
    run_with_known_race_rerun,
)

ROOT = Path(__file__).resolve().parents[1]
WHEEL_DIR = ROOT / "native" / "wheel"
if str(WHEEL_DIR) not in sys.path:
    sys.path.insert(
        0, str(WHEEL_DIR)
    )  # the harness imports parity_crash as a sibling of the script


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, WHEEL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parity_wheel = _load("parity_wheel")
report = _load("report_flaky_crash")

MISS = "1 of 200 committed vectors were not found by the next search: [(396, [162, 169, 427, 165, 454])]"
OTHER_TASK = "3 of 200 committed vectors were not found by another task: [(401, [1])]"
SEARCH_FAILED = "Vector search failed: com.arcadedb.exception.RecordNotFoundException: Record #1:489 not found"


def fake_request():
    """What the wrapper needs of pytest's `request`: a node with a list of user properties."""
    return SimpleNamespace(node=SimpleNamespace(user_properties=[]))


def body_failing(times, error):
    """A fake test body: attempt n raises `error()` for n <= times and returns after that; `calls` records the attempts."""
    calls = []

    def attempt(n):
        calls.append(n)
        if n <= times:
            raise error()

    return attempt, calls


def run(attempt, request=None):
    """(reruns used or the exception raised, the UserWarnings that the run issued)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = run_with_known_race_rerun(attempt, request, label="test_x")
        except Exception as exc:  # noqa: BLE001 - the tests below read it
            result = exc
    return result, [w for w in caught if issubclass(w.category, UserWarning)]


def test_a_body_that_passes_is_called_once_with_no_warning_and_no_count():
    attempt, calls = body_failing(0, AssertionError)
    request = fake_request()
    result, warned = run(attempt, request)
    assert (result, calls, warned) == (0, [1], [])
    assert request.node.user_properties == []


def test_a_body_that_fails_once_with_a_signature_and_then_passes_is_a_pass_with_one_warning_and_a_count():
    attempt, calls = body_failing(1, lambda: AssertionError(MISS))
    request = fake_request()
    result, warned = run(attempt, request)
    assert (result, calls) == (1, [1, 2])
    assert len(warned) == 1
    text = str(warned[0].message)
    assert KNOWN_RACE_ISSUE in text and "run 1 of 3" in text and "rerun 1 of 2" in text
    assert "1 of 200 committed vectors were not found" in text
    # the count the harness reads from the junit record
    assert request.node.user_properties == [(KNOWN_RACE_PROPERTY, 1)]


def test_a_body_that_fails_twice_and_passes_on_the_last_run_is_a_pass_with_two_warnings():
    attempt, calls = body_failing(2, lambda: arcadedb.ArcadeDBError(SEARCH_FAILED))
    request = fake_request()
    result, warned = run(attempt, request)
    assert (result, calls) == (2, [1, 2, 3])
    assert "rerun 1 of 2" in str(warned[0].message)
    assert "rerun 2 of 2" in str(warned[1].message)
    assert request.node.user_properties == [(KNOWN_RACE_PROPERTY, 2)]


def test_a_body_that_fails_on_every_run_fails_with_two_warnings_and_no_count():
    attempt, calls = body_failing(99, lambda: AssertionError(MISS))
    request = fake_request()
    result, warned = run(attempt, request)
    assert isinstance(result, AssertionError) and MISS in str(result)
    assert calls == [1, 2, 3] == list(range(1, 2 + KNOWN_RACE_RERUNS))
    assert len(warned) == 2
    assert request.node.user_properties == []
    assert any(KNOWN_RACE_ISSUE in note for note in result.__notes__)


def test_a_failure_with_another_message_is_not_retried():
    for error in (
        lambda: AssertionError("assert 499 == 500"),
        lambda: AssertionError("a vector was not found"),
        lambda: arcadedb.ArcadeDBError("Database is closed"),
        lambda: arcadedb.ArcadeDBError("Record not found"),
        lambda: ValueError("Vector search failed"),
        lambda: KeyError(MISS),
        lambda: RuntimeError(SEARCH_FAILED),
    ):
        attempt, calls = body_failing(99, error)
        request = fake_request()
        result, warned = run(attempt, request)
        assert isinstance(result, Exception), error
        assert (calls, warned, request.node.user_properties) == ([1], [], []), error


def test_a_second_failure_of_another_kind_ends_the_reruns_at_once():
    kinds = iter(
        [lambda: AssertionError(MISS), lambda: AssertionError("assert 1 == 2")]
    )

    def attempt(n):
        calls.append(n)
        raise next(kinds)()

    calls = []
    result, warned = run(attempt, fake_request())
    assert isinstance(result, AssertionError) and str(result) == "assert 1 == 2"
    assert (calls, len(warned)) == ([1, 2], 1)


def test_only_the_three_signatures_match():
    race = [
        AssertionError(MISS),
        AssertionError(OTHER_TASK),
        AssertionError(
            MISS + "\nassert [(396, [1])] == []\n  Left contains one more item"
        ),
        arcadedb.ArcadeDBError(SEARCH_FAILED),
        arcadedb.ArcadeDBError("Vector search failed"),
        arcadedb.ArcadeDBError("Record #1:446 not found"),
    ]
    not_race = [
        AssertionError("assert 499 == 500"),
        AssertionError("Record #1:446 not found"),  # the class is part of the signature
        AssertionError("Vector search failed"),
        arcadedb.ArcadeDBError("committed vectors were not found"),
        arcadedb.ArcadeDBError("Vector search"),
        arcadedb.ArcadeDBError("Record #x not found"),
        arcadedb.ArcadeDBError("Database is closed"),
        RuntimeError("Vector search failed"),
        TimeoutError("Record #1:446 not found"),
    ]
    assert [is_known_engine_race(e) for e in race] == [True] * len(race)
    assert [is_known_engine_race(e) for e in not_race] == [False] * len(not_race)
    # exactly three signatures, no more
    assert len(conftest.KNOWN_RACE_SIGNATURES) == 3


def test_the_number_of_reruns_is_two_and_no_environment_variable_raises_it():
    assert KNOWN_RACE_RERUNS == 2
    names = [
        "ARCADEDB_KNOWN_RACE_RERUNS",
        "KNOWN_RACE_RERUNS",
        "ARCADEDB_TEST_RERUNS",
        "ARCADEDB_RERUNS",
        "RERUNS",
        "ARCADEDB_FLAKY_RERUNS",
    ]
    saved = {n: os.environ.get(n) for n in names}
    try:
        for n in names:
            os.environ[n] = "50"
        attempt, calls = body_failing(99, lambda: AssertionError(MISS))
        result, warned = run(attempt)
        assert isinstance(result, AssertionError) and calls == [1, 2, 3]
        assert len(warned) == 2
    finally:
        for n, value in saved.items():
            if value is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = value
    # and the code that decides reads no environment at all
    for fn in (run_with_known_race_rerun, is_known_engine_race):
        source = inspect.getsource(fn)
        assert "environ" not in source and "getenv" not in source, fn.__name__


def test_only_the_vector_read_your_writes_file_uses_the_rerun():
    users = sorted(
        p.name
        for p in (ROOT / "tests").glob("*.py")
        if "run_with_known_race_rerun" in p.read_text(encoding="utf-8")
    )
    assert users == [
        "conftest.py",
        "test_known_engine_race_rerun.py",
        "test_vector_read_your_writes.py",
    ]


def test_every_test_of_the_vector_read_your_writes_file_goes_through_the_rerun():
    path = ROOT / "tests" / "test_vector_read_your_writes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tests = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert len(tests) == 4
    for node in tests:
        names = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        assert names == ["known_race_test"], node.name


class FakeDb:
    """Just enough database for the corpus of tests/test_vector_read_your_writes.py: it records where it was created and whether it was dropped."""

    created = []

    def __init__(self, path):
        self.path = path
        self.dropped = False
        FakeDb.created.append(self)

    def command(self, *args):
        pass

    def transaction(self):
        return contextlib.nullcontext()

    def create_vector_index(self, *args, **kwargs):
        return object()

    def drop(self):
        self.dropped = True


def test_each_run_of_a_vector_test_gets_a_new_database_and_every_database_is_dropped(
    tmp_path,
):
    from tests import test_vector_read_your_writes as vec

    seen = []

    @vec.known_race_test
    def test_x(corpus):
        db, _index = corpus
        seen.append(db)
        if len(seen) < 3:
            raise AssertionError(MISS)

    real = vec.arcadedb.create_database
    FakeDb.created = []
    vec.arcadedb.create_database = lambda path: FakeDb(path)
    try:
        request = fake_request()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            test_x(tmp_path, request)
    finally:
        vec.arcadedb.create_database = real
    assert seen == FakeDb.created and len(seen) == 3
    assert len({db.path for db in seen}) == 3  # three directories, never a reused one
    assert all(db.dropped for db in seen)
    assert request.node.user_properties == [(KNOWN_RACE_PROPERTY, 2)]
    # pytest sees the wrapper's own fixtures, not the body's `corpus`
    assert list(inspect.signature(test_x).parameters) == ["tmp_path", "request"]
    assert test_x.__name__ == "test_x"


# ---------------------------------------------------------------------------------------------------------------------------------
# the count: the junit property, the TOTAL line, the metadata and the job summary
# ---------------------------------------------------------------------------------------------------------------------------------

XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="0" tests="4">
<testcase classname="tests.test_vector_read_your_writes" name="test_a" time="1.0"><properties><property name="flaky_race" value="1"/></properties></testcase>
<testcase classname="tests.test_vector_read_your_writes" name="test_b" time="1.0"><properties><property name="flaky_race" value="2"/></properties></testcase>
<testcase classname="tests.test_vector_read_your_writes" name="test_c" time="1.0"/>
<testcase classname="tests.test_vector_read_your_writes" name="test_d" time="1.0"><properties><property name="flaky_race" value="1"/></properties><failure message="x">x</failure></testcase>
</testsuite></testsuites>
"""


def test_the_harness_counts_the_tests_that_passed_after_a_rerun_and_not_the_failed_ones(
    tmp_path,
):
    xml = tmp_path / "test_vector_read_your_writes.py.xml"
    xml.write_text(XML)
    found = parity_wheel.race_events(str(xml), "test_vector_read_your_writes.py")
    assert [(e["test"], e["reruns"]) for e in found] == [
        ("test_a", "1"),
        ("test_b", "2"),
    ]
    assert {e["file"] for e in found} == {"test_vector_read_your_writes.py"}
    xml.write_text(
        "<testsuites>"
    )  # an unreadable XML is BADXML for the harness, never a count
    assert parity_wheel.race_events(str(xml), "x.py") == []


@pytest.mark.parametrize("family", ["xunit1", "xunit2"])
def test_a_real_junit_record_carries_the_property_in_both_families(tmp_path, family):
    inner = tmp_path / "test_inner.py"
    inner.write_text(
        "def test_a(request):\n"
        f"    request.node.user_properties.append(({KNOWN_RACE_PROPERTY!r}, 2))\n\n\n"
        "def test_b():\n    pass\n"
    )
    xml = tmp_path / "inner.xml"
    r = subprocess.run(  # nosec B603 - fixed argument list, no shell
        [
            sys.executable,
            "-m",
            "pytest",
            str(inner),
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml}",
            "-o",
            f"junit_family={family}",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    found = parity_wheel.race_events(str(xml), "test_inner.py")
    assert [(e["test"], e["reruns"]) for e in found] == [("test_a", "2")]


def totals(**counts):
    tot = dict.fromkeys(
        [
            "passed",
            "failed",
            "error",
            "notyet",
            "na_jvm",
            "skipped",
            "timeout",
            "flaky_crash",
            "crash",
            "flaky_race",
        ],
        0,
    )
    tot["passed"] = 10
    tot.update(counts)
    return tot


def test_the_total_line_carries_flaky_race_after_the_crash_counts():
    zero = parity_wheel.total_row(totals())
    two = parity_wheel.total_row(totals(flaky_race=2, flaky_crash=1))
    assert (
        zero
        == "TOTAL\t10\t0\t0\t0\t0\t0\ttimeouts=0 flaky_crash=0 crash=0 flaky_race=0\t\t\t\n"
    )
    assert (
        two
        == "TOTAL\t10\t0\t0\t0\t0\t0\ttimeouts=0 flaky_crash=1 crash=0 flaky_race=2\t\t\t\n"
    )
    # still the 11 columns of the header, so every reader of the table keeps its column numbers
    assert len(zero.rstrip("\n").split("\t")) == 11 == len(two.rstrip("\n").split("\t"))


def write_race_meta(root, artifact, key, races):
    d = root / artifact
    d.mkdir(parents=True)
    (d / "parity-wheel.meta.json").write_text(
        json.dumps(
            {
                "key": key,
                "python": "3.12.7",
                "wheel_version": "26.12.1.dev0",
                "crash_events": [],
                "race_events": races,
            }
        )
    )


RACE = {"file": "test_vector_read_your_writes.py", "test": "test_a_task", "reruns": "1"}


def test_the_summary_lists_the_race_tests_posts_nothing_and_a_run_without_any_stays_quiet(
    tmp_path, capsys
):
    root = tmp_path / "artifacts"
    write_race_meta(root, "native-parity-macos-arm64-py3.12", "macos-arm64", [RACE])
    write_race_meta(root, "native-parity-linux-x86_64-py3.12", "linux-x86_64", [])
    races = report.load_race_events(str(root))
    assert [(r["leg"], r["test"]) for r in races] == [("macos-arm64", "test_a_task")]
    summary = tmp_path / "summary.md"
    posted = []
    rc = report.report(
        [],
        {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REPOSITORY": report.REPO},
        str(summary),
        False,
        run=lambda *a, **k: posted.append(a),
        races=races,
    )
    text = summary.read_text()
    assert rc == 0 and posted == []
    assert "FLAKY-RACE: 1 test(s)" in text and "`test_a_task`" in text
    assert "no crash event in this run" in capsys.readouterr().out
    # no race and no crash: nothing is written
    quiet = tmp_path / "quiet.md"
    assert report.report([], {}, str(quiet), races=()) == 0
    assert not quiet.exists()
    assert report.race_md([]) == ""


def test_no_text_of_the_rerun_has_an_em_dash():
    dash = "—"  # an escape, so that this file holds none itself
    for source in (
        inspect.getsource(run_with_known_race_rerun),
        inspect.getsource(report.race_md),
        (ROOT / "tests" / "test_vector_read_your_writes.py").read_text(
            encoding="utf-8"
        ),
    ):
        assert dash not in source
