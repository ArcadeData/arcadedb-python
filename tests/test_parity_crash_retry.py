"""The crash handling of the native parity harness (native/wheel/parity_wheel.py): record, retry once, report, stay strict.

The Windows legs of the native release run lost one test process each (ArcadeData/arcadedb-python#295), with no exit code and no trace.
The harness now records the exit code of a process that died without a result (`CRASH(rc=0xC0000005,3221225477)`), runs such a file once
more when --retry-crash is given (the workflow gives it on Windows legs only) and marks a retry that produced a result `FLAKY-CRASH`. A
failing test, a timeout, a second crash and every crash without the flag stay failures. These tests run the decision with a fake runner,
with real child processes that exit with a high code or die from a signal, and the harness itself against a small fixture repository
(real pytest children, the wheel check stubbed), so the code that ships is the code that is tested. Nothing here needs Windows: the exit
code of a crash differs by platform and the tests read it from the platform they run on.
"""

from __future__ import annotations

import importlib.util
import json
import re
import signal
import subprocess  # nosec B404 - test-controlled child processes
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WHEEL_DIR = ROOT / "native" / "wheel"
WORKFLOWS = ROOT / ".github" / "workflows"

# the harness imports parity_crash as a sibling of the script
if str(WHEEL_DIR) not in sys.path:
    sys.path.insert(0, str(WHEEL_DIR))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, WHEEL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


pc = _load("parity_crash")
parity_wheel = _load("parity_wheel")
report = _load("report_flaky_crash")

WIN = sys.platform == "win32"
# An exit code a test process can die with: the NTSTATUS of an access violation on Windows, 139 (128 + SIGSEGV) elsewhere (POSIX keeps 8 bits).
HIGH = 0xC0000005 if WIN else 139
HIGH_TEXT = "0xC0000005,3221225477" if WIN else "0x0000008B,139"
# os._exit takes a C int: the same NTSTATUS as a signed 32-bit number on Windows (the exit code stays 0xC0000005 to the parent)
HIGH_ARG = HIGH - (1 << 32) if WIN else HIGH
OTHER = 0xC0000374 if WIN else 134
OTHER_TEXT = "0xC0000374,3221226356" if WIN else "0x00000086,134"
EM_DASH = "\u2014"  # written as an escape so that this file holds none itself


def attempts(*items):
    """A fake runner: attempt n returns items[n - 1]; `calls` records which attempts ran."""
    calls = []

    def run(n):
        calls.append(n)
        return items[n - 1]

    return run, calls


RESULT = pc.Attempt(rc=0, result=True)
FAILED = pc.Attempt(rc=1, result=True)  # pytest wrote its XML, with failures in it
DIED = pc.Attempt(rc=HIGH, result=False)
DIED_AGAIN = pc.Attempt(rc=OTHER, result=False)
ON = lambda budget=3: pc.RetryPolicy(enabled=True, budget=budget)  # noqa: E731


# ---------------------------------------------------------------------------------------------------------------------------------
# the decision, with a fake runner
# ---------------------------------------------------------------------------------------------------------------------------------


def test_a_file_with_a_result_runs_once_even_when_the_retry_is_on():
    run, calls = attempts(RESULT, RESULT)
    out = pc.run_file(run, ON())
    assert calls == [1]
    assert out.status == ""
    assert (out.retried, out.flaky, out.hard_crash) == (False, False, False)


def test_a_file_with_failing_tests_is_never_retried():
    # exit code 1 with an XML is a test failure: the count of failures comes from the XML and stays strict
    run, calls = attempts(FAILED, RESULT)
    out = pc.run_file(run, ON())
    assert calls == [1]
    assert out.status == ""
    assert out.final is FAILED


def test_a_crash_without_the_retry_flag_is_a_crash_with_its_exit_code():
    # the default policy is what Linux and macOS use: strict, one run, the exit code recorded
    run, calls = attempts(DIED, RESULT)
    out = pc.run_file(run, pc.RetryPolicy())
    assert calls == [1]
    assert out.status == f"CRASH(rc={HIGH_TEXT})"
    assert out.hard_crash and not out.flaky and not out.retried


def test_a_crash_that_produces_a_result_when_run_again_is_flaky_not_clean():
    run, calls = attempts(DIED, RESULT)
    out = pc.run_file(run, ON())
    assert calls == [1, 2]
    assert out.status == f"FLAKY-CRASH(rc={HIGH_TEXT})"
    assert out.flaky and out.retried and not out.hard_crash
    assert out.final is RESULT


def test_a_second_crash_is_a_crash_with_both_exit_codes():
    run, calls = attempts(DIED, DIED_AGAIN)
    out = pc.run_file(run, ON())
    assert calls == [1, 2]
    assert out.status == f"CRASH(rc={HIGH_TEXT};retry rc={OTHER_TEXT})"
    assert out.hard_crash and not out.flaky


def test_a_timeout_and_an_unreadable_xml_are_not_crashes_and_are_not_retried():
    run, calls = attempts(pc.Attempt(rc=None, result=False, timed_out=True), RESULT)
    out = pc.run_file(run, ON())
    assert (calls, out.status, out.timed_out) == ([1], "TIMEOUT", True)
    run, calls = attempts(pc.Attempt(rc=1, result=False, bad_xml=True), RESULT)
    out = pc.run_file(run, ON())
    assert (calls, out.status) == ([1], "BADXML")


def test_a_retry_that_times_out_or_writes_a_bad_xml_fails_the_leg():
    run, _ = attempts(DIED, pc.Attempt(rc=None, result=False, timed_out=True))
    out = pc.run_file(run, ON())
    assert out.status == f"CRASH(rc={HIGH_TEXT};retry TIMEOUT)"
    assert out.timed_out
    run, _ = attempts(DIED, pc.Attempt(rc=1, result=False, bad_xml=True))
    assert pc.run_file(run, ON()).status == f"CRASH(rc={HIGH_TEXT};retry BADXML)"


def test_a_retry_that_runs_and_has_failures_is_flaky_and_the_counts_decide():
    # the status says the file crashed once; the failures of the second run are counted by the harness and fail the leg
    run, _ = attempts(DIED, FAILED)
    out = pc.run_file(run, ON())
    assert out.flaky
    assert pc.event_of("test_x.py", out, final_failed=2)["retry"] == "ran, 2 failed"


def test_the_retry_budget_of_a_leg_is_spent_by_the_files_not_the_crashes():
    policy = ON(budget=2)
    statuses = []
    for _ in range(3):
        run, calls = attempts(DIED, RESULT)
        out = pc.run_file(run, policy)
        statuses.append((out.status.split("(")[0], calls))
    # two files retried and flaky, the third crashes where the budget is spent: a wheel that crashes everywhere cannot pass on retries
    assert statuses == [("FLAKY-CRASH", [1, 2]), ("FLAKY-CRASH", [1, 2]), ("CRASH", [1])]
    assert policy.used == 2
    run, _ = attempts(DIED)
    assert pc.run_file(run, policy).status == f"CRASH(rc={HIGH_TEXT};retry budget spent)"


def test_exit_codes_are_written_as_ntstatus_and_decimal_or_as_a_signal():
    assert pc.format_rc(3221225477, windows=True) == "0xC0000005,3221225477"
    # the same NTSTATUS read as a signed number folds back to 32 bits on Windows
    assert pc.format_rc(-1073741819, windows=True) == "0xC0000005,3221225477"
    assert pc.format_rc(139, windows=False) == "0x0000008B,139"
    assert pc.format_rc(0, windows=False) == "0x00000000,0"
    assert pc.format_rc(-signal.SIGABRT, windows=False) == "-6,SIGABRT"
    assert pc.format_rc(-250, windows=False) == "-250,signal 250"
    assert pc.format_rc(None) == "none"


def test_a_crash_event_names_the_kind_and_what_the_retry_did():
    run, _ = attempts(DIED, RESULT)
    ev = pc.event_of("test_free_threading.py", pc.run_file(run, ON()), final_failed=0)
    assert ev == {
        "file": "test_free_threading.py",
        "kind": "FLAKY-CRASH",
        "first_rc": HIGH_TEXT,
        "retry_rc": "0x00000000,0",
        "retry": "passed",
        "status": f"FLAKY-CRASH(rc={HIGH_TEXT})",
    }
    run, _ = attempts(DIED, DIED_AGAIN)
    ev = pc.event_of("a.py", pc.run_file(run, ON()), final_failed=0)
    assert (ev["kind"], ev["retry"], ev["retry_rc"]) == ("CRASH", "crashed again", OTHER_TEXT)
    run, _ = attempts(DIED)
    ev = pc.event_of("a.py", pc.run_file(run, pc.RetryPolicy()), final_failed=0)
    assert (ev["kind"], ev["retry"], ev["retry_rc"]) == ("CRASH", "not retried", "")
    run, _ = attempts(RESULT)
    assert pc.event_of("a.py", pc.run_file(run, ON()), final_failed=0) is None


def test_a_result_followed_by_an_odd_exit_code_is_noticed_but_not_a_crash():
    # the junit XML is written at the end of the session; a process that dies in shutdown after it leaves a result
    assert pc.dirty_exit(pc.Attempt(rc=HIGH, result=True)) is True
    assert pc.dirty_exit(pc.Attempt(rc=1, result=True)) is False
    assert pc.dirty_exit(pc.Attempt(rc=0, result=True)) is False
    assert pc.dirty_exit(DIED) is False
    assert pc.is_crash(pc.Attempt(rc=HIGH, result=True)) is False


# ---------------------------------------------------------------------------------------------------------------------------------
# real child processes: a high exit code, a signal, and a fresh process for the retry
# ---------------------------------------------------------------------------------------------------------------------------------

EXIT_ONCE = """
import os, pathlib, sys
d, n, high = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
with open(d / "pids", "a") as fh:
    fh.write(f"{os.getpid()}\\n")
if not (d / "died").exists():
    (d / "died").write_text("1")
    os._exit(high)
(d / f"result{n}").write_text("ok")
"""

EXIT_ALWAYS = """
import os, sys
os._exit(int(sys.argv[3]))
"""

SEGV = """
import os, sys
if sys.platform == "win32":
    # a real access violation (0xC0000005); ctypes would turn one into an OSError, and raise(SIGSEGV) exits with 3
    import ctypes, faulthandler
    ctypes.windll.kernel32.SetErrorMode(0x8003)  # no crash dialog and no Windows Error Reporting for this child
    faulthandler._read_null()
else:
    import resource, signal
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # no core file for a test child
    os.kill(os.getpid(), signal.SIGSEGV)
"""


def real_runner(code, d):
    """attempt(n) runs `code` in a new python process; the process is a crash unless it wrote result<n> in directory d."""

    def run(n):
        r = subprocess.run(  # nosec B603 - fixed argument list, no shell
            [sys.executable, "-c", code, str(d), str(n), str(HIGH_ARG)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return pc.Attempt(rc=r.returncode, result=(d / f"result{n}").exists())

    return run


def test_a_real_process_that_exits_with_a_high_code_is_recorded_with_that_code(tmp_path):
    out = pc.run_file(real_runner(EXIT_ALWAYS, tmp_path), pc.RetryPolicy())
    assert out.final.rc == HIGH
    assert out.status == f"CRASH(rc={HIGH_TEXT})"


def test_a_real_process_killed_by_a_fault_is_recorded_as_one(tmp_path):
    out = pc.run_file(real_runner(SEGV, tmp_path), pc.RetryPolicy())
    if WIN:
        assert out.final.rc == 0xC0000005
        assert out.status == "CRASH(rc=0xC0000005,3221225477)"
    else:
        assert out.final.rc == -signal.SIGSEGV
        assert out.status == "CRASH(rc=-11,SIGSEGV)"


def test_the_retry_of_a_real_crash_is_a_new_process_and_a_pass_is_flaky(tmp_path):
    out = pc.run_file(real_runner(EXIT_ONCE, tmp_path), ON())
    pids = (tmp_path / "pids").read_text().split()
    assert len(pids) == 2 and pids[0] != pids[1]
    assert (tmp_path / "result1").exists() is False and (tmp_path / "result2").exists()
    assert out.status == f"FLAKY-CRASH(rc={HIGH_TEXT})"
    assert out.final.rc == 0


def test_a_real_file_that_crashes_every_time_fails_after_one_retry(tmp_path):
    out = pc.run_file(real_runner(EXIT_ALWAYS, tmp_path), ON())
    assert out.status == f"CRASH(rc={HIGH_TEXT};retry rc={HIGH_TEXT})"
    assert len(out.attempts) == 2


# ---------------------------------------------------------------------------------------------------------------------------------
# the harness against a small repository: real pytest children, the wheel check stubbed
# ---------------------------------------------------------------------------------------------------------------------------------

FIXTURE_FILES = {
    "test_ok.py": "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n",
    "test_fails.py": "def test_a():\n    assert False, 'a real failure'\n",
    "test_dies_once.py": textwrap.dedent(
        f"""
        import os, pathlib


        def test_a():
            marker = pathlib.Path(os.environ["DIES_ONCE_MARKER"])
            if not marker.exists():
                marker.write_text("1")
                os._exit({HIGH_ARG})
        """
    ),
    "test_dies_always.py": f"import os\n\n\ndef test_a():\n    os._exit({HIGH_ARG})\n",
}

IDENT = (
    "/x/site-packages/arcadedb/__init__.py 26.11.1.post2 native 26.11.1 abc1234\n"
    'META {"python": "3.14.0", "wheel_version": "26.11.1.post2", "engine_version": "26.11.1", "source_commit": "abc1234"}\n'
)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """run(*flags) -> (rows by file name, the TOTAL row, meta.json, the xml directory); the fixture repo is rebuilt for each call."""
    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        if "-I" in cmd and any("import json, sys, arcadedb" in str(c) for c in cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout=IDENT, stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(parity_wheel.subprocess, "run", run)
    # no taskset and no nice: the cores of this machine are not the harness's to choose
    monkeypatch.setattr(parity_wheel.shutil, "which", lambda name: None)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    counter = iter(range(100))

    def go(*flags):
        repo = tmp_path / f"repo{next(counter)}"
        (repo / "tests").mkdir(parents=True)
        (repo / "tests" / "conftest.py").write_text("# fixture\n")
        for name, text in FIXTURE_FILES.items():
            (repo / "tests" / name).write_text(text)
        monkeypatch.setenv("DIES_ONCE_MARKER", str(repo / "marker"))
        monkeypatch.setenv("KEY", "windows-x64")
        out = repo / "out" / "parity-wheel.tsv"
        parity_wheel.main(["--repo", str(repo), "--python", sys.executable, "--out", str(out), "--timeout", "120", *flags])
        lines = [ln.split("\t") for ln in out.read_text().splitlines()]
        header = lines[0]
        rows = {ln[0]: dict(zip(header, ln)) for ln in lines[1:]}
        meta = json.loads((repo / "out" / "parity-wheel.meta.json").read_text())
        return rows, meta, repo / "out" / "parity-wheel-xml"

    return go


def test_the_harness_is_strict_without_the_flag(harness):
    rows, meta, xmldir = harness()
    assert rows["test_ok.py"]["status"] == "" and rows["test_ok.py"]["passed"] == "2"
    assert rows["test_fails.py"]["failed"] == "1" and rows["test_fails.py"]["status"] == ""
    # both crashing files are CRASH rows with the exit code, one attempt each, no retry
    for name in ("test_dies_once.py", "test_dies_always.py"):
        assert rows[name]["status"] == f"CRASH(rc={HIGH_TEXT})", name
        assert rows[name]["attempts"] == "1", name
        assert rows[name]["exit"] == HIGH_TEXT, name
    assert rows["TOTAL"]["status"] == "timeouts=0 flaky_crash=0 crash=2"
    assert meta["retry_crash"] is False
    assert [e["kind"] for e in meta["crash_events"]] == ["CRASH", "CRASH"]
    assert not (xmldir / "test_dies_once.py.attempt1.log").exists()


def test_the_harness_retries_only_the_crashes_with_the_flag(harness):
    rows, meta, xmldir = harness("--retry-crash")
    # a clean file and a file with a failing test run once, whatever the flag says
    assert (rows["test_ok.py"]["attempts"], rows["test_ok.py"]["status"]) == ("1", "")
    assert (rows["test_fails.py"]["attempts"], rows["test_fails.py"]["status"]) == ("1", "")
    assert rows["test_fails.py"]["failed"] == "1"
    # the file that died once passes on the retry: FLAKY-CRASH with the first exit code, counted from the second run
    once = rows["test_dies_once.py"]
    assert once["status"] == f"FLAKY-CRASH(rc={HIGH_TEXT})"
    assert (once["attempts"], once["passed"], once["failed"]) == ("2", "1", "0")
    assert once["exit"] == "0x00000000,0"
    # the file that always dies is retried once and then fails the leg
    always = rows["test_dies_always.py"]
    assert always["status"] == f"CRASH(rc={HIGH_TEXT};retry rc={HIGH_TEXT})"
    assert always["attempts"] == "2"
    assert rows["TOTAL"]["status"] == "timeouts=0 flaky_crash=1 crash=1"
    # the log of the first attempt is kept beside the log of the retry
    assert (xmldir / "test_dies_once.py.attempt1.log").exists()
    assert (xmldir / "test_dies_once.py.log").exists()
    assert (xmldir / "test_dies_once.py.xml").exists()
    # the metadata names the wheel and the Python once and every crash event
    assert meta["wheel_version"] == "26.11.1.post2" and meta["python"] == "3.14.0"
    assert meta["retry_crash"] is True
    events = {e["file"]: e for e in meta["crash_events"]}
    assert set(events) == {"test_dies_once.py", "test_dies_always.py"}
    assert events["test_dies_once.py"]["kind"] == "FLAKY-CRASH"
    assert events["test_dies_once.py"]["retry"] == "passed"
    assert events["test_dies_always.py"]["retry"] == "crashed again"


def test_the_harness_retry_is_off_unless_asked_for():
    assert parity_wheel.parse_args(["--repo", "r", "--out", "o"]).retry_crash is False
    assert parity_wheel.parse_args(["--repo", "r", "--out", "o", "--retry-crash"]).retry_crash is True
    assert parity_wheel.parse_args(["--repo", "r", "--out", "o"]).max_crash_retries == 3


# ---------------------------------------------------------------------------------------------------------------------------------
# the faulthandler trace of a native run on Windows (tests/conftest.py)
# ---------------------------------------------------------------------------------------------------------------------------------

FAULT_CHILD = """
import faulthandler, os, sys

if os.name == "posix":
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # no core file for a test child
else:
    import ctypes

    ctypes.windll.kernel32.SetErrorMode(0x8003)  # no crash dialog and no Windows Error Reporting for this child

sys.path.insert(0, sys.argv[1])
from tests import conftest

trace = sys.argv[2]
os.environ[conftest.FAULT_FILE_ENV] = trace

# the JVM build ignores the variable: faulthandler stays off on Windows, as before
conftest.BACKEND = "jvm"
sys.platform = "win32"
faulthandler.enable()
conftest.pytest_configure(None)
assert not faulthandler.is_enabled(), "jvm on win32"
assert not os.path.exists(trace), "jvm on win32 opened the trace file"

# another platform keeps what it has, whatever the variable says
conftest.BACKEND = "native"
sys.platform = "linux"
faulthandler.enable()
conftest.pytest_configure(None)
assert faulthandler.is_enabled(), "native on linux"
assert not os.path.exists(trace), "native on linux opened the trace file"

# the native build on Windows without the variable: off, as before
sys.platform = "win32"
del os.environ[conftest.FAULT_FILE_ENV]
conftest.pytest_configure(None)
assert not faulthandler.is_enabled(), "native on win32 without the variable"

# the native build on Windows with the variable: on, and a fault leaves its trace in the file
os.environ[conftest.FAULT_FILE_ENV] = trace
conftest.pytest_configure(None)
assert faulthandler.is_enabled(), "native on win32 with the variable"
print("configured", flush=True)
faulthandler._read_null()  # a read of address 0: an access violation on Windows, SIGSEGV elsewhere
"""


def test_a_native_run_on_windows_keeps_a_faulthandler_trace_and_nothing_else_does(tmp_path):
    trace = tmp_path / "fault.txt"
    r = subprocess.run(  # nosec B603 - fixed argument list, no shell
        [sys.executable, "-c", FAULT_CHILD, str(ROOT), str(trace)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # every assert of the child passed (a failed one would end the child before it printed) and it then died of the fault
    assert "AssertionError" not in r.stderr, r.stderr
    assert "configured" in r.stdout, r.stderr
    assert r.returncode == (0xC0000005 if WIN else -signal.SIGSEGV), (r.returncode, r.stderr)
    text = trace.read_text()
    assert "Segmentation fault" in text or "access violation" in text
    assert "most recent call first" in text


# ---------------------------------------------------------------------------------------------------------------------------------
# the report: the job summary and one comment on issue 295
# ---------------------------------------------------------------------------------------------------------------------------------


def write_meta(root, artifact, key, python, wheel, events, retry=True):
    d = root / artifact
    d.mkdir(parents=True)
    (d / "parity-wheel.meta.json").write_text(
        json.dumps(
            {
                "key": key,
                "platform": "win32",
                "python": python,
                "wheel_version": wheel,
                "retry_crash": retry,
                "crash_events": events,
            }
        )
    )


FLAKY_EVENT = {
    "file": "test_free_threading.py",
    "kind": "FLAKY-CRASH",
    "first_rc": "0xC0000005,3221225477",
    "retry_rc": "",
    "retry": "passed",
    "status": "FLAKY-CRASH(rc=0xC0000005,3221225477)",
    "trace1": "test_free_threading.py.attempt1.fault.txt",
}
CRASH_EVENT = {
    "file": "test_vector.py",
    "kind": "CRASH",
    "first_rc": "0xC0000374,3221226356",
    "retry_rc": "0xC0000374,3221226356",
    "retry": "crashed again",
    "status": "CRASH(rc=0xC0000374,3221226356;retry rc=0xC0000374,3221226356)",
}
RELEASE_ENV = {
    "GITHUB_REPOSITORY": "ArcadeData/arcadedb-python",
    "GITHUB_RUN_ID": "42744219731",
    "GITHUB_RUN_ATTEMPT": "2",
    "GITHUB_EVENT_NAME": "push",
    "GITHUB_REF_TYPE": "tag",
    "GITHUB_REF_NAME": "26.11.1.post2",
    "GITHUB_SERVER_URL": "https://github.com",
    "GH_TOKEN": "token",
}


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "artifacts"
    write_meta(root, "native-parity-windows-x64-py3.11", "windows-x64", "3.11.9", "26.11.1.post2", [FLAKY_EVENT])
    write_meta(root, "native-parity-windows-x64-py3.12", "windows-x64", "3.12.7", "26.11.1.post2", [])
    write_meta(root, "native-parity-windows-x64-py3.14", "windows-x64", "3.14.0", "26.11.1.post2", [CRASH_EVENT])
    return root


class FakeGh:
    def __init__(self, returncode=0, stderr=""):
        self.calls, self.returncode, self.stderr = [], returncode, stderr

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, json.loads(kwargs["input"])["body"]))
        return subprocess.CompletedProcess(cmd, self.returncode, stdout="", stderr=self.stderr)


def test_the_events_of_every_leg_are_collected_with_the_identity_of_their_leg(artifacts):
    events = report.load_events(str(artifacts))
    assert [(e["file"], e["python"], e["wheel"]) for e in events] == [
        ("test_free_threading.py", "3.11.9", "26.11.1.post2"),
        ("test_vector.py", "3.14.0", "26.11.1.post2"),
    ]
    assert report.load_events(str(artifacts / "none")) == []


def test_a_release_posts_one_comment_listing_every_event_and_the_summary_has_the_table(artifacts, tmp_path, capsys):
    gh, summary = FakeGh(), tmp_path / "summary.md"
    code = report.report(report.load_events(str(artifacts)), RELEASE_ENV, str(summary), run=gh)
    assert code == 0
    assert len(gh.calls) == 1
    cmd, body = gh.calls[0]
    assert cmd[:5] == ["gh", "api", "--method", "POST", "repos/ArcadeData/arcadedb-python/issues/295/comments"]
    for text in (
        "https://github.com/ArcadeData/arcadedb-python/actions/runs/42744219731",
        "attempt 2",
        "26.11.1.post2",
        "test_free_threading.py",
        "test_vector.py",
        "FLAKY-CRASH 0xC0000005,3221225477",
        "3.11.9",
        "3.14.0",
        "native-parity-windows-x64-py3.11",
    ):
        assert text in body, text
    assert body.startswith("CRASH and FLAKY-CRASH in a native parity run.")
    assert EM_DASH not in body
    md = summary.read_text()
    assert md.startswith("### Parity crashes")
    assert "**FLAKY-CRASH: 1 file(s)" in md and "**CRASH: 1 file(s)" in md
    assert "| windows-x64 | 3.11.9 | 26.11.1.post2 | `test_free_threading.py` | FLAKY-CRASH 0xC0000005,3221225477 | passed |" in md
    assert EM_DASH not in md


def test_a_dispatch_posts_and_a_pull_request_a_branch_push_and_a_fork_do_not(artifacts):
    events = report.load_events(str(artifacts))
    posted = {}
    for name, env in {
        "dispatch": {**RELEASE_ENV, "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF_TYPE": "branch"},
        "pull_request": {**RELEASE_ENV, "GITHUB_EVENT_NAME": "pull_request", "GITHUB_REF_TYPE": "branch"},
        "branch push": {**RELEASE_ENV, "GITHUB_EVENT_NAME": "push", "GITHUB_REF_TYPE": "branch"},
        "schedule": {**RELEASE_ENV, "GITHUB_EVENT_NAME": "schedule", "GITHUB_REF_TYPE": "branch"},
        "fork": {**RELEASE_ENV, "GITHUB_REPOSITORY": "humemai/arcadedb-python"},
    }.items():
        gh = FakeGh()
        assert report.report(events, env, run=gh) == 0
        posted[name] = len(gh.calls)
    assert posted == {"dispatch": 1, "pull_request": 0, "branch push": 0, "schedule": 0, "fork": 0}


def test_a_token_that_cannot_write_issues_is_a_visible_warning_and_never_a_failure(artifacts, capsys):
    gh = FakeGh(returncode=1, stderr="gh: Resource not accessible by integration (HTTP 403)")
    assert report.report(report.load_events(str(artifacts)), RELEASE_ENV, run=gh) == 0
    out = capsys.readouterr().out
    assert len(gh.calls) == 1
    assert "::warning::the FLAKY-CRASH comment on #295 was skipped" in out
    assert "HTTP 403" in out
    # no token at all: gh is not even called
    gh = FakeGh()
    assert report.report(report.load_events(str(artifacts)), {**RELEASE_ENV, "GH_TOKEN": ""}, run=gh) == 0
    assert gh.calls == []
    assert "::warning::the FLAKY-CRASH comment on #295 was skipped: no GH_TOKEN" in capsys.readouterr().out


def test_a_run_without_a_crash_writes_nothing_and_posts_nothing(tmp_path, capsys):
    root = tmp_path / "a"
    write_meta(root, "native-parity-windows-x64-py3.12", "windows-x64", "3.12.7", "26.11.1.post2", [])
    gh, summary = FakeGh(), tmp_path / "summary.md"
    assert report.report(report.load_events(str(root)), RELEASE_ENV, str(summary), run=gh) == 0
    assert gh.calls == [] and not summary.exists()


def test_a_broken_artifact_directory_does_not_fail_the_report(tmp_path, capsys):
    (tmp_path / "a" / "x").mkdir(parents=True)
    (tmp_path / "a" / "x" / "parity-wheel.meta.json").write_text("{not json")
    assert report.main(["--artifacts", str(tmp_path / "a"), "--summary", ""]) == 0
    assert "could not be read" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------------------------------------
# the workflows: the flag on Windows legs only, and issues: write on one job only
# ---------------------------------------------------------------------------------------------------------------------------------


def jobs_of(text):
    """{job name: its text} for the jobs: block of a workflow file (a job starts at two spaces of indentation)."""
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def test_the_workflow_passes_the_retry_flag_on_windows_legs_only():
    text = (WORKFLOWS / "test-native.yml").read_text(encoding="utf-8")
    assert text.count("--retry-crash") == 2  # once in the comment of the header, once in the Windows branch
    assert re.search(r'if \[ "\$RUNNER_OS" = Windows \]; then retry=\(--retry-crash\); fi', text)
    jobs = jobs_of(text)
    assert 'retry=(--retry-crash)' in jobs["test"]
    assert '${retry[@]+"${retry[@]}"}' in jobs["test"]
    # the other jobs that run the harness never pass it
    for name, job in jobs.items():
        if name != "test":
            assert "--retry-crash" not in job, name
    # the examples, pyver and build jobs call no harness with it; the Windows-only condition is the only place it is set
    assert jobs["pyver"].count("parity_wheel.py") == 1


def test_issues_write_is_held_by_the_crash_report_job_and_the_release_call_only():
    jobs = jobs_of((WORKFLOWS / "test-native.yml").read_text(encoding="utf-8"))
    for name, job in jobs.items():
        assert ("issues: write" in job) == (name == "crash-report"), name
    report_job = jobs["crash-report"]
    assert "continue-on-error: true" in report_job
    assert "contents: read" in report_job
    assert "native/wheel/report_flaky_crash.py" in report_job
    release = jobs_of((WORKFLOWS / "release-native.yml").read_text(encoding="utf-8"))
    for name, job in release.items():
        assert ("issues: write" in job) == (name == "test"), name
    assert "uses: ./.github/workflows/test-native.yml" in release["test"]


def test_the_flaky_comment_text_has_no_em_dash_in_any_source_that_writes_it():
    for path in (WHEEL_DIR / "report_flaky_crash.py", WHEEL_DIR / "parity_crash.py", WHEEL_DIR / "parity_wheel.py"):
        assert EM_DASH not in path.read_text(encoding="utf-8"), path.name
