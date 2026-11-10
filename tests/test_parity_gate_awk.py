"""The gate of the native parity legs lets a FLAKY-CRASH row pass, says so, and fails every other status.

test-native.yml reads parity-wheel.tsv with an awk program in the step "Gate: no failure, error, crash, timeout or uncounted skip" (the
`test` job and the `pyver` job carry the same program). These tests cut that program out of the workflow file and run it with the awk
of the machine on small tables, so the code that ships is the code that is tested. A retried crash (FLAKY-CRASH) must not block a
release, and it must be impossible to miss; a crash that stayed a crash, a timeout, a failing test and a skip must still fail the leg.
The file is not collected on Windows (tests/conftest.py WINDOWS_NOT_APPLICABLE): the gate runs in the bash of a runner, as in
test_release_preflight.py.
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec B404 - test-controlled child processes
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "test-native.yml"
)
HEADER = "file\tpassed\tfailed\terror\tnotyet\tna_jvm\tskipped\tstatus\tfailing_tests\tattempts\texit\n"


def gate_programs():
    """The awk program of every gate step in the workflow, by job."""
    text = WORKFLOW.read_text(encoding="utf-8")
    found = re.findall(r"awk -F'\\t' '\n(.*?)' \"\$tsv\"", text, flags=re.S)
    assert len(found) == 2, "the test job and the pyver job each carry one gate"
    return found


def row(name, passed=3, failed=0, error=0, status="", attempts=1, exit_="0x00000000,0"):
    return f"{name}\t{passed}\t{failed}\t{error}\t0\t0\t0\t{status}\t\t{attempts}\t{exit_}\n"


def total(
    passed=10, failed=0, error=0, skipped=0, status="timeouts=0 flaky_crash=0 crash=0"
):
    return f"TOTAL\t{passed}\t{failed}\t{error}\t0\t0\t{skipped}\t{status}\t\t\t\n"


def run_gate(program, tmp_path, *rows):
    tsv = tmp_path / "parity-wheel.tsv"
    tsv.write_text(HEADER + "".join(rows))
    return subprocess.run(  # nosec B603 B607 - fixed argument list, no shell
        [
            "awk",
            "-F\\t",
            program,
            str(tsv),
        ],  # -F'\t' of the workflow: awk reads the backslash-t itself
        capture_output=True,
        text=True,
        env={**os.environ, "KEY": "windows-x64", "PY": "3.14"},
        timeout=60,
    )


@pytest.fixture(params=[0, 1], ids=["test-job", "pyver-job"])
def program(request):
    return gate_programs()[request.param]


def test_a_clean_table_passes_and_prints_no_flaky_line(program, tmp_path):
    r = run_gate(program, tmp_path, row("test_a.py"), total())
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("parity windows-x64 py3.14: passed=10 failed=0 error=0")
    assert "FLAKY-CRASH" not in r.stdout


def test_a_file_that_crashed_and_passed_on_its_retry_passes_the_gate_loudly(
    program, tmp_path
):
    status = "FLAKY-CRASH(rc=0xC0000005,3221225477)"
    r = run_gate(
        program,
        tmp_path,
        row("test_a.py"),
        row("test_free_threading.py", status=status, attempts=2),
        total(),
    )
    assert r.returncode == 0, r.stderr
    assert (
        f"FLAKY-CRASH (died without a result, passed on the one retry; first exit code in brackets): test_free_threading.py({status})"
        in r.stdout
    )


def test_a_crash_that_stayed_a_crash_fails_the_gate(program, tmp_path):
    for status in (
        "CRASH(rc=0xC0000005,3221225477)",
        "CRASH(rc=0xC0000005,3221225477;retry rc=0xC0000005,3221225477)",
        "CRASH(rc=0xC0000005,3221225477;retry budget spent)",
    ):
        r = run_gate(
            program, tmp_path, row("test_a.py", passed=0, status=status), total()
        )
        assert r.returncode == 1, status
        assert f"test_a.py({status})" in r.stderr, status


def test_a_flaky_file_does_not_hide_another_files_crash(program, tmp_path):
    r = run_gate(
        program,
        tmp_path,
        row(
            "test_flaky.py", status="FLAKY-CRASH(rc=0xC0000005,3221225477)", attempts=2
        ),
        row(
            "test_dead.py",
            passed=0,
            status="CRASH(rc=0xC0000374,3221226356;retry rc=0xC0000374,3221226356)",
            attempts=2,
        ),
        total(),
    )
    assert r.returncode == 1
    assert "test_dead.py(CRASH" in r.stderr and "test_flaky.py" not in r.stderr


def test_a_flaky_file_whose_retry_had_a_failing_test_fails_the_gate(program, tmp_path):
    # the status passes, the counts of the second run decide
    r = run_gate(
        program,
        tmp_path,
        row(
            "test_flaky.py",
            failed=1,
            status="FLAKY-CRASH(rc=0xC0000005,3221225477)",
            attempts=2,
        ),
        total(failed=1),
    )
    assert r.returncode == 1


def test_a_timeout_an_unreadable_xml_a_failure_and_a_skip_still_fail_the_gate(
    program, tmp_path
):
    assert (
        run_gate(
            program, tmp_path, row("test_a.py", status="TIMEOUT"), total()
        ).returncode
        == 1
    )
    assert (
        run_gate(
            program, tmp_path, row("test_a.py", status="BADXML"), total()
        ).returncode
        == 1
    )
    assert (
        run_gate(
            program, tmp_path, row("test_a.py", failed=1), total(failed=1)
        ).returncode
        == 1
    )
    assert (
        run_gate(program, tmp_path, row("test_a.py"), total(skipped=1)).returncode == 1
    )
    assert run_gate(program, tmp_path, row("test_a.py")).returncode == 1  # no TOTAL row


def test_a_file_that_died_after_its_result_does_not_fail_the_gate(program, tmp_path):
    # DIRTY-EXIT is a record in the metadata, not a status: the row is clean and only its exit column differs
    r = run_gate(program, tmp_path, row("test_a.py", exit_="0x0000008B,139"), total())
    assert r.returncode == 0, r.stderr
    assert "FLAKY-CRASH" not in r.stdout
