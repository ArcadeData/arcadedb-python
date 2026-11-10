"""parity_crash.py: what native/wheel/parity_wheel.py does when a test file's process dies, as pure code (no files, no subprocess).

A test file runs in its own process. Four things can happen to that process:

  result    pytest wrote its junit XML. Pass or fail is read from the XML; the exit code is not consulted (a failing test is a
            failure, never a crash, and it is never retried).
  TIMEOUT   the harness killed it after --timeout. Strict, never retried.
  BADXML    the XML exists and does not parse. Strict, never retried.
  crash     no XML at all: the process died before pytest wrote its result (access violation, abort, kill by the runner, an
            os._exit in a library). The exit code is the only evidence left, so it is recorded: CRASH(rc=0xC0000005,3221225477).

Retry (off by default, switched on with --retry-crash, which test-native.yml passes on Windows legs only): a crash is run once more in
a fresh process. If the second run produces a result the file is marked FLAKY-CRASH(rc=<first>) and counted from the second run (a
test failure in it still fails the leg); if it crashes again the status is CRASH(rc=<first>;retry rc=<second>) and the leg fails. A
retry budget per leg (default 3 files) keeps a wheel that crashes everywhere from passing on retries.

A process that wrote its result and then died in shutdown (DIRTY-EXIT) is recorded, never retried and never failed.

Everything here takes the facts of an attempt (Attempt) and returns the status and the attempts; the harness runs the processes.
"""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

CRASH = "CRASH"
FLAKY_CRASH = "FLAKY-CRASH"
DIRTY_EXIT = "DIRTY-EXIT"
TIMEOUT = "TIMEOUT"
BADXML = "BADXML"

# The exit codes pytest itself returns (0 ok, 1 failures, 2 interrupted or collection errors, 3 internal error, 4 usage, 5 nothing collected).
# Any other code after the XML was written means the process died later, in the interpreter's or a library's shutdown.
PYTEST_EXIT_CODES = range(0, 6)


def format_rc(rc: Optional[int], windows: Optional[bool] = None) -> str:
    """The exit code the way a crash is read: 0xC0000005,3221225477 (Windows NTSTATUS) or -11,SIGSEGV (a POSIX signal).

    Python reports an unsigned 32-bit code for a Windows process and the negative signal number for a POSIX one; a negative code on
    Windows is the same NTSTATUS read as signed, so it is folded back to 32 bits there.
    """
    if rc is None:
        return "none"
    if windows is None:
        windows = sys.platform == "win32"
    if windows or rc >= 0:
        return f"0x{rc & 0xFFFFFFFF:08X},{rc & 0xFFFFFFFF}"
    try:
        name = signal.Signals(-rc).name
    except ValueError:
        name = f"signal {-rc}"
    return f"{rc},{name}"


@dataclass(frozen=True)
class Attempt:
    """The facts about one run of one test file."""

    rc: Optional[int]  # the exit code; None when the harness killed the process (timed_out)
    result: bool  # the junit XML was written and parses
    timed_out: bool = False
    bad_xml: bool = False  # the XML exists and does not parse


def is_crash(attempt: Attempt) -> bool:
    """No result of any kind and not killed by the harness: the process died on its own."""
    return not (attempt.result or attempt.timed_out or attempt.bad_xml)


def dirty_exit(attempt: Attempt) -> bool:
    """The XML was written but the process ended with a code pytest never returns: it died in shutdown, after the tests."""
    return attempt.result and attempt.rc is not None and attempt.rc not in PYTEST_EXIT_CODES


def dirty_events(name: str, outcome: "Outcome") -> list:
    """One DIRTY-EXIT record per attempt that wrote its result and then died in shutdown (the metadata, the summary and the issue comment list it).

    It is a record and nothing else: the status of the file is what its XML says (the outcome is not retried and the leg is not failed).
    The log of such an attempt is the file's final log, name.log.
    """
    return [
        {
            "file": name,
            "kind": DIRTY_EXIT,
            "first_rc": format_rc(a.rc),
            "retry_rc": "",
            "retry": "not retried, result kept",
            "status": outcome.status,
            "attempt": n,
            "log": f"{name}.log",
        }
        for n, a in enumerate(outcome.attempts, 1)
        if dirty_exit(a)
    ]


@dataclass
class RetryPolicy:
    """Whether a crashed file may be run again, and how many files of one leg may be."""

    enabled: bool = False
    budget: int = 3
    used: int = 0

    def take(self) -> bool:
        if not self.enabled or self.used >= self.budget:
            return False
        self.used += 1
        return True


@dataclass(frozen=True)
class Outcome:
    attempts: Tuple[Attempt, ...]
    status: str  # "" when the file produced a result; else TIMEOUT, BADXML, CRASH(...) or FLAKY-CRASH(...)

    @property
    def final(self) -> Attempt:
        return self.attempts[-1]

    @property
    def retried(self) -> bool:
        return len(self.attempts) > 1

    @property
    def flaky(self) -> bool:
        return self.status.startswith(FLAKY_CRASH)

    @property
    def hard_crash(self) -> bool:
        return self.status.startswith(CRASH)

    @property
    def timed_out(self) -> bool:
        return self.final.timed_out


def run_file(run_attempt: Callable[[int], Attempt], policy: RetryPolicy) -> Outcome:
    """Run one test file by `run_attempt(1)` and, only for a crash and only when the policy allows, once more by `run_attempt(2)`."""
    first = run_attempt(1)
    if not is_crash(first):
        return Outcome((first,), TIMEOUT if first.timed_out else BADXML if first.bad_xml else "")
    crash = f"rc={format_rc(first.rc)}"
    if not policy.take():
        why = ";retry budget spent" if policy.enabled else ""
        return Outcome((first,), f"{CRASH}({crash}{why})")
    second = run_attempt(2)
    if second.result:
        return Outcome((first, second), f"{FLAKY_CRASH}({crash})")
    if second.timed_out:
        again = "TIMEOUT"
    elif second.bad_xml:
        again = "BADXML"
    else:
        again = f"rc={format_rc(second.rc)}"
    return Outcome((first, second), f"{CRASH}({crash};retry {again})")


def event_of(name: str, outcome: Outcome, final_failed: int) -> Optional[dict]:
    """The record of a crash for the metadata file (parity-wheel.meta.json), or None for a file that never crashed.

    `final_failed` is the number of failed or errored tests of the run that produced the result (0 when there is none).
    """
    if not (outcome.flaky or outcome.hard_crash):
        return None
    first, last = outcome.attempts[0], outcome.final
    if outcome.flaky:
        retry = "passed" if final_failed == 0 else f"ran, {final_failed} failed"
    elif outcome.retried:
        retry = "crashed again"
    else:
        retry = "not retried"
    return {
        "file": name,
        "kind": FLAKY_CRASH if outcome.flaky else CRASH,
        "first_rc": format_rc(first.rc),
        "retry_rc": format_rc(last.rc) if outcome.retried else "",
        "retry": retry,
        "status": outcome.status,
    }
