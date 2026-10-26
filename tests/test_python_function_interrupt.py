"""A real Ctrl-C during a statement that calls a Python function always arrives as KeyboardInterrupt.

The signal can land in the function, in the engine between two calls, or on entering the callback
(where the exception leaves it as JPype's PyExceptionProxy and the engine wraps it). Each of these
used to come out differently (KeyboardInterrupt, or ArcadeDBError). When the signal also lands while
JPype converts the engine's exception, JPype fails with ``RuntimeError: Fatal error occurred``
(jpype-project/jpype#1496), which was an ArcadeDBError in about 5% of the runs; it is a
KeyboardInterrupt now. Not run on Windows, where a
signal cannot be sent to the own process like this (see test_sigint.py).

The test needs Python's own SIGINT handler (``signal.default_int_handler``), and Python installs it
at startup only when SIGINT is not ignored. A pytest started as a background job inherits SIGINT as
ignored (a shell runs ``cmd &`` without job control that way, so do many launchers and agent
shells): ``os.kill(os.getpid(), SIGINT)`` is then dropped, no query is interrupted, and the test
failed every time on such a runner while it passed in CI. The ``default_sigint_handler`` fixture
installs the handler for the test and puts back what it found.
"""

import os
import signal
import subprocess  # nosec B404 - test-controlled child process
import sys
import threading
import time

import numpy as np
import pytest

import arcadedb


@pytest.fixture
def db(temp_db_path):
    with arcadedb.create_database(temp_db_path) as database:
        database.command("sql", "CREATE DOCUMENT TYPE T")
        database.insert_columns("T", {"n": np.arange(300_000, dtype=np.int64)})
        database.register_function("plus_one", lambda x: x + 1)
        yield database


@pytest.fixture
def default_sigint_handler():
    """KeyboardInterrupt on SIGINT for the test, whatever the runner left (see the module docstring)."""
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        # getsignal is None for a handler installed by C code: nothing to give back from Python
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def test_sigint_during_a_query_that_calls_python_is_a_keyboard_interrupt(
    db, default_sigint_handler
):
    interrupted = 0
    for delay in (0.05, 0.09, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33):
        timer = threading.Timer(delay, os.kill, (os.getpid(), signal.SIGINT))
        timer.start()
        try:
            db.query("sql", "SELECT sum(plus_one(n)) AS s FROM T").first()
            outcome = "completed"
        except KeyboardInterrupt:
            outcome = "interrupted"
            interrupted += 1
        finally:
            timer.cancel()
            timer.join()
        if outcome == "completed":
            try:  # a signal sent just after the end is delivered here, still a KeyboardInterrupt
                time.sleep(0.05)
            except KeyboardInterrupt:
                interrupted += 1
        # the database works as before
        assert db.count_type("T") == 300_000
    assert interrupted >= 1


class _JpypeLostTheRace:
    """Stands in for the Java database when JPype loses its race with Ctrl-C (jpype#1496).

    The engine handles a KeyboardInterrupt that left the Python function as a Java exception;
    JPype's conversion of it to Python fails with ``InterruptedException: sleep interrupted``
    and the call raises this plain RuntimeError, with nothing left for the bindings to read.
    """

    def query(self, *args):
        raise RuntimeError("Fatal error occurred")

    command = query


@pytest.mark.parametrize("method", ["query", "command"])
def test_jpypes_fatal_error_during_a_statement_is_a_keyboard_interrupt(
    db, method, monkeypatch
):
    monkeypatch.setattr(db, "_java_db", _JpypeLostTheRace())
    with pytest.raises(KeyboardInterrupt) as raised:
        getattr(db, method)("sql", "SELECT sum(plus_one(n)) AS s FROM T")
    # the text of JPype's error stays reachable
    assert "Fatal error occurred" in str(raised.value.__cause__)


def test_a_statement_that_fails_otherwise_is_still_an_arcadedb_error(db):
    with pytest.raises(arcadedb.ArcadeDBError) as raised:
        db.query("sql", "SELECT FROM NoSuchType")
    assert not isinstance(raised.value, KeyboardInterrupt)


_SIGINT_LOOP = r"""
import os, signal, tempfile, threading, time
import numpy as np
import arcadedb

# a parent started as a background job ignores SIGINT, and a child inherits that: Python then installs no handler
signal.signal(signal.SIGINT, signal.default_int_handler)
with tempfile.TemporaryDirectory() as directory:
    db = arcadedb.create_database(os.path.join(directory, "db"))
    db.command("sql", "CREATE DOCUMENT TYPE T")
    db.insert_columns("T", {"n": np.arange(150_000, dtype=np.int64)})
    db.register_function("plus_one", lambda x: x + 1)
    interrupted = 0
    for delay in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55):
        timer = threading.Timer(delay, os.kill, (os.getpid(), signal.SIGINT))
        timer.start()
        try:
            db.query("sql", "SELECT sum(plus_one(n)) AS s FROM T").first()
            done = True
        except KeyboardInterrupt:
            done = False
            interrupted += 1
        finally:
            timer.cancel()
            timer.join()
        if done:
            try:
                time.sleep(0.05)
            except KeyboardInterrupt:
                interrupted += 1
        assert db.count_type("T") == 150_000
    db.close()
    print("interrupted", interrupted)
"""


@pytest.mark.skipif(
    sys.platform == "win32", reason="a signal cannot be sent to the own process here"
)
def test_sigint_loop_in_a_fresh_process_ends_cleanly():
    """The same loop in its own process. A Ctrl-C raised as the Python callback is entered used to leave the engine a stale
    reply pointer, and the process aborted in ``free()`` (nonzero exit); pytest's own hooks hid it inside a test run.
    """
    result = subprocess.run(  # nosec B603 - fixed argv, no shell, test-owned code string
        [sys.executable, "-c", _SIGINT_LOOP],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert int(result.stdout.split()[-1]) >= 1, result.stdout
