"""open_or_create_database: one call that opens a database when it exists and creates it when it does not.

The same file runs on both builds (arcadedb-jvm and the native arcadedb). The issue is ArcadeData/arcadedb-python#474: a script that
keeps its data in one database could not run twice, because create_database fails on the second run and open_database on the first.
"""

import os
import subprocess  # nosec B404
import sys
import textwrap
import time

import pytest

import arcadedb
from arcadedb.exceptions import ArcadeDBError

ACCEPTED_RACE_OUTCOMES = ("OK", "LOCKED")


def test_first_run_creates_second_run_opens(tmp_path):
    path = str(tmp_path / "memory")
    assert not arcadedb.database_exists(path)

    with arcadedb.open_or_create_database(path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE Note")
        with db.transaction():
            db.command("sql", "INSERT INTO Note SET text = ?", "first run")
    assert arcadedb.database_exists(path)

    # the second run: the same call, the data of the first run is there, and the type is not created again
    with arcadedb.open_or_create_database(path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE Note IF NOT EXISTS")
        with db.transaction():
            db.command("sql", "INSERT INTO Note SET text = ?", "second run")
        texts = sorted(r.get("text") for r in db.query("sql", "SELECT FROM Note"))
    assert texts == ["first run", "second run"]

    # a third call after the second closed it
    with arcadedb.open_or_create_database(path) as db:
        assert db.query("sql", "SELECT count(*) AS n FROM Note").first().get("n") == 2


def test_the_existing_calls_keep_their_errors(tmp_path):
    path = str(tmp_path / "db")
    with pytest.raises(ArcadeDBError, match="does not exist"):
        arcadedb.open_database(path)
    arcadedb.open_or_create_database(path).close()
    with pytest.raises(ArcadeDBError, match="already exists"):
        arcadedb.create_database(path)
    arcadedb.open_database(path).close()


def test_factory_method_does_the_same(tmp_path):
    path = str(tmp_path / "db")
    factory = arcadedb.DatabaseFactory(path)
    assert not factory.exists()
    factory.open_or_create().close()
    assert factory.exists()
    db = factory.open_or_create()
    try:
        assert db.is_open()
        assert not db.is_read_only()
    finally:
        db.close()


def test_read_only_opens_an_existing_database(tmp_path):
    path = str(tmp_path / "db")
    with arcadedb.open_or_create_database(path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE Note")
    with arcadedb.open_or_create_database(path, read_only=True) as db:
        assert db.is_read_only()
        assert db.query("sql", "SELECT count(*) AS n FROM Note").first().get("n") == 0
        with pytest.raises(ArcadeDBError):
            db.command("sql", "CREATE DOCUMENT TYPE Other")


def test_read_only_never_creates(tmp_path):
    path = str(tmp_path / "missing")
    with pytest.raises(ArcadeDBError, match="does not exist"):
        arcadedb.open_or_create_database(path, read_only=True)
    assert not os.path.exists(path)
    assert not arcadedb.database_exists(path)


def test_a_path_that_is_a_file_raises_and_changes_nothing(tmp_path):
    path = tmp_path / "afile"
    path.write_text("not a database")
    with pytest.raises(ArcadeDBError):
        arcadedb.open_or_create_database(str(path))
    assert path.read_text() == "not a database"
    assert path.is_file()


def test_a_directory_without_a_database_gets_one_and_keeps_its_files(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "readme.txt").write_text("mine")
    assert not arcadedb.database_exists(str(folder))
    with arcadedb.open_or_create_database(str(folder)) as db:
        db.command("sql", "CREATE DOCUMENT TYPE Note")
    assert arcadedb.database_exists(str(folder))
    assert (folder / "readme.txt").read_text() == "mine"
    with arcadedb.open_or_create_database(str(folder)) as db:
        assert db.query("sql", "SELECT count(*) AS n FROM Note").first().get("n") == 0


def test_the_same_process_cannot_hold_a_path_twice(tmp_path):
    path = str(tmp_path / "db")
    with arcadedb.open_or_create_database(path):
        with pytest.raises(ArcadeDBError, match="already in use"):
            arcadedb.open_or_create_database(path)
    arcadedb.open_or_create_database(path).close()


def test_jvm_kwargs_are_accepted_like_open_database(tmp_path):
    # the JVM is usually started by an earlier test with other settings: an empty dict starts nothing new and conflicts with nothing
    path = str(tmp_path / "db")
    arcadedb.open_or_create_database(path, jvm_kwargs={}).close()
    arcadedb.open_or_create_database(path, jvm_kwargs={}, read_only=True).close()


def _child(code, *args):
    return subprocess.Popen(  # nosec B603
        [sys.executable, "-c", textwrap.dedent(code), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_file(path, timeout=120):
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.02)


HOLDER = """
    import os, sys, time
    import arcadedb
    path, ready, stop = sys.argv[1:4]
    db = arcadedb.open_or_create_database(path)
    open(ready, "w").close()
    deadline = time.monotonic() + 120
    while not os.path.exists(stop) and time.monotonic() < deadline:
        time.sleep(0.02)
    db.close()
"""


def test_locked_by_another_process_raises_the_engine_error_and_does_not_wait(tmp_path):
    path = str(tmp_path / "db")
    ready, stop = str(tmp_path / "ready"), str(tmp_path / "stop")
    holder = _child(HOLDER, path, ready, stop)
    try:
        _wait_for_file(ready)
        started = time.monotonic()
        with pytest.raises(ArcadeDBError, match="locked by another process"):
            arcadedb.open_or_create_database(path)
        # a hang detector, not a latency bound: the call returns the engine's error, it does not wait for the holder
        assert time.monotonic() - started < 60
    finally:
        open(stop, "w").close()
        out, err = holder.communicate(timeout=120)
    assert holder.returncode == 0, err
    # the holder let go: the call works again, and the database it created is intact
    with arcadedb.open_or_create_database(path) as db:
        assert db.is_open()


RACER = """
    import os, sys, time
    import arcadedb
    path, ready, go = sys.argv[1:4]
    arcadedb.database_exists(os.path.join(os.path.dirname(path), "warm-up"))  # start the engine before the signal
    open(ready, "w").close()
    deadline = time.monotonic() + 120
    while not os.path.exists(go) and time.monotonic() < deadline:
        time.sleep(0.001)
    try:
        arcadedb.open_or_create_database(path).close()
        print("RESULT OK")
    except arcadedb.ArcadeDBError as exc:
        print("RESULT", "LOCKED" if "locked by another process" in str(exc) else "ERROR", exc)
"""


@pytest.mark.parametrize("round_number", range(3))
def test_two_processes_racing_on_a_new_path(tmp_path, round_number):
    """Both processes call the function at the same moment on a path that holds no database.

    Neither may fail with "already exists" or "does not exist". The only other outcome is the engine's lock error, when one process
    still holds the database as the other one looks (the engine allows one process per database). At least one process gets the
    database, and the directory it made opens afterwards.
    """
    path = str(tmp_path / "db")
    go = str(tmp_path / "go")
    readies = [str(tmp_path / f"ready{i}") for i in range(2)]
    children = [_child(RACER, path, r, go) for r in readies]
    try:
        for r in readies:
            _wait_for_file(r)
        open(go, "w").close()
        outputs = [c.communicate(timeout=120) for c in children]
    finally:
        for c in children:
            if c.poll() is None:
                c.kill()
    outcomes = []
    for child, (out, err) in zip(children, outputs):
        assert child.returncode == 0, err
        line = [ln for ln in out.splitlines() if ln.startswith("RESULT")][-1]
        outcomes.append(line)
    kinds = [ln.split()[1] for ln in outcomes]
    assert all(k in ACCEPTED_RACE_OUTCOMES for k in kinds), outcomes
    assert "OK" in kinds, outcomes
    assert arcadedb.database_exists(path)
    with arcadedb.open_or_create_database(path) as db:
        assert db.is_open()
