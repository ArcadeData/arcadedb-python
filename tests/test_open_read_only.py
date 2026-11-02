"""Opening a database read-only (#383).

`open_database(path, read_only=True)` and `DatabaseFactory.open(read_only=True)`
open the files in `ComponentFile.MODE.READ_ONLY`. A mode is not a query, so it
is the one part of #383 the bindings wrap: every other item there is a SQL
statement (see `test_sql_constraints_and_maintenance.py`).
"""

import contextlib
import re

import pytest

import arcadedb
from arcadedb import ArcadeDBError


@pytest.fixture
def populated_path(temp_db_path):
    """A closed database holding two `Note` documents."""
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE Note")
        with db.transaction():
            db.command("sql", "INSERT INTO Note SET n = 1")
            db.command("sql", "INSERT INTO Note SET n = 2")
    return temp_db_path


def test_read_only_open_reads_and_refuses_writes(populated_path):
    with arcadedb.open_database(populated_path, read_only=True) as db:
        assert db.is_read_only()
        assert db.count_type("Note") == 2
        rows = db.query("sql", "SELECT n FROM Note ORDER BY n").to_list()
        assert [r["n"] for r in rows] == [1, 2]

        with pytest.raises(ArcadeDBError, match="read only"):
            with db.transaction():
                db.command("sql", "INSERT INTO Note SET n = 3")
        with pytest.raises(ArcadeDBError, match="read only"):
            db.command("sql", "CREATE DOCUMENT TYPE Other")

    # The refused writes left the files as they were.
    with arcadedb.open_database(populated_path) as db:
        assert not db.is_read_only()
        assert db.count_type("Note") == 2
        assert not db.schema.exists_type("Other")


def test_read_only_begin_reads_and_end_cleanly_and_the_write_is_refused(populated_path):
    """begin() returns on a read-only database; a write through Document.save() raises an
    ArcadeDBError like the rest of the API, and reads need no transaction.

    Whether commit() of that read-only transaction raises depends on the engine.
    Before ArcadeDB PR #11124 (issue #11057) it raised the read-only error. From
    26.11.1 on it returns, as does rollback(), because ending a transaction that
    wrote nothing is not a write. The 26.11.1-SNAPSHOT line spans both behaviours,
    so a version check cannot tell them apart: accept either outcome. Both must
    leave the data unchanged and no transaction active.
    """
    with arcadedb.open_database(populated_path, read_only=True) as db:
        db.begin()  # not refused
        assert db.count_type("Note") == 2
        with pytest.raises(ArcadeDBError, match="(?i)read.?only"):
            db.new_document("Note").set("n", 3).save()
        try:
            db.commit()  # engines from #11124 on: succeeds
        except ArcadeDBError as e:  # engines before #11124: the read-only error
            assert re.search(r"(?i)read.?only", str(e)), str(e)
            if db.is_transaction_active():  # the old engine may keep it open: end it
                with contextlib.suppress(ArcadeDBError):
                    db.rollback()
        assert not db.is_transaction_active()
        assert db.count_type("Note") == 2
        assert db.query("sql", "SELECT n FROM Note").to_list()


def test_read_only_rollback_of_a_read_transaction(populated_path):
    """rollback() of a read-only transaction: ArcadeDB PR #11124 (issue #11057) made it
    succeed from 26.11.1. Engines before that may raise the read-only error; either way
    the data is unchanged and no transaction stays active."""
    with arcadedb.open_database(populated_path, read_only=True) as db:
        db.begin()
        assert db.count_type("Note") == 2
        try:
            db.rollback()
        except ArcadeDBError as e:
            assert re.search(r"(?i)read.?only", str(e)), str(e)
            if db.is_transaction_active():  # the old engine may keep it open: end it
                with contextlib.suppress(ArcadeDBError):
                    db.rollback()
        assert not db.is_transaction_active()
        assert db.count_type("Note") == 2


def test_factory_open_read_only_and_default_is_writable(populated_path):
    factory = arcadedb.DatabaseFactory(populated_path)

    db = factory.open(read_only=True)
    try:
        assert db.is_read_only()
    finally:
        db.close()

    db = factory.open()
    try:
        assert not db.is_read_only()
        with db.transaction():
            db.command("sql", "INSERT INTO Note SET n = 3")
        assert db.count_type("Note") == 3
    finally:
        db.close()
