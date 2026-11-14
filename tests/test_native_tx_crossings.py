"""The native wheel spends the fewest library crossings a transaction needs, and each one is a single foreign call.

Every ``arcadedb_n_tx`` call is one trip into the native library. It used to return a malloc'd JSON text (about 60 bytes) that the
caller then read with ``strlen``, a copy into a ``str`` and ``arcadedb_free``: four foreign calls per crossing, twelve for
``with db.transaction()``. It now returns the transaction state as one integer (bit 0 ``active``, bits 1 to 31 ``depth``, bits 32 to
62 ``commits``, -1 for an error whose text ``lib.last_error`` holds), the way ``arcadedb_n_cmd`` returns a cursor id: one foreign
call, no reply text, no free, no JSON decode. The reply of begin, commit and rollback carries the whole state, so a caller that
needs the state right after one of these calls must not ask for it again with a mark.

The counts are crossings and foreign calls, not seconds: they are deterministic. The behaviour tests next to them pin what the
marks are for (a caller's transaction is never rolled back by a nested block, a retry rolls the failed attempt back) and that an
error reaches Python as the same typed exception with the same message, so a lower count can not buy a different result.

Needs the native library, so conftest.py does not collect this file on the JPype backend (a skip there would fail
scripts/check_test_skips.py); it runs in the native parity run.
"""

import contextlib
import json

import pytest

import arcadedb as arcadedb
from arcadedb import ArcadeDBError
from arcadedb._native import lib

_OPS = {0: "begin", 1: "commit", 2: "rollback", 3: "mark"}
_DEPTH_MASK = 0x7FFFFFFF


def _fields(state):
    """``(active, depth, commits)`` of the integer ``arcadedb_n_tx`` returns (the layout is pinned here, not read from the wheel)."""
    return bool(state & 1), (state >> 1) & _DEPTH_MASK, state >> 32


class _Crossings:
    """Counts every ``lib.tx`` and ``lib.call`` made while it is installed; ``tx`` is the sequence of transaction operations."""

    def __init__(self, monkeypatch):
        self.calls = []
        real_tx = lib.tx
        real_call = lib.call

        def counting_tx(handle, op):
            self.calls.append(_OPS[op])
            return real_tx(handle, op)

        def counting_call(name, *args):
            self.calls.append(name)
            return real_call(name, *args)

        monkeypatch.setattr(lib, "tx", counting_tx)
        monkeypatch.setattr(lib, "call", counting_call)

    @property
    def tx(self):
        return [c for c in self.calls if c in _OPS.values()]

    def reset(self):
        self.calls.clear()


class _ForeignCalls:
    """Every foreign (ctypes) call the transaction path can make, by name, while it is installed.

    The entry points of the library are counted through a proxy of the loaded library object, and the two helpers that turn a reply
    text into a ``str`` (``strlen`` and ``PyUnicode_FromStringAndSize``) by name. ``arcadedb_free`` is an entry point, so a reply
    text that is read and released shows up as ``arcadedb_n_tx``, ``_strlen``, ``_str_from_utf8``, ``arcadedb_free``.
    """

    def __init__(self, monkeypatch):
        self.names = []
        names = self.names
        real_lib = lib._lib

        class _Proxy:
            def __getattr__(self, attr):
                function = getattr(real_lib, attr)

                def counted(*args):
                    names.append(attr)
                    return function(*args)

                return counted

        monkeypatch.setattr(lib, "_lib", _Proxy())
        for helper in ("_strlen", "_str_from_utf8"):
            monkeypatch.setattr(
                lib, helper, self._counted(helper, getattr(lib, helper))
            )

    def _counted(self, name, function):
        def counted(*args):
            self.names.append(name)
            return function(*args)

        return counted

    @property
    def entries(self):
        # a batched cursor close that an earlier statement left waiting runs before a commit: it belongs to that statement
        return [n for n in self.names if n != "arcadedb_n_rs_close_many"]

    def reset(self):
        del self.names[:]


@pytest.fixture
def db(temp_db_path):
    database = arcadedb.create_database(temp_db_path)
    database.command("sql", "CREATE DOCUMENT TYPE R")
    with (
        database.transaction()
    ):  # the thread's transaction context exists from here on, as it does in a running application
        pass
    lib.flush_closes()  # a cursor closed earlier is not part of a call measured below
    try:
        yield database
    finally:
        # a test that fails with a transaction open (a begin without its end, a level a direct lib.tx call pushed) must not turn into
        # a teardown error as well: the engine refuses to drop a database that has a transaction open on the thread
        try:
            for _ in range(16):
                if not database.is_transaction_active():
                    break
                database.rollback()
        finally:
            database.drop()


@pytest.fixture
def crossings(db, monkeypatch):
    return _Crossings(monkeypatch)


@pytest.fixture
def foreign(db, monkeypatch):
    return _ForeignCalls(monkeypatch)


def _count(db):
    return db.query("sql", "SELECT count(*) AS c FROM R").first().get("c")


def _need_retry():
    error = ArcadeDBError(
        "Failed to commit transaction: ConcurrentModificationException"
    )
    error.native = {"retry": "need_retry", "retry_ms": 0}
    return error


# -- crossing counts


def test_run_in_transaction_happy_path_is_three_crossings(db, crossings):
    # the mark before begin (what this call owns), begin (its reply is the commit-count witness), commit
    assert db.run_in_transaction(lambda: 42) == 42
    assert crossings.calls == ["mark", "begin", "commit"]


def test_with_transaction_is_three_crossings(db, crossings):
    with db.transaction():
        pass
    assert crossings.calls == ["mark", "begin", "commit"]


def test_with_transaction_that_raises_is_three_crossings(db, crossings):
    with pytest.raises(ValueError):
        with db.transaction():
            raise ValueError("boom")
    assert crossings.calls == ["mark", "begin", "rollback"]


def test_run_in_transaction_that_raises_reads_the_state_once(db, crossings):
    # mark, begin, then one state read for the unwind: the rollback replies carry the state, so no is-active read and no commit-count read
    with pytest.raises(ValueError):
        db.run_in_transaction(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert crossings.calls == ["mark", "begin", "mark", "rollback"]
    assert not db.is_transaction_active()


def test_run_in_transaction_retry_costs_one_unwind_and_one_attempt(db, crossings):
    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) == 1:
            raise _need_retry()
        return "done"

    assert db.run_in_transaction(fn, backoff_s=0) == "done"
    assert len(attempts) == 2
    assert crossings.calls == [
        *["mark", "begin", "mark", "rollback"],  # the refused attempt, unwound
        *["mark", "begin", "commit"],  # the attempt that commits
    ]


# -- foreign calls: a crossing is one call into the library, with no reply text to read and free


def test_with_transaction_is_three_foreign_calls(db, foreign):
    # was twelve: each of the three crossings (mark, begin, commit) was the entry, strlen, the copy into a str and arcadedb_free
    with db.transaction():
        pass
    assert foreign.entries == ["arcadedb_n_tx"] * 3


def test_each_transaction_call_is_one_foreign_call_and_frees_nothing(db, foreign):
    for call in (
        db.is_transaction_active,
        db.begin,
        db.is_transaction_active,
        db.rollback,
        db.begin,
        db.commit,
        db._transaction_stack_mark,
    ):
        foreign.reset()
        call()
        assert foreign.entries == ["arcadedb_n_tx"], call.__name__


def test_a_transaction_that_raises_is_three_foreign_calls(db, foreign):
    with pytest.raises(ValueError):
        with db.transaction():
            raise ValueError("boom")
    assert foreign.entries == ["arcadedb_n_tx"] * 3


def test_a_transaction_never_reads_a_reply_text(db, monkeypatch):
    taken = []
    real = lib._take
    monkeypatch.setattr(lib, "_take", lambda *a: taken.append(1) or real(*a))
    with db.transaction():
        pass
    db.begin()
    assert db.is_transaction_active() is True
    db.rollback()
    assert db.run_in_transaction(lambda: 1) == 1
    assert taken == []


# -- what the marks are for: the same results at the lower count


def test_a_nested_block_that_fails_leaves_the_callers_transaction_open(db):
    with db.transaction():
        db.command("sql", "INSERT INTO R SET k = 1")
        with pytest.raises(ValueError):
            db.run_in_transaction(
                lambda: (
                    db.command("sql", "INSERT INTO R SET k = 2"),
                    (_ for _ in ()).throw(ValueError("boom")),
                )
            )
        assert db.is_transaction_active()
    assert _count(db) == 1


def test_a_nested_block_that_succeeds_commits_with_the_caller(db):
    with db.transaction():
        db.run_in_transaction(lambda: db.command("sql", "INSERT INTO R SET k = 1"))
        db.command("sql", "INSERT INTO R SET k = 2")
    assert _count(db) == 2


def test_nested_with_blocks_push_one_level_for_the_inner_block_only(db):
    # The depth is the size of the thread's stack of transaction contexts (LocalDatabase.getNestedTransactions). The stack always holds
    # the thread's base context once the thread has used the database (the fixture did), so it is 1 while idle. The outer begin
    # reuses that base context (depth stays 1, active); only a begin while a transaction is active pushes a context (2), and the commit
    # of that inner level pops it (1, still active). The outer commit ends the base context and leaves it on the stack (1, idle).
    base = db._transaction_stack_mark()[1]
    assert db._transaction_stack_mark() == (False, 1)
    with db.transaction():
        assert db._transaction_stack_mark() == (True, base)
        with db.transaction():
            assert db._transaction_stack_mark() == (True, base + 1)
            db.command("sql", "INSERT INTO R SET k = 1")
        assert db._transaction_stack_mark() == (True, base)
    assert db._transaction_stack_mark() == (False, base)
    assert _count(db) == 1


def test_a_refused_attempt_is_rolled_back_before_the_retry(db):
    attempts = []

    def fn():
        attempts.append(1)
        db.command("sql", "INSERT INTO R SET k = 1")
        if len(attempts) == 1:
            raise _need_retry()

    db.run_in_transaction(fn, backoff_s=0)
    assert len(attempts) == 2
    assert _count(db) == 1
    assert not db.is_transaction_active()


def test_a_duplicated_key_is_retried_once(db):
    attempts = []

    def fn():
        attempts.append(1)
        error = ArcadeDBError("duplicated key")
        error.native = {"retry": "duplicated_key", "retry_ms": 0}
        raise error

    with pytest.raises(ArcadeDBError):
        db.run_in_transaction(fn, backoff_s=0)
    assert len(attempts) == 2
    assert not db.is_transaction_active()


def test_an_error_without_a_retry_kind_is_raised_after_one_attempt(db):
    attempts = []

    def fn():
        attempts.append(1)
        raise ArcadeDBError("not a conflict")

    with pytest.raises(ArcadeDBError):
        db.run_in_transaction(fn, backoff_s=0)
    assert attempts == [1]
    assert not db.is_transaction_active()


def test_begin_commit_rollback_return_nothing(db):
    assert db.begin() is None
    assert db.rollback() is None
    assert db.begin() is None
    assert db.commit() is None


# -- decodes: a status the caller does not read is not a JSON decode, and a state the caller reads is not one either


class _Decodes:
    """Counts every ``json.loads`` while it is installed (an error reply is decoded with it, nothing else is)."""

    def __init__(self, monkeypatch):
        self.n = 0
        real = json.loads

        def counting(*args, **kwargs):
            self.n += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(json, "loads", counting)


@pytest.fixture
def decodes(db, monkeypatch):
    return _Decodes(monkeypatch)


def test_begin_commit_rollback_decode_nothing(db, decodes):
    db.begin()
    db.commit()
    db.begin()
    db.rollback()
    assert decodes.n == 0


def test_is_transaction_active_decodes_nothing_and_reports_the_state(db, decodes):
    assert db.is_transaction_active() is False
    db.begin()
    assert db.is_transaction_active() is True
    db.commit()
    assert db.is_transaction_active() is False
    assert decodes.n == 0


def test_with_transaction_decodes_nothing(db, decodes):
    # the mark reads two fields (active, depth) out of the integer, begin and commit read none: no JSON anywhere
    with db.transaction():
        pass
    assert decodes.n == 0


def test_a_state_still_carries_active_depth_and_commits(db):
    db.begin()
    db.begin()
    # the first begin reuses the thread's base context (depth 1), the second pushes one (2)
    assert db._transaction_stack_mark() == (True, 2)
    db.rollback()
    db.rollback()
    assert db._transaction_stack_mark()[0] is False
    active, depth, _ = _fields(db._begin_state())
    assert (active, depth) == (True, 1)
    active, _, _ = _fields(db._rollback_state())
    assert active is False


def test_the_state_comes_back_as_one_integer(db):
    handle = db._h
    idle = lib.tx(handle, 3)
    assert type(idle) is int and idle >= 0
    active, base, _ = _fields(idle)
    assert active is False
    # the first begin reuses the thread's base context (the depth stays), a begin inside it pushes one more (see the nested with test)
    one = lib.tx(handle, 0)
    assert type(one) is int and one >= 0
    assert _fields(one)[:2] == (True, base)
    two = lib.tx(handle, 0)
    assert _fields(two)[:2] == (True, base + 1)
    assert _fields(two)[2] == _fields(lib.tx(handle, 3))[2]  # a mark reads the count
    assert _fields(lib.tx(handle, 2))[:2] == (True, base)  # the inner level is gone
    assert _fields(lib.tx(handle, 2))[:2] == (False, base)  # the outer one ended


# -- errors: the same typed exception with the same message


def test_a_failed_call_returns_minus_one_and_the_error_is_in_last_error(db):
    assert lib.tx(2**62, 3) == -1  # a handle no database has
    error = json.loads(lib.last_error())
    assert error["ok"] is False
    assert error["cls"] == "java.lang.IllegalArgumentException"
    assert "unknown database handle" in error["error"]
    assert error["chain"][0][0] == error["cls"]


def test_a_commit_without_begin_raises_the_engines_error_unchanged(db):
    with pytest.raises(ArcadeDBError) as caught:
        db.commit()
    native = caught.value.native
    assert native["ok"] is False
    assert native["cls"] and "Transaction not begun" in native["error"]
    assert (
        str(caught.value)
        == f"Failed to commit transaction: {native['cls']}: {native['error']}"
    )
    assert caught.value.java_chain == native["chain"]
    assert caught.value.java_chain[0] == [native["cls"], native["error"]]
    assert db.is_transaction_active() is False


def test_a_rollback_with_nothing_to_roll_back_is_not_an_error(db):
    db.rollback()
    assert db.is_transaction_active() is False


def test_a_failed_statement_in_a_block_rolls_the_block_back_and_leaves_no_level(db):
    db.command("sql", "CREATE PROPERTY R.k INTEGER")
    db.command("sql", "CREATE INDEX ON R (k) UNIQUE")
    base = db._transaction_stack_mark()
    with pytest.raises(ArcadeDBError) as caught:
        with db.transaction():
            db.command("sql", "INSERT INTO R (k) VALUES (1), (1)")
    assert "DuplicatedKeyException" in str(caught.value)
    assert db._transaction_stack_mark() == base
    assert _count(db) == 0


def test_a_failure_reply_is_read_from_last_error_into_the_same_error(db, monkeypatch):
    reply = (
        '{"ok":false,"error":"boom","cls":"x.Y","chain":[["x.Y","boom"]],'
        '"retry":"need_retry","retry_ms":3}'
    )
    monkeypatch.setattr(lib, "tx", lambda handle, op: -1)
    monkeypatch.setattr(lib, "last_error", lambda: reply)
    for op, prefix in (
        (db.begin, "Failed to begin transaction"),
        (db.commit, "Failed to commit transaction"),
        (db.rollback, "Failed to rollback transaction"),
        (db.is_transaction_active, "Failed to check transaction status"),
        (db._transaction_stack_mark, "Failed to read the transaction state"),
    ):
        with pytest.raises(ArcadeDBError) as caught:
            op()
        assert str(caught.value) == f"{prefix}: x.Y: boom"
        assert caught.value.native["retry"] == "need_retry"
        assert caught.value.java_chain == [["x.Y", "boom"]]


# -- a closed database


def test_a_transaction_left_open_is_gone_after_a_close_and_reopen(temp_db_path):
    database = arcadedb.create_database(temp_db_path)
    reopened = None
    try:
        database.command("sql", "CREATE DOCUMENT TYPE R")
        database.begin()
        database.command("sql", "INSERT INTO R SET k = 1")
        assert database.is_transaction_active() is True
        database.close()
        for call in (
            database.begin,
            database.commit,
            database.rollback,
            database.is_transaction_active,
            database._transaction_stack_mark,
        ):
            with pytest.raises(ArcadeDBError, match="Database is closed"):
                call()
        reopened = arcadedb.open_database(temp_db_path)
        assert reopened.is_transaction_active() is False
        assert _count(reopened) == 0
    finally:
        if reopened is not None:
            reopened.drop()
        else:
            # a failure before the reopen: end what is open and drop, ignoring the refusals of a database already closed
            with contextlib.suppress(ArcadeDBError):
                database.rollback()
            with contextlib.suppress(ArcadeDBError):
                database.drop()
