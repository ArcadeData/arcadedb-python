"""What a transaction does, as the how-to page ``docs/modules/ROOT/pages/how-to/transactions.adoc`` says it (ArcadeData/arcadedb-python#267).

Each statement on that page is pinned here first: commit, rollback, the context manager, an exception inside it, a failed statement
inside a transaction, a second ``begin()``, the visibility of uncommitted rows, ``set_auto_transaction``, the defaults of
``run_in_transaction`` and the refusal of ``GraphBatch``. The tests use SQL and no Java object, so they run on the JVM build and the
native build alike.
"""

import inspect
import threading

import pytest

from arcadedb import ArcadeDBError


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Account")
    temp_db.command("sql", "CREATE PROPERTY Account.name STRING")
    temp_db.command("sql", "CREATE INDEX ON Account (name) UNIQUE")
    return temp_db


def _names(db):
    rows = db.query("sql", "SELECT name FROM Account ORDER BY name").to_list()
    return [row["name"] for row in rows]


def _insert(db, name):
    db.command("sql", "INSERT INTO Account SET name = ?", name)


def _count_in_another_thread(db):
    seen = []

    def read():
        seen.append(len(db.query("sql", "SELECT FROM Account").to_list()))

    thread = threading.Thread(target=read)
    thread.start()
    thread.join()
    return seen[0]


def test_the_context_manager_commits_on_exit(db):
    with db.transaction():
        assert db.is_transaction_active() is True
        _insert(db, "alice")
    assert db.is_transaction_active() is False
    assert _names(db) == ["alice"]


def test_an_exception_in_the_block_rolls_back_and_is_raised_unchanged(db):
    with db.transaction():
        _insert(db, "alice")

    with pytest.raises(ValueError, match="stop"):
        with db.transaction():
            _insert(db, "bob")
            raise ValueError("stop")

    assert db.is_transaction_active() is False
    assert _names(db) == ["alice"]


def test_begin_then_rollback_or_commit(db):
    db.begin()
    _insert(db, "alice")
    assert db.is_transaction_active() is True
    db.rollback()
    assert db.is_transaction_active() is False
    assert _names(db) == []

    db.begin()
    _insert(db, "alice")
    db.commit()
    assert db.is_transaction_active() is False
    assert _names(db) == ["alice"]


def test_a_write_commit_or_rollback_outside_a_transaction(db):
    with pytest.raises(ArcadeDBError, match="Transaction not begun"):
        _insert(db, "alice")
    assert _names(db) == []

    with pytest.raises(ArcadeDBError, match="Transaction not begun"):
        db.commit()

    # a rollback with nothing to roll back is not an error
    db.rollback()
    assert db.is_transaction_active() is False


def test_a_commit_inside_the_block_makes_the_commit_at_exit_fail(db):
    # the block committed, so the transaction is gone when the context manager commits again
    with pytest.raises(ArcadeDBError, match="Transaction not begun"):
        with db.transaction():
            _insert(db, "alice")
            db.commit()
    assert db.is_transaction_active() is False
    assert _names(db) == ["alice"]


def test_a_transaction_sees_its_own_rows_and_other_threads_see_them_after_the_commit(
    db,
):
    with db.transaction():
        _insert(db, "alice")
        assert _names(db) == ["alice"]
        assert _count_in_another_thread(db) == 0
    assert _count_in_another_thread(db) == 1


def test_a_failed_statement_is_undone_and_the_transaction_goes_on(db):
    with db.transaction():
        _insert(db, "alice")
        with pytest.raises(ArcadeDBError, match="DuplicatedKeyException"):
            # the first row is written before the second is refused
            db.command("sql", "INSERT INTO Account (name) VALUES ('bob'), ('bob')")
        assert db.is_transaction_active() is True
        _insert(db, "carol")
    # 26.11.1: the rows of the failed statement are gone, the others are committed
    assert _names(db) == ["alice", "carol"]


def test_a_second_begin_opens_a_nested_transaction_of_its_own(db):
    db.begin()
    _insert(db, "outer")
    db.begin()
    # the nested transaction does not see what the outer one has not committed
    assert _names(db) == []
    _insert(db, "inner")
    db.commit()
    # that commit ended the nested transaction only, and what it committed is visible to the outer one
    assert db.is_transaction_active() is True
    assert _names(db) == ["inner", "outer"]
    db.rollback()
    assert db.is_transaction_active() is False
    # the nested commit stands, the outer rollback took back only the outer rows
    assert _names(db) == ["inner"]


def test_nested_with_blocks_end_one_at_a_time(db):
    with pytest.raises(ValueError, match="stop"):
        with db.transaction():
            _insert(db, "outer")
            with db.transaction():
                _insert(db, "inner")
            raise ValueError("stop")
    assert db.is_transaction_active() is False
    assert _names(db) == ["inner"]


def test_an_exception_in_a_nested_block_rolls_back_only_that_block(db):
    with db.transaction():
        _insert(db, "outer")
        with pytest.raises(ValueError, match="stop"):
            with db.transaction():
                _insert(db, "inner")
                raise ValueError("stop")
        assert db.is_transaction_active() is True
    assert _names(db) == ["outer"]


def test_set_auto_transaction_commits_each_bare_write_alone(db):
    db.set_auto_transaction(True)
    try:
        _insert(db, "alice")
        assert db.is_transaction_active() is False
        assert _names(db) == ["alice"]
    finally:
        db.set_auto_transaction(False)
    with pytest.raises(ArcadeDBError, match="Transaction not begun"):
        _insert(db, "bob")


def test_run_in_transaction_returns_the_value_and_has_these_defaults(db):
    def work():
        _insert(db, "alice")
        return "done"

    assert db.run_in_transaction(work) == "done"
    assert db.is_transaction_active() is False
    assert _names(db) == ["alice"]

    parameters = inspect.signature(db.run_in_transaction).parameters
    assert parameters["retries"].default == 12
    assert parameters["backoff_s"].default == 0.005


def test_graph_batch_refuses_to_run_inside_your_transaction(db):
    db.command("sql", "CREATE VERTEX TYPE City")
    with db.graph_batch(parallel_flush=False) as batch:
        db.begin()
        try:
            with pytest.raises(ArcadeDBError):
                batch.create_vertices("City", 2)
            # the refusal leaves the caller's transaction as it was
            assert db.is_transaction_active() is True
        finally:
            db.rollback()
    assert db.count_type("City") == 0
