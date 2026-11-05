"""insert_many pauses CPython's cyclic garbage collector for the call when the input is a list or a tuple (JVM build).

The call allocates many objects the collector tracks (JPype crossings, chunk lists), and without the pause the collector re-traverses the
caller's live row list again and again (ArcadeData/arcadedb-python#235). A lazy input (a generator or any other iterator) is not paused:
user code that runs between the chunks can create reference cycles that only the collector frees. The previous state is restored in
``finally``: the collector comes back on only when it was on at entry, also after an exception.

The state is observed from inside the call: ``Database._insert_many_chunk`` (and ``_insert_many_parallel``) are wrapped to record
``gc.isenabled()`` and then run unchanged, so the rows are really stored and counted.
"""

import gc

import pytest

from arcadedb.core import Database
from arcadedb.exceptions import ArcadeDBError

ROWS = 25


def _count(db, type_name):
    q = f"SELECT count(*) AS n FROM {type_name}"  # nosec B608 - test-controlled
    return int(db.query("sql", q).to_list()[0]["n"])


@pytest.fixture
def collector_state():
    """Remember the collector's state and put it back, so a failing test cannot leave it off for the rest of the run."""
    was = gc.isenabled()
    yield
    if was:
        gc.enable()
    else:
        gc.disable()


@pytest.fixture
def seen(monkeypatch):
    """gc.isenabled() as seen inside each chunk insert and inside the parallel writer path of insert_many."""
    states = []
    real_chunk = Database._insert_many_chunk
    real_parallel = Database._insert_many_parallel

    def chunk(self, *args, **kwargs):
        states.append(gc.isenabled())
        return real_chunk(self, *args, **kwargs)

    def parallel(self, *args, **kwargs):
        states.append(gc.isenabled())
        return real_parallel(self, *args, **kwargs)

    monkeypatch.setattr(Database, "_insert_many_chunk", chunk)
    monkeypatch.setattr(Database, "_insert_many_parallel", parallel)
    return states


@pytest.fixture
def doc_type(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE GcDoc")
    return "GcDoc"


def _rows():
    return [{"id": i, "name": f"row_{i}"} for i in range(ROWS)]


@pytest.mark.parametrize("make", [list, tuple], ids=["list", "tuple"])
def test_the_collector_is_paused_during_a_call_with_a_sized_input(
    temp_db, doc_type, seen, collector_state, make
):
    gc.enable()

    written = temp_db.insert_many(doc_type, make(_rows()))

    assert written == ROWS
    assert _count(temp_db, doc_type) == ROWS
    assert seen == [False]


@pytest.mark.parametrize("make", [list, tuple], ids=["list", "tuple"])
def test_the_collector_is_enabled_again_after_the_call(
    temp_db, doc_type, seen, collector_state, make
):
    gc.enable()

    temp_db.insert_many(doc_type, make(_rows()))

    assert gc.isenabled()


def test_a_call_that_uses_several_chunks_is_paused_for_all_of_them(
    temp_db, doc_type, seen, collector_state
):
    gc.enable()
    rows = [{"id": i} for i in range(25)]

    # commit_every=10 splits the 25 rows into chunks of 10, 10 and 5
    assert temp_db.insert_many(doc_type, rows, commit_every=10) == 25

    assert seen == [False, False, False]
    assert gc.isenabled()


def test_a_parallel_call_with_a_list_is_paused_too(
    temp_db, doc_type, seen, collector_state
):
    gc.enable()

    written = temp_db.insert_many(doc_type, _rows(), parallel=True)

    assert written == ROWS
    assert _count(temp_db, doc_type) == ROWS
    assert seen == [False]
    assert gc.isenabled()


def test_the_collector_is_enabled_again_after_an_exception(
    temp_db, doc_type, monkeypatch, collector_state
):
    gc.enable()
    states = []

    def failing_chunk(self, *args, **kwargs):
        states.append(gc.isenabled())
        raise RuntimeError("chunk failed")

    monkeypatch.setattr(Database, "_insert_many_chunk", failing_chunk)

    with pytest.raises(RuntimeError, match="chunk failed"):
        temp_db.insert_many(doc_type, _rows())

    # the pause was in force when the failure happened, and the exception did not leave it on
    assert states == [False]
    assert gc.isenabled()
    assert not temp_db.is_transaction_active()


def test_a_collector_the_caller_disabled_stays_disabled(
    temp_db, doc_type, seen, collector_state
):
    gc.disable()

    written = temp_db.insert_many(doc_type, _rows())

    assert written == ROWS
    assert seen == [False]
    assert not gc.isenabled()


def test_a_collector_the_caller_disabled_stays_disabled_after_an_exception(
    temp_db, doc_type, monkeypatch, collector_state
):
    gc.disable()

    def failing_chunk(self, *args, **kwargs):
        raise RuntimeError("chunk failed")

    monkeypatch.setattr(Database, "_insert_many_chunk", failing_chunk)

    with pytest.raises(RuntimeError, match="chunk failed"):
        temp_db.insert_many(doc_type, _rows())

    assert not gc.isenabled()


def test_a_generator_is_not_paused(temp_db, doc_type, seen, collector_state):
    gc.enable()
    in_generator = []

    def rows():
        for row in _rows():
            in_generator.append(gc.isenabled())
            yield row

    written = temp_db.insert_many(doc_type, rows())

    assert written == ROWS
    assert _count(temp_db, doc_type) == ROWS
    assert in_generator == [True] * ROWS
    assert seen == [True]
    assert gc.isenabled()


@pytest.mark.parametrize(
    "make", [iter, lambda rows: (r for r in rows)], ids=["iterator", "genexp"]
)
def test_other_lazy_inputs_are_not_paused(
    temp_db, doc_type, seen, collector_state, make
):
    gc.enable()

    written = temp_db.insert_many(doc_type, make(_rows()))

    assert written == ROWS
    assert seen == [True]
    assert gc.isenabled()


def test_a_parallel_generator_is_not_paused(temp_db, doc_type, seen, collector_state):
    gc.enable()

    written = temp_db.insert_many(doc_type, iter(_rows()), parallel=True)

    assert written == ROWS
    assert seen == [True]
    assert gc.isenabled()


def test_an_empty_list_leaves_the_collector_as_it_was(
    temp_db, doc_type, collector_state
):
    gc.enable()
    assert temp_db.insert_many(doc_type, []) == 0
    assert gc.isenabled()
    gc.disable()
    assert temp_db.insert_many(doc_type, []) == 0
    assert not gc.isenabled()


def test_a_closed_database_raises_before_the_collector_is_touched(
    temp_db, collector_state
):
    gc.enable()
    temp_db.close()

    with pytest.raises(ArcadeDBError, match="closed"):
        temp_db.insert_many("GcDoc", _rows())

    assert gc.isenabled()
