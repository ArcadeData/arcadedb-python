"""Native ``insert_columns(parallel=True)`` takes the columnar entry and stores what the row path of ``insert_many(parallel=True)`` stores.

Before, a parallel call skipped the columnar fast path: the columns became one dict per row, then one JSON text per chunk of 10,000
rows for ``arcadedb_n_async insert_parallel``. Now the same ``arcadedb_n_insert_colp`` crossing that the synchronous call uses carries
``"parallel": true`` and the image hands the records to the async writers, as ``DocumentBatcher.insertColumnsParallel`` does on the
JVM wheel. Needs the native library, so conftest.py collects this file on the native backend only. Each test loads the same data
through the row path (the private reference: ``insert_many(parallel=True)`` with dict rows) and through the columns, compares the
stored records value by value and type by type, and spies on the entry points so a silent fall back cannot pass.
"""

import numpy as np
import pytest

from arcadedb._native import bulk, extras, lib  # noqa: E402
from arcadedb.exceptions import ArcadeDBError  # noqa: E402

COLP, ROWS = "arcadedb_n_insert_colp", "arcadedb_n_insert_rows"


class Spy:
    """Counts the crossings by name and the calls that hand a chunk of dict rows to the parallel writers."""

    def __init__(self, monkeypatch):
        self.calls, self.row_chunks = [], 0
        real_call, real_rows = lib.call, extras.insert_parallel

        def call(name, *args):
            self.calls.append(name)
            return real_call(name, *args)

        def insert_parallel(database, type_name, rows):
            self.row_chunks += 1
            return real_rows(database, type_name, rows)

        monkeypatch.setattr(lib, "call", call)
        monkeypatch.setattr(extras, "insert_parallel", insert_parallel)

    def count(self, name):
        return self.calls.count(name)


@pytest.fixture
def spy(monkeypatch):
    return Spy(monkeypatch)


def _count(db, name):
    return db.query("sql", f"SELECT count(*) AS n FROM {name}").to_list()[0][
        "n"
    ]  # nosec B608 - test-owned type name


def _types(db, *names, kind="DOCUMENT", buckets=3, unique=False):
    for name in names:
        db.command("sql", f"CREATE {kind} TYPE {name} BUCKETS {buckets}")
        db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")
        db.command("sql", f"CREATE PROPERTY {name}.val DOUBLE")
        db.command("sql", f"CREATE PROPERTY {name}.flag BOOLEAN")
        db.command("sql", f"CREATE PROPERTY {name}.txt STRING")
        db.command("sql", f"CREATE PROPERTY {name}.vec ARRAY_OF_FLOATS")
        if unique:
            db.command("sql", f"CREATE INDEX ON {name} (id) UNIQUE")


def _stored(db, name):
    """Every record of a type as (value, python type) pairs per property, ordered by id (the writers store in no fixed order)."""
    sql = f"SELECT FROM {name} ORDER BY id"  # nosec B608 - test-owned type name
    return [
        {k: (v, type(v).__name__) for k, v in row.items() if not k.startswith("@")}
        for row in db.query("sql", sql).to_list()
    ]


def _columns(n, dim=4, seed=11):
    rng = np.random.default_rng(seed)
    ids = np.arange(n, dtype=np.int64)
    return {
        "id": ids,
        "val": (ids * 37 % 10007) / 7.0,
        "flag": ids % 3 == 0,
        "txt": ["row-%d-%s" % (k, "é" * (k % 5)) for k in range(n)],
        "vec": rng.standard_normal((n, dim), dtype=np.float32),
    }


def _rows(columns):
    names = list(columns)
    lists = [c.tolist() if hasattr(c, "tolist") else c for c in columns.values()]
    return [dict(zip(names, vals)) for vals in zip(*lists)]


def test_a_parallel_call_uses_the_columnar_entry_and_stores_the_row_paths_records(
    temp_db, spy
):
    _types(temp_db, "Ref", "Col")
    columns = _columns(3_100)
    assert temp_db.insert_many("Ref", _rows(columns), parallel=True) == 3_100
    assert spy.row_chunks == 1 and spy.count(COLP) == 0  # the reference is the row path

    spy.calls.clear()
    spy.row_chunks = 0
    assert temp_db.insert_columns("Col", columns, parallel=True) == 3_100
    assert spy.count(COLP) == 1, spy.calls  # one crossing for the whole call
    assert spy.row_chunks == 0 and spy.count(ROWS) == 0  # no dict rows, no JSON text

    expected = _stored(temp_db, "Ref")
    assert len(expected) == 3_100 and _count(temp_db, "Col") == 3_100
    assert expected[7]["id"][1] == "int" and expected[7]["vec"][1] == "list"
    assert _stored(temp_db, "Col") == expected


def test_nulls_and_awkward_strings_are_stored_as_the_row_path_stores_them(temp_db, spy):
    _types(temp_db, "Ref", "Col")
    columns = {
        "id": list(range(8)),
        "txt": ["", "a", None, "snow☃man", "x" * 300, None, "\n\t", "z"],
        "val": [1.5, None, 2.5, 1e300, -1e-300, 0.0, 3.0, None],
    }
    temp_db.insert_many("Ref", _rows(columns), parallel=True)
    spy.calls.clear()
    temp_db.insert_columns("Col", columns, parallel=True)
    assert spy.count(COLP) == 1
    got, want = _stored(temp_db, "Col"), _stored(temp_db, "Ref")
    assert got == want and got[2]["txt"][0] is None


def test_a_vertex_type_gets_vertices_in_parallel(temp_db, spy):
    _types(temp_db, "Ref", "Col", kind="VERTEX")
    columns = _columns(1_500)
    temp_db.insert_many("Ref", _rows(columns), parallel=True)
    spy.calls.clear()
    temp_db.insert_columns("Col", columns, parallel=True)
    assert spy.count(COLP) == 1
    assert _stored(temp_db, "Col") == _stored(temp_db, "Ref")
    kinds = temp_db.query(
        "sql", "SELECT FROM schema:types WHERE name = 'Col'"
    ).to_list()
    assert kinds[0]["type"] == "vertex"


def test_a_record_the_writers_reject_is_reported_after_the_wait(temp_db, spy):
    _types(temp_db, "Col", unique=True)
    ids = np.arange(400, dtype=np.int64)
    ids[300] = ids[7]  # a duplicate key
    with pytest.raises(ArcadeDBError, match="failed record"):
        temp_db.insert_columns("Col", {"id": ids, "txt": ["t"] * 400}, parallel=True)
    assert spy.count(COLP) == 1 and spy.row_chunks == 0
    stored = _count(temp_db, "Col")
    assert 0 < stored <= 400  # the call had waited for the writers: nothing lands later


def test_a_value_the_declared_type_refuses_raises_and_the_call_is_final(temp_db, spy):
    _types(temp_db, "Col")
    values = [str(i) for i in range(1_000)]
    values[700] = "not a number"
    with pytest.raises(
        ArcadeDBError, match="may have been stored|failed record"
    ) as err:
        temp_db.insert_columns("Col", {"id": values}, parallel=True)
    assert "rolled back" not in str(err.value)
    assert spy.count(COLP) == 1
    stored = _count(temp_db, "Col")
    temp_db.async_executor().wait_completion()
    assert _count(temp_db, "Col") == stored <= 1_000


def test_an_edge_type_is_refused_as_before(temp_db):
    temp_db.command("sql", "CREATE EDGE TYPE Link")
    with pytest.raises(ArcadeDBError, match="edge type"):
        temp_db.insert_columns("Link", {"id": [1, 2, 3]}, parallel=True)


def test_a_missing_type_is_the_schemas_error(temp_db):
    with pytest.raises(ArcadeDBError, match="NoSuchType"):
        temp_db.insert_columns("NoSuchType", {"id": [1, 2, 3]}, parallel=True)


def test_a_column_the_buffers_cannot_carry_still_goes_through_the_row_path(
    temp_db, spy
):
    _types(temp_db, "Col")
    columns = {"id": [1, 2, 3, 4], "txt": ["a", 2, "c", 4.5]}  # a mixed-type column
    assert temp_db.insert_columns("Col", columns, parallel=True) == 4
    assert spy.count(COLP) == 0 and spy.row_chunks == 1
    assert _count(temp_db, "Col") == 4


def test_the_switch_that_turns_the_columnar_path_off_applies_to_parallel(
    temp_db, spy, monkeypatch
):
    _types(temp_db, "Col")
    monkeypatch.setattr(bulk, "_ENABLED", False)
    assert temp_db.insert_columns("Col", _columns(50), parallel=True) == 50
    assert spy.count(COLP) == 0 and spy.row_chunks == 1
    assert _count(temp_db, "Col") == 50


def test_no_rows_is_zero_and_writes_nothing(temp_db, spy):
    _types(temp_db, "Col")
    assert temp_db.insert_columns("Col", {"id": []}, parallel=True) == 0
    assert spy.count(COLP) == 0 and _count(temp_db, "Col") == 0


def test_the_synchronous_call_is_unchanged(temp_db, spy):
    _types(temp_db, "Col", unique=True)
    assert temp_db.insert_columns("Col", _columns(500), commit_every=100) == 500
    assert spy.count(COLP) == 1 and spy.row_chunks == 0
    assert _count(temp_db, "Col") == 500


def test_insert_arrow_parallel_takes_the_columnar_entry(temp_db, spy, monkeypatch):
    import pyarrow as pa

    _types(temp_db, "Ref", "ByTable")
    columns = _columns(2_000)
    table = pa.table(
        {
            "id": columns["id"],
            "val": pa.array(
                [None if i % 7 == 0 else v for i, v in enumerate(columns["val"])]
            ),
            "txt": pa.array(columns["txt"]),
        }
    )
    with (
        monkeypatch.context() as m
    ):  # the private reference: the row path of the old code
        m.setattr(bulk, "_ENABLED", False)
        assert (
            temp_db.insert_arrow("Ref", table, batch_size=700, parallel=True) == 2_000
        )
    assert spy.count(COLP) == 0 and spy.row_chunks >= 1

    spy.calls.clear()
    spy.row_chunks = 0
    assert (
        temp_db.insert_arrow("ByTable", table, batch_size=700, parallel=True) == 2_000
    )
    assert spy.count(COLP) == 3 and spy.row_chunks == 0  # one crossing per batch

    expected = _stored(temp_db, "Ref")
    assert len(expected) == 2_000 and expected[0]["val"][0] is None
    assert _stored(temp_db, "ByTable") == expected
