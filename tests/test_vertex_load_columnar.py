"""``insert_dataframe`` and ``insert_arrow`` load a vertex type through the columnar insert, and the stored vertices, their indexes and
their buckets are those of the old route, ``GraphBatch.create_vertices`` (ArcadeData/arcadedb-python #319).

Both routes are ``newVertex``, ``set`` per property and ``save`` on the caller's thread inside a transaction, so the records, the
index entries and the bucket each vertex lands in come out the same; the one difference the engine code shows is that a GraphBatch
turns the write-ahead log off for its load (its ``useWAL`` default) while the columnar insert uses the database's own settings, as the
document loads of the same calls always did. The old route is the private reference here: a type loaded through
``graph_batch().create_vertices`` and a type loaded through ``insert_dataframe`` are compared value by value and type by type, with the
index lookups and the per-bucket record counts. A spy on ``create_vertices`` proves which route ran (the old code calls it, the new code
must not). Runs on both wheels.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from arcadedb.exceptions import ArcadeDBError

N = 37  # not a multiple of the batch size or of the bucket count


def _type(db, name, buckets=3):
    db.command("sql", f"CREATE VERTEX TYPE {name} BUCKETS {buckets}")
    db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    db.command("sql", f"CREATE PROPERTY {name}.grp INTEGER")
    db.command("sql", f"CREATE PROPERTY {name}.emb ARRAY_OF_FLOATS")
    db.command("sql", f"CREATE INDEX ON {name} (id) UNIQUE")
    db.command("sql", f"CREATE INDEX ON {name} (grp) NOTUNIQUE")


def _frame(n=N):
    ids = np.arange(n, dtype=np.int64)
    return pd.DataFrame(
        {
            "id": ids,
            "grp": (ids % 5).astype(np.int32),
            "name": [None if i % 9 == 0 else f"n{i}-é" for i in range(n)],
            "score": [None if i % 4 == 0 else i * 0.25 for i in range(n)],
            "born": pd.to_datetime(
                [
                    dt.datetime(1969, 12, 31, 23, 59, 59, 999999) + dt.timedelta(days=i)
                    for i in range(n)
                ]
            ),
            "emb": [np.array([i, i + 0.5, -i], dtype=np.float32) for i in range(n)],
        }
    )


def _rows_of(frame):
    out = []
    for rec in frame.to_dict("records"):
        row = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and v != v):
                v = None  # an Arrow null, as the old route of insert_dataframe handed it over
            elif isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.generic):
                v = v.item()
            row[k] = v
        out.append(row)
    return out


def _old_route(db, name, frame):
    with db.graph_batch() as batch:
        batch.create_vertices(name, _rows_of(frame))


def _stored(db, name):
    sql = f"SELECT FROM {name} ORDER BY id"  # nosec B608 - test-owned type name
    return [
        {k: (type(v).__name__, v) for k, v in rec.items() if not k.startswith("@")}
        for rec in db.query("sql", sql).to_list()
    ]


def _bucket_counts(db, name):
    counts = []
    for i in range(3):
        sql = f"SELECT count(*) AS n FROM bucket:{name}_{i}"  # nosec B608 - test-owned names
        counts.append(db.query("sql", sql).first().get("n"))
    return sorted(counts)


def _index_view(db, name):
    by_id, by_grp = {}, {}
    for i in (0, 7, N - 1):
        sql = f"SELECT id, name FROM {name} WHERE id = {i}"  # nosec B608 - test-owned names
        by_id[i] = [
            (r.get("id"), r.get("name")) for r in db.query("sql", sql).to_list()
        ]
    for g in range(5):
        sql = f"SELECT id FROM {name} WHERE grp = {g}"  # nosec B608 - test-owned names
        by_grp[g] = sorted(r.get("id") for r in db.query("sql", sql).to_list())
    return by_id, by_grp


@pytest.fixture
def old_and_new(temp_db):
    _type(temp_db, "Old")
    _type(temp_db, "New")
    _old_route(temp_db, "Old", _frame())
    return temp_db


def test_the_vertices_indexes_and_buckets_equal_those_of_graph_batch(
    old_and_new, monkeypatch
):
    db = old_and_new
    with db.graph_batch() as batch:
        graph_batch_class = type(batch)

    def refused(*args, **kwargs):
        raise AssertionError(
            "a vertex load through insert_dataframe went through GraphBatch.create_vertices"
        )

    monkeypatch.setattr(graph_batch_class, "create_vertices", refused)
    assert db.insert_dataframe("New", _frame(), batch_size=10) == N
    expected = _stored(db, "Old")
    assert len(expected) == N and expected[7]["emb"][0] == "list"
    assert expected[0].get("name", ("NoneType", None))[1] is None
    assert sum(_bucket_counts(db, "Old")) == N and max(_bucket_counts(db, "Old")) < N
    assert _stored(db, "New") == expected
    assert _bucket_counts(db, "New") == _bucket_counts(db, "Old")
    assert _index_view(db, "New") == _index_view(db, "Old")
    assert db.lookup_by_key("New", ["id"], [7]).get("name") == "n7-é"
    assert (
        db.query("sql", "SELECT outE().size() AS e FROM New WHERE id = 7")
        .first()
        .get("e")
        == 0
    )  # a vertex


def test_insert_arrow_loads_a_vertex_type_the_same_way(old_and_new):
    db = old_and_new
    assert db.insert_arrow("New", pa.Table.from_pandas(_frame()), batch_size=16) == N
    assert _stored(db, "New") == _stored(db, "Old")


def test_a_duplicate_key_fails_and_rolls_the_batch_back(temp_db):
    _type(temp_db, "Dup")
    frame = _frame(20)
    frame.loc[15, "id"] = 3  # the unique index refuses it
    with pytest.raises(ArcadeDBError, match="Duplicat"):
        temp_db.insert_dataframe("Dup", frame, batch_size=50)
    assert temp_db.is_transaction_active() is False
    assert temp_db.query("sql", "SELECT count(*) AS n FROM Dup").first().get("n") == 0


def test_a_vertex_load_refuses_to_run_inside_a_transaction_the_caller_opened(temp_db):
    _type(temp_db, "Mine")
    with temp_db.transaction():
        with pytest.raises(
            ArcadeDBError, match="cannot run inside a transaction the caller opened"
        ):
            temp_db.insert_dataframe("Mine", _frame(5))
    assert temp_db.query("sql", "SELECT count(*) AS n FROM Mine").first().get("n") == 0
    assert temp_db.insert_dataframe("Mine", _frame(5)) == 5  # outside one it loads


def test_a_document_load_still_joins_the_callers_transaction(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Doc")
    with pytest.raises(RuntimeError, match="stop"):
        with temp_db.transaction():
            temp_db.insert_dataframe(
                "Doc", pd.DataFrame({"id": [1, 2, 3]}), batch_size=2
            )
            raise RuntimeError("stop")
    assert temp_db.query("sql", "SELECT count(*) AS n FROM Doc").first().get("n") == 0
