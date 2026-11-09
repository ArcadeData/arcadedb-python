"""GraphBatch.create_vertices: float32 vectors of any property cross as one matrix, on both wheels (#337).

A row value that is a Java ``float[]`` (what ``to_java_float_array`` returns on the JVM wheel) or a numpy float32 array (what it returns on
the native wheel) is a vector candidate for every property, declared ``ARRAY_OF_FLOATS`` or not. Before, the JVM wheel refused a Java
``float[]`` as a candidate and sent the whole load through the property-matrix path (a JPype conversion of every value of every row), and
both wheels wrote one tag per row for a float32 array. Now one float32 matrix per property and chunk crosses (``take_vector_columns``).

This file pins what is stored (bit for bit, for both wheels, with the one call both wheels share) and the shared split itself. The crossing
counts are in test_graph_batch_vector_crossings_jvm.py (JPype) and test_native_vertex_vector_crossings.py (ctypes).
"""

import numpy as np
import pytest

import arcadedb
from arcadedb import type_conversion as tc

N = 40
TYPES = ["Declared", "Free"]  # embedding declared ARRAY_OF_FLOATS, and not declared
SHAPES = ["to_java_float_array", "float32_ndarray"]


def _vectors(dim, n=N, seed=5):
    """float32 values a decimal text of 6 digits would not hold: -0.0, the largest, the smallest normal and the smallest subnormal."""
    x = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    x[0, 0] = np.float32(-0.0)
    x[1, 0] = np.finfo(np.float32).max
    x[1, 1] = np.finfo(np.float32).tiny
    x[1, 2] = np.finfo(np.float32).smallest_subnormal
    return x


def _make(shape):
    if shape == "to_java_float_array":
        return (
            arcadedb.to_java_float_array
        )  # a float[] on the JVM wheel, a float32 ndarray on the native wheel
    return lambda vector: vector


def _bits(stored):
    return (
        np.array(stored, dtype=np.float64).astype(np.float32).view(np.uint32).tolist()
    )


@pytest.fixture
def db(temp_db):
    for name in ("Declared", "Bulk", "One"):
        temp_db.command("sql", f"CREATE VERTEX TYPE {name}")
        temp_db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    temp_db.command("sql", "CREATE PROPERTY Declared.embedding ARRAY_OF_FLOATS")
    temp_db.command("sql", "CREATE PROPERTY Declared.born DATETIME")
    temp_db.command("sql", "CREATE VERTEX TYPE Free")
    temp_db.command("sql", "CREATE PROPERTY Free.id LONG")
    temp_db.command("sql", "CREATE PROPERTY Free.born DATETIME")
    temp_db.command("sql", "CREATE PROPERTY Bulk.embedding ARRAY_OF_FLOATS")
    temp_db.command("sql", "CREATE PROPERTY One.embedding ARRAY_OF_FLOATS")
    return temp_db


def _stored(db, type_name, prop="embedding"):
    sql = f"SELECT id, {prop} AS v FROM {type_name}"  # nosec B608 - test-owned names
    return {r["id"]: r["v"] for r in db.query("sql", sql).to_list()}


def _count(db, type_name):
    sql = f"SELECT count(*) AS c FROM {type_name}"  # nosec B608 - test-owned name
    return db.query("sql", sql).first().get("c")


@pytest.mark.parametrize("type_name", TYPES)
@pytest.mark.parametrize("dim", [64, 96])
@pytest.mark.parametrize("shape", SHAPES)
def test_float32_vectors_are_stored_bit_equal(db, type_name, dim, shape):
    x = _vectors(dim)
    make = _make(shape)
    with db.graph_batch() as gb:
        rids = gb.create_vertices(
            type_name, [{"id": i, "embedding": make(x[i])} for i in range(N)]
        )
    assert len(rids) == N and len(set(rids)) == N
    stored = _stored(db, type_name)
    assert sorted(stored) == list(range(N))
    for i in range(N):
        assert _bits(stored[i]) == x[i].view(np.uint32).tolist()
    for i in (0, 17, N - 1):  # the RIDs come back in row order
        sql = f"SELECT id FROM {rids[i]}"  # nosec B608 - a RID from this test
        assert db.query("sql", sql).first().get("id") == i


@pytest.mark.parametrize("shape", SHAPES)
def test_the_matrix_stores_what_the_per_vertex_path_stores(db, shape):
    x = _vectors(32, 12)
    rows = [{"id": i, "embedding": _make(shape)(x[i])} for i in range(12)]
    with db.graph_batch() as gb:
        gb.create_vertices("Bulk", rows)
    with db.graph_batch() as gb:
        for row in rows:
            gb.create_vertex("One", **row)
    assert _stored(db, "Bulk") == _stored(db, "One")
    assert _count(db, "Bulk") == 12


def test_float64_numpy_rows_keep_their_documented_values(db):
    x64 = np.random.default_rng(3).random((N, 64)) * 1000.0 + 1e-3
    rows = [{"id": i, "embedding": x64[i]} for i in range(N)]
    with db.graph_batch() as gb:
        gb.create_vertices("Declared", rows)
        gb.create_vertices("Free", rows)
    declared, free = _stored(db, "Declared"), _stored(db, "Free")
    for i in range(N):
        # declared ARRAY_OF_FLOATS: the engine stores floats, so the doubles round to float32
        assert _bits(declared[i]) == x64[i].astype(np.float32).view(np.uint32).tolist()
        # undeclared: the list keeps its doubles, exactly
        assert np.array(free[i], dtype=np.float64).tolist() == x64[i].tolist()


def test_lists_of_floats_keep_their_documented_values(db):
    x64 = np.random.default_rng(4).random((N, 40)) * 1000.0 + 1e-3
    rows = [{"id": i, "embedding": x64[i].tolist()} for i in range(N)]
    with db.graph_batch() as gb:
        gb.create_vertices("Declared", rows)
        gb.create_vertices("Free", rows)
    declared, free = _stored(db, "Declared"), _stored(db, "Free")
    for i in range(N):
        assert _bits(declared[i]) == x64[i].astype(np.float32).view(np.uint32).tolist()
        assert np.array(free[i], dtype=np.float64).tolist() == x64[i].tolist()


@pytest.mark.parametrize("type_name", TYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ragged_null_and_missing_vectors_take_the_fallback(db, type_name, shape):
    x = _vectors(16, 6)
    make = _make(shape)
    rows = [
        {"id": 0, "embedding": make(x[0])},
        {"id": 1, "embedding": make(x[1][:9])},  # another length
        {"id": 2, "embedding": None},
        {"id": 3},  # no vector
        {"id": 4, "embedding": make(x[4])},
        None,  # a vertex without properties
    ]
    with db.graph_batch() as gb:
        assert len(gb.create_vertices(type_name, rows)) == 6
    stored = _stored(db, type_name)
    assert (
        _count(db, type_name) == 6 and None in stored
    )  # the vertex of the None row has no id
    assert _bits(stored[0]) == x[0].view(np.uint32).tolist()
    assert _bits(stored[1]) == x[1][:9].view(np.uint32).tolist()
    assert stored[2] is None and stored[3] is None
    assert _bits(stored[4]) == x[4].view(np.uint32).tolist()


@pytest.mark.parametrize("type_name", TYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_a_float64_row_among_float32_rows_loses_nothing(db, type_name, shape):
    x = _vectors(16, 8)
    make = _make(shape)
    rows = [{"id": i, "embedding": make(x[i]), "tag": f"t{i}"} for i in range(8)]
    rows[5] = {"id": 5, "tag": "t5", "embedding": make(x[5].astype(np.float64))}
    with db.graph_batch() as gb:
        gb.create_vertices(type_name, rows)
    stored, tags = _stored(db, type_name), _stored(db, type_name, "tag")
    assert [tags[i] for i in range(8)] == [f"t{i}" for i in range(8)]
    for i in range(8):
        if type_name == "Free" and i == 5 and shape == "float32_ndarray":
            # a float64 array on an undeclared property keeps its doubles (it is no float32 vector)
            assert (
                np.array(stored[i], dtype=np.float64).tolist()
                == x[5].astype(np.float64).tolist()
            )
        else:
            assert _bits(stored[i]) == x[i].view(np.uint32).tolist()


@pytest.mark.parametrize("type_name", TYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_the_other_properties_stay_with_their_rows(db, type_name, shape):
    x = _vectors(16, 12)
    make = _make(shape)
    rows = []
    for i in range(12):
        row = {
            "id": i,
            "name": f"n{i}",
            "score": i * 0.5,
            "flag": i % 2 == 0,
            "note": None,
            "embedding": make(x[i]),
        }
        if i % 4 == 3:  # another key order: the rows no longer share one
            row = dict(reversed(list(row.items())))
        rows.append(row)
    with db.graph_batch() as gb:
        gb.create_vertices(type_name, rows)
    sql = f"SELECT id, name, score, flag, embedding FROM {type_name} ORDER BY id"  # nosec B608 - test-owned name
    got = db.query("sql", sql).to_list()
    assert [r["id"] for r in got] == list(range(12))
    for i, r in enumerate(got):
        assert (
            r["name"] == f"n{i}" and r["score"] == i * 0.5 and r["flag"] is (i % 2 == 0)
        )
        assert _bits(r["embedding"]) == x[i].view(np.uint32).tolist()


@pytest.mark.parametrize("type_name", TYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_two_vector_properties_and_a_date_time_in_one_load(db, type_name, shape):
    from datetime import datetime

    x, y = _vectors(8, 20), _vectors(8, 20, seed=9)
    make = _make(shape)
    rows = [
        {
            "id": i,
            "embedding": make(x[i]),
            "second": make(y[i]),
            "born": datetime(2020, 1, 2, 3, 4, 5),
        }
        for i in range(20)
    ]
    with db.graph_batch() as gb:
        gb.create_vertices(type_name, rows)
    first, second = _stored(db, type_name), _stored(db, type_name, "second")
    for i in range(20):
        assert _bits(first[i]) == x[i].view(np.uint32).tolist()
        assert _bits(second[i]) == y[i].view(np.uint32).tolist()
    sql = f"SELECT born FROM {type_name} WHERE id = 7"  # nosec B608 - test-owned name
    assert db.query("sql", sql).first().get("born") == datetime(2020, 1, 2, 3, 4, 5)


@pytest.mark.parametrize("type_name", TYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_a_load_over_the_chunk_boundary(db, type_name, shape):
    make = _make(shape)
    with db.graph_batch() as gb:
        # the rows per crossing: the JVM wheel's 100,000, the native wheel's commit_every (1000 unless graph_batch() got another value)
        chunk = getattr(gb, "_vertex_chunk", None) or gb._BULK_CHUNK
        n = chunk + 3
        x = _vectors(4, n)
        rids = gb.create_vertices(
            type_name, [{"id": i, "embedding": make(x[i])} for i in range(n)]
        )
    assert len(rids) == n and len(set(rids)) == n
    assert _count(db, type_name) == n
    stored = _stored(db, type_name)
    got = np.array([stored[i] for i in range(n)], dtype=np.float64).astype(np.float32)
    assert (got.view(np.uint32) == x.view(np.uint32)).all()
    for i in (
        0,
        chunk - 1,
        chunk,
        n - 1,
    ):  # the RIDs come back in row order across the boundary
        sql = f"SELECT id FROM {rids[i]}"  # nosec B608 - a RID from this test
        assert db.query("sql", sql).first().get("id") == i


@pytest.mark.parametrize("shape", SHAPES)
def test_insert_many_takes_the_same_vectors(db, shape):
    db.command("sql", "CREATE DOCUMENT TYPE Doc")
    db.command("sql", "CREATE PROPERTY Doc.id LONG")
    db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    x = _vectors(32, N)
    make = _make(shape)
    assert (
        db.insert_many("Doc", [{"id": i, "embedding": make(x[i])} for i in range(N)])
        == N
    )
    stored = _stored(db, "Doc")
    for i in range(N):
        assert _bits(stored[i]) == x[i].view(np.uint32).tolist()


# ---- the split itself (no database): the one function both wheels call


def _rows(x, **extra):
    return [{"a": i, "b": f"s{i}", "v": x[i], **extra} for i in range(len(x))]


def test_take_vector_columns_takes_a_float32_column_of_any_property():
    x = _vectors(5, 7)
    names, matrices, rest = tc.take_vector_columns(_rows(x), (), undeclared=True)
    assert names == ["v"] and matrices[0].dtype == np.float32
    assert matrices[0].shape == (7, 5) and matrices[0].flags.c_contiguous
    assert matrices[0].tobytes() == x.tobytes()  # bit for bit
    assert rest == [{"a": i, "b": f"s{i}"} for i in range(7)]
    assert all(list(row) == ["a", "b"] for row in rest)


def test_take_vector_columns_leaves_an_undeclared_column_that_is_no_float32_vector():
    x = _vectors(5, 7)
    as_lists = [{"v": row.tolist()} for row in x]
    as_doubles = [{"v": row.astype(np.float64)} for row in x]
    assert tc.take_vector_columns(as_lists, (), undeclared=True) is None
    assert tc.take_vector_columns(as_doubles, (), undeclared=True) is None
    assert tc.take_vector_columns(as_lists, ["v"])[1][0].shape == (7, 5)
    assert tc.take_vector_columns(as_doubles, ["v"])[1][0].shape == (7, 5)
    assert (
        tc.take_vector_columns(_rows(x), ()) is None
    )  # undeclared names are tried only when asked


def test_take_vector_columns_declines_a_ragged_null_or_missing_column():
    x = _vectors(5, 4)
    for rows in (
        [{"v": x[0]}, {"v": x[1][:3]}],
        [{"v": x[0]}, {"v": None}],
        [{"v": x[0]}, {"w": 1}],
        [{"v": x[0]}, None],
        [{"v": np.zeros(0, dtype=np.float32)}],
        [{"v": x}],  # a 2-D array per row
    ):
        assert tc.take_vector_columns(rows, ["v"], undeclared=True) is None


def test_take_vector_columns_rebuilds_the_rest_rows_in_each_rows_own_key_order():
    x = _vectors(3, 2)
    rows = [{"a": 1, "v": x[0], "b": 2}, {"b": 3, "v": x[1], "a": 4}]
    rest = tc.take_vector_columns(rows, (), undeclared=True)[2]
    assert rest == [{"a": 1, "b": 2}, {"b": 3, "a": 4}]
    assert [list(row) for row in rest] == [["a", "b"], ["b", "a"]]
    only = tc.take_vector_columns([{"v": x[0]}, {"v": x[1]}], (), undeclared=True)[2]
    assert only == [{}, {}] and only[0] is not only[1]
    two = [{"x": r, "y": r * 2, "k": i} for i, r in enumerate(x)]
    names, _, rest = tc.take_vector_columns(two, (), undeclared=True)
    assert sorted(names) == ["x", "y"] and rest == [{"k": 0}, {"k": 1}]


def test_take_vector_columns_takes_a_strided_or_read_only_row():
    wide = _vectors(10, 6)
    strided = tc.take_vector_columns(
        [{"v": wide[i, ::2]} for i in range(6)], (), undeclared=True
    )[1][0]
    assert strided.tobytes() == np.ascontiguousarray(wide[:, ::2]).tobytes()
    frozen = wide.copy()
    frozen.setflags(write=False)
    matrix = tc.take_vector_columns(
        [{"v": frozen[i]} for i in range(6)], (), undeclared=True
    )[1][0]
    assert matrix.tobytes() == wide.tobytes() and matrix.flags.writeable
