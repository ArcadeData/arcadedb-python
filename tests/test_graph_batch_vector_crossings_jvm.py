"""GraphBatch.create_vertices (the JVM wheel): a Java float[] or float32 array crosses JPype as one matrix per property and chunk (#337).

Before, a Java ``float[]`` (what ``to_java_float_array`` returns) was no vector candidate: ``classify_vertex_rows`` gave up on the first
row, and the whole load went through ``_to_java_property_matrix``, which converts every value of every row with ``convert_python_to_java``,
builds one ``Object[]`` per row and one ``Object[][]`` for the call (an unwrapped Python loop that was 59 to 61 percent of the wall of the
e2 and l3d loads in the untimed ingest profile of 26.11.1). A float32 numpy array on a property that is not declared ``ARRAY_OF_FLOATS``
took the same way. Now ``take_vector_columns`` stacks the column once (numpy reads a Java array through the buffer protocol), one
``float[]`` copy and one ``FloatVectors.Matrix`` cross per property, and one ``VertexBatcher.createVerticesJson`` call per chunk.

The JPype crossings are counted by wrapping the bridge classes and the Python-side conversion; the stored values are in
test_vertex_vector_loads.py. The JPype bridge does not exist on the native wheel (its counts are in test_native_vertex_vector_crossings.py),
so conftest.py does not collect this file there.
"""

import jpype
import numpy as np
import pytest

import arcadedb
from arcadedb import graph_batch as gbm
from arcadedb import type_conversion as tc
from arcadedb.graph_batch import GraphBatch

ROWS, CHUNK, DIM = 250, 100, 16  # three chunks: 100, 100, 50


class _Counting:
    """Forwards every call of a Java class (or of the bridge's FloatVectors) and records its name."""

    def __init__(self, target, log):
        self._target, self._log = target, log

    def __getattr__(self, name):
        attribute = getattr(self._target, name)
        if not callable(attribute):
            return attribute

        def call(*args):
            self._log.append(name)
            return attribute(*args)

        return call


@pytest.fixture
def crossings(monkeypatch):
    """What the load does: property-matrix fallbacks, per-value conversions, bridge calls, Matrix constructions, per-row inserts."""
    log = {"fallback": [], "convert": [], "bridge": [], "matrix": [], "per_row": []}

    real_matrix_path = GraphBatch._to_java_property_matrix

    def fallback(self, rows):
        log["fallback"].append(len(rows))
        return real_matrix_path(self, rows)

    monkeypatch.setattr(GraphBatch, "_to_java_property_matrix", fallback)

    real_convert = tc.convert_python_to_java

    def convert(value):
        log["convert"].append(type(value).__name__)
        return real_convert(value)

    monkeypatch.setattr(tc, "convert_python_to_java", convert)
    monkeypatch.setattr(gbm, "convert_python_to_java", convert)

    real_jclass = jpype.JClass

    def jclass(name, *args, **kwargs):
        cls = real_jclass(name, *args, **kwargs)
        if name == "com.arcadedb.python.VertexBatcher":
            return _Counting(cls, log["bridge"])
        return cls

    monkeypatch.setattr(jpype, "JClass", jclass)

    real_vectors = tc._float_vectors_class

    def vectors_class():
        vectors = real_vectors()
        return None if vectors is None else _Counting(vectors, log["matrix"])

    monkeypatch.setattr(tc, "_float_vectors_class", vectors_class)

    from arcadedb import core

    real_per_row = core.Database._insert_many_per_row

    def per_row(self, type_name, rows):
        log["per_row"].append(len(rows))
        return real_per_row(self, type_name, rows)

    monkeypatch.setattr(core.Database, "_insert_many_per_row", per_row)
    monkeypatch.setattr(GraphBatch, "_BULK_CHUNK", CHUNK)
    return log


@pytest.fixture
def db(temp_db):
    for name in ("Declared", "Free", "Doc"):
        kind = "DOCUMENT" if name == "Doc" else "VERTEX"
        temp_db.command("sql", f"CREATE {kind} TYPE {name}")
        temp_db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    temp_db.command("sql", "CREATE PROPERTY Declared.embedding ARRAY_OF_FLOATS")
    temp_db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    return temp_db


def _x(n=ROWS, dim=DIM):
    return np.random.default_rng(7).standard_normal((n, dim)).astype(np.float32)


def _floats(x):
    return [jpype.JArray(jpype.JFloat)(row) for row in x]


@pytest.mark.parametrize("type_name", ["Declared", "Free"])
@pytest.mark.parametrize("shape", ["java_float_array", "float32_ndarray"])
def test_a_float32_vector_crosses_as_one_matrix_per_chunk(
    db, crossings, type_name, shape
):
    x = _x()
    vectors = _floats(x) if shape == "java_float_array" else list(x)
    rows = [{"id": i, "embedding": vectors[i]} for i in range(ROWS)]
    with db.graph_batch(parallel_flush=False) as gb:
        assert len(gb.create_vertices(type_name, rows)) == ROWS
    assert (
        crossings["fallback"] == []
    ), "the property-matrix path ran (a JPype conversion per value and per row)"
    assert crossings["convert"] == [], "a value was converted one at a time"
    assert (
        crossings["bridge"] == ["createVerticesJson"] * 3
    )  # one crossing per chunk of 100
    # one float32 matrix per chunk: `declared` is asked once per call, `Matrix` once per chunk
    assert crossings["matrix"].count("Matrix") == 3
    sql = f"SELECT id, embedding FROM {type_name}"  # nosec B608 - test-owned name
    stored = {r["id"]: r["embedding"] for r in db.query("sql", sql).to_list()}
    for i in range(ROWS):
        assert np.array(stored[i], dtype=np.float32).tobytes() == x[i].tobytes()


def test_two_vector_properties_cross_as_two_matrices_per_chunk(db, crossings):
    x, y = _x(), _x(dim=8)
    rows = [
        {"id": i, "embedding": jpype.JArray(jpype.JFloat)(x[i]), "second": y[i]}
        for i in range(ROWS)
    ]
    with db.graph_batch(parallel_flush=False) as gb:
        gb.create_vertices("Declared", rows)
    assert crossings["fallback"] == [] and crossings["convert"] == []
    assert crossings["bridge"] == ["createVerticesJson"] * 3
    assert crossings["matrix"].count("Matrix") == 6


def test_a_ragged_column_still_takes_the_property_matrix_path_for_its_chunk(
    db, crossings
):
    x = _x()
    vectors = _floats(x)
    rows = [{"id": i, "embedding": vectors[i]} for i in range(ROWS)]
    rows[120] = {"id": 120, "embedding": jpype.JArray(jpype.JFloat)(x[120][:5])}
    with db.graph_batch(parallel_flush=False) as gb:
        assert len(gb.create_vertices("Declared", rows)) == ROWS
    assert crossings["fallback"] == [100], "only the chunk of row 120"
    assert crossings["bridge"] == ["createVerticesJson"] * 2
    stored = {
        r["id"]: r["embedding"]
        for r in db.query("sql", "SELECT id, embedding FROM Declared").to_list()
    }
    assert np.array(stored[120], dtype=np.float32).tobytes() == x[120][:5].tobytes()
    assert np.array(stored[0], dtype=np.float32).tobytes() == x[0].tobytes()


def test_a_java_double_array_is_a_vector_only_on_a_declared_property(db, crossings):
    x64 = np.random.default_rng(8).random((ROWS, DIM)) * 1000.0 + 1e-3
    rows = [
        {"id": i, "embedding": jpype.JArray(jpype.JDouble)(x64[i])} for i in range(ROWS)
    ]
    with db.graph_batch(parallel_flush=False) as gb:
        gb.create_vertices("Declared", rows)
    assert (
        crossings["fallback"] == []
        and crossings["bridge"] == ["createVerticesJson"] * 3
    )
    declared = {
        r["id"]: r["embedding"]
        for r in db.query("sql", "SELECT id, embedding FROM Declared").to_list()
    }
    for i in (
        0,
        99,
        100,
        ROWS - 1,
    ):  # the engine stores floats for the declared property
        want = x64[i].astype(np.float32).tobytes()
        assert np.array(declared[i], dtype=np.float32).tobytes() == want
    del crossings["fallback"][:], crossings["bridge"][:]
    with db.graph_batch(parallel_flush=False) as gb:
        gb.create_vertices("Free", rows)
    assert crossings["fallback"] == [
        ROWS
    ], "undeclared: the double[] keeps the matrix path"
    free = {
        r["id"]: r["embedding"]
        for r in db.query("sql", "SELECT id, embedding FROM Free").to_list()
    }
    assert np.array(free[5], dtype=np.float64).tolist() == x64[5].tolist()


def test_insert_many_sends_a_java_float_array_column_through_the_matrix(db, crossings):
    x = _x(50)
    arrays = _floats(x)
    rows = [{"id": i, "embedding": arrays[i]} for i in range(50)]
    assert db.insert_many("Doc", rows) == 50
    assert crossings["per_row"] == [], "the column went row by row"
    assert crossings["matrix"].count("Matrix") == 1
    stored = {
        r["id"]: r["embedding"]
        for r in db.query("sql", "SELECT id, embedding FROM Doc").to_list()
    }
    for i in range(50):
        assert np.array(stored[i], dtype=np.float32).tobytes() == x[i].tobytes()


def test_take_vector_columns_reads_java_arrays_through_one_stack(db):
    float_array, double_array = jpype.JArray(jpype.JFloat), jpype.JArray(jpype.JDouble)
    x = _x(6, 5)
    floats = [{"v": float_array(row), "k": i} for i, row in enumerate(x)]
    names, matrices, rest = tc.take_vector_columns(floats, (), undeclared=True)
    assert names == ["v"] and matrices[0].dtype == np.float32
    assert matrices[0].tobytes() == x.tobytes() and rest == [{"k": i} for i in range(6)]
    doubles = [{"v": double_array(row.astype(np.float64))} for row in x]
    assert tc.take_vector_columns(doubles, (), undeclared=True) is None
    assert tc.take_vector_columns(doubles, ["v"])[1][0].tobytes() == x.tobytes()
    ints = [{"v": jpype.JArray(jpype.JInt)([1, 2, 3])} for _ in range(3)]
    assert tc.take_vector_columns(ints, ["v"], undeclared=True) is None
    assert not tc.is_vector_candidate(ints[0]["v"])
    assert tc.is_vector_candidate(floats[0]["v"]) and tc.is_vector_candidate(
        doubles[0]["v"]
    )
    assert isinstance(arcadedb.to_java_float_array(x[0]), float_array)
