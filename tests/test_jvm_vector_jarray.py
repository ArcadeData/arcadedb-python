"""A Java float[] or double[] vector crosses as one float32 matrix, like a list or a numpy row (JVM build).

``arcadedb.to_java_float_array`` returns a JPype ``float[]`` on the JVM build (a float32 ndarray on the native one), and the
benchmark harnesses build every vector with it. ``is_vector_candidate`` used to accept only a list or an ndarray, so a chunk of
such vectors left the json-bulk path for the property-matrix path, 2 to 2.4 times slower per vertex. These tests pin that the
chunk takes the split path, that the stored values are the ones the declined path stores, and that a Java array of another
kind, or a chunk with a hole, still declines. The path is observed with spies on ``split_float_vectors`` and
``GraphBatch._to_java_property_matrix``, so an unchanged module fails the first assert and not only a timing.
"""

import jpype
import numpy as np
import pytest

import arcadedb
from arcadedb import graph_batch, type_conversion
from arcadedb.graph_batch import GraphBatch

DIM = 16
N = 40


@pytest.fixture
def vx_db(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE Vx")
    temp_db.command("sql", "CREATE PROPERTY Vx.id LONG")
    temp_db.command("sql", "CREATE PROPERTY Vx.embedding ARRAY_OF_FLOATS")
    temp_db.command("sql", "CREATE VERTEX TYPE Ref")
    temp_db.command("sql", "CREATE PROPERTY Ref.id LONG")
    temp_db.command("sql", "CREATE PROPERTY Ref.embedding ARRAY_OF_FLOATS")
    temp_db.command("sql", "CREATE DOCUMENT TYPE Doc")
    temp_db.command("sql", "CREATE PROPERTY Doc.id LONG")
    temp_db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    return temp_db


@pytest.fixture
def paths(monkeypatch):
    """Spies: how often the vector split ran and took a column, and how often the property-matrix path ran."""
    seen = {"split": 0, "taken": 0, "matrix": 0}
    real_split = graph_batch.split_float_vectors
    real_matrix = GraphBatch._to_java_property_matrix

    def split(*args, **kwargs):
        seen["split"] += 1
        result = real_split(*args, **kwargs)
        if result is not None:
            seen["taken"] += 1
        return result

    def matrix(self, rows):
        seen["matrix"] += 1
        return real_matrix(self, rows)

    monkeypatch.setattr(graph_batch, "split_float_vectors", split)
    monkeypatch.setattr(GraphBatch, "_to_java_property_matrix", matrix)
    return seen


def _doubles(n=N, dim=DIM, seed=11):
    """float64 values a float32 cannot hold, so a path that kept doubles or lost bits shows."""
    return np.random.default_rng(seed).random((n, dim)) * 1000.0 + 1e-3


def _jfloat(row):
    return arcadedb.to_java_float_array(row)


def _jdouble(row):
    return jpype.JArray(jpype.JDouble)(row)


def _stored(db, type_name="Vx"):
    """{id: stored vector as float32 bytes} for every record of the type."""
    query = (
        f"SELECT id, embedding FROM {type_name}"  # nosec B608 - test-owned type name
    )
    out = {}
    for r in db.query("sql", query).to_list():
        out[r["id"]] = np.array(r["embedding"], dtype=np.float32).tobytes()
    return out


def _create(db, rows):
    with db.graph_batch() as gb:
        return gb.create_vertices("Vx", rows)


def test_to_java_float_array_is_a_float_array_on_this_build(
    vx_db,
):  # vx_db starts the JVM
    assert type(_jfloat([1.0, 2.0])) is jpype.JArray(jpype.JFloat)


class TestDecision:
    def _split(self, db, rows):
        return type_conversion.split_float_vectors(db.get_java_database(), "Vx", rows)

    def test_float_and_double_arrays_are_taken_out(self, vx_db):
        x = _doubles()
        for make in (_jfloat, _jdouble):
            rows = [{"id": i, "embedding": make(x[i].tolist())} for i in range(N)]
            rest, names, matrices = self._split(vx_db, rows)
            assert [str(n) for n in names] == ["embedding"]
            assert len(matrices) == 1
            assert rest == [{"id": i} for i in range(N)]

    def test_candidates(self, vx_db):
        assert type_conversion.is_vector_candidate(_jfloat([1.0]))
        assert type_conversion.is_vector_candidate(_jdouble([1.0]))
        assert not type_conversion.is_vector_candidate(jpype.JArray(jpype.JInt)([1]))
        assert not type_conversion.is_vector_candidate(jpype.JArray(jpype.JLong)([1]))
        assert not type_conversion.is_vector_candidate(
            jpype.JArray(jpype.JString)(["a"])
        )
        assert not type_conversion.is_vector_candidate("a string")
        assert not type_conversion.is_vector_candidate(None)

    @pytest.mark.parametrize(
        "case",
        [
            "int_array_first",
            "int_array_after_float_array",
            "long_array",
            "none_after_float_array",
            "missing_after_float_array",
            "ragged",
            "scalar_after_float_array",
            "empty",
        ],
    )
    def test_anything_else_is_declined(self, vx_db, case):
        # built here: a Java array type needs the JVM, which is not running while pytest collects
        ints = jpype.JArray(jpype.JInt)([1, 2])
        rows = {
            "int_array_first": [{"embedding": ints}] * 3,
            "int_array_after_float_array": [
                {"embedding": _jfloat([1.0, 2.0])},
                {"embedding": ints},
            ],
            "long_array": [{"embedding": jpype.JArray(jpype.JLong)([1, 2])}] * 3,
            "none_after_float_array": [
                {"embedding": _jfloat([1.0, 2.0])},
                {"embedding": None},
            ],
            "missing_after_float_array": [
                {"embedding": _jfloat([1.0, 2.0])},
                {"id": 3},
            ],
            "ragged": [
                {"embedding": _jfloat([1.0, 2.0])},
                {"embedding": _jfloat([1.0, 2.0, 3.0])},
            ],
            "scalar_after_float_array": [
                {"embedding": _jfloat([1.0, 2.0])},
                {"embedding": 1.5},
            ],
            "empty": [{"embedding": _jfloat([])}] * 3,
        }[case]
        assert self._split(vx_db, rows) is None


class TestCreateVertices:
    def test_float_arrays_take_the_split_path_and_store_the_floats(self, vx_db, paths):
        x = _doubles()
        x32 = x.astype(np.float32)
        rows = [{"id": i, "embedding": _jfloat(x32[i].tolist())} for i in range(N)]
        rids = _create(vx_db, rows)
        assert len(rids) == N and len(set(rids)) == N
        assert paths["taken"] == 1, "the chunk goes through the float32 matrix"
        assert paths["matrix"] == 0, "the property-matrix path did not run"
        stored = _stored(vx_db)
        assert sorted(stored) == list(range(N))
        for i in range(N):
            assert stored[i] == x32[i].tobytes(), i

    def test_double_arrays_store_what_the_declined_path_stores(
        self, vx_db, paths, monkeypatch
    ):
        x = _doubles()
        rows = [{"id": i, "embedding": _jdouble(x[i].tolist())} for i in range(N)]
        _create(vx_db, rows)
        assert paths["taken"] == 1 and paths["matrix"] == 0
        for i, got in _stored(vx_db).items():
            assert got == x[i].astype(np.float32).tobytes(), i
        # the reference is the path the chunk took before: the json-bulk path declines, the property matrix stores
        monkeypatch.setattr(GraphBatch, "_create_vertices_json_bulk", lambda *a: None)
        with vx_db.graph_batch() as gb:
            gb.create_vertices("Ref", rows)
        assert paths["matrix"] == 1
        reference = _stored(vx_db, "Ref")
        assert reference == _stored(
            vx_db
        ), "the matrix path and the split path store the same bits"

    def test_the_rids_come_back_in_row_order(self, vx_db, paths):
        x = _doubles()
        rows = [{"id": i, "embedding": _jfloat(x[i].tolist())} for i in range(N)]
        rids = _create(vx_db, rows)
        assert paths["matrix"] == 0
        for i in (0, 17, N - 1):
            query = f"SELECT id FROM {rids[i]}"  # nosec B608 - a RID from this test
            assert vx_db.query("sql", query).first().get("id") == i

    def test_float_arrays_and_lists_in_one_chunk(self, vx_db, paths):
        x = _doubles()
        x32 = x.astype(np.float32)
        rows = [
            {
                "id": i,
                "embedding": (_jfloat(x32[i].tolist()) if i % 2 else x32[i].tolist()),
            }
            for i in range(N)
        ]
        _create(vx_db, rows)
        assert paths["taken"] == 1 and paths["matrix"] == 0
        for i, got in _stored(vx_db).items():
            assert got == x32[i].tobytes(), i

    def test_a_chunk_with_a_none_vector_declines_as_before(self, vx_db, paths):
        x32 = _doubles(10).astype(np.float32)
        rows = [{"id": i, "embedding": _jfloat(x32[i].tolist())} for i in range(10)]
        rows[4] = {"id": 4, "embedding": None}
        rows[7] = {"id": 7}
        _create(vx_db, rows)
        assert paths["taken"] == 0, "a hole keeps the vectors out of the matrix"
        assert paths["matrix"] == 1
        stored = _stored(vx_db)
        for i in range(10):
            if i in (4, 7):
                continue
            assert stored[i] == x32[i].tobytes(), i
        count = vx_db.query("sql", "SELECT count(*) AS c FROM Vx").first().get("c")
        assert count == 10

    def test_an_int_array_still_declines(self, vx_db, paths):
        rows = [
            {"id": i, "embedding": jpype.JArray(jpype.JInt)([i, i + 1, i + 2])}
            for i in range(5)
        ]
        try:
            _create(vx_db, rows)
        except arcadedb.ArcadeDBError:
            pass  # the engine may refuse an int[] for a float vector: either way the matrix path was the one asked
        assert paths["taken"] == 0
        assert paths["matrix"] == 1

    def test_other_properties_still_cross_in_the_json(self, vx_db, paths):
        vx_db.command("sql", "CREATE PROPERTY Vx.name STRING")
        x32 = _doubles(5).astype(np.float32)
        rows = [
            {"id": i, "name": f"v{i}", "embedding": _jfloat(x32[i].tolist())}
            for i in range(5)
        ]
        _create(vx_db, rows)
        assert paths["taken"] == 1 and paths["matrix"] == 0
        got = {
            r["id"]: r["name"]
            for r in vx_db.query("sql", "SELECT id, name FROM Vx").to_list()
        }
        assert got == {i: f"v{i}" for i in range(5)}


class TestInsertMany:
    """insert_many shares is_vector_candidate and split_float_vectors, so a Java array takes the same matrix there."""

    def test_float_arrays_cross_as_a_matrix(self, vx_db, monkeypatch):
        calls = []
        real = type_conversion.split_float_vectors
        from arcadedb import core

        def spy(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append(result is not None)
            return result

        monkeypatch.setattr(core, "split_float_vectors", spy)
        x32 = _doubles().astype(np.float32)
        rows = [{"id": i, "embedding": _jfloat(x32[i].tolist())} for i in range(N)]
        assert vx_db.insert_many("Doc", rows) == N
        assert calls == [True], "the chunk's vectors left the JSON text"
        for i, got in _stored(vx_db, "Doc").items():
            assert got == x32[i].tobytes(), i
