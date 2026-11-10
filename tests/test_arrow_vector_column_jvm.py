"""insert_arrow and insert_dataframe with a vector column on the JVM wheel: the column crosses as one float32 matrix, not as one Python list of
Python floats per row, and what is stored is unchanged.

A ``FixedSizeList<float32>`` (or list, or large list) column into an ``ARRAY_OF_FLOATS`` property went ``chunk.to_pylist()`` (one Python float
per element, 31.5 MiB of Python heap for one 10,000 x 96 batch against a 3.7 MiB flat buffer), then ``_column_to_java`` converted every row
with ``convert_python_to_java`` (an ``ArrayList<Double>`` of 96 boxed doubles per row, 100,000 conversions for the 10 x 10,000 rows of the
study) and the engine narrowed each list back to a ``float[]``. In the untimed profile that conversion was about 63 percent of the call,
and the boxed doubles are what the JVM heap held (``heap used`` 183 to 195 MiB at the end against 69 MiB live after a collection). Now a
float32 column with one length and no null stays an Arrow buffer, ``_float32_matrix`` lays it out as one 2-D float32 numpy matrix, and
``_column_to_java`` sends it as one ``float[]`` inside the bridge's ``FloatVectors.Matrix`` (the route a 2-D numpy column of
``insert_columns`` already took); the documents get a row each Java-side.

Three tests fail on the code before (they count the per-row conversions, the matrix crossings and the Python heap of one batch); the others
pin what the old route stored, so the new one must store the same: count, an order-independent digest of every row's vector bytes read back
through SQL and through the result API, both again after close and reopen, the special float32 values, a null row, a null element, vectors
of different length, a wrong dimension against a vector index, float64 and integer lists (which are NOT taken: the old route refuses a
value past the float range and rounds the rest, a float32 cast would store an infinity), and the properties that are not declared
``ARRAY_OF_FLOATS`` (they keep a list). The JVM build only: it counts JPype crossings and reads Java-side values, so conftest.py does not
collect this file on the native wheel (whose vector columns stay Arrow arrays, test_native_columnar_memory.py).
"""

import hashlib
import tracemalloc

import numpy as np
import pyarrow as pa
import pytest

import arcadedb
from arcadedb import core
from arcadedb.exceptions import ArcadeDBError

F32 = np.float32
SHAPES = ["fixed_size", "list", "large_list"]
TYPES = ["VERTEX", "DOCUMENT"]


def _matrix(n, dim, start=0, specials=True):
    """float32 vectors, a pure function of the row number, with the values a decimal text of 6 digits would not hold in rows 0 and 1."""
    r = np.arange(start, start + n, dtype=np.int64)
    j = np.arange(dim, dtype=np.int64)
    m = (((r[:, None] * 257 + j[None, :] * 17) % 1000).astype(F32)) / F32(1000.0)
    if specials and start == 0 and n > 1 and dim > 3:
        m[0, 0] = F32(-0.0)
        m[1, 0] = np.finfo(F32).max
        m[1, 1] = np.finfo(F32).tiny
        m[1, 2] = np.finfo(F32).smallest_subnormal
    return m


def _column(matrix, shape):
    dim = matrix.shape[1]
    flat = pa.array(matrix.reshape(-1), type=pa.float32())
    if shape == "fixed_size":
        return pa.FixedSizeListArray.from_arrays(flat, dim)
    kind = pa.list_ if shape == "list" else pa.large_list
    rows = [row.tolist() for row in matrix]
    return pa.array(rows, type=kind(pa.float32()))


def _table(matrix, shape, start=0):
    n = matrix.shape[0]
    return pa.table(
        {
            "k": pa.array(np.arange(start, start + n), type=pa.int32()),
            "embedding": _column(matrix, shape),
        }
    )


def _make(db, name, kind="VERTEX", embedding="ARRAY_OF_FLOATS", index=None):
    db.command("sql", f"CREATE {kind} TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.k INTEGER")
    if embedding:
        db.command("sql", f"CREATE PROPERTY {name}.embedding {embedding}")
    if index:
        db.command(
            "sql",
            f"CREATE INDEX ON {name} (embedding) LSM_VECTOR METADATA {{dimensions: {index}, similarity: 'COSINE'}}",
        )


def _count(db, name):
    sql = f"SELECT count(*) AS c FROM {name}"  # nosec B608 - test-owned name
    return db.query("sql", sql).first().get("c")


def _key(k, vector_bytes):
    return int.from_bytes(
        hashlib.sha256(int(k).to_bytes(8, "little") + vector_bytes).digest()[:16],
        "little",
    )


def _expected_digest(matrix, start=0):
    """The digest the stored table must have, from the generator, never from the database."""
    h = 0
    for i, row in enumerate(matrix):
        h ^= _key(start + i, row.tobytes())
    return h


def _sql_digest(db, name):
    h, n = 0, 0
    sql = f"SELECT k, embedding FROM {name}"  # nosec B608 - test-owned name
    for row in db.query("sql", sql).to_list():
        h ^= _key(
            row["k"],
            np.asarray(row["embedding"], dtype=np.float64).astype(F32).tobytes(),
        )
        n += 1
    return n, h


def _api_digest(db, name):
    """The same digest through the result API (one Result per record, ``get`` per property), not through ``to_list``."""
    h, n = 0, 0
    result_set = db.query("sql", f"SELECT FROM {name}")  # nosec B608 - test-owned name
    for result in result_set:
        h ^= _key(
            result.get("k"),
            np.asarray(result.get("embedding"), dtype=np.float64).astype(F32).tobytes(),
        )
        n += 1
    return n, h


def _bits(values):
    return np.asarray(values, dtype=np.float64).astype(F32).view(np.uint32).tolist()


@pytest.fixture
def crossings(monkeypatch):
    """What one load does on the JPype side: the per-value conversions and the bridge's Matrix constructions."""
    log = {"convert": [], "matrix": []}
    real_convert = core.convert_python_to_java
    real_class = core._java_class

    def convert(value):
        log["convert"].append(type(value).__name__)
        return real_convert(value)

    class _Counting:
        def __init__(self, target):
            self._target = target

        def __call__(self, *args):
            log["matrix"].append(len(args[0]))
            return self._target(*args)

        def __getattr__(self, name):
            return getattr(self._target, name)

    def java_class(name):
        cls = real_class(name)
        return (
            _Counting(cls) if name == "com.arcadedb.python.FloatVectors$Matrix" else cls
        )

    monkeypatch.setattr(core, "convert_python_to_java", convert)
    monkeypatch.setattr(core, "_java_class", java_class)
    return log


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_count_digest_and_reopen_of_a_batched_vector_load(
    temp_db, temp_db_path_of, shape, kind
):
    """1,300 rows in batches of 500 (500, 500, 300: the batches are slices of one table): count, SQL digest, API digest, then the same after reopen."""
    n, dim = 1_300, 24
    matrix = _matrix(n, dim)
    _make(temp_db, "Vec", kind)
    assert temp_db.insert_arrow("Vec", _table(matrix, shape), batch_size=500) == n
    want = (n, _expected_digest(matrix))
    assert _count(temp_db, "Vec") == n
    assert _sql_digest(temp_db, "Vec") == want
    assert _api_digest(temp_db, "Vec") == want
    path = temp_db_path_of(temp_db)
    temp_db.close()
    reopened = arcadedb.open_database(path)
    try:
        assert _count(reopened, "Vec") == n
        assert _sql_digest(reopened, "Vec") == want
        assert _api_digest(reopened, "Vec") == want
    finally:
        reopened.close()


@pytest.fixture
def temp_db_path_of():
    def path_of(db):
        return db.get_java_database().getDatabasePath()

    return path_of


def test_a_load_in_several_chunks_of_one_table_keeps_every_row(temp_db):
    """Two tables concatenated (a ChunkedArray of two chunks) cut into batches of 4 (offsets that start inside a chunk)."""
    a, b = _matrix(3, 6, 0), _matrix(4, 6, 3)
    _make(temp_db, "Vec")
    table = pa.concat_tables([_table(a, "fixed_size", 0), _table(b, "fixed_size", 3)])
    assert temp_db.insert_arrow("Vec", table, batch_size=4) == 7
    assert _sql_digest(temp_db, "Vec") == (
        7,
        _expected_digest(a, 0) ^ _expected_digest(b, 3),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_special_float32_values_are_stored_bit_for_bit(temp_db, shape):
    special = np.array(
        [
            [
                -0.0,
                np.finfo(F32).max,
                np.finfo(F32).tiny,
                np.finfo(F32).smallest_subnormal,
                np.nan,
                np.inf,
                -np.inf,
                16777217.0,
            ]
        ],
        dtype=F32,
    )
    _make(temp_db, "Vec")
    assert temp_db.insert_arrow("Vec", _table(special, shape)) == 1
    stored = temp_db.query("sql", "SELECT embedding FROM Vec").first().get("embedding")
    assert len(stored) == 8
    got = np.asarray(stored, dtype=np.float64).astype(F32)
    assert np.isnan(got[4])
    keep = [0, 1, 2, 3, 5, 6, 7]
    assert (
        got.view(np.uint32)[keep].tolist() == special.view(np.uint32)[0, keep].tolist()
    )
    assert got[7] == F32(
        16777216.0
    )  # 16777217.0 is not a float32; the row holds the float32 that was written


def test_a_null_row_stays_null_and_its_neighbours_are_exact(temp_db):
    _make(temp_db, "Vec")
    table = pa.table(
        {
            "k": pa.array([0, 1, 2], pa.int32()),
            "embedding": pa.array(
                [[1.0, 2.0], None, [3.0, 4.0]], pa.list_(pa.float32(), 2)
            ),
        }
    )
    assert temp_db.insert_arrow("Vec", table) == 3
    rows = {
        r["k"]: r["embedding"]
        for r in temp_db.query("sql", "SELECT k, embedding FROM Vec").to_list()
    }
    assert rows[1] is None
    assert _bits(rows[0]) == _bits([1.0, 2.0]) and _bits(rows[2]) == _bits([3.0, 4.0])


def test_a_null_element_is_refused_and_the_batch_is_rolled_back(temp_db):
    _make(temp_db, "Vec")
    table = pa.table(
        {
            "k": pa.array([0, 1], pa.int32()),
            "embedding": pa.array(
                [[1.0, None, 3.0], [4.0, 5.0, 6.0]], pa.list_(pa.float32())
            ),
        }
    )
    with pytest.raises(ArcadeDBError, match="null element"):
        temp_db.insert_arrow("Vec", table)
    assert _count(temp_db, "Vec") == 0


def test_vectors_of_different_length_are_stored_as_given_without_an_index_and_refused_with_one(
    temp_db,
):
    _make(temp_db, "Free")
    ragged = pa.table(
        {
            "k": pa.array([0, 1, 2], pa.int32()),
            "embedding": pa.array(
                [[1.0, 2.0], [1.0, 2.0, 3.0], [4.0]], pa.list_(pa.float32())
            ),
        }
    )
    assert temp_db.insert_arrow("Free", ragged) == 3
    rows = {
        r["k"]: r["embedding"]
        for r in temp_db.query("sql", "SELECT k, embedding FROM Free").to_list()
    }
    assert {k: len(v) for k, v in rows.items()} == {0: 2, 1: 3, 2: 1}
    _make(temp_db, "Indexed", index=2)
    with pytest.raises(
        ArcadeDBError,
        match="Vector dimension does not match index dimension 2: got float.. of length 3",
    ):
        temp_db.insert_arrow("Indexed", ragged.slice(0, 2))
    assert _count(temp_db, "Indexed") == 0


def test_a_fixed_size_vector_of_the_wrong_dimension_is_refused_by_the_vector_index(
    temp_db,
):
    _make(temp_db, "Indexed", index=2)
    wrong = _table(_matrix(2, 3, specials=False), "fixed_size")
    with pytest.raises(
        ArcadeDBError,
        match="Vector dimension does not match index dimension 2: got float.. of length 3",
    ):
        temp_db.insert_arrow("Indexed", wrong)
    assert _count(temp_db, "Indexed") == 0
    right = _matrix(2, 2, specials=False)
    assert temp_db.insert_arrow("Indexed", _table(right, "fixed_size")) == 2
    assert _sql_digest(temp_db, "Indexed") == (2, _expected_digest(right))


def test_float64_and_integer_lists_keep_what_the_old_route_stored(temp_db):
    """Not taken by the matrix route: a float64 value past the float range is refused (a cast would store an infinity), the rest rounds."""
    _make(temp_db, "Dbl")
    ok = pa.table(
        {
            "k": pa.array([0], pa.int32()),
            "embedding": pa.array([[0.1, 16777217.0, -2.5]], pa.list_(pa.float64())),
        }
    )
    assert temp_db.insert_arrow("Dbl", ok) == 1
    got = temp_db.query("sql", "SELECT embedding FROM Dbl").first().get("embedding")
    assert _bits(got) == _bits([0.1, 16777217.0, -2.5])
    assert got[1] == 16777216.0
    _make(temp_db, "Over")
    over = pa.table(
        {
            "k": pa.array([0], pa.int32()),
            "embedding": pa.array([[1e39, 1.0]], pa.list_(pa.float64())),
        }
    )
    with pytest.raises(ArcadeDBError, match="1.0E39"):
        temp_db.insert_arrow("Over", over)
    assert _count(temp_db, "Over") == 0
    _make(temp_db, "Ints")
    ints = pa.table(
        {
            "k": pa.array([0], pa.int32()),
            "embedding": pa.array([[16777217, 2**40, -3]], pa.list_(pa.int64())),
        }
    )
    assert temp_db.insert_arrow("Ints", ints) == 1
    assert temp_db.query("sql", "SELECT embedding FROM Ints").first().get(
        "embedding"
    ) == [16777216.0, 1099511627776.0, -3.0]


@pytest.mark.parametrize(
    "embedding", [None, "LIST"], ids=["undeclared", "declared_list"]
)
def test_a_property_that_is_not_array_of_floats_keeps_a_list_and_takes_no_matrix(
    temp_db, crossings, embedding
):
    _make(temp_db, "Free", embedding=embedding)
    matrix = _matrix(5, 4, specials=False)
    assert temp_db.insert_arrow("Free", _table(matrix, "fixed_size")) == 5
    assert crossings["matrix"] == []
    assert _sql_digest(temp_db, "Free") == (5, _expected_digest(matrix))


def test_empty_vectors_are_stored_as_empty_lists(temp_db):
    _make(temp_db, "Vec")
    table = pa.table(
        {
            "k": pa.array([0, 1], pa.int32()),
            "embedding": pa.array([[], []], pa.list_(pa.float32())),
        }
    )
    assert temp_db.insert_arrow("Vec", table) == 2
    assert [
        r["embedding"]
        for r in temp_db.query("sql", "SELECT embedding FROM Vec ORDER BY k").to_list()
    ] == [[], []]


# ---- the three tests that fail on the code before the fix: they count what the load does, never how long it takes


@pytest.mark.parametrize("shape", SHAPES)
def test_a_vector_column_converts_no_python_value_per_row(temp_db, crossings, shape):
    """Before: one ``convert_python_to_java`` per row (an ArrayList of boxed doubles). After: the vector column makes none."""
    n = 600
    _make(temp_db, "Vec")
    assert temp_db.insert_arrow("Vec", _table(_matrix(n, 8), shape), batch_size=n) == n
    assert crossings["convert"] == []


def test_a_vector_column_crosses_as_one_matrix_per_batch(temp_db, crossings):
    """1,300 rows in batches of 500: three batches, so three ``FloatVectors.Matrix`` of one flat float[] each (500 x 24, 500 x 24, 300 x 24)."""
    _make(temp_db, "Vec")
    assert (
        temp_db.insert_arrow(
            "Vec", _table(_matrix(1_300, 24), "fixed_size"), batch_size=500
        )
        == 1_300
    )
    assert crossings["matrix"] == [500 * 24, 500 * 24, 300 * 24]


def test_the_python_heap_of_one_batch_stays_below_twice_the_flat_buffer(temp_db):
    """10,000 x 96 float32 is a 3.66 MiB buffer; the per-row Python lists of floats were 31.5 MiB at their peak (tracemalloc sees Python and
    numpy allocations, not the Java heap and not Arrow's pool). The bound is the buffer twice, nothing about time.
    """
    n, dim = 10_000, 96
    table = _table(_matrix(n, dim, specials=False), "fixed_size")
    _make(temp_db, "Vec")
    _make(temp_db, "Twin")
    assert (
        temp_db.insert_arrow("Twin", table.slice(0, 500)) == 500
    )  # first-call imports and caches stay out of the measurement
    flat = n * dim * 4
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        assert temp_db.insert_arrow("Vec", table, batch_size=n) == n
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert (
        peak < 2 * flat
    ), f"{peak} bytes of Python heap for a {flat} byte vector column"
