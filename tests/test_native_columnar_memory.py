"""Native: a document or vertex load by ``insert_arrow`` or ``insert_dataframe`` keeps its transient memory to one batch, and a vector column
is never turned into Python floats on its way to the columnar entry (ArcadeData/arcadedb-python #333 and #335, the extra anonymous memory of
the post1 candidate's native wheel).

Three properties are pinned, none of them with an absolute size:

* ``_column_values(..., arrow=True)`` returns an Arrow list of numbers (a vector column) as the Arrow array, which ``columnar.pack`` reads in
  place; the JVM build (``arrow=False``) still gets Python lists. One ``to_pylist()`` of a 64-dimensional vector column makes 64 Python
  floats per row (about 2 KB a row), which a 10,000-row batch held for the whole crossing.
* the Python-side peak of a whole load is a small fraction of what one batch of Python floats costs (``tracemalloc`` sees numpy and Python
  objects, not Arrow's memory or the image's heap, which is why the ratio is generous).
* ``columnar.pack`` is called once per batch and never with more than ``batch_size`` rows, and what it laid out is released when the batch
  is sent (the arrays of batch k are gone before batch k + 1 is packed).

Needs the native library, so conftest.py collects this file on the native backend only.
"""

import ctypes
import gc
import tracemalloc
import weakref

import numpy as np
import pyarrow as pa
import pytest

from arcadedb import dataframe
from arcadedb._native import columnar

DIM = 64


def _vectors(n, start=0, dim=DIM):
    r = np.arange(start, start + n, dtype=np.int64)
    j = np.arange(dim, dtype=np.int64)
    return (
        ((r[:, None] * 131 + j[None, :] * 17) % 1000).astype(np.float32)
    ) / np.float32(1000.0)


def _fixed(mat):
    return pa.FixedSizeListArray.from_arrays(
        pa.array(mat.reshape(-1), type=pa.float32()), mat.shape[1]
    )


def _bytes(addr_len):
    addr, length = addr_len
    return ctypes.string_at(addr, length) if length else b""


def _peak(fn):
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        fn()
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize(
    "make",
    [
        lambda m: _fixed(m),
        lambda m: pa.array(m.astype(np.float64).tolist(), type=pa.list_(pa.float64())),
        lambda m: pa.array(m.astype(np.int32).tolist(), type=pa.large_list(pa.int32())),
    ],
    ids=["fixed_size_list_f32", "list_f64", "large_list_i32"],
)
def test_a_vector_column_stays_an_arrow_array_on_the_native_load(make):
    chunk = make(_vectors(10))
    kept = dataframe._column_values("embedding", chunk, pa, native=True, arrow=True)
    assert isinstance(kept, pa.Array) and kept.type == chunk.type
    assert (
        dataframe._column_values("embedding", chunk, pa, native=True, arrow=False)
        == chunk.to_pylist()
    )
    assert isinstance(
        dataframe._column_values("embedding", chunk, pa, native=False, arrow=False),
        list,
    )


def test_a_list_that_is_not_a_list_of_numbers_still_becomes_python_values():
    for chunk in (
        pa.array([["a", "b"], ["c"]]),
        pa.array([[{"x": 1}], [{"x": 2}]]),
        pa.array([[[1.0], [2.0]], [[3.0]]]),
    ):
        got = dataframe._column_values("c", chunk, pa, native=True, arrow=True)
        assert isinstance(got, list) and got == chunk.to_pylist()


def test_the_vector_layout_from_arrow_is_the_layout_from_python_lists():
    mat = _vectors(300)
    table = pa.table({"embedding": _fixed(mat)}).slice(
        7, 211
    )  # a slice: the offset of the array is not zero
    declared = columnar.Declared(known={"embedding": "ARRAY_OF_FLOATS"})
    kept = dataframe._column_values(
        "embedding", table.column(0).combine_chunks(), pa, True, True
    )
    listed = dataframe._column_values(
        "embedding", table.column(0).combine_chunks(), pa, True, False
    )
    rows_a, (col_a,) = columnar.pack({"embedding": kept}, declared)
    rows_b, (col_b,) = columnar.pack({"embedding": listed}, declared)
    assert (
        rows_a == rows_b == 211
        and col_a.kind == col_b.kind == "f"
        and col_a.dim == col_b.dim == DIM
    )
    assert _bytes(col_a.data) == _bytes(col_b.data) == mat[7:218].tobytes()


def test_the_python_side_peak_of_a_vector_batch_is_a_fraction_of_the_python_float_copy():
    batch = _fixed(_vectors(10_000))
    declared = columnar.Declared(known={"embedding": "ARRAY_OF_FLOATS"})
    as_python = _peak(
        lambda: columnar.pack(
            {
                "embedding": dataframe._column_values(
                    "embedding", batch, pa, True, False
                )
            },
            declared,
        )
    )
    in_place = _peak(
        lambda: columnar.pack(
            {"embedding": dataframe._column_values("embedding", batch, pa, True, True)},
            declared,
        )
    )
    assert (
        as_python > 10 * 1024 * 1024
    ), as_python  # 640,000 Python floats: the copy this change removes (about 20 MiB)
    assert in_place * 8 < as_python, (
        in_place,
        as_python,
    )  # measured: about 100 times less


class _PackSpy:
    """Wraps ``columnar.pack``: the row count of every call and a weak reference to every array it laid out."""

    def __init__(self, monkeypatch):
        self.rows, self.arrays, self.alive_at_next_call = [], [], []
        real = columnar.pack

        def pack(columns, declared):
            gc.collect()
            self.alive_at_next_call.append(
                sum(ref() is not None for refs in self.arrays for ref in refs)
            )
            out = real(columns, declared)
            self.rows.append(None if out is None else out[0])
            self.arrays.append(
                [weakref.ref(c.keep[0]) for c in out[1]] if out is not None else []
            )
            return out

        monkeypatch.setattr(columnar, "pack", pack)


def _vector_type(db, kind="VERTEX"):
    db.command("sql", f"CREATE {kind} TYPE V")
    db.command("sql", "CREATE PROPERTY V.id LONG")
    db.command("sql", "CREATE PROPERTY V.embedding ARRAY_OF_FLOATS")


@pytest.mark.parametrize("kind", ["VERTEX", "DOCUMENT"])
def test_every_batch_is_packed_once_and_released_before_the_next(
    temp_db, monkeypatch, kind
):
    _vector_type(temp_db, kind)
    spy = _PackSpy(monkeypatch)
    n, batch = 2_505, 1_000
    table = pa.table(
        {"id": pa.array(np.arange(n, dtype=np.int64)), "embedding": _fixed(_vectors(n))}
    )
    assert temp_db.insert_arrow("V", table, batch_size=batch) == n
    assert spy.rows == [
        1_000,
        1_000,
        505,
    ]  # one pack per batch, never more than batch_size rows
    assert spy.alive_at_next_call == [
        0,
        0,
        0,
    ], spy.alive_at_next_call  # batch k is gone when batch k + 1 is packed
    gc.collect()
    assert not any(
        ref() is not None for refs in spy.arrays for ref in refs
    )  # and the last one when the call returns


def test_a_dataframe_load_is_packed_per_batch_too(temp_db, monkeypatch):
    import pandas as pd

    temp_db.command("sql", "CREATE VERTEX TYPE P")
    temp_db.command("sql", "CREATE PROPERTY P.id LONG")
    temp_db.command("sql", "CREATE PROPERTY P.name STRING")
    spy = _PackSpy(monkeypatch)
    frame = pd.DataFrame(
        {"id": np.arange(250, dtype=np.int64), "name": [f"p{i}" for i in range(250)]}
    )
    assert temp_db.insert_dataframe("P", frame, batch_size=100) == 250
    assert spy.rows == [100, 100, 50] and spy.alive_at_next_call == [0, 0, 0]


def test_the_python_side_peak_of_a_whole_vector_load_stays_below_one_batch_of_python_floats(
    temp_db,
):
    _vector_type(temp_db)
    n, batch = 40_000, 10_000
    table = pa.table(
        {"id": pa.array(np.arange(n, dtype=np.int64)), "embedding": _fixed(_vectors(n))}
    )
    one_batch_of_python_floats = _peak(
        lambda: table.slice(0, batch).column("embedding").to_pylist()
    )
    whole_load = _peak(lambda: temp_db.insert_arrow("V", table, batch_size=batch))
    assert one_batch_of_python_floats > 10 * 1024 * 1024
    assert whole_load * 2 < one_batch_of_python_floats, (
        whole_load,
        one_batch_of_python_floats,
    )  # measured: about 7 times less
    assert temp_db.query("sql", "SELECT count(*) AS n FROM V").first().get("n") == n


def _stored(db, key):
    return [
        float(x)
        for x in db.query("sql", "SELECT embedding FROM V WHERE id = ?", key)
        .first()
        .get("embedding")
    ]


@pytest.mark.parametrize("kind", ["VERTEX", "DOCUMENT"])
def test_the_stored_vectors_are_the_ones_the_python_lists_stored(temp_db, kind):
    _vector_type(temp_db, kind)
    mat = _vectors(120)
    table = pa.table(
        {"id": pa.array(np.arange(120, dtype=np.int64)), "embedding": _fixed(mat)}
    )
    assert (
        temp_db.insert_arrow("V", table.slice(5, 100), batch_size=30) == 100
    )  # a sliced table, four batches
    for key in (5, 6, 34, 35, 64, 104):
        assert (
            max(abs(a - b) for a, b in zip(_stored(temp_db, key), mat[key].tolist()))
            < 1e-6
        )
    assert temp_db.query("sql", "SELECT count(*) AS n FROM V").first().get("n") == 100


def test_a_vector_column_with_a_null_or_a_list_property_takes_the_row_path_with_the_same_values(
    temp_db,
):
    _vector_type(temp_db)
    temp_db.command("sql", "CREATE VERTEX TYPE L")
    temp_db.command("sql", "CREATE PROPERTY L.id LONG")
    temp_db.command("sql", "CREATE PROPERTY L.tags LIST")
    with_null = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "embedding": pa.array(
                [[0.5, 0.25], None, [1.0, 2.0]], type=pa.list_(pa.float32())
            ),
        }
    )
    assert temp_db.insert_arrow("V", with_null) == 3
    assert [
        float(x)
        for x in temp_db.query("sql", "SELECT embedding FROM V WHERE id = 1")
        .first()
        .get("embedding")
    ] == [0.5, 0.25]
    assert (
        temp_db.query("sql", "SELECT embedding FROM V WHERE id = 2")
        .first()
        .get("embedding")
        is None
    )
    tags = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "tags": pa.array([[1, 2, 3], [4]], type=pa.list_(pa.int64())),
        }
    )
    assert temp_db.insert_arrow("L", tags) == 2
    assert list(
        temp_db.query("sql", "SELECT tags FROM L WHERE id = 1").first().get("tags")
    ) == [1, 2, 3]
