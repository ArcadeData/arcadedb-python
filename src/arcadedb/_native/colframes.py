"""Column frames on the native build: the glue between ``arcadedb_n_next_col`` and the decoder both wheels share (``arcadedb._columnar``).

A result of the native build reaches Python in two forms. The rows after the first reply are column frames, read by the shared decoder.
The first reply of a query (up to ``_FIRST_BATCH`` rows) is JSON text that the row APIs read, and it is the only part of a result that a
columnar read holds as dicts: the rows that are still buffered when the columnar read starts go through the rules of :mod:`.columns`
and are merged with the frames as one more chunk. When the cursor still holds those rows (``arcadedb_n_next_col`` replays them), they
are not taken from the dicts at all; a result that the first reply held whole has no cursor, so its rows are the only form it has.
That chunk goes away with the JSON reply.

Two things in a chunk of dicts cannot be known from the values: whether a list of floats is a ``float[]`` (a frame says so with its
column type) and nothing else. A column of lists of floats that float32 holds exactly is a vector column in the chunk when the frames
say so, or when there are no frames; when the frames carry it as a list column, the chunk keeps the list.
"""

from datetime import datetime

from .. import _columnar
from . import columns as _rows
from .codec import loads_tagged


def json_values(data, side, count):
    """The values of a ``json`` column of a frame: the native library's tagged JSON, restored as a row's values are."""
    return loads_tagged(bytes(data).decode("utf-8"))[0]


def decode_columns(raw, micros):
    return _columnar.decode_columns_frame(memoryview(raw), (), json_values, micros)


def decode_arrow(raw):
    return _columnar.decode_arrow_frame(memoryview(raw), (), json_values)


def _micros_part(values):
    """A column of datetimes for ``to_dataframe``: ``datetime64[us]`` when every value is a naive datetime, else the objects
    themselves (a DATE, a time-zone-aware datetime), as a frame's ``json`` column gives them.
    """
    import numpy as np

    if all(v is None or (type(v) is datetime and v.tzinfo is None) for v in values):
        return np.array(
            [
                np.datetime64("NaT", "us") if v is None else np.datetime64(v, "us")
                for v in values
            ],
            dtype="datetime64[us]",
        )
    return list(values)


def _first_part(batches, name):
    return next((p[name] for _, p in batches if name in p), None)


def columns_result(head_rows, columns, micros, batches):
    """The ``to_columns`` dict of the buffered rows (dicts) followed by the decoded frames ``batches`` (a list of ``(count, parts)``)."""
    names = []
    parts_list = []
    if head_rows:
        parts = _rows.to_columns(head_rows, columns)
        for name in list(parts):
            part = parts[name]
            values = [row.get(name) for row in head_rows]
            if getattr(part, "ndim", 1) == 2:
                # a column of float lists that float32 holds: a vector column unless the frames carry it as lists
                other = _first_part(batches, name)
                if other is not None and getattr(other, "ndim", 1) != 2:
                    parts[name] = values
            elif micros and getattr(getattr(part, "dtype", None), "kind", "") == "M":
                parts[name] = _micros_part(values)
        names.extend(parts)
        parts_list.append((len(head_rows), parts))
    for count, parts in batches:
        for name in parts:
            if name not in names:
                names.append(name)
        parts_list.append((count, parts))
    if not names:
        return {}
    return _columnar.merge_columns(names, parts_list)


def _vector_arrow(values):
    """The ``fixed_size_list<float32>`` array of a column of float lists that float32 holds, or None."""
    import pyarrow as pa

    vec = _rows._float_vectors(values)
    if vec is None:
        return None
    arr = pa.FixedSizeListArray.from_arrays(pa.array(vec.reshape(-1)), vec.shape[1])
    if any(v is None for v in values):
        arr = pa.array(arr.to_pylist(), type=arr.type, mask=[v is None for v in values])
    return arr


def arrow_result(head_rows, columns, batches):
    """The ``pyarrow.Table`` of the buffered rows (dicts) followed by the decoded frames ``batches`` (a list of ``(count, chunks)``)."""
    import pyarrow as pa

    names = []
    chunks_list = []
    if head_rows:
        table = _rows.to_arrow(head_rows, columns)
        chunks = {}
        for name in table.column_names:
            column = table.column(name)
            arr = (
                column.chunk(0)
                if column.num_chunks == 1
                else pa.concat_arrays(column.chunks)
            )
            if pa.types.is_null(arr.type):
                # a frame types an all-null column as text
                arr = pa.nulls(len(arr), pa.string())
            elif pa.types.is_list(arr.type) and arr.type.value_type == pa.float64():
                other = _first_part(batches, name)
                if other is None or pa.types.is_fixed_size_list(other.type):
                    vector = _vector_arrow([row.get(name) for row in head_rows])
                    if vector is not None:
                        arr = vector
            chunks[name] = arr
        names.extend(chunks)
        chunks_list.append((len(head_rows), chunks))
    for count, chunks in batches:
        for name in chunks:
            if name not in names:
                names.append(name)
        chunks_list.append((count, chunks))
    return _columnar.merge_arrow(names, chunks_list)
