"""Column frames on the native build: the glue between ``arcadedb_n_next_col`` and the decoder both wheels share (``arcadedb._columnar``).

A result of the native build reaches Python in two forms. The rows after the first reply are column frames, which the library writes and
the shared decoder reads. The first reply of a query (up to ``_FIRST_BATCH`` rows) is JSON text that the row APIs read, and a columnar read
finds those rows (and any the caller read by row) as dicts in the result's buffer. They cannot be read again from the cursor, so they are
written here as one more frame, in the layout the library writes, by the rules the library's encoder follows, and read by the same
decoder: a column has one type however a result splits into the first reply and frames. The JSON reply names the properties that were
``float[]`` or ``double[]`` (``fv`` and ``dv``): a list of numbers in a row cannot say that by itself, and a vector column is a column of
declared arrays, not a column of lists that happen to fit float32. This frame goes away with the JSON reply.
"""

import json
import struct
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from .. import _columnar
from .codec import loads_tagged

_EPOCH = datetime(1970, 1, 1)
_MS = timedelta(milliseconds=1)
_US = timedelta(microseconds=1)
_INT64 = (-(1 << 63), (1 << 63) - 1)
_TWO53 = 1 << 53


def json_values(data, side, count):
    """The values of a ``json`` column of a frame from the library: its tagged JSON, restored as a row's values are."""
    return loads_tagged(bytes(data).decode("utf-8"))[0]


def _head_json_values(data, side, count):
    """The values of a ``json`` column of the head frame: the buffered values themselves, by the index the column carries."""
    return side[int.from_bytes(bytes(data), "little")]


def _naive_utc(v):
    if isinstance(v, datetime):
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return v
    return datetime(v.year, v.month, v.day)


def _kind(values, micros):
    """The column type of a frame that the values would get from the library's encoder: ``i8``, ``f8``, ``b1``, ``dt``, ``dtu``,
    ``str``, ``dec`` or ``json``; ``null`` for a column of nothing but nulls."""
    kinds = set()
    big_int = False
    for v in values:
        if v is None:
            continue
        t = type(v)
        if t is bool:
            k = "b1"
        elif t is int:
            if not _INT64[0] <= v <= _INT64[1]:
                k = "json"  # a BigInteger
            else:
                k = "i8"
                if not -_TWO53 <= v <= _TWO53:
                    big_int = True
        elif t is float:
            k = "f8"
        elif t is datetime or t is date:
            k = "dt"
        elif t is str:
            k = "str"
        elif t is Decimal:
            k = "dec"
        else:
            k = "json"
        kinds.add(k)
        if k == "json":
            break
    if not kinds:
        return "null"
    if kinds == {"i8", "f8"}:
        # an integer beyond 2**53 next to a floating value has no exact double: the values stay what they are
        return "json" if big_int else "f8"
    if len(kinds) > 1:
        return "json"
    kind = kinds.pop()
    if kind == "dt":
        if not micros:
            return "dt"
        if all(v is None or (type(v) is datetime and v.tzinfo is None) for v in values):
            return "dtu"
        return "json"
    return kind


def _vector_dim(values):
    """The one length of the lists of a column of float lists, or 0 when it is not one (a non-list, an empty or a ragged list)."""
    dim = 0
    for v in values:
        if v is None:
            continue
        if type(v) is not list or not v or any(type(x) is not float for x in v):
            return 0
        if dim == 0:
            dim = len(v)
        elif len(v) != dim:
            return 0
    return dim


def _bitmap(values):
    out = bytearray((len(values) + 7) // 8)
    for i, v in enumerate(values):
        if v is None:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def _column(values, vector, micros, side):
    """``(type, data, dim)`` of one column: the buffer the library's encoder writes for it. ``vector`` is ``"f"`` or ``"d"`` for a
    property that was a ``float[]`` or a ``double[]``, else None."""
    import numpy as np

    kind = _kind(values, micros)
    nan = float("nan")
    if kind == "i8":
        return (
            kind,
            np.array([0 if v is None else v for v in values], dtype="<i8").tobytes(),
            0,
        )
    if kind == "f8":
        return (
            kind,
            np.array([nan if v is None else v for v in values], dtype="<f8").tobytes(),
            0,
        )
    if kind == "b1":
        return kind, np.array([bool(v) for v in values], dtype=np.uint8).tobytes(), 0
    if kind in ("dt", "dtu"):
        unit = _US if kind == "dtu" else _MS
        ticks = [0 if v is None else (_naive_utc(v) - _EPOCH) // unit for v in values]
        return kind, np.array(ticks, dtype="<i8").tobytes(), 0
    if kind in ("str", "dec", "null"):
        offsets, blob = [0], bytearray()
        for v in values:
            if v is not None:
                text = format(v, "f") if kind == "dec" else v
                blob += text.encode("utf-8", "replace")
            offsets.append(len(blob))
        data = struct.pack(f"<{len(offsets)}i", *offsets) + bytes(blob)
        return ("str" if kind == "null" else kind), data, 0
    if vector:
        dim = _vector_dim(values)
        if dim:
            ctype, dt = ("f4v", "<f4") if vector == "f" else ("f8v", "<f8")
            rows = [[nan] * dim if v is None else v for v in values]
            return ctype, np.array(rows, dtype=dt).tobytes(), dim
    side.append(values)
    return "json", (len(side) - 1).to_bytes(4, "little"), 0


def _head_frame(rows, columns, hints, micros):
    """``(frame, side)`` for the dict rows: every column of ``columns``, or the union of the rows' keys in order of first appearance."""
    if columns is None:
        seen = {}
        known = seen.keys()
        for row in rows:
            if not row.keys() <= known:
                for k in row:
                    seen.setdefault(k, None)
        columns = list(seen)
    side, heads, body = [], [], bytearray()
    for name in columns:
        values = [row.get(name) for row in rows]
        ctype, data, dim = _column(values, hints.get(name), micros, side)
        nulls = _bitmap(values)
        head = {"name": name, "type": ctype, "nulls": len(nulls), "bytes": len(data)}
        if dim:
            head["dim"] = dim
        heads.append(head)
        body += nulls + data
    header = json.dumps({"count": len(rows), "cols": heads}).encode("utf-8")
    return memoryview(struct.pack("<i", len(header)) + header + bytes(body)), side


def decode_columns(raw, micros):
    return _columnar.decode_columns_frame(memoryview(raw), (), json_values, micros)


def decode_arrow(raw):
    return _columnar.decode_arrow_frame(memoryview(raw), (), json_values)


def columns_result(head_rows, columns, micros, batches, hints):
    """The ``to_columns`` dict of the buffered rows (dicts) followed by the decoded frames ``batches`` (a list of ``(count, parts)``)."""
    names = []
    parts_list = []
    if head_rows:
        frame, side = _head_frame(head_rows, columns, hints, micros)
        count, parts, _ = _columnar.decode_columns_frame(
            frame, side, _head_json_values, micros
        )
        names.extend(parts)
        parts_list.append((count, parts))
    for count, parts in batches:
        for name in parts:
            if name not in names:
                names.append(name)
        parts_list.append((count, parts))
    return _columnar.merge_columns(names, parts_list, columns)


def arrow_result(head_rows, columns, batches, hints):
    """The ``pyarrow.Table`` of the buffered rows (dicts) followed by the decoded frames ``batches`` (a list of ``(count, chunks)``)."""
    names = []
    chunks_list = []
    if head_rows:
        frame, side = _head_frame(head_rows, columns, hints, False)
        count, chunks, _ = _columnar.decode_arrow_frame(frame, side, _head_json_values)
        names.extend(chunks)
        chunks_list.append((count, chunks))
    for count, chunks in batches:
        for name in chunks:
            if name not in names:
                names.append(name)
        chunks_list.append((count, chunks))
    return _columnar.merge_arrow(names, chunks_list, columns)
