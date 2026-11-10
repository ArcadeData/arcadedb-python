"""The column frame of both wheels, and the one decoder that reads it.

A frame is what ``ColumnBatcher`` (the JVM wheel's bridge jar) writes and what the native library's ``arcadedb_n_next_col`` writes: a
little-endian ``int32`` header length, a JSON header (``count``, ``cols``: name, type, null-bitmap bytes, data bytes and, for a
vector, ``dim``), then for each column its null bitmap (bit i of byte i/8 set means row i is null) and its buffer. A column type is
``i8`` (int64), ``f8`` (float64), ``b1`` (one byte per bool), ``dt`` and ``dtu`` (int64 epoch milliseconds and microseconds),
``f4v`` and ``f8v`` (``dim`` floats per row), ``str`` (int32 offsets, count + 1 of them, then the UTF-8 bytes: Arrow's own string
layout), ``dec`` (a BigDecimal as text in the string layout) and ``json`` (a JSON array of the column, for values no fixed layout
carries: nested lists and maps, RIDs, a DATE next to datetimes, a mixed column).

The interface and the format are shared, the plumbing is per wheel: what differs is how the bytes get here (JPype or ctypes) and the
text of a ``json`` column (the JVM wheel's tagged JSON with a side array of Java objects, the native library's tagged JSON), so the
decoders take ``json_values(data, side, count)`` from the wheel. Nothing here imports JPype.
"""

import json
from typing import Any, Callable, Dict, List, Tuple


def _cast_all_to_string(arrs: list, pa) -> list:
    """
    Cast every array to a string type.

    Tolerates a type pyarrow cannot cast directly (e.g. a vector/list column)
    by going through Python objects instead.
    """
    unified = []
    for a in arrs:
        try:
            unified.append(a.cast(pa.string()))
        except pa.ArrowException:
            unified.append(
                pa.array(
                    [None if v is None else str(v) for v in a.to_pylist()],
                    type=pa.string(),
                )
            )
    return unified


def _decode_strings(data, count: int, mask) -> list:
    """The strings of one column buffer: int32 offsets, then the UTF-8 blob.

    ``mask`` is the null mask (True = null) or None when the column has no nulls.
    """
    import numpy as np

    offs = np.frombuffer(data[: (count + 1) * 4], dtype="<i4")
    chars = bytes(data[(count + 1) * 4 :])
    if mask is None:
        return [chars[offs[i] : offs[i + 1]].decode("utf-8") for i in range(count)]
    return [
        None if mask[i] else chars[offs[i] : offs[i + 1]].decode("utf-8")
        for i in range(count)
    ]


def _decode_decimals(data, count: int, mask) -> list:
    """The DECIMAL values of one column buffer, as ``Decimal`` objects (None for
    null). The bridge writes each ``BigDecimal`` in full as text, so no digit is
    lost to a double."""
    from decimal import Decimal

    return [
        None if s is None else Decimal(s) for s in _decode_strings(data, count, mask)
    ]


def _missing_column_part(count: int, template):
    """A batch of ``count`` rows that lack a column some other batch has: nulls in
    the shape ``to_columns`` uses for that column (NaN for numbers, NaT for
    datetimes, None for everything else)."""
    import numpy as np

    if isinstance(template, np.ndarray):
        if template.ndim == 2:
            return np.full((count, template.shape[1]), np.nan, dtype=template.dtype)
        if template.dtype.kind in "iuf":
            return np.full(count, np.nan)
        if template.dtype.kind == "M":
            return np.full(count, np.datetime64("NaT"), dtype=template.dtype)
        if template.dtype.kind == "O":
            return np.full(count, None, dtype=object)
    return [None] * count


def _is_untyped_arrow_chunk(a, pa) -> bool:
    """True for a chunk that says nothing about its column's type: all values
    null (a batch whose rows lack the property, or carry it as null), or a list
    column holding only empty lists, which Arrow types ``list<null>``."""
    if len(a) == 0 or a.null_count == len(a) or pa.types.is_null(a.type):
        return True
    return pa.types.is_list(a.type) and pa.types.is_null(a.type.value_type)


def _common_decimal_type(types: list, pa):
    """One decimal type every given decimal type fits in (decimal128 up to 38
    digits, decimal256 up to 76), or None when it would need more."""
    scale = max(t.scale for t in types)
    precision = max(t.precision - t.scale for t in types) + scale
    if precision <= 38:
        return pa.decimal128(precision, scale)
    if precision <= 76:
        return pa.decimal256(precision, scale)
    return None


def _unify_arrow_chunk_types(arrs: list, pa) -> list:
    """
    Unify a column's per-batch Arrow arrays onto one Arrow type.

    ArcadeDB is schemaless per document, so a property's Java type can
    legally vary row to row; ColumnBatcher infers each batch's column type
    independently, so two batches of the same result set can produce
    incompatible Arrow types for the same column (#7108) - e.g. int64 in one
    batch, float64 in the next. ``pa.chunked_array`` requires every chunk to
    share one type, so leaving them as-is raises at concatenation time on a
    result set that is otherwise entirely legal.

    Numeric types widen (int64 -> float64, matching to_columns' pandas-style
    promotion) when every value survives the cast exactly; pyarrow's cast is
    safe by default and raises rather than silently lose precision, which an
    int64 outside float64's +-2**53 exact range would (code review on #7108),
    so that case - and anything else, including a json or vector column that
    varied - degrades to string instead, so the column always ends up with
    exactly one type rather than an exception.

    int64/float64 is the only numeric pair worth special-casing: ColumnBatcher
    (src/java/.../ColumnBatcher.java) already widens every
    Java integer type (Long/Integer/Short/Byte) to the same "i8" wire encoding
    and every Java floating type (Double/Float) to "f8" *within* one batch, so
    across batches the only numeric type mismatch this ever has to unify is
    exactly this pair - there is no int32/int64 or bool/int64 case to widen.
    """
    types = {a.type for a in arrs}
    if len(types) <= 1:
        return arrs

    # A chunk that carries no type information (every value null, or only empty
    # lists) takes the type the other chunks agree on, so the column's type does
    # not depend on where the batch boundaries fall (#114).
    untyped = [_is_untyped_arrow_chunk(a, pa) for a in arrs]
    if any(untyped) and not all(untyped):
        typed = _unify_arrow_chunk_types(
            [a for a, u in zip(arrs, untyped) if not u], pa
        )
        target = typed[0].type
        filled = iter(typed)
        out = []
        for a, u in zip(arrs, untyped):
            if not u:
                out.append(next(filled))
            elif a.null_count == len(a):
                out.append(pa.nulls(len(a), type=target))
            else:
                try:
                    out.append(a.cast(target))
                except pa.ArrowException:
                    return _cast_all_to_string(arrs, pa)
        return out

    if types <= {pa.int64(), pa.float64()}:
        try:
            return [a.cast(pa.float64()) for a in arrs]
        except pa.ArrowException:
            pass

    if all(pa.types.is_list(t) or pa.types.is_fixed_size_list(t) for t in types):
        # vectors and lists of numbers: a dimension that changes between frames makes a list, float32 next to float64 is float64
        # (the widening is exact), never text
        values = {t.value_type for t in types}
        if values <= {pa.float32(), pa.float64()}:
            # always float64: the widening is exact, and a frame whose rows differ in length is a json column (list<double>) on its
            # own, so float32 here would make the type depend on whether the change falls inside a frame or between two
            sizes = {
                t.list_size if pa.types.is_fixed_size_list(t) else None for t in types
            }
            target = (
                pa.list_(pa.float64(), next(iter(sizes)))
                if len(sizes) == 1 and None not in sizes
                else pa.list_(pa.float64())
            )
            try:
                return [pa.array(a.to_pylist(), type=target) for a in arrs]
            except pa.ArrowException:
                pass

    if all(pa.types.is_decimal(t) for t in types):
        common = _common_decimal_type(list(types), pa)
        if common is not None:
            return [a.cast(common) for a in arrs]

    return _cast_all_to_string(arrs, pa)


MAX_FRAME_ROWS = 2**31 - 1  # a frame's row limit crosses to the library as a C int


def frame_rows(batch_size) -> int:
    """The row limit of a frame for ``batch_size``: at least 1 (a ValueError otherwise), at most what a C int holds, so a huge
    ``batch_size`` means "as many as one frame takes" instead of wrapping around to zero or a negative number.
    """
    rows = int(batch_size)
    if rows < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    return min(rows, MAX_FRAME_ROWS)


def empty_columns(columns) -> Dict[str, Any]:
    """The ``to_columns`` dict of a result with no rows: no column, or the ones the caller asked for, empty."""
    return {} if columns is None else {name: [] for name in columns}


def empty_arrow(columns):
    """The ``to_arrow`` table of a result with no rows: no column, or the ones the caller asked for, empty and of the null type."""
    import pyarrow as pa

    return pa.table({} if columns is None else {name: pa.nulls(0) for name in columns})


def _null_mask(buf, pos: int, nulls_len: int, count: int):
    """``(null_bits, mask)`` of a column: ``mask`` is a bool array (True = null) or None when no row is null."""
    import numpy as np

    null_bits = np.frombuffer(buf[pos : pos + nulls_len], dtype=np.uint8)
    if not null_bits.any():
        return null_bits, None
    return null_bits, np.unpackbits(null_bits, bitorder="little")[:count].astype(bool)


def read_header(buf) -> Tuple[Dict[str, Any], int]:
    """``(header, position of the first column)`` of a frame."""
    hlen = int.from_bytes(buf[:4], "little")
    return json.loads(bytes(buf[4 : 4 + hlen])), 4 + hlen


def decode_columns_frame(
    buf, side, json_values: Callable, micros: bool = False
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """``(count, parts, header)`` of one frame as ``to_columns`` columns: ``parts`` maps a column name to a numpy array
    (int64, float64, bool, datetime64) or a Python list (text, JSON-typed values, a bool column with nulls); a count of 0 has
    no parts. A vector column is a (count, dim) matrix, a null row of it NaN.

    ``micros`` is the ``to_dataframe`` form: a naive datetime column is ``datetime64[us]`` (``dtu``), and a vector column is a
    list with one numpy array per cell and None for a null (a NaN row would read as a vector of NaN, not as a missing value).
    """
    import numpy as np

    header, pos = read_header(buf)
    count = header["count"]
    if count == 0:
        return 0, {}, header
    parts: Dict[str, Any] = {}
    for col in header["cols"]:
        name, ctype = col["name"], col["type"]
        nulls_len = col["nulls"]
        _, mask = _null_mask(buf, pos, nulls_len, count)
        has_nulls = mask is not None
        pos += nulls_len
        data = buf[pos : pos + col["bytes"]]
        pos += col["bytes"]

        if ctype == "i8":
            # a copy: np.frombuffer over bytes is read-only, and every array of to_columns is writable
            arr = np.frombuffer(data, dtype="<i8").copy()
            if has_nulls:
                arr = arr.astype(np.float64)
                arr[mask] = np.nan
            values = arr
        elif ctype == "f8":
            # the writer already puts NaN into null slots
            values = np.frombuffer(data, dtype="<f8").copy()
        elif ctype == "dt":
            arr = np.frombuffer(data, dtype="<i8").astype("datetime64[ms]")
            if has_nulls:
                arr[mask] = np.datetime64("NaT", "ms")
            values = arr
        elif ctype == "dtu":
            arr = np.frombuffer(data, dtype="<i8").astype("datetime64[us]")
            if has_nulls:
                arr[mask] = np.datetime64("NaT", "us")
            values = arr
        elif ctype == "b1":
            arr = np.frombuffer(data, dtype=np.uint8).astype(bool)
            if has_nulls:
                values = [None if mask[i] else bool(arr[i]) for i in range(count)]
            else:
                values = arr
        elif ctype in ("f4v", "f8v"):
            # fixed-dimension vector column -> 2-D array (count, dim); an empty vector has no "dim" in the header. The null rows are
            # made NaN here, whatever the writer put in the slot: a row of zeros would read as a real zero vector
            dim = col.get("dim", 0)
            dt = "<f4" if ctype == "f4v" else "<f8"
            if dim == 0:
                values = np.empty((count, 0), dtype=dt)
            else:
                values = np.frombuffer(data, dtype=dt).reshape(count, dim).copy()
            if has_nulls:
                values[mask] = np.nan
            if micros:
                values = [
                    None if has_nulls and mask[i] else values[i] for i in range(count)
                ]
        elif ctype == "json":
            values = json_values(data, side, count)
        elif ctype == "dec":
            # an object array of Decimal (None for null): a double would lose digits, and the dtype would follow the data
            values = np.empty(count, dtype=object)
            values[:] = _decode_decimals(data, count, mask)
        else:  # str
            values = _decode_strings(data, count, mask)
        parts[name] = values
    return count, parts, header


def _is_all_none(part, count: int) -> bool:
    return (
        isinstance(part, list) and len(part) == count and all(v is None for v in part)
    )


# float64 holds every integer in +-2**53 exactly, and not every one beyond it
_TWO53 = 1 << 53


def _int64_beyond_2_53_next_to_float64(parts) -> bool:
    """True when the parts of one column hold an int64 array with a value beyond +-2**53 and a float64 array: ``np.concatenate``
    would make the integers doubles and round them. One frame of such a column is a list of exact Python values (the writer's
    ``json`` column); the merge of frames does the same, so the column does not depend on where the frames end.
    """
    import numpy as np

    arrays = [p for p in parts if isinstance(p, np.ndarray) and p.ndim == 1]
    if not any(a.dtype.kind == "f" for a in arrays):
        return False
    return any(
        a.dtype.kind == "i" and a.size and (a.max() > _TWO53 or a.min() < -_TWO53)
        for a in arrays
    )


def merge_columns(
    names: List[str], batches: List[Tuple[int, Dict[str, Any]]], columns=None
) -> Dict[str, Any]:
    """The ``to_columns`` dict of the parts of every frame. The type of a column belongs to the whole result, not to where the frames
    end: a frame that lacks the column, or holds only nulls for it, takes the type of the frames that have values (NaN, NaT or None
    in its rows). An int64 frame with a value beyond +-2**53 next to a float64 frame makes a list of exact Python values, not a
    float64 array. With no rows at all the result is :func:`empty_columns`."""
    import numpy as np

    if not batches or not names:
        return empty_columns(columns)
    out: Dict[str, Any] = {}
    for name in names:
        typed = [
            p[name] for c, p in batches if name in p and not _is_all_none(p[name], c)
        ]
        template = typed[0] if typed else next(p[name] for _, p in batches if name in p)
        parts = [
            (
                p[name]
                if name in p and not (typed and _is_all_none(p[name], c))
                else _missing_column_part(c, template)
            )
            for c, p in batches
        ]
        np_parts = [p for p in parts if not isinstance(p, list)]
        if (
            parts
            and len(np_parts) == len(parts)
            and not _int64_beyond_2_53_next_to_float64(typed)
        ):
            try:
                out[name] = np.concatenate(parts) if len(parts) > 1 else parts[0]
                continue
            except Exception:  # nosec B110 - fall through to generic merge
                pass
        column: list = []
        for p in parts:
            column.extend(p if isinstance(p, list) else p.tolist())
        out[name] = column
    return out


def _pandas_would_round(values) -> bool:
    """True for a list that holds an integer beyond +-2**53 next to a float: pandas infers float64 for a list of ints and floats,
    and an integer that is not an exact double comes out rounded."""
    kinds = set(map(type, values))  # one C-speed pass: a text column ends here
    if int not in kinds or float not in kinds:
        return False
    return any(type(v) is int and not -_TWO53 <= v <= _TWO53 for v in values)


def dataframe_from_columns(columns):
    """The ``pandas.DataFrame`` of the ``to_columns`` dict ``columns`` (what both wheels' ``to_dataframe`` builds). A column that is a
    list holding an integer beyond +-2**53 next to a float is an ``object`` column of the exact Python values, as ``to_columns()``
    returns them: pandas would make it float64 and round the integer (#340). Every other column is handed to pandas as it was.
    """
    import pandas as pd

    exact = {
        name: pd.Series(values, dtype=object)
        for name, values in columns.items()
        if isinstance(values, list) and _pandas_would_round(values)
    }
    return pd.DataFrame({**columns, **exact} if exact else columns)


def decode_arrow_frame(
    buf, side, json_values: Callable
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    """``(count, chunks, header)`` of one frame as ``to_arrow`` columns: ``chunks`` maps a column name to a ``pyarrow.Array``.
    Integers keep their type and their nulls (``to_columns`` would make them float64 with NaN), strings wrap the frame's own
    offsets and bytes, vectors are fixed-size lists of float32 or float64."""
    import numpy as np
    import pyarrow as pa

    header, pos = read_header(buf)
    count = header["count"]
    if count == 0:
        return 0, {}, header
    chunks: Dict[str, Any] = {}
    for col in header["cols"]:
        name, ctype = col["name"], col["type"]
        nulls_len = col["nulls"]
        null_bits, mask = _null_mask(buf, pos, nulls_len, count)
        has_nulls = mask is not None
        pos += nulls_len
        data = buf[pos : pos + col["bytes"]]
        pos += col["bytes"]

        if ctype == "i8":
            # kept as int64 WITH validity, where to_columns promotes the whole column to float64/NaN
            arr = pa.array(np.frombuffer(data, dtype="<i8"), mask=mask)
        elif ctype == "f8":
            arr = pa.array(np.frombuffer(data, dtype="<f8"), mask=mask)
        elif ctype == "dt":
            ts = np.frombuffer(data, dtype="<i8").astype("datetime64[ms]")
            arr = pa.array(ts, type=pa.timestamp("ms"), mask=mask)
        elif ctype == "dtu":
            ts = np.frombuffer(data, dtype="<i8").astype("datetime64[us]")
            arr = pa.array(ts, type=pa.timestamp("us"), mask=mask)
        elif ctype == "b1":
            # stays a bool column; to_columns degrades this to a list
            vals = np.frombuffer(data, dtype=np.uint8).astype(bool)
            arr = pa.array(vals, mask=mask)
        elif ctype in ("f4v", "f8v"):
            # an empty vector has no "dim" in the header
            dim = col.get("dim", 0)
            dt = "<f4" if ctype == "f4v" else "<f8"
            if dim == 0:
                # no fixed-size list of size 0: empty lists, as the row readers give
                arr = pa.array(
                    [
                        None if mask is not None and mask[i] else []
                        for i in range(count)
                    ],
                    type=pa.list_(pa.float32() if ctype == "f4v" else pa.float64()),
                )
            else:
                flat = np.frombuffer(data, dtype=dt)
                child = pa.array(flat)
                arr = pa.FixedSizeListArray.from_arrays(child, dim)
                if mask is not None:
                    arr = pa.array(arr.to_pylist(), type=arr.type, mask=mask)
        elif ctype == "json":
            values = json_values(data, side, count)
            try:
                arr = pa.array(values)
            except pa.ArrowException:
                # mixed types in one column (an int in one row, a string in the next): the same string column a mixed
                # column across batches ends up as, not a raw ArrowInvalid
                arr = pa.array(
                    [None if v is None else str(v) for v in values], type=pa.string()
                )
        elif ctype == "dec":
            decimals = _decode_decimals(data, count, mask)
            try:
                arr = pa.array(decimals)
            except pa.ArrowException:
                # more than the 76 digits decimal256 holds
                arr = pa.array(
                    [None if d is None else str(d) for d in decimals], type=pa.string()
                )
        else:  # str: int32 offsets + utf8 blob IS Arrow's layout
            off_bytes = bytes(data[: (count + 1) * 4])
            chars = bytes(data[(count + 1) * 4 :])
            validity = (
                None if not has_nulls else pa.py_buffer(np.invert(null_bits).tobytes())
            )
            arr = pa.StringArray.from_buffers(
                count, pa.py_buffer(off_bytes), pa.py_buffer(chars), validity
            )
        chunks[name] = arr
    return count, chunks, header


def merge_arrow(
    names: List[str], batches: List[Tuple[int, Dict[str, Any]]], columns=None
):
    """The ``pyarrow.Table`` of the chunks of every frame: one chunked column per name, one type per column (see
    :func:`_unify_arrow_chunk_types`), a frame that lacks a column adds nulls."""
    import pyarrow as pa

    if not batches or not names:
        return empty_arrow(columns)
    return pa.table(
        {
            n: pa.chunked_array(
                _unify_arrow_chunk_types(
                    [
                        chunks[n] if n in chunks else pa.nulls(c)
                        for c, chunks in batches
                    ],
                    pa,
                )
            )
            for n in names
        }
    )
