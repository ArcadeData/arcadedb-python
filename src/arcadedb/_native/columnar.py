"""Packing of whole columns for ``arcadedb_n_insert_colp``: the columnar bulk insert of the native backend.

A column crosses as raw memory, not as JSON text: the image reads each buffer in place and builds the documents without a parser. This
module only decides how a column is laid out and hands the addresses over; it needs no native library, so the layouts are unit-tested
without one. A column that cannot be laid out (a nested value, a mixed-type column, a numpy datetime64, an integer past 64 bits, a string
with a lone surrogate) makes :func:`pack` return ``None`` and the caller uses the JSON path for that call; nothing is half sent.

Kinds (the image's, ``BulkOps.insertColumns``): ``q`` int64, ``d`` float64, ``b`` bool byte, ``i`` int32, ``e`` float32, ``f`` float32
matrix (``dim`` per row), ``s`` UTF-8 string (``int32 offsets[n+1]`` and the bytes they index: Arrow's string layout). A column may
carry a null bitmap (Arrow's: bit ``i`` of byte ``i // 8``, least significant first, 1 = present).

Temporal, decimal and binary kinds (the values the JSON path sends as tagged values, so the stored value is the same object): ``T`` a
date-time, int64 ticks since 1970-01-01T00:00:00 of the UTC wall clock, ``dim`` ticks per second (1, 1000, 10**6 or 10**9: an Arrow
timestamp keeps its unit, a Python ``datetime`` is microseconds), read as ``LocalDateTime``; ``D`` a date, int32 days since the epoch,
read as ``LocalDate``; ``n`` a decimal, the string layout holding ``str(Decimal)``, read as ``BigDecimal``; ``x`` bytes, the string
layout, read as ``byte[]``. A time-zone-aware ``datetime`` or Arrow timestamp is its UTC wall clock and a naive one is taken as it is,
the rule of ``_tag_datetime``; nothing here decides a time zone.

``i`` and ``e`` and the float matrix from a list of lists are chosen only for a property declared ``INTEGER``, ``FLOAT`` and
``ARRAY_OF_FLOATS``: the engine would convert to those types anyway, so the stored value is the one the JSON path stores. For an
undeclared property an ``int32`` column crosses as ``q`` (a Long) and a ``float32`` column as ``d`` (a Double), as the JSON path stores
them.
"""

from datetime import date, datetime, timezone
from decimal import Decimal

from .schema_objects import (  # the schema-entry reader shared with NativeType (#195)
    _name_type,
)

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

_INT64_MAX = 2**63 - 1
_INT32_MAX = 2**31 - 1
_NoneType = type(None)
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
_UTC = timezone.utc
_BLOB_KINDS = (
    "snx"  # the kinds laid out as int32 offsets[n + 1] plus the bytes they index
)
_TICKS = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
# the seconds that java.time.LocalDateTime.ofEpochSecond takes (years -999999999 to 999999999): beyond them it throws in the image, in the
# middle of a batch. Only a timestamp[s] can pass them: an int64 of milliseconds is at most 9.2e15 seconds
_LDT_MIN_SECONDS = -31_557_014_167_219_200
_LDT_MAX_SECONDS = 31_556_889_864_403_199


class Packed:
    """One column laid out for the image: ``data``, ``aux`` and ``nulls`` are ``(address, bytes)``; ``keep`` holds what the addresses point into."""

    __slots__ = ("name", "kind", "dim", "n", "data", "aux", "nulls", "keep")

    def __init__(self, name, kind, dim, array, aux=None, nulls=None, keep=()):
        self.name = name
        self.kind = kind
        self.dim = dim
        self.n = int(array.shape[0]) - (
            1 if kind in _BLOB_KINDS else 0
        )  # a string, decimal or bytes column's array is its n + 1 offsets
        self.data = (array.ctypes.data, array.nbytes)
        self.aux = aux or (0, 0)
        self.nulls = (nulls.ctypes.data, nulls.nbytes) if nulls is not None else (0, 0)
        self.keep = (array, nulls, *keep)


class Declared:
    """The declared property types of a type, read only when a column needs them (``INTEGER``, ``FLOAT``, ``ARRAY_OF_FLOATS``).

    One ingest call makes one object (:func:`declared_for`) and every chunk of the call shares it, so the call reads ``schema:types``
    (every type and property in one reply) once, not once per chunk. The object is dropped when the call ends: nothing is kept across
    calls, so a schema change made between two calls is always seen.

    ``known`` (a dict) is the caller's own snapshot and is used as it is. Otherwise ``db`` is asked the first time :meth:`get` is
    called, and again when a statement has run on ``db`` since (its ``_command_count`` moved): a schema change made while the call
    runs (the rows come from a generator that runs a ``CREATE PROPERTY``, say) is a statement on the handle, so the next chunk sees it.
    A read that fails is not kept: the next :meth:`get` asks again.
    """

    def __init__(self, db=None, type_name=None, known=None):
        self._db, self._type, self._known = db, type_name, known
        self._pinned = known is not None
        self._stamp = None

    def get(self, name):
        if not self._pinned:
            stamp = getattr(self._db, "_command_count", None)
            if self._known is None or stamp != self._stamp:
                known = _declared_types(self._db, self._type)
                if known is None:
                    self._known = None
                    return None
                self._known, self._stamp = known, stamp
        return self._known.get(name)


def declared_for(db, type_name, declared=None):
    """The :class:`Declared` of one ingest call: ``declared`` itself when it is one (a call that goes on from one entry to another
    shares it), a dict as the caller's snapshot, None as read from the schema when a column needs it.
    """
    if isinstance(declared, Declared):
        return declared
    return Declared(db, type_name, declared)


def _declared_types(db, type_name):
    """``{property: TYPE}`` of ``type_name`` and its parent types (the nearest declaration wins); None when it cannot be read."""
    try:
        rows = {
            r["name"]: r for r in db.query("sql", "SELECT FROM schema:types").to_list()
        }
    except Exception:  # nosec B110 - the insert reports the real error
        return None
    out, seen, queue = {}, set(), [type_name]
    while queue:
        current = queue.pop(0)
        if current in seen or current not in rows:
            continue
        seen.add(current)
        for entry in rows[current].get("properties") or []:
            name, kind = _name_type(entry)  # a dict since #195, the text form before
            if name and kind:
                out.setdefault(name, str(kind))
        queue.extend(rows[current].get("parentTypes") or [])
    return out


def _bitmap(present):
    """Arrow validity bitmap of a bool array (True = present)."""
    return _np.packbits(present, bitorder="little")


def _numpy_column(name, values, declared, nulls=None):
    """A 1-D or 2-D numpy array as a column, or None."""
    k = values.dtype.kind
    if (
        values.ndim == 2
        and values.dtype
        == _np.float32  # only a float32 matrix is a column of vectors, as on the JPype path
        and values.shape[1] > 0
        and nulls is None
    ):
        arr = _np.ascontiguousarray(values, dtype=_np.float32)
        return Packed(name, "f", int(values.shape[1]), arr)
    if values.ndim != 1:
        return None
    if k == "b":
        return Packed(
            name, "b", 0, _np.ascontiguousarray(values).view(_np.uint8), nulls=nulls
        )
    if k in "iu":
        if (
            k == "u"
            and values.dtype.itemsize == 8
            and values.size
            and int(values.max()) > _INT64_MAX
        ):
            return None
        if k == "i" and values.dtype.itemsize <= 4 and declared.get(name) == "INTEGER":
            return Packed(
                name,
                "i",
                0,
                _np.ascontiguousarray(values, dtype=_np.int32),
                nulls=nulls,
            )
        return Packed(
            name, "q", 0, _np.ascontiguousarray(values, dtype=_np.int64), nulls=nulls
        )
    if k == "f":
        if values.dtype.itemsize == 4 and declared.get(name) == "FLOAT":
            return Packed(
                name,
                "e",
                0,
                _np.ascontiguousarray(values, dtype=_np.float32),
                nulls=nulls,
            )
        return Packed(
            name, "d", 0, _np.ascontiguousarray(values, dtype=_np.float64), nulls=nulls
        )
    return None


def _string_column(name, texts, nulls=None):
    """A list of str (a null already replaced by '') as a string column, or None (a lone surrogate, or 2 GiB of text)."""
    n = len(texts)
    blob = "".join(texts)
    if blob.isascii():
        data = blob.encode("ascii")
        lengths = _np.fromiter(map(len, texts), dtype=_np.int64, count=n)
    else:
        try:
            encoded = [t.encode("utf-8") for t in texts]
        except UnicodeEncodeError:
            return None  # a lone surrogate: the JSON path decides what happens to it
        data = b"".join(encoded)
        lengths = _np.fromiter(map(len, encoded), dtype=_np.int64, count=n)
    if len(data) > _INT32_MAX:
        return None
    offsets = _np.zeros(n + 1, dtype=_np.int32)
    _np.cumsum(lengths, out=offsets[1:], dtype=_np.int32)
    blob_array = (
        _np.frombuffer(data, dtype=_np.uint8) if data else _np.zeros(0, dtype=_np.uint8)
    )
    aux = (blob_array.ctypes.data, blob_array.nbytes) if data else (0, 0)
    return Packed(name, "s", 0, offsets, aux=aux, nulls=nulls, keep=(blob_array, data))


def _blob_column(name, kind, items, nulls=None):
    """A list of bytes (a null already replaced by b'') as an offsets-plus-bytes column of ``kind``, or None past 2 GiB."""
    n = len(items)
    data = b"".join(items)
    if len(data) > _INT32_MAX:
        return None
    lengths = _np.fromiter(map(len, items), dtype=_np.int64, count=n)
    offsets = _np.zeros(n + 1, dtype=_np.int32)
    _np.cumsum(lengths, out=offsets[1:], dtype=_np.int32)
    blob_array = (
        _np.frombuffer(data, dtype=_np.uint8) if data else _np.zeros(0, dtype=_np.uint8)
    )
    aux = (blob_array.ctypes.data, blob_array.nbytes) if data else (0, 0)
    return Packed(name, kind, 0, offsets, aux=aux, nulls=nulls, keep=(blob_array, data))


def _epoch_micros(value):
    """Microseconds since 1970-01-01T00:00:00 of the UTC wall clock of a ``datetime`` (a naive one is taken as it is, as ``_tag_datetime``)."""
    if value.utcoffset() is not None:
        value = value.astimezone(_UTC)
    seconds = (
        (value.toordinal() - _EPOCH_ORDINAL) * 86400
        + value.hour * 3600
        + value.minute * 60
        + value.second
    )
    return seconds * 1_000_000 + value.microsecond


def _list_column(name, values, declared):
    """A list or tuple of Python values as a column: one type (None allowed) or a list of equal-length float arrays; else None."""
    n = len(values)
    types = set(map(type, values))
    present = None
    if _NoneType in types:
        types.discard(_NoneType)
        present = _np.fromiter((v is not None for v in values), dtype=bool, count=n)
        nulls = _bitmap(present)
    else:
        nulls = None
    if not types:  # every value is None
        return _string_column(name, [""] * n, nulls)
    if len(types) != 1:
        return None
    (only,) = types
    if only is str:
        texts = (
            values if present is None else [v if v is not None else "" for v in values]
        )
        return _string_column(name, texts, nulls)
    if only is int or only is float or only is bool:
        filler = 0 if only is int else 0.0 if only is float else False
        vals = (
            values
            if present is None
            else [v if v is not None else filler for v in values]
        )
        dtype = (
            _np.int64 if only is int else _np.float64 if only is float else _np.uint8
        )
        try:
            arr = _np.array(vals, dtype=dtype)
        except OverflowError:
            return None  # an integer beyond 64 bits
        kind = "q" if only is int else "d" if only is float else "b"
        return Packed(name, kind, 0, arr, nulls=nulls)
    if issubclass(only, datetime):
        if only.__name__ == "NaTType":
            return None  # the JSON path stores it as a null
        try:
            ticks = [0 if v is None else _epoch_micros(v) for v in values]
        except (OverflowError, ValueError):
            return None  # the JSON path raises what it raised before
        return Packed(
            name, "T", 1_000_000, _np.array(ticks, dtype=_np.int64), nulls=nulls
        )
    if only is date:
        days = [0 if v is None else v.toordinal() - _EPOCH_ORDINAL for v in values]
        return Packed(name, "D", 0, _np.array(days, dtype=_np.int32), nulls=nulls)
    if only is Decimal:
        texts = []
        for v in values:
            if v is None:
                texts.append(b"")
            elif v.is_finite():
                texts.append(str(v).encode("ascii"))
            else:
                return None  # the JSON path sends a non-finite Decimal down the per-row path
        return _blob_column(name, "n", texts, nulls)
    if only is bytes or only is bytearray:
        return _blob_column(
            name, "x", values if present is None else [v or b"" for v in values], nulls
        )
    if only is list and declared.get(name) == "ARRAY_OF_FLOATS" and present is None:
        try:
            arr = _np.array(values, dtype=_np.float32)
        except (ValueError, TypeError):
            return None  # ragged, or not numbers
        if arr.ndim != 2 or arr.shape[1] == 0:
            return None
        return Packed(name, "f", int(arr.shape[1]), arr)
    if (
        only is _np.ndarray
        and present is None
        and all(
            v.ndim == 1 and v.dtype == _np.float32 and v.shape == values[0].shape
            for v in values
        )
        and values[0].shape[0] > 0
    ):
        return Packed(
            name,
            "f",
            int(values[0].shape[0]),
            _np.ascontiguousarray(_np.stack(values), dtype=_np.float32),
        )
    return None


def _arrow_column(name, arr, declared):
    """A ``pyarrow`` array as a column (strings in place), or None."""
    import pyarrow as pa

    types = pa.types
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.chunk(0) if arr.num_chunks == 1 else None
        if arr is None:
            return None
    if types.is_dictionary(arr.type):
        arr = arr.cast(arr.type.value_type)
    t, n = arr.type, len(arr)
    nulls = (
        _bitmap(arr.is_valid().to_numpy(zero_copy_only=False))
        if arr.null_count
        else None
    )
    is_text = types.is_string(t) or types.is_large_string(t)
    if is_text or types.is_binary(t) or types.is_large_binary(t):
        bufs = arr.buffers()
        off_buf, data_buf = bufs[1], bufs[2]
        if types.is_string(t) or types.is_binary(t):
            offsets = _np.frombuffer(off_buf, dtype=_np.int32)[
                arr.offset : arr.offset + n + 1
            ]
        else:
            wide = _np.frombuffer(off_buf, dtype=_np.int64)[
                arr.offset : arr.offset + n + 1
            ]
            if len(wide) and int(wide[-1]) > _INT32_MAX:
                return None
            offsets = wide.astype(_np.int32)
        offsets = _np.ascontiguousarray(offsets)
        size = 0 if data_buf is None else min(data_buf.size, _INT32_MAX)
        aux = (data_buf.address, size) if size else (0, 0)
        return Packed(
            name,
            "s" if is_text else "x",
            0,
            offsets,
            aux=aux,
            nulls=nulls,
            keep=(arr, off_buf, data_buf),
        )
    if types.is_timestamp(t):
        values = arr.cast(pa.int64()).fill_null(0).to_numpy(zero_copy_only=False)
        if t.unit == "s" and values.size:
            if values.min() < _LDT_MIN_SECONDS or values.max() > _LDT_MAX_SECONDS:
                return None  # the row path raises its usual error for it, before anything is stored
        return Packed(
            name,
            "T",
            _TICKS[t.unit],
            _np.ascontiguousarray(values, dtype=_np.int64),
            nulls=nulls,
        )
    if types.is_date(t):
        if types.is_date64(t):  # milliseconds since the epoch, floor-divided to days
            raw = arr.cast(pa.int64()).fill_null(0).to_numpy(zero_copy_only=False)
            days = raw // 86_400_000
            if days.size and (
                days.min() < -(_INT32_MAX + 1) or days.max() > _INT32_MAX
            ):
                return None  # int32 days would wrap: the row path raises its usual error for it
        else:
            days = arr.cast(pa.int32()).fill_null(0).to_numpy(zero_copy_only=False)
        return Packed(
            name, "D", 0, _np.ascontiguousarray(days, dtype=_np.int32), nulls=nulls
        )
    if types.is_decimal(t):
        texts = [b"" if v is None else str(v).encode("ascii") for v in arr.to_pylist()]
        return _blob_column(name, "n", texts, nulls)
    if types.is_integer(t) or types.is_floating(t) or types.is_boolean(t):
        if arr.null_count:
            arr = arr.fill_null(False if types.is_boolean(t) else 0)
        return _numpy_column(name, arr.to_numpy(zero_copy_only=False), declared, nulls)
    if (
        (types.is_list(t) or types.is_large_list(t) or types.is_fixed_size_list(t))
        and not arr.null_count
        and n
        and declared.get(name) == "ARRAY_OF_FLOATS"
        and (types.is_floating(t.value_type) or types.is_integer(t.value_type))
    ):
        flat = arr.flatten()
        if flat.null_count or len(flat) % n:
            return None
        dim = len(flat) // n
        if dim == 0:
            return None
        if not types.is_fixed_size_list(t):
            import pyarrow.compute as pc

            lengths = pc.list_value_length(arr)
            if pc.min(lengths).as_py() != dim or pc.max(lengths).as_py() != dim:
                return None
        values = flat.to_numpy(zero_copy_only=False).reshape(n, dim)
        return Packed(name, "f", dim, _np.ascontiguousarray(values, dtype=_np.float32))
    return None


def pack(columns, declared):
    """``(rows, [Packed])`` for a dict of name -> column, or None when any column cannot be laid out or the columns differ in length.

    A column is a numpy array (1-D ints, floats and bools, or a 2-D float32 matrix), a list or tuple of Python values of one type (None
    allowed), a list of equal-length 1-D float32 arrays, or a ``pyarrow`` array.
    """
    items = list(
        columns.items()
    )  # a dict, or a DataFrame (its columns are pandas Series: the row path takes those)
    if _np is None or not items:
        return None
    rows, packed = None, []
    for name, values in items:
        if type(values).__module__.split(".")[0] == "pyarrow":
            col = _arrow_column(name, values, declared)
        elif isinstance(values, _np.ndarray):
            if values.dtype.kind in "OUS":
                col = (
                    _list_column(name, values.tolist(), declared)
                    if values.ndim == 1
                    else None
                )
            else:
                col = _numpy_column(name, values, declared)
        elif isinstance(values, (list, tuple)) and values:
            col = _list_column(name, values, declared)
        else:
            col = None  # a pandas Series, a generator, an empty list: the row path handles them
        if col is None:
            return None
        if rows is not None and col.n != rows:
            return None  # the row path reports the ragged columns
        rows = col.n
        packed.append(col)
    return (rows, packed) if rows else None


def spec_of(packed):
    """The address table the image reads: six int64 per column (data, aux and null-bitmap address and byte count)."""
    import ctypes

    flat = []
    for col in packed:
        flat.extend((*col.data, *col.aux, *col.nulls))
    return (ctypes.c_int64 * len(flat))(*flat)


def header_of(rows, commit_every, packed, fid=None):
    """The JSON header text: row count, commit interval and ``[name, kind, dim]`` per column; with a failure token ``fid`` it also says
    ``"parallel": true``, which hands the records to the async executor (the image then ignores the commit interval and counts the
    failed records on that token).
    """
    import json

    header = {
        "n": rows,
        "commit": min(int(commit_every), 2**31 - 1),
        "cols": [[c.name, c.kind, c.dim] for c in packed],
    }
    if fid is not None:
        header["parallel"] = True
        header["fid"] = fid
    return json.dumps(header)


def rows_to_columns(rows):
    """A list of dicts with the same keys as a dict of name -> list of values, or None (no rows, different keys, a non-str key)."""
    if not rows:
        return None
    try:
        keys = list(rows[0])
        if any(type(key) is not str for key in keys):
            return None
        if set(map(len, rows)) != {len(keys)}:
            return None
        return {key: [row[key] for row in rows] for key in keys}
    except (
        KeyError,
        TypeError,
    ):  # same size but different keys, or a row that is not a dict
        return None
