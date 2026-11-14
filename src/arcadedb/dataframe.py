"""
ArcadeDB Python Bindings - DataFrame and Arrow ingest

Loads a pandas or Polars DataFrame, or an Arrow table, into a type in batches
(``Database.insert_dataframe`` and ``Database.insert_arrow``). Arrow is the one
internal form: a pandas frame becomes an Arrow table, a Polars frame is already
one, and each Arrow column becomes the typed array ``Database.insert_columns``
sends across the bridge, for a document type and for a vertex type alike (the
vertex type gets vertices from the same ``DocumentBatcher.insertColumns``; it
refuses to run inside a transaction the caller opened, as ``GraphBatch`` does).

The native backend (the ``arcadedb`` wheel) shares everything but the
last step: the checks, the type mapping, the batching and the error wording are
the same code. A batch of documents or vertices goes over by address through
the columnar entry (``arcadedb_n_insert_colp``) when every column can cross as
memory (numbers, bools, strings, float vectors, date-times, dates, decimals and bytes),
else as one JSON text through ``Database.insert_many`` (one crossing per 10,000 rows),
where a value JSON cannot carry (a date-time, a decimal, bytes) travels as a tagged
value the image decodes.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import jpype

from .exceptions import ArcadeDBError

_PYARROW_HINT = (
    "pyarrow is required for {what}. Install with: uv add pyarrow "
    "(or uv add 'arcadedb-jvm[arrow]')"
)

# Declared property types (the engine's Type names) that an Arrow column kind
# loads into. The engine converts the value (an integer into a LONG or a
# DECIMAL, a list into an ARRAY_OF_FLOATS) and refuses what does not fit (a
# LONG past INTEGER's range). What it also does, silently, is truncate a float
# into an integer property and parse a string into a number, so the kinds that
# would lose or guess are refused here, before anything is written.
_DATETIMES = {"DATETIME", "DATETIME_SECOND", "DATETIME_MICROS", "DATETIME_NANOS"}
_ARRAYS = {
    "ARRAY_OF_SHORTS",
    "ARRAY_OF_INTEGERS",
    "ARRAY_OF_LONGS",
    "ARRAY_OF_FLOATS",
    "ARRAY_OF_DOUBLES",
}
_ALLOWED = {
    "integer": {
        "BYTE",
        "SHORT",
        "INTEGER",
        "LONG",
        "FLOAT",
        "DOUBLE",
        "DECIMAL",
        "STRING",
    },
    "floating": {"FLOAT", "DOUBLE", "DECIMAL", "STRING"},
    "bool": {"BOOLEAN", "STRING"},
    "string": {"STRING", "LINK"},
    "binary": {"BINARY"},
    "timestamp": _DATETIMES | {"STRING"},
    "date": {"DATE"} | _DATETIMES | {"STRING"},
    "decimal": {"DECIMAL", "DOUBLE", "FLOAT", "STRING"},
    "list": {"LIST"} | _ARRAYS,
    "struct": {"MAP", "EMBEDDED"},
}


def _import_pyarrow(what: str):
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError(_PYARROW_HINT.format(what=what)) from exc
    return pa


def _kind(pa, arrow_type) -> Optional[str]:
    """The mapping family of an Arrow type, or None when it has no ArcadeDB mapping."""
    types = pa.types
    if types.is_dictionary(arrow_type):
        return _kind(pa, arrow_type.value_type)
    if types.is_integer(arrow_type):
        return "integer"
    if types.is_floating(arrow_type):
        return "floating"
    if types.is_boolean(arrow_type):
        return "bool"
    if (
        types.is_string(arrow_type)
        or types.is_large_string(arrow_type)
        or (hasattr(types, "is_string_view") and types.is_string_view(arrow_type))
    ):
        return "string"
    if types.is_binary(arrow_type) or types.is_large_binary(arrow_type):
        return "binary"
    if types.is_timestamp(arrow_type):
        return "timestamp"
    if types.is_date(arrow_type):
        return "date"
    if types.is_decimal(arrow_type):
        return "decimal"
    if (
        types.is_list(arrow_type)
        or types.is_large_list(arrow_type)
        or types.is_fixed_size_list(arrow_type)
    ):
        return "list"
    if types.is_struct(arrow_type):
        return "struct"
    if types.is_null(arrow_type):
        return "null"
    return None


def to_arrow_table(data, *, what: str):
    """Normalize a pandas or Polars DataFrame, or anything Arrow can read, to a ``pyarrow.Table``."""
    pa = _import_pyarrow(what)
    module = type(data).__module__.split(".")[0]
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    if isinstance(data, pa.RecordBatchReader):
        return data.read_all()
    if module == "pandas" and hasattr(data, "columns"):
        names = list(data.columns)
        bad = [n for n in names if not isinstance(n, str)]
        if bad:
            raise ValueError(
                f"column names must be strings, got {bad[0]!r}; rename the columns first"
            )
        if len(set(names)) != len(names):
            raise ValueError("a DataFrame with duplicate column names cannot be loaded")
        # NaN in a float column is pandas' missing value: Arrow stores it as a null.
        return pa.Table.from_pandas(data, preserve_index=False)
    if module == "polars":
        if hasattr(data, "collect") and not hasattr(data, "to_arrow"):
            raise TypeError(
                "a Polars LazyFrame is not loaded as it is; call .collect() first"
            )
        return data.to_arrow()
    if hasattr(data, "__arrow_c_stream__") or hasattr(data, "__arrow_c_array__"):
        return pa.table(data)  # any library that speaks the Arrow PyCapsule interface
    raise TypeError(
        f"{what} takes a pandas or Polars DataFrame or an Arrow table, "
        f"not {type(data).__name__}"
    )


def _is_native(db) -> bool:
    """True for the native backend's database (no JVM: schema reads are SQL, values cross as tagged JSON)."""
    return not hasattr(db, "_java_db")


_PROPERTY = re.compile(r"name: (.+?), type: ([A-Z_]+)")


def _property_name_type(entry):
    """(name, type) of an entry of a native ``schema:types`` row's ``properties``: a dict since #195, the engine's text form
    of the row before it; (None, None) when neither. Kept here, not imported from ``_native``, so the JVM build never loads it.
    """
    if isinstance(entry, dict):
        return entry.get("name"), entry.get("type")
    m = _PROPERTY.search(entry) if isinstance(entry, str) else None
    return (m.group(1), m.group(2)) if m else (None, None)


def _declared_types_native(
    db, type_name: str, names: List[str]
) -> Tuple[bool, Dict[str, str]]:
    """:func:`_declared_types` for the native backend: the ``schema:types`` rows, walking the parent types."""
    rows = {r["name"]: r for r in db.query("sql", "SELECT FROM schema:types").to_list()}
    row = rows.get(type_name)
    if row is None:
        raise ArcadeDBError(
            f"Cannot load into '{type_name}': the type does not exist "
            f"(create it first, with or without declared properties)"
        )
    kind = row.get("type")
    if kind == "edge":
        raise ArcadeDBError(
            f"Cannot load into '{type_name}': it is an edge type. Load the vertices, "
            f"then buffer the edges with graph_batch().new_edges()"
        )
    declared: Dict[str, str] = {}
    seen, queue = set(), [type_name]
    while queue:  # the type first, then its parents: the nearest declaration wins
        current = queue.pop(0)
        if current in seen or current not in rows:
            continue
        seen.add(current)
        for entry in rows[current].get("properties") or []:
            name, kind_name = _property_name_type(entry)
            if name in names and kind_name:
                declared.setdefault(name, str(kind_name))
        queue.extend(rows[current].get("parentTypes") or [])
    return kind == "vertex", declared


def _declared_types(
    db, type_name: str, names: List[str]
) -> Tuple[bool, Dict[str, str]]:
    """``(is_vertex_type, {column: declared property type name})`` for the columns the type declares.

    Raises ArcadeDBError when the type does not exist or is an edge type.
    """
    if _is_native(db):
        return _declared_types_native(db, type_name, names)
    java_schema = db._java_db.getSchema()
    if not java_schema.existsType(type_name):
        raise ArcadeDBError(
            f"Cannot load into '{type_name}': the type does not exist "
            f"(create it first, with or without declared properties)"
        )
    java_type = java_schema.getType(type_name)
    if isinstance(java_type, jpype.JClass("com.arcadedb.schema.EdgeType")):
        raise ArcadeDBError(
            f"Cannot load into '{type_name}': it is an edge type. Load the vertices, "
            f"then buffer the edges with graph_batch().new_edges()"
        )
    is_vertex = isinstance(java_type, jpype.JClass("com.arcadedb.schema.VertexType"))
    declared = {}
    for name in names:
        prop = java_type.getPolymorphicPropertyIfExists(name)
        if prop is not None:
            declared[name] = str(prop.getType().name())
    return is_vertex, declared


def _check_declared(
    type_name: str, schema, kinds: Dict[str, str], declared: Dict[str, str]
):
    """Refuse a column whose Arrow kind does not load into the property type the type declares."""
    for name, declared_type in declared.items():
        kind = kinds[name]
        if kind == "null":
            continue  # nothing but nulls: any property accepts it
        if declared_type not in _ALLOWED[kind]:
            raise TypeError(
                f"column {name!r} has Arrow type {schema.field(name).type} but "
                f"'{type_name}.{name}' is declared {declared_type}; cast the column to a "
                f"matching type first (e.g. df[{name!r}].astype(...))"
            )


def _temporal_column(chunk, pa):
    """A timestamp or date chunk as one ``TemporalColumn`` of the bridge, keeping the unit's precision.

    The integer epoch values cross as ONE ``long[]`` (a buffer copy), the null rows as one ``boolean[]`` when there is a null, and the
    bridge builds each ``LocalDateTime`` or ``LocalDate`` as it fills the document of that row: one call per column and batch, where a
    ``LocalDateTime.ofEpochSecond`` or ``LocalDate.ofEpochDay`` call per cell (and one conversion per element in ``insert_columns``) came
    before. ``insert_columns`` takes the column as it is.

    DATETIME is a UTC wall clock: a timestamp with a zone is already UTC epoch time, and a timestamp without
    one is taken as the wall clock it shows (the same rule ``Document.set`` applies to a ``datetime``), so both
    are read from the integer epoch value, in the unit of the column. Going through ``datetime`` would cut a nanosecond value to
    microseconds. A date32 is days, a date64 is milliseconds (the bridge divides by 86,400,000, rounding toward negative infinity).
    """
    import numpy as np

    types = pa.types
    if types.is_date(chunk.type):
        date64 = types.is_date64(chunk.type)
        raw = chunk.cast(pa.int64() if date64 else pa.int32())
        ticks, per_unit, factory = raw, 86_400_000 if date64 else 1, "dates"
    else:
        per_second = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
        ticks = chunk.cast(pa.int64())
        per_unit, factory = per_second[chunk.type.unit], "dateTimes"
    values = ticks.fill_null(0).to_numpy(zero_copy_only=False)
    nulls = None
    if chunk.null_count:
        nulls = jpype.JArray(jpype.JBoolean)(
            chunk.is_null().to_numpy(zero_copy_only=False)
        )
    return getattr(jpype.JClass("com.arcadedb.python.TemporalColumn"), factory)(
        jpype.JArray(jpype.JLong)(np.ascontiguousarray(values, dtype=np.int64)),
        per_unit,
        nulls,
    )


def _temporal_to_tagged(chunk, pa):
    """A timestamp or date chunk as tagged values (``T`` date-time, ``D`` date) for the native backend, with the same
    reading rules as :func:`_temporal_column`: ISO text of the integer epoch value, so a nanosecond value keeps its
    nanoseconds (a Python ``datetime`` would cut them to microseconds).
    """
    import numpy as np

    from ._native.codec import TAG

    types = pa.types
    mask = chunk.is_null().to_numpy(zero_copy_only=False)
    if types.is_date(chunk.type):
        raw = chunk.cast(pa.int64() if types.is_date64(chunk.type) else pa.int32())
        days = raw.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        if types.is_date64(chunk.type):
            days = days // 86_400_000
        text = days.astype("datetime64[D]").astype(str).tolist()
        tag = "D"
    else:
        unit = chunk.type.unit
        values = chunk.cast(pa.int64()).fill_null(0).to_numpy(zero_copy_only=False)
        text = (
            values.astype(np.int64).astype(f"datetime64[{unit}]").astype(str).tolist()
        )
        tag = "T"
    return [None if m else {TAG: [tag, s]} for s, m in zip(text, mask)]


def _scalars_to_tagged(kind: str, chunk):
    """A decimal or binary chunk as tagged values (``n`` decimal text, ``b`` base64) for the native backend."""
    from ._native.codec import TAG

    values = chunk.to_pylist()
    if kind == "decimal":
        return [None if v is None else {TAG: ["n", str(v)]} for v in values]
    import base64

    return [
        None if v is None else {TAG: ["b", base64.b64encode(v).decode("ascii")]}
        for v in values
    ]


def _is_number_list(pa, arrow_type) -> bool:
    """A list, large list or fixed-size list of integers or floats: the column the columnar entry lays out as a float32 matrix."""
    types = pa.types
    if not (
        types.is_list(arrow_type)
        or types.is_large_list(arrow_type)
        or types.is_fixed_size_list(arrow_type)
    ):
        return False
    return types.is_floating(arrow_type.value_type) or types.is_integer(
        arrow_type.value_type
    )


_JAVA_INT_MAX = 2**31 - 1


def _float32_matrix(pa, chunk):
    """A list, large list or fixed-size list column of float32 with no null row, no null element and one length, as one 2-D float32
    numpy matrix (a view of Arrow's value buffer: ``flatten`` follows a slice's offset), else None. None means "not this shape" and
    leaves the column to ``to_pylist``, which keeps what that route stored or refused: a null row stays null, a null element is
    refused, vectors of different length are stored as given or refused by a vector index. Only float32 is taken: a float64 or integer
    value is narrowed by the engine (and refused past the float range), which a cast here would not do.
    """
    types = pa.types
    arrow_type = chunk.type
    if not (
        types.is_list(arrow_type)
        or types.is_large_list(arrow_type)
        or types.is_fixed_size_list(arrow_type)
    ):
        return None
    n = len(chunk)
    if not types.is_float32(arrow_type.value_type) or not n or chunk.null_count:
        return None
    flat = chunk.flatten()
    if flat.null_count or len(flat) % n or len(flat) > _JAVA_INT_MAX:
        return None
    dim = len(flat) // n
    if dim == 0:
        return None
    if not types.is_fixed_size_list(arrow_type):
        import pyarrow.compute as pc

        lengths = pc.list_value_length(chunk)
        if pc.min(lengths).as_py() != dim or pc.max(lengths).as_py() != dim:
            return None
    return flat.to_numpy(zero_copy_only=False).reshape(n, dim)


def _column_values(
    name: str,
    chunk,
    pa,
    native: bool = False,
    arrow: bool = False,
    float_vectors: bool = False,
):
    """One Arrow array as what ``Database.insert_columns`` takes: a numpy array when the column crosses as one
    buffer (integers, floats and booleans without nulls), else a list of values (``None`` is a null). On the JVM backend a timestamp or
    date column is one bridge ``TemporalColumn`` instead (``_temporal_column``: one ``long[]`` of epoch ticks, no value per cell).

    ``float_vectors`` (the JVM backend, a property declared ``ARRAY_OF_FLOATS``): a float32 vector column of one length and no null is
    one 2-D float32 matrix, which ``insert_columns`` sends as one ``float[]``; ``to_pylist`` made one Python float per element and
    ``insert_columns`` one ``ArrayList`` of boxed doubles per row (see ``_float32_matrix`` for what is left to it).

    ``arrow`` (the native backend's document and vertex load): a string, binary, timestamp, date or decimal column, a list of
    numbers (a vector) and a numeric column with nulls, stay Arrow arrays, which the columnar entry reads in place (the offsets,
    the bytes, the integer ticks, the vector values and the validity bitmap) instead of as Python values; ``_insert_documents_native``
    turns them into Python or tagged values for the row path when a batch cannot go columnar.
    """
    types = pa.types
    if types.is_dictionary(chunk.type):  # a pandas categorical or a Polars Categorical
        chunk = chunk.cast(chunk.type.value_type)
    kind = _kind(pa, chunk.type)
    if arrow and (
        (
            kind == "string"
            and (types.is_string(chunk.type) or types.is_large_string(chunk.type))
        )
        or (kind in ("integer", "floating", "bool") and chunk.null_count)
    ):
        if not (
            kind == "integer"
            and types.is_uint64(chunk.type)
            and len(chunk) > chunk.null_count
        ):  # uint64 keeps the range check below
            return chunk
    if kind in ("integer", "floating", "bool"):
        if native and types.is_uint64(chunk.type) and len(chunk) > chunk.null_count:
            import pyarrow.compute as pc

            if (
                pc.max(chunk).as_py() > 2**63 - 1
            ):  # insert_columns refuses it on the JPype path, with this text
                raise ValueError(
                    f"column {name!r} holds a value beyond the 64-bit signed range"
                )
        if chunk.null_count == 0:
            values = chunk.to_numpy(zero_copy_only=False)
            if values.dtype.kind == "f" and values.dtype.itemsize < 8:
                values = values.astype("float64")  # float16/32 widen exactly
            return values
        return chunk.to_pylist()
    if arrow and kind in ("timestamp", "date", "decimal", "binary"):
        return chunk
    if arrow and kind == "list" and _is_number_list(pa, chunk.type):
        return chunk  # a vector column: the columnar entry reads the values in place (to_pylist makes one Python float per element)
    if float_vectors and kind == "list":
        matrix = _float32_matrix(pa, chunk)
        if matrix is not None:
            return matrix
    if kind in ("timestamp", "date"):
        return _temporal_to_tagged(chunk, pa) if native else _temporal_column(chunk, pa)
    if native and kind in ("decimal", "binary"):
        return _scalars_to_tagged(kind, chunk)
    if kind == "null":
        return [None] * len(chunk)
    # strings, binary, decimal, list, struct: Python values, which insert_columns converts element by element
    # (str, bytes -> byte[], Decimal -> BigDecimal, list -> LIST, dict -> MAP)
    return chunk.to_pylist()


def insert_table(
    db,
    type_name: str,
    table,
    *,
    batch_size: int,
    parallel: bool,
    what: str,
) -> int:
    """Load a ``pyarrow.Table`` into ``type_name`` in batches of ``batch_size`` rows; see ``Database.insert_arrow``."""
    pa = _import_pyarrow(what)
    db._check_not_closed()
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    names = list(table.column_names)
    if not names:
        raise ValueError(f"{what} needs at least one column")
    if len(set(names)) != len(names):
        raise ValueError("a table with duplicate column names cannot be loaded")

    kinds = {}
    for field in table.schema:
        kind = _kind(pa, field.type)
        if kind is None:
            raise TypeError(
                f"column {field.name!r} has Arrow type {field.type}, which has no ArcadeDB "
                f"mapping; convert it first (supported: integers, floats, bool, string, "
                f"binary, timestamp, date, decimal, list, struct, null)"
            )
        kinds[field.name] = kind
    is_vertex, declared = _declared_types(db, type_name, names)
    _check_declared(type_name, table.schema, kinds, declared)
    if parallel and is_vertex:
        raise ValueError(
            "parallel=True writes documents through the async executor; "
            f"'{type_name}' is a vertex type, which loads through graph_batch()"
        )
    if table.num_rows == 0:
        return 0

    native = _is_native(db)
    written = 0
    # a document load inside the caller's own transaction joins it (insert_columns commits
    # nothing then); a vertex load refuses to run inside one, as GraphBatch.createVertices does.
    # parallel=True never joins it: the async writers commit the records themselves, so a rollback
    # of the caller's transaction does not remove them
    in_transaction = bool(db.is_transaction_active())
    in_caller_transaction = in_transaction and not parallel
    if is_vertex and in_transaction:
        raise ArcadeDBError(
            f"Failed to create batch vertices for type '{type_name}': a vertex load commits its own batches "
            f"and cannot run inside a transaction the caller opened: commit or roll back that transaction first"
        )
    for batch in table.to_batches(max_chunksize=batch_size):
        if batch.num_rows == 0:
            continue
        columns = {
            name: _column_values(
                name,
                batch.column(i),
                pa,
                native,
                arrow=native,
                float_vectors=not native and declared.get(name) == "ARRAY_OF_FLOATS",
            )
            for i, name in enumerate(names)
        }
        try:
            if native:
                _insert_documents_native(
                    db, type_name, names, columns, batch_size, parallel, declared
                )
            else:
                db.insert_columns(
                    type_name, columns, commit_every=batch_size, parallel=parallel
                )
        except ArcadeDBError as exc:
            if written and in_caller_transaction:
                # the batches joined the caller's transaction: nothing is committed
                # until the caller commits, and a rollback removes them
                raise ArcadeDBError(
                    f"{exc}; the {written} row(s) of the batches before this one "
                    f"were written in your transaction, not committed"
                ) from exc
            if written and parallel:
                # the async writers commit on their own, inside the caller's transaction or not
                raise ArcadeDBError(
                    f"{exc}; the {written} row(s) of the batches before this one "
                    f"were handed to the async writers, which commit on their own: "
                    f"a rollback of your transaction does not remove them"
                ) from exc
            if written:
                raise ArcadeDBError(
                    f"{exc}; the {written} row(s) of the batches before this one "
                    f"were committed"
                ) from exc
            raise
        written += batch.num_rows
        del (
            columns,
            batch,
        )  # sent: drop this batch's columns before the next batch is built, so two batches are never alive together
    return written


def _arrow_row_values(col) -> list:
    """A column kept as an Arrow array, as the values the row path takes: Python values, or tagged values for a timestamp, date,
    decimal or binary column (the form it had before the columnar entry took them, so the stored value is unchanged).
    """
    import pyarrow as pa

    kind = _kind(pa, col.type)
    if kind in ("timestamp", "date"):
        return _temporal_to_tagged(col, pa)
    if kind in ("decimal", "binary"):
        return _scalars_to_tagged(kind, col)
    return col.to_pylist()


def _insert_documents_native(
    db,
    type_name: str,
    names: List[str],
    columns: Dict[str, Any],
    batch_size: int,
    parallel: bool,
    declared: Optional[Dict[str, str]] = None,
):
    """One batch of documents or vertices on the native backend. When every column can cross as memory (numbers, bools, strings,
    nulls, date-times, dates, decimals, bytes, float vectors for an ``ARRAY_OF_FLOATS`` property) the batch goes through the
    columnar entry, no Python values and no JSON text; otherwise the columns become one dict per row and go to ``insert_many`` (one crossing per 10,000 rows). Either way
    the batch is one transaction, or joins the caller's. With ``parallel`` the same columnar entry hands the records to the async
    writers (no transaction), and the call waits for them and raises when a record failed, as ``insert_many(parallel=True)`` does.
    """
    if parallel:
        from ._native.bulk import insert_columns_parallel

        if (
            insert_columns_parallel(
                db, type_name, columns, {} if declared is None else declared
            )
            is not None
        ):
            return
    else:
        from ._native.bulk import insert_columns_fast

        if (
            insert_columns_fast(
                db,
                type_name,
                columns,
                batch_size,
                declared={} if declared is None else declared,
                prefix=f"Failed to bulk-insert into '{type_name}'",
            )
            is not None
        ):
            return
    lists = []
    for name in names:
        col = columns[name]
        if hasattr(
            col, "to_pylist"
        ):  # a column kept as an Arrow array for the columnar entry
            lists.append(_arrow_row_values(col))
        else:
            # jvm-payload: not-a-crossing -- no JVM here: insert_many serialises the rows to JSON text
            lists.append(col.tolist() if hasattr(col, "tolist") else col)
    # the call's declared types (read once by insert_table) go on to the row path, so it does not read the schema again per chunk
    db._insert_many(
        type_name,
        [dict(zip(names, row)) for row in zip(*lists)],
        batch_size,
        parallel,
        declared,
    )
