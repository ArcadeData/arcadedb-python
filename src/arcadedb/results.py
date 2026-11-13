"""
ArcadeDB Python Bindings - Result Set Wrappers

ResultSet and Result classes for wrapping query results.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import jpype as _jpype
from jpype import JException

from ._columnar import (
    dataframe_from_columns,
    decode_arrow_frame,
    decode_columns_frame,
    empty_arrow,
    empty_columns,
    frame_rows,
    merge_arrow,
    merge_columns,
)
from ._logging import get_logger
from .callbacks import raise_pending_interrupt
from .exceptions import ArcadeDBError
from .graph import Document, Edge, Vertex
from .type_conversion import convert_java_to_python

_logger = get_logger(__name__)
_BRIDGE_CLASSES: dict = {}


# Rows per crossing for ResultSet.to_list(): large enough that the per-batch
# cost (one crossing, one decoder) disappears. TypedRows also ends a batch once
# its JSON text reaches a few megabytes, so rows of any width cross in pieces
# that stay that small.
_TYPED_BATCH_ROWS = 1000

_TYPED_TAG = "\u0001"
_TYPED_SIMPLE = {
    "n": Decimal,
    "i": int,
    "D": date.fromisoformat,
    "T": datetime.fromisoformat,
    "Z": datetime.fromisoformat,
    "s": set,
}


# NaN and the infinities: JSON has no literal for them and the engine's writers make null of them, so the
# bridge (RowBatcher, ColumnBatcher) writes each one as {"\u0001": ["f", "NaN"]} (the text is Double.toString).
_NONFINITE_KIND = "f"
_NONFINITE_TEXT = frozenset(("NaN", "Infinity", "-Infinity"))


def _nonfinite_hook(obj):
    """The json object_hook that turns the bridge's non-finite tag back into a float."""
    tagged = obj.get(_TYPED_TAG)
    if (
        isinstance(tagged, list)
        and len(tagged) == 2
        and tagged[0] == _NONFINITE_KIND
        and tagged[1] in _NONFINITE_TEXT
    ):
        return float(tagged[1])
    return obj


def _loads_nonfinite(text):
    """``json.loads`` of bridge JSON (a str or UTF-8 bytes) that may hold the non-finite tag.

    The hook runs only when the tag character appears in the text (the writer
    escapes it as ``\\u0001``), so a batch with no NaN or infinity parses at
    the speed of the plain C decoder.
    """
    if isinstance(text, (bytes, bytearray, memoryview)):
        text = bytes(text).decode("utf-8")
    if "\\u0001" in text or _TYPED_TAG in text:
        return json.loads(text, object_hook=_nonfinite_hook)
    return json.loads(text)


def _typed_hook(side):
    """The json object_hook that restores the values TypedRows tagged."""

    def hook(obj):
        tagged = obj.get(_TYPED_TAG)
        if tagged is None:
            return obj
        kind, payload = tagged
        restore = _TYPED_SIMPLE.get(kind)
        if restore is not None:
            return restore(payload)
        if kind == "x":
            return convert_java_to_python(side[payload])
        # "r": a row whose property name is the tag key, sent as names and values
        pair = side[payload]
        return {str(n): convert_java_to_python(v) for n, v in zip(pair[0], pair[1])}

    return hook


def _typed_json_values(data, side, count):
    """The values of a ``json`` column of a ColumnBatcher frame: the tagged JSON (TypedRows) with its side array of Java objects."""
    return json.loads(bytes(data), object_hook=_typed_hook(side))


def _loads_typed_row(java_row):
    """The dict of one row or record that ``ReaderCalls.rowJson`` / ``documentJson`` wrote.

    ``java_row`` is ``{String json, Object[] side}``. The side array is read only
    when the text carries the tag character (the writer escapes it as
    ``\\u0001``): a row of plain values costs one array read and one string read,
    and the C ``json`` decoder, and no other JPype call.
    """
    text = str(java_row[0])
    if "\\u0001" not in text:
        return json.loads(text)
    return json.loads(text, object_hook=_typed_hook(java_row[1]))


def _loads_typed_batch(java_batch):
    """The rows of one ``TypedRows.nextRows`` batch, ``{String json, Object[] side, Boolean drained}``.

    The hook, and the read of the side array that builds it, are needed only when the text holds a
    tag. TypedRows writes every tag as ``{"\\u0001": [kind, payload]}`` with the escape spelled out
    (``tagged``, and the ``r`` row of ``appendRow``), writes a control character of a string as the
    same escape (``quote``), and never writes the raw character, and every entry of the side array
    is referred to by such a tag. So a text without the escape holds no tag and no side index: the
    hook would return every object unchanged, and ``json.loads(text)`` is the same value without it.
    A string that holds U+0001, or the text of the escape, also puts the escape into the text; that
    only keeps the hook, which then changes nothing.
    """
    text = str(java_batch[0])
    if "\\u0001" not in text:
        return json.loads(text)
    return json.loads(text, object_hook=_typed_hook(java_batch[1]))


def _loads_names(java_text):
    """The property names ``ReaderCalls.rowNames`` / ``documentNames`` wrote, as a tuple."""
    return tuple(json.loads(str(java_text)))


def _bridge_class(name):
    """Cached bridge-class lookup; None (with one warning) if unavailable.

    In an installed wheel the bridge jar is always on the classpath, so the
    None path only occurs in broken source checkouts."""
    if name in _BRIDGE_CLASSES:
        return _BRIDGE_CLASSES[name]
    import jpype

    if not jpype.isJVMStarted():
        # Asked too early: say so, but do not cache it, or every later call
        # would take the slow path for the life of the process.
        return None
    try:
        cls = jpype.JClass(f"com.arcadedb.python.{name}")
    except Exception:
        cls = None
        _logger.warning(
            "bridge class %s unavailable; falling back to the slow per-row "
            "path (is the bridge jar missing from this build?)",
            name,
        )
    _BRIDGE_CLASSES[name] = cls
    return cls


def _read_error(exc: BaseException) -> ArcadeDBError:
    """The ArcadeDBError for a Java exception raised while rows were read.

    The engine plans lazily, so a statement's error can surface on the first
    or any later row, after ``query()`` or ``command()`` returned (#173). Raise
    it ``from`` the Java exception, which ``str()`` then names as the cause.
    """
    error = ArcadeDBError(f"Reading the result set failed: {exc}")
    error.__cause__ = exc
    raise_pending_interrupt(
        error, exc
    )  # raises KeyboardInterrupt when a Python function left one
    return error


def _java_class_name(value: Any) -> str:
    return str(value.getClass().getName())


class ResultSet:
    """Iterator wrapper for ArcadeDB query results.

    The Java result set is closed as soon as it is exhausted (by iteration or
    any of the ``to_*`` methods), by ``first()``/``one()`` once they have their
    row, and when this object is freed. Since the engine's parallel scan
    (ArcadeData/arcadedb#8524, 26.10.1) an unclosed result set keeps its scan's
    producer threads parked for up to ``arcadedb.parallelScanAbandonedTimeout``
    (10 minutes), and a few of them stall the next query that needs the pool
    (ArcadeData/arcadedb#8594).

    A result set read to its end reads as empty afterwards. One closed before
    its end (by ``first()``, ``one()``, ``close()``, or leaving its ``with``
    block) raises ArcadeDBError when read again: the rows it had not returned
    are gone, and returning nothing would hide that. So does one whose
    ``Database`` was closed before it was read to its end: a record row is
    loaded lazily from the open database, and an empty row would hide that.

    A result set keeps its ``Database`` alive, so a function may open a
    database, query it, and return the result without closing anything.

    The engine plans and computes rows lazily, so an error in the statement
    can surface while its rows are read rather than in ``query()``. Every way
    of reading them (iteration, ``to_list()``, ``to_json_list()``, the
    columnar readers, ``first()``, ``one()``, ``count()``) raises it as
    ArcadeDBError, with the Java exception as its cause.
    """

    def __init__(self, java_result_set, database=None):
        self._java_result_set = java_result_set
        self._database = database  # strong reference, see the class docstring
        self._closed = False
        self._exhausted = False

    def __iter__(self) -> Iterator["Result"]:
        return self

    def _readable(self) -> bool:
        """True while rows can still come; False once read to the end.

        Raises ArcadeDBError for a result set closed before its end, or whose
        database was closed before its end. What a closed Java result set
        returns is the engine's business and has changed between builds, so it
        is never asked.
        """
        database = self._database
        if database is not None and database._closed and not self._exhausted:
            raise ArcadeDBError(
                "Database is closed: the rows of this result set cannot be "
                "read after their database was closed. Read them before "
                "closing the database."
            )
        if not self._closed:
            return True
        if self._exhausted:
            return False
        raise ArcadeDBError(
            "This result set was closed before all its rows were read (by "
            "first(), one(), close(), or leaving its with block), so the rows "
            "it had not returned are gone. Run the query again to read them."
        )

    def _finish(self) -> None:
        self._exhausted = True
        self.close()

    def _drained(self) -> None:
        """The bridge returned its last batch and closed the Java result set itself.

        ``RowBatcher.nextJsonBatch`` and ``RowAccess.nextRows`` return fewer rows
        than asked for only when the result set is drained, and close it before
        returning, so Python marks the set finished without another JVM
        crossing (a JPype call is 3 to 4 microseconds, a fifth of a one-row
        read through ``to_json_list()``).
        """
        self._exhausted = True
        self._closed = True

    def __next__(self) -> "Result":
        if self._readable():
            try:
                reader = _bridge_class("ReaderCalls")
                if reader is not None:
                    # hasNext() and next() in ONE crossing: the row is still read
                    # when it is asked for, and the end is the same null as
                    # hasNext() false
                    java_row = reader.nextOrNull(self._java_result_set)
                    if java_row is not None:
                        return Result(java_row, self._database)
                elif self._java_result_set.hasNext():
                    return Result(self._java_result_set.next(), self._database)
            except JException as exc:
                raise _read_error(exc) from exc
        if not self._closed:
            self._finish()
        raise StopIteration

    def to_list(self, convert_types: bool = True) -> List[Dict[str, Any]]:
        """
        Convert all results to list of dictionaries.

        Rows come over from the JVM in batches (the bridge's ``TypedRows``):
        Java writes each batch as JSON, tags the values JSON cannot carry
        exactly (DECIMAL, DATE, DATETIME, sets) and hands everything else
        (RIDs, embedded documents, ``float[]``, ...) over as the engine's own
        object, so each value has the Python type it always had and no value
        crosses the JVM boundary on its own. That is within a few percent of
        ``to_json_list()`` on a wide scan.

        For large results, ``to_columns()``, ``to_dataframe()`` or
        ``to_arrow()`` move the data as columns and are faster still.
        ``to_json_list()`` returns the same shape with JSON-native values
        (DATE and DATETIME as epoch-millisecond integers, DECIMAL as float),
        which only matters when you want that form.

        Args:
            convert_types: Convert Java types to Python (default: True)

        Returns:
            List of dictionaries with result data

        Example:
            >>> results = db.query("sql", "SELECT FROM User LIMIT 10")
            >>> users = results.to_list()
            >>> print(users[0])
            {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}
        """
        if convert_types:
            # The whole result is consumed, so rows can come over in batches:
            # one crossing per batch instead of hasNext/next and a Result
            # wrapper per row. iter_dicts() reads one row per crossing instead,
            # because a batch taken ahead would consume rows a lazy caller may
            # still want.
            typed_rows = _bridge_class("TypedRows")
            if typed_rows is not None:
                return self._to_list_typed(typed_rows)
            row_access = _bridge_class("RowAccess")
            if row_access is not None:
                out: List[Dict[str, Any]] = []
                if not self._readable():
                    return out
                while True:
                    try:
                        batch = row_access.nextRows(self._java_result_set, 512)
                    except JException as exc:
                        raise _read_error(exc) from exc
                    count = len(batch)
                    for pair in batch:
                        out.append(
                            {
                                str(name): convert_java_to_python(value)
                                for name, value in zip(pair[0], pair[1])
                            }
                        )
                    if count < 512:
                        # a short batch is the last one, and the bridge closed
                        # the result set: no second call just to see an empty batch
                        self._drained()
                        return out
        return list(self.iter_dicts(convert_types=convert_types))

    def _to_list_typed(self, typed_rows) -> List[Dict[str, Any]]:
        """``to_list()`` through the bridge's ``TypedRows``.

        Java writes each batch as JSON and tags the values JSON cannot carry
        exactly (DECIMAL, dates and datetimes, sets) and refers to everything
        else (RIDs, embedded documents, non-finite floats, ...) by index into
        a side array of the engine's own objects, so every value comes back
        as the type ``convert_java_to_python`` gives it. The C ``json`` module
        parses the text and one hook call per tagged value restores it; a batch
        with no tag in its text is parsed without the hook and its side array
        is not read (``_loads_typed_batch``).
        """
        out: List[Dict[str, Any]] = []
        if not self._readable():
            return out
        while True:
            try:
                java_batch = typed_rows.nextRows(
                    self._java_result_set, _TYPED_BATCH_ROWS
                )
            except JException as exc:
                raise _read_error(exc) from exc
            out.extend(_loads_typed_batch(java_batch))
            if java_batch[2]:
                # the result set is drained, and the bridge closed it: no
                # second call just to see an empty batch. (A batch also ends
                # early, short of _TYPED_BATCH_ROWS, once its JSON text is a
                # few megabytes, so a short batch alone does not say drained.)
                self._drained()
                return out

    def iter_dicts(self, convert_types: bool = True) -> Iterator[Dict[str, Any]]:
        """
        Iterate results as dictionaries.

        Each row is read when you ask for it, in one crossing into the JVM
        (``TypedRows``, as for ``to_list()``), so stopping early or changing
        records inside the loop behaves as it always did.

        Args:
            convert_types: Convert Java types to Python (default: True)

        Yields:
            Result rows as dictionaries
        """
        typed_rows = _bridge_class("TypedRows") if convert_types else None
        if typed_rows is None:
            for result in self:
                yield result.to_dict(convert_types=convert_types)
            return
        # One row per crossing, read when the caller asks for it, exactly as
        # before (a batch taken ahead would read rows a lazy caller may not
        # want, or that its own loop body is about to change). It replaces the
        # hasNext, next, wrapper, and per-property reads of the row-by-row
        # path with one crossing and one JSON parse.
        while self._readable():
            try:
                java_batch = typed_rows.nextRows(self._java_result_set, 1)
            except JException as exc:
                raise _read_error(exc) from exc
            rows = _loads_typed_batch(java_batch)
            if not rows:
                # nextRows closed the drained result set itself
                self._drained()
                return
            yield rows[0]

    def close(self) -> None:
        """
        Close the underlying Java result set. Idempotent.

        Exhausting the result set closes it already; call this (or use the
        result set as a context manager) when you stop reading early. An
        unclosed result set is not only held memory: since 26.10.1's parallel
        scan it can hold engine threads that other queries need
        (ArcadeData/arcadedb#8594).
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._java_result_set.close()
        except Exception:  # nosec B110 - close() is best-effort hygiene
            pass

    def __del__(self):
        # A result set abandoned mid-iteration (a `break`) is freed right away
        # under CPython's reference counting, so its engine-side cursor is
        # released then rather than after the engine's abandonment timeout.
        # Best effort: the JVM may already be gone at interpreter exit.
        if not getattr(self, "_closed", True):
            try:
                if _jpype.isJVMStarted():
                    self.close()
            except Exception:  # nosec B110 - finalizer must never raise
                pass

    def __enter__(self) -> "ResultSet":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def to_json_list(self, batch_size: int = 10_000) -> List[Dict[str, Any]]:
        """
        Bulk-materialize all rows via batched Java-side JSON serialization.

        The fast path for large result sets: rows are serialized to JSON in
        batches on the Java side (one JPype crossing per batch instead of
        several per row) and parsed with the C json module. Measured ~5.5x
        faster than ``to_list()`` used to be on a 10,000-row, nine-property
        scan (578 ms against 103 ms, laptop, 2026-09-27); ``to_list()`` now
        reads rows the same way and keeps the Python types.

        Values carry JSON-native types. Numbers, strings, booleans, lists and
        nested maps convert as expected, but DATE and DATETIME values arrive
        as epoch-millisecond integers (not ``datetime``) and DECIMALs as
        floats. Use ``to_list()`` when you want ``date``, ``datetime`` and
        ``Decimal`` values: it costs about the same.

        A record id (a ``LINK`` value, ``@rid``) arrives as its text
        (``"#10:5"``), at the top level and inside a list or a map, and a map key
        that is not a string arrives as text (``"1"``, ``"true"``, ``"null"``),
        on both builds. On the native build the conversion over the rows runs
        only for a result in which a reply carried a value that JSON cannot
        carry (DECIMAL, DATE, DATETIME, a set, NaN, a record id, ...); any other
        result is returned as parsed.

        NaN, Infinity and -Infinity arrive as Python floats (``float('nan')``,
        ``float('inf')``), at the top level and inside a list or a map, as on
        the native build. JSON has no literal for them, so the bridge writes
        each as a tag that is turned back into a float here: the result is not
        strict JSON, and ``json.dumps(..., allow_nan=False)`` of it raises.

        The encoding is one rule for every row, at the top level and inside a
        list or a map: a DATE is the epoch milliseconds of its UTC midnight (not
        a count of days) whatever the JVM time zone, the same number
        ``to_columns()`` carries as ``datetime64[ms]``. A date too far from 1970
        for a long of milliseconds (past about year 292 million) arrives as its
        ISO-8601 text instead of failing the batch.

        This is NOT the form of ``Result.to_json()`` for a record row (``SELECT
        FROM T``), which is the engine's round-trip JSON of the record: there a
        DATE column at the top level is epoch DAYS, with the ``@rid``, ``@type``
        and ``@cat`` metadata. Do not mix the two when decoding a DATE.

        Args:
            batch_size: Rows serialized per Java crossing (default 10000)

        Returns:
            List of dictionaries with JSON-native values

        Example:
            >>> rows = db.query("sql", "SELECT FROM Doc").to_json_list()
        """
        rows: List[Dict[str, Any]] = []
        for batch in self.iter_json_batches(batch_size=batch_size):
            rows.extend(batch)
        return rows

    def iter_json_batches(
        self, batch_size: int = 10_000
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Yield rows as lists of dicts, one Java-serialized batch at a time.

        Streaming counterpart of :meth:`to_json_list` with the same JSON-native
        type semantics; bounds memory to one batch. Falls back to chunked
        per-row conversion when the bridge jar is unavailable.

        On the native build the same property holds: ``batch_size`` bounds
        memory (one reply of the cursor plus one batch), and the next reply is
        read only when the next batch is asked for.

        One laptop study on the native build, 1,000,000 rows read with ``batch_size=10_000``, 10 clean pairs of this change against the previous behaviour; each figure is the ratio of the new value to the old one, so below 1 is lower. Peak RSS: 0.286 (95% interval 0.284 to 0.289). Time to the first batch: 0.010 (95% interval 0.009 to 0.010). Wall time of the whole read: 0.958 (95% interval 0.952 to 0.966). This is one machine and one result shape, not a speed promise.
        """
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        row_batcher = _bridge_class("RowBatcher")
        if row_batcher is None:
            chunk: List[Dict[str, Any]] = []
            for row in self.iter_dicts():
                chunk.append(row)
                if len(chunk) >= batch_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
            return

        if not self._readable():
            return
        size = int(batch_size)
        while True:
            try:
                java_batch = row_batcher.nextJsonBatch(self._java_result_set, size)
            except JException as exc:
                raise _read_error(exc) from exc
            batch = _loads_nonfinite(str(java_batch))
            if len(batch) < size:
                # nextJsonBatch stops short only when the result set is drained, so a
                # short batch is the last one: no second call (a JVM crossing, a
                # str(), and a json.loads) just to see "[]". It closed the result
                # set itself, so there is no close() crossing either.
                self._drained()
                if batch:
                    yield batch
                return
            yield batch

    def to_dataframe(self, convert_types: bool = True):
        """
        Convert results to pandas DataFrame.

        Requires pandas to be installed.

        Values are the ones ``to_list()`` returns, with two exceptions that are
        the same on both builds. A DATETIME keeps its microseconds (a
        ``datetime64[us]`` column, where ``to_columns()`` and ``to_arrow()``
        carry milliseconds), a DATE is a ``datetime.date`` in an object column,
        and a ``Decimal`` stays one. A column of ``float[]`` vectors holds one
        numpy ``float32`` array per cell (a row of the matrix ``to_columns()``
        returns, no copy into a Python list per row) and None for a null;
        ``to_list()`` gives lists of floats. A column that holds an integer
        beyond +-2**53 next to a float is an ``object`` column of the exact
        Python values (``float64`` would round the integer), as ``to_columns()``
        returns them; ``to_arrow()`` gives the exact text.

        Args:
            convert_types: Convert Java types to Python (default: True)

        Returns:
            pandas DataFrame

        Raises:
            ImportError: If pandas is not installed

        Example:
            >>> results = db.query("sql", "SELECT FROM User")
            >>> df = results.to_dataframe()
            >>> print(df.describe())
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: uv add pandas"
            ) from exc

        if convert_types:
            # fast path: columnar binary transport straight into typed
            # columns (numpy dtypes, real datetime64) — measured ~2x the JSON
            # row path and ~6x the per-row path on 100k-row scans
            # Datetimes cross with their microseconds (datetime64[us]) and dates as
            # date objects, the values to_list() returns; to_columns() and
            # to_arrow() keep the documented milliseconds.
            columns = self._read_columns(25_000, None, micros=True)
            if columns is not None:
                # a vector column is already one numpy array per cell, None for a null
                return dataframe_from_columns(columns)

        return pd.DataFrame(self.to_list(convert_types=convert_types))

    def to_polars(
        self, batch_size: int = 25_000, columns: Optional[Sequence[str]] = None
    ):
        """
        Bulk-materialize all rows as a ``polars.DataFrame``, through Arrow.

        Reads :meth:`to_arrow` and hands the table to Polars, so nulls keep
        their type (a nullable integer stays an integer column) and a DECIMAL
        column is a Polars ``Decimal``. A column of ``float[]`` vectors is a
        Polars ``Array(Float32, n)`` (``to_arrow()`` types it as a fixed-size
        list of float32). Requires polars and pyarrow
        (``uv add 'arcadedb-jvm[polars]'``).

        Args:
            batch_size: Rows per bridge crossing, as in :meth:`to_arrow`.
            columns: Optional column names to pin, as in :meth:`to_columns`.

        Returns:
            polars DataFrame

        Raises:
            ImportError: If polars or pyarrow is not installed, or the bundled
                bridge jar is missing (the cases in which :meth:`to_arrow`
                returns None).

        Example:
            >>> df = db.query("sql", "SELECT FROM User").to_polars()
        """
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "polars is required for to_polars(). "
                "Install with: uv add 'arcadedb-jvm[polars]'"
            ) from exc
        table = self.to_arrow(batch_size=batch_size, columns=columns)
        if table is None:
            raise ImportError(
                "to_polars() reads through Arrow, which needs pyarrow and numpy "
                "and the bundled bridge jar. "
                "Install with: uv add 'arcadedb-jvm[polars]'"
            )
        return pl.from_arrow(table)

    def to_columns(
        self, batch_size: int = 25_000, columns: Optional[Sequence[str]] = None
    ):
        """
        Bulk-materialize all rows as columns: dict of column name -> numpy
        array (int64/float64/bool/datetime64[ms]) or Python list (strings and
        JSON-typed values). The fastest bulk path (~1.2x Java-native scans,
        measured), ideal for feeding pandas/numpy.

        Null handling follows pandas conventions: int/datetime columns with
        nulls are promoted to float64 with NaN / datetime64 NaT; string and
        JSON columns use None. A DECIMAL column is an object array of exact
        ``Decimal`` values (#115). The type of a column belongs to the whole
        result, not to where a batch ends: a batch that lacks the column or
        holds only nulls for it takes the type of the batches that have
        values. The arrays are writable. A vector column is a ``(count, dim)`` matrix
        and a null vector is a row of NaN in it (``to_dataframe()`` and ``to_arrow()`` keep
        it a null). An empty result is ``{}``, or, when
        ``columns`` is given, those columns as empty lists.

        The columns are the union of every row's property names, in order of
        first appearance, because a document is schemaless: a property the
        first row lacks is still a column (#113). Finding them costs one pass
        over each row's names; pass ``columns`` (a list of names) to read exactly
        those, as a projection would, and skip it. A row lacking one of them
        reads null, and properties not listed are left out.

        List and map values (and a column of mixed types) are decoded as
        ``to_list()`` decodes them: a ``date``, ``datetime`` or ``Decimal``
        inside a list or map stays one, and a map keeps its non-str keys. The
        ``datetime64[ms]`` rule above is for a column whose values are all
        datetimes.

        Returns None when numpy or the bridge jar is unavailable (callers
        fall back to row-based paths).
        """
        return self._read_columns(batch_size, columns, micros=False)

    def _read_columns(self, batch_size, columns, micros):
        """``to_columns()``; with ``micros`` a column of datetimes is a
        ``datetime64[us]`` array and a column holding DATE values is an object
        column of ``date`` objects, which is what ``to_dataframe()`` needs to
        return the values ``to_list()`` does."""
        try:
            __import__("numpy")  # availability check
        except ImportError:
            return None

        column_batcher = _bridge_class("ColumnBatcher")
        if column_batcher is None:
            return None

        # Every batch reports the union of its own rows' property names. The
        # result's columns are the union over the batches, in order of
        # first appearance (#113); a batch lacking one gets nulls for it when merged.
        names: List[str] = []
        batches: List[Tuple[int, Dict[str, Any]]] = []

        # An empty column spec makes Java derive the column set from the
        # rows of each batch; a pinned one is read as given. JSON, not
        # ";".join: a name may legally contain a semicolon, a quote or a
        # backslash.
        spec = json.dumps(list(columns)) if columns is not None else ""
        size = frame_rows(batch_size)
        if not self._readable():
            return empty_columns(columns)
        while True:
            try:
                java_batch = column_batcher.nextColumnBatch(
                    self._java_result_set, size, spec, micros
                )
            except JException as exc:
                raise _read_error(exc) from exc
            buf = memoryview(bytes(java_batch[0]))
            count, parts, _ = decode_columns_frame(
                buf, java_batch[1], _typed_json_values, micros
            )
            if count == 0:
                self._finish()
                break
            for name in parts:
                if name not in names:
                    names.append(name)
            batches.append((count, parts))

        return merge_columns(names, batches, columns)

    def to_arrow(
        self, batch_size: int = 25_000, columns: Optional[Sequence[str]] = None
    ):
        """
        Bulk-materialize all rows as a ``pyarrow.Table``.

        Reads the same columnar buffer as :meth:`to_columns` (one JSON header
        plus packed little-endian columns and a null bitmap each), so there is
        no extra work on the Java side. What differs is what happens to nulls
        and to strings.

        ``to_columns`` follows pandas conventions, which are lossy in two
        places: a nullable int64 column is promoted to float64 with NaN, which
        silently loses precision above 2**53 and loses the type; and a nullable
        boolean column falls back to a Python list. Arrow carries a validity
        bitmap alongside the values, so both keep their type here.

        Strings are also cheaper: the buffer already holds int32 offsets
        followed by a UTF-8 blob, which is exactly Arrow's string layout, so
        the column is wrapped rather than decoded one Python str at a time.

        The columns and the optional ``columns`` argument are as in
        :meth:`to_columns`. A column's type does not depend on ``batch_size``
        (#114), and a DECIMAL column is a decimal Arrow column (#115).

        Returns None when pyarrow, numpy, or the bridge jar is unavailable, so
        callers can fall back to :meth:`to_columns` the same way that method
        falls back to the row-based paths.
        """
        try:
            __import__("numpy")  # availability check
            __import__("pyarrow")  # availability check
        except ImportError:
            return None

        column_batcher = _bridge_class("ColumnBatcher")
        if column_batcher is None:
            return None

        # As in to_columns: every batch reports its own rows' union of property
        # names, the table's columns are the union over the batches in order of
        # first appearance (#113), and a batch lacking a column adds nulls.
        names: List[str] = []
        batches: List[Tuple[int, Dict[str, Any]]] = []

        spec = json.dumps(list(columns)) if columns is not None else ""
        size = frame_rows(batch_size)
        if not self._readable():
            return empty_arrow(columns)
        while True:
            try:
                java_batch = column_batcher.nextColumnBatch(
                    self._java_result_set, size, spec, False
                )
            except JException as exc:
                raise _read_error(exc) from exc
            buf = memoryview(bytes(java_batch[0]))
            count, chunks, _ = decode_arrow_frame(
                buf, java_batch[1], _typed_json_values
            )
            if count == 0:
                self._finish()
                break
            for name in chunks:
                if name not in names:
                    names.append(name)
            batches.append((count, chunks))

        return merge_arrow(names, batches, columns)

    def iter_chunks(
        self, size: int = 1000, convert_types: bool = True
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate results in chunks for memory-efficient processing.

        Useful for processing large result sets without loading
        everything into memory at once.

        Args:
            size: Chunk size (default: 1000)
            convert_types: Convert Java types to Python (default: True)

        Yields:
            List of dictionaries (up to size elements)

        Example:
            >>> results = db.query("sql", "SELECT FROM User")
            >>> for chunk in results.iter_chunks(size=1000):
            ...     process_batch(chunk)  # chunk is list of dicts
        """
        chunk = []
        for row in self.iter_dicts(convert_types=convert_types):
            chunk.append(row)
            if len(chunk) >= size:
                yield chunk
                chunk = []

        if chunk:  # Yield remaining items
            yield chunk

    def count(self) -> int:
        """
        Count the remaining results without building a list.

        Note:
            This consumes the remaining rows from the current result set.
            After calling `count()`, the iterator is exhausted.

        Returns:
            Number of remaining results

        Example:
            >>> count = db.query("sql", "SELECT FROM User").count()
            >>> print(f"Found {count} users")
        """
        row_access = _bridge_class("RowAccess")
        if row_access is not None and self._readable() and not self._closed:
            # hasNext() and next() per row, and the close, in ONE crossing
            try:
                count = int(row_access.countAndClose(self._java_result_set))
            except JException as exc:
                raise _read_error(exc) from exc
            self._drained()
            return count
        count = 0
        for _ in self:
            count += 1
        return count

    def first(self) -> Optional["Result"]:
        """
        Get first result or None if no results.

        Returns:
            First Result or None

        Example:
            >>> user = db.query("sql", "SELECT FROM User WHERE id = 1").first()
            >>> if user:
            ...     print(user.get("name"))
        """
        row_access = _bridge_class("RowAccess")
        if row_access is not None and self._readable() and not self._closed:
            # hasNext(), next(), and close() in ONE crossing instead of three
            try:
                java_row = row_access.firstAndClose(self._java_result_set)
            except JException as exc:
                self._closed = True  # firstAndClose closes on every path
                raise _read_error(exc) from exc
            self._closed = True
            if java_row is None:
                self._exhausted = True
                return None
            return Result(java_row, self._database)
        try:
            return next(iter(self))
        except StopIteration:
            return None
        finally:
            # the rest is never read: release the cursor now
            self.close()

    def one(self) -> "Result":
        """
        Get single result, raise error if not exactly one.

        Returns:
            The single Result

        Raises:
            ValueError: If zero or multiple results

        Example:
            >>> user = db.query("sql", "SELECT FROM User WHERE email = ?",
            ...                 "alice@example.com").one()
            >>> print(user.get("name"))
        """
        row_access = _bridge_class("RowAccess")
        try:
            if row_access is not None and self._readable() and not self._closed:
                # up to two rows and the close in ONE crossing instead of hasNext()/next() per row
                try:
                    rows = row_access.upToTwoAndClose(self._java_result_set)
                except JException as exc:
                    self._closed = True  # upToTwoAndClose closes on every path
                    raise _read_error(exc) from exc
                self._closed = True
                self._exhausted = len(rows) < 2
                if len(rows) == 0:
                    raise ValueError("Query returned no results")
                if len(rows) > 1:
                    raise ValueError("Query returned multiple results")
                return Result(rows[0], self._database)
            iterator = iter(self)
            try:
                result = next(iterator)
            except StopIteration as exc:
                raise ValueError("Query returned no results") from exc

            try:
                next(iterator)
                raise ValueError("Query returned multiple results")
            except StopIteration:
                return result
        finally:
            self.close()


class Result:
    """Wrapper for a single result from a query.

    A result keeps its ``Database`` alive. A row that is a record
    (``SELECT FROM T``) raises ArcadeDBError when read after that database was
    closed, like a record; a projection or a command result holds its own values
    and stays readable.
    """

    def __init__(self, java_result, database=None):
        self._java_result = java_result
        self._database = database  # strong reference, see the class docstring
        self._property_names_cache: Optional[Tuple[str, ...]] = None

    def _check_open(self) -> None:
        database = self._database
        if database is not None and database._closed and self._needs_database():
            raise ArcadeDBError(
                "Database is closed: a record row cannot be read after its "
                "database was closed. Read what you need before closing it."
            )

    def _needs_database(self) -> bool:
        """True for a row that is a record (``SELECT FROM T``): the engine
        loads its properties lazily from the open database. A projection or a
        command result holds its own values and stays readable, as it did
        before results held their database (an IMPORT DATABASE result is read
        after the import's database is closed, in example 16)."""
        try:
            return bool(self._java_result.isElement())
        except Exception:  # nosec B110 - an unreadable row is treated as lazy
            return True

    def _property_names_tuple(self) -> Tuple[str, ...]:
        self._check_open()
        if self._property_names_cache is None:
            reader = _bridge_class("ReaderCalls")
            if reader is not None:
                # the names as one JSON text instead of one JPype read per name
                self._property_names_cache = _loads_names(
                    reader.rowNames(self._java_result)
                )
            else:
                self._property_names_cache = tuple(
                    str(name) for name in self._java_result.getPropertyNames()
                )
        return self._property_names_cache

    def has_property(self, name: str) -> bool:
        """
        Check if a property exists in the result.

        Args:
            name: Property name

        Returns:
            True if property exists, False otherwise

        Example:
            >>> result = db.query("sql", "SELECT FROM User LIMIT 1").first()
            >>> if result.has_property("email"):
            ...     print(result.get("email"))
        """
        self._check_open()
        return self._java_result.hasProperty(name)

    def get(self, name: str, convert_types: bool = True) -> Any:
        """
        Get a property value from the result with automatic type conversion.

        Args:
            name: Property name

        Returns:
            Property value as a Python type, or None if not found

        Example:
            >>> result = db.query("sql", "SELECT FROM User WHERE id = 1").first()
            >>> age = result.get("age")  # Returns Python int
            >>> email = result.get("email")  # Returns Python str
            >>> phone = result.get("phone")  # Returns None if not found
            >>> phone = result.get("phone") or "unknown"  # Use default pattern
        """
        value = self.get_raw(name)
        if convert_types:
            return convert_java_to_python(value)
        return value

    def get_raw(self, name: str) -> Any:
        """
        Get a property value without Java-to-Python conversion.

        Args:
            name: Property name

        Returns:
            Raw Java-backed property value or None if not found
        """
        self._check_open()
        row_access = _bridge_class("RowAccess")
        if row_access is not None:
            # hasProperty() then getProperty() in ONE crossing
            return row_access.propertyOrNull(self._java_result, name)
        if not self._java_result.hasProperty(name):
            return None
        return self._java_result.getProperty(name)

    def get_rid(self) -> Optional[str]:
        """
        Get the Record ID (RID) if available.

        Returns:
            RID string (e.g., "#10:5") or None

        Example:
            >>> result = db.query("sql", "SELECT FROM User LIMIT 1").first()
            >>> print(result.get_rid())
            #10:5
        """
        identity = self._java_result.getIdentity()
        if identity.isPresent():
            return str(identity.get().toString())
        return None

    def get_vertex(self) -> Optional[Vertex]:
        """
        Get the underlying Vertex object if available.

        Returns:
            Vertex object or None
        """
        self._check_open()
        vertex = self._java_result.getVertex()
        if vertex.isPresent():
            return Vertex(vertex.get(), self._database)
        return None

    def get_edge(self) -> Optional[Edge]:
        """
        Get the underlying Edge object if available.

        Returns:
            Edge object or None
        """
        self._check_open()
        edge = self._java_result.getEdge()
        if edge.isPresent():
            return Edge(edge.get(), self._database)
        return None

    def get_element(self) -> Optional[Document]:
        """
        Get the underlying Element (Document, Vertex, or Edge) if available.

        Returns:
            Document, Vertex, or Edge object or None
        """
        self._check_open()
        element = self._java_result.getElement()
        if element.isPresent():
            return Document.wrap(element.get(), self._database)
        return None

    def get_property_names(self) -> List[str]:
        """
        Get all property names (alternative to property_names property).

        Returns:
            List of property names

        Example:
            >>> result = db.query("sql", "SELECT FROM User LIMIT 1").first()
            >>> names = result.get_property_names()
            >>> print(names)
            ['name', 'email', 'age', 'created_at']
        """
        return list(self._property_names_tuple())

    @property
    def property_names(self) -> List[str]:
        """
        Get all property names in this result.

        Returns:
            List of property names

        Example:
            >>> result = db.query("sql", "SELECT FROM User LIMIT 1").first()
            >>> print(result.property_names)
            ['name', 'email', 'age', 'created_at']
        """
        return list(self._property_names_tuple())

    def to_dict(self, convert_types: bool = True) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Args:
            convert_types: Convert Java types to Python (default: True)

        Returns:
            Dictionary with all properties

        Example:
            >>> result = db.query("sql", "SELECT FROM User LIMIT 1").first()
            >>> user_dict = result.to_dict()
            >>> print(user_dict)
            {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}
        """
        self._check_open()
        if convert_types:
            # One crossing for the whole row, and no JPype read of a name or a
            # value: Java writes the row as the typed JSON of TypedRows (the
            # transport of to_list()), so a row of plain values costs the call,
            # one array read and one string read. The values have the types
            # convert_java_to_python gives them.
            reader = _bridge_class("ReaderCalls")
            if reader is not None:
                row = _loads_typed_row(reader.rowJson(self._java_result))
                if self._property_names_cache is None:
                    self._property_names_cache = tuple(row)
                return row
            # Without ReaderCalls: one crossing for the whole row (names and
            # values) instead of one per property, then a JPype read per name and
            # value. The values are the engine's own objects, converted exactly
            # as below.
            row_access = _bridge_class("RowAccess")
            if row_access is not None:
                pair = row_access.namesAndValues(self._java_result)
                names = tuple(str(name) for name in pair[0])
                if self._property_names_cache is None:
                    self._property_names_cache = names
                return {
                    name: convert_java_to_python(value)
                    for name, value in zip(names, pair[1])
                }

        property_names = self._property_names_tuple()
        if not convert_types:
            return {
                name: self._java_result.getProperty(name) for name in property_names
            }

        return {
            name: convert_java_to_python(self._java_result.getProperty(name))
            for name in property_names
        }

    def to_json(self) -> str:
        """
        Convert result to JSON string.

        This is the engine's own JSON of the row. For a record row (``SELECT
        FROM T``) it is the record's round-trip form, which ``fromJSON`` reads
        back: it carries the ``@rid``, ``@type`` and ``@cat`` metadata and writes
        a top-level DATE as epoch DAYS (a DATETIME as epoch milliseconds, or as
        text when its column keeps microseconds or nanoseconds). A DATE inside a
        list or a map, and a DATE in a projection row (``SELECT a, b FROM T``),
        is written as the epoch milliseconds of its UTC midnight, the form
        ``ResultSet.to_json_list()`` uses everywhere; a DATETIME in a projection
        row is written as text in the schema's date-time format. The engine's
        JSON is not one encoding across row shapes, which is why
        ``to_json_list()`` does not delegate to it.

        Returns:
            JSON string representation

        Example:
            >>> result = db.query("sql", "SELECT FROM User LIMIT 1").first()
            >>> print(result.to_json())
            {"name": "Alice", "age": 30, "email": "alice@example.com"}
        """
        self._check_open()
        return str(self._java_result.toJSON())

    def __repr__(self) -> str:
        """String representation of the result."""
        try:
            rid = self.get_rid()
            property_names = self._property_names_tuple()
            names_preview = ", ".join(repr(name) for name in property_names[:3])
            if len(property_names) > 3:
                names_preview += ", ..."
            if rid is not None:
                return f"Result(rid={rid!r}, properties=[{names_preview}])"
            return f"Result(properties=[{names_preview}])"
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return f"Result({self._java_result})"
