"""Native backend coverage beyond the DSL core: GraphBatch, the async executor, database settings, import/export and Python
functions in SQL, over the ``arcadedb_n_gb*``, ``arcadedb_n_async``, ``arcadedb_n_dbset``, ``arcadedb_n_import``,
``arcadedb_n_export`` and ``arcadedb_n_cb`` entry points (``NativeExtras.java``).

Callbacks (async completions, error handlers, Python functions called from SQL) go through ONE ctypes function pointer that
``_ensure_upcall`` registers with the library: the engine thread (or an async writer thread) calls ``_dispatch(id, json)``,
ctypes takes the GIL for the call, and the reply is JSON (see the header of ``NativeExtras.java`` for GIL and exception rules).
"""

import ctypes
import itertools
import json
import threading
import weakref
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Optional

from ..callbacks import EventHandle as _BaseEventHandle
from ..callbacks import check_event, check_function
from ..exceptions import ArcadeDBError
from ..graph_batch import GraphBatch as _BaseGraphBatch
from . import interrupts, lib, not_yet
from .codec import _PLAIN, _default, _prep, loads

# -------------------------------------------------------------------------------------------------------- upcall

# (Java long id, char *json, void **out): the reply goes through *out, not through a return value (see _dispatch)
_UPCALL_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.c_longlong, ctypes.c_char_p, ctypes.c_void_p
)
_callbacks: dict = {}
_callback_ids = itertools.count(1)
_callbacks_lock = threading.Lock()
_upcall_ref = None


def _enc(value) -> str:
    """JSON text of a value with the tagged encoding of query parameters."""
    return json.dumps(_prep(value), default=_default)


def _dispatch(cid, raw, out):
    """Run callback ``cid`` with the message ``raw`` and write the JSON reply to ``*out`` (Java zeroed it before the call).

    The reply never travels as a return value. A C callback that returns without setting its return value hands the engine
    whatever the return register held, here the pointer of an earlier reply, which Java had already freed: an abort in
    ``free()``. That is what happens when a Ctrl-C is raised as this function is entered, before its ``try``. Through ``*out``
    the engine sees NULL then, fails the statement, and the error becomes a KeyboardInterrupt (``interrupts.convert``).
    The memory comes from the library (``arcadedb_n_alloc``): Java copies the text and frees it, so it must be nothing else.
    """
    try:
        fn = _callbacks.get(cid)
        message = loads(raw.decode("utf-8")) if raw else {}
        reply = fn(message) if fn is not None else None
        data = None if reply is None else _enc(reply).encode("utf-8")
    except BaseException as exc:  # nosec B110 - an exception must not cross the C frame
        interrupts.note(exc)
        data = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
    if data is None:
        return
    size = len(data) + 1
    pointer = lib.alloc(size)
    if pointer:
        ctypes.memmove(pointer, data + b"\0", size)
        ctypes.c_void_p.from_address(out).value = pointer


def _ensure_upcall() -> None:
    global _upcall_ref
    if _upcall_ref is not None:
        return
    lib.call_void(
        "arcadedb_n_set_upcall", ctypes.cast(_make_pointer(), ctypes.c_void_p)
    )


def _make_pointer():
    global _upcall_ref
    _upcall_ref = _UPCALL_TYPE(_dispatch)
    return _upcall_ref


def _forget(cid) -> None:
    """Drop a callback the engine no longer holds (an unregistered function, a removed listener, a replaced function)."""
    if cid:
        with _callbacks_lock:
            _callbacks.pop(cid, None)


def _register(fn: Callable) -> int:
    _ensure_upcall()
    with _callbacks_lock:
        cid = next(_callback_ids)
        _callbacks[cid] = fn
    return cid


def _error_from(message: dict) -> ArcadeDBError:
    reply = message.get("err") or {}
    exc = ArcadeDBError(f"{reply.get('cls')}: {reply.get('error')}")
    exc.native = reply
    exc.java_chain = reply.get("chain") or []
    return exc


def _rid_of(value) -> str:
    """The RID text of a record wrapper, a RID or its text."""
    get_rid = getattr(value, "get_rid", None)
    return get_rid() if get_rid is not None else str(value)


def _call(prefix: str, name: str, *args) -> dict:
    from .database import _call as call

    return call(prefix, name, *args)


# -------------------------------------------------------------------------------------------------------- GraphBatch


class NativeGraphBatch(_BaseGraphBatch):
    """``GraphBatch`` over the engine's batch builder in the image (same names and behaviour as the JPype wrapper)."""

    _BULK_CHUNK = 10_000  # rows per crossing: small enough that a chunk's text, rows and reply die in the young generation

    def __init__(self, database, handle: int):
        self._db = database
        self._gb = handle
        self._closed = False

    @classmethod
    def create(cls, database, **options) -> "NativeGraphBatch":
        wal_flush = options.get("wal_flush")
        if wal_flush is not None:
            options["wal_flush"] = cls._to_java_wal_flush_name(wal_flush)
        given = {k: v for k, v in options.items() if v is not None}
        try:
            reply = _call(
                "Failed to create GraphBatch",
                "arcadedb_n_gb_create",
                database._h,
                lib.b(_enc(given)),
            )
        except ArcadeDBError:
            raise
        return cls(database, reply["h"])

    @classmethod
    def _to_java_wal_flush_name(cls, mode: str) -> str:
        normalized = mode.lower()
        if normalized not in cls._VALID_WAL_FLUSH_MODES:
            raise ValueError(
                f"Invalid wal_flush mode: {mode}. Must be one of: {list(cls._VALID_WAL_FLUSH_MODES.keys())}"
            )
        return cls._VALID_WAL_FLUSH_MODES[normalized]

    def _op(self, prefix: str, op: str, **args) -> dict:
        return _call(prefix, "arcadedb_n_gb", self._gb, lib.b(op), lib.b(_enc(args)))

    def new_vertex(self, type_name: str):
        from .records import Vertex

        self._check_not_closed()
        return Vertex(self._db, type_name)

    def create_vertex(self, type_name: str, **properties):
        from .records import Vertex

        self._check_not_closed()
        db = self._db
        started = False
        try:
            if not db.is_transaction_active():
                started = True
                db.begin()
            reply = self._op(
                f"Failed to create batch vertex of type '{type_name}'",
                "create_vertex",
                type=type_name,
                props=self._to_java_varargs(properties),
            )
            if started:
                db.commit()
            return Vertex(db, type_name, reply["rid"], properties)
        except BaseException as e:
            if started:
                try:
                    if db.is_transaction_active():
                        db.rollback()
                except (
                    Exception
                ):  # nosec B110 - the original error is the one to report
                    pass
            if isinstance(e, ArcadeDBError) or not isinstance(e, Exception):
                raise
            raise ArcadeDBError(
                f"Failed to create batch vertex of type '{type_name}': {e}"
            ) from e

    def _to_java_varargs(self, properties: dict) -> dict:
        """The properties of one vertex as the library takes them (the step the JPype path converts to Java values in)."""
        if not isinstance(properties, dict):
            raise TypeError("vertex properties must be a dict")
        for key in properties:
            if type(key) is not str:
                raise TypeError(f"property names must be str, got {key!r}")
        return properties

    def _create_vertices_json_bulk(self, type_name: str, rows):
        """The bulk path (JSON rows in, RIDs out): the RIDs, or None for rows JSON would change (the caller falls back)."""
        from ..type_conversion import json_bulk_scalar_ok

        for row in rows:
            for key, value in (row or {}).items():
                if type(key) is not str or not json_bulk_scalar_ok(value):
                    return None
        return self._op(
            f"Failed to create batch vertices for type '{type_name}'",
            "create_vertices",
            type=type_name,
            rows=[r or {} for r in rows],
        )["rids"]

    def create_vertices(self, type_name: str, count_or_properties) -> list:
        self._check_not_closed()
        db = self._db
        owns_transaction = not db.is_transaction_active()
        prefix = f"Failed to create batch vertices for type '{type_name}'"
        try:
            chunk = self._BULK_CHUNK
            if isinstance(count_or_properties, int):
                rids = []
                for start in range(0, count_or_properties, chunk):
                    rids.extend(
                        self._op(
                            prefix,
                            "create_vertices",
                            type=type_name,
                            count=min(chunk, count_or_properties - start),
                        )["rids"]
                    )
                if not count_or_properties:
                    rids = self._op(prefix, "create_vertices", type=type_name, count=0)[
                        "rids"
                    ]
                return rids
            rows = list(count_or_properties)
            for row in rows:
                if row is not None and not isinstance(row, dict):
                    raise TypeError(
                        "create_vertices() property rows must be dicts or None"
                    )
            # chunked, as the JPype wrapper does: one crossing, one JSON text and one reply per chunk, so a 1M-vertex load
            # never holds the text, the parsed rows and the reply of all of it at once (the Serial GC copies what lives)
            rids = []
            for start in range(0, len(rows), chunk):
                rids.extend(
                    self._op(
                        prefix,
                        "create_vertices",
                        type=type_name,
                        rows=[r or {} for r in rows[start : start + chunk]],
                    )["rids"]
                )
            if not rows:
                rids = self._op(prefix, "create_vertices", type=type_name, rows=[])[
                    "rids"
                ]
            return rids
        except BaseException as e:
            if owns_transaction:
                try:
                    if db.is_transaction_active():
                        db.rollback()
                except Exception:  # nosec B110
                    pass
            if isinstance(e, (TypeError, ArcadeDBError)) or not isinstance(
                e, Exception
            ):
                raise
            raise ArcadeDBError(f"{prefix}: {e}") from e

    @staticmethod
    def _rid_text(value) -> str:
        rid = getattr(value, "get_rid", None)
        if rid is not None:
            return rid()
        return str(value)

    def new_edge(
        self,
        source_vertex_or_rid,
        edge_type: str,
        destination_vertex_or_rid,
        **properties,
    ):
        self._check_not_closed()
        self._op(
            f"Failed to buffer batch edge '{edge_type}'",
            "new_edge",
            src=self._rid_text(source_vertex_or_rid),
            type=edge_type,
            dst=self._rid_text(destination_vertex_or_rid),
            props=properties,
        )
        return self

    def new_edges(self, source_rids, edge_type: str, destination_rids, properties=None):
        self._check_not_closed()
        source_rids = [self._rid_text(r) for r in source_rids]
        destination_rids = [self._rid_text(r) for r in destination_rids]
        if properties is not None:
            properties = list(properties)
        lengths = [len(source_rids), len(destination_rids)]
        if properties is not None:
            lengths.append(len(properties))
        if len(set(lengths)) != 1:
            names = ["source_rids", "destination_rids", "properties"]
            got = ", ".join(f"{n} {k}" for n, k in zip(names, lengths))
            raise ArcadeDBError(
                f"Failed to buffer batch edges '{edge_type}': source_rids, destination_rids and properties must have "
                f"the same length ({got})"
            )
        for p in properties or ():
            for key in p or ():
                if not isinstance(key, str):
                    raise ArcadeDBError(
                        f"Failed to buffer batch edges '{edge_type}': property names must be str, got {key!r}"
                    )
        chunk = self._BULK_CHUNK
        if properties is None:
            # no properties: the RIDs of a chunk go over as two ';'-joined texts (as EdgeBatcher.newEdgesJoined takes them on the
            # JPype path) and are read straight into RIDs, not as two JSON lists of Strings
            for start in range(0, len(source_rids), chunk):
                self._op(
                    f"Failed to buffer batch edges '{edge_type}'",
                    "new_edges",
                    src=";".join(source_rids[start : start + chunk]),
                    type=edge_type,
                    dst=";".join(destination_rids[start : start + chunk]),
                    props=None,
                )
            return self
        for start in range(0, len(source_rids), chunk):
            self._op(
                f"Failed to buffer batch edges '{edge_type}'",
                "new_edges",
                src=source_rids[start : start + chunk],
                type=edge_type,
                dst=destination_rids[start : start + chunk],
                props=(
                    None
                    if properties is None
                    else [p or {} for p in properties[start : start + chunk]]
                ),
            )
        return self

    def flush(self):
        self._check_not_closed()
        self._op("Failed to flush GraphBatch", "flush")
        return self

    def close(self):
        if self._closed:
            return
        refused = False
        try:
            self._op("Failed to close GraphBatch", "close")
        except ArcadeDBError as e:
            # an IllegalStateException while the caller has a transaction open: the batch is still open with its work pending
            refused = (getattr(e, "native", None) or {}).get(
                "cls"
            ) == "java.lang.IllegalStateException" and bool(
                self._db.is_transaction_active()
            )
            raise
        finally:
            if not refused:
                self._closed = True

    def get_total_edges_created(self) -> int:
        self._check_not_closed()
        return int(self._op("Failed to read GraphBatch", "stats")["total_edges"])

    def get_buffered_edge_count(self) -> int:
        self._check_not_closed()
        return int(self._op("Failed to read GraphBatch", "stats")["buffered"])

    def get_deferred_incoming_edge_count(self) -> int:
        self._check_not_closed()
        return int(self._op("Failed to read GraphBatch", "stats")["deferred"])


# -------------------------------------------------------------------------------------------------------- async


class NativeAsyncExecutor:
    """The database's async executor (one per database) with the JPype wrapper's names; callbacks run on the engine's writer
    threads and take the GIL, as the JPype proxies do."""

    def __init__(self, database):
        self._db = database
        self._closed = False
        self._owner = database
        self._cb_ids: list = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def is_closed(self) -> bool:
        return self._closed

    def _op(self, op: str, **args) -> dict:
        return _call(
            "Async executor failed",
            "arcadedb_n_async",
            self._db._h,
            lib.b(op),
            lib.b(_enc(args)),
        )

    def _settings(self) -> dict:
        return self._op("get")

    def _set(self, key, value):
        self._op("set", key=key, value=value)
        return self

    # configuration
    def set_parallel_level(self, level: int):
        if level < 1:
            raise ValueError("parallel_level must be at least 1")
        return self._set("parallel", level)

    def set_commit_every(self, count: int):
        if count < 1:
            raise ValueError(
                f"commit_every must be >= 1 (got {count}); the async executor always commits in batches"
            )
        return self._set("commit_every", count)

    def set_transaction_use_wal(self, use_wal: bool):
        return self._set("wal", bool(use_wal))

    def set_transaction_sync(self, sync_mode: str):
        valid = {"no": "NO", "yes_nometadata": "YES_NOMETADATA", "yes_full": "YES_FULL"}
        if sync_mode not in valid:
            raise ValueError(f"sync_mode must be one of {set(valid)}, got {sync_mode}")
        return self._set("sync", valid[sync_mode])

    def set_back_pressure(self, percentage: int):
        if percentage < 0 or percentage > 100:
            raise ValueError("back_pressure must be between 0 and 100")
        return self._set("back_pressure", percentage)

    def get_parallel_level(self) -> int:
        return int(self._settings()["parallel"])

    def get_back_pressure(self) -> int:
        return int(self._settings()["back_pressure"])

    def get_commit_every(self) -> int:
        return int(self._settings()["commit_every"])

    def is_transaction_use_wal(self) -> bool:
        return bool(self._settings()["wal"])

    def get_transaction_sync(self) -> str:
        return self._settings()["sync"].lower()

    def get_thread_count(self) -> int:
        return int(self._settings()["threads"])

    def is_processing(self) -> bool:
        return bool(self._settings()["processing"])

    def is_pending(self) -> bool:
        return self.is_processing()

    def kill(self):
        self._op("kill")

    # work
    def _record_callback(self, callback, kind="d"):
        from .records import _wrap_dict

        def handler(message):
            callback(_wrap_dict(self._db, message["rec"], kind))

        return _register(handler)

    def _error_callback(self, callback):
        return _register(lambda message: callback(_error_from(message)))

    def create_record(self, document, callback=None, error_callback=None):
        kind = "v" if hasattr(document, "get_out_edges") else "d"
        cb = self._record_callback(callback, kind) if callback is not None else 0
        err = self._error_callback(error_callback) if error_callback is not None else 0
        self._op(
            "create_record",
            type=document._type,
            kind=kind,
            props=document._props,
            cb=cb,
            err=err,
        )

    def _statement(self, op, language, text, callback, args, params, error_callback):
        from .database import NativeResultSet

        args = tuple(args or ())
        if args and params:
            raise ValueError("Use either positional args or named params, not both")
        cb = 0
        if callback is not None:

            def on_result(message):
                # the JPype wrapper calls the callback once per row of the completed statement
                for row in NativeResultSet(self._db, message["reply"]):
                    callback(row)

            cb = _register(on_result)
        err = self._error_callback(error_callback) if error_callback is not None else 0
        if params:
            values = dict(params)
        elif args:
            values = (
                dict(args[0])
                if len(args) == 1 and isinstance(args[0], Mapping)
                else list(args)
            )
        else:
            values = None
        self._op(op, lang=language, text=text, params=values, cb=cb, err=err)

    def query(
        self, language, query_text, callback, args=None, error_callback=None, **params
    ):
        self._statement(
            "query", language, query_text, callback, args, params, error_callback
        )

    def command(
        self,
        language,
        command_text,
        callback=None,
        args=None,
        error_callback=None,
        **params,
    ):
        self._statement(
            "command", language, command_text, callback, args, params, error_callback
        )

    def append_samples(
        self, type_name, timestamps, *column_values, primitive: bool = False
    ):
        def plain(values):
            kind = getattr(getattr(values, "dtype", None), "kind", None)
            if (
                kind == "b"
            ):  # a numpy bool array is a 0/1 numeric column (a list of bools is not: the engine refuses it)
                values = values.astype("int64")
            return values.tolist() if hasattr(values, "tolist") else list(values)

        stamps = [int(t) for t in plain(timestamps)]
        if primitive:
            # the engine's primitive TimeSeriesBatch is a JVM-boundary optimisation (no boxing per value); here the rows cross
            # as one JSON text either way, so the columns take the same path, with the primitive path's length check
            for index, values in enumerate(column_values):
                if len(values) != len(stamps):
                    raise ValueError(
                        f"column {index} has {len(values)} values for {len(stamps)} timestamps"
                    )
        self._op(
            "append_samples",
            type=type_name,
            ts=stamps,
            cols=[plain(c) for c in column_values],
        )

    def _edge_callback(self, callback):
        """The handler of an edge-created event: ``callback(edge, created_source, created_destination)``."""
        from .records import _wrap_dict

        def handler(message):
            callback(
                _wrap_dict(self._db, message["edge"], "e"),
                bool(message["src"]),
                bool(message["dst"]),
            )

        return _register(handler)

    def new_edge(
        self,
        source_vertex,
        edge_type: str,
        destination_vertex_or_rid,
        light: bool = False,
        callback=None,
        **properties,
    ):
        cb = self._edge_callback(callback) if callback is not None else 0
        self._op(
            "new_edge",
            src=_rid_of(source_vertex),
            type=edge_type,
            dst=_rid_of(destination_vertex_or_rid),
            light=bool(light),
            props=properties,
            cb=cb,
        )

    def new_edge_by_keys(
        self,
        source_vertex_type: str,
        source_key_names,
        source_key_values,
        destination_vertex_type: str,
        destination_key_names,
        destination_key_values,
        create_vertex_if_not_exist: bool,
        edge_type: str,
        bidirectional: bool,
        light: bool,
        callback=None,
        **properties,
    ):
        def listed(names, values, what):
            if isinstance(names, str):
                names = [names]
                values = list(values) if isinstance(values, (list, tuple)) else [values]
            names, values = list(names), list(values)
            if len(names) != len(values):
                raise ValueError(
                    f"{what}_key_names and {what}_key_values must have same size"
                )
            return names, values

        src_names, src_values = listed(source_key_names, source_key_values, "source")
        dst_names, dst_values = listed(
            destination_key_names, destination_key_values, "destination"
        )
        cb = self._edge_callback(callback) if callback is not None else 0
        self._op(
            "new_edge_by_keys",
            src_type=source_vertex_type,
            src_names=src_names,
            src_values=src_values,
            dst_type=destination_vertex_type,
            dst_names=dst_names,
            dst_values=dst_values,
            create=bool(create_vertex_if_not_exist),
            type=edge_type,
            bidirectional=bool(bidirectional),
            light=bool(light),
            props=properties,
            cb=cb,
        )

    def scan_type(
        self,
        type_name: str,
        callback,
        polymorphic: bool = True,
        error_callback=None,
    ):
        """Scan every record of a type on the writer threads: ``callback(record) -> bool`` (False stops the scan; None goes on)."""
        from .records import _wrap_dict

        def on_record(message):
            result = callback(_wrap_dict(self._db, message["rec"], "d"))
            return {"v": True if result is None else bool(result)}

        cb = _register(on_record)
        err = 0
        if error_callback is not None:

            def on_error(message):
                result = error_callback(
                    message.get("rid"), _error_from({"err": message.get("err")})
                )
                return {"v": True if result is None else bool(result)}

            err = _register(on_error)
        self._op(
            "scan_type", type=type_name, polymorphic=bool(polymorphic), cb=cb, err=err
        )

    def transaction(
        self,
        tx_block: Callable[[], None],
        retries: Optional[int] = None,
        ok_callback: Optional[Callable[[], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        slot: Optional[int] = None,
    ):
        """Run ``tx_block`` on an async writer inside a transaction (the engine begins it and commits or rolls it back, retrying a conflict ``retries`` times)."""

        def run(message):
            tx_block()

        args = {"cb": _register(run)}
        if retries is not None:
            args["retries"] = int(retries)
        if ok_callback is not None:
            args["ok"] = _register(lambda message: ok_callback())
        if error_callback is not None:
            args["err"] = self._error_callback(error_callback)
        if slot is not None:
            args["slot"] = int(slot)
        self._op("transaction", **args)

    def wait_completion(self, timeout_ms: Optional[int] = None):
        if timeout_ms is None:
            self._op("wait")
        elif timeout_ms < 0:
            raise ValueError(f"timeout_ms must be None or >= 0, got {timeout_ms}")
        elif timeout_ms == 0:
            if self.is_processing():
                raise TimeoutError("Async operations have not completed (0ms poll)")
        elif not self._op("wait", timeout_ms=timeout_ms)["done"]:
            raise TimeoutError(
                f"Async operations did not complete within {timeout_ms}ms"
            )

    def on_ok(self, callback: Callable[[], None]):
        self._op("on_ok", cb=_register(lambda message: callback()))
        return self

    def on_error(self, callback: Callable[[Exception], None]):
        self._op("on_error", cb=self._error_callback(callback))
        return self

    def close(self):
        if self._closed:
            return
        try:
            self._op("close")
        finally:
            self._closed = True
            owner, self._owner = self._owner, None
            if owner is not None:
                owner._discard_async_executor(self)


def insert_parallel(database, type_name: str, rows) -> dict:
    """Hand ``rows`` (a list of dicts) to the async writers; returns the failure token for :func:`failures`."""
    return _call(
        "Failed to bulk-insert",
        "arcadedb_n_async",
        database._h,
        b"insert_parallel",
        lib.b(_enc({"type": type_name, "rows": rows})),
    )


def failures(database, token: int) -> dict:
    return _call(
        "Failed to read the parallel writers",
        "arcadedb_n_async",
        database._h,
        b"failures",
        lib.b(_enc({"fid": token})),
    )


# -------------------------------------------------------------------------------------------------------- settings


def db_set(database, op: str, value: str = "") -> dict:
    return _call(
        f"Failed to set {op}", "arcadedb_n_dbset", database._h, lib.b(op), lib.b(value)
    )


# -------------------------------------------------------------------------------------------------------- import / export


def import_documents(database, settings: dict) -> Optional[dict]:
    from ..importer import _stringify_import_setting

    text = json.dumps({k: _stringify_import_setting(v) for k, v in settings.items()})
    return _call("Failed to import", "arcadedb_n_import", database._h, lib.b(text))[
        "stats"
    ]


def run_import(database, settings: dict, parallel, commit_every, wal) -> Optional[dict]:
    """The importer with the async executor's settings overridden for the run and restored after it (as the JPype path does)."""
    executor = database._async()
    saved = executor._settings()
    saved_ryw = database.is_read_your_writes()
    try:
        database.set_read_your_writes(False)
        if parallel is not None:
            executor.set_parallel_level(parallel)
        if commit_every is not None:
            executor.set_commit_every(commit_every)
        if wal is not None:
            executor.set_transaction_use_wal(wal)
        return import_documents(database, settings)
    finally:
        try:
            executor.wait_completion()
        finally:
            database.set_read_your_writes(saved_ryw)
            executor._set("parallel", saved["parallel"])
            executor._set("commit_every", saved["commit_every"])
            executor._set("wal", saved["wal"])


def export_database(
    database, path: str, fmt: str, overwrite: bool, settings: dict
) -> dict:
    return _call(
        "Database export failed",
        "arcadedb_n_export",
        database._h,
        lib.b(path),
        lib.b(fmt),
        1 if overwrite else 0,
        lib.b(json.dumps(settings)),
    )["stats"]


# -------------------------------------------------------------------------------------------------------- Python functions in SQL and record events

_MAX_LONG = 2**63


def _check_result(value) -> None:
    """A function result the engine can carry: an int past a Java long is an OverflowError, as on the JPype path."""
    t = type(value)
    if t is int:
        if not -_MAX_LONG <= value < _MAX_LONG:
            raise OverflowError("Python int too large to convert to Java long")
    elif t is list or t is tuple:
        for item in value:
            _check_result(item)
    elif t is dict:
        for item in value.values():
            _check_result(item)


def _record_or_value(database, value):
    """A record the engine sent as a dict with ``@rid``, ``@type`` and ``@cat`` becomes a Document, Vertex or Edge."""
    if type(value) is dict and "@type" in value and "@cat" in value:
        from .records import _wrap_dict

        return _wrap_dict(database, value)
    return value


def _function_handler(fn: Callable, database_ref) -> Callable:
    def handler(message):
        database = database_ref()
        result = fn(*[_record_or_value(database, a) for a in message.get("args", [])])
        _check_result(result)
        return {"v": result}

    return handler


_veto = threading.local()


def veto_count() -> int:
    """How many ``before_*`` callbacks vetoed a change on this thread so far (``save`` tells a veto from a failure by it)."""
    return getattr(_veto, "count", 0)


def _event_handler(event: str, fn: Callable, database_ref) -> Callable:
    is_before = event.startswith("before_")

    def handler(message):
        from .records import _wrap_dict

        result = fn(_wrap_dict(database_ref(), message["rec"]))
        vetoed = is_before and result is False
        if vetoed:
            _veto.count = veto_count() + 1
        return {"v": not vetoed}

    return handler


class NativeEventHandle(_BaseEventHandle):
    """A registered record event callback; ``remove()`` stops it (also a context manager, as on the JPype backend)."""

    def __init__(self, registrations, event, type_name, handle, cid):
        super().__init__(None, event, type_name, handle, None)
        self._registrations = registrations
        self._cid = cid

    def remove(self) -> None:
        if not self._active:
            return
        self._active = False
        self._registrations._events.discard(self)
        try:
            _call(
                f"Failed to remove '{self.event}' callback",
                "arcadedb_n_cb",
                lib.b("ev_remove"),
                lib.b(json.dumps({"h": self._java_handle})),
            )
        finally:
            _forget(self._cid)


class Registrations:
    """The Python functions and event callbacks one ``NativeDatabase`` registered, to remove them when it closes.

    The engine keeps functions for the whole process, so each database removes its own; the checks and the replies are those
    of the JPype backend (``arcadedb.callbacks``), the engine side is ``arcadedb_n_cb`` in ``NativeExtras.java``.
    """

    def __init__(self, database):
        # weak: the callbacks hold the database only for wrapping records, and the registry must not keep it alive
        self._database_ref = weakref.ref(database)
        self._h = database._h
        self._functions: dict = {}  # lower-case name -> (engine handle, callback id)
        self._events: set = set()

    def _cb(self, prefix: str, op: str, **args) -> dict:
        reply = _call(prefix, "arcadedb_n_cb", lib.b(op), lib.b(json.dumps(args)))
        if "refused" in reply:
            raise ArcadeDBError(reply["refused"])
        return reply

    def register_function(
        self, name, fn, min_args, max_args, deterministic, replace
    ) -> None:
        min_args, max_args = check_function(name, fn, min_args, max_args)
        cid = _register(_function_handler(fn, self._database_ref))
        try:
            reply = self._cb(
                f"Failed to register function '{name}'",
                "fn_register",
                name=name,
                cb=cid,
                min=min_args,
                max=min(max_args, 2**31 - 1),
                det=bool(deterministic),
                replace=bool(replace),
            )
        except BaseException:
            _forget(cid)
            raise
        _forget(reply.get("dropped"))  # the function this one replaced
        self._functions[name.lower()] = (reply["h"], cid)

    def unregister_function(self, name: str) -> None:
        entry = self._functions.pop(str(name).lower(), None)
        if entry is None:
            raise ArcadeDBError(
                f"No Python function '{name}' is registered through this database"
            )
        try:
            self._cb(f"Failed to unregister '{name}'", "fn_unregister", h=entry[0])
        finally:
            _forget(entry[1])

    def on(self, event: str, fn: Callable, type_name: Optional[str]):
        check_event(event, fn)
        cid = _register(_event_handler(event, fn, self._database_ref))
        try:
            reply = self._cb(
                f"Failed to register '{event}' callback",
                "ev_register",
                db=self._h,
                type=type_name,
                event=event,
                cb=cid,
            )
        except BaseException:
            _forget(cid)
            raise
        handle = NativeEventHandle(self, event, type_name, reply["h"], cid)
        self._events.add(handle)
        return handle

    def clear(self) -> None:
        """Remove everything this database registered (called when it closes); a failure here must not stop the close."""
        functions, self._functions = self._functions, {}
        events, self._events = self._events, set()
        for handle, cid in functions.values():
            try:
                self._cb("", "fn_unregister", h=handle)
            except Exception:  # nosec B110 - the library may be shutting down
                pass
            _forget(cid)
        for event in events:
            try:
                event.remove()
            except Exception:  # nosec B110
                pass
