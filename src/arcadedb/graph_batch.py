"""
ArcadeDB Python Bindings - GraphBatch wrapper

Pythonic wrapper for ArcadeDB's high-throughput graph ingest API.
"""

from typing import Any, Iterable, Optional

import jpype

from ._logging import get_logger, log_swallowed_exception
from .exceptions import ArcadeDBError
from .graph import Document, Vertex
from .type_conversion import (
    convert_python_to_java,
    declared_float_vectors,
    declared_scalars_ok,
    json_bulk_dumps,
    json_bulk_dumps_tagged,
    json_bulk_scalar_ok,
    json_bulk_shape_ok,
    screen_vertex_rows,
    split_float_vectors,
    vector_width,
)

_LOGGER = get_logger(__name__)


class GraphBatch:
    """Wrapper for Java GraphBatch with builder-backed configuration.

    create_vertices(), flush() and close(), and new_edge()/new_edges() when the
    buffer fills, manage their own transactions. From engine 26.10.1 they refuse
    to run inside a transaction you opened (an ArcadeDBError) and leave it
    untouched; on 26.9.1 and earlier they commit it (ArcadeData/arcadedb#9242).
    Call them outside your own transactions either way. Commit sizes and what
    an interrupted batch leaves behind are in
    docs/modules/ROOT/pages/how-to/bulk-loads.adoc.
    """

    _VALID_WAL_FLUSH_MODES = {
        "no": "NO",
        "yes_nometadata": "YES_NOMETADATA",
        "yes_full": "YES_FULL",
    }

    def __init__(
        self,
        java_database,
        java_graph_batch,
    ):
        self._java_db = java_database
        self._java_graph_batch = java_graph_batch
        self._closed = False

    @classmethod
    def create(
        cls,
        java_database,
        *,
        batch_size: Optional[int] = None,
        expected_edge_count: Optional[int] = None,
        edge_list_initial_size: Optional[int] = None,
        light_edges: Optional[bool] = None,
        bidirectional: Optional[bool] = None,
        commit_every: Optional[int] = None,
        use_wal: Optional[bool] = None,
        wal_flush: Optional[str] = None,
        pre_allocate_edge_chunks: Optional[bool] = None,
        parallel_flush: Optional[bool] = None,
        commit_retries: Optional[int] = None,
        commit_retry_delay_ms: Optional[int] = None,
        chunk_cache_capacity: Optional[int] = None,
        max_deferred_incoming_edges: Optional[int] = None,
    ) -> "GraphBatch":
        """Build a GraphBatch from the Java builder API."""
        try:
            builder = java_database.batch()

            if batch_size is not None:
                builder = builder.withBatchSize(batch_size)
            if expected_edge_count is not None:
                builder = builder.withExpectedEdgeCount(expected_edge_count)
            if edge_list_initial_size is not None:
                builder = builder.withEdgeListInitialSize(edge_list_initial_size)
            if light_edges is not None:
                builder = builder.withLightEdges(light_edges)
            if bidirectional is not None:
                builder = builder.withBidirectional(bidirectional)
            if commit_every is not None:
                builder = builder.withCommitEvery(commit_every)
            if use_wal is not None:
                builder = builder.withWAL(use_wal)
            if wal_flush is not None:
                builder = builder.withWALFlush(cls._to_java_wal_flush(wal_flush))
            if pre_allocate_edge_chunks is not None:
                builder = builder.withPreAllocateEdgeChunks(pre_allocate_edge_chunks)
            if parallel_flush is not None:
                builder = builder.withParallelFlush(parallel_flush)
            if commit_retries is not None:
                builder = builder.withCommitRetries(commit_retries)
            if commit_retry_delay_ms is not None:
                # Java takes a long here; JPype picks the overload from the
                # Python int, but be explicit so a large delay cannot land on
                # an int-typed candidate.
                builder = builder.withCommitRetryDelay(
                    jpype.JLong(commit_retry_delay_ms)
                )
            if chunk_cache_capacity is not None:
                builder = builder.withChunkCacheCapacity(chunk_cache_capacity)
            if max_deferred_incoming_edges is not None:
                builder = builder.withMaxDeferredIncomingEdges(
                    max_deferred_incoming_edges
                )

            return cls(java_database, builder.build())
        except ValueError:
            raise
        except Exception as e:
            raise ArcadeDBError(f"Failed to create GraphBatch: {e}") from e

    def __enter__(self) -> "GraphBatch":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def new_vertex(self, type_name: str) -> Vertex:
        """Create an unsaved vertex through the batch API."""
        self._check_not_closed()
        try:
            return Vertex(self._java_graph_batch.newVertex(type_name))
        except Exception as e:
            raise ArcadeDBError(
                f"Failed to create new batch vertex of type '{type_name}': {e}"
            ) from e

    def create_vertex(self, type_name: str, **properties) -> Vertex:
        """Create and save a single vertex, optionally with properties."""
        self._check_not_closed()
        started_transaction = False
        try:
            if not self._java_db.isTransactionActive():
                # Flag first: an interrupt right after begin() must still roll
                # back; the handler re-checks isTransactionActive() anyway.
                started_transaction = True
                self._java_db.begin()

            if properties:
                java_vertex = self._java_graph_batch.createVertex(
                    type_name, *self._to_java_varargs(properties)
                )
            else:
                java_vertex = self._java_graph_batch.createVertex(type_name)

            if started_transaction:
                self._java_db.commit()

            return Vertex(java_vertex)
        except BaseException as e:
            # BaseException, not Exception: a KeyboardInterrupt/SystemExit
            # between the begin() above and the commit() must not leak the
            # transaction this method started (#7882). Only ordinary failures
            # are translated to ArcadeDBError; the rest propagate unchanged.
            if started_transaction:
                try:
                    if self._java_db.isTransactionActive():
                        self._java_db.rollback()
                except Exception:
                    log_swallowed_exception(_LOGGER, "during batch vertex rollback")
            if not isinstance(e, Exception):
                raise
            raise ArcadeDBError(
                f"Failed to create batch vertex of type '{type_name}': {e}"
            ) from e

    def create_vertices(
        self,
        type_name: str,
        count_or_properties: int | Iterable[Optional[dict[str, Any]]],
    ) -> list[str]:
        """
        Create multiple vertices efficiently and return their RIDs.

        Call it outside your own transactions: from engine 26.10.1 it raises
        ArcadeDBError inside one and leaves it untouched; on 26.9.1 and earlier
        it commits the transaction open on this thread, yours included
        (ArcadeData/arcadedb#9242).

        Args:
            type_name: Vertex type name.
            count_or_properties: Either an integer vertex count or an iterable of
                per-vertex property dictionaries. Use `None` or `{}` for vertices
                without properties when passing an iterable. A property declared
                ``ARRAY_OF_FLOATS`` whose value in every row is a list, a numpy
                array or a Java ``float[]`` or ``double[]`` of numbers of one length
                crosses as one binary float32 matrix instead of JSON text, and so
                does any other property whose value in every row is a float32
                numpy array or a Java ``float[]`` (``to_java_float_array``) of one
                length: one buffer per property and chunk, stored as a ``float[]``.
                A list (not one of only numbers), tuple, dict, date-time, date,
                Decimal or bytes value also stays on the bulk path: it crosses
                inside the JSON as a tagged value and is stored as the
                per-element path stores it. A list of only numbers keeps the
                property-matrix path, which crosses it as one double[].

        Large inputs cross in chunks of 100,000 rows on this build. On the native build (the ``arcadedb`` package) the chunk is the
        batch's ``commit_every``, 1000 unless ``graph_batch()`` got another value, and each chunk is its own transaction. Chunks that
        already went in stay stored if a later chunk fails.
        """
        self._check_not_closed()
        # From engine 26.10.1 the engine refuses to run inside a transaction the
        # caller opened and leaves it untouched, so the rollback below is only
        # ever for a transaction this call began. On 26.9.1 and earlier it
        # joined the caller's transaction, committed it with the vertices, and
        # on a retryable error rolled it back before retrying in one of its own
        # (ArcadeData/arcadedb#9242).
        owns_transaction = not self._java_db.isTransactionActive()
        try:
            if isinstance(count_or_properties, int):
                java_rids = self._java_graph_batch.createVertices(
                    type_name, count_or_properties
                )
                return [str(rid) for rid in java_rids]

            rows = list(count_or_properties)
            rids = self._create_vertices_json_bulk(type_name, rows)
            if rids is not None:
                return rids

            java_rids = self._java_graph_batch.createVertices(
                type_name,
                self._to_java_property_matrix(rows),
            )
            return [str(rid) for rid in java_rids]
        except BaseException as e:
            # As in create_vertex: a failure other than a retryable one (a
            # duplicate key, a KeyboardInterrupt) leaves the engine's
            # transaction open, so a later write outside any transaction is
            # accepted and lost at close (#121).
            if owns_transaction:
                try:
                    if self._java_db.isTransactionActive():
                        self._java_db.rollback()
                except Exception:
                    log_swallowed_exception(_LOGGER, "during batch vertices rollback")
            if not isinstance(e, Exception):
                raise
            raise ArcadeDBError(
                f"Failed to create batch vertices for type '{type_name}': {e}"
            ) from e

    _BULK_CHUNK = 100_000  # rows per boundary crossing in bulk paths
    # Rows with vector columns cross in blocks of about this many floats (1 MiB of float32) over all their vector columns, at least
    # _VECTOR_BLOCK_MIN_ROWS rows: 10,000 rows of 96 floats cross in 4 blocks, not in one. The engine builds the vertices of a crossing
    # from the JSON text, one float[] buffer and one copy of every row's vector, and a smaller crossing keeps less of that alive when
    # a collection runs (the peak RSS of a load of Java float[] vectors was 1.1 to 1.3 times the 26.11.1 wheel's, which never copies them).
    _VECTOR_BLOCK_FLOATS = 262_144
    _VECTOR_BLOCK_MIN_ROWS = 512

    def _create_vertices_json_bulk(self, type_name, rows):
        """Bulk path: rows cross as one JSON string, RIDs return as one joined
        string (two bulk copies instead of per-value marshaling, measured
        ~2.4x). The float vectors of ARRAY_OF_FLOATS properties, and the float32
        arrays and Java float[] of any property, cross as one binary matrix
        beside the JSON instead of as text (split_float_vectors).
        A list (not one of only numbers), tuple, dict, date-time, date, Decimal
        or bytes value crosses inside the JSON as the tagged value
        json_bulk_dumps_tagged writes and VertexBatcher decodes (the stored
        value is what the property-matrix path stores). An input whose values
        are all scalars is written by json_bulk_dumps. The values are checked
        by their types and columns, in C (type_conversion.screen_vertex_rows),
        not in a loop over the rows: a scalar the text would change (a NaN, an
        integer beyond 64 bits, a lone surrogate, a subnormal) anywhere in the
        input sends the whole input through the property-matrix path, and
        rows of exact scalars only are then written with no further check.
        Other rows are checked in the one json.dumps pass of each chunk (see
        type_conversion._dumps_once); a chunk holding a date-time inside a
        list, or a vector column the matrix cannot carry, goes through the
        property-matrix path on its own. Returns None when unavailable/unsuitable
        (a property name that is not a str, a type the text does not carry) so
        the caller falls back to the property-matrix path for the whole input."""
        try:
            vertex_batcher = jpype.JClass("com.arcadedb.python.VertexBatcher")
            declared = set(declared_float_vectors(self._java_db, type_name))
        except Exception:
            return None

        verdict = screen_vertex_rows(rows, declared)
        if verdict is None:
            # a property name that is not a str (json.dumps would write 1 as "1" where the matrix path refuses it), or a type the
            # tagged JSON does not carry (a set, a numpy or pandas value, any subclass, a list of only numbers: the matrix path
            # converts those)
            return None
        tagged, vetted = verdict

        if vetted:
            # every value is an exact scalar the text carries unchanged (screened in C above): the chunks are written without a check
            def dumps(chunk_rows):
                return json_bulk_dumps(chunk_rows, vetted=True)

        else:
            dumps = json_bulk_dumps_tagged if tagged else json_bulk_dumps
        # chunked so huge ingests never materialize one giant JSON string; rows that hold vectors cross in blocks of about
        # _VECTOR_BLOCK_FLOATS floats per crossing, so the float[] buffer of a matrix and what the engine builds from it stay small
        step = self._BULK_CHUNK
        width = vector_width(rows, declared)
        if width:
            step = min(
                step,
                max(self._VECTOR_BLOCK_MIN_ROWS, self._VECTOR_BLOCK_FLOATS // width),
            )
        rids = []
        for start in range(0, len(rows), step):
            # the previous chunk's rest rows, text and matrices go before this chunk's are built: two float[] buffers were alive at once
            split = rest = text = None
            chunk = rows[start : start + step]
            split = split_float_vectors(
                self._java_db, type_name, chunk, declared, undeclared=True
            )
            rest = split[0] if split else chunk
            text = None
            if declared_scalars_ok(rest, declared):
                try:
                    text = dumps([row or {} for row in rest])
                except (TypeError, ValueError):
                    text = None  # a value the text cannot carry, see the docstring
            if text is None:
                # a list the matrix cannot carry (ragged, a null, not numbers)
                # is left in this chunk, or a value the text would change: the
                # property-matrix path takes the chunk
                java_rids = self._java_graph_batch.createVertices(
                    type_name, self._to_java_property_matrix(chunk)
                )
                rids.extend(str(rid) for rid in java_rids)
                continue
            if split:
                joined = vertex_batcher.createVerticesJson(
                    self._java_graph_batch, type_name, text, split[1], split[2]
                )
            else:
                joined = vertex_batcher.createVerticesJson(
                    self._java_graph_batch, type_name, text
                )
            joined = str(joined)
            if joined:
                rids.extend(joined.split(";"))
        return rids

    def new_edge(
        self,
        source_vertex_or_rid,
        edge_type: str,
        destination_vertex_or_rid,
        **properties,
    ) -> "GraphBatch":
        """Buffer an edge for creation during flush or close."""
        self._check_not_closed()
        try:
            source_rid = self._to_java_rid(source_vertex_or_rid)
            destination_rid = self._to_java_rid(destination_vertex_or_rid)
            if properties:
                self._java_graph_batch.newEdge(
                    source_rid,
                    edge_type,
                    destination_rid,
                    *self._to_java_varargs(properties),
                )
            else:
                self._java_graph_batch.newEdge(source_rid, edge_type, destination_rid)
            return self
        except Exception as e:
            raise ArcadeDBError(
                f"Failed to buffer batch edge '{edge_type}': {e}"
            ) from e

    def new_edges(
        self,
        source_rids,
        edge_type: str,
        destination_rids,
        properties=None,
    ) -> "GraphBatch":
        """
        Buffer many edges with one JPype crossing per call.

        The bulk counterpart of :meth:`new_edge` (which costs one boundary
        crossing per edge — measured ~24x slower than Java-native for large
        edge lists). RIDs may be strings ("#1:0") or objects with a string
        representation.

        Args:
            source_rids: Sequence of source vertex RIDs.
            edge_type: Edge type name.
            destination_rids: Sequence of destination RIDs (same length).
            properties: Optional sequence of per-edge property dicts (same
                length). Scalars and a list (not one of only numbers), tuple,
                dict, date-time, date, Decimal or bytes value take the bulk
                path (the last ones as tagged values inside the JSON). A list
                of only numbers, a value the text cannot carry (a NaN, an
                integer beyond 64 bits, a date-time inside a list or dict, a
                non-str dict key) or any other type sends the whole call to
                per-edge buffering.
        """
        self._check_not_closed()
        from .results import _bridge_class

        # Read once, so the lengths can be compared before anything is
        # buffered and a fallback to per-edge buffering can start over. zip()
        # stops at the shortest sequence without a word, so a properties list
        # one short used to drop the last edge and none of the paths said so.
        source_rids = list(source_rids)
        destination_rids = list(destination_rids)
        if properties is not None:
            properties = list(properties)
        lengths = [len(source_rids), len(destination_rids)]
        if properties is not None:
            lengths.append(len(properties))
        if len(set(lengths)) != 1:
            names = ["source_rids", "destination_rids", "properties"]
            got = ", ".join(f"{n} {k}" for n, k in zip(names, lengths))
            raise ArcadeDBError(
                f"Failed to buffer batch edges '{edge_type}': source_rids, "
                f"destination_rids and properties must have the same length "
                f"({got})"
            )

        # A property name that is not a str is refused on every path, so look
        # for one before the first edge is buffered: the per-edge fallbacks
        # below would otherwise buffer the edges before it and leave them
        # behind for close() to persist after this call raised.
        for p in properties or ():
            for key in p or ():
                if not isinstance(key, str):
                    raise ArcadeDBError(
                        f"Failed to buffer batch edges '{edge_type}': property "
                        f"names must be str, got {key!r}"
                    )

        edge_batcher = _bridge_class("EdgeBatcher")
        if edge_batcher is None:
            # bridge jar unavailable (warned once): per-edge fallback,
            # documented ~24x slower than the bulk path
            props_iter = properties or [None] * len(source_rids)
            for src, dst, p in zip(source_rids, destination_rids, props_iter):
                self.new_edge(src, edge_type, dst, **(p or {}))
            return self

        try:
            if properties is None:
                # One joined string crosses the boundary in a single bulk
                # copy; a Python list of N strings would be converted
                # element-by-element by JPype and eat the batching win.
                # Chunked so huge ingests never build one giant string.
                sources = [str(r) for r in source_rids]
                dests = [str(r) for r in destination_rids]
                for start in range(0, len(sources), self._BULK_CHUNK):
                    edge_batcher.newEdgesJoined(
                        self._java_graph_batch,
                        ";".join(sources[start : start + self._BULK_CHUNK]),
                        edge_type,
                        ";".join(dests[start : start + self._BULK_CHUNK]),
                    )
                return self

            rows = []
            tagged = False  # a value that needs the tagged text; False keeps the untagged code of before
            for src, dst, p in zip(source_rids, destination_rids, properties):
                row = {"_src": str(src), "_dst": str(dst)}
                for key, value in (p or {}).items():
                    ok = type(key) is str and key not in ("_src", "_dst")
                    if ok and not json_bulk_scalar_ok(value):
                        ok = json_bulk_shape_ok(value)
                        tagged = tagged or ok
                    if not ok:
                        # a type the tagged JSON does not carry (a set, a numpy or
                        # pandas value, a list of only numbers), or a scalar JSON
                        # would change (an integer beyond 64 bits, NaN or Infinity,
                        # a lone surrogate), or a property name that is not a str
                        # (JSON would write 1 as "1"): per-edge fallback preserves
                        # types exactly and refuses the name
                        props_iter = properties
                        for s2, d2, p2 in zip(
                            source_rids, destination_rids, props_iter
                        ):
                            self.new_edge(s2, edge_type, d2, **(p2 or {}))
                        return self
                    row[key] = value
                rows.append(row)

            if not tagged:
                import json

                for start in range(0, len(rows), self._BULK_CHUNK):
                    edge_batcher.newEdgesJson(
                        self._java_graph_batch,
                        edge_type,
                        json.dumps(rows[start : start + self._BULK_CHUNK]),
                    )
                return self

            # Every chunk is written before the first one crosses, so a value the
            # text cannot carry (a NaN, an integer beyond 64 bits or a date-time
            # inside a list or map, a subnormal) can still start the per-edge
            # path over with nothing buffered.
            try:
                texts = [
                    json_bulk_dumps_tagged(rows[start : start + self._BULK_CHUNK])
                    for start in range(0, len(rows), self._BULK_CHUNK)
                ]
            except (TypeError, ValueError):
                for s2, d2, p2 in zip(source_rids, destination_rids, properties):
                    self.new_edge(s2, edge_type, d2, **(p2 or {}))
                return self
            for text in texts:
                edge_batcher.newEdgesJson(self._java_graph_batch, edge_type, text)
            return self
        except Exception as e:
            raise ArcadeDBError(
                f"Failed to buffer batch edges '{edge_type}': {e}"
            ) from e

    def flush(self) -> "GraphBatch":
        """Flush buffered edges to disk.

        Call it outside your own transactions: from engine 26.10.1 it raises
        ArcadeDBError inside one when edges are buffered, leaving the
        transaction and the buffer as they were, and so do new_edge() and
        new_edges() when the buffer fills and flushes. On 26.9.1 and earlier
        they commit the transaction open on this thread, yours included
        (ArcadeData/arcadedb#9242).
        """
        self._check_not_closed()
        try:
            self._java_graph_batch.flush()
            return self
        except Exception as e:
            raise ArcadeDBError(f"Failed to flush GraphBatch: {e}") from e

    def close(self):
        """Flush remaining work and finalize deferred incoming edges.

        Leaving a ``with`` block calls this. Call it outside your own
        transactions: from engine 26.10.1, inside one with work still pending,
        it raises ArcadeDBError and the batch stays open, so a later close()
        after your transaction ends writes that work. On 26.9.1 and earlier it
        commits the transaction open on this thread, yours included
        (ArcadeData/arcadedb#9242).
        """
        if self._closed:
            return
        refused = False
        try:
            self._java_graph_batch.close()
        except Exception as e:
            # A close the engine refused because of the caller's transaction
            # released nothing: the batch is still open with its work pending,
            # and marking the wrapper closed would make the retry a no-op that
            # drops that work and keeps the database's batch guard held.
            refused = self._is_callers_transaction_refusal(e)
            raise ArcadeDBError(f"Failed to close GraphBatch: {e}") from e
        finally:
            if not refused:
                self._closed = True

    def _is_callers_transaction_refusal(self, error: Exception) -> bool:
        """True when the engine refused a call because the caller has a
        transaction open (IllegalStateException, ArcadeData/arcadedb#9242)."""
        return isinstance(
            error, jpype.JClass("java.lang.IllegalStateException")
        ) and bool(self._java_db.isTransactionActive())

    def get_total_edges_created(self) -> int:
        self._check_not_closed()
        return int(self._java_graph_batch.getTotalEdgesCreated())

    def get_buffered_edge_count(self) -> int:
        self._check_not_closed()
        return int(self._java_graph_batch.getBufferedEdgeCount())

    def get_deferred_incoming_edge_count(self) -> int:
        self._check_not_closed()
        return int(self._java_graph_batch.getDeferredIncomingEdgeCount())

    def _check_not_closed(self):
        if self._closed:
            raise ArcadeDBError("GraphBatch is closed")

    @classmethod
    def _to_java_wal_flush(cls, mode: str):
        normalized_mode = mode.lower()
        if normalized_mode not in cls._VALID_WAL_FLUSH_MODES:
            raise ValueError(
                f"Invalid wal_flush mode: {mode}. Must be one of: "
                f"{list(cls._VALID_WAL_FLUSH_MODES.keys())}"
            )

        FlushType = jpype.JClass("com.arcadedb.engine.WALFile$FlushType")
        return getattr(FlushType, cls._VALID_WAL_FLUSH_MODES[normalized_mode])

    def _unwrap_record(self, record):
        java_document = getattr(record, "_java_document", None)
        return java_document if java_document is not None else record

    def _to_java_varargs(self, properties: dict[str, Any]) -> list[Any]:
        varargs = []
        for key, value in properties.items():
            varargs.append(key)
            varargs.append(convert_python_to_java(value))
        return varargs

    def _to_java_property_matrix(
        self, properties_iterable: Iterable[Optional[dict[str, Any]]]
    ):
        rows = []
        for properties in properties_iterable:
            if properties is None:
                rows.append(None)
                continue
            if not isinstance(properties, dict):
                raise TypeError("create_vertices() property rows must be dicts or None")
            row = self._to_java_varargs(properties)
            rows.append(jpype.JArray(jpype.JObject)(row))

        return jpype.JArray(jpype.JObject[:])(rows)

    def _to_java_rid(self, value):
        value = self._unwrap_record(value)
        if hasattr(value, "getIdentity"):
            return value.getIdentity()
        if isinstance(value, str):
            RID = jpype.JClass("com.arcadedb.database.RID")
            return RID(value)
        if isinstance(value, Document):
            return value.get_identity()
        return value
