"""What the JPype ``Schema`` returns as Java objects, rebuilt from ``schema:types`` and ``schema:indexes`` rows for the native
backend: types and indexes with the engine's camelCase accessors (``getName``, ``existsProperty``, ``getTypeName``, ...) and the
vector index wrapper over ``vectorNeighbors``. Read-only views: DDL stays SQL statements.
"""

import json
import re

from ..exceptions import ArcadeDBError
from ..vector import VectorIndex as _BaseVectorIndex
from . import needs_jvm
from .codec import _np

_PROPERTY = re.compile(r"name: (.+?), type: ([A-Z_]+)")

# find_nearest_approximate crosses in arcadedb_n_vindex_bin. The answer buffer is sized for this many bucket indexes of the type at
# first (each answers with up to k hits); a type with more of them costs one more crossing, and the larger size is remembered.
_VINDEX_PARTS = 8
# an answer buffer larger than this is not asked for: the JSON route answers
_VINDEX_MAX_BYTES = 1 << 30
# int64 words per hit: bucket, position, the score as the bits of a double
_HIT_WORDS = 3
_RID_PAIRS = re.compile(r"#\d{1,9}:\d{1,18}(?:;#\d{1,9}:\d{1,18})*")
_DIGITS = re.compile(r"\d+")
_NUMBERS = frozenset((float, int))
_FLOAT_DTYPES = (
    (_np.dtype(_np.float16), _np.dtype(_np.float32), _np.dtype(_np.float64))
    if _np is not None
    else ()
)
# an int up to this is a double, so float32 rounds it as Java's floatValue() of the parsed JSON number does
_EXACT_INT = 1 << 53


def _name_type(entry):
    """(name, type) of an entry of a ``schema:types`` row's ``properties``: a dict since #195 (the codec writes a nested row
    as a map), the engine's text form of the row before it; (None, None) when neither.
    """
    if isinstance(entry, dict):
        return entry.get("name"), entry.get("type")
    m = _PROPERTY.search(entry) if isinstance(entry, str) else None
    return (m.group(1), m.group(2)) if m else (None, None)


class NativeType:
    """A type as a ``schema:types`` row."""

    def __init__(self, row: dict):
        self._row = row

    def getName(self) -> str:
        return self._row["name"]

    def getType(self) -> str:
        return self._row.get("type")

    def getPropertyNames(self):
        return [n for n, _ in map(_name_type, self._row.get("properties") or []) if n]

    def existsProperty(self, name: str) -> bool:
        return name in self.getPropertyNames()

    def getParentTypes(self):
        return list(self._row.get("parentTypes") or [])

    getSuperTypes = getParentTypes

    def getBuckets(self):
        return list(self._row.get("buckets") or [])

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        needs_jvm(f"the Java object a Schema method returns ({name})")

    def __repr__(self):
        return f"<Type {self.getName()}>"


class NativeIndex:
    """An index as a ``schema:indexes`` row."""

    def __init__(self, row: dict):
        self._row = row

    def getName(self) -> str:
        return self._row["name"]

    def getTypeName(self) -> str:
        return self._row.get("typeName")

    def getPropertyNames(self):
        props = self._row.get("properties") or []
        return list(props[0]) if props and isinstance(props[0], list) else list(props)

    def isUnique(self) -> bool:
        return bool(self._row.get("unique"))

    def getType(self) -> str:
        return self._row.get("indexType")

    def isAutomatic(self) -> bool:
        return bool(self._row.get("automatic"))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        needs_jvm(f"the Java object a Schema method returns ({name})")

    def __repr__(self):
        return f"<Index {self.getName()}>"


class _IndexShim:
    """The few LSMVectorIndex methods the shared ``VectorIndex`` code calls, over statements."""

    def __init__(self, database, type_name, prop, name):
        self._db, self._type, self._prop, self._name = database, type_name, prop, name
        self._vindex_parts = _VINDEX_PARTS

    def getName(self):
        return self._name

    def getTypeName(self):
        return self._type

    def getPropertyNames(self):
        return [self._prop]

    def engine_info(self) -> dict:
        """The engine's configuration of the index (the defaults it chose included), read once: it does not change after the index exists."""
        info = self.__dict__.get("_engine_info")
        if info is None:
            from . import lib
            from .extras import _call, _enc

            info = self._engine_info = _call(
                "Failed to read the vector index metadata",
                "arcadedb_n_vindex",
                self._db._h,
                lib.b("metadata"),
                lib.b(_enc({"type": self._type, "prop": self._prop})),
            )
        return info

    def getMetadata(self):
        """The index's quantization and pq_clusters, shaped like the Java metadata object."""
        from types import SimpleNamespace

        info = self.engine_info()
        return SimpleNamespace(
            quantizationType=str(info["quantization"] or "NONE"),
            pqClusters=int(info["pq_clusters"]),
        )

    def getIdPropertyName(self):
        return self._db._vector_id_properties.get((self._type, self._prop), "id")

    def countEntries(self):
        count_sql = f"SELECT count(*) AS c FROM `{self._type}` WHERE `{self._prop}` IS NOT NULL"  # nosec B608 - backtick-quoted schema identifiers, no values
        row = self._db.query("sql", count_sql).first()
        return int(row.get("c"))

    def warmUp(self):
        # the engine loads a persisted graph on the first search: one search now, on a vector of the index's dimension
        dims = self.engine_info()["dimensions"]
        if dims:
            self._db.query(
                "sql", "SELECT vectorNeighbors(?, ?, 1)", self._name, [1.0] * dims
            ).close()

    def buildVectorGraphNow(self):
        # the engine's own rebuild of every bucket index (LSMVectorIndex.buildVectorGraphNow), as the JPype wrapper calls it:
        # a search builds a graph only when there is none, so it left the vectors written since in the delta (#162)
        from . import lib
        from .extras import _call, _enc

        _call(
            "Failed to rebuild vector graph",
            "arcadedb_n_vindex",
            self._db._h,
            lib.b("build"),
            lib.b(_enc({"type": self._type, "prop": self._prop})),
        )


def _rid_text(value) -> str:
    get_rid = getattr(value, "get_rid", None)
    return get_rid() if get_rid is not None else str(value)


def _vindex_query(query_vector):
    """The query of an approximate search as a contiguous float32 array, for ``arcadedb_n_vindex_bin``; None when the JSON route must
    carry it.

    The engine searches with float32 whatever it is given: the JSON route sends doubles and Java narrows them with ``floatValue()``, which
    rounds to nearest even, as numpy's cast to float32 does, so the library reads the same floats from the address. What goes by
    address: a one-dimensional ``float16``, ``float32`` or ``float64`` array of the native byte order, and a list or tuple of ``float``
    and ``int`` (an int beyond 2**53 is not a double and stays JSON). Everything else keeps the JSON route (it reads whatever the
    object's ``tolist()`` or ``list()`` gives and reports what the engine makes of it): a generator, any other array or sequence, an
    entry that is not a number, a bool, a NaN or an infinity (also one that float32 overflows to), and an empty query.
    """
    if _np is None:
        return None
    if type(query_vector) is _np.ndarray:
        if query_vector.ndim != 1 or query_vector.dtype not in _FLOAT_DTYPES:
            return None
        source = query_vector
    elif type(query_vector) in (list, tuple):
        kinds = set(map(type, query_vector))
        if not kinds or not kinds <= _NUMBERS:
            return None
        if int in kinds and any(
            abs(v) > _EXACT_INT for v in query_vector if type(v) is int
        ):
            return None
        source = query_vector
    else:
        return None
    try:
        with _np.errstate(over="ignore", invalid="ignore"):
            f32 = _np.asarray(source, dtype=_np.float32)
    except (OverflowError, ValueError, TypeError):
        return None
    if f32.size == 0 or not _np.isfinite(f32).all():
        return None
    return _np.require(f32, requirements=["C", "A"])


def _vindex_allowed(texts):
    """The RIDs of an approximate search's filter (``"#bucket:position"`` texts) as an int64 array of (bucket, position) pairs, or None
    when one of them is not that plain form (the JSON route hands the text to the engine's own RID parser).
    """
    try:
        joined = ";".join(texts)
    except TypeError:
        return None
    numbers = _DIGITS.findall(joined) if _RID_PAIRS.fullmatch(joined) else ()
    # one text that holds a ';' is two RIDs to the pattern but one bad RID to the engine: the count tells them apart
    if not numbers or len(numbers) != 2 * len(texts):
        return None
    return _np.array(numbers, dtype=_np.int64)


def _query_vector_param(query_vector):
    """The query vector as the one parameter the library takes it in.

    A numeric 1-d vector (an ndarray, or a list/tuple of numbers: numpy walks a list in C) becomes a float32 array, which
    ``encode_params`` sends through the binary float32 side buffer (tag ``B``): no ``tolist()``, no float32-exactness check of a
    list, and no JSON number text for a float64 query whose values float32 would round. The engine reads the vector as float32
    whatever it is given, so the rounding is the same. Anything else (text or None entries, a generator, numpy missing) stays a
    list and reaches the engine as it was given; numpy would turn a None into nan.
    """
    if _np is not None:
        try:
            arr = _np.asarray(query_vector)
        except (
            Exception
        ):  # nosec B110 - ragged or unconvertible input: the list path below reports it
            arr = None
        if arr is not None and arr.ndim == 1 and arr.dtype.kind in "fiu":
            return arr.astype(_np.float32, copy=False)
    return (
        query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
    )


def _quote_name(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


# find_nearest(include_vector=False): the engine's row of a hit is the record's properties plus the columns @rid, @type and distance
# (ScoredRecordRow; before 26.11.1, ArcadeData/arcadedb#10898, also a nested `record` column holding every property a second time).
# `*` copies the row, `!name` drops the indexed vector and the `record` column. A property of the record called distance or record is
# hidden by those columns in the row, so such a hit holds them back too (_LAZY_COLUMNS) and loads them with the vector.
_LAZY_COLUMNS = ("record", "distance")
_HITS_FROM = "FROM (SELECT expand(vectorNeighbors(?, ?, ?)))"


class NativeVectorIndex(_BaseVectorIndex):
    """``VectorIndex`` over ``vectorNeighbors``: ``find_nearest`` returns ``[(record, distance)]`` like the JPype wrapper."""

    def __init__(self, database, type_name, prop, name=None):
        shim = _IndexShim(database, type_name, prop, name or f"{type_name}[{prop}]")
        super().__init__(shim, database)
        self._shim = shim

    def _resolve_lsm_indexes(self):
        return [self._shim]

    def _check_live(self):
        # every search and counter names the index in its statement, so a rebuilt index is found by name on each call:
        # there is no cached engine object to go stale (the JPype wrapper re-resolves one here)
        return

    def _run_search(
        self,
        *,
        query_vector,
        k,
        allowed_rids,
        approximate,
        ef_search=None,
        include_vector=True,
    ):
        if approximate:
            return self._run_approximate(query_vector, k, allowed_rids)
        if not include_vector:
            return self._run_search_without_vector(
                query_vector, k, allowed_rids, ef_search
            )
        from .records import _wrap_dict

        vector = _query_vector_param(query_vector)
        # the statement's own options: efSearch, and filter (the RIDs the search may return), as the JPype path passes them
        options = {}
        if ef_search is not None:
            options["efSearch"] = int(ef_search)
        if allowed_rids:
            options["filter"] = [_rid_text(r) for r in allowed_rids]
        if options:
            rows = self._database.query(
                "sql",
                "SELECT expand(vectorNeighbors(?, ?, ?, ?))",
                self._shim._name,
                vector,
                int(k),
                options,
            )
        else:
            rows = self._database.query(
                "sql",
                "SELECT expand(vectorNeighbors(?, ?, ?))",
                self._shim._name,
                vector,
                int(k),
            )
        out = []
        for row in rows:
            nested = row.get("record")
            if nested is None and row.get("@rid") is not None:
                # From ArcadeDB 26.11.1 (ArcadeData/arcadedb#10898) a vector row lists the record's
                # properties once, on the row itself, and no longer under "record"
                nested = {k: v for k, v in row.to_dict().items() if k != "distance"}
            record = (
                None
                if nested is None
                else _wrap_dict(
                    self._database, nested, self._database._kind_of(nested.get("@type"))
                )
            )
            out.append((record, float(row.get("distance"))))
        return self._sort_results(out)[:k]

    def _run_search_without_vector(self, query_vector, k, allowed_rids, ef_search):
        """``find_nearest(include_vector=False)``: the hits come back without the indexed vector, which loads on first use.

        ``expand(vectorNeighbors(..))`` alone ships every hit's vector, and the engine reads each record to build the row whatever
        the columns asked for. Here the statement drops the vector (and the nested ``record`` column of an engine before 26.11.1),
        and ``_Mixin._fill`` loads the record, once, by RID, the first time the vector or the whole record is needed.
        """
        from .records import _wrap_partial

        vector = _query_vector_param(query_vector)
        options = {}
        if ef_search is not None:
            options["efSearch"] = int(ef_search)
        if allowed_rids:
            options["filter"] = [_rid_text(r) for r in allowed_rids]
        prop = self._shim._prop
        dropped = ", ".join(
            "!" + _quote_name(name) for name in dict.fromkeys((prop, "record"))
        )
        sql = f"SELECT *, {dropped} {_HITS_FROM}"  # nosec B608 - dropped is backtick-quoted schema names, the values go through ? parameters
        if options:
            sql = sql.replace("vectorNeighbors(?, ?, ?)", "vectorNeighbors(?, ?, ?, ?)")
            rows = self._database.query(
                "sql", sql, self._shim._name, vector, int(k), options
            )
        else:
            rows = self._database.query("sql", sql, self._shim._name, vector, int(k))
        omitted = (prop, *_LAZY_COLUMNS)
        out = []
        for row in rows:
            props = row.to_dict()
            rid = props.pop("@rid", None)
            type_name = props.pop("@type", None)
            distance = props.pop("distance")
            if rid is None:
                record = None
            else:
                record = _wrap_partial(
                    self._database,
                    _rid_text(rid),
                    type_name,
                    self._database._kind_of(type_name),
                    props,
                    omitted,
                )
            out.append((record, float(distance)))
        return self._sort_results(out)[:k]

    def _run_approximate(self, query_vector, k, allowed_rids):
        """The PQ-scored search of the engine (no statement runs it): the engine returns (RID, distance), the records are loaded here."""
        from . import lib

        vector = _vindex_query(query_vector) if lib.VINDEX_BIN else None
        k = int(k)
        allowed = [_rid_text(r) for r in allowed_rids] if allowed_rids else None
        hits = None if vector is None else self._approximate_packed(vector, k, allowed)
        if hits is None:
            hits = self._approximate_json(query_vector, k, allowed)
        out = [(self._database.lookup_by_rid(rid), score) for rid, score in hits]
        return self._sort_results(out)[:k]

    def _approximate_packed(self, vector, k, allowed):
        """``[(rid text, score)]`` from ``arcadedb_n_vindex_bin``, or None when the JSON route has to carry the call (a k that is not a C
        int, a filter that is not plain ``#bucket:position`` texts, an answer buffer of more than a gibibyte, or one that the engine keeps
        outgrowing). The library fills one buffer of int64 words, three per hit, and answers with their number; the query, the filter and
        the buffer are numpy arrays that live until the call returns.
        """
        from . import lib
        from .database import _error

        if not -(1 << 31) <= k < (1 << 31):
            return None
        pairs, count = None, 0
        if allowed:
            pairs = _vindex_allowed(allowed)
            if pairs is None:
                return None
            count = len(pairs) // 2
        shim = self._shim
        room = max(k, 1)
        for _ in range(3):
            cap = room * shim._vindex_parts
            if cap * _HIT_WORDS * 8 > _VINDEX_MAX_BYTES:
                return None
            out = _np.empty(cap * _HIT_WORDS, dtype=_np.int64)
            n = lib.vindex_bin(
                self._database._h,
                lib.bc(shim._type),
                lib.bc(shim._prop),
                k,
                vector.ctypes.data,
                vector.size,
                pairs.ctypes.data if count else 0,
                count,
                out.ctypes.data,
                cap,
            )
            if n < 0:
                raise _error("Approximate search failed", json.loads(lib.last_error()))
            if n <= cap:
                words, end = out[: n * _HIT_WORDS], n * _HIT_WORDS
                rids = [
                    f"#{b}:{p}"
                    for b, p in zip(
                        words[0::_HIT_WORDS].tolist(), words[1::_HIT_WORDS].tolist()
                    )
                ]
                scores = out.view(_np.float64)[2:end:_HIT_WORDS].tolist()
                return list(zip(rids, scores))
            # more hits than the buffer holds: nothing was written; ask again with room for them, and keep that size
            shim._vindex_parts = -(-n // room)
        return None

    def _approximate_json(self, query_vector, k, allowed):
        """``[(rid text, score)]`` through the JSON text of ``arcadedb_n_vindex``: the route for every call the binary entry cannot carry."""
        from . import lib
        from .extras import _call, _enc

        vector = (
            query_vector.tolist()
            if hasattr(query_vector, "tolist")
            else list(query_vector)
        )
        reply = _call(
            "Approximate search failed",
            "arcadedb_n_vindex",
            self._database._h,
            lib.b("approximate"),
            lib.b(
                _enc(
                    {
                        "type": self._shim._type,
                        "prop": self._shim._prop,
                        "vector": vector,
                        "k": k,
                        "allowed": allowed,
                    }
                )
            ),
        )
        return [(rid, float(distance)) for rid, distance in reply["hits"]]

    def find_nearest_approximate(self, query_vector, k=10, allowed_rids=None):
        """PQ approximate search over PRODUCT quantization, the engine's own (``findNeighborsFromVectorApproximate``)."""
        if self.get_quantization() != "PRODUCT":
            raise ArcadeDBError("Approximate search requires quantization=PRODUCT (PQ)")
        try:
            return self._run_search(
                query_vector=query_vector,
                k=k,
                allowed_rids=allowed_rids,
                approximate=True,
            )
        except ArcadeDBError:
            raise
        except Exception as e:
            raise ArcadeDBError(f"Approximate search failed: {e}") from e

    def _lookup_query_vector_by_key(self, key):
        type_name, prop = self._shim._type, self._shim._prop
        id_prop = self._shim.getIdPropertyName()
        lookup_sql = f"SELECT `{prop}` AS `query_vector` FROM `{type_name}` WHERE `{id_prop}` = ? LIMIT 1"  # nosec B608 - backtick-quoted schema identifiers; the key goes through the ? parameter
        row = self._database.query("sql", lookup_sql, key).first()
        if row is None:
            raise ArcadeDBError(
                f"No record found in type '{type_name}' where {id_prop} = {key!r}"
            )
        if row.get("query_vector") is None:
            raise ArcadeDBError(
                f"Record found for {id_prop} = {key!r} but property '{prop}' is null"
            )
        return row.get("query_vector")

    def _live_info(self):
        # read afresh, the native side of the JPype wrapper's _check_live: a dropped index raises here (the glue names it),
        # REBUILD INDEX builds new bucket indexes, and a re-created index has its own configuration; a memoized read
        # answered all three from the old index
        self._shim.__dict__.pop("_engine_info", None)
        return self._shim.engine_info()

    def get_size(self):
        self._live_info()
        return super().get_size()

    def get_stats(self):
        """The engine's live counters of the primary bucket index, from its ``schema:index:<name>`` row (the row's own columns left out)."""
        bucket = self._live_info()["bucket"]
        rows = self._database.query(
            "sql",
            f"SELECT FROM schema:index:`{bucket}`",  # nosec B608 - a backtick-quoted index name the engine reported
        ).to_list()
        if not rows:
            raise ArcadeDBError(
                f"Failed to read vector index stats: no index {bucket!r}"
            )
        return {k: v for k, v in rows[0].items() if k not in _INDEX_ROW_COLUMNS}

    def get_metadata(self):
        """The engine's configuration of the index: what this session created and what an index opened from disk carries, defaults included."""
        info = self._live_info()
        return {
            "index_name": self._shim._name,
            "bucket_index_name": info["bucket"],
            "type_name": info["type_name"],
            "vector_property": self._shim._prop,
            **{k: info[k] for k in _METADATA_KEYS},
            # present when the engine the image links has them, as on the JVM build (26.10.2 has not; the glue emits them from 26.11.1 on)
            **{k: info[k] for k in _OPTIONAL_METADATA_KEYS if k in info},
        }

    def get_quantization(self):
        return str(self._live_info()["quantization"] or "NONE")


# the descriptive columns of a schema:index row; every other column is a counter of IndexInternal.getStats()
_INDEX_ROW_COLUMNS = frozenset(
    {
        "name",
        "indexType",
        "typeName",
        "properties",
        "keyTypes",
        "unique",
        "automatic",
        "compacting",
        "valid",
        "supportsOrderedIterations",
        "nullStrategy",
        "associatedBucketId",
        "fileId",
        "size",
        "upgradeWarning",
        "typeIndexName",
    }
)
_OPTIONAL_METADATA_KEYS = ("rerank_oversample", "fused_pq")
_METADATA_KEYS = (
    "dimensions",
    "similarity_function",
    "id_property",
    "quantization",
    "max_connections",
    "beam_width",
    "ef_search",
    "graph_build_cache_size",
    "mutations_before_rebuild",
    "store_vectors_in_graph",
    "add_hierarchy",
    "build_state",
    "pq_subspaces",
    "pq_clusters",
    "pq_center_globally",
    "pq_training_limit",
)


def _check_vector_arguments(
    quantization,
    encoding,
    pq_subspaces,
    pq_clusters,
    pq_center_globally,
    pq_training_limit,
    rerank_oversample=None,
    fused_pq=None,
):
    quantization_kind = quantization.upper() if quantization else ""
    if any(
        v is not None
        for v in (
            pq_subspaces,
            pq_clusters,
            pq_center_globally,
            pq_training_limit,
            fused_pq,
        )
    ):
        if quantization_kind != "PRODUCT":
            raise ValueError("PQ parameters require quantization='PRODUCT'")
    # PRODUCT and, from 26.11.1 (ArcadeData/arcadedb#10968), BINARY rerank on the stored
    # vectors; every other quantization ignores the setting.
    if rerank_oversample is not None and quantization_kind not in ("PRODUCT", "BINARY"):
        raise ValueError(
            "rerank_oversample requires quantization='PRODUCT' or 'BINARY'"
        )
    if (
        encoding
        and encoding.upper() == "INT8"
        and quantization
        and quantization.upper() == "INT8"
    ):
        raise ValueError(
            "encoding='INT8' cannot be combined with quantization='INT8'; use quantization='NONE' for native INT8 storage"
        )


def create_vector_index(
    db,
    vertex_type,
    vector_property,
    dimensions,
    id_property,
    distance_function,
    max_connections,
    beam_width,
    quantization,
    encoding,
    graph_build_cache_size,
    mutations_before_rebuild,
    store_vectors_in_graph,
    add_hierarchy,
    pq_subspaces,
    pq_clusters,
    pq_center_globally,
    pq_training_limit,
    rerank_oversample,
    fused_pq,
    build_graph_now,
):
    """``CREATE INDEX .. LSM_VECTOR METADATA {..}`` with the JPype builder's defaults and checks; returns the wrapper."""
    try:
        _check_vector_arguments(
            quantization,
            encoding,
            pq_subspaces,
            pq_clusters,
            pq_center_globally,
            pq_training_limit,
            rerank_oversample,
            fused_pq,
        )
    except ValueError as e:
        raise ArcadeDBError(f"Failed to create vector index: {e}") from e
    meta = {
        "dimensions": int(dimensions),
        "similarity": distance_function,
        "maxConnections": int(max_connections),
        "beamWidth": int(beam_width),
    }
    if id_property:
        meta["idPropertyName"] = id_property
    if quantization:
        meta["quantization"] = quantization
    if encoding:
        meta["encoding"] = encoding
    for key, value in (
        ("pqSubspaces", pq_subspaces),
        ("pqClusters", pq_clusters),
        ("pqCenterGlobally", pq_center_globally),
        ("pqTrainingLimit", pq_training_limit),
    ):
        if value is not None:
            meta[key] = bool(value) if key == "pqCenterGlobally" else int(value)
    if rerank_oversample is not None:
        meta["rerankOversample"] = int(rerank_oversample)
    if fused_pq is not None:
        meta["fusedPQ"] = bool(fused_pq)
    if store_vectors_in_graph:
        meta["storeVectorsInGraph"] = True
    if add_hierarchy is not None:
        meta["addHierarchy"] = bool(add_hierarchy)
    if graph_build_cache_size is not None:
        meta["graphBuildCacheSize"] = int(graph_build_cache_size)
    if mutations_before_rebuild is not None:
        meta["mutationsBeforeRebuild"] = int(mutations_before_rebuild)
    try:
        db.command(
            "sql",
            f"CREATE INDEX ON `{vertex_type}` (`{vector_property}`) LSM_VECTOR METADATA {json.dumps(meta)}",
        ).close()
        key = (vertex_type, vector_property)
        db._vector_meta[key] = meta
        db._vector_dimensions[key] = int(dimensions)
        db._vector_id_properties[key] = id_property or "id"
        index = NativeVectorIndex(db, vertex_type, vector_property)
        if build_graph_now:
            index.build_graph_now()
        return index
    except Exception as e:
        raise ArcadeDBError(f"Failed to create vector index: {e}") from e
