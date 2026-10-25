"""What the JPype ``Schema`` returns as Java objects, rebuilt from ``schema:types`` and ``schema:indexes`` rows for the native
backend: types and indexes with the engine's camelCase accessors (``getName``, ``existsProperty``, ``getTypeName``, ...) and the
vector index wrapper over ``vectorNeighbors``. Read-only views: DDL stays SQL statements.
"""

import json
import re

from ..exceptions import ArcadeDBError
from ..vector import VectorIndex as _BaseVectorIndex
from . import needs_jvm

_PROPERTY = re.compile(r"name: (.+?), type: ([A-Z_]+)")


class NativeType:
    """A type as a ``schema:types`` row."""

    def __init__(self, row: dict):
        self._row = row

    def getName(self) -> str:
        return self._row["name"]

    def getType(self) -> str:
        return self._row.get("type")

    def getPropertyNames(self):
        return [
            m.group(1)
            for m in map(_PROPERTY.search, self._row.get("properties") or [])
            if m
        ]

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
        """Just what the PRODUCT-readiness check reads: the index's quantization and pq_clusters."""
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
        # the engine loads a persisted graph on the first search: run that search now
        self.buildVectorGraphNow()

    def buildVectorGraphNow(self):
        # the engine builds the graph on the first search: one search now, on a vector of the index's dimension
        dims = self.engine_info()["dimensions"]
        if dims:
            self._db.query(
                "sql", "SELECT vectorNeighbors(?, ?, 1)", self._name, [1.0] * dims
            ).close()


def _rid_text(value) -> str:
    get_rid = getattr(value, "get_rid", None)
    return get_rid() if get_rid is not None else str(value)


class NativeVectorIndex(_BaseVectorIndex):
    """``VectorIndex`` over ``vectorNeighbors``: ``find_nearest`` returns ``[(record, distance)]`` like the JPype wrapper."""

    def __init__(self, database, type_name, prop, name=None):
        shim = _IndexShim(database, type_name, prop, name or f"{type_name}[{prop}]")
        super().__init__(shim, database)
        self._shim = shim

    def _resolve_lsm_indexes(self):
        return [self._shim]

    def _run_search(
        self, *, query_vector, k, allowed_rids, approximate, ef_search=None
    ):
        if approximate:
            return self._run_approximate(query_vector, k, allowed_rids)
        from .records import _wrap_dict

        vector = (
            query_vector.tolist()
            if hasattr(query_vector, "tolist")
            else list(query_vector)
        )
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
            record = (
                None
                if nested is None
                else _wrap_dict(
                    self._database, nested, self._database._kind_of(nested.get("@type"))
                )
            )
            out.append((record, float(row.get("distance"))))
        return self._sort_results(out)[:k]

    def _run_approximate(self, query_vector, k, allowed_rids):
        """The PQ-scored search of the engine (no statement runs it): the engine returns (RID, distance), the records are loaded here."""
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
                        "k": int(k),
                        "allowed": (
                            [_rid_text(r) for r in allowed_rids]
                            if allowed_rids
                            else None
                        ),
                    }
                )
            ),
        )
        out = [
            (self._database.lookup_by_rid(rid), float(distance))
            for rid, distance in reply["hits"]
        ]
        return self._sort_results(out)[:k]

    def find_nearest_approximate(self, query_vector, k=10, allowed_rids=None):
        """PQ approximate search over PRODUCT quantization, the engine's own (``findNeighborsFromVectorApproximate``)."""
        if self.get_quantization() != "PRODUCT":
            raise ArcadeDBError("Approximate search requires quantization=PRODUCT (PQ)")
        self._ensure_product_quantization_ready()
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

    def get_stats(self):
        """The engine's live counters of the primary bucket index, from its ``schema:index:<name>`` row (the row's own columns left out)."""
        bucket = self._shim.engine_info()["bucket"]
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
        info = self._shim.engine_info()
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
        return str(self._shim.engine_info()["quantization"] or "NONE")


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
    if any(
        v is not None
        for v in (
            pq_subspaces,
            pq_clusters,
            pq_center_globally,
            pq_training_limit,
            rerank_oversample,
            fused_pq,
        )
    ):
        if not quantization or quantization.upper() != "PRODUCT":
            raise ValueError("PQ parameters require quantization='PRODUCT'")
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
