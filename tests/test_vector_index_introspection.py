"""Vector index introspection and PRODUCT-quantization search: the same behaviour on the JVM wheel and the native wheel.

``get_stats`` (live counters), ``get_metadata`` and ``get_quantization`` of an index that this session did not create (a
reopened database), and ``find_nearest_approximate`` over PRODUCT quantization. The tests use no Java object, so they run
on both backends.
"""

import pytest

import arcadedb

# keys the metadata dict has on both wheels (the engine's configuration of the index)
METADATA_KEYS = {
    "index_name",
    "bucket_index_name",
    "type_name",
    "vector_property",
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
}


def _make_type(db):
    db.command("sql", "CREATE VERTEX TYPE Doc")
    db.command("sql", "CREATE PROPERTY Doc.name STRING")
    db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")


def _insert(db, rows):
    with db.transaction():
        for name, vector in rows:
            db.command(
                "sql",
                "INSERT INTO Doc SET name = ?, embedding = ?",
                name,
                arcadedb.to_java_float_array(vector),
            )


UNIT_VECTORS = [("x", [1.0, 0.0, 0.0]), ("y", [0.0, 1.0, 0.0]), ("z", [0.0, 0.0, 1.0])]
# as many vectors as PQ clusters: each vector is its own centroid, so the approximate answer is the exact one
PQ_VECTORS = UNIT_VECTORS + [("w", [-1.0, 0.0, 0.0])]


# the engine reports these two only from 26.11.1 on, on both builds
OPTIONAL_METADATA_KEYS = {"rerank_oversample", "fused_pq"}


class TestGetStats:
    def test_counters_start_at_zero_and_track_inserts(self, temp_db):
        _make_type(temp_db)
        index = temp_db.create_vector_index("Doc", "embedding", dimensions=3)

        stats = index.get_stats()
        assert isinstance(stats, dict) and stats
        assert stats["totalVectors"] == 0
        assert isinstance(stats["dimensions"], int)
        assert stats["vectorFetchFromDocuments"] >= 0

        _insert(temp_db, UNIT_VECTORS[:2])
        assert index.get_stats()["totalVectors"] == 2

    def test_values_are_plain_scalars_and_the_listing_columns_are_not_counters(
        self, temp_db
    ):
        _make_type(temp_db)
        index = temp_db.create_vector_index("Doc", "embedding", dimensions=3)
        _insert(temp_db, UNIT_VECTORS)

        stats = index.get_stats()
        assert all(
            v is None or isinstance(v, (int, float, bool, str)) for v in stats.values()
        )
        # a schema:index row also names the index and describes it; get_stats is the counters only
        for descriptive in ("name", "indexType", "typeName", "properties", "fileId"):
            assert descriptive not in stats

    def test_searches_move_the_search_counter(self, temp_db):
        _make_type(temp_db)
        index = temp_db.create_vector_index("Doc", "embedding", dimensions=3)
        _insert(temp_db, UNIT_VECTORS)

        before = index.get_stats()["searchOperations"]
        index.find_nearest([1.0, 0.0, 0.0], k=1)
        assert index.get_stats()["searchOperations"] > before


class TestGetMetadataOfAReopenedIndex:
    @staticmethod
    def _create(path, **kwargs):
        with arcadedb.create_database(path) as db:
            _make_type(db)
            index = db.create_vector_index(
                "Doc", "embedding", dimensions=4, id_property="name", **kwargs
            )
            _insert(
                db,
                [
                    ("a", [1.0, 0.0, 0.0, 0.0]),
                    ("b", [0.0, 1.0, 0.0, 0.0]),
                    ("c", [0.0, 0.0, 1.0, 0.0]),
                ],
            )
            return index.get_metadata()

    def test_metadata_after_reopen_equals_the_creating_session(self, temp_db_path):
        created = self._create(
            temp_db_path,
            distance_function="EUCLIDEAN",
            max_connections=16,
            beam_width=64,
            add_hierarchy=False,
        )
        assert set(created) - OPTIONAL_METADATA_KEYS == METADATA_KEYS

        with arcadedb.open_database(temp_db_path) as db:
            index = db.schema.get_vector_index("Doc", "embedding")
            reopened = index.get_metadata()

        assert set(reopened) - OPTIONAL_METADATA_KEYS == METADATA_KEYS
        assert reopened["dimensions"] == 4
        assert reopened["similarity_function"] == "EUCLIDEAN"
        assert reopened["id_property"] == "name"
        assert reopened["max_connections"] == 16
        assert reopened["beam_width"] == 64
        assert reopened["add_hierarchy"] is False
        # the build state is runtime state; everything else is the index's configuration
        created.pop("build_state")
        reopened.pop("build_state")
        assert reopened == created

    def test_product_quantization_settings_survive_a_reopen(self, temp_db_path):
        self._create(temp_db_path, quantization="PRODUCT", pq_clusters=2)

        with arcadedb.open_database(temp_db_path) as db:
            index = db.schema.get_vector_index("Doc", "embedding")
            assert index.get_quantization() == "PRODUCT"
            meta = index.get_metadata()
        assert meta["quantization"] == "PRODUCT"
        assert meta["pq_clusters"] == 2

    @pytest.mark.parametrize("quantization", ["NONE", "INT8", "BINARY"])
    def test_quantization_of_a_reopened_index(self, temp_db_path, quantization):
        self._create(temp_db_path, quantization=quantization)
        with arcadedb.open_database(temp_db_path) as db:
            index = db.schema.get_vector_index("Doc", "embedding")
            assert index.get_quantization() == quantization
            assert index.get_metadata()["quantization"] == quantization

    def test_the_default_quantization_is_the_same_after_a_reopen(self, temp_db_path):
        created = self._create(temp_db_path)
        with arcadedb.open_database(temp_db_path) as db:
            index = db.schema.get_vector_index("Doc", "embedding")
            assert index.get_quantization() == created["quantization"]


class TestFindNearestApproximate:
    @staticmethod
    def _pq_index(db, rows):
        _make_type(db)
        index = db.create_vector_index(
            "Doc", "embedding", dimensions=3, quantization="PRODUCT", pq_clusters=4
        )
        _insert(db, rows)
        index.build_graph_now()
        return index

    def test_returns_the_nearest_record_and_its_distance(self, temp_db):
        index = self._pq_index(temp_db, PQ_VECTORS)

        results = index.find_nearest_approximate([0.9, 0.1, 0.0], k=1)

        assert len(results) == 1
        record, distance = results[0]
        assert record.get("name") == "x"
        assert isinstance(distance, float)

    def test_k_results_come_back_nearest_first(self, temp_db):
        index = self._pq_index(temp_db, PQ_VECTORS)

        results = index.find_nearest_approximate([0.9, 0.1, 0.0], k=3)

        assert [r.get("name") for r, _ in results][0] == "x"
        assert len(results) == 3
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_allowed_rids_restrict_the_answer(self, temp_db):
        index = self._pq_index(temp_db, PQ_VECTORS)
        rids = {
            row.get("name"): row.get("rid")
            for row in temp_db.query("sql", "SELECT name, @rid AS rid FROM Doc")
        }

        results = index.find_nearest_approximate(
            [0.9, 0.1, 0.0], k=3, allowed_rids=[str(rids["y"]), str(rids["z"])]
        )

        names = {r.get("name") for r, _ in results}
        assert names <= {"y", "z"} and names

    def test_a_typeindex_wrapper_from_the_schema_searches_too(self, temp_db):
        self._pq_index(temp_db, PQ_VECTORS)
        index = temp_db.schema.get_vector_index("Doc", "embedding")

        results = index.find_nearest_approximate([0.0, 0.9, 0.1], k=1)

        assert [r.get("name") for r, _ in results] == ["y"]

    def test_without_product_quantization_it_raises(self, temp_db):
        _make_type(temp_db)
        index = temp_db.create_vector_index("Doc", "embedding", dimensions=3)
        _insert(temp_db, UNIT_VECTORS)
        index.build_graph_now()

        with pytest.raises(arcadedb.ArcadeDBError):
            index.find_nearest_approximate([0.9, 0.1, 0.0], k=1)


class TestRebuildThreshold:
    """``mutations_before_rebuild`` is the FLOOR of the rebuild threshold (#110).

    The engine rebuilds after max(mutations_before_rebuild, min(graph size x rebuildGraphRatio, maxPendingMutations))
    changes; with the defaults (ratio 0.2, cap 50000) a 1000-vector graph gives 200 for any floor up to 200. The
    vectors are distinct and nonzero (cosine is the default similarity).
    """

    ROWS = [
        (f"v{i}", [1.0 + i % 10, 1.0 + i // 10 % 10, 1.0 + i // 100])
        for i in range(1000)
    ]

    @pytest.mark.parametrize("floor, expected", [(10, 200), (100, 200), (5000, 5000)])
    def test_the_threshold_is_the_larger_of_the_floor_and_the_ratio_term(
        self, temp_db, floor, expected
    ):
        _make_type(temp_db)
        _insert(temp_db, self.ROWS)
        index = temp_db.create_vector_index(
            "Doc", "embedding", dimensions=3, mutations_before_rebuild=floor
        )

        stats = index.get_stats()
        assert stats["totalVectors"] == len(self.ROWS)
        assert stats["effectiveMutationsThreshold"] == expected

    def test_alter_database_sets_the_ratio_the_threshold_reads(self, temp_db):
        # the route the docs give for a database-scope setting (#112): SQL, no Python wrapper
        _make_type(temp_db)
        _insert(temp_db, self.ROWS)
        index = temp_db.create_vector_index("Doc", "embedding", dimensions=3)
        assert index.get_stats()["effectiveMutationsThreshold"] == 200

        temp_db.command(
            "sql", "ALTER DATABASE `arcadedb.vectorIndex.rebuildGraphRatio` 0.5"
        )

        assert index.get_stats()["effectiveMutationsThreshold"] == 500


class TestWrapperAcrossRebuildAndDrop:
    """A ``VectorIndex`` held across SQL ``REBUILD INDEX`` keeps serving; across a drop it raises (#116).

    ``REBUILD INDEX`` replaces the engine's index object. The JVM wrapper used to keep the old one, which answered
    ``find_nearest`` with no rows and ``totalVectors`` 0 without an error.
    """

    ROWS = [(f"v{i}", [1.0 + i % 7, 1.0 + i // 7, 2.0 + i % 3]) for i in range(40)]

    def _held_index(self, db):
        _make_type(db)
        _insert(db, self.ROWS)
        index = db.create_vector_index("Doc", "embedding", dimensions=3)
        assert len(index.find_nearest([1.0, 1.0, 2.0], k=5)) == 5
        assert index.get_stats()["totalVectors"] == len(self.ROWS)
        return index

    def test_old_wrapper_serves_after_rebuild_index(self, temp_db):
        index = self._held_index(temp_db)
        quantization = index.get_quantization()  # create_vector_index defaults to INT8

        temp_db.command("sql", "REBUILD INDEX `Doc[embedding]`")

        assert len(index.find_nearest([1.0, 1.0, 2.0], k=5)) == 5
        assert index.get_stats()["totalVectors"] == len(self.ROWS)
        assert index.get_size() == len(self.ROWS)
        assert index.get_metadata()["index_name"] == "Doc[embedding]"
        assert index.get_quantization() == quantization
        index.warm_up()

    def test_old_wrapper_follows_a_second_rebuild(self, temp_db):
        index = self._held_index(temp_db)

        for _ in range(2):
            temp_db.command("sql", "REBUILD INDEX `Doc[embedding]`")
            assert len(index.find_nearest([1.0, 1.0, 2.0], k=5)) == 5
            assert index.get_stats()["totalVectors"] == len(self.ROWS)

    def test_dropped_index_raises_and_never_answers_empty(self, temp_db):
        index = self._held_index(temp_db)

        temp_db.command("sql", "DROP INDEX `Doc[embedding]`")

        with pytest.raises(arcadedb.ArcadeDBError) as excinfo:
            index.find_nearest([1.0, 1.0, 2.0], k=5)
        if arcadedb.__backend__ == "jvm":
            # the JVM wrapper names the index itself; the native one passes the engine's own error on
            assert "Doc[embedding]" in str(excinfo.value)
            with pytest.raises(arcadedb.ArcadeDBError, match=r"Doc\[embedding\]"):
                index.get_stats()
            with pytest.raises(arcadedb.ArcadeDBError, match=r"Doc\[embedding\]"):
                index.get_size()
