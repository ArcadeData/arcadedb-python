import pytest

import arcadedb


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db_path = str(tmp_path / "test_vector_params_db")
    db = arcadedb.create_database(db_path)
    yield db
    db.drop()


class TestVectorParams:
    """Verify that vector index parameters are correctly passed to Java."""

    @staticmethod
    def _get_primary_metadata(index):
        java_index = index._java_index
        idx_to_check = java_index
        if "TypeIndex" in java_index.getClass().getName():
            idx_to_check = java_index.getSubIndexes().get(0)
        return idx_to_check.getMetadata()

    def test_quantization_param(self, test_db):
        """Test sending quantization parameter."""
        test_db.command("sql", "CREATE VERTEX TYPE QuantDoc")
        test_db.command("sql", "CREATE PROPERTY QuantDoc.embedding ARRAY_OF_FLOATS")

        # Create with INT8
        index = test_db.create_vector_index(
            "QuantDoc", "embedding", dimensions=3, quantization="INT8"
        )

        # Verify via wrapper convenience method
        assert index.get_quantization() == "INT8"

        # Verify through the engine's own metadata of the index
        assert index.get_metadata()["quantization"] == "INT8"

    def test_store_vectors_in_graph_param(self, test_db):
        """Test sending store_vectors_in_graph parameter."""
        test_db.command("sql", "CREATE VERTEX TYPE StoreDoc")
        test_db.command("sql", "CREATE PROPERTY StoreDoc.embedding ARRAY_OF_FLOATS")

        # Create with store_vectors_in_graph=True
        index = test_db.create_vector_index(
            "StoreDoc", "embedding", dimensions=3, store_vectors_in_graph=True
        )

        assert index.get_metadata()["store_vectors_in_graph"] is True

        # and the default stays off
        test_db.command("sql", "CREATE VERTEX TYPE NoStoreDoc")
        test_db.command("sql", "CREATE PROPERTY NoStoreDoc.embedding ARRAY_OF_FLOATS")
        plain = test_db.create_vector_index("NoStoreDoc", "embedding", dimensions=3)
        assert plain.get_metadata()["store_vectors_in_graph"] is False

    def test_encoding_param(self, test_db):
        """Test sending encoding parameter for native INT8 storage."""
        test_db.command("sql", "CREATE VERTEX TYPE EncodingDoc")
        test_db.command("sql", "CREATE PROPERTY EncodingDoc.embedding BINARY")

        index = test_db.create_vector_index(
            "EncodingDoc",
            "embedding",
            dimensions=4,
            quantization="NONE",
            encoding="INT8",
        )

        metadata = self._get_primary_metadata(index)
        assert str(metadata.encoding) == "INT8"
        assert str(metadata.quantizationType) == "NONE"

    def test_encoding_int8_rejects_default_int8_quantization(self, test_db):
        """encoding=INT8 should reject the default INT8 quantization combination."""
        test_db.command("sql", "CREATE VERTEX TYPE EncodingGuardDoc")
        test_db.command("sql", "CREATE PROPERTY EncodingGuardDoc.embedding BINARY")

        with pytest.raises(
            arcadedb.ArcadeDBError,
            match="encoding='INT8'.*quantization='INT8'",
        ):
            test_db.create_vector_index(
                "EncodingGuardDoc",
                "embedding",
                dimensions=4,
                encoding="INT8",
            )

    def test_add_hierarchy_param(self, test_db):
        """Test sending add_hierarchy parameter."""
        test_db.command("sql", "CREATE VERTEX TYPE HierDoc")
        test_db.command("sql", "CREATE PROPERTY HierDoc.embedding ARRAY_OF_FLOATS")

        index = test_db.create_vector_index(
            "HierDoc", "embedding", dimensions=3, add_hierarchy=True
        )

        assert index.get_metadata()["add_hierarchy"] is True

        test_db.command("sql", "CREATE VERTEX TYPE FlatDoc")
        test_db.command("sql", "CREATE PROPERTY FlatDoc.embedding ARRAY_OF_FLOATS")
        flat = test_db.create_vector_index(
            "FlatDoc", "embedding", dimensions=3, add_hierarchy=False
        )
        assert flat.get_metadata()["add_hierarchy"] is False

    def test_params_persistence(self, tmp_path):
        """Verify parameters persist after reload."""
        db_path = str(tmp_path / "test_vector_params_persist")

        # 1. Create and Configure
        with arcadedb.create_database(db_path) as db:
            db.command("sql", "CREATE VERTEX TYPE Doc")
            db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")

            db.create_vector_index(
                "Doc",
                "embedding",
                dimensions=3,
                quantization="INT8",
                store_vectors_in_graph=True,
                add_hierarchy=True,
            )

        # 2. Reopen and Check
        with arcadedb.open_database(db_path) as db:
            index_rows = db.query(
                "sql", "SELECT name FROM schema:indexes WHERE typeName = 'Doc'"
            ).to_list()
            assert len(index_rows) > 0

            rs = db.query(
                "sql",
                "SELECT vectorNeighbors('Doc[embedding]', [1.0, 0.0, 0.0], 1) as res",
            ).to_list()
            assert len(rs) == 1
            neighbors = rs[0].get("res")
            assert neighbors is not None

    def test_per_index_cache_params(self, test_db):
        """Test per-index cache and rebuild overrides."""
        test_db.command("sql", "CREATE VERTEX TYPE CacheDoc")
        test_db.command("sql", "CREATE PROPERTY CacheDoc.embedding ARRAY_OF_FLOATS")

        # location_cache_size was removed by the engine (#5559, #5568). The
        # binding must reject it with an explanation rather than forward it and
        # let index creation blow up inside withMetadata().
        with pytest.raises(ValueError, match="no longer supported"):
            test_db.create_vector_index(
                "CacheDoc", "embedding", dimensions=4, location_cache_size=123
            )

        index = test_db.create_vector_index(
            "CacheDoc",
            "embedding",
            dimensions=4,
            graph_build_cache_size=456,
            mutations_before_rebuild=789,
        )

        metadata = index.get_metadata()
        assert metadata["graph_build_cache_size"] == 456
        assert metadata["mutations_before_rebuild"] == 789

    def test_quantization_none(self, test_db):
        """Test sending quantization parameter NONE."""
        test_db.command("sql", "CREATE VERTEX TYPE QuantNoneDoc")
        test_db.command("sql", "CREATE PROPERTY QuantNoneDoc.embedding ARRAY_OF_FLOATS")

        index = test_db.create_vector_index(
            "QuantNoneDoc", "embedding", dimensions=3, quantization="NONE"
        )

        assert index.get_quantization() == "NONE"

        assert index.get_metadata()["quantization"] == "NONE"

    def test_quantization_binary(self, test_db):
        """Test sending quantization parameter BINARY."""
        test_db.command("sql", "CREATE VERTEX TYPE QuantBinaryDoc")
        test_db.command(
            "sql", "CREATE PROPERTY QuantBinaryDoc.embedding ARRAY_OF_FLOATS"
        )

        index = test_db.create_vector_index(
            "QuantBinaryDoc", "embedding", dimensions=128, quantization="BINARY"
        )

        assert index.get_quantization() == "BINARY"

        assert index.get_metadata()["quantization"] == "BINARY"

    def test_quantization_product(self, test_db):
        """Test sending quantization parameter PRODUCT (PQ)."""
        test_db.command("sql", "CREATE VERTEX TYPE QuantProductDoc")
        test_db.command(
            "sql", "CREATE PROPERTY QuantProductDoc.embedding ARRAY_OF_FLOATS"
        )

        index = test_db.create_vector_index(
            "QuantProductDoc", "embedding", dimensions=64, quantization="PRODUCT"
        )

        assert index.get_quantization() == "PRODUCT"

        assert index.get_metadata()["quantization"] == "PRODUCT"

    @staticmethod
    def _engine_default_max_connections(db) -> int:
        """Read maxConnections off an index the engine configured by itself.

        create_vector_index always sets maxConnections, so the engine default
        shows only on an index SQL creates without it. The value tracks the
        engine the wheel carries rather than a number copied into the Python
        layer, and it reads the same on both builds.
        """
        db.command("sql", "CREATE VERTEX TYPE DefaultProbe")
        db.command("sql", "CREATE PROPERTY DefaultProbe.embedding ARRAY_OF_FLOATS")
        db.command(
            "sql",
            "CREATE INDEX ON DefaultProbe (embedding) LSM_VECTOR METADATA "
            '{"dimensions": 4, "similarity": "EUCLIDEAN"}',
        )
        probe = db.schema.get_vector_index("DefaultProbe", "embedding")
        return int(probe.get_metadata()["max_connections"])

    def test_wrapper_default_matches_engine_default(self, test_db):
        """The Python default must equal the engine default, not a copy of it.

        create_vector_index always calls withMaxConnections, so the wrapper
        default fully shadows the engine default for callers who omit it.
        """
        import inspect

        signature = inspect.signature(type(test_db).create_vector_index)
        wrapper_default = signature.parameters["max_connections"].default

        assert wrapper_default == self._engine_default_max_connections(test_db)

    def test_omitted_max_connections_reaches_the_index(self, test_db):
        """Omitting the argument must land the engine default on the index."""
        engine_default = self._engine_default_max_connections(test_db)

        test_db.command("sql", "CREATE VERTEX TYPE DefaultDegreeDoc")
        test_db.command(
            "sql", "CREATE PROPERTY DefaultDegreeDoc.embedding ARRAY_OF_FLOATS"
        )

        index = test_db.create_vector_index(
            "DefaultDegreeDoc", "embedding", dimensions=4
        )

        assert index.get_metadata()["max_connections"] == engine_default

    def test_jvm_heap_check(self):
        """Verify JVM memory settings from Java level."""
        import jpype

        runtime = jpype.JPackage("java.lang").Runtime.getRuntime()
        max_memory = runtime.maxMemory()
        total_memory = runtime.totalMemory()
        free_memory = runtime.freeMemory()

        print(f"\n=== JVM Memory Stats ===")
        print(f"Max Memory:   {max_memory / (1024**3):.2f} GB ({max_memory} bytes)")
        print(f"Total Memory: {total_memory / (1024**2):.2f} MB")
        print(f"Free Memory:  {free_memory / (1024**2):.2f} MB")

        # Verify it's a reasonable size (at least 1GB, reflecting -Xmx4g default)
        assert max_memory > 1 * 1024 * 1024 * 1024
