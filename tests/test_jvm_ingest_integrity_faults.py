"""The JVM wheel's column conversion and bulk vertex path with a rows-dropped fault in them (ArcadeData/arcadedb-python#336).

``arcadedb.core._column_to_java`` converts every column of ``insert_columns`` (and of ``insert_arrow`` and ``insert_dataframe``,
which call it), and ``GraphBatch._create_vertices_json_bulk`` carries the rows of ``create_vertices``. Each test puts a fault into
one of them and runs the same checks as ``test_ingest_integrity.py``: they must report it. A file about the JPype path, so
``NATIVE_NOT_APPLICABLE`` lists it.
"""

import pytest

import arcadedb
from arcadedb import core
from tests.test_ingest_integrity import (
    BY_ID,
    FOUND,
    MUTATION_SIZE,
    IntegrityError,
    run_cases,
)


@pytest.mark.parametrize("family", ["cols", "arrow", "df"])
def test_a_column_converted_one_row_short_is_reported(
    store, small_chunks, monkeypatch, family
):
    real = core._column_to_java
    seen = []

    def faulty(name, values):
        seen.append(name)
        return real(name, values[:-1])

    monkeypatch.setattr(core, "_column_to_java", faulty)
    with pytest.raises(IntegrityError) as found:
        run_cases(store, BY_ID[family], [MUTATION_SIZE])
    assert FOUND in str(found.value), found.value
    assert seen, "the fault was not in the path"


def test_a_bulk_vertex_chunk_one_row_short_is_reported(
    store, small_chunks, monkeypatch
):
    real = arcadedb.GraphBatch._create_vertices_json_bulk
    seen = []

    def faulty(self, type_name, rows):
        seen.append(len(rows))
        return real(self, type_name, rows[:-1])

    monkeypatch.setattr(arcadedb.GraphBatch, "_create_vertices_json_bulk", faulty)
    with pytest.raises(IntegrityError) as found:
        run_cases(store, BY_ID["vertices"], [MUTATION_SIZE])
    assert FOUND in str(found.value), found.value
    assert seen, "the fault was not in the path"
