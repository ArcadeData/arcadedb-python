"""GraphBatch.create_vertices (the JVM wheel) decides its rows by their types and writes each chunk once.

Before, `_create_vertices_json_bulk` called `json_bulk_scalar_ok` on every value of every row before the first chunk was written, and
the chunk was then walked again by json.dumps. Now `type_conversion.classify_vertex_rows` takes a census of the types in C, and the
value checks come from the one json.dumps pass of each chunk (`json_bulk_dumps`, as `json_bulk_dumps_tagged` already did for the tagged
text). What stays: the whole input goes the property-matrix way for a property name that is not a str or a type the text does not carry
(`_create_vertices_json_bulk` returns None); what changes: a scalar the text would change (a subnormal, a NaN, an integer beyond 64 bits,
a lone surrogate) sends its own chunk, not the whole input, through the matrix path, as it already did for a tagged chunk.
"""

import pytest

import arcadedb
from arcadedb import graph_batch as gbm
from arcadedb import type_conversion as tc
from arcadedb.graph_batch import GraphBatch


def _rows(n):
    return [
        {"id": i, "name": f"n{i}", "score": i * 0.5, "flag": i % 2 == 0, "note": None}
        for i in range(n)
    ]


def test_scalar_rows_are_not_judged_value_by_value(temp_db_path, monkeypatch):
    def refuse(value):
        raise AssertionError("a per-value gate ran over a scalar table")

    monkeypatch.setattr(tc, "json_bulk_scalar_ok", refuse)
    monkeypatch.setattr(gbm, "json_bulk_scalar_ok", refuse)
    monkeypatch.setattr(tc, "json_bulk_shape_ok", refuse)
    monkeypatch.setattr(gbm, "json_bulk_shape_ok", refuse)
    rows = _rows(2_000)
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE V")
        with db.graph_batch(parallel_flush=False) as batch:
            assert len(batch._create_vertices_json_bulk("V", rows)) == len(rows)
        count = db.query("sql", "SELECT count(*) AS c FROM V").first().get("c")
        assert count == len(rows)


def test_the_whole_input_still_goes_the_matrix_way_for_a_name_or_type_the_text_does_not_carry(
    temp_db_path,
):
    rows = _rows(5)
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE V")
        with db.graph_batch(parallel_flush=False) as batch:
            assert len(batch._create_vertices_json_bulk("V", rows)) == len(rows)
            assert batch._create_vertices_json_bulk("V", rows + [{1: "a"}]) is None
            assert batch._create_vertices_json_bulk("V", rows + [{"k": {1, 2}}]) is None


@pytest.mark.parametrize("value", [5e-324, float("nan"), 2**70, "a\ud800b"])
def test_a_scalar_the_text_changes_sends_its_chunk_through_the_matrix_path(
    temp_db_path, monkeypatch, value
):
    monkeypatch.setattr(GraphBatch, "_BULK_CHUNK", 100)
    calls = []
    real = GraphBatch._to_java_property_matrix

    def spy(self, chunk):
        calls.append(len(chunk))
        return real(self, chunk)

    monkeypatch.setattr(GraphBatch, "_to_java_property_matrix", spy)
    rows = [{"id": i, "x": 1.5} for i in range(250)]
    rows[150] = {"id": 150, "x": value}
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE V")
        with db.graph_batch(parallel_flush=False) as batch:
            try:
                rids = batch.create_vertices("V", rows)
            except (
                Exception
            ):  # noqa: BLE001 - the matrix path refuses a value that no entry point stores
                rids = None
        assert calls == [100], "only the chunk of row 150 takes the matrix path"
        if value == 5e-324:
            # stored as the double every other entry point stores; the text of a subnormal was read as a DECIMAL
            assert len(rids) == 250
            stored = db.query("sql", "SELECT x FROM V WHERE id = 150").first().get("x")
            assert stored == 5e-324 and type(stored) is float


def test_the_stored_type_does_not_depend_on_which_chunk_held_a_bad_value(
    temp_db_path, monkeypatch
):
    """An undeclared int property is stored as the same type in the chunks that crossed as JSON text and in the chunk that took the
    property-matrix path because a NaN sat in another property of one of its rows (the 100,000-row chunking of the real class).
    """
    n = 250_000
    rows = [{"id": i, "k": i % 1000} for i in range(n)]
    rows[150_000] = {"id": 150_000, "k": 1, "x": float("nan")}  # chunk 2 of 3 only
    probes = {5: "chunk 1", 150_001: "chunk 2", 249_999: "chunk 3"}
    calls = []
    real = GraphBatch._to_java_property_matrix

    def spy(self, chunk):
        calls.append(len(chunk))
        return real(self, chunk)

    monkeypatch.setattr(GraphBatch, "_to_java_property_matrix", spy)
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE V")
        with db.graph_batch(parallel_flush=False) as batch:
            assert len(batch.create_vertices("V", rows)) == n
        assert calls == [100_000], "only the chunk with the NaN takes the matrix path"
        ids = ", ".join(str(i) for i in probes)
        query = (
            f"SELECT id, k, k.javaType() AS t FROM V WHERE id IN [{ids}]"  # nosec B608
        )
        got = db.query("sql", query).to_list()
        by_id = {r["id"]: (r["k"], str(r["t"])) for r in got}
        assert set(by_id) == set(probes)
        assert {i: by_id[i][0] for i in probes} == {i: i % 1000 for i in probes}
        types = {probes[i]: by_id[i][1] for i in probes}
        assert (
            len(set(types.values())) == 1
        ), f"the stored type depends on the chunk: {types}"
