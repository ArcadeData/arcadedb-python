"""GraphBatch.create_vertices (the JVM wheel) decides its rows by their types and writes each chunk once.

Before, `_create_vertices_json_bulk` called `json_bulk_scalar_ok` on every value of every row before the first chunk was written, and
the chunk was then walked again by json.dumps. Now `type_conversion.classify_vertex_rows` takes a census of the types in C, and the
value checks come from the one json.dumps pass of each chunk (`json_bulk_dumps`, as `json_bulk_dumps_tagged` already did for the tagged
text). The rule that keeps the stored type of a property from depending on which chunk held a bad value: an input of one chunk is decided
by its own dumps before anything is sent; an input of several chunks first has its columns of exact scalars screened in C over the whole
input (`classify_vertex_rows(..., screen_scalars=True)`), and a bad scalar (a NaN, an integer beyond 64 bits, a lone surrogate, a
subnormal) anywhere sends the WHOLE input the property-matrix way, as it did before. A name that is not a str or a type the text does not
carry does the same (`_create_vertices_json_bulk` returns None).
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


@pytest.mark.parametrize("chunk", [100_000, 500], ids=["one_chunk", "screened"])
def test_scalar_rows_are_not_judged_value_by_value(temp_db_path, monkeypatch, chunk):
    def refuse(value):
        raise AssertionError("a per-value gate ran over a scalar table")

    monkeypatch.setattr(GraphBatch, "_BULK_CHUNK", chunk)
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
@pytest.mark.parametrize("chunks", [1, 3], ids=["one_chunk", "three_chunks"])
def test_a_scalar_the_text_changes_sends_the_whole_input_through_the_matrix_path(
    temp_db_path, monkeypatch, value, chunks
):
    monkeypatch.setattr(GraphBatch, "_BULK_CHUNK", 100 if chunks == 3 else 1_000)
    calls = []
    real = GraphBatch._to_java_property_matrix

    def spy(self, rows):
        calls.append(len(rows))
        return real(self, rows)

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
        assert calls == [250], "the whole input takes the matrix path, once"
        if value == 5e-324:
            # stored as the double every other entry point stores; the text of a subnormal was read as a DECIMAL
            assert len(rids) == 250
            stored = db.query("sql", "SELECT x FROM V WHERE id = 150").first().get("x")
            assert stored == 5e-324 and type(stored) is float


@pytest.mark.parametrize(
    "bad_row", [5, 150_000, 249_999], ids=["first_chunk", "middle_chunk", "last_chunk"]
)
def test_the_stored_type_does_not_depend_on_which_chunk_held_a_bad_value(
    temp_db_path, monkeypatch, bad_row
):
    """An undeclared int property is stored as the same Java type in every chunk, whichever chunk holds a NaN in another property
    (the 100,000-row chunking of the real class): the NaN sends the whole input the matrix way, so no chunk is stored from the JSON text.
    """
    n = 250_000
    rows = [{"id": i, "k": i % 1000} for i in range(n)]
    rows[bad_row] = {"id": bad_row, "k": 1, "x": float("nan")}
    probes = {7: "chunk 1", 150_001: "chunk 2", 249_999: "chunk 3"}
    probes.pop(bad_row, None)
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
        assert calls == [n], "the whole input takes the matrix path, once"
        types = _stored_types(db, probes)
        assert (
            len(set(types.values())) == 1
        ), f"the stored type depends on the chunk: {types}"


def test_a_clean_input_of_several_chunks_is_stored_alike_in_every_chunk(
    temp_db_path, monkeypatch
):
    n = 250_000
    rows = [{"id": i, "k": i % 1000} for i in range(n)]
    probes = {7: "chunk 1", 150_001: "chunk 2", 249_999: "chunk 3"}
    calls = []
    real = GraphBatch._to_java_property_matrix
    monkeypatch.setattr(
        GraphBatch,
        "_to_java_property_matrix",
        lambda self, chunk: calls.append(len(chunk)) or real(self, chunk),
    )
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE V")
        with db.graph_batch(parallel_flush=False) as batch:
            assert len(batch.create_vertices("V", rows)) == n
        assert calls == [], "a clean input never takes the matrix path"
        types = _stored_types(db, probes)
        assert (
            len(set(types.values())) == 1
        ), f"the stored type depends on the chunk: {types}"


def _stored_types(db, probes):
    ids = ", ".join(str(i) for i in probes)
    query = f"SELECT id, k, k.javaType() AS t FROM V WHERE id IN [{ids}]"  # nosec B608
    got = db.query("sql", query).to_list()
    by_id = {r["id"]: (r["k"], str(r["t"])) for r in got}
    assert set(by_id) == set(probes)
    assert {i: by_id[i][0] for i in probes} == {i: i % 1000 for i in probes}
    return {probes[i]: by_id[i][1] for i in probes}
