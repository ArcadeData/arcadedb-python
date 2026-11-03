"""GraphBatch.create_vertices and new_edges keep a list, tuple, dict, date-time, date, Decimal or bytes property on the bulk path.

Those values made ``json_bulk_scalar_ok`` refuse the whole call, which sent every row through ``_to_java_property_matrix`` (one JPype
conversion per element, measured 91.4 against 9.3 us per row for a list property). They now cross inside the JSON text as the tagged
values ``json_bulk_dumps_tagged`` writes (insert_many's format) and the bridge classes ``VertexBatcher`` and ``EdgeBatcher`` decode with
``DocumentBatcher.untag``. Each test stores the same rows through the property-matrix path (``_create_vertices_json_bulk`` switched
off, or ``new_edge`` per edge) and through the bulk path, compares the stored values AND their Python types, and spies on the
fallback entry points to prove which path ran. A value the text cannot carry (NaN, an integer beyond 64 bits, a date-time inside a list or
map, a subnormal, a non-str dict key) or a type the tags do not carry (a set, a numpy scalar) still takes the old path.
"""

import datetime as dt
import decimal
import math

import numpy as np
import pytest

import arcadedb
from arcadedb import graph_batch as gbm
from arcadedb import type_conversion
from arcadedb.graph_batch import GraphBatch

UTC9 = dt.timezone(dt.timedelta(hours=9))

# name -> value; each is stored by the property-matrix path (convert_python_to_java), so the bulk path has a stored value to match
VALUES = {
    "list_strings": ["a", "b", "é中\U0001f600", ""],
    "list_mixed": [1, 2.5, "x", None, True],
    "list_nested": [[1, 2], [3.5], []],
    "list_empty": [],
    "dict_flat": {"a": 1, "b": "two", "c": 3.5, "d": None, "e": False},
    "dict_nested": {"a": [1, 2], "b": {"c": "x", "d": [0.5]}},
    "dict_empty": {},
    "naive_dt": dt.datetime(2026, 5, 6, 7, 8, 9, 123456),
    "aware_dt": dt.datetime(2026, 5, 6, 7, 8, 9, 654321, tzinfo=UTC9),
    "dt_whole_second": dt.datetime(2026, 5, 6, 7, 8, 9),
    "date": dt.date(2026, 5, 6),
    "decimal": decimal.Decimal("12345.6789"),
    "bytes": b"\x00\xff abc\x7f",
    "plain_int": 7,
    "plain_str": "text",
}

# lists and tuples of nothing but numbers keep main's property-matrix path (one double[] per list, about 3.8 times faster than the text)
NUMERIC_LISTS = {
    "list_floats": [0.1, 1 / 3, 2.5e-3, 1e300, -0.0, 7.0],
    "list_floats_long": [i * 0.37 for i in range(300)],
    "list_ints": [1, 2, 3, 2**40, -(2**63)],
    "list_int_float": [1, 2.5],
    "tuple_ints": (4, 5, 6),
}

# a declared property per value where the type matters, so the engine's conversion to the declared type is compared too
DECLARED = {
    "list_floats": "LIST",
    "list_strings": "LIST",
    "dict_flat": "MAP",
    "naive_dt": "DATETIME_MICROS",
    "aware_dt": "DATETIME_NANOS",
    "dt_whole_second": "DATETIME",
    "date": "DATE",
    "decimal": "DECIMAL",
    "bytes": "BINARY",
}


@pytest.fixture
def matrix_calls(monkeypatch):
    """Count the property-matrix path: the only caller of GraphBatch._to_java_property_matrix."""
    calls = []
    real = GraphBatch._to_java_property_matrix

    def counting(self, rows):
        calls.append(None)
        return real(self, rows)

    monkeypatch.setattr(GraphBatch, "_to_java_property_matrix", counting)
    return calls


@pytest.fixture
def edge_calls(monkeypatch):
    """Count the per-edge path: GraphBatch.new_edge, which new_edges falls back to."""
    calls = []
    real = GraphBatch.new_edge

    def counting(self, *args, **kwargs):
        calls.append(None)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(GraphBatch, "new_edge", counting)
    return calls


def _schema(db, declared):
    for name in ("Ref", "Bulk"):
        db.command("sql", f"CREATE VERTEX TYPE {name}")
        db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")
        if declared:
            for prop, ty in DECLARED.items():
                db.command("sql", f"CREATE PROPERTY {name}.{prop} {ty}")
    for name in ("ERef", "EBulk"):
        db.command("sql", f"CREATE EDGE TYPE {name}")
        db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")
        if declared:
            for prop, ty in DECLARED.items():
                db.command("sql", f"CREATE PROPERTY {name}.{prop} {ty}")


def _stored(db, type_name, names):
    """Every record by id as {property: (python type name, repr)}, so a changed type is as visible as a changed value."""
    out = []
    sql = f"SELECT FROM {type_name} ORDER BY id"  # nosec B608 - test-owned type name
    for rec in db.query("sql", sql).to_list():
        out.append(
            {
                k: (type(v).__name__, repr(v))
                for k, v in sorted(rec.items())
                if k in names
            }
        )
    return out


def _names(rows):
    return {k for row in rows for k in row}


def _rows(names):
    return [{"id": i, name: VALUES[name]} for i, name in enumerate(names)]


def _vertices(db, type_name, rows):
    with db.graph_batch(parallel_flush=False) as batch:
        return batch.create_vertices(type_name, rows)


def _reference_vertices(monkeypatch, db, type_name, rows):
    """The reference: create_vertices without its JSON bulk path, which is the path it took for these values before."""
    with monkeypatch.context() as mp:
        mp.setattr(GraphBatch, "_create_vertices_json_bulk", lambda self, t, r: None)
        return _vertices(db, type_name, rows)


def _edges(db, edge_type, rows, per_edge=False):
    with db.transaction():
        src = [db.new_vertex("Bulk").save().get_rid()]
        dst = [db.new_vertex("Bulk").save().get_rid()]
    with db.graph_batch(parallel_flush=False) as batch:
        if per_edge:
            for row in rows:
                batch.new_edge(src[0], edge_type, dst[0], **row)
        else:
            batch.new_edges(src * len(rows), edge_type, dst * len(rows), rows)


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "schemaless"])
def test_vertices_every_shape_is_stored_as_the_matrix_path_stores_it(
    temp_db_path, monkeypatch, matrix_calls, declared
):
    rows = _rows(list(VALUES))
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, declared)
        _reference_vertices(monkeypatch, db, "Ref", rows)
        assert len(matrix_calls) == 1
        matrix_calls.clear()
        rids = _vertices(db, "Bulk", rows)
        assert matrix_calls == [], "the chunk left the bulk path"
        assert len(rids) == len(rows) and all(r.startswith("#") for r in rids)
        names = _names(rows)
        assert _stored(db, "Bulk", names) == _stored(db, "Ref", names)
        assert len(_stored(db, "Bulk", names)) == len(rows)


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "schemaless"])
def test_vertices_one_row_with_every_shape_at_once(
    temp_db_path, monkeypatch, matrix_calls, declared
):
    row = {"id": 0, **VALUES}
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, declared)
        _reference_vertices(monkeypatch, db, "Ref", [row, None, {}])
        matrix_calls.clear()
        _vertices(db, "Bulk", [row, None, {}])
        assert matrix_calls == []
        names = set(row)
        assert _stored(db, "Bulk", names)[0] == _stored(db, "Ref", names)[0]


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "schemaless"])
def test_edges_every_shape_is_stored_as_the_per_edge_path_stores_it(
    temp_db_path, edge_calls, declared
):
    rows = _rows(list(VALUES))
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, declared)
        _edges(db, "ERef", rows, per_edge=True)
        assert len(edge_calls) == len(rows)
        edge_calls.clear()
        _edges(db, "EBulk", rows)
        assert edge_calls == [], "the edges left the bulk path"
        names = _names(rows)
        assert _stored(db, "EBulk", names) == _stored(db, "ERef", names)
        assert len(_stored(db, "EBulk", names)) == len(rows)


# a value the text cannot carry, or whose type the tags do not carry: the old path takes it, and stores what it always stored
OFF_THE_BULK_PATH = {
    "nan_in_list": ["x", math.nan],
    "nan_in_number_list": [1.0, math.nan],
    "inf_in_dict": {"a": math.inf},
    "subnormal_in_list": ["x", 5e-324],
    "datetime_in_list": [dt.datetime(2026, 5, 6, 7, 8, 9)],
    "datetime_in_dict": {"when": dt.datetime(2026, 5, 6, 7, 8, 9)},
    "int_key_dict": {1: "a"},
    "tag_marker_dict": {"\u0001": ["T", "2026-05-06T00:00:00"]},
    "nan_scalar": math.nan,
}


@pytest.mark.parametrize("name", list(OFF_THE_BULK_PATH))
def test_vertices_a_value_the_text_cannot_carry_keeps_the_old_path(
    temp_db_path, monkeypatch, matrix_calls, name
):
    value = OFF_THE_BULK_PATH[name]
    rows = [{"id": 0, "k": ["a", "b"]}, {"id": 1, "k": value}, {"id": 2, "k": "x"}]
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        _reference_vertices(monkeypatch, db, "Ref", rows)
        matrix_calls.clear()
        rids = _vertices(db, "Bulk", rows)
        assert len(matrix_calls) >= 1, "the value went through the JSON text"
        assert len(rids) == 3
        assert _stored(db, "Bulk", {"id", "k"}) == _stored(db, "Ref", {"id", "k"})


@pytest.mark.parametrize("name", list(OFF_THE_BULK_PATH))
def test_edges_a_value_the_text_cannot_carry_keeps_the_old_path(
    temp_db_path, edge_calls, name
):
    value = OFF_THE_BULK_PATH[name]
    rows = [{"id": 0, "k": ["a", "b"]}, {"id": 1, "k": value}, {"id": 2, "k": "x"}]
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        _edges(db, "ERef", rows, per_edge=True)
        edge_calls.clear()
        _edges(db, "EBulk", rows)
        assert len(edge_calls) == 3, "the edges went through the JSON text"
        assert _stored(db, "EBulk", {"id", "k"}) == _stored(db, "ERef", {"id", "k"})
        count = db.query("sql", "SELECT count(*) AS c FROM EBulk").first().get("c")
        assert count == 3


def test_edges_a_late_value_the_text_cannot_carry_buffers_each_edge_once(
    temp_db_path, edge_calls, monkeypatch
):
    """Every chunk is written before the first crosses: a bad value in the last chunk starts the per-edge path over with nothing buffered."""
    monkeypatch.setattr(GraphBatch, "_BULK_CHUNK", 3)
    rows = [{"id": i, "k": [f"s{i}", "z"]} for i in range(8)]
    rows[7]["k"] = ["x", math.nan]
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        _edges(db, "EBulk", rows)
        assert len(edge_calls) == 8
        count = db.query("sql", "SELECT count(*) AS c FROM EBulk").first().get("c")
        assert count == 8


def test_vertices_only_the_chunk_with_the_bad_value_leaves_the_bulk_path(
    temp_db_path, monkeypatch, matrix_calls
):
    monkeypatch.setattr(GraphBatch, "_BULK_CHUNK", 3)
    rows = [{"id": i, "k": [f"s{i}", "z"], "d": {"x": i}} for i in range(8)]
    rows[4]["k"] = ["x", math.nan]
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        rids = _vertices(db, "Bulk", rows)
        assert len(rids) == 8 and len(set(rids)) == 8
        assert len(matrix_calls) == 1, "only rows 3 to 5 needed the old path"
        got = db.query("sql", "SELECT FROM Bulk ORDER BY id").to_list()
        assert [r["id"] for r in got] == list(range(8))
        assert got[0]["k"] == ["s0", "z"] and got[0]["d"] == {"x": 0}
        assert got[4]["k"][0] == "x" and math.isnan(got[4]["k"][1])
        assert got[5]["k"] == ["s5", "z"] and got[7]["d"] == {"x": 7}


def test_vertices_a_ragged_list_in_a_declared_vector_property_keeps_the_old_path(
    temp_db_path, matrix_calls
):
    """An ARRAY_OF_FLOATS property whose lists the float matrix cannot carry still goes to the matrix path with its chunk."""
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE Vx")
        db.command("sql", "CREATE PROPERTY Vx.id INTEGER")
        db.command("sql", "CREATE PROPERTY Vx.embedding ARRAY_OF_FLOATS")
        db.command("sql", "CREATE PROPERTY Vx.tags LIST")
        rows = [
            {"id": 0, "embedding": [1.0, 2.0], "tags": ["a"]},
            {"id": 1, "embedding": [1.0], "tags": ["b", "c"]},
        ]
        _vertices(db, "Vx", rows)
        assert len(matrix_calls) == 1
        got = db.query("sql", "SELECT FROM Vx ORDER BY id").to_list()
        assert [list(r["embedding"]) for r in got] == [[1.0, 2.0], [1.0]]
        assert [r["tags"] for r in got] == [["a"], ["b", "c"]]


def test_vertices_a_declared_vector_property_and_a_list_property_share_a_chunk(
    temp_db_path, matrix_calls
):
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE Vx")
        db.command("sql", "CREATE PROPERTY Vx.id INTEGER")
        db.command("sql", "CREATE PROPERTY Vx.embedding ARRAY_OF_FLOATS")
        rows = [
            {
                "id": i,
                "embedding": [float(i), 0.5, 0.25],
                "tags": [f"t{i}", "z"],
                "meta": {"n": i},
                "seen": dt.datetime(2026, 1, 1, 0, 0, i),
            }
            for i in range(4)
        ]
        _vertices(db, "Vx", rows)
        assert matrix_calls == []
        got = db.query("sql", "SELECT FROM Vx ORDER BY id").to_list()
        assert [list(r["embedding"]) for r in got] == [
            [float(i), 0.5, 0.25] for i in range(4)
        ]
        assert [r["tags"] for r in got] == [[f"t{i}", "z"] for i in range(4)]
        assert [r["meta"] for r in got] == [{"n": i} for i in range(4)]
        assert [r["seen"] for r in got] == [
            dt.datetime(2026, 1, 1, 0, 0, i) for i in range(4)
        ]


@pytest.mark.parametrize(
    "value, ok",
    [
        (None, True),
        (1, True),
        (1.5, True),
        ("text", True),
        (["a", 1], True),
        (("a", 1), True),
        ([[1, 2]], True),
        ([None, 1.5], True),
        ([True, 1], True),
        ([], True),
        ([1, 2], False),
        ((1, 2.5), False),
        ({"a": 1}, True),
        (dt.datetime(2026, 5, 6), True),
        (dt.date(2026, 5, 6), True),
        (decimal.Decimal("1.5"), True),
        (b"bytes", True),
        (bytearray(b"ab"), True),
        ({1, 2}, False),
        (dt.time(1, 2), False),
        (math.nan, False),
        (2**63, False),
        ("a\ud800b", False),
    ],
)
def test_json_bulk_tagged_ok(value, ok):
    assert type_conversion.json_bulk_tagged_ok(value) is ok


def test_json_bulk_tagged_ok_refuses_subclasses_and_numpy_values():
    pd = pytest.importorskip("pandas")
    assert not type_conversion.json_bulk_tagged_ok(pd.Timestamp("2026-05-06"))
    assert not type_conversion.json_bulk_tagged_ok(np.float32(1.5))
    assert not type_conversion.json_bulk_tagged_ok(np.arange(3))


def test_the_bulk_method_returns_none_for_a_type_the_tags_do_not_carry(temp_db_path):
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE V")
        with db.graph_batch(parallel_flush=False) as batch:
            assert batch._create_vertices_json_bulk("V", [{"k": {1, 2}}]) is None
            rids = batch._create_vertices_json_bulk(
                "V", [{"k": ["a"]}, {"k": {"a": 1}}]
            )
            assert rids is not None and len(rids) == 2


@pytest.mark.parametrize("name", list(NUMERIC_LISTS))
def test_vertices_a_list_of_only_numbers_keeps_the_matrix_path(
    temp_db_path, monkeypatch, matrix_calls, name
):
    """One double[] per list beats the text (probe 2026-11-04: 30.8 against 117.4 us per row for 128 doubles): the input takes main's path."""
    rows = [{"id": i, "k": NUMERIC_LISTS[name]} for i in range(3)]
    mixed = rows + [
        {"id": 3, "k": {"a": 1}}
    ]  # a shape that would take the text, beside it
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        _reference_vertices(monkeypatch, db, "Ref", mixed)
        matrix_calls.clear()
        _vertices(db, "Bulk", rows)
        assert len(matrix_calls) == 1, "a numeric list left the matrix path"
        names = {"id", "k"}
        assert _stored(db, "Bulk", names) == _stored(db, "Ref", names)[:3]
        _vertices(db, "Bulk", mixed)
        assert (
            len(matrix_calls) == 2
        ), "one numeric list sent the whole call down the text"


@pytest.mark.parametrize("name", list(NUMERIC_LISTS))
def test_edges_a_list_of_only_numbers_keeps_the_per_edge_path(
    temp_db_path, edge_calls, name
):
    rows = [{"id": i, "k": NUMERIC_LISTS[name]} for i in range(3)]
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        _edges(db, "ERef", rows, per_edge=True)
        edge_calls.clear()
        _edges(db, "EBulk", rows)
        assert len(edge_calls) == 3
        assert _stored(db, "EBulk", {"id", "k"}) == _stored(db, "ERef", {"id", "k"})


@pytest.fixture
def untagged_only(monkeypatch):
    """Spy on what an input of scalars must not touch: the tagged writer and the shape gate (main's code has neither)."""
    calls = {"tagged": 0, "shape": 0}
    real_dumps, real_shape = gbm.json_bulk_dumps_tagged, gbm.json_bulk_shape_ok

    def dumps(*a, **k):
        calls["tagged"] += 1
        return real_dumps(*a, **k)

    def shape(*a, **k):
        calls["shape"] += 1
        return real_shape(*a, **k)

    monkeypatch.setattr(gbm, "json_bulk_dumps_tagged", dumps)
    monkeypatch.setattr(gbm, "json_bulk_shape_ok", shape)
    return calls


def test_vertices_an_input_of_scalars_takes_mains_code(
    temp_db_path, matrix_calls, untagged_only
):
    rows = [
        {"id": i, "p": i, "s": f"t{i}", "f": i * 0.5, "b": i % 2 == 0, "n": None}
        for i in range(50)
    ]
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        assert len(_vertices(db, "Bulk", rows)) == 50
        assert matrix_calls == []
        assert untagged_only == {"tagged": 0, "shape": 0}


def test_edges_an_input_of_scalars_takes_mains_code(
    temp_db_path, edge_calls, untagged_only
):
    rows = [{"id": i, "p": i, "s": f"t{i}"} for i in range(50)]
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        _edges(db, "EBulk", rows)
        assert edge_calls == []
        assert untagged_only == {"tagged": 0, "shape": 0}


def test_vertices_a_shape_takes_the_tagged_writer(
    temp_db_path, matrix_calls, untagged_only
):
    rows = [{"id": i, "p": ["a", i]} for i in range(5)]
    with arcadedb.create_database(temp_db_path) as db:
        _schema(db, False)
        _vertices(db, "Bulk", rows)
        assert matrix_calls == []
        assert untagged_only["tagged"] == 1 and untagged_only["shape"] == 5
