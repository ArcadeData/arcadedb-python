"""Native GraphBatch.create_vertices: the vertices that the columnar route stores are the ones the JSON route stores (ArcadeData/arcadedb-python #388, rank 7).

The same rows are loaded into two vertex types of one schema, once with ``lib.GB_COLP`` on (``arcadedb_n_gb_colp``: columns by address, the
RIDs back in an int64 buffer) and once with it off (the JSON text of ``arcadedb_n_gb``), and every stored record is compared: the property
order, the type and the value of each property (``None`` and ``NaN`` included), and the RIDs the call returned (a ``list`` of ``str`` in row
order). A shape the columns cannot carry must come out the same because it takes the JSON route; the entry that each shape crossed is checked
too, so a shape cannot move from one route to the other unseen. Needs the native library, so conftest.py collects this file on the native
backend only; the routing, the header and the RID decode are in test_native_graph_batch_frame_pack.py (both backends, no library).
"""

import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from arcadedb._native import lib

COLP = "arcadedb_n_gb_colp"


@pytest.fixture
def db(temp_db):
    for name in (
        "A",
        "B",
    ):  # the same schema twice: A is loaded with the columnar route, B with the JSON route
        temp_db.command("sql", f"CREATE VERTEX TYPE {name}")
        temp_db.command("sql", f"CREATE PROPERTY {name}.i LONG")
        temp_db.command("sql", f"CREATE PROPERTY {name}.n INTEGER")
        temp_db.command("sql", f"CREATE PROPERTY {name}.f FLOAT")
        temp_db.command("sql", f"CREATE PROPERTY {name}.v ARRAY_OF_FLOATS")
    return temp_db


def _vectors(n, dim=6):
    x = np.random.default_rng(5).standard_normal((n, dim)).astype(np.float32)
    x[0, 0] = np.float32(-0.0)
    return x


def _rows(n, make):
    return [{"i": i, **make(i)} for i in range(n)]


SHAPES = {
    "scalars": (lambda i: {"s": f"né{i}", "d": i * 0.5, "b": i % 2 == 0}, True),
    "big integers": (
        lambda i: {"big": [2**53 + 1, -(2**63), 2**63 - 1, 7][i % 4]},
        True,
    ),
    "nan inf and negative zero": (
        lambda i: {"x": [float("nan"), float("inf"), float("-inf"), -0.0, 1.5][i % 5]},
        True,
    ),
    "empty and odd strings": (
        lambda i: {"s": ["", "a\u0000b", "\U0001f600", "x" * 300][i % 4]},
        True,
    ),
    "nulls in a column": (
        lambda i: {"s": None if i % 3 == 0 else f"s{i}", "d": None if i % 2 else 1.0},
        True,
    ),
    "only nulls": (lambda i: {"s": None}, True),
    "dates and datetimes": (
        lambda i: {
            "day": date(2020, 1, 1) + timedelta(days=i),
            "at": datetime(2020, 1, 2, 3, 4, 5, 123456 + i),
            "utc": datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=9))),
        },
        True,
    ),
    "decimals and bytes": (
        lambda i: {"dec": Decimal(f"{i}.50"), "raw": bytes([i, 0, 255])},
        True,
    ),
    "declared integer and float": (lambda i: {"n": i, "f": i * 0.1}, True),
    "float32 vectors": (lambda i: {"emb": _vectors(40)[i]}, True),
    "declared vectors from lists": (lambda i: {"v": [i * 0.5, i + 0.25, 3.0]}, True),
    "declared vectors from float32 arrays": (lambda i: {"v": _vectors(40)[i]}, True),
    "float64 vectors": (lambda i: {"emb": _vectors(40)[i].astype(np.float64)}, False),
    "nested lists and maps": (
        lambda i: {"l": [i, "x", [1, 2]], "m": {"k": i, "z": None}},
        False,
    ),
    "numpy scalars": (
        lambda i: {"a": np.int64(i), "b": np.float64(i) / 3, "c": np.bool_(i % 2)},
        False,
    ),
    "mixed int and float": (lambda i: {"a": i if i % 2 else i + 0.5}, False),
    "a set and a tuple": (lambda i: {"s": {i}, "t": (i, 1)}, False),
    "a pandas timestamp with nanoseconds": (
        lambda i: {
            "t": pd.Timestamp("2020-01-01 00:00:00.000000123") + pd.Timedelta(i, "ns")
        },
        False,
    ),
}


def _dump(db, type_name):
    """Every stored record as (property, type, repr) in its stored order, sorted by the row index."""
    out = {}
    sql = f"SELECT FROM {type_name}"  # nosec B608 - test-owned name
    for row in db.query("sql", sql).to_list():
        props = [
            (k, type(v).__name__, repr(v))
            for k, v in row.items()
            if not str(k).startswith("@")
        ]
        out[row["i"]] = props
    return [out[i] for i in sorted(out)]


def _load(db, monkeypatch, type_name, rows, columnar):
    calls = []
    real = lib.call

    def spy(name, *args):
        calls.append(name)
        return real(name, *args)

    monkeypatch.setattr(lib, "call", spy)
    monkeypatch.setattr(lib, "GB_COLP", columnar)
    try:
        with db.graph_batch() as gb:
            rids = gb.create_vertices(type_name, rows)
    finally:
        monkeypatch.undo()
    return rids, calls


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("n", [1, 40])
def test_the_columnar_route_stores_what_the_json_route_stores(
    db, monkeypatch, shape, n
):
    make, carried = SHAPES[shape]
    if shape == "mixed int and float" and n == 1:
        carried = True  # one row holds one type: a column of one float
    rows = _rows(n, make)
    rids_on, calls_on = _load(db, monkeypatch, "A", rows, True)
    rids_off, calls_off = _load(db, monkeypatch, "B", rows, False)
    assert (COLP in calls_on) is carried, (shape, calls_on)
    assert COLP not in calls_off
    assert _dump(db, "A") == _dump(db, "B")
    for rids in (rids_on, rids_off):
        assert type(rids) is list and len(rids) == n
        assert all(type(r) is str and re.fullmatch(r"#\d+:\d+", r) for r in rids)


def test_the_rids_are_the_vertices_in_row_order_across_chunks(db, monkeypatch):
    rows = [
        {"i": i, "s": f"p{i}"} for i in range(2500)
    ]  # the default chunk is 1000 rows: three crossings
    rids, calls = _load(db, monkeypatch, "A", rows, True)
    assert calls.count(COLP) == 3
    by_rid = {
        str(r["rid"]): r["i"]
        for r in db.query("sql", "SELECT i, @rid AS rid FROM A").to_list()
    }
    assert [by_rid[r] for r in rids] == list(range(2500))
    assert len(set(rids)) == 2500


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"i": 0, "a": 1, "b": 2},
            {"i": 1, "b": 4, "a": 3},
        ],  # the same keys in another order
        [{"i": 0, "a": 1, "b": 2}, {"i": 1, "a": 3}],  # a missing key
        [{"i": 0, "a": None}, {"i": 1}],  # None against a missing key
    ],
)
def test_rows_with_other_keys_keep_the_json_route_and_the_stored_order(
    db, monkeypatch, rows
):
    _, calls_on = _load(db, monkeypatch, "A", rows, True)
    _load(db, monkeypatch, "B", rows, False)
    assert COLP not in calls_on
    assert _dump(db, "A") == _dump(db, "B")


def test_a_none_row_among_dict_rows_is_an_empty_vertex_on_both_routes(db, monkeypatch):
    rows = [{"i": 0, "a": 1}, None, {"i": 2, "a": 3}]
    rids_on, calls_on = _load(db, monkeypatch, "A", rows, True)
    rids_off, _ = _load(db, monkeypatch, "B", rows, False)
    assert COLP not in calls_on and len(rids_on) == len(rids_off) == 3
    for type_name in ("A", "B"):
        sql = f"SELECT count(*) AS c FROM {type_name}"  # nosec B608 - test-owned name
        assert db.query("sql", sql).first().get("c") == 3


def test_an_integer_past_64_bits_is_refused_by_both_routes_alike(db, monkeypatch):
    rows = [{"i": 0, "a": 2**70}]
    errors = []
    for type_name, columnar in (("A", True), ("B", False)):
        with pytest.raises(Exception) as raised:
            _load(db, monkeypatch, type_name, rows, columnar)
        errors.append(
            str(raised.value).replace(f"'{type_name}'", "'T'")
        )  # the type name is the only difference
    assert errors[0] == errors[1]
    assert db.query("sql", "SELECT count(*) AS c FROM A").first().get("c") == 0


def test_an_empty_batch_and_a_batch_of_one(db, monkeypatch):
    rids, calls = _load(db, monkeypatch, "A", [], True)
    assert rids == [] and COLP not in calls
    rids, calls = _load(db, monkeypatch, "A", [{"i": 0, "s": "one"}], True)
    assert COLP in calls and len(rids) == 1 and rids[0].startswith("#")


def test_a_failing_chunk_stores_nothing_of_itself_and_keeps_the_chunks_before_it(
    db, monkeypatch
):
    db.command("sql", "CREATE INDEX ON A (i) UNIQUE")
    rows = [{"i": i, "s": "x"} for i in range(1000)] + [
        {"i": i, "s": "y"} for i in range(5)
    ]  # the 2nd chunk repeats keys
    with pytest.raises(Exception):
        _load(db, monkeypatch, "A", rows, True)
    assert db.query("sql", "SELECT count(*) AS c FROM A").first().get("c") == 1000


def test_the_paper_shaped_loads_take_the_columnar_route_end_to_end(
    temp_db, monkeypatch
):
    """l2 (Person rows in chunks) and e2 (Product rows with a float32 vector): every chunk is one arcadedb_n_gb_colp crossing, no create_vertices
    crossing of the JSON entries, no codec hook; the vertices and the vectors are stored, the RIDs come back as a list of str.
    """
    from arcadedb._native import codec

    temp_db.command("sql", "CREATE VERTEX TYPE Person")
    temp_db.command("sql", "CREATE VERTEX TYPE Product")
    temp_db.command("sql", "CREATE PROPERTY Product.embedding ARRAY_OF_FLOATS")
    calls, hooks = [], {"prep": 0, "default": 0}
    real_call, real_prep, real_default = lib.call, codec._prep, codec._default

    def spy(name, *args):
        calls.append(
            (name, args[1] if name in ("arcadedb_n_gb", "arcadedb_n_gb_bin") else None)
        )
        return real_call(name, *args)

    def counting(key, real):
        def wrapper(*args, **kwargs):
            hooks[key] += 1
            return real(*args, **kwargs)

        return wrapper

    people = [
        {"id": i, "name": f"p{i}", "age": 20 + i % 50, "city": f"c{i % 7}"}
        for i in range(2500)
    ]
    x = _vectors(2100, 16)
    products = [
        {"id": i, "title": f"t{i}", "price": i * 0.25, "embedding": x[i]}
        for i in range(2100)
    ]
    with (
        temp_db.graph_batch() as gb
    ):  # the default commit_every: 1000 rows per crossing
        monkeypatch.setattr(lib, "call", spy)
        monkeypatch.setattr(codec, "_prep", counting("prep", real_prep))
        monkeypatch.setattr(codec, "_default", counting("default", real_default))
        rids_people = gb.create_vertices("Person", people)
        rids_products = gb.create_vertices("Product", products)
        monkeypatch.undo()
    assert [
        c for c in calls if c[1] == b"create_vertices"
    ] == []  # no JSON entry carried a chunk
    assert [c[0] for c in calls].count(
        COLP
    ) == 3 + 3  # 1000, 1000, 500 and 1000, 1000, 100
    assert hooks == {"prep": 0, "default": 0}
    assert len(rids_people) == 2500 and len(rids_products) == 2100
    assert all(type(r) is str for r in rids_people + rids_products)
    stored = {
        r["id"]: r["embedding"]
        for r in temp_db.query("sql", "SELECT id, embedding FROM Product").to_list()
    }
    assert len(stored) == 2100
    assert all(
        np.array(stored[i], dtype=np.float32).tobytes() == x[i].tobytes()
        for i in range(2100)
    )
    assert (
        temp_db.query("sql", "SELECT count(*) AS c FROM Person").first().get("c")
        == 2500
    )
