"""RID and list columns travel as binary column kinds, and read back as the JSON form gave them (refs #388, #315).

A column frame has carried a RID, a list or a map as one JSON array of the column (the ``json`` kind), parsed per cell in Python. The
writers of both wheels now write ``rid`` (bucket and position, two int64 per row) and ``li8`` / ``lf8`` (a list of int64 or of float64 per
row: offsets, a bitmap of null elements, one flat array) for those columns, and keep ``json`` for every other shape: maps, nested
lists, lists of strings, of booleans or of RIDs, lists that mix ints and floats, a list of nothing but nulls.

Three kinds of check, all on the wheel that runs the tests (no branch on ``arcadedb.__backend__`` except where a wheel's frames differ in
how they start, noted below):

* ``test_the_frames_*``: which kind each shape gets, read from the frame headers the readers decode (a frame header is what the Java
  writers produced: this fails on a wheel whose writers still write ``json``).
* ``test_readers_agree_with_the_json_route``: ``to_columns``, ``to_arrow``, ``to_dataframe`` and ``to_polars`` against the row route
  (``to_list``: the tagged JSON of the rows, which this change does not touch) cell by cell, and the Arrow type the JSON form would have
  had.
* ``test_readers_match_the_tag_fixture``: the same four readers against ``fixtures/frame_kinds_post3.json``, the output of the released
  26.11.1.post3 wheels (JSON route, one entry per wheel) for the same table, stored before the change and never regenerated from this code. RID values compare
  by shape (``#N:N``): the bucket numbers of a type depend on the machine, and each RID is checked against the row route instead.

On the native wheel the first 256 rows of a result are read as JSON text and written as a frame in Python (``json`` kind: the values
themselves), the rows after them come from the library's frames, so every table here has more than 256 rows.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import arcadedb
from arcadedb import _columnar

ROWS = 700
FIXTURE = Path(__file__).parent / "fixtures" / "frame_kinds_post3.json"
INF = float("inf")


def _g1(i):
    k = i % 5
    return {
        "n": i,
        "li": [[i, i + 1, i + 2], [], None, [-(2**62), 2**53 + 1, 0], [7]][k],
        "lf": [[0.5, 1.5], [], None, [-0.0, 1e300, 5e-324], [i / 7]][k],
        "le": [],
        "lm": [[1, 2], [1.5]][i % 2],
        "lmix": [1, 2.5],
        "ls": ["a", "b"],
        "lb": [True, False],
        "mp": {"a": i, "b": "x"},
        "nl": [[1, 2], [3]],
        "mixc": 5 if i % 2 else "x",
    }


def _g3(i):
    k = i % 5
    return {
        "n": i,
        "lin": [[4, None], [None], [1, 2], [], None][k],
        "lfn": [[0.1, None], [1.5], [None, 2.5], [], None][k],
    }


def _g4(i):
    k = i % 5
    return {"n": i, "lfx": [[math.nan, INF, -INF, 0.1], [1.5], [-INF], [], None][k]}


GROUPS = {
    "g1": ("SELECT n, li, lf, le, lm, lmix, ls, lb, mp, nl, mixc FROM g1 ORDER BY n"),
    "g2": "SELECT n, lnk, rls, @rid AS rid FROM g2 ORDER BY n",
    "g3": "SELECT n, lin, lfn FROM g3 ORDER BY n",
    "g4": "SELECT n, lfx FROM g4 ORDER BY n",
}
# the kind of each column in the frames of a read (the shapes in the table above)
KINDS = {
    "g1": {
        "li": "li8",
        "lf": "lf8",
        "le": "json",
        "lm": "json",
        "lmix": "json",
        "ls": "json",
        "lb": "json",
        "mp": "json",
        "nl": "json",
        "mixc": "json",
    },
    "g2": {"lnk": "rid", "rls": "json", "rid": "rid"},
    "g3": {"lin": "li8", "lfn": "lf8"},
    "g4": {"lfx": "lf8"},
}


def build(db, group):
    """Create and fill the table ``group`` (several hundred rows: see the module docstring)."""
    if group == "g2":
        db.command("sql", "CREATE DOCUMENT TYPE t")
        db.insert_many("t", [{"k": k} for k in range(5)])
        rids = [
            str(r["rid"])
            for r in db.query("sql", "SELECT @rid AS rid FROM t ORDER BY k").to_list()
        ]
        db.command("sql", "CREATE DOCUMENT TYPE g2")
        db.command("sql", "CREATE PROPERTY g2.lnk LINK")
        with db.transaction():
            for i in range(ROWS):
                link = "null" if i % 9 == 0 else rids[i % 5]
                db.command(
                    "sql",
                    f"INSERT INTO g2 SET n = {i}, lnk = {link}, rls = [{rids[i % 5]}, {rids[(i + 1) % 5]}]",
                )
        return
    db.command("sql", f"CREATE DOCUMENT TYPE {group}")
    make = {"g1": _g1, "g3": _g3, "g4": _g4}[group]
    db.insert_many(group, [make(i) for i in range(ROWS)])


# ---------------------------------------------------------------- normal form of a value, for comparing and for the stored fixture
_RID_TEXT = re.compile(r"^#-?\d+:-?\d+$")


def norm(v):
    """A JSON-able, exact form of a value: type tags, ``repr`` of floats (NaN, -0.0), RIDs by their shape."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return ["b", bool(v)]
    if isinstance(v, (int, np.integer)):
        return ["i", str(int(v))]
    if isinstance(v, (float, np.floating)):
        return ["f", repr(float(v))]
    if isinstance(v, str):
        return ["s", "#N:N" if _RID_TEXT.match(v) else v]
    if hasattr(v, "getBucketId") or type(v).__name__ == "RID":
        return ["rid", "#N:N"]
    if isinstance(v, np.ndarray):
        return ["nd", str(v.dtype), list(v.shape), [norm(x) for x in v.tolist()]]
    if isinstance(v, (list, tuple)):
        return ["l", [norm(x) for x in v]]
    if isinstance(v, dict):
        return ["d", [[str(k), norm(x)] for k, x in v.items()]]
    return ["?", type(v).__name__, str(v)]


def _column(kind, values):
    flat = [norm(v) for v in values]
    digest = hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest()
    return {
        "type": kind,
        "n": len(flat),
        "sha": digest,
        "sample": flat[:12] + flat[256:268],
    }


def _read_columns(rs, bs):
    return {
        k: (
            type(v).__name__
            + (f":{v.dtype}:{v.shape[1:]}" if isinstance(v, np.ndarray) else ""),
            list(v),
        )
        for k, v in rs.to_columns(batch_size=bs).items()
    }


def _read_arrow(rs, bs):
    table = rs.to_arrow(batch_size=bs)
    return {
        name: (str(table.column(name).type), table.column(name).to_pylist())
        for name in table.column_names
    }


def _read_dataframe(rs, bs):
    df = rs.to_dataframe()
    return {name: (str(df[name].dtype), df[name].tolist()) for name in df.columns}


def _read_polars(rs, bs):
    df = rs.to_polars(batch_size=bs)
    return {name: (str(df[name].dtype), df[name].to_list()) for name in df.columns}


READERS = {
    "to_columns": _read_columns,
    "to_arrow": _read_arrow,
    "to_dataframe": _read_dataframe,
    "to_polars": _read_polars,
}


def snapshot(db, group):
    """``{reader: {column: type, size, digest of the exact values, a sample}}`` of every reader on ``group`` (an error is its class name)."""
    out = {}
    for name, read in READERS.items():
        for bs in (25_000, 100) if name != "to_dataframe" else (25_000,):
            key = f"{name}:{bs}"
            try:
                cols = read(db.query("sql", GROUPS[group]), bs)
                out[key] = {
                    c: _column(kind, values) for c, (kind, values) in cols.items()
                }
            except (
                Exception
            ) as exc:  # noqa: BLE001 - the stored form of a failure is its class
                out[key] = {"error": type(exc).__name__}
    return out


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    db = arcadedb.create_database(str(tmp_path_factory.mktemp("frame_kinds") / "db"))
    for group in GROUPS:
        build(db, group)
    yield db
    db.close()


# ---------------------------------------------------------------------------------------------------- which kind each shape gets
@pytest.fixture
def recorded_kinds(monkeypatch):
    seen = []
    real = _columnar.read_header

    def read_header(buf):
        header, pos = real(buf)
        seen.append({c["name"]: c["type"] for c in header["cols"]})
        return header, pos

    monkeypatch.setattr(_columnar, "read_header", read_header)
    return seen


@pytest.mark.parametrize("group", list(GROUPS))
@pytest.mark.parametrize("reader", ["to_columns", "to_arrow", "to_dataframe"])
def test_the_frames_carry_rid_and_number_lists_in_binary_kinds(
    tables, recorded_kinds, group, reader
):
    getattr(tables.query("sql", GROUPS[group]), reader)()
    assert recorded_kinds, "no frame was decoded"
    for name, expected in KINDS[group].items():
        # every frame that has the column: the read ends with an empty frame (no columns), and on the native wheel the first 256 rows
        # are a Python-written frame (always json) that is decoded after the library's frames, so neither is "the last frame"
        kinds = [frame[name] for frame in recorded_kinds if name in frame]
        assert kinds, name
        if expected == "json":
            assert set(kinds) == {"json"}, (name, kinds)
        else:
            assert expected in kinds and set(kinds) <= {expected, "json"}, (name, kinds)


# ------------------------------------------------------------------------------------------------ the values the JSON route gave
def _json_form_arrow(values):
    """The Arrow array the ``json`` kind made of a column of Python values."""
    try:
        return pa.array(values)
    except pa.ArrowException:
        return pa.array(
            [None if v is None else str(v) for v in values], type=pa.string()
        )


def _same(have, want):
    return norm(list(have) if not isinstance(have, list) else have) == norm(
        list(want) if not isinstance(want, list) else want
    )


@pytest.mark.parametrize("group", list(GROUPS))
def test_readers_agree_with_the_json_route(tables, group):
    sql = GROUPS[group]
    rows = tables.query("sql", sql).to_list()
    names = list(rows[0])
    want = {name: [row.get(name) for row in rows] for name in names}

    columns = tables.query("sql", sql).to_columns(batch_size=100)
    table = tables.query("sql", sql).to_arrow(batch_size=100)
    frame = tables.query("sql", sql).to_dataframe()
    for name in names:
        # values: a RID is the wheel's own RID object, equal to the row route's, cell by cell
        assert _same(columns[name], want[name]), name
        if name not in ("n", "mixc"):
            expected = _json_form_arrow(want[name])
            assert table.column(name).type == expected.type, (
                name,
                table.column(name).type,
                expected.type,
            )
            assert norm(table.column(name).to_pylist()) == norm(
                expected.to_pylist()
            ), name
        assert _same(frame[name].tolist(), want[name]), name
    for name in KINDS[group]:
        if KINDS[group][name] == "rid":
            # the text of every RID, and its type, are the row route's
            for have, w in zip(columns[name], want[name]):
                assert (have is None) == (w is None), name
                assert have is None or (
                    str(have) == str(w) and type(have) is type(w)
                ), name


@pytest.mark.parametrize("group", list(GROUPS))
def test_readers_match_the_tag_fixture(tables, group):
    stored = json.loads(FIXTURE.read_text())[arcadedb.__backend__]
    assert (
        group in stored and "build_error" not in stored[group]
    ), f"no stored output for {group}: {stored.get(group)}"
    now = snapshot(tables, group)
    assert now.keys() == stored[group].keys()
    for key in now:
        assert now[key] == stored[group][key], (group, key)
