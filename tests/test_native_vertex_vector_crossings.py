"""Native GraphBatch.create_vertices: float32 vectors are split and encoded as columns, not row by row (#337).

Before, every float32 array of a row was found by ``json.dumps`` through a Python hook (``codec._gb_default``, then ``_default``: about a
dozen ``isinstance`` tests and one array copy per row), the rows were walked by ``codec._prep`` first (one Python call per value), a tag
``{"\\u0001":["B",offset,count]}`` was written and parsed back in Java per row, and ``np.concatenate`` joined one small array per row
(the 29 to 31 percent prep share of the e2 and l3d loads in the untimed ingest profile of 26.11.1). Now ``take_vector_columns`` stacks each
float32 column once, ``codec.encode_vertex_args`` writes the other properties with one ``json.dumps`` (no hook when they are plain scalars)
and the matrices cross as the front of the side buffer, described by one ``"vecs"`` entry, which the library
(``NativeExtras.withVectors``) turns into one ``float[]`` per row. One ``arcadedb_n_gb_bin`` crossing per chunk, as before.

Needs the native library, so conftest.py collects this file on the native backend only. The stored values are in test_vertex_vector_loads.py
(both wheels) and test_native_graph_batch_binary_vectors.py.
"""

import ctypes
import json
from datetime import datetime

import numpy as np
import pytest

from arcadedb._native import codec, lib

CHUNK, ROWS, DIM = 100, 250, 16  # three chunks: 100, 100, 50


class _Spy:
    """The create_vertices crossings (entry, JSON text, side buffer, float count) and the per-row Python calls of the codec."""

    def __init__(self, monkeypatch):
        self.crossings, self.hooks = [], {"gb_default": 0, "default": 0, "prep": 0}
        real_call, real_gb_default = lib.call, codec._gb_default
        real_default, real_prep = codec._default, codec._prep

        def call(name, *args):
            if (
                name in ("arcadedb_n_gb", "arcadedb_n_gb_bin")
                and args[1] == b"create_vertices"
            ):
                buf = args[3] if name == "arcadedb_n_gb_bin" else None
                count = args[4] if name == "arcadedb_n_gb_bin" else 0
                self.crossings.append((name, args[2].decode(), buf, count))
            return real_call(name, *args)

        def counting(key, real):
            def wrapper(*args, **kwargs):
                self.hooks[key] += 1
                return real(*args, **kwargs)

            return wrapper

        monkeypatch.setattr(lib, "call", call)
        monkeypatch.setattr(
            codec, "_gb_default", counting("gb_default", real_gb_default)
        )
        monkeypatch.setattr(codec, "_default", counting("default", real_default))
        monkeypatch.setattr(codec, "_prep", counting("prep", real_prep))

    def entries(self):
        return [c[0] for c in self.crossings]


@pytest.fixture
def spy(monkeypatch):
    return _Spy(monkeypatch)


@pytest.fixture
def db(temp_db):
    for name in ("Declared", "Free"):
        temp_db.command("sql", f"CREATE VERTEX TYPE {name}")
        temp_db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    temp_db.command("sql", "CREATE PROPERTY Declared.embedding ARRAY_OF_FLOATS")
    temp_db.command("sql", "CREATE PROPERTY Free.born DATETIME")
    return temp_db


def _x(n=ROWS, dim=DIM, seed=7):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _stored(db, type_name, prop="embedding"):
    sql = f"SELECT id, {prop} AS v FROM {type_name}"  # nosec B608 - test-owned names
    return {r["id"]: r["v"] for r in db.query("sql", sql).to_list()}


def _assert_bit_equal(stored, x):
    assert sorted(stored) == list(range(len(x)))
    for i in range(len(x)):
        assert np.array(stored[i], dtype=np.float32).tobytes() == x[i].tobytes()


@pytest.mark.parametrize("type_name", ["Declared", "Free"])
def test_a_float32_column_is_one_matrix_with_no_hook_call_per_row(db, spy, type_name):
    x = _x()
    rows = [{"id": i, "embedding": x[i]} for i in range(ROWS)]
    with db.graph_batch(commit_every=CHUNK) as gb:
        assert len(gb.create_vertices(type_name, rows)) == ROWS
    assert spy.entries() == ["arcadedb_n_gb_bin"] * 3  # one crossing per chunk of 100
    assert spy.hooks == {"gb_default": 0, "default": 0, "prep": 0}, spy.hooks
    assert [c[3] for c in spy.crossings] == [CHUNK * DIM, CHUNK * DIM, 50 * DIM]
    for _, text, buf, count in spy.crossings:
        args = json.loads(text)
        assert args["vecs"] == {"names": ["embedding"], "dims": [DIM], "offs": [0]}
        assert '"B"' not in text and '"F"' not in text  # no tag and no number text
        assert all(set(row) == {"id"} for row in args["rows"])
        assert isinstance(
            buf, ctypes.Array
        )  # a view of the matrix, not a copy of its bytes
    _assert_bit_equal(_stored(db, type_name), x)


def test_two_columns_and_a_date_time_keep_the_buffer_layout(db, spy):
    x, y = _x(), _x(dim=8, seed=8)
    when = datetime(2020, 1, 2, 3, 4, 5)
    rows = [
        {"id": i, "embedding": x[i], "second": y[i], "born": when} for i in range(ROWS)
    ]
    with db.graph_batch(commit_every=CHUNK) as gb:
        gb.create_vertices("Free", rows)
    assert spy.entries() == ["arcadedb_n_gb_bin"] * 3
    first = json.loads(spy.crossings[0][1])
    assert first["vecs"] == {
        "names": ["embedding", "second"],
        "dims": [DIM, 8],
        "offs": [0, CHUNK * DIM],
    }
    assert spy.crossings[0][3] == CHUNK * DIM + CHUNK * 8
    _assert_bit_equal(_stored(db, "Free"), x)
    _assert_bit_equal(_stored(db, "Free", "second"), y)
    sql = "SELECT born FROM Free WHERE id = 7"
    assert db.query("sql", sql).first().get("born") == when


def test_a_float32_array_that_does_not_qualify_follows_the_matrices_in_the_buffer(
    db, spy
):
    x = _x(60)
    rows = [{"id": i, "embedding": x[i], "short": x[i][: 2 + i % 2]} for i in range(60)]
    with db.graph_batch(commit_every=CHUNK) as gb:
        gb.create_vertices("Free", rows)
    assert spy.entries() == ["arcadedb_n_gb_bin"]
    text, count = spy.crossings[0][1], spy.crossings[0][3]
    assert json.loads(text)["vecs"]["offs"] == [0]
    assert json.loads(text)["rows"][0]["short"] == {"\u0001": ["B", 60 * DIM, 2]}
    assert count == 60 * DIM + sum(2 + i % 2 for i in range(60))
    _assert_bit_equal(_stored(db, "Free"), x)
    short = _stored(db, "Free", "short")
    for i in range(60):
        assert (
            np.array(short[i], dtype=np.float32).tobytes()
            == x[i][: 2 + i % 2].tobytes()
        )


def test_a_float64_or_list_column_keeps_the_text_entry(db, spy):
    x = _x(30)
    rows = [{"id": i, "embedding": x[i].astype(np.float64)} for i in range(30)]
    with db.graph_batch(commit_every=CHUNK) as gb:
        gb.create_vertices("Free", rows)
        gb.create_vertices(
            "Declared", [{"id": i, "embedding": x[i].tolist()} for i in range(30)]
        )
    assert spy.entries() == ["arcadedb_n_gb"] * 2
    assert all('"vecs"' not in c[1] for c in spy.crossings)
    _assert_bit_equal(_stored(db, "Declared"), x)


def test_a_library_without_the_entry_keeps_the_text_path(db, spy, monkeypatch):
    monkeypatch.setattr(lib, "GB_BIN", False)
    x = _x(30)
    with db.graph_batch(commit_every=CHUNK) as gb:
        gb.create_vertices("Free", [{"id": i, "embedding": x[i]} for i in range(30)])
    assert spy.entries() == ["arcadedb_n_gb"]
    assert '"F"' in spy.crossings[0][1] and '"vecs"' not in spy.crossings[0][1]
    _assert_bit_equal(_stored(db, "Free"), x)


def test_a_vector_column_outside_the_side_buffer_is_refused(db):
    x = _x(4)
    with db.graph_batch() as gb:
        text = json.dumps(
            {
                "type": "Free",
                "rows": [{"id": i} for i in range(4)],
                "vecs": {"names": ["embedding"], "dims": [DIM], "offs": [8]},
            }
        )
        buf = x.tobytes()  # 4 vectors of 16 floats: offset 8 leaves room for 56, not 64
        reply = json.loads(
            lib.call(
                "arcadedb_n_gb_bin",
                gb._gb,
                b"create_vertices",
                text.encode(),
                buf,
                x.size,
            )
        )
    assert not reply.get("ok")
    assert "outside the side buffer" in json.dumps(reply)
    assert db.query("sql", "SELECT count(*) AS c FROM Free").first().get("c") == 0


def test_the_encoder_in_isolation():
    x = _x(5, 4)
    rows = [{"id": i} for i in range(5)]
    text, buf, count = codec.encode_vertex_args("T", rows, ["v"], [x])
    assert json.loads(text) == {
        "type": "T",
        "rows": rows,
        "vecs": {"names": ["v"], "dims": [4], "offs": [0]},
    }
    assert count == 20 and bytes(buf) == x.tobytes()
    frozen = x.copy()
    frozen.setflags(
        write=False
    )  # a read-only matrix cannot be viewed writable: its bytes cross instead
    assert codec.encode_vertex_args("T", rows, ["v"], [frozen])[1] == x.tobytes()
    assert (
        codec._plain_rows(rows)
        and codec._plain_rows([{}, {}])
        and codec._plain_rows([])
    )
    assert not codec._plain_rows([{"a": np.float64(1)}])
    assert not codec._plain_rows([{"a": [1]}])
    assert not codec._plain_rows([{1: 2}])
