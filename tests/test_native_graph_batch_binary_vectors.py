"""GraphBatch float32 vectors cross the native library in a binary side buffer, not as number text.

``create_vertices`` rows and ``new_edges`` props that hold a float32 ``ndarray`` go through ``arcadedb_n_gb_bin`` (the float32 buffer is
read in Java through the ``["B", offset, count]`` tag, as ``arcadedb_n_exec_bin`` reads query parameters; a ``create_vertices`` column of
float32 arrays of one length crosses as one matrix described by ``"vecs"`` instead, see test_native_vertex_vector_crossings.py); everything else keeps the text
of ``arcadedb_n_gb``: a vector given as a list, a float64 array, a row with no vector. The stored values do not change: the same float32
as the ``"F"``-tagged text gave, bit for bit. A library without the entry (``lib.GB_BIN`` false) keeps the text path.

The calls are observed through ``lib.call`` (the ctypes door of the native library, as in test_native_tx_crossings.py).
Needs the native library, so conftest.py collects this file on the native backend only; the stored values of the same loads are also
pinned for both builds in test_float_vector_bulk.py.
"""

import json

import numpy as np
import pytest

from arcadedb._native import lib

N = 30


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE Vx")
    temp_db.command("sql", "CREATE PROPERTY Vx.id LONG")
    temp_db.command("sql", "CREATE PROPERTY Vx.embedding ARRAY_OF_FLOATS")
    temp_db.command("sql", "CREATE VERTEX TYPE Free")  # embedding undeclared
    temp_db.command("sql", "CREATE EDGE TYPE Ex")
    temp_db.command("sql", "CREATE PROPERTY Ex.embedding ARRAY_OF_FLOATS")
    return temp_db


class _Calls:
    """The ``(entry point, operation, JSON text, float count)`` of every GraphBatch call made while it is installed."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = lib.call

        def spy(name, *args):
            if name in ("arcadedb_n_gb", "arcadedb_n_gb_bin"):
                count = args[4] if name == "arcadedb_n_gb_bin" else 0
                self.calls.append((name, args[1].decode(), args[2].decode(), count))
            return real(name, *args)

        monkeypatch.setattr(lib, "call", spy)

    def entries(self, op):
        return [c[0] for c in self.calls if c[1] == op]


@pytest.fixture
def calls(db, monkeypatch):
    return _Calls(monkeypatch)


def _vectors(dim, n=N, seed=3):
    """float32 vectors a decimal text of 6 digits would not hold: bit-equality shows a path that rounded."""
    x = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    x[0, 0] = np.float32(-0.0)
    x[1, 0] = np.finfo(np.float32).max
    x[1, 1] = np.finfo(np.float32).tiny
    x[1, 2] = np.finfo(np.float32).smallest_subnormal
    return x


def _stored(db, type_name):
    """{id: stored vector as float32} for every record of the type."""
    query = (
        f"SELECT id, embedding FROM {type_name}"  # nosec B608 - test-owned type name
    )
    return {
        r["id"]: np.array(r["embedding"], dtype=np.float64).astype(np.float32)
        for r in db.query("sql", query).to_list()
    }


def _assert_bit_equal(stored, x):
    assert sorted(stored) == list(range(len(x)))
    for i in range(len(x)):
        assert stored[i].shape == x[i].shape
        assert stored[i].view(np.uint32).tolist() == x[i].view(np.uint32).tolist()


@pytest.mark.parametrize("dim", [64, 96, 128])
def test_float32_vectors_take_the_binary_entry_and_are_stored_bit_equal(db, calls, dim):
    x = _vectors(dim)
    with db.graph_batch() as gb:
        rids = gb.create_vertices(
            "Vx", [{"id": i, "embedding": x[i]} for i in range(N)]
        )
    assert len(rids) == N
    assert lib.GB_BIN is True  # the loaded library has the entry and the default is on
    assert calls.entries("create_vertices") == ["arcadedb_n_gb_bin"]
    _, _, text, count = next(c for c in calls.calls if c[1] == "create_vertices")
    assert count == N * dim
    assert (
        '"vecs"' in text and '"B"' not in text and '"F"' not in text
    )  # one matrix, no tag per row (#337)
    assert str(float(x[2, 0])) not in text  # the numbers are not in the text
    assert len(text) < N * 80  # a row is its other properties, not dim numbers
    _assert_bit_equal(_stored(db, "Vx"), x)


@pytest.mark.parametrize("dim", [64, 96, 128])
def test_the_binary_path_stores_what_the_text_path_stored(db, calls, monkeypatch, dim):
    x = _vectors(dim)
    rows = [{"id": i, "embedding": x[i]} for i in range(N)]
    with db.graph_batch() as gb:
        gb.create_vertices("Vx", rows)
    binary = _stored(db, "Vx")
    with db.transaction():
        db.command("sql", "DELETE FROM Vx")
    monkeypatch.setattr(lib, "GB_BIN", False)
    del calls.calls[:]
    with db.graph_batch() as gb:
        gb.create_vertices("Vx", rows)
    assert calls.entries("create_vertices") == ["arcadedb_n_gb"]
    text = next(c for c in calls.calls if c[1] == "create_vertices")[2]
    assert '"F"' in text and '"B"' not in text
    for i in range(N):
        assert (
            _stored(db, "Vx")[i].view(np.uint32).tolist()
            == binary[i].view(np.uint32).tolist()
        )


def test_a_library_without_the_entry_keeps_the_text_path(db, calls, monkeypatch):
    # the probe flag as the loader leaves it for a library built before arcadedb_n_gb_bin
    monkeypatch.setattr(lib, "GB_BIN", False)
    x = _vectors(64)
    with db.graph_batch() as gb:
        gb.create_vertices("Vx", [{"id": i, "embedding": x[i]} for i in range(N)])
        gb.create_vertex("Vx", id=N, embedding=x[0])
    assert set(c[0] for c in calls.calls) == {"arcadedb_n_gb"}
    stored = _stored(db, "Vx")
    assert stored.pop(N).view(np.uint32).tolist() == x[0].view(np.uint32).tolist()
    _assert_bit_equal(stored, x)


def test_rows_without_a_float32_array_keep_the_text_entry(db, calls):
    x = _vectors(64, 4)
    rows = [
        {"id": 0, "embedding": x[0].tolist()},  # a list of floats
        {"id": 1, "embedding": x[1].astype(np.float64)},  # float64 keeps its doubles
        {"id": 2, "embedding": None},
        {"id": 3, "embedding": np.zeros(0, dtype=np.float32)},  # empty: no buffer
        {"id": 4},
        None,
    ]
    with db.graph_batch() as gb:
        gb.create_vertices("Free", rows)
    assert calls.entries("create_vertices") == ["arcadedb_n_gb"]
    got = {
        r["id"]: r["embedding"]
        for r in db.query("sql", "SELECT id, embedding FROM Free").to_list()
    }
    assert got[0] == pytest.approx(x[0].tolist())
    assert got[1] == pytest.approx(x[1].astype(np.float64).tolist())
    assert got[2] is None and got[4] is None
    assert list(got[3] or []) == []
    assert db.query("sql", "SELECT count(*) AS c FROM Free").first().get("c") == 6


def test_none_and_list_vectors_store_as_without_the_binary_entry(
    db, calls, monkeypatch
):
    x = _vectors(64, 8)
    rows = [{"id": i, "embedding": x[i].tolist()} for i in range(8)]
    rows[5] = {"id": 5, "embedding": None}
    with db.graph_batch() as gb:
        gb.create_vertices("Vx", rows)
    assert calls.entries("create_vertices") == ["arcadedb_n_gb"]
    with_flag = _stored_or_none(db)
    with db.transaction():
        db.command("sql", "DELETE FROM Vx")
    monkeypatch.setattr(lib, "GB_BIN", False)
    with db.graph_batch() as gb:
        gb.create_vertices("Vx", rows)
    assert _stored_or_none(db) == with_flag
    assert with_flag[5] is None
    assert with_flag[0] == x[0].astype(np.float64).tolist()


def _stored_or_none(db):
    return {
        r["id"]: (
            None
            if r["embedding"] is None
            else np.array(r["embedding"], dtype=np.float32).astype(np.float64).tolist()
        )
        for r in db.query("sql", "SELECT id, embedding FROM Vx").to_list()
    }


def test_a_mix_of_vector_and_other_rows_keeps_every_offset(db, calls):
    x = _vectors(64, 9)
    rows = []
    for i in range(9):
        if i % 3 == 0:
            rows.append({"id": i, "embedding": None})
        elif i % 3 == 1:
            rows.append({"id": i, "embedding": x[i]})
        else:
            rows.append({"id": i, "embedding": x[i], "other": x[i][:3]})
    with db.graph_batch() as gb:
        gb.create_vertices("Vx", rows)
    assert calls.entries("create_vertices") == ["arcadedb_n_gb_bin"]
    stored = _stored_or_none(db)
    for i in range(9):
        if i % 3 == 0:
            assert stored[i] is None
        else:
            assert stored[i] == x[i].astype(np.float64).tolist()
    other = db.query("sql", "SELECT other FROM Vx WHERE id = 2").first().get("other")
    assert np.array(other, dtype=np.float32).tolist() == x[2][:3].tolist()


def test_every_chunk_of_a_load_has_its_own_buffer(db, calls):
    x = _vectors(64, 2500)
    with db.graph_batch() as gb:  # commit_every 1000: three crossings
        rids = gb.create_vertices(
            "Vx", [{"id": i, "embedding": x[i]} for i in range(len(x))]
        )
    assert len(rids) == len(x)
    assert calls.entries("create_vertices") == ["arcadedb_n_gb_bin"] * 3
    assert [c[3] for c in calls.calls if c[1] == "create_vertices"] == [
        64000,
        64000,
        32000,
    ]
    _assert_bit_equal(_stored(db, "Vx"), x)


def test_edge_props_with_a_float32_vector_take_the_binary_entry(db, calls):
    x = _vectors(64, 4)
    with db.graph_batch() as gb:
        rids = gb.create_vertices("Vx", 4)
        gb.new_edges(
            rids[:2], "Ex", rids[2:], [{"embedding": x[0]}, {"embedding": x[1]}]
        )
        gb.new_edge(rids[0], "Ex", rids[3], embedding=x[2])
    assert calls.entries("new_edges") == ["arcadedb_n_gb_bin"]
    assert calls.entries("new_edge") == ["arcadedb_n_gb_bin"]
    assert calls.entries("create_vertices") == ["arcadedb_n_gb"]  # a count, no rows
    got = sorted(
        np.array(r["embedding"], dtype=np.float32).view(np.uint32).tolist()
        for r in db.query("sql", "SELECT embedding FROM Ex").to_list()
    )
    want = sorted(x[i].view(np.uint32).tolist() for i in range(3))
    assert got == want


def test_a_tag_outside_the_side_buffer_is_refused(db):
    with db.graph_batch() as gb:
        text = json.dumps(
            {
                "type": "Vx",
                "rows": [{"id": 1, "embedding": {"\u0001": ["B", 0, 100]}}],
            }
        )
        buf = np.zeros(10, dtype=np.float32).tobytes()
        reply = json.loads(
            lib.call(
                "arcadedb_n_gb_bin",
                gb._gb,
                b"create_vertices",
                text.encode(),
                buf,
                10,
            )
        )
    assert not reply.get("ok")
    assert "outside the side buffer" in json.dumps(reply)
    assert db.query("sql", "SELECT count(*) AS c FROM Vx").first().get("c") == 0
