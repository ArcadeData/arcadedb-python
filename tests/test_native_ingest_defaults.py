"""The native build's ingest defaults, observed on the native library (conftest.py collects this file on the native backend only).

``insert_many`` commits every 1000 rows, ``graph_batch`` commits every 1000 and ``create_vertices`` sends (and commits) 1000 rows per
call into the image. An explicit value wins, and the semantics stay: rows that are already committed stay committed if a later chunk fails.
"""

import inspect

import pytest

import arcadedb
from arcadedb._native import extras
from arcadedb.exceptions import ArcadeDBError

_COUNT_SQL = {  # fixed statements, one for each type the tests create
    "Dup": "SELECT count(*) AS c FROM Dup",
    "V": "SELECT count(*) AS c FROM V",
    "W": "SELECT count(*) AS c FROM W",
}


def _count(db, type_name):
    return list(db.query("sql", _COUNT_SQL[type_name]))[0].get("c")


def _unique_type(db, name):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.k INTEGER")
    db.command("sql", f"CREATE INDEX ON {name} (k) UNIQUE")


def test_insert_many_signature_default_is_1000(temp_db):
    param = inspect.signature(temp_db.insert_many).parameters["commit_every"]
    assert param.default == 1000


@pytest.mark.parametrize(
    "kwargs, kept",
    [
        ({}, 2000),  # the default: 1000-row transactions, the third one fails
        ({"commit_every": 1000}, 2000),
        ({"commit_every": 1500}, 1500),  # an explicit value wins
        (
            {"commit_every": 10_000},
            0,
        ),  # the old default: nothing was committed before the failure
        ({"commit_every": 0}, 0),  # one transaction
    ],
)
def test_insert_many_commit_size_decides_what_a_failed_load_keeps(
    temp_db, kwargs, kept
):
    _unique_type(temp_db, "Dup")
    rows = [{"k": i} for i in range(2500)]
    rows[2400] = {"k": 0}  # a duplicate key inside the third 1000 rows
    with pytest.raises(ArcadeDBError):
        temp_db.insert_many("Dup", rows, **kwargs)
    assert temp_db.is_transaction_active() is False
    assert _count(temp_db, "Dup") == kept


class _Spy:
    def __init__(self, monkeypatch):
        self.created = (
            []
        )  # the option dicts that went to arcadedb_n_gb_create (and every other call that encodes a dict)
        self.vertex_calls = (
            []
        )  # the row or vertex count of each create_vertices crossing
        real_enc = extras._enc
        real_op = extras.NativeGraphBatch._op

        def enc(value):
            if isinstance(value, dict):
                self.created.append(dict(value))
            return real_enc(value)

        def op(this, prefix, name, **args):
            if name == "create_vertices":
                rows = args.get("rows")
                self.vertex_calls.append(
                    len(rows) if rows is not None else args.get("count")
                )
            return real_op(this, prefix, name, **args)

        monkeypatch.setattr(extras, "_enc", enc)
        monkeypatch.setattr(extras.NativeGraphBatch, "_op", op)

    def commit_every_given(self):
        return [d["commit_every"] for d in self.created if "commit_every" in d]


def _vertices(db, name):
    db.command("sql", f"CREATE VERTEX TYPE {name}")


def test_graph_batch_default_commit_every_is_1000(temp_db, monkeypatch):
    spy = _Spy(monkeypatch)
    _vertices(temp_db, "V")
    with temp_db.graph_batch(parallel_flush=False) as gb:
        gb.create_vertices("V", [{"id": i} for i in range(2500)])
        gb.create_vertices("V", 2500)
    assert spy.commit_every_given() == [1000]
    assert spy.vertex_calls == [1000, 1000, 500, 1000, 1000, 500]
    assert _count(temp_db, "V") == 5000


@pytest.mark.parametrize(
    "commit_every, calls",
    [
        (
            250,
            [250] * 10,
        ),  # an explicit value wins, for the engine and for the create_vertices chunk
        (
            0,
            [2500],
        ),  # 0: one commit per flush, and the create_vertices chunk is the old 10,000
    ],
)
def test_graph_batch_explicit_commit_every_wins(
    temp_db, monkeypatch, commit_every, calls
):
    spy = _Spy(monkeypatch)
    _vertices(temp_db, "W")
    with temp_db.graph_batch(commit_every=commit_every, parallel_flush=False) as gb:
        gb.create_vertices("W", [{"id": i} for i in range(2500)])
    assert spy.commit_every_given() == [commit_every]
    assert spy.vertex_calls == calls
    assert _count(temp_db, "W") == 2500


def test_graph_batch_default_constant_and_edges_chunk():
    assert extras.NativeGraphBatch.DEFAULT_COMMIT_EVERY == 1000
    assert (
        extras.NativeGraphBatch._BULK_CHUNK == 10_000
    )  # new_edges is not part of the measured change
    assert arcadedb.__backend__ == "native"
