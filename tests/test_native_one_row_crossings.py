"""A created or updated record costs the fewest native crossings (conftest.py collects this file on the native backend only).

``save()`` of a new record and ``Vertex.new_edge`` read the row the statement returns, so they used the lazy form of the command
(``arcadedb_n_cmd``: a cursor id, then a second crossing, ``arcadedb_n_next_bin``, for the row they always read). They now use the
form whose reply carries the first rows (``arcadedb_n_exec``): one crossing. ``save()`` of a saved record spent four crossings (the
transaction check, the SELECT of the stale-record check, and two for the UPDATE) and now spends three. The stale-record check still
reads the stored record first: the engine's reply to an UPDATE comes after the write, and the check must refuse before writing
anything inside the caller's transaction, so it cannot be read from the update's own result.

The counts are crossings (the ctypes doors ``cmd``, ``call``, ``call_bin``, ``exec_query`` and the batched cursor close), not seconds:
deterministic. The oracle for the values and the errors is the lazy form itself (``db.command(...).first()``), kept for every other caller.
"""

import datetime as dt
from decimal import Decimal

import numpy as np
import pytest

from arcadedb import ArcadeDBError
from arcadedb._native import lib


class _Crossings:
    """Every call into the native library made while installed, by entry point."""

    def __init__(self, monkeypatch):
        self.names = []
        for attr in ("cmd", "call", "call_bin", "exec_query"):
            monkeypatch.setattr(lib, attr, self._spy(attr, getattr(lib, attr)))
        real_close = lib._lib.arcadedb_n_rs_close_many

        def close_many(*args):
            self.names.append("arcadedb_n_rs_close_many")
            return real_close(*args)

        monkeypatch.setattr(lib._lib, "arcadedb_n_rs_close_many", close_many)

    def _spy(self, attr, real):
        def spy(*args, **kwargs):
            self.names.append(args[0] if attr in ("call", "call_bin") else attr)
            return real(*args, **kwargs)

        return spy

    def reset(self):
        lib.flush_closes()  # a cursor closed earlier is not part of the call measured
        del self.names[:]


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE D")
    temp_db.command("sql", "CREATE VERTEX TYPE V")
    temp_db.command("sql", "CREATE EDGE TYPE E")
    return temp_db


@pytest.fixture
def crossings(db, monkeypatch):
    return _Crossings(monkeypatch)


def _plain(values):
    return {k: v for k, v in values.items() if not k.startswith("@")}


# -- create


def test_saving_a_new_document_is_one_crossing(db, crossings):
    with db.transaction():
        doc = db.new_document("D")
        doc.set("a", 1)
        crossings.reset()
        doc.save()
        assert crossings.names == [
            "exec_query"
        ]  # was: arcadedb_n_cmd, then arcadedb_n_next_bin
    assert doc.get_identity() is not None


def test_saving_a_new_vertex_with_a_float_array_is_one_crossing(db, crossings):
    with db.transaction():
        vertex = db.new_vertex("V")
        vertex.set("v", np.arange(16, dtype=np.float32))
        crossings.reset()
        vertex.save()
        assert crossings.names == [
            "arcadedb_n_exec_bin"
        ]  # the float side buffer: the long path, still one crossing


def test_new_edge_is_one_crossing_with_and_without_properties(db, crossings):
    with db.transaction():
        a, b = db.new_vertex("V"), db.new_vertex("V")
        a.save()
        b.save()
        crossings.reset()
        plain = a.new_edge("E", b)
        assert crossings.names == ["exec_query"]
        crossings.reset()
        weighted = a.new_edge("E", b, w=1.5, tag="x")
        assert crossings.names == ["exec_query"]
    assert plain.get_identity() != weighted.get_identity()
    assert weighted.get("w") == 1.5 and weighted.get("tag") == "x"


CREATE_CASES = {
    "ints and text": {"a": 1, "b": "text", "c": None, "d": True, "e": 2.5},
    "non-ascii": {"s": "café 日本語 \U0001f600"},
    "date and time": {
        "d": dt.date(2026, 11, 9),
        "t": dt.datetime(2026, 11, 9, 12, 30, 15, 123000),
    },
    "decimal": {"x": Decimal("1.50")},
    "list and map": {"l": [1, 2, 3], "m": {"k": [1.5, "v"]}},
    "float array": {"v": np.arange(16, dtype=np.float32)},
    "nothing": {},
}


@pytest.mark.parametrize("case", list(CREATE_CASES))
def test_a_created_record_has_the_values_the_lazy_form_gives(db, case):
    values = CREATE_CASES[case]
    with db.transaction():
        doc = db.new_document("D")
        for name, value in values.items():
            doc.set(name, value)
        doc.save()
        # the oracle: the same statement through the lazy command form every other caller still uses
        want = db.command("sql", "INSERT INTO D CONTENT :row", {"row": values}).first()
    got = db.lookup_by_rid(str(doc.get_identity()))
    assert doc.get_type_name() == want.get_element().get_type_name() == "D"
    assert _plain(doc.to_dict()) == _plain(want.to_dict()) == _plain(got.to_dict())


def test_a_create_that_the_engine_refuses_fails_with_the_lazy_forms_message(db):
    db.command("sql", "CREATE PROPERTY D.k STRING")
    db.command("sql", "CREATE INDEX ON D (k) UNIQUE")
    with db.transaction():
        db.command("sql", "INSERT INTO D SET k = 'dup'").close()
    with pytest.raises(ArcadeDBError) as lazy:
        with db.transaction():
            db.command(
                "sql", "INSERT INTO D CONTENT :row", {"row": {"k": "dup"}}
            ).first()
    with pytest.raises(ArcadeDBError) as eager:
        with db.transaction():
            doc = db.new_document("D")
            doc.set("k", "dup")
            doc.save()
    assert str(eager.value) == str(lazy.value)
    assert type(eager.value) is type(lazy.value)


# -- update


def _saved(db, **values):
    with db.transaction():
        doc = db.new_document("D")
        for name, value in values.items():
            doc.set(name, value)
        doc.save()
    return str(doc.get_identity())


def test_updating_a_saved_record_is_three_crossings(db, crossings):
    rid = _saved(db, a=1, name="x")
    with db.transaction():
        record = db.lookup_by_rid(rid).modify()
        record.set("a", 2)
        crossings.reset()
        record.save()
        # the transaction check, the SELECT of the stale-record check, the UPDATE (was four: the UPDATE took two)
        assert crossings.names == ["arcadedb_n_tx", "exec_query", "exec_query"]
    assert db.lookup_by_rid(rid).get("a") == 2


def test_an_update_that_reads_its_row_back_is_three_crossings(db, crossings):
    rid = _saved(db, a=1)
    with db.transaction():
        record = db.lookup_by_rid(rid).modify()
        record.set("brand_new", "value")  # a new property is read back (RETURN AFTER)
        crossings.reset()
        record.save()
        assert crossings.names == ["arcadedb_n_tx", "exec_query", "exec_query"]
    assert db.lookup_by_rid(rid).get("brand_new") == "value"


def test_a_stale_record_is_still_refused_before_anything_is_written(db):
    rid = _saved(db, a=1, b=1)
    mine = db.lookup_by_rid(rid).modify()
    with db.transaction():
        db.command(
            "sql", f"UPDATE {rid} SET b = 9"
        ).close()  # nosec B608 - a record id this test read
    mine.set("a", 2)
    with pytest.raises(ArcadeDBError, match="ConcurrentModificationException"):
        with db.transaction():
            mine.save()
    stored = db.lookup_by_rid(rid)
    assert (stored.get("a"), stored.get("b")) == (1, 9)  # nothing of mine was written


def test_an_update_outside_a_transaction_still_raises_before_reading(db, crossings):
    rid = _saved(db, a=1)
    record = db.lookup_by_rid(rid).modify()
    record.set("a", 2)
    crossings.reset()
    with pytest.raises(ArcadeDBError, match="Transaction not begun"):
        record.save()
    assert crossings.names == ["arcadedb_n_tx"]
