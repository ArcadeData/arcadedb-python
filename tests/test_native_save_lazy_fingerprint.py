"""Native build: save() of a record refuses to overwrite a record that changed after it was read, and the check costs a read nothing.

The fingerprint of the values a record held as read is taken lazily, at ``modify()``, at the first ``set()`` of a saved record or at
``save()``, never when a read decodes a row into a record. The seam is ``arcadedb._native.records._fingerprint``: these tests count its
calls. The refusal itself is also tested in test_graph_api.py, which runs on both builds.
"""

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pytest

import arcadedb
from arcadedb._native import records


@pytest.fixture
def fingerprints(monkeypatch):
    """The list of calls to the fingerprint function (it still computes the real value)."""
    calls = []
    real = records._fingerprint

    def counting(props):
        calls.append(dict(props))
        return real(props)

    monkeypatch.setattr(records, "_fingerprint", counting)
    return calls


def _create(db):
    db.command("sql", "CREATE VERTEX TYPE V")
    db.command("sql", "CREATE EDGE TYPE E")
    with db.transaction():
        a = db.new_vertex("V")
        a.set("a", 1)
        a.set("b", 1)
        a.save()
        b = db.new_vertex("V")
        b.set("a", 2)
        b.save()
        a.new_edge("E", b, w=1.5)
        return str(a.get_identity())


def _bump_b_sql(rid):
    """The statement that changes b of the record at ``rid`` (a RID the test read)."""
    return f"UPDATE {rid} SET b = 2"  # nosec B608 - rid is a record id the test read


def _stored(db, rid):
    select_sql = (
        f"SELECT a, b FROM {rid}"  # nosec B608 - rid is a record id the test read
    )
    row = list(db.query("sql", select_sql))[0]
    return row.get("a"), row.get("b")


def test_reads_never_compute_the_fingerprint(temp_db_path, fingerprints):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)
        fingerprints.clear()

        rows = list(db.query("sql", "SELECT FROM V"))
        records_read = [row.get_element() for row in rows]
        by_rid = db.lookup_by_rid(rid)
        vertex = records_read[0]
        edges = vertex.get_out_edges("E") + vertex.get_in_edges("E")
        ends = [edge.get_in() for edge in edges] + [edge.get_out() for edge in edges]
        cypher = [
            row.get_element() for row in db.query("opencypher", "MATCH (n:V) RETURN n")
        ]
        assert len(records_read) == 2 and by_rid is not None and ends and cypher
        assert fingerprints == []  # nothing was fingerprinted by any read path

        mutable = by_rid.modify()
        assert (
            len(fingerprints) == 1
        )  # modify() takes it, once, from the values the record holds as read
        assert fingerprints[0] == by_rid.to_dict()
        by_rid.modify()
        assert len(fingerprints) == 1  # and caches it on the record

        mutable.set("a", 10)  # the snapshot is already there: no work
        assert len(fingerprints) == 1
        with db.transaction():
            mutable.save()  # the stored values to compare with, and the stored values after the write
        assert len(fingerprints) == 3
        assert _stored(db, rid) == (10, 1)


def test_modify_then_save_after_a_concurrent_change_still_raises(
    temp_db_path, fingerprints
):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)

        mutable = db.lookup_by_rid(rid).modify()
        with (
            db.transaction()
        ):  # another transaction changes the stored record after the read
            db.command("sql", _bump_b_sql(rid))
        mutable.set("a", 10)
        with pytest.raises(arcadedb.ArcadeDBError) as excinfo:
            with db.transaction():
                mutable.save()
        assert "ConcurrentModificationException" in str(excinfo.value)
        assert (
            excinfo.value.native["retry"] == "need_retry"
        )  # run_in_transaction retries it
        assert _stored(db, rid) == (1, 2)  # nothing of the refused save is stored

        # a fresh read saves, in a transaction
        fresh = db.lookup_by_rid(rid).modify()
        fresh.set("a", 10)
        with db.transaction():
            fresh.save()
        assert _stored(db, rid) == (10, 2)


def test_a_change_between_the_read_and_modify_is_not_lost_when_modify_comes_late(
    temp_db_path,
):
    """The fingerprint is of the values read, not of what is stored when modify() runs."""
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)

        read = db.lookup_by_rid(rid)
        with db.transaction():
            db.command("sql", _bump_b_sql(rid))
        mutable = (
            read.modify()
        )  # late: the stored record is already newer than the values held
        mutable.set("a", 10)
        with pytest.raises(arcadedb.ArcadeDBError):
            with db.transaction():
                mutable.save()
        assert _stored(db, rid) == (1, 2)


def test_a_record_created_here_then_changed_elsewhere_is_refused(
    temp_db_path, fingerprints
):
    """A record this object created has no fingerprint until its first change: taken before that change, from the stored values."""
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE D")
        with db.transaction():
            created = db.new_document("D")
            created.set("a", 1)
            created.set("b", 1)
            created.save()
        assert (
            fingerprints == []
        )  # creating and saving a new record fingerprints nothing
        rid = str(created.get_identity())
        with db.transaction():
            db.command("sql", _bump_b_sql(rid))
        created.set("a", 9)
        with pytest.raises(arcadedb.ArcadeDBError):
            with db.transaction():
                created.save()
        assert _stored(db, rid) == (1, 2)


# ---- the fingerprint itself: a direct byte encoding, equal for equal values whatever path read them


def _medium_record():
    """About what a medium record holds: 20 mixed properties and a 768-float vector (float32 values, as the engine stores them)."""
    rng = np.random.default_rng(7)
    props = {
        "i": 12,
        "big": 2**40,
        "d": 3.5,
        "neg_zero": -0.0,
        "t": True,
        "f": False,
        "n": None,
        "s": "text with é and 日",
        "empty": "",
        "day": date(2020, 1, 2),
        "at": datetime(2020, 1, 2, 3, 4, 5, 678000),
        "dec": Decimal("12.50"),
        "ints": [1, 2, 3],
        "mixed": [1, "a", 2.5, None],
        "strs": ["x", "y"],
        "map": {"b": 1, "a": [0.5, 0.25], "c": {"z": "q"}},
        "doubles": [0.1, 0.2],
        "blob": b"\x00\x01",
        "neg": -7,
        "zero": 0,
    }
    props["vec"] = rng.random(768).astype(np.float32).tolist()
    assert len(props) == 21
    return props


def test_every_property_and_every_vector_element_changes_the_fingerprint():
    props = _medium_record()
    base = records._fingerprint(props)
    assert records._fingerprint(dict(props)) == base
    seen = {base}
    for name in props:
        if name == "vec":
            continue
        changed = dict(props)
        value = props[name]
        if isinstance(value, bool):
            changed[name] = not value
        elif value is None:
            changed[name] = 0
        elif isinstance(value, (int, float)):
            changed[name] = value + 1
        elif isinstance(value, str):
            changed[name] = value + "!"
        elif isinstance(value, (datetime, date)):
            changed[name] = value.replace(year=2021)
        elif isinstance(value, Decimal):
            changed[name] = Decimal("12.5")  # same number, other scale: another value
        elif isinstance(value, bytes):
            changed[name] = value + b"\x02"
        elif isinstance(value, list):
            changed[name] = value + [value[0]]
        else:
            changed[name] = {**value, "new": 1}
        fingerprint = records._fingerprint(changed)
        assert fingerprint != base, name
        seen.add(fingerprint)
    for index in (0, 1, 383, 767):
        changed = dict(props)
        changed["vec"] = list(props["vec"])
        changed["vec"][index] = changed["vec"][index] + 0.5
        fingerprint = records._fingerprint(changed)
        assert fingerprint != base, index
        seen.add(fingerprint)
    shorter = {**props, "vec": props["vec"][:-1]}
    assert records._fingerprint(shorter) != base
    assert len(seen) == len(props) + 4


def test_the_fingerprint_does_not_depend_on_order_or_on_list_versus_array():
    props = _medium_record()
    base = records._fingerprint(props)
    reordered = dict(reversed(list(props.items())))
    reordered["map"] = {
        "c": {"z": "q"},
        "a": [0.5, 0.25],
        "b": 1,
    }
    assert records._fingerprint(reordered) == base
    # a vector read as a numpy float32 array is the vector read as a list of Python floats
    as_array = {**props, "vec": np.asarray(props["vec"], dtype=np.float32)}
    assert records._fingerprint(as_array) == base
    assert (
        records._fingerprint({**props, "ints": (1, 2, 3)}) == base
    )  # a tuple is a list
    # the types stay apart: 1, 1.0 and True are three values; so are a list, a set, a map, a string and bytes
    kinds = [1, 1.0, True, "1", [1], {1}, {"1": 1}, b"1"]
    assert len({records._fingerprint({"k": k}) for k in kinds}) == len(kinds)
    assert records._fingerprint({"k": {1, 2}}) == records._fingerprint({"k": {2, 1}})
    assert records._fingerprint({"k": float("nan")}) == records._fingerprint(
        {"k": float("nan")}
    )
    assert records._fingerprint({"k": 0.0}) != records._fingerprint({"k": -0.0})
    assert records._fingerprint({"k": 2**70}) != records._fingerprint({"k": 2**70 + 1})
    assert records._fingerprint({"k": [2**70, 1]}) != records._fingerprint(
        {"k": [2**70, 2]}
    )


def test_one_record_has_one_fingerprint_on_every_read_path(temp_db_path):
    """lookup_by_rid, a SELECT and the row UPDATE .. RETURN AFTER gives back decode to the same fingerprint, and a change to the stored
    record (one vector element) changes it."""
    props = _medium_record()
    vec = props.pop("vec")
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE M")
        db.command("sql", "CREATE PROPERTY M.vec ARRAY_OF_FLOATS")
        with db.transaction():
            vertex = db.new_vertex("M")
            for name, value in {**props, "vec": vec}.items():
                vertex.set(name, value)
            vertex.save()
            rid = str(vertex.get_identity())

        by_rid = db.lookup_by_rid(rid)
        select_sql = (
            f"SELECT FROM {rid}"  # nosec B608 - rid is a record id the test read
        )
        selected = db.query("sql", select_sql).first()
        with db.transaction():
            after = db.command(
                "sql",
                f"UPDATE {rid} MERGE :row RETURN AFTER",  # nosec B608 - rid is a record id the test read
                {"row": {"i": props["i"]}},
            ).first()
        one = records._fingerprint(by_rid._props)
        assert records._fingerprint(selected._props) == one
        assert records._fingerprint(after._props) == one
        assert len(by_rid._props["vec"]) == 768

        changed = list(by_rid._props["vec"])
        changed[500] = changed[500] + 0.5
        with db.transaction():
            db.command(
                "sql",
                f"UPDATE {rid} SET vec = :v",  # nosec B608 - rid is a record id the test read
                {"v": changed},
            ).close()
        assert records._fingerprint(db.lookup_by_rid(rid)._props) != one
