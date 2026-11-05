"""Native build: save() of a record refuses to overwrite a record that changed after it was read, and the check is cheap.

A record keeps a copy of the values it held as read, taken lazily (at ``modify()`` or at the first ``set()`` of a saved record, never when
a read decodes a row), and ``save()`` compares the stored row with that copy by value. After the UPDATE the values just sent are the new
copy when the engine stores them exactly as sent, so the row is not read back; otherwise the UPDATE returns the stored row. The seams the
tests count are ``arcadedb._native.records._snap`` (the copy) and the statement texts handed to the engine. The refusal itself is also
tested in test_graph_api.py, which runs on both builds.
"""

import copy
import hashlib
import math
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

import arcadedb
from arcadedb._native import records
from arcadedb._native.database import NativeDatabase


@pytest.fixture
def snaps(monkeypatch):
    """The list of calls to the copy function (it still computes the real copy)."""
    calls = []
    real = records._snap

    def counting(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(records, "_snap", counting)
    return calls


@pytest.fixture
def hashes(monkeypatch):
    """Every hash the save path could compute: blake2b anywhere, and the record fingerprint of earlier versions of this module (counted when a
    version has one)."""
    calls = []
    real_hash = hashlib.blake2b

    def counting_hash(*args, **kwargs):
        calls.append(("blake2b", args))
        return real_hash(*args, **kwargs)

    monkeypatch.setattr(hashlib, "blake2b", counting_hash)
    real_fingerprint = getattr(records, "_fingerprint", None)
    if real_fingerprint is not None:

        def counting_fingerprint(props):
            calls.append(("fingerprint", props))
            return real_fingerprint(props)

        monkeypatch.setattr(records, "_fingerprint", counting_fingerprint)
    return calls


@pytest.fixture
def texts(monkeypatch):
    """Every statement text handed to the engine, in order, while the fixture is active."""
    seen = []
    original = NativeDatabase._exec

    def recording(self, flags, prefix, language, command, args):
        seen.append(command)
        return original(self, flags, prefix, language, command, args)

    monkeypatch.setattr(NativeDatabase, "_exec", recording)
    return seen


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


def test_reads_never_copy_the_values(temp_db_path, snaps):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)
        snaps.clear()

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
        assert snaps == []  # nothing was copied by any read path

        mutable = by_rid.modify()
        assert len(snaps) == 1  # modify() copies, once, the values the record holds
        assert snaps[0] == by_rid.to_dict()
        by_rid.modify()
        assert len(snaps) == 1  # and caches the copy on the record

        mutable.set("a", 10)  # the copy is already there: no work
        assert len(snaps) == 1
        with db.transaction():
            mutable.save()
        assert _stored(db, rid) == (10, 1)


def test_no_hash_is_computed_on_the_save_path(temp_db_path, hashes):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)
        mutable = db.lookup_by_rid(rid).modify()
        mutable.set("a", 10)
        with db.transaction():
            mutable.save()
            mutable.set("b", 5)
            mutable.save()
        again = db.lookup_by_rid(rid).modify()
        again.set("name", "x")  # a new property: the read-back path
        with db.transaction():
            again.save()
        assert _stored(db, rid) == (10, 5)
        assert hashes == []


def test_modify_then_save_after_a_concurrent_change_still_raises(temp_db_path):
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
    """The copy is of the values read, not of what is stored when modify() runs."""
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


def test_a_record_created_here_then_changed_elsewhere_is_refused(temp_db_path, snaps):
    """A record this object created has no copy until its first change: taken before that change, from the stored values."""
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE D")
        with db.transaction():
            created = db.new_document("D")
            created.set("a", 1)
            created.set("b", 1)
            created.save()
        assert snaps == []  # creating and saving a new record copies nothing
        rid = str(created.get_identity())
        with db.transaction():
            db.command("sql", _bump_b_sql(rid))
        created.set("a", 9)
        with pytest.raises(arcadedb.ArcadeDBError):
            with db.transaction():
                created.save()
        assert _stored(db, rid) == (1, 2)


# ---- the copy of the values as read: nothing the caller does to the record changes it


def _vector(n=768, seed=7):
    return np.random.default_rng(seed).random(n).astype(np.float32).tolist()


def _rich_record(db):
    """A vertex with a scalar, a list, a map and a 768-float vector (float32 values, as the engine stores them); returns its RID."""
    db.command("sql", "CREATE VERTEX TYPE M")
    db.command("sql", "CREATE PROPERTY M.vec ARRAY_OF_FLOATS")
    with db.transaction():
        vertex = db.new_vertex("M")
        vertex.set("name", "n")
        vertex.set("tags", ["x", "y"])
        vertex.set("meta", {"k": [1, 2], "z": {"q": "r"}})
        vertex.set("vec", _vector())
        vertex.save()
        return str(vertex.get_identity())


def test_the_copy_is_not_changed_by_set_or_by_changing_a_nested_value_in_place(
    temp_db_path,
):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _rich_record(db)
        read = db.lookup_by_rid(rid)
        mutable = read.modify()
        as_read = copy.deepcopy(mutable.to_dict())
        assert mutable._base == as_read
        assert mutable._base is not mutable._props
        for name in ("tags", "meta", "vec"):
            assert mutable._base[name] is not mutable._props[name]  # copied, not shared

        mutable.set("name", "changed")  # set() writes into the record's own dict
        mutable.set("tags", ["other"])
        assert mutable._base == as_read

        mutable.get("meta")["k"].append(3)  # a nested list changed in place
        mutable.get("meta")["z"]["new"] = 1
        mutable.get("vec")[5] = 0.5
        assert mutable.get("vec")[5] == 0.5 and mutable._props["meta"]["k"] == [1, 2, 3]
        assert mutable._base == as_read
        assert read.modify()._base == as_read  # the read object's copy is untouched too

        # the changes in place are the record's changes: the save writes them, and the object's next save is not a conflict
        with db.transaction():
            mutable.save()
        fresh = db.lookup_by_rid(rid)
        assert fresh.get("meta") == {"k": [1, 2, 3], "z": {"q": "r", "new": 1}}
        assert fresh.get("vec")[5] == 0.5 and fresh.get("tags") == ["other"]
        mutable.get("tags").append("again")
        with db.transaction():
            mutable.save()
        assert db.lookup_by_rid(rid).get("tags") == ["other", "again"]


def test_a_concurrent_change_of_a_nested_vector_is_detected(temp_db_path):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _rich_record(db)
        mutable = db.lookup_by_rid(rid).modify()
        other = _vector()
        other[767] = float(np.float32(other[767] + 0.25))  # one element, the last
        with db.transaction():
            db.command(
                "sql",
                f"UPDATE {rid} SET vec = :v",  # nosec B608 - rid is a record id the test read
                {"v": other},
            ).close()
        mutable.set("name", "mine")
        with pytest.raises(arcadedb.ArcadeDBError) as excinfo:
            with db.transaction():
                mutable.save()
        assert "ConcurrentModificationException" in str(excinfo.value)
        fresh = db.lookup_by_rid(rid)
        assert fresh.get("name") == "n" and fresh.get("vec")[767] == other[767]

        # and a nested map changed by another transaction
        mutable = db.lookup_by_rid(rid).modify()
        with db.transaction():
            db.command(
                "sql",
                f"UPDATE {rid} SET meta = :m",  # nosec B608 - rid is a record id the test read
                {"m": {"k": [1, 2], "z": {"q": "R"}}},
            ).close()
        mutable.set("name", "mine")
        with pytest.raises(arcadedb.ArcadeDBError):
            with db.transaction():
                mutable.save()


def test_a_vetoed_update_leaves_the_copy_as_it_was(temp_db_path):
    """A before_update veto writes nothing and returns no error; the next save, after the veto is lifted, is not a conflict."""
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE D")
        with db.transaction():
            doc = db.new_document("D")
            doc.set("name", "a")
            doc.save()
            rid = str(doc.get_identity())
        veto = db.on("before_update", lambda r: r.get("name") != "frozen", "D")
        mutable = db.lookup_by_rid(rid).modify()
        mutable.set("name", "frozen")
        with db.transaction():
            mutable.save()
        assert db.lookup_by_rid(rid).get("name") == "a"  # nothing written
        veto.remove()
        mutable.set("name", "thawed")
        with db.transaction():
            mutable.save()
        assert db.lookup_by_rid(rid).get("name") == "thawed"


# ---- which values are stored exactly as sent (the parity test behind records._stored_as_sent)

_F32 = _vector(8, seed=3)
_NOT_F32 = [0.1, 0.2, 0.3]
_MS = datetime(2021, 1, 2, 3, 4, 5, 678000)
_US = datetime(2021, 1, 2, 3, 4, 5, 678901)

# (id, declared type of the property or None, value stored first, value sent by set(), True when the read-back is skipped)
CASES = [
    ("str", "STRING", "a", "héllo 日本", True),
    ("str_empty", "STRING", "a", "", True),
    ("bool", "BOOLEAN", True, False, True),
    ("int", "INTEGER", 1, 2, True),
    ("int_negative_long", "LONG", 1, -(2**40), True),
    ("int_untyped", None, 1, 2**40, True),
    ("float_f32_exact", "DOUBLE", 0.5, 0.25, True),
    ("float_neg_zero", "DOUBLE", 0.5, -0.0, True),
    ("float_untyped_f32_exact", None, 0.5, 0.75, True),
    ("float_double", "DOUBLE", 0.5, 0.1, False),  # fine in a DOUBLE, rounded by a FLOAT
    ("float_in_FLOAT_property", "FLOAT", 0.5, 0.1, False),  # reads back 0.10000000149
    ("float_nan", "DOUBLE", 0.5, float("nan"), False),
    ("float_inf", "DOUBLE", 0.5, float("inf"), True),
    ("date", "DATE", date(2020, 1, 2), date(2021, 3, 4), True),
    ("datetime_ms", "DATETIME", datetime(2020, 1, 2, 3, 4, 5), _MS, False),
    ("datetime_us", "DATETIME", datetime(2020, 1, 2, 3, 4, 5), _US, False),
    ("decimal", "DECIMAL", Decimal("1.5"), Decimal("12.50"), True),
    ("set_into_list_property", "LIST", [1], {1, 2, 3}, False),  # reads back as a list
    (
        "bytes",
        "BINARY",
        b"\x01",
        b"\x00\x01\xff",
        False,
    ),  # reads back as a list of ints
    ("bytes_untyped", None, 1, b"\x00\x01\xff", False),
    ("set_untyped", None, 1, {1, 2, 3}, False),
    (
        "datetime_aware_untyped",
        None,
        1,
        datetime(2021, 1, 2, 3, 4, 5, 678000, tzinfo=timezone.utc),
        False,
    ),
    ("datetime_untyped", None, datetime(2020, 1, 2), _US, False),
    ("numpy_float32_scalar", "DOUBLE", 0.5, np.float32(0.25), True),
    ("numpy_float64_scalar_not_f32", "DOUBLE", 0.5, np.float64(0.1), False),
    ("numpy_int64_scalar", "LONG", 1, np.int64(5), True),
    ("numpy_bool", "BOOLEAN", True, np.bool_(False), True),
    ("list_float32_values", "ARRAY_OF_FLOATS", _F32, _F32[::-1], True),
    ("list_float32_values_untyped", None, _F32, _F32[::-1], True),
    ("list_double_values", "ARRAY_OF_FLOATS", _F32, _NOT_F32, False),  # read as float32
    ("list_double_values_in_LIST", "LIST", [0.5, 0.25], _NOT_F32, False),
    (
        "ndarray_float32",
        "ARRAY_OF_FLOATS",
        _F32,
        np.asarray(_F32[::-1], dtype=np.float32),
        True,
    ),
    (
        "ndarray_float64_f32_values",
        "ARRAY_OF_FLOATS",
        _F32,
        np.asarray(_F32[::-1], dtype=np.float64),
        True,
    ),
    ("ndarray_float64_not_f32", "ARRAY_OF_FLOATS", _F32, np.asarray(_NOT_F32), False),
    ("ndarray_int", "LIST", [1, 2], np.array([3, 4, 5]), True),
    ("list_ints", "LIST", [1], [1, 2, 3], True),
    ("list_strs", "LIST", ["a"], ["b", "c"], True),
    ("list_empty", "LIST", [1], [], True),
    ("list_from_empty", "LIST", [], [1], False),  # the element kind was never read
    ("list_other_element_kind", "LIST", [1, 2], ["a"], False),
    ("list_with_none", "LIST", [1, None], [None, 2], True),
    ("tuple", "LIST", [1], (1, 2), True),
    ("list_of_maps", "LIST", [{"a": 1}], [{"a": 1}, {"b": [1, 2]}], True),
    ("map", "MAP", {"a": 1}, {"b": 1, "a": [0.5, 0.25], "c": {"z": "q"}}, True),
    ("map_with_datetime", "MAP", {"a": 1}, {"d": _MS}, False),
    ("map_with_set", "MAP", {"a": 1}, {"s": {1, 2}}, False),
    (
        "list_of_datetimes",
        "LIST",
        [datetime(2020, 1, 2)],
        [datetime(2021, 1, 2)],
        False,
    ),
    ("int_into_DOUBLE_property", "DOUBLE", 1.5, 3, False),  # reads back as 3.0
    ("float_into_INTEGER_property", "INTEGER", 1, 3.0, False),  # reads back as 3
    ("str_into_INTEGER_property", "INTEGER", 1, "3", False),
    ("str_into_DATE_property", "DATE", date(2020, 1, 2), "2021-01-02", False),
    ("int_into_STRING_property", "STRING", "a", 5, False),
    ("none_before", "STRING", None, "x", False),
    ("none_after", "STRING", "a", None, False),
]


def _updates(texts):
    return [text for text in texts if text.startswith("UPDATE")]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_a_value_is_stored_as_sent_exactly_when_the_read_back_is_skipped(
    temp_db_path, texts, case
):
    """For every kind: the baseline after a save equals what a fresh read returns (so the same object's next save is not a conflict), and the
    statement has no RETURN AFTER exactly for the kinds the engine stores as sent."""
    _, declared, first, sent, skips_read_back = case
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE D")
        if declared:
            db.command("sql", f"CREATE PROPERTY D.p {declared}")
        with db.transaction():
            doc = db.new_document("D")
            doc.set("o", 1)
            doc.set("p", first)
            doc.save()
            rid = str(doc.get_identity())

        mutable = db.lookup_by_rid(rid).modify()
        mutable.set("p", sent)
        del texts[:]
        with db.transaction():
            mutable.save()
        updates = _updates(texts)
        assert len(updates) == 1
        assert ("RETURN AFTER" not in updates[0]) is skips_read_back, updates[0]

        stored = db.lookup_by_rid(rid)._props
        assert sorted(stored) == ["o", "p"]
        assert records._same(
            stored, mutable._base
        ), "the baseline differs from what a read returns"

        # the same object saves again, twice (a baseline that differed from the stored row would be refused here)
        mutable.set("o", 2)
        with db.transaction():
            mutable.save()
            mutable.set("o", 3)
            mutable.save()
        assert db.lookup_by_rid(rid).get("o") == 3
        assert records._same(db.lookup_by_rid(rid)._props, mutable._base)


def test_a_new_property_is_read_back_and_then_known(temp_db_path, texts):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)
        mutable = db.lookup_by_rid(rid).modify()
        mutable.set(
            "fresh", "x"
        )  # no value of this property was read: its type is unknown
        del texts[:]
        with db.transaction():
            mutable.save()
        assert "RETURN AFTER" in _updates(texts)[0]
        mutable.set("fresh", "y")  # now it was read as a str: exact
        del texts[:]
        with db.transaction():
            mutable.save()
        assert "RETURN AFTER" not in _updates(texts)[0]
        assert db.lookup_by_rid(rid).get("fresh") == "y"


def test_a_stored_vector_reads_as_a_list_of_python_floats(temp_db_path):
    """The codec decodes a float vector to a list (``tolist()`` of the float32 buffer), never to an array: the copy and the comparison
    work on lists, and an array that is sent is copied as the list it reads back as."""
    with arcadedb.create_database(temp_db_path) as db:
        rid = _rich_record(db)
        stored = db.lookup_by_rid(rid).get("vec")
        assert type(stored) is list and set(map(type, stored)) == {float}
        array = np.asarray(stored, dtype=np.float32)
        snapped = records._snap(array)
        assert type(snapped) is list and snapped == stored
        assert records._same(snapped, stored)
        mutable = db.lookup_by_rid(rid).modify()
        mutable.set("vec", array[::-1].copy())
        with db.transaction():
            mutable.save()
        assert db.lookup_by_rid(rid).get("vec") == stored[::-1]
        assert records._same(mutable._base["vec"], stored[::-1])


# ---- the two helpers on their own


def test_same_is_strict_about_types_and_special_floats():
    same = records._same
    assert same(1, 1) and not same(1, 1.0) and not same(1, True) and not same(0, False)
    assert same(float("nan"), float("nan"))
    assert not same(0.0, -0.0) and same(-0.0, -0.0)
    assert same([1.0, float("nan")], [1.0, float("nan")])
    assert not same([0.0, 1.0], [-0.0, 1.0])
    assert not same([1, True], [True, 1]) and not same([1, 2], [1.0, 2.0])
    assert not same([1, 2], [1, 2, 3]) and not same([1, 2], (1, 2))
    vec = _vector()
    changed = list(vec)
    changed[500] += 0.5
    assert same(vec, list(vec)) and not same(vec, changed)
    assert same({"a": [1, {"b": 2}], "c": 1}, {"c": 1, "a": [1, {"b": 2}]})
    assert not same({"a": 1}, {"a": 1, "b": 2}) and not same({"a": 1}, {"b": 1})
    assert same({1, 2}, {2, 1}) and not same({1}, {True})
    assert same(Decimal("1.50"), Decimal("1.50"))
    assert not same(Decimal("1.5"), Decimal("1.50"))
    assert same(date(2020, 1, 2), date(2020, 1, 2))
    assert not same(date(2020, 1, 2), datetime(2020, 1, 2))
    assert not same(None, 0) and same(None, None) and same("a", "a")
    assert not same(math.inf, -math.inf) and same(math.inf, math.inf)


def test_snap_copies_and_normalises():
    snap = records._snap
    nested = {"l": [1, [2, 3]], "m": {"k": [4]}, "s": {1, 2}, "t": (1, 2)}
    copied = snap(nested)
    assert copied == {"l": [1, [2, 3]], "m": {"k": [4]}, "s": {1, 2}, "t": [1, 2]}
    assert copied["l"] is not nested["l"] and copied["l"][1] is not nested["l"][1]
    assert copied["m"]["k"] is not nested["m"]["k"] and copied["s"] is not nested["s"]
    nested["l"][1].append(9)
    nested["m"]["k"].append(9)
    assert copied["l"] == [1, [2, 3]] and copied["m"] == {"k": [4]}
    # arrays and NumPy scalars are the lists and scalars a read gives
    array = np.asarray([0.5, 0.25], dtype=np.float32)
    assert snap(array) == [0.5, 0.25] and type(snap(array)[0]) is float
    assert snap(np.float32(0.5)) == 0.5 and type(snap(np.float32(0.5))) is float
    assert type(snap(np.int64(3))) is int and type(snap(np.bool_(True))) is bool
    assert snap(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    held = snap(array)
    array[1] = 8  # the copy is not the array
    assert held == [0.5, 0.25]
    values = [datetime(2020, 1, 2), date(2020, 1, 2), Decimal("1"), b"x", None, "s"]
    assert snap(values) == values and snap(values) is not values
