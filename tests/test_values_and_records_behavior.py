"""What Python values and record objects do, as the how-to page ``docs/modules/ROOT/pages/how-to/values-and-records.adoc`` says it
(ArcadeData/arcadedb-python#267, rows 3, 9 and 12).

Each statement on that page is pinned here first: how Python values go in and what comes back (scalars, ``Decimal``, ``datetime`` and
``date``, bytes, lists, maps, sets, embedded documents), what a query row and a record object give you (``get``, ``to_dict``, item and
attribute access, identity), which of them is a copy of the record as it was read, how ``modify()``, ``save()`` and a stale save
behave, and what the ``Schema`` API does next to plain SQL. The tests use SQL and the wrapper objects only, no Java object, so they
run on the JVM build and the native build alike.
"""

import re
import struct
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from arcadedb import ArcadeDBError, IndexType, PropertyType
from arcadedb.graph import Document, Edge, Vertex


def _first(db, sql, *args):
    return db.query("sql", sql, *args).first()


# --- Python values in, Python values out ------------------------------------------------------------


def test_scalars_come_back_as_the_python_type_they_went_in_as(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Scalars")
    with temp_db.transaction():
        temp_db.command(
            "sql",
            "INSERT INTO Scalars SET s = ?, i = ?, f = ?, b = ?, n = ?",
            "text",
            2**40,
            0.1,
            True,
            None,
        )
    row = _first(temp_db, "SELECT s, i, f, b, n FROM Scalars")
    assert row.get("s") == "text" and type(row.get("s")) is str
    assert row.get("i") == 2**40 and type(row.get("i")) is int
    assert row.get("f") == 0.1 and type(row.get("f")) is float
    assert row.get("b") is True
    assert row.get("n") is None


def test_the_declared_type_of_a_property_wins_over_the_python_type(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Declared")
    temp_db.command("sql", "CREATE PROPERTY Declared.ratio DOUBLE")
    temp_db.command("sql", "CREATE PROPERTY Declared.count INTEGER")
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO Declared SET ratio = ?, count = ?", 3, 7)
    row = _first(temp_db, "SELECT ratio, count FROM Declared")
    assert row.get("ratio") == 3.0 and type(row.get("ratio")) is float
    assert row.get("count") == 7 and type(row.get("count")) is int


@pytest.mark.parametrize("declared", [False, True], ids=["undeclared", "declared"])
def test_a_decimal_keeps_every_digit(temp_db, declared):
    value = Decimal("12345678901234567890.123456789012345678")
    temp_db.command("sql", "CREATE DOCUMENT TYPE Money")
    if declared:
        temp_db.command("sql", "CREATE PROPERTY Money.amount DECIMAL")
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO Money SET amount = ?", value)
    got = _first(temp_db, "SELECT amount FROM Money").get("amount")
    assert type(got) is Decimal and got == value
    # a lookup by the same Decimal finds the record
    assert (
        _first(temp_db, "SELECT count(*) AS n FROM Money WHERE amount = ?", value).get(
            "n"
        )
        == 1
    )


NAIVE = datetime(2026, 10, 1, 12, 34, 56, 789123)
AWARE = datetime(2026, 10, 1, 21, 34, 56, 789123, tzinfo=timezone(timedelta(hours=9)))
EPOCH_MS = 1790858096789  # 2026-10-01T12:34:56.789Z


def test_a_naive_datetime_is_a_utc_wall_clock_and_an_aware_one_is_converted(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Event")
    temp_db.command("sql", "CREATE PROPERTY Event.at DATETIME_MICROS")
    with temp_db.transaction():
        for key, value in (("naive", NAIVE), ("aware", AWARE)):
            temp_db.command("sql", "INSERT INTO Event SET k = ?, at = ?", key, value)
    for key in ("naive", "aware"):
        row = _first(
            temp_db, "SELECT at, at.asLong() AS ms FROM Event WHERE k = ?", key
        )
        # both are 12:34:56.789123 UTC, and what comes back has no time zone
        assert row.get("at") == NAIVE and row.get("at").tzinfo is None, key
        assert row.get("ms") == EPOCH_MS, key
    # a lookup by either value finds both records
    for value in (NAIVE, AWARE):
        assert (
            _first(temp_db, "SELECT count(*) AS n FROM Event WHERE at = ?", value).get(
                "n"
            )
            == 2
        )


def test_the_precision_of_the_declared_type_decides_what_a_datetime_keeps(temp_db):
    value = datetime(2021, 2, 3, 4, 5, 6, 123456)
    temp_db.command("sql", "CREATE DOCUMENT TYPE Stamp")
    for name, declared in (
        ("sec", "DATETIME_SECOND"),
        ("ms", "DATETIME"),
        ("us", "DATETIME_MICROS"),
    ):
        temp_db.command("sql", f"CREATE PROPERTY Stamp.{name} {declared}")
    with temp_db.transaction():
        temp_db.command(
            "sql", "INSERT INTO Stamp SET sec = :v, ms = :v, us = :v", {"v": value}
        )
    row = _first(temp_db, "SELECT sec, ms, us FROM Stamp")
    assert row.get("sec") == datetime(2021, 2, 3, 4, 5, 6)
    assert row.get("ms") == datetime(2021, 2, 3, 4, 5, 6, 123000)
    assert row.get("us") == value


def test_a_date_is_a_date_and_an_undeclared_value_keeps_its_kind(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Day")
    temp_db.command("sql", "CREATE PROPERTY Day.declared DATE")
    with temp_db.transaction():
        temp_db.command(
            "sql",
            "INSERT INTO Day SET declared = :d, loose = :d, moment = :t",
            {"d": date(2026, 10, 1), "t": NAIVE},
        )
    row = _first(temp_db, "SELECT declared, loose, moment FROM Day")
    assert type(row.get("declared")) is date and row.get("declared") == date(
        2026, 10, 1
    )
    assert type(row.get("loose")) is date and row.get("loose") == date(2026, 10, 1)
    # a datetime is not a date: it stays a datetime
    assert type(row.get("moment")) is datetime


def test_bytes_come_back_as_a_list_of_signed_ints(temp_db):
    payload = b"\xff\x00\xfe\x80Hello"
    temp_db.command("sql", "CREATE DOCUMENT TYPE Blob")
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO Blob SET data = ?", payload)
    got = _first(temp_db, "SELECT data FROM Blob").get("data")
    assert got == list(struct.unpack(f"{len(payload)}b", payload))
    assert bytes(b & 0xFF for b in got) == payload


def test_lists_and_maps_are_lists_and_dicts_to_any_depth(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Shapes")
    nested = {"tags": ["a", "b"], "inner": {"n": 1, "ok": True}, "none": None}
    with temp_db.transaction():
        temp_db.command(
            "sql",
            "INSERT INTO Shapes SET mixed = ?, tup = ?, nested = ?",
            [1, "a", None, 2.5, True],
            (1, 2, 3),
            nested,
        )
    row = _first(temp_db, "SELECT mixed, tup, nested FROM Shapes")
    assert (
        row.get("mixed") == [1, "a", None, 2.5, True] and type(row.get("mixed")) is list
    )
    # a tuple goes in as a list and comes back as one
    assert row.get("tup") == [1, 2, 3] and type(row.get("tup")) is list
    assert row.get("nested") == nested and type(row.get("nested")) is dict


def test_a_set_is_a_set_until_the_commit_and_a_list_after_it(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Bag")
    with temp_db.transaction():
        bag = (
            temp_db.new_document("Bag").set("roles", {"admin", "user", "admin"}).save()
        )
        assert bag.get("roles") == {"admin", "user"}
    for got in (
        temp_db.lookup_by_rid(bag.get_rid()).get("roles"),
        _first(temp_db, "SELECT roles FROM Bag").get("roles"),
    ):
        assert type(got) is list and sorted(got) == ["admin", "user"]


def test_an_embedded_document_reads_back_as_a_dict_through_to_dict(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Emb")
    temp_db.command("sql", "CREATE DOCUMENT TYPE Holder")
    temp_db.command("sql", "CREATE PROPERTY Holder.em EMBEDDED OF Emb")
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO Holder SET em = {'@type': 'Emb', 'q': 1}")
    row = _first(temp_db, "SELECT FROM Holder")
    for value in (
        row.to_dict()["em"],
        temp_db.query("sql", "SELECT FROM Holder").to_list()[0]["em"],
    ):
        assert isinstance(value, dict) and value["q"] == 1
    assert _first(temp_db, "SELECT em.q AS q FROM Holder").get("q") == 1


def test_get_to_dict_and_to_list_give_the_same_values(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Same")
    temp_db.command("sql", "CREATE PROPERTY Same.d DATE")
    temp_db.command("sql", "CREATE PROPERTY Same.t DATETIME_MICROS")
    with temp_db.transaction():
        temp_db.command(
            "sql",
            "INSERT INTO Same SET s = 'x', i = 1, f = 2.5, n = :n, d = :d, t = :t, li = [1, 2], ma = {'k': 'v'}",
            {"n": Decimal("1.50"), "d": date(2026, 10, 1), "t": NAIVE},
        )
    row = _first(temp_db, "SELECT s, i, f, n, d, t, li, ma FROM Same")
    as_dict = row.to_dict()
    as_listed = temp_db.query(
        "sql", "SELECT s, i, f, n, d, t, li, ma FROM Same"
    ).to_list()[0]
    assert set(as_dict) == set(as_listed) == {"s", "i", "f", "n", "d", "t", "li", "ma"}
    for name in as_dict:
        values = (row.get(name), as_dict[name], as_listed[name])
        assert len({type(v) for v in values}) == 1, name
        assert values[0] == values[1] == values[2], name


# --- Query rows -------------------------------------------------------------------------------------


@pytest.fixture
def people(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE Person")
    temp_db.command("sql", "CREATE EDGE TYPE Knows")
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO Person SET name = 'Ada', age = 36")
    return temp_db


def test_a_row_is_read_with_get_not_with_brackets_or_attributes(people):
    row = _first(people, "SELECT FROM Person")
    assert row.get("name") == "Ada"
    with pytest.raises(TypeError):
        row["name"]
    with pytest.raises(AttributeError):
        row.name
    # to_list() and to_dict() are plain dicts, which take brackets
    listed = people.query("sql", "SELECT FROM Person").to_list()
    assert type(listed[0]) is dict and listed[0]["name"] == "Ada"
    assert type(row.to_dict()) is dict and row.to_dict()["name"] == "Ada"


def test_a_missing_property_is_none_and_has_property_tells_it_from_a_value(people):
    row = _first(people, "SELECT FROM Person")
    assert row.get("nothing") is None
    assert row.has_property("nothing") is False
    assert row.has_property("name") is True
    assert sorted(row.get_property_names()) == ["age", "name"]
    assert sorted(row.property_names) == ["age", "name"]
    assert row.to_dict() == {"name": "Ada", "age": 36}


def test_to_dict_is_a_new_dict_each_time(people):
    row = _first(people, "SELECT FROM Person")
    changed = row.to_dict()
    changed["name"] = "someone else"
    assert row.get("name") == "Ada"
    assert row.to_dict()["name"] == "Ada"


def test_a_record_row_has_a_rid_and_an_element_and_a_projection_row_has_neither(people):
    record_row = _first(people, "SELECT FROM Person")
    assert re.fullmatch(r"#\d+:\d+", record_row.get_rid())
    assert isinstance(record_row.get_element(), Vertex)
    assert isinstance(record_row.get_vertex(), Vertex)
    assert record_row.get_element().get_rid() == record_row.get_rid()

    projection = _first(people, "SELECT name FROM Person")
    assert projection.get_rid() is None
    assert projection.get_element() is None
    assert projection.get("name") == "Ada"


def test_the_rows_of_a_document_type_wrap_as_documents_and_of_an_edge_type_as_edges(
    people,
):
    people.command("sql", "CREATE DOCUMENT TYPE Note")
    with people.transaction():
        people.command("sql", "INSERT INTO Note SET text = 'hi'")
        a = people.new_vertex("Person").set("name", "Bob").save()
        b = people.new_vertex("Person").set("name", "Cy").save()
        a.new_edge("Knows", b, since=2020)
    note = _first(people, "SELECT FROM Note").get_element()
    assert isinstance(note, Document)
    assert not isinstance(note, (Vertex, Edge))
    edge = _first(people, "SELECT FROM Knows").get_element()
    assert isinstance(edge, Edge) and edge.get("since") == 2020
    # a Vertex and an Edge are Documents too
    assert isinstance(edge, Document) and isinstance(a, Document)


# --- Record objects ---------------------------------------------------------------------------------


def test_new_document_set_and_save_chain_and_return_the_same_object(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    with temp_db.transaction():
        task = temp_db.new_document("Task")
        assert isinstance(task, Document)
        assert task.set("title", "write docs") is task
        assert task.set("priority", 2).save() is task
        assert re.fullmatch(r"#\d+:\d+", task.get_rid())
        assert task.get_type_name() == "Task"
        # the record is readable by a statement in the same transaction
        assert _first(temp_db, "SELECT count(*) AS n FROM Task").get("n") == 1
    assert temp_db.lookup_by_rid(task.get_rid()).to_dict() == {
        "title": "write docs",
        "priority": 2,
    }


def test_save_is_a_write_and_needs_a_transaction(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    task = temp_db.new_document("Task").set("title", "x")
    with pytest.raises(ArcadeDBError, match="Transaction not begun"):
        task.save()
    assert _first(temp_db, "SELECT count(*) AS n FROM Task").get("n") == 0


def test_a_record_that_was_read_is_immutable_until_modify(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    with temp_db.transaction():
        rid = temp_db.new_document("Task").set("n", 1).save().get_rid()
    record = temp_db.lookup_by_rid(rid)
    with pytest.raises(AttributeError, match="immutable"):
        record.set("n", 2)
    mutable = record.modify()
    assert mutable is not record
    assert mutable.set("n", 2) is mutable


def test_modify_and_save_change_the_stored_record_and_not_the_object_that_was_read(
    temp_db,
):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    with temp_db.transaction():
        rid = temp_db.new_document("Task").set("n", 1).save().get_rid()
    record = temp_db.lookup_by_rid(rid)
    assert record.get("n") == 1

    with temp_db.transaction():
        mutable = record.modify().set("n", 2)
        mutable.save()
        # a statement in the same transaction sees the change; the object that was read does not
        assert _first(temp_db, "SELECT n FROM Task").get("n") == 2
        assert record.get("n") == 1
    assert record.get("n") == 1
    assert temp_db.lookup_by_rid(rid).get("n") == 2


def test_set_without_save_stores_nothing(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    with temp_db.transaction():
        rid = temp_db.new_document("Task").set("n", 1).save().get_rid()
    with temp_db.transaction():
        temp_db.lookup_by_rid(rid).modify().set("n", 99)
    assert _first(temp_db, "SELECT n FROM Task").get("n") == 1


def test_a_record_and_a_row_are_the_record_as_it_was_read_a_statement_does_not_update_them(
    temp_db,
):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    with temp_db.transaction():
        rid = temp_db.new_document("Task").set("n", 1).save().get_rid()
    record = temp_db.lookup_by_rid(rid)
    row = _first(temp_db, "SELECT FROM Task")
    assert record.get("n") == 1 and row.get("n") == 1

    with temp_db.transaction():
        temp_db.command("sql", "UPDATE Task SET n = 2")
        assert _first(temp_db, "SELECT n FROM Task").get("n") == 2
        assert record.get("n") == 1 and row.get("n") == 1
    assert record.to_dict() == {"n": 1} and row.to_dict() == {"n": 1}
    assert temp_db.lookup_by_rid(rid).get("n") == 2


def test_saving_a_record_that_changed_after_it_was_read_is_refused(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    with temp_db.transaction():
        rid = temp_db.new_document("Task").set("n", 1).set("m", 1).save().get_rid()
    mutable = temp_db.lookup_by_rid(rid).modify()
    with temp_db.transaction():
        temp_db.command("sql", "UPDATE Task SET m = 2")
    mutable.set("n", 10)
    with pytest.raises(ArcadeDBError) as refused:
        with temp_db.transaction():
            mutable.save()
    assert "oncurrent" in str(refused.value) or rid in str(refused.value)
    # nothing of the refused save is stored, and the other change is kept
    assert _first(temp_db, "SELECT n, m FROM Task").to_dict() == {"n": 1, "m": 2}

    # a fresh read sees the new state and saves
    fresh = temp_db.lookup_by_rid(rid).modify().set("n", 10)
    with temp_db.transaction():
        fresh.save()
    assert _first(temp_db, "SELECT n, m FROM Task").to_dict() == {"n": 10, "m": 2}


def test_two_reads_of_one_record_are_two_objects_with_one_rid(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    with temp_db.transaction():
        rid = temp_db.new_document("Task").set("n", 1).save().get_rid()
    first = temp_db.lookup_by_rid(rid)
    second = temp_db.lookup_by_rid(rid)
    third = _first(temp_db, "SELECT FROM Task").get_element()
    assert first is not second
    assert first != second  # no __eq__: objects are equal only to themselves
    assert first.get_rid() == second.get_rid() == third.get_rid() == rid


def test_lookup_by_rid_raises_for_a_record_that_is_not_there_and_delete_removes_one(
    temp_db,
):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Task")
    with temp_db.transaction():
        rid = temp_db.new_document("Task").set("n", 1).save().get_rid()
    with temp_db.transaction():
        temp_db.lookup_by_rid(rid).delete()
    assert _first(temp_db, "SELECT count(*) AS n FROM Task").get("n") == 0
    with pytest.raises(ArcadeDBError):
        temp_db.lookup_by_rid(rid)


def test_lookup_by_key_needs_an_index_and_returns_a_record_or_none(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE User")
    temp_db.command("sql", "CREATE PROPERTY User.email STRING")
    with temp_db.transaction():
        temp_db.command(
            "sql", "INSERT INTO User SET email = 'ada@example.com', name = 'Ada'"
        )
    # no index on email yet: lookup_by_key never scans
    with pytest.raises(ArcadeDBError):
        temp_db.lookup_by_key("User", ["email"], ["ada@example.com"])
    temp_db.command("sql", "CREATE INDEX ON User (email) UNIQUE")
    found = temp_db.lookup_by_key("User", ["email"], ["ada@example.com"])
    assert isinstance(found, Vertex) and found.get("name") == "Ada"
    assert temp_db.lookup_by_key("User", ["email"], ["nobody@example.com"]) is None


def test_vertices_and_edges_are_created_saved_first_and_walked_by_method(people):
    with people.transaction():
        ada = people.new_vertex("Person").set("name", "Ada2").save()
        bob = people.new_vertex("Person").set("name", "Bob").save()
        edge = ada.new_edge("Knows", bob, since=2020)
        assert isinstance(edge, Edge)
        assert edge.get("since") == 2020
        assert edge.get_out().get_rid() == ada.get_rid()
        assert edge.get_in().get_rid() == bob.get_rid()
        assert [e.get_rid() for e in ada.get_out_edges("Knows")] == [edge.get_rid()]
        assert [e.get_rid() for e in bob.get_in_edges("Knows")] == [edge.get_rid()]
        assert ada.get_in_edges() == []
    stored = _first(people, "SELECT FROM Knows").get_edge()
    assert stored.get("since") == 2020


# --- The Schema API next to SQL ---------------------------------------------------------------------


def test_the_schema_api_and_sql_make_the_same_schema_and_need_no_transaction(temp_db):
    schema = temp_db.schema
    schema.create_vertex_type("Person")
    schema.create_property("Person", "email", "STRING")
    schema.create_property("Person", "age", PropertyType.INTEGER)
    schema.create_index("Person", ["email"], unique=True, index_type=IndexType.HASH)
    temp_db.command("sql", "CREATE DOCUMENT TYPE Note")
    temp_db.command("sql", "CREATE PROPERTY Note.text STRING")
    temp_db.command("sql", "CREATE INDEX ON Note (text) UNIQUE")
    assert temp_db.is_transaction_active() is False

    for name in ("Person", "Note"):
        assert schema.exists_type(name) is True
    assert schema.exists_type("Nothing") is False
    assert schema.exists_index("Person[email]") is True
    assert schema.exists_index("Note[text]") is True

    # a type made by the API takes SQL, and the index it made is enforced
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO Person SET email = 'a@x.org', age = 3")
    with pytest.raises(ArcadeDBError, match="DuplicatedKeyException"):
        with temp_db.transaction():
            temp_db.command("sql", "INSERT INTO Person SET email = 'a@x.org', age = 4")
    row = _first(temp_db, "SELECT age FROM Person")
    assert row.get("age") == 3 and type(row.get("age")) is int


def test_creating_what_exists_fails_unless_it_is_get_or_create_or_if_not_exists(
    temp_db,
):
    schema = temp_db.schema
    schema.create_vertex_type("Person")
    with pytest.raises(ArcadeDBError):
        schema.create_vertex_type("Person")
    with pytest.raises(ArcadeDBError):
        temp_db.command("sql", "CREATE VERTEX TYPE Person")
    schema.get_or_create_vertex_type("Person")
    temp_db.command("sql", "CREATE VERTEX TYPE Person IF NOT EXISTS")
    schema.create_property("Person", "name", "STRING")
    schema.get_or_create_property("Person", "name", "STRING")
    temp_db.command("sql", "CREATE PROPERTY Person.name IF NOT EXISTS STRING")
    assert schema.exists_type("Person") is True


def test_constraints_and_other_property_settings_are_sql(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Account")
    temp_db.command(
        "sql", "CREATE PROPERTY Account.name STRING (mandatory true, notnull true)"
    )
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO Account SET name = 'ok'")
    with pytest.raises(ArcadeDBError):
        with temp_db.transaction():
            temp_db.command("sql", "INSERT INTO Account SET other = 1")
    assert _first(temp_db, "SELECT count(*) AS n FROM Account").get("n") == 1


def test_drop_type_and_drop_index_by_the_api_and_by_sql(temp_db):
    schema = temp_db.schema
    schema.create_document_type("A")
    schema.create_property("A", "k", "STRING")
    schema.create_index("A", ["k"], unique=False)
    assert schema.exists_index("A[k]") is True
    schema.drop_index("A[k]")
    assert schema.exists_index("A[k]") is False
    schema.drop_type("A")
    assert schema.exists_type("A") is False

    temp_db.command("sql", "CREATE DOCUMENT TYPE B")
    temp_db.command("sql", "DROP TYPE B")
    assert schema.exists_type("B") is False
