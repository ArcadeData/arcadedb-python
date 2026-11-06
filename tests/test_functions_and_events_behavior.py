"""What Python functions in statements and record events do, as the how-to page
``docs/modules/ROOT/pages/how-to/functions-and-events.adoc`` says it (ArcadeData/arcadedb-python#267, row 8).

``test_python_callbacks.py`` already pins the basics: a function in SELECT and WHERE, the names and argument counts, replace and
unregister, the types a function returns, a failing function as a query error, a record as a wrapper, each event with its record,
the type filter, the veto, ``remove()``, a failing callback and the registration errors. ``test_native_udf.py`` and
``test_python_function_interrupt.py`` pin the native upcall and a real Ctrl-C. This file pins what the page adds: what a function
receives from a stored value, what an exception does to a transaction, the scope of a function (the process) and of an event (one
database), the threads, and what a callback may do (read, write, re-enter, change the record). The tests use SQL, openCypher and the
wrapper objects only, no Java object, so they run on the JVM build and the native build alike.
"""

import threading
from datetime import date, datetime
from decimal import Decimal

import pytest

import arcadedb
from arcadedb import ArcadeDBError


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Person")
    temp_db.command("sql", "CREATE DOCUMENT TYPE Audit")
    temp_db.command("sql", "CREATE VERTEX TYPE City")
    temp_db.command("sql", "CREATE EDGE TYPE Road")
    return temp_db


def _insert(db, *names, type_name="Person"):
    with db.transaction():
        statement = (
            f"INSERT INTO {type_name} SET name = ?"  # nosec B608 - literal type names
        )
        for name in names:
            db.command("sql", statement, name)


def _names(db, type_name="Person"):
    statement = f"SELECT name FROM {type_name}"  # nosec B608 - literal type names
    rows = db.query("sql", statement).to_list()
    return sorted(row["name"] for row in rows)


def _count_in_another_thread(db, type_name):
    seen = []
    statement = f"SELECT FROM {type_name}"  # nosec B608 - literal type names

    def read():
        seen.append(len(db.query("sql", statement).to_list()))

    thread = threading.Thread(target=read)
    thread.start()
    thread.join()
    return seen[0]


# --- what a function receives and returns ----------------------------------------------------------


def test_the_literals_of_a_statement_arrive_as_the_python_types_they_look_like(db):
    db.register_function("kinds", lambda *args: [type(a).__name__ for a in args])
    row = db.query(
        "sql", "SELECT kinds(1, 2.5, 'a', true, null, [1, 2], {'k': 'v'}) AS k"
    ).first()
    assert row.get("k") == [
        "int",
        "float",
        "str",
        "bool",
        "NoneType",
        "list",
        "dict",
    ]


def test_a_stored_datetime_date_and_decimal_arrive_as_python_values(db):
    db.command("sql", "CREATE DOCUMENT TYPE Booking")
    db.command("sql", "CREATE PROPERTY Booking.at DATETIME_MICROS")
    db.command("sql", "CREATE PROPERTY Booking.day DATE")
    db.command("sql", "CREATE PROPERTY Booking.price DECIMAL")
    with db.transaction():
        db.command(
            "sql",
            "INSERT INTO Booking SET at = ?, day = ?, price = ?, tags = ?, meta = ?",
            datetime(2026, 10, 1, 12, 34, 56, 789123),
            date(2026, 10, 1),
            Decimal("19.99"),
            ["a", "b"],
            {"k": 1},
        )
    seen = []
    db.register_function("seen", lambda *args: seen.append(args) or 0)
    db.query(
        "sql", "SELECT seen(at, day, price, tags, meta) AS s FROM Booking"
    ).to_list()
    assert len(seen) == 1
    at, day, price, tags, meta = seen[0]
    # what the page says about values: a datetime has no time zone, a date is a date, a Decimal keeps its digits
    assert type(at) is datetime and at == datetime(2026, 10, 1, 12, 34, 56, 789123)
    assert type(day) is date and day == date(2026, 10, 1)
    assert type(price) is Decimal and price == Decimal("19.99")
    assert tags == ["a", "b"] and meta == {"k": 1}


def test_a_vertex_an_edge_and_a_document_arrive_as_their_own_wrapper(db):
    with db.transaction():
        db.command("sql", "INSERT INTO Person SET name = 'Ann'")
        db.command("sql", "CREATE VERTEX City SET name = 'A'")
        db.command("sql", "CREATE VERTEX City SET name = 'B'")
        db.command(
            "sql",
            "CREATE EDGE Road FROM (SELECT FROM City WHERE name = 'A') "
            "TO (SELECT FROM City WHERE name = 'B') SET km = 5",
        )
    db.register_function(
        "kind", lambda record: [type(record).__name__, record.get_type_name()]
    )
    for type_name, expected in (
        ("Person", ["Document", "Person"]),
        ("City", ["Vertex", "City"]),
        ("Road", ["Edge", "Road"]),
    ):
        statement = f"SELECT kind(@this) AS k FROM {type_name} LIMIT 1"  # nosec B608 - literal type names
        assert db.query("sql", statement).first().get("k") == expected


def test_a_tuple_result_is_a_list(db):
    db.register_function("pair", lambda: (1, "a"))
    assert db.query("sql", "SELECT pair() AS p").first().get("p") == [1, "a"]


def test_opencypher_passes_and_returns_the_same_python_values(db):
    with db.transaction():
        db.command("opencypher", "CREATE (:City {name: 'Oslo'})")
    seen = []

    def describe(*args):
        seen.append(args)
        return [len(args), "ok"]

    db.register_function("py.describe", describe)
    rows = db.query(
        "opencypher",
        "MATCH (c:City) RETURN py.describe(1, 2.5, 'a', true, null, [1, 2]) AS d",
    ).to_list()
    assert seen == [(1, 2.5, "a", True, None, [1, 2])]
    assert [type(a) for a in seen[0]] == [int, float, str, bool, type(None), list]
    assert rows == [{"d": [6, "ok"]}]


def test_the_function_runs_once_per_row(db):
    _insert(db, "a", "b", "c")
    calls = []
    db.register_function("note", lambda name: calls.append(name) or name)
    db.query("sql", "SELECT note(name) AS n FROM Person").to_list()
    assert sorted(calls) == ["a", "b", "c"]


def test_a_function_can_query_the_database_it_runs_in(db):
    _insert(db, "a", "b")

    def people():
        return len(db.query("sql", "SELECT FROM Person").to_list())

    db.register_function("people", people)
    assert db.query("sql", "SELECT people() AS n").first().get("n") == 2


# --- what an exception does ------------------------------------------------------------------------


def test_a_function_that_raises_undoes_its_statement_and_the_transaction_goes_on(db):
    _insert(db, "Ann", "Bob", "Cy")

    def refuse_bob(name):
        if name == "Bob":
            raise ValueError("no Bob")
        return name.upper()

    db.register_function("refuse_bob", refuse_bob)
    with db.transaction():
        db.command("sql", "INSERT INTO Audit SET note = 'before'")
        with pytest.raises(ArcadeDBError, match="ValueError: no Bob"):
            db.command("sql", "UPDATE Person SET shout = refuse_bob(name)")
        assert db.is_transaction_active() is True
        db.command("sql", "INSERT INTO Audit SET note = 'after'")
    # the rows that the failed UPDATE had reached before Bob are not changed, and the other statements are committed
    shouted = db.query("sql", "SELECT FROM Person WHERE shout IS NOT NULL").to_list()
    assert shouted == []
    notes = db.query("sql", "SELECT note FROM Audit").to_list()
    assert sorted(row["note"] for row in notes) == ["after", "before"]


@pytest.mark.parametrize("event", ["before_create", "after_create"])
def test_a_callback_that_raises_undoes_its_statement_and_the_transaction_goes_on(
    db, event
):
    def refuse_bad(record):
        if record.get("name") == "bad":
            raise RuntimeError("no bad")

    db.on(event, refuse_bad, "Person")
    with db.transaction():
        db.command("sql", "INSERT INTO Person SET name = 'first'")
        with pytest.raises(ArcadeDBError, match=f"{event}.*RuntimeError: no bad"):
            db.command("sql", "INSERT INTO Person SET name = 'bad'")
        assert db.is_transaction_active() is True
        db.command("sql", "INSERT INTO Person SET name = 'second'")
    assert _names(db) == ["first", "second"]


def test_keyboard_interrupt_in_a_callback_reaches_the_caller_as_one(db):
    def interrupt(record):
        raise KeyboardInterrupt("stop")

    db.on("before_create", interrupt, "Person")
    with pytest.raises(KeyboardInterrupt) as error:
        with db.transaction():
            db.command("sql", "INSERT INTO Person SET name = 'x'")
    assert isinstance(error.value.__cause__, ArcadeDBError)
    assert "before_create" in str(error.value.__cause__)
    assert db.is_transaction_active() is False
    assert _names(db) == []


# --- scope: a function belongs to the process, an event to one database ----------------------------


def test_a_function_is_seen_by_every_open_database_and_goes_with_the_one_that_registered_it(
    db, tmp_path
):
    other = arcadedb.create_database(str(tmp_path / "other"))
    try:
        db.register_function("shared_fn", lambda: "hi")
        assert other.query("sql", "SELECT shared_fn() AS x").first().get("x") == "hi"
        # the name is taken for the process, and only the database that registered it can remove it
        with pytest.raises(ArcadeDBError, match="already registered"):
            other.register_function("shared_fn", lambda: "no")
        with pytest.raises(ArcadeDBError, match="No Python function"):
            other.unregister_function("shared_fn")
        db.close()
        with pytest.raises(ArcadeDBError, match="shared_fn"):
            other.query("sql", "SELECT shared_fn() AS x").to_list()
    finally:
        other.close()


def test_an_event_callback_belongs_to_the_database_that_registered_it(db, tmp_path):
    other = arcadedb.create_database(str(tmp_path / "other"))
    try:
        other.command("sql", "CREATE DOCUMENT TYPE Person")
        seen = []
        db.on("after_create", lambda record: seen.append(record.get("name")))
        _insert(other, "elsewhere")
        _insert(db, "here")
        assert seen == ["here"]
    finally:
        other.close()


def test_closing_a_database_drops_its_callbacks(temp_db_path):
    seen = []
    with arcadedb.create_database(temp_db_path) as first:
        first.command("sql", "CREATE DOCUMENT TYPE Person")
        first.on("after_create", lambda record: seen.append("type"), "Person")
        first.on("after_create", lambda record: seen.append("database"))
    with arcadedb.open_database(temp_db_path) as second:
        _insert(second, "after reopen")
        assert _names(second) == ["after reopen"]
    assert seen == []


@pytest.mark.parametrize("answer", [None, True, 0, "", []])
def test_only_false_vetoes_a_before_callback(db, answer):
    db.on("before_create", lambda record, answer=answer: answer, "Person")
    _insert(db, "kept")
    assert _names(db) == ["kept"]


def test_the_type_filter_matches_the_exact_type_and_not_its_subtypes(db):
    db.command("sql", "CREATE DOCUMENT TYPE Animal")
    db.command("sql", "CREATE DOCUMENT TYPE Dog EXTENDS Animal")
    on_animal, on_everything = [], []
    db.on("after_create", lambda r: on_animal.append(r.get_type_name()), "Animal")
    db.on("after_create", lambda r: on_everything.append(r.get_type_name()))
    with db.transaction():
        db.command("sql", "INSERT INTO Dog SET name = 'rex'")
        db.command("sql", "INSERT INTO Animal SET name = 'generic'")
    assert on_animal == ["Animal"]
    assert sorted(on_everything) == ["Animal", "Dog"]


def test_an_edge_fires_the_event_of_its_own_type(db):
    seen = []
    db.on(
        "after_create",
        lambda r: seen.append((type(r).__name__, r.get_type_name())),
        "Road",
    )
    with db.transaction():
        db.command("sql", "CREATE VERTEX City SET name = 'A'")
        db.command("sql", "CREATE VERTEX City SET name = 'B'")
        db.command(
            "sql",
            "CREATE EDGE Road FROM (SELECT FROM City WHERE name = 'A') "
            "TO (SELECT FROM City WHERE name = 'B')",
        )
    assert seen == [("Edge", "Road")]


# --- when an event fires ---------------------------------------------------------------------------


def test_after_events_fire_before_the_commit_and_a_rollback_does_not_unfire_them(db):
    log = []
    for event in ("before_create", "after_create"):
        db.on(event, lambda record, event=event: log.append(event), "Person")
    with pytest.raises(ValueError, match="stop"):
        with db.transaction():
            db.command("sql", "INSERT INTO Person SET name = 'x'")
            # both events have fired, and no other thread sees the row yet
            assert log == ["before_create", "after_create"]
            assert _count_in_another_thread(db, "Person") == 0
            raise ValueError("stop")
    assert _names(db) == []
    assert log == ["before_create", "after_create"]


def test_a_before_callback_sees_the_table_without_the_record_and_an_after_callback_with_it(
    db,
):
    counts = {}

    def count(label):
        def callback(record):
            counts[label] = len(db.query("sql", "SELECT FROM Person").to_list())

        return callback

    db.on("before_create", count("before"), "Person")
    db.on("after_create", count("after"), "Person")
    _insert(db, "first")
    assert counts == {"before": 0, "after": 1}


# --- what a callback may do ------------------------------------------------------------------------


def test_a_write_in_a_callback_belongs_to_the_transaction_that_fired_it(db):
    db.on(
        "after_create",
        lambda record: db.command(
            "sql", "INSERT INTO Audit SET note = ?", "created " + record.get("name")
        ),
        "Person",
    )
    _insert(db, "ann")
    notes = db.query("sql", "SELECT note FROM Audit").to_list()
    assert [row["note"] for row in notes] == ["created ann"]

    with pytest.raises(ValueError, match="stop"):
        with db.transaction():
            db.command("sql", "INSERT INTO Person SET name = 'bob'")
            raise ValueError("stop")
    notes = db.query("sql", "SELECT note FROM Audit").to_list()
    assert [row["note"] for row in notes] == ["created ann"]
    assert _names(db) == ["ann"]


def test_a_callback_for_every_record_fires_for_the_writes_that_callbacks_make(db):
    log = []

    def audit(record):
        log.append(record.get_type_name())
        if (
            record.get_type_name() != "Audit"
        ):  # without this guard the callback would never end
            db.command("sql", "INSERT INTO Audit SET note = 'seen'")

    db.on("after_create", audit)
    _insert(db, "ann")
    assert log == ["Person", "Audit"]


def test_changing_the_record_in_before_create_is_stored_on_the_jvm_build_only(db):
    def stamp(record):
        record.set("stamp", "x")

    db.on("before_create", stamp, "Person")
    _insert(db, "ann")
    row = db.query("sql", "SELECT FROM Person").first()
    expected = "x" if arcadedb.__backend__ == "jvm" else None
    assert row.get("stamp") == expected


# --- threads ---------------------------------------------------------------------------------------


def test_a_function_runs_on_the_thread_of_its_statement_and_on_an_engine_thread_for_async_writes(
    db,
):
    _insert(db, "a")
    seen = []
    db.register_function("who", lambda x: seen.append(threading.get_ident()) or x)

    own = []

    def run():
        own.append(threading.get_ident())
        db.query("sql", "SELECT who(name) AS w FROM Person").to_list()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert seen == own
    db.query("sql", "SELECT who(name) AS w FROM Person").to_list()
    assert seen[-1] == threading.get_ident()

    executor = db.async_executor()
    executor.command("sql", "INSERT INTO Audit SET note = who('async')")
    executor.wait_completion()
    assert len(seen) == 3 and seen[-1] != threading.get_ident()


def test_a_function_that_waits_does_not_block_other_threads(db):
    _insert(db, "a")
    entered, release = threading.Event(), threading.Event()

    def park(name):
        entered.set()
        release.wait(30)
        return name

    db.register_function("park", park)
    outcome = []

    def run():
        outcome.append(
            db.query("sql", "SELECT park(name) AS p FROM Person").first().get("p")
        )

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(30)
        # the statement of the other thread is inside the function, and this thread still runs statements
        assert db.query("sql", "SELECT 1 + 1 AS two").first().get("two") == 2
        assert outcome == []
    finally:
        release.set()
        thread.join(30)
    assert outcome == ["a"]


def test_the_event_of_an_async_write_runs_on_an_engine_thread(db):
    idents = []
    db.on("after_create", lambda record: idents.append(threading.get_ident()), "Person")
    executor = db.async_executor()
    document = db.new_document("Person")
    document.set("name", "async")
    executor.create_record(document)
    executor.wait_completion()
    assert len(idents) == 1 and idents[0] != threading.get_ident()
    assert _names(db) == ["async"]


def test_the_type_name_of_a_record_is_a_python_str_on_the_native_build_and_a_java_string_on_the_jvm_build(
    db,
):
    seen = {}

    def look(record):
        seen["type_name"] = record.get_type_name()
        seen["name"] = record.get("name")
        seen["dict_name"] = record.to_dict()["name"]
        seen["rid"] = record.get_rid()

    db.on("after_create", look, "Person")
    _insert(db, "x")
    # a property, a to_dict value and the RID are Python str on both builds
    for key in ("name", "dict_name", "rid"):
        assert type(seen[key]) is str, key
    assert seen["name"] == "x" and seen["dict_name"] == "x"
    # the type name is a Python str on both builds, in a callback and outside one
    value = seen["type_name"]
    assert value == "Person" and type(value) is str
    outside = db.query("sql", "SELECT FROM Person").first().get_element()
    assert outside.get_type_name() == "Person" and type(outside.get_type_name()) is str
