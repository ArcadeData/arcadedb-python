"""The JVM build's async executor defines each JPype proxy class once, not once per callback.

Every callback of an async call (``create_record``, ``command``, ``query``, ``new_edge``, ``scan_type``,
``transaction``, ``on_ok``, ``on_error``) is a Java object that calls back into Python. The executor used to define a new
``@jpype.JImplements`` class for each one, which is a Python type, a Java proxy class and a method table per callback,
kept for the life of the process (the same cost that ``callbacks.py`` removed for Python functions in SQL).
The classes are now defined once per kind, and the Python callables of one call are state of its instance.

These tests count the class definitions, check that no type objects pile up, and check that what a callback sees is
unchanged: each callback gets the arguments of its own call (no crosstalk between callbacks that share a class), and
the delivery order of a one-writer executor is the submission order.
"""

import gc
import threading

import jpype
import pytest

from arcadedb import async_executor

# the proxy kinds of the executor: new_record, ok, new_edge, error, result, document, error_record, transaction_scope
KINDS = 8


@pytest.fixture
def items_db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Item")
    temp_db.command("sql", "CREATE PROPERTY Item.n INTEGER")
    temp_db.command("sql", "CREATE VERTEX TYPE P")
    temp_db.command("sql", "CREATE PROPERTY P.n INTEGER")
    temp_db.command("sql", "CREATE EDGE TYPE K")
    return temp_db


@pytest.fixture
def definitions(monkeypatch):
    """Count the ``jpype.JImplements`` class definitions, starting from an empty class cache."""
    count = {"n": 0}
    real = jpype.JImplements

    def counting(*args, **kwargs):
        count["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(jpype, "JImplements", counting)
    # raising=False: a tree without the cache has no such name, and then every registration defines a class
    monkeypatch.setattr(async_executor, "_PROXY_CLASSES", {}, raising=False)
    return count


def _register_every_kind(executor, db, rounds):
    """Build the Java proxy of every kind ``rounds`` times, without submitting anything."""
    noop = lambda *_a: None  # noqa: E731
    for _ in range(rounds):
        executor._create_new_record_callback(noop)
        executor._create_ok_callback(noop)
        executor._create_new_edge_callback(noop)
        executor._create_error_callback(noop)
        executor._create_result_callback(noop, noop)
        executor._create_document_callback(noop)
        executor._create_error_record_callback(noop)
        executor._create_transaction_scope(noop)


def test_proxy_classes_are_defined_once_per_kind_across_executors(
    items_db, definitions
):
    for _ in range(3):
        executor = items_db.async_executor()
        _register_every_kind(executor, items_db, rounds=50)
        executor.close()

    assert 0 < definitions["n"] <= KINDS, definitions["n"]


def test_a_callback_heavy_load_defines_no_class_after_the_first_round(
    items_db, definitions
):
    executor = items_db.async_executor().set_parallel_level(1).set_commit_every(50)
    seen = []
    # the first round defines the classes
    executor.command(
        "sql", "INSERT INTO Item SET n = ?", callback=seen.append, args=(-1,)
    )
    doc = items_db.new_document("Item")
    doc.set("n", -2)
    executor.create_record(doc, callback=seen.append, error_callback=seen.append)
    executor.query("sql", "SELECT FROM Item", seen.append)
    executor.wait_completion()
    after_first = definitions["n"]

    for i in range(200):
        executor.command(
            "sql", "INSERT INTO Item SET n = ?", callback=seen.append, args=(i,)
        )
        doc = items_db.new_document("Item")
        doc.set("n", i)
        executor.create_record(doc, callback=seen.append, error_callback=seen.append)
        executor.query("sql", "SELECT FROM Item WHERE n = ?", seen.append, args=(i,))
    executor.wait_completion()
    executor.close()

    assert after_first <= KINDS
    assert definitions["n"] == after_first
    assert items_db.count_type("Item") == 2 + 400


def test_registrations_do_not_pile_up_type_objects(items_db):
    executor = items_db.async_executor()
    _register_every_kind(executor, items_db, rounds=2)  # warm: the classes exist

    def types():
        gc.collect()
        return sum(1 for o in gc.get_objects() if isinstance(o, type))

    before = types()
    _register_every_kind(executor, items_db, rounds=100)  # 800 proxies
    grown = types() - before
    executor.close()

    assert grown < 50, f"{grown} new type objects after 800 registrations"


def test_concurrent_first_use_defines_the_class_once(items_db, definitions):
    executor = items_db.async_executor()
    barrier = threading.Barrier(8)
    errors = []

    def first_use():
        try:
            barrier.wait(30)
            executor._create_ok_callback(lambda: None)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=first_use) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    executor.close()

    assert not errors, errors
    assert definitions["n"] == 1, definitions["n"]


def test_each_command_callback_gets_its_own_rows(items_db):
    """Callbacks share a proxy class; none of them may see another call's result or error."""
    got = {tag: [] for tag in "abc"}
    failed = {"x": [], "y": []}
    executor = items_db.async_executor().set_parallel_level(1).set_commit_every(1)

    for i, tag in enumerate("abc"):
        executor.command(
            "sql",
            "INSERT INTO Item SET n = ?",
            callback=lambda r, tag=tag: got[tag].append(r.get("n")),
            error_callback=lambda e: failed["y"].append(e),
            args=(i,),
        )
    executor.command(
        "sql",
        "INSERT INTO NoSuchType SET n = 1",
        callback=lambda r: failed["y"].append(r),
        error_callback=lambda e: failed["x"].append(str(e)),
    )
    executor.wait_completion()
    executor.close()

    assert got == {"a": [0], "b": [1], "c": [2]}
    assert len(failed["x"]) == 1 and "NoSuchType" in failed["x"][0]
    assert failed["y"] == []


def test_query_callbacks_keep_row_order_and_do_not_mix(items_db):
    with items_db.transaction():
        for i in range(30):
            items_db.command("sql", "INSERT INTO Item SET n = ?", i)
    low, high = [], []
    executor = items_db.async_executor().set_parallel_level(1)
    executor.query(
        "sql",
        "SELECT n FROM Item WHERE n < 10 ORDER BY n",
        lambda r: low.append(r.get("n")),
    )
    executor.query(
        "sql",
        "SELECT n FROM Item WHERE n >= 20 ORDER BY n",
        lambda r: high.append(r.get("n")),
    )
    executor.wait_completion()
    executor.close()

    assert low == list(range(10))
    assert high == list(range(20, 30))


def test_create_record_callbacks_run_in_submission_order_on_one_writer(items_db):
    seen, errors = [], []
    executor = items_db.async_executor().set_parallel_level(1).set_commit_every(25)
    for i in range(200):
        doc = items_db.new_document("Item")
        doc.set("n", i)
        executor.create_record(
            doc,
            callback=lambda d: seen.append(d.get("n")),
            error_callback=errors.append,
        )
    executor.wait_completion()
    executor.close()

    assert errors == []
    assert seen == list(range(200))
    assert items_db.count_type("Item") == 200


def test_create_record_callback_delivers_every_record_at_parallel_four(items_db):
    seen = []
    lock = threading.Lock()

    def on_record(d):
        with lock:
            seen.append(d.get("n"))

    executor = items_db.async_executor().set_parallel_level(4).set_commit_every(25)
    for i in range(200):
        doc = items_db.new_document("Item")
        doc.set("n", i)
        executor.create_record(doc, callback=on_record)
    executor.wait_completion()
    executor.close()

    assert sorted(seen) == list(range(200))


def test_new_edge_callback_gets_the_edge_and_the_two_flags(items_db):
    with items_db.transaction():
        a = items_db.new_vertex("P").set("n", 1).save()
        b = items_db.new_vertex("P").set("n", 2).save()
    calls = []
    executor = items_db.async_executor().set_parallel_level(1)
    executor.new_edge(a, "K", b, callback=lambda *args: calls.append(args))
    executor.wait_completion()
    executor.close()

    assert len(calls) == 1
    edge, created_source, created_dest = calls[0]
    assert str(edge.getTypeName()) == "K"
    assert created_source in (True, False) and created_dest in (True, False)


def test_scan_type_and_error_callbacks_still_see_every_record(items_db):
    with items_db.transaction():
        for i in range(10):
            items_db.command("sql", "INSERT INTO Item SET n = ?", i)
    seen, lock = [], threading.Lock()

    def on_record(record):
        with lock:
            seen.append(int(record.get("n")))
        # None means "keep going"

    executor = items_db.async_executor()
    executor.scan_type("Item", on_record, error_callback=lambda rid, exc: True)
    executor.wait_completion()
    executor.close()

    assert sorted(seen) == list(range(10))


def test_transactions_run_their_own_block_and_report_to_their_own_handlers(items_db):
    ran, ok, failed = [], [], []
    executor = items_db.async_executor().set_parallel_level(1)

    def good():
        ran.append("good")
        items_db.command("sql", "INSERT INTO Item SET n = 1")

    def bad():
        ran.append("bad")
        raise ValueError("boom")

    executor.transaction(
        good,
        retries=1,
        ok_callback=lambda: ok.append("good"),
        error_callback=lambda e: failed.append(("good", e)),
    )
    executor.transaction(
        bad,
        retries=1,
        ok_callback=lambda: ok.append("bad"),
        error_callback=lambda e: failed.append(("bad", e)),
    )
    executor.wait_completion(30_000)
    executor.close()

    assert ran == ["good", "bad"]
    assert ok == ["good"]
    assert [tag for tag, _ in failed] == ["bad"]
    assert items_db.count_type("Item") == 1


def test_executor_wide_handlers_still_fire(items_db):
    ok, errors = [], []
    executor = (
        items_db.async_executor()
        .set_parallel_level(1)
        .set_commit_every(1)
        .on_ok(lambda: ok.append(1))
        .on_error(lambda e: errors.append(str(e)))
    )
    for i in range(5):
        doc = items_db.new_document("Item")
        doc.set("n", i)
        executor.create_record(doc)
    executor.wait_completion()
    executor.close()

    assert len(ok) >= 1
    assert errors == []
