"""Live failures give the same class and code on both builds (ArcadeData/arcadedb-python#475).

Same pattern as ``test_cross_wheel_golden.py``: there is no branch on ``arcadedb.__backend__`` here, and the literals below are
asserted on ``arcadedb`` (native) and on ``arcadedb-jvm`` alike, so a literal that holds on both IS the parity check. Each scenario
makes the engine fail the way an agent's statement fails and checks the class, the ``code``, the ``cause`` and the text of the
error. A ``cause`` of ``None`` in the table means that it is not compared: the two builds word those failures differently (the
native build's engine reports ``CommandExecutionException`` where the JVM build's reports ``SchemaException``, and the Python
layer of the native build raises the missing record itself), and the class and the code are the same all the same.
"""

import json
import subprocess  # nosec B404 - test-controlled child process
import sys
import threading

import pytest

import arcadedb
from arcadedb import ArcadeDBError

E = "com.arcadedb.exception."


def _db(tmp_path, name="db"):
    """A database with a document type T (unique ``id``), a vertex type V and an edge type E. Writes use ``db.transaction()``."""
    db = arcadedb.create_database(str(tmp_path / name))
    db.command("sql", "CREATE DOCUMENT TYPE T")
    db.command("sql", "CREATE PROPERTY T.id INTEGER")
    db.command("sql", "CREATE INDEX ON T (id) UNIQUE")
    db.command("sql", "CREATE VERTEX TYPE V")
    db.command("sql", "CREATE EDGE TYPE E")
    return db


def _insert(db, value=1):
    with db.transaction():
        db.command("sql", "INSERT INTO T SET id = ?", value)


# ------------------------------------------------------------------------------------------------------ the scenarios
# Each one takes the temporary directory, runs with a database that it closes itself, and must raise.


def open_missing(tmp_path):
    arcadedb.open_database(str(tmp_path / "missing"))


def create_existing(tmp_path):
    arcadedb.create_database(str(tmp_path / "db")).close()
    arcadedb.create_database(str(tmp_path / "db"))


def read_only_write(tmp_path):
    db = _db(tmp_path)
    db.close()
    with arcadedb.open_database(str(tmp_path / "db"), read_only=True) as ro:
        with ro.transaction():
            ro.command("sql", "INSERT INTO T SET id = 1")


def on_a_database(action):
    def scenario(tmp_path):
        with _db(tmp_path) as db:
            action(db)

    scenario.__name__ = action.__name__
    return scenario


def sql_syntax_query(db):
    db.query("sql", "SELEC FROM T").to_list()


def sql_syntax_command(db):
    db.command("sql", "SELEC FROM T")


def cypher_syntax(db):
    db.query("opencypher", "MATC (n) RETURN n").to_list()


def duplicate_in_transaction(db):
    _insert(db)
    _insert(db)


def duplicate_in_one_transaction(db):
    with db.transaction():
        db.command("sql", "INSERT INTO T SET id = 1")
        db.command("sql", "INSERT INTO T SET id = 1")


def duplicate_with_auto_transaction(db):
    db.set_auto_transaction(True)
    db.command("sql", "INSERT INTO T SET id = 1")
    db.command("sql", "INSERT INTO T SET id = 1")


def duplicate_in_insert_many(db):
    db.insert_many("T", [{"id": 1}, {"id": 1}])


def duplicate_in_save(db):
    _insert(db)
    with db.transaction():
        document = db.new_document("T")
        document.set("id", 1)
        document.save()


def lookup_missing_rid(db):
    db.lookup_by_rid("#9:9")


def lookup_deleted_record(db):
    _insert(db)
    rid = db.query("sql", "SELECT FROM T").first().get_rid()
    with db.transaction():
        db.command("sql", "DELETE FROM T")
    db.lookup_by_rid(rid)


def edge_between_missing_records(db):
    with db.transaction():
        db.command("sql", "CREATE EDGE E FROM #9:9 TO #9:8")


def write_in_strict_mode(db):
    db.set_auto_transaction(False)
    db.command("sql", "INSERT INTO T SET id = 1")


def save_in_strict_mode(db):
    db.set_auto_transaction(False)
    document = db.new_document("T")
    document.set("id", 1)
    document.save()


def commit_with_nothing_open(db):
    db.commit()


def query_after_close(db):
    db.close()
    db.query("sql", "SELECT FROM T")


def command_after_close(db):
    db.close()
    db.command("sql", "CREATE DOCUMENT TYPE Q")


def timeout_of_a_query(db):
    db.insert_many("T", [{"id": i + 10} for i in range(200_000)])
    db.query(
        "sql", "SELECT count(*) FROM T WHERE (id * id) % 7 = 3 TIMEOUT 1 EXCEPTION"
    ).to_list()


def timeout_of_a_command(db):
    db.insert_many("T", [{"id": i + 10} for i in range(200_000)])
    with db.transaction():
        db.command("sql", "UPDATE T SET k = id + 1 TIMEOUT 1 EXCEPTION")


def conflict_of_two_transactions(db):
    _insert(db)
    errors = []
    barrier = threading.Barrier(2, timeout=60)

    def slow_writer():
        try:
            with db.transaction():
                db.command("sql", "UPDATE T SET k = 2 WHERE id = 1")
                barrier.wait()  # the other writer commits now
                barrier.wait()
        except BaseException as e:  # noqa: BLE001 - handed to the main thread
            errors.append(e)

    thread = threading.Thread(target=slow_writer)
    thread.start()
    barrier.wait()
    try:
        with db.transaction():
            db.command("sql", "UPDATE T SET k = 3 WHERE id = 1")
    finally:
        barrier.wait()
        thread.join(timeout=60)
    if errors:
        raise errors[0]


def conflict_after_a_commit_in_the_block(db):
    _insert(db)

    def other_writer():
        with db.transaction():
            db.command("sql", "UPDATE T SET k = 2 WHERE id = 1")

    with db.transaction():
        db.command("sql", "INSERT INTO T SET id = 2")
        db.commit()  # stored from here on: the block must not be run again
        db.begin()
        db.command("sql", "UPDATE T SET k = 3 WHERE id = 1")
        writer = threading.Thread(target=other_writer)
        writer.start()
        writer.join(timeout=60)


def unknown_type_in_query(db):
    db.query("sql", "SELECT FROM Nope").to_list()


def unknown_type_in_insert(db):
    with db.transaction():
        db.command("sql", "INSERT INTO Nope SET a = 1")


def drop_unknown_type(db):
    db.command("sql", "DROP TYPE Nope")


def create_a_type_twice(db):
    db.command("sql", "CREATE DOCUMENT TYPE T")


def create_a_property_twice(db):
    db.command("sql", "CREATE PROPERTY T.id INTEGER")


def create_an_index_twice(db):
    db.command("sql", "CREATE INDEX ON T (id) UNIQUE")


def index_on_a_missing_property(db):
    db.command("sql", "CREATE INDEX ON T (nothere) NOTUNIQUE")


def drop_a_missing_property(db):
    db.command("sql", "DROP PROPERTY T.nope")


def drop_a_missing_index(db):
    db.command("sql", "DROP INDEX `nope[x]`")


def mandatory_property_missing(db):
    db.command("sql", "CREATE PROPERTY T.m STRING (mandatory true)")
    with db.transaction():
        db.command("sql", "INSERT INTO T SET id = 1")


def not_null_property_null(db):
    db.command("sql", "CREATE PROPERTY T.m STRING (notnull true)")
    with db.transaction():
        db.command("sql", "INSERT INTO T SET id = 1, m = null")


def property_above_max(db):
    db.command("sql", "CREATE PROPERTY T.k INTEGER (max 5)")
    with db.transaction():
        db.command("sql", "INSERT INTO T SET id = 1, k = 50")


def readonly_property_changed(db):
    db.command("sql", "CREATE PROPERTY T.r STRING (readonly true)")
    with db.transaction():
        db.command("sql", "INSERT INTO T SET id = 1, r = 'a'")
    with db.transaction():
        db.command("sql", "UPDATE T SET r = 'b'")


def text_for_an_integer(db):
    with db.transaction():
        db.command("sql", "INSERT INTO T SET id = 'abc'")


def write_through_query(db):
    db.query("sql", "INSERT INTO T SET id = 1")


def parameter_not_passed(db):
    db.query("sql", "SELECT FROM T WHERE id = :x", {}).to_list()


def unknown_query_language(db):
    db.query("nolang", "SELECT FROM T")


def unknown_property_type(db):
    db.command("sql", "CREATE PROPERTY T.z FOOBAR")


def division_by_zero(db):
    db.query("sql", "SELECT 1/0 AS x").to_list()


def unknown_function(db):
    db.query("sql", "SELECT nosuchfn(1) AS x").to_list()


# (scenario, expected class name, expected code, expected cause; None: the cause is not compared, see the docstring)
SCENARIOS = [
    (
        open_missing,
        "DatabaseNotFoundError",
        "database_not_found",
        E + "DatabaseNotFoundException",
    ),
    (
        create_existing,
        "DatabaseExistsError",
        "database_exists",
        E + "DatabaseOperationException",
    ),
    (
        read_only_write,
        "ArcadeDBError",
        "database_read_only",
        E + "DatabaseIsReadOnlyException",
    ),
    (
        on_a_database(sql_syntax_query),
        "QuerySyntaxError",
        "syntax_error",
        E + "CommandSQLParsingException",
    ),
    (
        on_a_database(sql_syntax_command),
        "QuerySyntaxError",
        "syntax_error",
        E + "CommandSQLParsingException",
    ),
    (
        on_a_database(cypher_syntax),
        "QuerySyntaxError",
        "syntax_error",
        E + "CommandParsingException",
    ),
    (
        on_a_database(duplicate_in_transaction),
        "DuplicateKeyError",
        "duplicate_key",
        E + "DuplicatedKeyException",
    ),
    (
        on_a_database(duplicate_in_one_transaction),
        "DuplicateKeyError",
        "duplicate_key",
        E + "DuplicatedKeyException",
    ),
    (
        on_a_database(duplicate_with_auto_transaction),
        "DuplicateKeyError",
        "duplicate_key",
        E + "DuplicatedKeyException",
    ),
    (
        on_a_database(duplicate_in_insert_many),
        "DuplicateKeyError",
        "duplicate_key",
        E + "DuplicatedKeyException",
    ),
    (
        on_a_database(duplicate_in_save),
        "DuplicateKeyError",
        "duplicate_key",
        E + "DuplicatedKeyException",
    ),
    (
        on_a_database(lookup_missing_rid),
        "RecordNotFoundError",
        "record_not_found",
        None,
    ),
    (
        on_a_database(lookup_deleted_record),
        "RecordNotFoundError",
        "record_not_found",
        None,
    ),
    (
        on_a_database(edge_between_missing_records),
        "RecordNotFoundError",
        "record_not_found",
        E + "RecordNotFoundException",
    ),
    (
        on_a_database(write_in_strict_mode),
        "TransactionError",
        "transaction_not_begun",
        E + "TransactionException",
    ),
    (
        on_a_database(save_in_strict_mode),
        "TransactionError",
        "transaction_not_begun",
        E + "TransactionException",
    ),
    (
        on_a_database(commit_with_nothing_open),
        "TransactionError",
        "transaction_not_begun",
        E + "TransactionException",
    ),
    (on_a_database(query_after_close), "DatabaseClosedError", "database_closed", None),
    (
        on_a_database(command_after_close),
        "DatabaseClosedError",
        "database_closed",
        None,
    ),
    (
        on_a_database(timeout_of_a_query),
        "QueryTimeoutError",
        "timeout",
        E + "TimeoutException",
    ),
    (
        on_a_database(timeout_of_a_command),
        "QueryTimeoutError",
        "timeout",
        E + "TimeoutException",
    ),
    # the packages raise this one themselves, so no Java class decides it
    (
        on_a_database(conflict_after_a_commit_in_the_block),
        "PartialCommitError",
        "partial_commit",
        None,
    ),
    (
        on_a_database(conflict_of_two_transactions),
        "ConflictError",
        "conflict",
        E + "ConcurrentModificationException",
    ),
    (
        on_a_database(unknown_type_in_query),
        "SchemaError",
        "type_not_found",
        E + "SchemaException",
    ),
    (
        on_a_database(unknown_type_in_insert),
        "SchemaError",
        "type_not_found",
        E + "SchemaException",
    ),
    (
        on_a_database(drop_unknown_type),
        "SchemaError",
        "type_not_found",
        E + "SchemaException",
    ),
    (
        on_a_database(create_a_type_twice),
        "SchemaError",
        "already_exists",
        E + "AlreadyExistsException",
    ),
    (
        on_a_database(create_a_property_twice),
        "SchemaError",
        "already_exists",
        E + "AlreadyExistsException",
    ),
    (
        on_a_database(create_an_index_twice),
        "SchemaError",
        "already_exists",
        E + "AlreadyExistsException",
    ),
    (
        on_a_database(index_on_a_missing_property),
        "SchemaError",
        "schema",
        E + "SchemaException",
    ),
    (on_a_database(drop_a_missing_property), "SchemaError", "schema", None),
    (on_a_database(drop_a_missing_index), "SchemaError", "schema", None),
    (
        on_a_database(mandatory_property_missing),
        "ValidationError",
        "validation",
        E + "ValidationException",
    ),
    (
        on_a_database(not_null_property_null),
        "ValidationError",
        "validation",
        E + "ValidationException",
    ),
    (
        on_a_database(property_above_max),
        "ValidationError",
        "validation",
        E + "ValidationException",
    ),
    (
        on_a_database(readonly_property_changed),
        "ValidationError",
        "validation",
        E + "ValidationException",
    ),
    (
        on_a_database(text_for_an_integer),
        "ValidationError",
        "type_mismatch",
        "java.lang.IllegalArgumentException",
    ),
    (
        on_a_database(write_through_query),
        "ArcadeDBError",
        "query_not_idempotent",
        E + "QueryNotIdempotentException",
    ),
    (
        on_a_database(parameter_not_passed),
        "ArcadeDBError",
        "missing_parameter",
        E + "CommandParameterMissingException",
    ),
    # failures that the table does not know stay ArcadeDBError with the code unknown, with the same text as before
    (
        on_a_database(unknown_query_language),
        "ArcadeDBError",
        "unknown",
        "java.lang.IllegalArgumentException",
    ),
    (
        on_a_database(unknown_property_type),
        "ArcadeDBError",
        "unknown",
        "java.lang.IllegalArgumentException",
    ),
    (
        on_a_database(division_by_zero),
        "ArcadeDBError",
        "unknown",
        E + "ArithmeticErrorException",
    ),
    (on_a_database(unknown_function), "ArcadeDBError", "unknown", None),
]


@pytest.mark.parametrize("row", SCENARIOS, ids=[row[0].__name__ for row in SCENARIOS])
def test_a_failure_has_the_same_class_code_and_cause_on_both_builds(tmp_path, row):
    scenario, class_name, code, cause = row
    with pytest.raises(ArcadeDBError) as caught:
        scenario(tmp_path)
    error = caught.value
    assert (type(error).__name__, error.code) == (class_name, code)
    if cause is not None:
        assert error.cause == cause
        assert cause in str(error)  # the Java text stays in the message
    if code == "unknown":
        assert error.hint is None
        assert not str(error).startswith("[")
    else:
        assert error.hint
        assert str(error).startswith(f"[{code}] {error.hint} ")
    data = error.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert data["code"] == code and data["type"] == class_name


def test_every_code_of_the_table_but_the_lock_is_reached_by_a_live_failure_here():
    """A row of the table that no scenario reaches would be checked by a literal only: this lists what is left."""
    from arcadedb import exceptions

    live = {row[2] for row in SCENARIOS} | {"database_locked"}
    assert {rule.code for rule in exceptions._RULES} <= live


# ---------------------------------------------------------------------------------------- a second process holds the lock

CHILD = """
import sys
import arcadedb

try:
    arcadedb.open_database(sys.argv[1])
except arcadedb.ArcadeDBError as e:
    print(type(e).__name__, e.code, e.cause)
"""


def test_a_database_that_another_process_has_open_is_locked(tmp_path):
    with arcadedb.create_database(str(tmp_path / "db")):
        done = subprocess.run(  # nosec B603 - fixed argv, no shell, test-owned
            [sys.executable, "-c", CHILD, str(tmp_path / "db")],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    assert done.stdout.split() == [
        "DatabaseLockedError",
        "database_locked",
        "com.arcadedb.utility.LockException",
    ], (done.stdout, done.stderr[-500:])
