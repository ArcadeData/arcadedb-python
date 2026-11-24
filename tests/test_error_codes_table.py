"""The table of error codes (ArcadeData/arcadedb-python#475): one literal per row, run on both builds.

``arcadedb.exceptions._RULES`` decides the class, the ``code``, the ``hint`` and the ``cause`` of an ``ArcadeDBError``. Each literal
below is an error text, with the Java chain behind it, that the engine produces (collected by running both builds, see
``test_error_codes_parity.py`` for the live failures). ``test_every_row_of_the_table_has_a_literal`` fails when a row is added
without one. No database is opened here, so the file runs on the JVM build and the native build alike.
"""

import copy
import pickle  # nosec B403 - a round trip of an object this test built
import re
from pathlib import Path

import pytest

import arcadedb
from arcadedb import exceptions
from arcadedb.exceptions import ArcadeDBError, _classify

E = "com.arcadedb.exception."

# (id, text of the error, Java chain (outermost first), expected class, expected code, expected cause)
CASES = [
    (
        "duplicate_key",
        "Command failed: com.arcadedb.exception.DuplicatedKeyException: Duplicated key [1] found on index 'T[id]' already assigned to record #1:0",
        [
            (
                E + "DuplicatedKeyException",
                "Duplicated key [1] found on index 'T[id]' already assigned to record #1:0",
            )
        ],
        "DuplicateKeyError",
        "duplicate_key",
        E + "DuplicatedKeyException",
    ),
    (
        "record_not_found_class",
        "Failed to lookup RID '#9:9': com.arcadedb.exception.RecordNotFoundException: Record #9:9 not found",
        [(E + "RecordNotFoundException", "Record #9:9 not found")],
        "RecordNotFoundError",
        "record_not_found",
        E + "RecordNotFoundException",
    ),
    (
        "record_not_found_vertex",
        "Command failed: com.arcadedb.exception.VertexNotFoundException: Vertex #9:9 not found",
        [(E + "VertexNotFoundException", "Vertex #9:9 not found")],
        "RecordNotFoundError",
        "record_not_found",
        E + "VertexNotFoundException",
    ),
    (
        "record_not_found_text",
        "Failed to lookup RID '#9:9': record not found",
        [],
        "RecordNotFoundError",
        "record_not_found",
        None,
    ),
    (
        "transaction_not_begun",
        "Command failed: com.arcadedb.exception.TransactionException: Transaction not begun",
        [(E + "TransactionException", "Transaction not begun")],
        "TransactionError",
        "transaction_not_begun",
        E + "TransactionException",
    ),
    (
        "transaction_not_begun_text",
        "Failed to save record: com.arcadedb.exception.TransactionException: Transaction not begun (wrap the write in `with db.transaction():` or call begin() first)",
        [],
        "TransactionError",
        "transaction_not_begun",
        E + "TransactionException",
    ),
    (
        "database_locked",
        "Failed to open database: com.arcadedb.exception.DatabaseOperationException: Error on creating new database instance",
        [
            (
                E + "DatabaseOperationException",
                "Error on creating new database instance",
            ),
            (
                "com.arcadedb.utility.LockException",
                "Database 'db19' is locked by another process (path=/tmp/db19)",
            ),
        ],
        "DatabaseLockedError",
        "database_locked",
        "com.arcadedb.utility.LockException",
    ),
    (
        "database_exists",
        "Failed to create database: com.arcadedb.exception.DatabaseOperationException: Database '/tmp/db7' already exists",
        [(E + "DatabaseOperationException", "Database '/tmp/db7' already exists")],
        "DatabaseExistsError",
        "database_exists",
        E + "DatabaseOperationException",
    ),
    (
        "database_not_found",
        "Failed to open database: com.arcadedb.exception.DatabaseNotFoundException: Database '/tmp/db21' does not exist",
        [(E + "DatabaseNotFoundException", "Database '/tmp/db21' does not exist")],
        "DatabaseNotFoundError",
        "database_not_found",
        E + "DatabaseNotFoundException",
    ),
    (
        "database_closed_class",
        "Command failed: com.arcadedb.exception.DatabaseIsClosedException: Database 'db' is closed",
        [(E + "DatabaseIsClosedException", "Database 'db' is closed")],
        "DatabaseClosedError",
        "database_closed",
        E + "DatabaseIsClosedException",
    ),
    (
        "database_closed_text",
        "Database is closed",
        [],
        "DatabaseClosedError",
        "database_closed",
        None,
    ),
    (
        "database_read_only",
        "Command failed: com.arcadedb.exception.DatabaseIsReadOnlyException: Cannot execute command on a read only database",
        [
            (
                E + "DatabaseIsReadOnlyException",
                "Cannot execute command on a read only database",
            )
        ],
        "ArcadeDBError",
        "database_read_only",
        E + "DatabaseIsReadOnlyException",
    ),
    (
        "syntax_error_sql",
        "Query failed: com.arcadedb.exception.CommandSQLParsingException: SQL syntax error at line 1, column 0: no viable alternative at input 'SELEC'",
        [
            (
                E + "CommandSQLParsingException",
                "SQL syntax error at line 1, column 0: no viable alternative at input 'SELEC'",
            )
        ],
        "QuerySyntaxError",
        "syntax_error",
        E + "CommandSQLParsingException",
    ),
    (
        "syntax_error_cypher",
        "Query failed: com.arcadedb.exception.CommandParsingException: Syntax error at line 1:0 - mismatched input 'MATC', expected: an expression",
        [
            (
                E + "CommandParsingException",
                "Syntax error at line 1:0 - mismatched input 'MATC', expected: an expression",
            ),
            ("org.antlr.v4.runtime.InputMismatchException", None),
        ],
        "QuerySyntaxError",
        "syntax_error",
        E + "CommandParsingException",
    ),
    (
        "timeout",
        "Query failed: com.arcadedb.exception.TimeoutException: the command exceeded the TIMEOUT clause of 1ms",
        [(E + "TimeoutException", "the command exceeded the TIMEOUT clause of 1ms")],
        "QueryTimeoutError",
        "timeout",
        E + "TimeoutException",
    ),
    (
        "timeout_partial",
        "Query failed: com.arcadedb.exception.PartialResultTimeoutException: partial result",
        [(E + "PartialResultTimeoutException", "partial result")],
        "QueryTimeoutError",
        "timeout",
        E + "PartialResultTimeoutException",
    ),
    (
        "conflict",
        "Failed to commit transaction: com.arcadedb.exception.ConcurrentModificationException: Slot rebase not possible on page PageId(db6/1/0) slot 0 (concurrent change to the same record). Please retry the operation",
        [
            (
                E + "ConcurrentModificationException",
                "Slot rebase not possible on page PageId(db6/1/0) slot 0 (concurrent change to the same record). Please retry the operation",
            )
        ],
        "ConflictError",
        "conflict",
        E + "ConcurrentModificationException",
    ),
    (
        "conflict_lock_timeout",
        "Command failed: com.arcadedb.exception.LockTimeoutException: Timeout on locking resource",
        [(E + "LockTimeoutException", "Timeout on locking resource")],
        "ConflictError",
        "conflict",
        E + "LockTimeoutException",
    ),
    (
        "conflict_need_retry",
        "Command failed: com.arcadedb.exception.NeedRetryException: retry",
        [(E + "NeedRetryException", "retry")],
        "ConflictError",
        "conflict",
        E + "NeedRetryException",
    ),
    (
        "type_not_found_class",
        "Query failed: com.arcadedb.exception.SchemaException: Type with name 'Nope' was not found",
        [(E + "SchemaException", "Type with name 'Nope' was not found")],
        "SchemaError",
        "type_not_found",
        E + "SchemaException",
    ),
    (
        "type_not_found_text",
        "Cannot load into 'Nope': the type does not exist (create it first)",
        [],
        "SchemaError",
        "type_not_found",
        None,
    ),
    (
        "already_exists",
        "Command failed: com.arcadedb.exception.AlreadyExistsException: Type T already exists",
        [(E + "AlreadyExistsException", "Type T already exists")],
        "SchemaError",
        "already_exists",
        E + "AlreadyExistsException",
    ),
    (
        "already_exists_type",
        "Command failed: com.arcadedb.exception.TypeAlreadyExistsException: Cannot create type 'T' because already exists",
        [
            (
                E + "TypeAlreadyExistsException",
                "Cannot create type 'T' because already exists",
            )
        ],
        "SchemaError",
        "already_exists",
        E + "TypeAlreadyExistsException",
    ),
    (
        "schema_class",
        "Command failed: com.arcadedb.exception.SchemaException: Cannot create the index on type 'T.nothere' because the property does not exist",
        [
            (
                E + "SchemaException",
                "Cannot create the index on type 'T.nothere' because the property does not exist",
            )
        ],
        "SchemaError",
        "schema",
        E + "SchemaException",
    ),
    (
        "schema_text_native_property",
        "Command failed: com.arcadedb.exception.CommandExecutionException: Property 'nope' not found on class T",
        [(E + "CommandExecutionException", "Property 'nope' not found on class T")],
        "SchemaError",
        "schema",
        E + "CommandExecutionException",
    ),
    (
        "schema_text_native_index",
        "Command failed: com.arcadedb.exception.CommandExecutionException: Index not found: nope[x]",
        [(E + "CommandExecutionException", "Index not found: nope[x]")],
        "SchemaError",
        "schema",
        E + "CommandExecutionException",
    ),
    (
        "type_mismatch",
        "Command failed: java.lang.IllegalArgumentException: Cannot convert type 'class java.lang.String' to 'INTEGER' found in property 'id' (caused by java.lang.NumberFormatException)",
        [
            (
                "java.lang.IllegalArgumentException",
                "Cannot convert type 'class java.lang.String' to 'INTEGER' found in property 'id'",
            ),
            ("java.lang.NumberFormatException", 'For input string: "abc"'),
        ],
        "ValidationError",
        "type_mismatch",
        "java.lang.IllegalArgumentException",
    ),
    (
        "validation",
        "Command failed: com.arcadedb.exception.ValidationException: The property 'T.m' is mandatory, but not found on record: @T[id=1]",
        [
            (
                E + "ValidationException",
                "The property 'T.m' is mandatory, but not found on record: @T[id=1]",
            )
        ],
        "ValidationError",
        "validation",
        E + "ValidationException",
    ),
    (
        "query_not_idempotent",
        "Query failed: com.arcadedb.exception.QueryNotIdempotentException: Query 'INSERT INTO T SET id = 1' is not idempotent",
        [
            (
                E + "QueryNotIdempotentException",
                "Query 'INSERT INTO T SET id = 1' is not idempotent",
            )
        ],
        "ArcadeDBError",
        "query_not_idempotent",
        E + "QueryNotIdempotentException",
    ),
    (
        "missing_parameter",
        "Query failed: com.arcadedb.exception.CommandParameterMissingException: Expected parameter(s): x",
        [(E + "CommandParameterMissingException", "Expected parameter(s): x")],
        "ArcadeDBError",
        "missing_parameter",
        E + "CommandParameterMissingException",
    ),
]


def _ids(cases):
    return [case[0] for case in cases]


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_a_row_decides_class_code_and_cause(case):
    _, text, chain, class_name, code, cause = case
    rule, found = _classify(text, tuple(chain))
    assert rule is not None
    assert (rule.cls.__name__, rule.code, found) == (class_name, code, cause)


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_the_same_failure_with_and_without_the_chain_gives_the_same_class_and_code(
    case,
):
    """The JVM build hands the chain of the Java exception, the native build the chain of the reply, a text-only error neither."""
    _, text, chain, class_name, code, _ = case
    from_chain = ArcadeDBError._classified(text, chain)
    assert (type(from_chain).__name__, from_chain.code) == (class_name, code)
    if (
        len(chain) <= 1
    ):  # a text names the outer class only; the inner ones come with the chain
        from_text = ArcadeDBError(text)
        assert (type(from_text).__name__, from_text.code) == (class_name, code)


def test_every_row_of_the_table_has_a_literal():
    """A new row needs a literal above; a row that an earlier row hides is found here too."""
    hit = set()
    for _, text, chain, _, _, _ in CASES:
        rule, _ = _classify(text, tuple(chain))
        hit.add(id(rule))
    missing = [rule.code for rule in exceptions._RULES if id(rule) not in hit]
    assert not missing, f"rows without a literal that reaches them: {missing}"


def test_a_row_names_a_class_of_this_module_and_a_hint_of_one_sentence():
    for rule in exceptions._RULES:
        assert issubclass(rule.cls, ArcadeDBError)
        assert re.fullmatch(r"[a-z]+(_[a-z]+)*", rule.code)
        assert rule.hint.endswith(".") and "\n" not in rule.hint
        assert rule.java or rule.message is not None, rule.code
        assert chr(0x2014) not in rule.hint


def test_the_class_of_a_code_is_the_same_in_every_row_of_that_code():
    seen = {}
    for rule in exceptions._RULES:
        assert seen.setdefault(rule.code, (rule.cls, rule.hint)) == (
            rule.cls,
            rule.hint,
        )


def test_a_failure_the_table_does_not_know_is_an_arcadedb_error_with_code_unknown():
    error = ArcadeDBError("Something odd happened")
    assert type(error) is ArcadeDBError
    assert (error.code, error.hint, error.cause) == ("unknown", None, None)
    assert str(error) == "Something odd happened"
    other = ArcadeDBError._classified(
        "Command failed: com.arcadedb.exception.CommandExecutionException: boom",
        [(E + "CommandExecutionException", "boom"), ("java.io.IOException", "disk")],
    )
    assert type(other) is ArcadeDBError and other.code == "unknown"
    assert other.cause == "java.io.IOException"
    assert (
        str(other)
        == "Command failed: com.arcadedb.exception.CommandExecutionException: boom"
    )


def test_a_transaction_exception_with_another_message_is_unknown():
    error = ArcadeDBError._classified(
        "Command failed: com.arcadedb.exception.TransactionException: something else",
        [(E + "TransactionException", "something else")],
    )
    assert (type(error), error.code) == (ArcadeDBError, "unknown")


def test_no_arguments_and_a_non_text_argument_are_accepted():
    assert ArcadeDBError().code == "unknown"
    assert ArcadeDBError(12).code == "unknown"
    assert str(ArcadeDBError()) == ""


def test_str_keeps_the_old_text_as_its_tail_after_the_code_and_the_hint():
    text = "Command failed: com.arcadedb.exception.DuplicatedKeyException: Duplicated key [1] found on index 'T[id]' already assigned to record #1:0"
    error = ArcadeDBError(text)
    shown = str(error)
    assert shown.startswith("[duplicate_key] " + error.hint + " ")
    assert shown.endswith(text)
    assert error.args == (text,)
    assert not isinstance(error, (KeyError, ValueError))


def test_the_old_caused_by_suffix_stays_in_the_tail():
    error = ArcadeDBError._classified(
        "Failed to open database: x", [(E + "DatabaseOperationException", "outer")]
    )
    error.java_chain = [
        (E + "DatabaseOperationException", "outer"),
        ("com.arcadedb.utility.LockException", "locked"),
    ]
    assert str(error).endswith(
        "Failed to open database: x (caused by com.arcadedb.utility.LockException: locked)"
    )


def test_to_dict_has_the_five_keys_and_is_plain_data():
    error = ArcadeDBError._classified(
        "Command failed: com.arcadedb.exception.TransactionException: Transaction not begun",
        [(E + "TransactionException", "Transaction not begun")],
    )
    data = error.to_dict()
    assert list(data) == ["type", "code", "hint", "cause", "message"]
    assert data["type"] == "TransactionError"
    assert data["code"] == "transaction_not_begun"
    assert data["cause"] == E + "TransactionException"
    assert (
        data["message"]
        == "Command failed: com.arcadedb.exception.TransactionException: Transaction not begun"
    )
    assert data["hint"] == error.hint
    unknown = ArcadeDBError("odd").to_dict()
    assert unknown == {
        "type": "ArcadeDBError",
        "code": "unknown",
        "hint": None,
        "cause": None,
        "message": "odd",
    }


def test_the_hint_for_a_missing_transaction_is_true_with_auto_transaction_on_and_off():
    hint = ArcadeDBError._classified(
        "Command failed: com.arcadedb.exception.TransactionException: Transaction not begun",
        [(E + "TransactionException", "Transaction not begun")],
    ).hint
    assert "with db.transaction():" in hint and "db.set_auto_transaction(True)" in hint


def test_a_subclass_built_by_hand_carries_its_code_and_hint():
    error = exceptions.DuplicateKeyError("my own text")
    assert error.code == "duplicate_key"
    assert error.hint == exceptions._RULE_OF_CODE["duplicate_key"].hint
    assert str(error) == f"[duplicate_key] {error.hint} my own text"
    assert isinstance(error, ArcadeDBError)


def test_the_subclasses_are_public_and_under_the_base_class():
    names = [
        "ConflictError",
        "DatabaseClosedError",
        "DatabaseExistsError",
        "DatabaseLockedError",
        "DatabaseNotFoundError",
        "DuplicateKeyError",
        "QuerySyntaxError",
        "QueryTimeoutError",
        "RecordNotFoundError",
        "SchemaError",
        "TransactionError",
        "ValidationError",
    ]
    for name in names:
        assert name in arcadedb.__all__
        assert issubclass(getattr(arcadedb, name), arcadedb.ArcadeDBError)
        assert getattr(arcadedb, name) is getattr(exceptions, name)


def test_a_raise_site_picks_the_subclass_from_the_text_inside_an_except_block():
    try:
        try:
            raise ValueError("not a Java exception")
        except ValueError as e:
            raise ArcadeDBError(
                "Command failed: com.arcadedb.exception.DuplicatedKeyException: Duplicated key [1]"
            ) from e
    except exceptions.DuplicateKeyError as error:
        assert error.code == "duplicate_key"
        assert error.__cause__.args == ("not a Java exception",)


def test_a_wrapper_around_a_classified_error_keeps_the_class_and_does_not_repeat_the_head():
    text = "Command failed: com.arcadedb.exception.DuplicatedKeyException: Duplicated key [1]"
    try:
        try:
            raise ArcadeDBError(text)
        except ArcadeDBError as e:
            raise ArcadeDBError(f"Failed to commit transaction: {e}") from e
    except exceptions.DuplicateKeyError as outer:
        shown = str(outer)
        assert shown.count("[duplicate_key]") == 1
        assert shown.endswith("Failed to commit transaction: " + text)
        assert outer.args == ("Failed to commit transaction: " + text,)


def test_a_statement_that_a_syntax_error_echoes_does_not_decide_the_code():
    text = "Query failed: com.arcadedb.exception.CommandSQLParsingException: SQL syntax error at line 1: SELECT 'record not found' FROM"
    error = ArcadeDBError._classified(
        text,
        [
            (
                E + "CommandSQLParsingException",
                "SQL syntax error at line 1: SELECT 'record not found' FROM",
            )
        ],
    )
    assert error.code == "syntax_error"


def test_pickle_and_copy_keep_the_class_and_the_attributes():
    error = ArcadeDBError._classified(
        "Failed to create database: com.arcadedb.exception.DatabaseOperationException: Database 'x' already exists",
        [(E + "DatabaseOperationException", "Database 'x' already exists")],
    )
    for clone in (
        pickle.loads(pickle.dumps(error)),  # nosec B301 - same
        copy.copy(error),
        copy.deepcopy(error),
    ):
        assert type(clone) is exceptions.DatabaseExistsError
        assert (clone.code, clone.hint, clone.cause) == (
            error.code,
            error.hint,
            error.cause,
        )
        assert str(clone) == str(error)


def test_the_docs_page_lists_every_code_class_and_hint_of_the_table():
    page = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "modules"
        / "ROOT"
        / "pages"
        / "how-to"
        / "errors.adoc"
    )
    if not page.is_file():
        pytest.fail("docs/modules/ROOT/pages/how-to/errors.adoc is missing")
    text = page.read_text(encoding="utf-8")
    for rule in exceptions._RULES:
        assert f"`{rule.code}`" in text, rule.code
        assert f"`{rule.cls.__name__}`" in text, rule.cls.__name__
        assert rule.hint in text, rule.code
