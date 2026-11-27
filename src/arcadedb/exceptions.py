"""
ArcadeDB Python Bindings - Exception Classes

All custom exceptions for the ArcadeDB Python bindings.

Every failure is an ``ArcadeDBError``. The error also carries ``code`` (a short stable name of the cause), ``hint`` (one sentence on
what to do next), ``cause`` (the Java exception class) and ``to_dict()``, and a subclass for the causes a program is likely to
branch on. The table that decides all of it is ``_RULES`` at the end of this module.
"""

import re
import sys
from typing import NamedTuple, Optional, Tuple


def _java_root_cause(exc):
    """The innermost cause of a Java exception, or None.

    None when ``exc`` is not a Java exception or has no nested cause (its own
    message is then already in the wrapping text).
    """
    get_cause = getattr(exc, "getCause", None)
    if get_cause is None:
        return None
    root, seen = None, set()
    cause = get_cause()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        root = cause
        cause = cause.getCause()
    return root


_MAX_CAUSE_DEPTH = 64
_RETRY_CLASSES = None


def _retry_classes():
    """``(NeedRetryException, DuplicatedKeyException, GlobalConfiguration)``.

    Resolved once, by name through the JVM rather than the ``com`` import
    hook, which a ``com/`` folder on ``sys.path`` shadows. Two threads racing
    the first call resolve the same classes, so the unlocked assignment is
    harmless. None while the JVM is down: no Java exception can exist then.
    """
    global _RETRY_CLASSES
    if _RETRY_CLASSES is None:
        import jpype

        if not jpype.isJVMStarted():
            return None
        _RETRY_CLASSES = (
            jpype.JClass("com.arcadedb.exception.NeedRetryException"),
            jpype.JClass("com.arcadedb.exception.DuplicatedKeyException"),
            jpype.JClass("com.arcadedb.GlobalConfiguration"),
        )
    return _RETRY_CLASSES


def _java_exceptions(exc):
    """Every Java exception ``exc`` carries, outermost first.

    Follows the Python ``__cause__`` chain the binding builds with
    ``raise ArcadeDBError(...) from e`` down to the first Java exception, then
    that exception's own ``getCause()`` chain: the engine wraps a conflict in a
    generic outer exception on some paths (#9487), and the type that decides
    whether to retry is the one inside.
    """
    # Bounded as well as deduplicated: JPype may hand out a new Python wrapper
    # for the same Java object on every getCause(), so id() alone cannot see a
    # cycle in a Java cause chain.
    seen = set()
    for _ in range(_MAX_CAUSE_DEPTH):
        if exc is None or id(exc) in seen:
            return
        seen.add(id(exc))
        if hasattr(exc, "getCause") and hasattr(exc, "getClass"):
            yield exc
            exc = exc.getCause()
        else:
            exc = exc.__cause__


def _bounded_retry_after_ms(retry_after_ms, global_configuration):
    """The part of a refusal's wait a retry loop honors (#8617).

    All of it up to ``arcadedb.network.retryAfterMaxWait``, the cap the Java
    client applies, so a misbehaving peer cannot park the caller.
    """
    if retry_after_ms <= 0:
        return 0
    cap = int(global_configuration.NETWORK_RETRY_AFTER_MAX_WAIT.getValueAsLong())
    return min(retry_after_ms, max(0, cap))


def _retry_verdict(exc):
    """Whether ``exc`` is a conflict a transaction retry can absorb.

    Returns ``(kind, retry_after_ms)``: ``kind`` is ``"need_retry"`` for any
    ``NeedRetryException`` (``ConcurrentModificationException``,
    ``LockTimeoutException``, ...), ``"duplicated_key"`` for a
    ``DuplicatedKeyException`` and None for everything else;
    ``retry_after_ms`` is the wait the refusing side asked for
    (``NeedRetryException.getRetryAfterMs()``, #8617), 0 when it did not say.

    The decision is by Java TYPE, the way ``LocalDatabase.transaction()``
    makes it with ``catch (NeedRetryException | DuplicatedKeyException e)``,
    so a subclass is covered without being named. Searching the message for
    class names missed every subclass that was not listed, which is how a
    ``LockTimeoutException`` escaped ``run_in_transaction`` on its first
    attempt (#9629).
    """
    classes = _retry_classes()
    if classes is None:
        return None, 0
    need_retry, duplicated_key, global_configuration = classes
    for java_exc in _java_exceptions(exc):
        if isinstance(java_exc, need_retry):
            return "need_retry", _bounded_retry_after_ms(
                int(java_exc.getRetryAfterMs()), global_configuration
            )
        if isinstance(java_exc, duplicated_key):
            return "duplicated_key", 0
    return None, 0


def _java_chain_of(exc):
    """``((Java class name, message or None), ...)`` of the Java exceptions behind ``exc``, outermost first.

    ``exc`` is the exception being handled when an error is built. A Java exception gives its own ``getCause()`` chain, an
    ``ArcadeDBError`` the chain it was built from (a wrapper that re-raises it keeps the classification), anything else nothing.
    """
    if exc is None:
        return ()
    if isinstance(exc, ArcadeDBError):
        return exc._chain
    chain = []
    try:
        for java_exc in _java_exceptions(exc):
            message = java_exc.getMessage()
            chain.append(
                (
                    str(java_exc.getClass().getName()),
                    None if message is None else str(message),
                )
            )
    except Exception:  # nosec B110 - building an error must never fail
        pass
    return tuple(chain)


class ArcadeDBError(Exception):
    """Base exception for ArcadeDB errors.

    When raised from a Java exception, ``str()`` also names the Java root
    cause: the engine often wraps the reason (for example "is locked by
    another process") in a generic outer exception.

    What a program or a model reads to repair its statement or call:

    * ``code``: a short stable name of the cause (``"duplicate_key"``,
      ``"record_not_found"``, ``"transaction_not_begun"``, ...), or
      ``"unknown"`` for a failure that the table of codes does not know.
    * ``hint``: one sentence on what to do next, or None for ``"unknown"``.
    * ``cause``: the name of the Java exception class that decided the code
      (the root cause of the Java chain for ``"unknown"``), or None when no
      Java exception is involved.
    * ``to_dict()``: those three, the class name and the message.

    ``str()`` of a classified error is ``[code] hint`` followed by the text
    that ``str()`` had before (the Java class and message stay at its end).
    An unknown error prints exactly that older text.

    ``ArcadeDBError(text)`` picks the subclass itself, from the Java
    exception being handled (or, with none, from the class names and the
    wording in ``text``), so ``except DuplicateKeyError`` works for every
    raise site. Both builds give the same class and code for the same
    failure.
    """

    code = "unknown"
    hint = None
    cause = None
    _chain = ()
    _args = None

    def __new__(cls, *args, **kwargs):
        normalized = None
        if cls is ArcadeDBError:
            text = str(args[0]) if args else ""
            try:
                handled = sys.exc_info()[1]
                if isinstance(handled, ArcadeDBError) and handled.hint:
                    # a wrapper that prints the error it caught ("Failed to X: {e}") gets its text without the head,
                    # which the new error adds itself
                    shown = str(handled)
                    if shown in text:
                        text = text.replace(shown, handled._text())
                        args = (text,) + args[1:]
                        normalized = args
                chain = _java_chain_of(handled)
                rule, cause = _classify(text, chain)
            except Exception:  # nosec B110 - building an error must never fail
                chain, rule, cause = (), None, None
            target = ArcadeDBError if rule is None else rule.cls
        else:
            chain, cause = (), None
            rule = _RULE_OF_CODE.get(cls.code)
            target = cls
        self = super().__new__(target, *args, **kwargs)
        self._decorate(rule, cause, chain)
        self._args = normalized
        return self

    def __init__(self, *args, **kwargs):
        # the call that built the error passes its own arguments again; a wrapper's text without the head replaces them
        super().__init__(*(self._args if self._args is not None else args), **kwargs)

    def _decorate(self, rule, cause, chain):
        self._chain = chain
        if rule is not None:
            self.code = rule.code
            self.hint = rule.hint
        self.cause = cause

    @classmethod
    def _classified(cls, text, chain):
        """The error for ``text`` and a Java chain that arrived as data (the native build's reply)."""
        chain = tuple((str(name), message) for name, message in chain)
        rule, cause = _classify(text, chain)
        target = ArcadeDBError if rule is None else rule.cls
        exc = Exception.__new__(target, text)
        Exception.__init__(exc, text)
        exc._decorate(rule, cause, chain)
        return exc

    def to_dict(self):
        """``{"type", "code", "hint", "cause", "message"}``: the class name, the three attributes and the text without the head."""
        return {
            "type": type(self).__name__,
            "code": self.code,
            "hint": self.hint,
            "cause": self.cause,
            "message": self._text(),
        }

    def _text(self):
        text = super().__str__()
        chain = getattr(self, "java_chain", None)
        if chain and len(chain) > 1:
            # native backend: the cause chain came over as text, innermost last
            name, message = chain[-1]
            if not message or message not in text:
                detail = f"{name}: {message}" if message and message != "None" else name
                text = f"{text} (caused by {detail})"
            return text
        try:
            root = _java_root_cause(self.__cause__)
            if root is not None:
                message = root.getMessage()
                message = "" if message is None else str(message)
                if not message or message not in text:
                    name = str(root.getClass().getName())
                    detail = f"{name}: {message}" if message else name
                    text = f"{text} (caused by {detail})"
        except Exception:  # nosec B110 - never let str() of an error fail
            pass
        return text

    def __str__(self):
        text = self._text()
        if self.hint is None:
            return text
        return f"[{self.code}] {self.hint} {text}"


class DuplicateKeyError(ArcadeDBError):
    """A unique index already holds the key (``duplicate_key``)."""

    code = "duplicate_key"


class RecordNotFoundError(ArcadeDBError):
    """No record has the RID (``record_not_found``)."""

    code = "record_not_found"


class TransactionError(ArcadeDBError):
    """A transaction call was made in the wrong state (``transaction_not_begun``)."""

    code = "transaction_not_begun"


class DatabaseExistsError(ArcadeDBError):
    """``create_database`` found a database at the path (``database_exists``)."""

    code = "database_exists"


class DatabaseNotFoundError(ArcadeDBError):
    """``open_database`` found no database at the path (``database_not_found``)."""

    code = "database_not_found"


class DatabaseLockedError(ArcadeDBError):
    """Another process has the database open (``database_locked``)."""

    code = "database_locked"


class DatabaseClosedError(ArcadeDBError):
    """The call needs an open database and this one was closed (``database_closed``)."""

    code = "database_closed"


class QuerySyntaxError(ArcadeDBError):
    """The statement does not parse (``syntax_error``)."""

    code = "syntax_error"


class QueryTimeoutError(ArcadeDBError):
    """The statement ran past its TIMEOUT clause (``timeout``)."""

    code = "timeout"


class ConflictError(ArcadeDBError):
    """Another transaction changed the same data first, so a retry can succeed (``conflict``)."""

    code = "conflict"


class PartialCommitError(ArcadeDBError):
    """A block lost a conflict after part of its work was committed, so running it again applies that part twice (``partial_commit``).

    Not a :class:`ConflictError`, and the conflict is not its ``__cause__``, so that no retry loop runs the block again. When
    ``run_in_transaction`` or ``with db.transaction():`` raise it, the conflict is its ``__context__``.
    """

    code = "partial_commit"


class SchemaError(ArcadeDBError):
    """A type, property or index is missing, exists already, or refuses the change (``type_not_found``, ``already_exists``, ``schema``)."""

    code = "schema"


class ValidationError(ArcadeDBError):
    """A record broke a property constraint or a value has the wrong type (``validation``, ``type_mismatch``)."""

    code = "validation"


class _Rule(NamedTuple):
    """One row of the table of codes.

    A rule matches when one of its ``java`` class names is in the Java chain of the error (``java`` empty: any chain) and its
    ``message`` pattern is found in the text of the error or in a message of the chain (``message`` None: any text). The first
    rule that matches decides, so a row with a general condition comes after the rows with a narrower one.
    """

    code: str
    cls: type
    hint: str
    java: Tuple[str, ...] = ()
    message: Optional["re.Pattern"] = None


_E = "com.arcadedb.exception."

_HINT_RECORD_NOT_FOUND = (
    "No record has this RID: check the RID, or look the record up by a property with a query "
    "(it may have been deleted, or it was never committed)."
)
_HINT_DATABASE_CLOSED = (
    "The database was closed: open it again with `open_database(path)`, and read what you need from a "
    "result before you close the database."
)
_HINT_TYPE_NOT_FOUND = (
    "No type has this name: check the spelling against `SELECT name FROM schema:types`, or create "
    "the type first with `CREATE DOCUMENT TYPE`, `CREATE VERTEX TYPE` or `CREATE EDGE TYPE`."
)
_HINT_SCHEMA = (
    "The schema does not allow this change: read the message, and compare it with the current "
    "schema (`SELECT FROM schema:types`)."
)

# The table that decides the class, the code and the hint of every error. The Java names and the texts are the ones the engine
# produces, collected by running both builds (tests/test_error_codes_table.py holds a literal for every row). A row with a
# ``message`` and no ``java`` is for a failure that the Python layer raises itself or that one build words differently.
_RULES = (
    # first: the engine wraps this refusal in a timeout when a COMMIT RETRY block ran out of time, and the half that is committed
    # matters more than the deadline
    _Rule(
        "partial_commit",
        PartialCommitError,
        "Part of the work is already committed: check what was applied before you run any of it again, because a rerun of the "
        "whole block applies that part a second time.",
        java=(_E + "TransactionException",),
        message=re.compile(r"committed part of its work before failing"),
    ),
    _Rule(
        "duplicate_key",
        DuplicateKeyError,
        "A unique index already holds this key: use another value, or update the record that has it "
        "(the message names the index and the record).",
        java=(_E + "DuplicatedKeyException",),
    ),
    _Rule(
        "record_not_found",
        RecordNotFoundError,
        _HINT_RECORD_NOT_FOUND,
        java=(_E + "RecordNotFoundException", _E + "VertexNotFoundException"),
    ),
    _Rule(
        "transaction_not_begun",
        TransactionError,
        "No transaction is open: wrap the write in `with db.transaction():`, or call "
        "`db.set_auto_transaction(True)` so that each statement commits on its own.",
        java=(_E + "TransactionException",),
        message=re.compile(r"Transaction not begun"),
    ),
    _Rule(
        "database_locked",
        DatabaseLockedError,
        "Another process has this database open: close it there first, because only one process can "
        "open a database at a time.",
        java=("com.arcadedb.utility.LockException",),
    ),
    _Rule(
        "database_exists",
        DatabaseExistsError,
        "A database already exists at this path: open it with `open_database(path)`, or choose another path.",
        java=(_E + "DatabaseOperationException",),
        message=re.compile(r"already exists"),
    ),
    _Rule(
        "database_not_found",
        DatabaseNotFoundError,
        "No database exists at this path: check the path, or create one with `create_database(path)`.",
        java=(_E + "DatabaseNotFoundException",),
    ),
    _Rule(
        "database_closed",
        DatabaseClosedError,
        _HINT_DATABASE_CLOSED,
        java=(_E + "DatabaseIsClosedException",),
    ),
    _Rule(
        "database_read_only",
        ArcadeDBError,
        "The database was opened with `read_only=True`: open it again without that argument to write.",
        java=(_E + "DatabaseIsReadOnlyException",),
    ),
    _Rule(
        "syntax_error",
        QuerySyntaxError,
        "The statement does not parse: fix the token that the message points at (line and column), and "
        "check the keywords against the ArcadeDB SQL or openCypher syntax.",
        java=(_E + "CommandSQLParsingException", _E + "CommandParsingException"),
    ),
    _Rule(
        "timeout",
        QueryTimeoutError,
        "The statement ran past its TIMEOUT clause: add a filter, a LIMIT or an index so that it "
        "finishes sooner, or raise the timeout.",
        java=(_E + "TimeoutException", _E + "PartialResultTimeoutException"),
    ),
    _Rule(
        "conflict",
        ConflictError,
        "Another transaction changed the same data first: run the whole transaction again, for example "
        "with `db.run_in_transaction(fn)`, which retries it.",
        java=(
            _E + "ConcurrentModificationException",
            _E + "LockTimeoutException",
            _E + "NeedRetryException",
        ),
    ),
    _Rule(
        "type_not_found",
        SchemaError,
        _HINT_TYPE_NOT_FOUND,
        java=(_E + "SchemaException",),
        message=re.compile(r"Type with name '[^']*' was not found"),
    ),
    _Rule(
        "already_exists",
        SchemaError,
        "A type, property or index with this name already exists: choose another name, or add "
        "`IF NOT EXISTS` to the CREATE statement of a type or an index.",
        java=(_E + "AlreadyExistsException", _E + "TypeAlreadyExistsException"),
    ),
    _Rule(
        "schema",
        SchemaError,
        _HINT_SCHEMA,
        java=(_E + "SchemaException",),
    ),
    _Rule(
        "type_mismatch",
        ValidationError,
        "A value does not fit the type of its property: send a value of the declared type "
        "(the message names the property and the type).",
        java=("java.lang.IllegalArgumentException", _E + "TypeMismatchException"),
        message=re.compile(r"Cannot convert type"),
    ),
    _Rule(
        "validation",
        ValidationError,
        "A property constraint (mandatory, not null, min, max, regexp or read-only) refused the record: "
        "change the value that the message names, or change the constraint in the schema.",
        java=(_E + "ValidationException",),
    ),
    _Rule(
        "query_not_idempotent",
        ArcadeDBError,
        "`db.query` runs reads only: send a statement that changes data with `db.command`.",
        java=(_E + "QueryNotIdempotentException",),
    ),
    _Rule(
        "missing_parameter",
        ArcadeDBError,
        "The statement has a parameter that was not passed: give a value for each `?` in order, or "
        "pass a dict that holds every `:name`.",
        java=(_E + "CommandParameterMissingException",),
    ),
    # Rows that match a text only come last: a statement that a syntax error echoes can hold any words, so a class decides first.
    _Rule(
        "record_not_found",
        RecordNotFoundError,
        _HINT_RECORD_NOT_FOUND,
        message=re.compile(r"\brecord not found\b"),
    ),
    _Rule(
        "database_closed",
        DatabaseClosedError,
        _HINT_DATABASE_CLOSED,
        message=re.compile(r"^Database is closed"),
    ),
    _Rule(
        "type_not_found",
        SchemaError,
        _HINT_TYPE_NOT_FOUND,
        message=re.compile(r"\bType '[^']*' does not exist|\bthe type does not exist"),
    ),
    _Rule(
        "schema",
        SchemaError,
        _HINT_SCHEMA,
        message=re.compile(r"Property '[^']*' not found on class|Index not found: "),
    ),
)

_RULE_OF_CODE = {}
for _rule in _RULES:
    _RULE_OF_CODE.setdefault(_rule.code, _rule)
del _rule

# A Java class name inside a text, for an error that has no Java chain (the Python layer words some errors like the JVM build:
# "Failed to save record: com.arcadedb.exception.TransactionException: Transaction not begun")
_CLASS_IN_TEXT = re.compile(
    r"\b((?:com|org|java)\.(?:[A-Za-z0-9_$]+\.)*[A-Za-z0-9_$]*(?:Exception|Error))\b"
)


def _partial_commit_error(block, conflict):
    """The ``PartialCommitError`` for a ``block`` that lost ``conflict`` after it had committed part of its work (#504).

    The engine's own retry loops raise a ``TransactionException`` with this wording for the same refusal (engine 26.12.1), so
    ``_RULES`` gives the two the same class and code. The caller raises the result while it handles ``conflict`` and without
    ``from``: the conflict stays visible as ``__context__``, and nothing that follows the cause chain to decide on a retry
    (``_retry_verdict``) finds it.
    """
    chain = _java_chain_of(conflict)
    if chain:
        conflicts = _RULE_OF_CODE["conflict"].java
        name, message = next(
            (link for link in chain if link[0] in conflicts), chain[-1]
        )
    else:
        name, message = type(conflict).__name__, str(conflict)
    text = (
        f"{block} committed part of its work before failing with {name.rsplit('.', 1)[-1]} ({message}). It is not retried, "
        "because a replay would apply the already committed part a second time: check what was applied before running it again"
    )
    error = PartialCommitError(text)
    # what a wrapper that re-raises this error classifies it by: the exception the engine raises for the same refusal
    error._chain = ((_E + "TransactionException", text),)
    return error


def _classify(text, chain=()):
    """``(rule, cause)`` for an error with this ``text`` and Java ``chain``: the first matching row, or ``(None, root cause)``."""
    names = [name for name, _ in chain] or _CLASS_IN_TEXT.findall(text)
    messages = [message for _, message in chain if message]
    messages.append(text)
    for rule in _RULES:
        hit = None
        if rule.java:
            hit = next((name for name in names if name in rule.java), None)
            if hit is None:
                continue
        if rule.message is not None and not any(
            rule.message.search(message) for message in messages
        ):
            continue
        return rule, hit or (names[-1] if names else None)
    return None, (names[-1] if names else None)
