"""Record wrappers for the native backend: ``Document``, ``Vertex`` and ``Edge`` as Python objects over SQL statements.

The JPype wrappers hold a Java record; here a record is its type, its RID (``None`` until saved) and its properties as a
dict, and every write is one DSL statement (``CREATE VERTEX .. CONTENT``, ``INSERT .. CONTENT``, ``UPDATE .. MERGE``,
``DELETE``, ``CREATE EDGE``). They subclass the JPype wrapper classes so ``isinstance`` checks keep their meaning.
"""

import copy
import math
from array import array
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List

from ..exceptions import ArcadeDBError
from ..graph import Document as _BaseDocument
from ..graph import Edge as _BaseEdge
from ..graph import Vertex as _BaseVertex
from . import needs_jvm
from .codec import RID

_JAVA_OBJECT = None


def _java_object():
    """A stand-in for the Java object of a record (any use of it says so); the class is looked up on first use, as database.py imports this module."""
    global _JAVA_OBJECT
    if _JAVA_OBJECT is None:
        from .database import _JavaObject as _JAVA_OBJECT
    return _JAVA_OBJECT()


_UNSAVED = "#-1:-1"

_MISSING = object()
_IMMUTABLE = frozenset(
    (str, int, float, bool, type(None), bytes, date, datetime, Decimal, RID)
)
_PLAIN = frozenset((str, int, bool, type(None)))
_FLOAT_ONLY = frozenset((float,))


def _snap(value):
    """A copy of ``value`` that nothing the caller holds can change, in the form a read of the stored value gives: a list for a list, a tuple and
    a float or int array (``tolist()`` widens float32 to the same double a read gives), a Python scalar for a NumPy scalar, a set for a set.
    ``set()`` writes into the record's own dict and a caller can change a nested list in place, so the values as read are copied, not kept.
    """
    t = type(value)
    if t in _IMMUTABLE:
        return value
    if t is list or t is tuple:
        if set(map(type, value)) <= _IMMUTABLE:
            return list(value)
        return [_snap(item) for item in value]
    if t is dict:
        if set(map(type, value.values())) <= _IMMUTABLE:
            return dict(value)
        return {key: _snap(item) for key, item in value.items()}
    tolist = getattr(value, "tolist", None)  # a numpy array or scalar
    if tolist is not None:
        listed = tolist()
        return listed if type(listed) is not list else _snap(listed)
    if isinstance(value, Mapping):
        return {key: _snap(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snap(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {_snap(item) for item in value}
    return copy.deepcopy(value)


def _same_float(a: float, b: float) -> bool:
    if a == b:
        return a != 0.0 or math.copysign(1.0, a) == math.copysign(
            1.0, b
        )  # 0.0 and -0.0 are two values
    return a != a and b != b  # NaN is the value NaN, equal to itself


def _same_after_commit(a, b) -> bool:
    """Two values of different types are one value in one case: a read inside the transaction that wrote a set, or bytes, gives the object just
    written, and a read after the commit gives what the engine stored, a list (a set's elements, bytes as signed ints). The stored row is
    ``a`` and the baseline ``b``, so a list against a set or bytes."""
    if type(a) is not list:
        return False
    if type(b) is set:
        return (
            len(a) == len(b) and set(a) == b and set(map(type, a)) == set(map(type, b))
        )
    if type(b) is bytes:
        return a == [x - 256 if x > 127 else x for x in b]
    return False


def _same(a, b) -> bool:
    """Whether the stored value ``a`` and the baseline ``b`` are one value: the same type (``1``, ``1.0`` and ``True`` are three values) and equal, a NaN equal to a NaN, ``0.0`` not
    equal to ``-0.0``, a Decimal equal only at the same scale. Both are decoded values (a read never gives an array), so a list is compared as a
    list; a float list, a vector, is compared in one call (its bytes when a zero is among the values).
    """
    t = type(a)
    if t is not type(b):
        return _same_after_commit(a, b)
    if t is float:
        return _same_float(a, b)
    if t is list:
        if len(a) != len(b):
            return False
        kinds = set(map(type, a))
        if kinds != set(map(type, b)):
            return False
        if kinds == _FLOAT_ONLY:
            if (
                a == b
            ):  # in C; equal floats of one type are one value unless a zero is among them (0.0 == -0.0)
                return (
                    0.0 not in a or array("d", a).tobytes() == array("d", b).tobytes()
                )
            return all(
                map(_same_float, a, b)
            )  # a NaN is not equal to itself in the list comparison
        if len(kinds) <= 1 and kinds <= _PLAIN:
            return a == b
        return all(map(_same, a, b))
    if t is dict:
        if len(a) != len(b):
            return False
        for key, item in a.items():
            other = b.get(key, _MISSING)
            if other is _MISSING or not _same(item, other):
                return False
        return True
    if t is set:
        return a == b and set(map(type, a)) == set(map(type, b))
    if t is Decimal:
        return (a == b and a.as_tuple() == b.as_tuple()) or (a.is_nan() and b.is_nan())
    if t is datetime:
        return a == b and a.tzinfo == b.tzinfo
    return bool(a == b)


def _exact_value(value) -> bool:
    """Whether the engine stores this (snapped) value so that a read gives back the same value, whatever the property's declared type allows.
    A kind that a type can round, truncate or turn into another kind is not on the list: a float the float32 of a FLOAT or ARRAY_OF_FLOATS
    property would round (a float is exact only when float32 holds it), a datetime (milliseconds, or seconds, by the property's precision;
    a time zone is dropped), a set (read back as a list), bytes (read back as a list of ints), a RID.
    """
    t = type(value)
    if t is str or t is bool or value is None:
        return True
    if t is int:
        return -9223372036854775808 <= value <= 9223372036854775807
    if t is float:
        return (
            array("f", (value,))[0] == value
        )  # false for NaN, and for a double float32 rounds
    if t is date:
        return True
    if t is Decimal:
        return value.is_finite()
    if t is list:
        kinds = set(map(type, value))
        if kinds == _FLOAT_ONLY:
            return array("f", value).tolist() == value
        return all(map(_exact_value, value))
    if t is dict:
        return all(type(key) is str for key in value) and all(
            map(_exact_value, value.values())
        )
    return False


def _stored_as_sent(new, old) -> bool:
    """Whether a changed value (``new``, snapped) is stored exactly as sent, so that the values just written can be the next baseline without a
    read. The engine converts a value to the declared type of its property (an int sent to a DOUBLE property reads back as 3.0, a str to a
    DATE one as a date), and the record object does not know the declaration. What it does know is what a read gave for this property before:
    a value of that same type is one the property holds as it is, so only a value of the type this property was last read as, with a kind that
    is exact (``_exact_value``), counts. A new property, a ``None`` before and a list that changes its element kinds do not: they are read
    back."""
    if type(new) is not type(old) or old is None:
        return False
    if type(new) is list:
        if not new:
            return True
        if not old or set(map(type, new)) != set(map(type, old)):
            return False
    return _exact_value(new)


def _concurrent_modification(rid: str, type_name: str) -> ArcadeDBError:
    """The error the engine raises when a record changed after it was read, in the form of a failed native reply (so that
    ``run_in_transaction`` retries it, as it retries the JVM build's)."""
    from .database import _error

    cls = "com.arcadedb.exception.ConcurrentModificationException"
    message = (
        f"Record {rid} was modified by a concurrent transaction after it was read. "
        "Please retry the operation"
    )
    return _error(
        f"Failed to save a {type_name} record",
        {
            "ok": False,
            "error": message,
            "cls": cls,
            "chain": [[cls, message]],
            "retry": "need_retry",
            "retry_ms": 0,
        },
    )


def _q(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


class _Mixin:
    """What the three native wrappers share (state in ``_db``, ``_type``, ``_rid``, ``_props``, ``_mutable``)."""

    _CREATE = "INSERT INTO {t} CONTENT :row"
    # the record id goes over as a parameter (a RID, passed back it is a link), never in the text: the engine caches a parsed statement by its
    # text and keeps a record id in a FROM target as part of it, so a pasted id would make one cache entry per record
    _SELECT = "SELECT FROM :rid"
    _UPDATE = "UPDATE :rid MERGE :row RETURN AFTER"
    _UPDATE_SENT = "UPDATE :rid MERGE :row"  # no read-back: used when every changed value is stored exactly as sent (_stored_as_sent)
    _DELETE = "DELETE FROM :rid"

    def __init__(
        self,
        database,
        type_name,
        rid=None,
        props=None,
        mutable=False,
        own=False,
        base=None,
    ):
        self._java_document = _java_object()
        self._database = database
        self._db = database
        self._type = type_name
        self._rid = rid or None
        # own: the caller built props for this record and keeps no other reference (the row of a lookup), so the dict is taken as it is
        self._props: Dict[str, Any] = (
            props if own and props is not None else dict(props or {})
        )
        self._mutable = mutable or not rid
        # a copy of the values this record held as read (None: not taken yet). A read does not take it: modify(), the first set() of a saved
        # record or save() does, from the values then held, which are the read-time values. save() refuses to overwrite a changed record.
        # The dict is never changed in place (a save makes a new one), so the objects modify() hands it to can share it
        self._base = base
        self._property_names_cache = None
        # names of properties a search hit was built without (see ``_wrap_partial``); None for every other record
        self._lazy = None

    # -- reading

    def _fill(self) -> None:
        """Load the properties a search hit was built without, in one lookup by RID (engine order); nothing to do for any other record."""
        if self._lazy:
            self._props = dict(self._db.lookup_by_rid(self._rid)._props)
            self._lazy = None

    def _check_open(self) -> None:
        database = self._database
        if database is not None and database._closed and self._rid:
            raise ArcadeDBError(
                "Database is closed: a record cannot be read after its database was closed. "
                "Read what you need before closing it."
            )

    def get(self, name: str, convert_types: bool = True) -> Any:
        self._check_open()
        if self._lazy and name in self._lazy:
            self._fill()
        return self._props.get(name)

    def get_raw(self, name: str) -> Any:
        return self.get(name)

    def has_property(self, name: str) -> bool:
        self._check_open()
        if self._lazy and name in self._lazy:
            self._fill()
        return name in self._props

    def get_property_names(self) -> List[str]:
        self._check_open()
        self._fill()
        return list(self._props)

    def to_dict(self, convert_types: bool = True) -> Dict[str, Any]:
        self._check_open()
        self._fill()
        return dict(self._props)

    def get_rid(self) -> str:
        return self._rid or _UNSAVED

    def get_identity(self):
        # a row can already carry @rid as an RID (vector-search rows of some engine builds): wrapping it again made str() fail
        rid = self.get_rid()
        return rid if isinstance(rid, RID) else RID(rid)

    def get_type_name(self) -> str:
        self._check_open()
        return self._type

    def get_java_document(self):
        needs_jvm("get_java_document")

    # -- writing

    def set(self, name: str, value: Any):
        if not self._mutable:
            raise AttributeError(
                f"{type(self).__name__} is immutable. Call .modify() first to get a mutable version."
            )
        if self._base is None and self._rid:
            self._base = _snap(
                self._props
            )  # a saved record's values as read, before the first change to it
        self._props[name] = value
        return self

    def save(self):
        """Create the record, or write the changed properties of a saved one.

        A saved record is written back only if the stored record still has the values this object was read with. Otherwise it raises
        ArcadeDBError ("ConcurrentModificationException" in the message, retryable by ``run_in_transaction``) and writes nothing: read the
        record again and repeat the change. The check compares values (the engine keeps no version), so a change that was changed back is
        not seen. Saving an existing record needs an active transaction, like every write; outside one it raises ArcadeDBError
        ("Transaction not begun"). Creating a new record is not checked.
        """
        self._fill()
        db = self._db
        if self._rid is None:
            from .extras import veto_count

            vetoes = veto_count()
            row = db._command_first_row(
                "sql", self._CREATE.format(t=_q(self._type)), {"row": self._props}
            ).first()
            if row is None and veto_count() != vetoes:
                return self  # a before_create callback vetoed it: nothing is written, no error (as on the JPype backend)
            self._adopt(row)
        else:
            self._save_update(db)
        return self

    def _save_update(self, db) -> None:
        """UPDATE .. MERGE of a saved record, refused when the stored record is no longer the one this object was read from.

        The engine keeps no record version. A copy of the values this object held as read (taken once, at modify() or here) is compared by
        value with the values stored now, in the caller's transaction; a stored record that differs raises ConcurrentModificationException and
        nothing is written. The copy then becomes the values this save wrote, so the object's own save is not another transaction's change:
        when every changed value is one the engine stores exactly as sent (``_stored_as_sent``) the statement returns no row and the values
        sent are the new copy; otherwise the statement returns the stored row (RETURN AFTER) and that row is.

        Like every write (and like the JVM build), it needs an active transaction while auto-transaction is off, the engine default:
        outside one it raises ArcadeDBError ("Transaction not begun") before it reads or writes anything. This method opens no
        transaction of its own. With auto-transaction on the check cannot refuse, so the state is not asked for (one crossing less).
        The engine's own refusal cannot stand in for the check with it off: the engine runs the before-update callbacks and the
        validation before it looks for a transaction, and an UPDATE of a record that is gone raises nothing.
        """
        if not db._auto_transaction and not db.is_transaction_active():
            raise ArcadeDBError(
                "Failed to save record: com.arcadedb.exception.TransactionException: Transaction not begun "
                "(wrap the write in `with db.transaction():` or call begin() first)"
            )
        base = self._base
        if base is None:
            base = self._base = _snap(self._props)
        stored = db.query("sql", self._SELECT, {"rid": RID(self._rid)}).first()
        if stored is not None and not _same(stored._props, base):
            raise _concurrent_modification(self._rid, self._type)
        sent = dict(base)
        exact = True
        for name, value in self._props.items():
            old = base.get(name, _MISSING)
            if old is not _MISSING and _same(value, old):
                continue  # as read: the copy of it stays
            value = _snap(value)
            if old is not _MISSING and _same(value, old):
                continue  # a tuple or an array of what was read
            sent[name] = value
            if exact and (old is _MISSING or not _stored_as_sent(value, old)):
                exact = False
        from .extras import veto_count

        vetoes = veto_count()
        params = {"rid": RID(self._rid), "row": self._props}
        if exact:
            row = db._command_first_row("sql", self._UPDATE_SENT, params).first()
            if row is not None and row.get("count") and veto_count() == vetoes:
                self._base = sent
        else:
            after = db._command_first_row("sql", self._UPDATE, params).first()
            if (
                after is not None
            ):  # a veto or a record that is gone gives no row: nothing was written, the copy stays
                self._base = after._props

    def _adopt(self, row) -> None:
        if row is None or not row.get_rid():
            raise ArcadeDBError(
                f"Failed to save a {self._type} record: the statement returned no record"
            )
        self._rid = row.get_rid()
        self._props = row.to_dict()
        self._base = None  # the stored values are the ones now held: copied when the first change or the next save needs it

    def delete(self):
        if self._rid is None:
            raise ArcadeDBError("Cannot delete a record that was never saved")
        self._db.command("sql", self._DELETE, {"rid": RID(self._rid)}).close()

    def modify(self):
        self._check_open()
        self._fill()
        if self._rid and self._base is None:
            self._base = _snap(
                self._props
            )  # the values held now are the ones read: copied here, used by save()
        return type(self)(
            self._db,
            self._type,
            self._rid,
            self._props,
            mutable=True,
            base=self._base,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} rid={self.get_rid()} type={self._type}>"


class Document(_Mixin, _BaseDocument):
    pass


class Vertex(_Mixin, _BaseVertex):
    _CREATE = "CREATE VERTEX {t} CONTENT :row"

    def new_edge(self, label: str, target: "Vertex", **kwargs) -> "Edge":
        self._check_open()
        if self._rid is None or getattr(target, "_rid", None) is None:
            raise ArcadeDBError(
                "Both vertices of an edge must be saved before new_edge()"
            )
        sql = f"CREATE EDGE {_q(label)} FROM :src TO :dst"
        params = {"src": RID(self._rid), "dst": RID(target._rid)}
        if kwargs:
            sql += " CONTENT :row"
            params["row"] = kwargs
        row = self._db._command_first_row("sql", sql, params).first()
        edge = Edge(self._db, label, mutable=False)
        edge._adopt(row)
        return edge

    def _edges(self, fn: str, labels) -> List["Edge"]:
        self._check_open()
        edges_sql = _edges_sql(fn, labels)
        rows = self._db.query("sql", edges_sql, {"rid": RID(self._rid)})
        database = self._db
        return [
            _wrap_meta(database, props, meta) for props, meta in rows._drain_pairs()
        ]

    def get_out_edges(self, *labels: str) -> List["Edge"]:
        return self._edges("outE", labels)

    def get_in_edges(self, *labels: str) -> List["Edge"]:
        return self._edges("inE", labels)

    def get_both_edges(self, *labels: str) -> List["Edge"]:
        return self._edges("bothE", labels)


class Edge(_Mixin, _BaseEdge):

    def _end(self, fn: str) -> "Vertex":
        self._check_open()
        row = self._db.query("sql", _END_SQL[fn], {"rid": RID(self._rid)}).first()
        return _wrap(self._db, row, True)

    def get_in(self) -> Vertex:
        return self._end("inV")

    def get_out(self) -> Vertex:
        return self._end("outV")


_CLASS_OF = {"v": Vertex, "e": Edge}

# The statement of a step or an edge end is built once per function and label list, not once per call: the same text object goes to the
# engine every time (it keys its statement cache by text). The label list is bounded so a program that builds labels cannot grow it.
_END_SQL: Dict[str, str] = {
    fn: f"SELECT expand({fn}()) FROM :rid"  # nosec B608 - fn is a fixed function name, the record id is a parameter
    for fn in ("inV", "outV")
}
_EDGES_SQL: Dict[Any, str] = {}
_EDGES_SQL_MAX = 512


def _edges_sql(fn: str, labels) -> str:
    key = (fn, labels)
    try:
        sql = _EDGES_SQL.get(key)
    except TypeError:
        sql = None  # a label that cannot be hashed: building the text below fails as it always did
    if sql is None:
        args = ", ".join("'" + label.replace("'", "\\'") + "'" for label in labels)
        sql = f"SELECT expand({fn}({args})) FROM :rid"  # nosec B608 - fn is a fixed function name, labels are escaped literals, the record id is a parameter
        if len(_EDGES_SQL) < _EDGES_SQL_MAX:
            _EDGES_SQL[key] = sql
    return sql


def _wrap(database, row, own=False):
    """The wrapper for a result row that is a record (vertex, edge or document); None for a projection.

    ``own``: nobody else holds the row, so the record takes its property dict as it is instead of a copy.
    """
    meta = row._meta
    if not meta:
        return None
    rid, type_name, kind = meta
    return _CLASS_OF.get(kind, Document)(database, type_name, rid, row._props, own=own)


def _wrap_meta(database, props, meta):
    """:func:`_wrap` for the ``(props, meta)`` pair of a row the caller drained itself (nothing else holds the dict)."""
    if not meta:
        return None
    rid, type_name, kind = meta
    return _CLASS_OF.get(kind, Document)(database, type_name, rid, props, own=True)


def _wrap_dict(database, record: dict, kind: str = "d"):
    """The wrapper for a nested record the engine sent as a dict with ``@rid``, ``@type`` and ``@cat`` members."""
    props = dict(record)
    rid = props.pop("@rid", None)
    type_name = props.pop("@type", None)
    kind = props.pop("@cat", kind)
    return _CLASS_OF.get(kind, Document)(database, type_name, rid, props)


def _wrap_partial(database, rid, type_name, kind, props, omitted):
    """The wrapper for a search hit built without the properties named in ``omitted`` (the indexed vector: not shipped with every hit).

    Reading one of them, or anything that needs the whole record (``to_dict``, ``get_property_names``, ``modify``, ``save``), loads the
    record once by RID; ``get`` and ``has_property`` of any other property are answered from what was shipped.
    """
    cls = {"v": Vertex, "e": Edge}.get(kind, Document)
    record = cls(database, type_name, rid, props)
    record._lazy = frozenset(omitted)
    return record
