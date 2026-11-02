"""Record wrappers for the native backend: ``Document``, ``Vertex`` and ``Edge`` as Python objects over SQL statements.

The JPype wrappers hold a Java record; here a record is its type, its RID (``None`` until saved) and its properties as a
dict, and every write is one DSL statement (``CREATE VERTEX .. CONTENT``, ``INSERT .. CONTENT``, ``UPDATE .. MERGE``,
``DELETE``, ``CREATE EDGE``). They subclass the JPype wrapper classes so ``isinstance`` checks keep their meaning.
"""

import hashlib
from collections.abc import Mapping
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

_SCALARS = frozenset((str, int, float, bool, type(None)))


def _canon(value):
    """``value`` as plain nested lists, tuples and scalars that compare and print the same for two equal values, whatever order a
    dict or a set was built in."""
    if type(value) in _SCALARS:
        return value
    if isinstance(value, Mapping):
        return (
            "m",
            sorted(((_canon(k), _canon(v)) for k, v in value.items()), key=repr),
        )
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return ("s", sorted((_canon(v) for v in value), key=repr))
    tolist = getattr(value, "tolist", None)  # a numpy array
    if tolist is not None:
        return ("a", str(getattr(value, "dtype", "")), _canon(tolist()))
    return value  # datetime, date, Decimal, RID and bytes print by value


def _fingerprint(props: Dict[str, Any]) -> bytes:
    """A fingerprint of the values a record holds. Called only when a record is about to be changed (``modify()``, the first ``set()`` of a
    saved record, ``save()``), never when one is read: the read path keeps the values the record already holds, and this runs on them
    later. Two reads of one stored record decode to equal values, so equal fingerprints mean an unchanged record.
    """
    return hashlib.blake2b(repr(_canon(props)).encode(), digest_size=16).digest()


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
    _DELETE = "DELETE FROM {rid}"

    def __init__(
        self, database, type_name, rid=None, props=None, mutable=False, token=None
    ):
        self._java_document = _java_object()
        self._database = database
        self._db = database
        self._type = type_name
        self._rid = rid or None
        self._props: Dict[str, Any] = dict(props or {})
        self._mutable = mutable or not rid
        # fingerprint of the values this record held as read (None: not taken yet). A read does not take it: modify(), the first set() of a
        # saved record or save() does, from the values then held, which are the read-time snapshot. save() refuses to overwrite a changed record
        self._token = token
        self._property_names_cache = None

    # -- reading

    def _check_open(self) -> None:
        database = self._database
        if database is not None and database._closed and self._rid:
            raise ArcadeDBError(
                "Database is closed: a record cannot be read after its database was closed. "
                "Read what you need before closing it."
            )

    def get(self, name: str, convert_types: bool = True) -> Any:
        self._check_open()
        return self._props.get(name)

    def get_raw(self, name: str) -> Any:
        return self.get(name)

    def has_property(self, name: str) -> bool:
        self._check_open()
        return name in self._props

    def get_property_names(self) -> List[str]:
        self._check_open()
        return list(self._props)

    def to_dict(self, convert_types: bool = True) -> Dict[str, Any]:
        self._check_open()
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
        if self._token is None and self._rid:
            self._token = _fingerprint(
                self._props
            )  # a saved record's snapshot, before the first change to it
        self._props[name] = value
        return self

    def save(self):
        db = self._db
        if self._rid is None:
            from .extras import veto_count

            vetoes = veto_count()
            row = db.command(
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

        The engine keeps no record version. The values this object held as read are fingerprinted (once, at modify() or here) and compared
        with the values stored now, in the caller's transaction; a stored record that differs raises ConcurrentModificationException and
        nothing is written. The fingerprint then moves to the stored values after this save, so the object's own save is not another
        transaction's change.

        Like every write (and like the JVM build), it needs an active transaction while auto-transaction is off, the engine default:
        outside one it raises ArcadeDBError ("Transaction not begun") before it reads or writes anything. This method opens no
        transaction of its own."""
        if not db.is_transaction_active() and not db._auto_transaction:
            raise ArcadeDBError(
                "Failed to save record: com.arcadedb.exception.TransactionException: Transaction not begun "
                "(wrap the write in `with db.transaction():` or call begin() first)"
            )
        token = self._token
        if token is None:
            token = self._token = _fingerprint(self._props)
        select_sql = (
            f"SELECT FROM {self._rid}"  # nosec B608 - _rid is the record's own id
        )
        stored = db.query("sql", select_sql).first()
        if stored is not None and _fingerprint(stored._props) != token:
            raise _concurrent_modification(self._rid, self._type)
        after = db.command(
            "sql",
            f"UPDATE {self._rid} MERGE :row RETURN AFTER",
            {"row": self._props},
        ).first()
        self._token = _fingerprint(after._props) if after is not None else None

    def _adopt(self, row) -> None:
        if row is None or not row.get_rid():
            raise ArcadeDBError(
                f"Failed to save a {self._type} record: the statement returned no record"
            )
        self._rid = row.get_rid()
        self._props = row.to_dict()
        self._token = None  # the stored values are the ones now held: fingerprinted when the first change or the next save needs it

    def delete(self):
        if self._rid is None:
            raise ArcadeDBError("Cannot delete a record that was never saved")
        self._db.command("sql", self._DELETE.format(rid=self._rid)).close()

    def modify(self):
        self._check_open()
        if self._rid and self._token is None:
            self._token = _fingerprint(
                self._props
            )  # the values held now are the ones read: cached here, used by save()
        return type(self)(
            self._db,
            self._type,
            self._rid,
            self._props,
            mutable=True,
            token=self._token,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} rid={self.get_rid()} type={self._type}>"


class Document(_Mixin, _BaseDocument):
    pass


class Vertex(_Mixin, _BaseVertex):
    _CREATE = "CREATE VERTEX {t} CONTENT :row"
    _DELETE = "DELETE FROM {rid}"

    def new_edge(self, label: str, target: "Vertex", **kwargs) -> "Edge":
        self._check_open()
        if self._rid is None or getattr(target, "_rid", None) is None:
            raise ArcadeDBError(
                "Both vertices of an edge must be saved before new_edge()"
            )
        sql = f"CREATE EDGE {_q(label)} FROM {self._rid} TO {target._rid}"
        if kwargs:
            sql += " CONTENT :row"
        row = self._db.command(
            "sql", sql, *(({"row": kwargs},) if kwargs else ())
        ).first()
        edge = Edge(self._db, label, mutable=False)
        edge._adopt(row)
        return edge

    def _edges(self, fn: str, labels) -> List["Edge"]:
        self._check_open()
        args = ", ".join("'" + label.replace("'", "\\'") + "'" for label in labels)
        edges_sql = f"SELECT expand({fn}({args})) FROM {self._rid}"  # nosec B608 - fn is a fixed function name, labels are escaped literals, _rid is the record's own id
        rows = self._db.query("sql", edges_sql)
        return [_wrap(self._db, row) for row in rows]

    def get_out_edges(self, *labels: str) -> List["Edge"]:
        return self._edges("outE", labels)

    def get_in_edges(self, *labels: str) -> List["Edge"]:
        return self._edges("inE", labels)

    def get_both_edges(self, *labels: str) -> List["Edge"]:
        return self._edges("bothE", labels)


class Edge(_Mixin, _BaseEdge):

    def _end(self, fn: str) -> "Vertex":
        self._check_open()
        end_sql = f"SELECT expand({fn}()) FROM {self._rid}"  # nosec B608 - fn is a fixed function name, _rid is the record's own id
        row = self._db.query("sql", end_sql).first()
        return _wrap(self._db, row)

    def get_in(self) -> Vertex:
        return self._end("inV")

    def get_out(self) -> Vertex:
        return self._end("outV")


_CLASS_OF = {"v": Vertex, "e": Edge}


def _wrap(database, row):
    """The wrapper for a result row that is a record (vertex, edge or document); None for a projection."""
    meta = row._meta
    if not meta:
        return None
    rid, type_name, kind = meta
    return _CLASS_OF.get(kind, Document)(database, type_name, rid, row._props)


def _wrap_dict(database, record: dict, kind: str = "d"):
    """The wrapper for a nested record the engine sent as a dict with ``@rid``, ``@type`` and ``@cat`` members."""
    props = dict(record)
    rid = props.pop("@rid", None)
    type_name = props.pop("@type", None)
    kind = props.pop("@cat", kind)
    return _CLASS_OF.get(kind, Document)(database, type_name, rid, props)
