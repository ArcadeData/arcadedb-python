"""Python <-> native JSON: query parameters out, typed rows in."""

import json
import os
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal

try:  # numpy is optional
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

TAG = "\u0001"
_TAG_ESCAPED = "\\u0001"  # TAG as it appears in reply text: NativeCodec.quote writes it as a \u escape
_EPOCH_NAIVE = datetime(1970, 1, 1)
_EPOCH_AWARE = datetime(1970, 1, 1, tzinfo=timezone.utc)

# A list of this many Python floats or more may cross in the float32 side buffer (see _exact_float_list). A shorter one is cheaper
# as JSON text, and a short IN list is the common use of a list parameter.
_FLOAT_LIST_MIN = 64


class _ExactFloats:
    """A list of Python floats that float32 holds exactly: ``f32`` is the side-buffer form, ``values`` the list it stands for."""

    __slots__ = ("f32", "values")

    def __init__(self, f32, values):
        self.f32 = f32
        self.values = values


def _exact_float_list(value):
    """``value`` as an :class:`_ExactFloats` when it is a list of at least ``_FLOAT_LIST_MIN`` Python floats (no int, bool, None,
    numpy scalar) that all survive the round trip double -> float32 -> double unchanged; None otherwise.

    The engine then reads the same ``List<Double>`` it read from the JSON text, value for value: the check is exact, never a
    tolerance. A list from a float32 model (``ndarray.tolist()``) passes; a list of doubles that float32 would round (0.1), NaN,
    or an infinity fails the check and stays JSON text, so no result changes. The text costs 340 ns a number
    (611 us for 1536), the buffer 60 us.
    """
    if (
        _np is None
        or len(value) < _FLOAT_LIST_MIN
        or set(map(type, value)) != _FLOAT_ONLY
    ):
        return None
    doubles = _np.array(value, dtype=_np.float64)
    with _np.errstate(
        over="ignore"
    ):  # a double beyond float32 becomes an infinity, which fails the check below
        f32 = doubles.astype(_np.float32)
    if (
        not _np.array_equal(f32.astype(_np.float64), doubles)
        or not _np.isfinite(f32).all()
    ):
        return None  # NaN and the infinities stay text: they are what the engine's JSON reader decides about, not this check
    return _ExactFloats(f32, value)


def _default(value, _bin=None):
    """json.dumps hook: the values JSON cannot carry, as tagged objects (see Json.java).

    ``_bin`` (binary mode): ``[arrays, floats_so_far]``; a float ndarray then goes to the side buffer as float32 and the JSON carries
    ``["B", offset, count]`` instead of the number list (no float formatting in Python, no text parsing in Java).
    """
    if isinstance(value, RID):
        return {TAG: ["R", value.text]}
    if isinstance(value, _ExactFloats):
        if _bin is None:
            return value.values
        off = _bin[1]
        _bin[0].append(value.f32)
        _bin[1] = off + value.f32.size
        return {TAG: ["BL", off, value.f32.size]}
    if isinstance(value, Decimal):
        return {TAG: ["n", str(value)]}
    if isinstance(value, datetime):
        utc = (
            value.astimezone(timezone.utc).replace(tzinfo=None)
            if value.utcoffset() is not None
            else value
        )
        return {TAG: ["T", utc.isoformat()]}
    if isinstance(value, date):
        return {TAG: ["D", value.isoformat()]}
    if isinstance(value, (set, frozenset)):
        return {TAG: ["s", list(value)]}
    if isinstance(value, (bytes, bytearray, memoryview)):
        import base64

        return {TAG: ["b", base64.b64encode(bytes(value)).decode("ascii")]}
    if _np is not None:
        if isinstance(value, _np.ndarray):
            if value.dtype.kind == "f":
                if value.dtype != _np.float32:
                    # a float64 (or float16, longdouble) array keeps its doubles, as on the JVM build and in row_default. The
                    # float32 side buffer and the "F" tag (a float[] in Java) would both round it to float32 (0.1 -> 0.10000000149011612).
                    # Only a lone positional ``?`` is a query vector sent as float32: encode_params narrows that one before it gets here.
                    return value.tolist()
                if _bin is not None:
                    arr = _np.ascontiguousarray(value.ravel(), dtype=_np.float32)
                    off = _bin[1]
                    _bin[0].append(arr)
                    _bin[1] = off + arr.size
                    return {TAG: ["B", off, arr.size]}
                return {TAG: ["F", value.tolist()]}
            return value.tolist()
        if isinstance(value, _np.bool_):
            return bool(value)
        if isinstance(value, _np.integer):
            return int(value)
        if isinstance(value, _np.floating):
            return float(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"cannot pass {type(value).__name__} as a parameter on the native backend"
    )


_PLAIN = frozenset((str, int, float, bool, type(None)))
_FLOAT_ONLY = frozenset((float,))
_dumps = json.dumps
# the C function that json.dumps writes a str with
_enc_str = json.encoder.encode_basestring_ascii
_fast = os.environ.get("ARCADEDB_NATIVE_FASTENC", "1") != "0"


def _prep(value, binary=False):
    """A value ready for json.dumps: a dict with a key that is not a str becomes a tagged ``[[k, v], ..]`` map (JSON would turn
    the key into text; the JPype path keeps it), and containers are walked for such dicts. Plain scalars pass untouched.

    ``binary``: a long list of floats that float32 holds exactly is returned as an :class:`_ExactFloats` for the side buffer.
    """
    t = type(value)
    if t in _PLAIN:
        return value
    if isinstance(value, Mapping):
        if all(type(k) is str for k in value):
            return {k: _prep(v, binary) for k, v in value.items()}
        return {
            TAG: [
                "m",
                [[_prep(k, binary), _prep(v, binary)] for k, v in value.items()],
            ]
        }
    if t is list or t is tuple:
        if len(value) > 8:
            if binary and t is list:
                exact = _exact_float_list(value)
                if exact is not None:
                    return exact
            if set(map(type, value)) <= _PLAIN:
                return value  # nothing inside to rewrite: skip the per-element walk (69 us for 1536 numbers)
        return [_prep(v, binary) for v in value]
    return value


def _query_vector(value):
    """A float ndarray among the positional parameters (``?``) is a query vector: the engine stores vectors as float32, so it is
    sent as float32 (the fast side buffer), as ``Database._convert_args`` does on the JVM build and as documented in "A query
    vector". A named parameter, ``CONTENT :row`` and a record's properties are values, not query vectors: a float64 array there
    keeps its doubles (see :func:`_default`).
    """
    if (
        _np is not None
        and isinstance(value, _np.ndarray)
        and value.dtype.kind == "f"
        and value.dtype != _np.float32
    ):
        return value.astype(_np.float32)
    return value


def encode_params(args, binary=False):
    """The ``params`` text for ``arcadedb_n_exec``: '' for none, else ``{"named":{..}}`` or ``{"pos":[..]}``.

    Same selection as the JPype path: a lone list/tuple is the positional list itself, a lone mapping (also alone in
    a list) is the named map, anything else is positional.
    """
    if not args:
        return ("", None, 0) if binary else ""
    values = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    plain = _PLAIN
    if _fast and len(values) == 1:
        # one int or str parameter (the point read, WHERE id = ?): the text is written directly, the bytes json.dumps({"pos": [v]}) gives,
        # without the set of types and the encoder call. Only the exact types: a bool, an int subclass or a float takes the general path
        v = values[0]
        t = type(v)
        if t is int:
            text = '{"pos": [%d]}' % v
            return (text, None, 0) if binary else text
        if t is str:
            text = '{"pos": [%s]}' % _enc_str(v)
            return (text, None, 0) if binary else text
    if _fast and set(map(type, values)) <= plain:
        # all plain scalars (the per-row INSERT of a load): the cached default encoder, no default hook, no per-call encoder object
        text = _dumps({"pos": values if type(values) is list else list(values)})
        return (text, None, 0) if binary else text
    if len(values) == 1 and isinstance(values[0], Mapping):
        named = values[0]
        if _fast:
            # every value a plain scalar or an RID (a named point read, the record id of a step or an edge end): no float side buffer
            # can arise, so the cached encoder writes the text, with an RID as the tagged object the hook would have returned. The
            # text is the same bytes; what is skipped is _prep, the hook closure and the per-call encoder object.
            out = {}
            for k, v in named.items():
                t = type(v)
                if t is RID:
                    v = {TAG: ["R", v.text]}
                elif t not in plain:
                    break
                out[k] = v
            else:
                text = _dumps({"named": out})
                return (text, None, 0) if binary else text
        if not all(type(v) in plain for v in named.values()):
            named = {k: _prep(v, binary) for k, v in named.items()}
        return _dump({"named": dict(named)}, binary)
    if not all(type(v) in plain for v in values):
        values = [_prep(_query_vector(v), binary) for v in values]
    return _dump({"pos": list(values)}, binary)


def encode_lookup(keys, values):
    """The JSON text ``arcadedb_n_lookup`` takes, ``{"keys": [..], "values": [..]}`` in the tagged encoding of query parameters
    (the text ``json.dumps(_prep(obj), default=_default)`` gives). Plain values are written by the cached encoder: for them
    ``_prep`` returns its argument and the hook is never called."""
    obj = {"keys": [str(k) for k in keys], "values": list(values)}
    if _fast and set(map(type, obj["values"])) <= _PLAIN:
        return _dumps(obj)
    return json.dumps(_prep(obj), default=_default)


def _dump(obj, binary):
    """JSON text; binary mode returns ``(text, float32 side buffer bytes or None, float count)``."""
    if not binary:
        return json.dumps(obj, default=_default)
    state = [[], 0]
    text = json.dumps(obj, default=lambda v: _default(v, state))
    arrays = state[0]
    if not arrays:
        return text, None, 0
    buf = arrays[0].tobytes() if len(arrays) == 1 else _np.concatenate(arrays).tobytes()
    return text, buf, state[1]


def _gb_default(value, state):
    """json.dumps hook for a GraphBatch op in binary mode: a one-dimensional non-empty float32 ndarray goes to the side buffer (the
    ``["B", offset, count]`` tag, a float[] in Java, the value the ``"F"`` text gives); every other value as :func:`_default`
    writes it, so a list, a float64 array or a 2-D array keeps the text it has always had.
    """
    if (
        _np is not None
        and isinstance(value, _np.ndarray)
        and value.dtype == _np.float32
        and value.ndim == 1
        and value.size
    ):
        return _default(value, state)
    return _default(value)


def encode_gb_args(args):
    """The ``args`` of ``arcadedb_n_gb_bin``: ``(text, float32 side buffer bytes or None, float count)``, text as :func:`_prep` and
    :func:`_default` write it for ``arcadedb_n_gb`` except that the float32 vectors of the rows and props are in the buffer.
    """
    state = [[], 0]
    text = json.dumps(_prep(args), default=lambda v: _gb_default(v, state))
    arrays = state[0]
    if not arrays:
        return text, None, 0
    buf = arrays[0].tobytes() if len(arrays) == 1 else _np.concatenate(arrays).tobytes()
    return text, buf, state[1]


_INT64_MAX = 2**63 - 1
_NS_DATETIME = None if _np is None else _np.dtype("datetime64[ns]")


def sample_array(values, timestamps=False):
    """The int64 or float64 array that stands for a column of ``append_samples`` in the address table of ``arcadedb_n_async_bin``, or
    None when the column keeps the JSON text it has always had. It is the array whose ``.tolist()`` the text path writes, value for
    value: an int, uint (uint64 only inside the int64 range) or bool array is int64, a float array narrower than long double is float64
    (a float32 widens exactly), and, for the timestamps only, a ``datetime64[ns]`` array without NaT is its int64 count (what ``.tolist()``
    gives for that unit). Anything else, a list, a subclass of ndarray, a 2-D or empty array, a float array as timestamps, other
    ``datetime64`` units (``int()`` refuses their datetimes on the text path), a ``datetime64[ns]`` of non-native byte order, returns None.
    """
    if (
        _np is None
        or type(values) is not _np.ndarray
        or values.ndim != 1
        or not values.size
    ):
        return None
    dtype = values.dtype
    kind = dtype.kind
    if kind in "ib":
        return _np.ascontiguousarray(values, dtype=_np.int64)
    if kind == "u":
        if dtype.itemsize == 8 and int(values.max()) > _INT64_MAX:
            return None
        return _np.ascontiguousarray(values, dtype=_np.int64)
    if kind == "f":
        if timestamps or dtype.itemsize > 8:
            return None
        return _np.ascontiguousarray(values, dtype=_np.float64)
    if kind == "M" and timestamps and dtype == _NS_DATETIME:
        counts = _np.ascontiguousarray(values.view(_np.int64))
        if (
            counts == _np.iinfo(_np.int64).min
        ).any():  # NaT: the text path raises on its None
            return None
        return counts
    return None


def row_default(value):
    """json.dumps hook for the rows of a bulk load: a NumPy float array that is not float32 is the list it would have been
    (a float64 array stored on an undeclared property keeps its doubles, as on the JPype path); the rest as :func:`_default`.
    """
    if (
        _np is not None
        and isinstance(value, _np.ndarray)
        and value.dtype.kind == "f"
        and value.dtype != _np.float32
    ):
        return value.tolist()
    return _default(value)


class RID:
    """A record id as the engine returns it in a row; passed back as a parameter it is a link, not a string.

    The JVM build hands out the engine's own ``RID`` in the same places, so this class offers what user code written
    against that one uses: ``str(rid)`` is the text (``"#10:5"``), two RIDs of one record are equal and hash alike,
    RIDs sort by bucket then position, and ``getBucketId()``, ``getPosition()``, ``toString()``, ``isValid()``,
    ``equals()`` and ``compareTo()`` answer like the Java methods. Unlike the Java RID, an RID here also equals its text.
    """

    __slots__ = ("text",)

    def __init__(self, text):
        # An RID of an RID is the same id: a hit of find_nearest(include_vector=True) already carries the row's RID and the record
        # wrappers wrap their rid again (save, delete, new_edge); the fast parameter path writes .text into the JSON text (#265)
        self.text = text.text if isinstance(text, RID) else text

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"RID({self.text!r})"

    def __eq__(self, other):
        return self.text == (other.text if isinstance(other, RID) else other)

    def __hash__(self):
        return hash(self.text)

    def _numbers(self):
        """(bucket id, position) from the text; (-1, -1) for a text that is not ``#bucket:position``."""
        bucket, sep, position = self.text.lstrip("#").partition(":")
        try:
            return (int(bucket), int(position)) if sep else (-1, -1)
        except ValueError:
            return (-1, -1)

    def __lt__(self, other):
        return (
            self._numbers() < other._numbers()
            if isinstance(other, RID)
            else NotImplemented
        )

    def __le__(self, other):
        return (
            self._numbers() <= other._numbers()
            if isinstance(other, RID)
            else NotImplemented
        )

    def __gt__(self, other):
        return (
            self._numbers() > other._numbers()
            if isinstance(other, RID)
            else NotImplemented
        )

    def __ge__(self, other):
        return (
            self._numbers() >= other._numbers()
            if isinstance(other, RID)
            else NotImplemented
        )

    # the Java RID's methods, by their Java names (the JVM build's RID is that object)
    def getBucketId(self):
        return self._numbers()[0]

    def getPosition(self):
        return self._numbers()[1]

    def toString(self):
        return self.text

    def isValid(self):
        bucket, position = self._numbers()
        return bucket > -1 and position > -1

    def equals(self, other):
        return isinstance(other, RID) and self.text == other.text

    def compareTo(self, other):
        mine, theirs = self._numbers(), other._numbers()
        return (mine > theirs) - (mine < theirs)


def _restore_float(text):
    return float(text)


def _restore_map(pairs):
    return {k: v for k, v in pairs}


_SIMPLE = {
    "n": Decimal,
    "i": int,
    "D": date.fromisoformat,
    "T": datetime.fromisoformat,
    "Z": datetime.fromisoformat,
    "s": set,
    "f": _restore_float,
    "m": _restore_map,
    "R": RID,
}


def _hook(obj):
    tagged = obj.get(TAG)
    # a column may be called like the tag: only a lone [kind, payload] pair of a known kind is a tagged value
    if (
        tagged is None
        or len(obj) != 1
        or type(tagged) is not list
        or len(tagged) != 2
        or tagged[0] not in _SIMPLE
    ):
        return obj
    kind, payload = tagged
    return _SIMPLE[kind](payload)


def loads(text: str, floats=None):
    """Decode a reply, restoring tagged values. ``floats``: the float32 tail of a binary reply (``["B", offset, count]`` values)."""
    # A tagged value always puts the escaped tag in the text (NativeCodec.quote escapes every character below 0x20). Without it the reply
    # holds none and the per-object hook, a Python call for every decoded object, is skipped (#151). A string or a key that merely
    # contains U+0001 takes the hook path as before.
    if _TAG_ESCAPED not in text:
        return json.loads(text)
    if floats is None:
        return json.loads(text, object_hook=_hook)

    def hook(obj):
        tagged = obj.get(TAG)
        if (
            tagged is not None
            and len(obj) == 1
            and type(tagged) is list
            and len(tagged) == 3
            and tagged[0] == "B"
        ):
            return floats[
                tagged[1] : tagged[1] + tagged[2]
            ].tolist()  # float32 widens to the same double as the JPype path
        return _hook(obj)

    return json.loads(text, object_hook=hook)


def loads_tagged(text: str, floats=None):
    """:func:`loads` that also says whether a tagged value was restored: ``(value, restored)``.

    ``restored`` is False for a reply without the escaped tag, and for one whose only tags are ``["B", offset, count]``
    (they decode to plain float lists). Then every value is a ``json.loads`` type and :func:`json_native` returns it
    unchanged, which is what lets ``to_json_list`` skip its walk. The decoded value is the one :func:`loads` returns.
    """
    if _TAG_ESCAPED not in text:
        return json.loads(text), False
    restored = False

    def hook(obj):
        nonlocal restored
        tagged = obj.get(TAG)
        if tagged is None or len(obj) != 1 or type(tagged) is not list:
            return obj
        size = len(tagged)
        if size == 3 and floats is not None and tagged[0] == "B":
            return floats[tagged[1] : tagged[1] + tagged[2]].tolist()
        if size == 2 and tagged[0] in _SIMPLE:
            restored = True
            return _SIMPLE[tagged[0]](tagged[1])
        return obj

    value = json.loads(text, object_hook=hook)
    return value, restored


# --- JSON-native form of a row (ResultSet.to_json_list): DATE/DATETIME as epoch millis, DECIMAL as float, an RID as its text
# (``"#10:5"``), sets as lists, and a map key that is not a string as the string the JVM build's JSON object writes.


def _epoch_millis(delta):
    """Whole milliseconds of a timedelta since the epoch, in integer arithmetic and floored toward negative infinity: the
    ``Instant.toEpochMilli`` of the JVM build. ``days`` floors and ``seconds`` and ``microseconds`` are never negative, so a
    time before 1970 with microseconds (``.999999``) is not rounded toward zero, and no float product can fall just under an
    integer (``total_seconds() * 1000`` did both).
    """
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def _json_key(key):
    """The text of a map key that is not a string, as the JVM build's JSON object has it (``String.valueOf``)."""
    if type(key) is str:
        return key
    if isinstance(key, RID):
        return key.text
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    return str(key)


def json_native(value):
    if isinstance(value, RID):
        return value.text
    if isinstance(value, datetime):
        return _epoch_millis(
            value - _EPOCH_AWARE if value.tzinfo is not None else value - _EPOCH_NAIVE
        )
    if isinstance(value, date):
        return _epoch_millis(
            datetime.combine(value, datetime.min.time()) - _EPOCH_NAIVE
        )
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (set, frozenset)):
        return [json_native(v) for v in value]
    if isinstance(value, list):
        return [json_native(v) for v in value]
    if isinstance(value, dict):
        return {
            (k if type(k) is str else _json_key(k)): json_native(v)
            for k, v in value.items()
        }
    return value
