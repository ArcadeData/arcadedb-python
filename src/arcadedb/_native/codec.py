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
        if not all(type(v) in plain for v in named.values()):
            named = {k: _prep(v, binary) for k, v in named.items()}
        return _dump({"named": dict(named)}, binary)
    if not all(type(v) in plain for v in values):
        values = [_prep(v, binary) for v in values]
    return _dump({"pos": list(values)}, binary)


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
        self.text = text

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


# --- JSON-native form of a row (ResultSet.to_json_list): DATE/DATETIME as epoch millis, DECIMAL as float.


def json_native(value):
    if isinstance(value, datetime):
        return (
            int((value.replace(tzinfo=None) - _EPOCH_NAIVE).total_seconds() * 1000)
            if value.tzinfo is None
            else int(value.timestamp() * 1000)
        )
    if isinstance(value, date):
        return int(
            (
                datetime.combine(value, datetime.min.time()) - _EPOCH_NAIVE
            ).total_seconds()
            * 1000
        )
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (set, frozenset)):
        return [json_native(v) for v in value]
    if isinstance(value, list):
        return [json_native(v) for v in value]
    if isinstance(value, dict):
        return {k: json_native(v) for k, v in value.items()}
    return value
