"""A TypedRows batch with no tagged value is decoded without a ``json`` object_hook and without reading its side array (#315).

``ResultSet.to_list()`` and ``iter_dicts()`` read rows through the bridge's ``TypedRows``: a batch is
``{String json, Object[] side, Boolean drained}``. Java writes a value JSON cannot carry exactly as
``{"\\u0001": [kind, payload]}`` and Python restores it with an ``object_hook``, which reads the side
array. A batch of plain values (strings, integers, finite floats, booleans, nulls, lists, maps) holds no
tag, so the hook and the read of the side array are two JPype operations and a Python call per object
for nothing. The decoder now looks for the tag escape in the batch text first, as ``_loads_typed_row``
already did for one row.

Two things are pinned here, and neither may cost the other:

* the guard fires: a batch without the tag escape is parsed by plain ``json.loads`` and ``batch[1]`` is
  not read (the hook factory and the ``object_hook`` argument are counted);
* the guard never fires on a tagged batch: every shape the bridge tags keeps the hook, and the rows are
  the ones the always-hooked decoder gives, which is what ``to_list()`` did before.

Only the JVM build has this reader (``NATIVE_NOT_APPLICABLE`` in ``conftest.py``).
"""

import contextlib
import json
import types

import jpype
import pytest

from arcadedb import results

# the wide table with every property type (fixture ``db``), the row-by-row reference and the in-memory result sets
from tests.test_typed_rows import _in_memory, db, exact, reference  # noqa: F401

ESCAPE = "\\u0001"
BATCH = 1000  # results._TYPED_BATCH_ROWS: the rows of one to_list() crossing


class _Batch:
    """A TypedRows batch that logs which of its three elements are read (0 json, 1 side, 2 drained)."""

    def __init__(self, target, reads):
        self._target = target
        self._reads = reads

    def __getitem__(self, index):
        self._reads.append(index)
        return self._target[index]


class _Bridge:
    """The bridge class TypedRows, with every batch it returns wrapped in a _Batch."""

    def __init__(self, real, reads):
        self._real = real
        self._reads = reads

    def nextRows(self, java_result_set, size):
        return _Batch(self._real.nextRows(java_result_set, size), self._reads)


class _Json:
    """The ``json`` module as results.py sees it, logging the keyword arguments of every ``loads``."""

    def __init__(self, calls):
        self._calls = calls

    def loads(self, text, **kwargs):
        self._calls.append(kwargs)
        return json.loads(text, **kwargs)

    def __getattr__(self, name):
        return getattr(json, name)


@contextlib.contextmanager
def watching():
    """What the decoder of results.py does while the block runs.

    ``loads``: the keyword arguments of every json.loads call, ``hooks``: the side arrays the hook factory
    was built with (one entry per batch decoded with a hook), ``reads``: the batch elements read.
    Nothing is recorded outside the block, so the reference rows are computed first.
    """
    seen = types.SimpleNamespace(loads=[], hooks=[], reads=[])
    real_hook = results._typed_hook

    def counting_hook(side):
        seen.hooks.append(side)
        return real_hook(side)

    bridge = _Bridge(results._bridge_class("TypedRows"), seen.reads)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(results, "_typed_hook", counting_hook)
        patch.setattr(results, "json", _Json(seen.loads))
        patch.setitem(results._BRIDGE_CLASSES, "TypedRows", bridge)
        yield seen


def _hooked(seen):
    """How many json.loads calls got an object_hook."""
    return sum(1 for kwargs in seen.loads if kwargs.get("object_hook") is not None)


def _texts(java_result_set, size):
    """The JSON text of every batch TypedRows writes for the result set."""
    typed_rows = jpype.JClass("com.arcadedb.python.TypedRows")
    texts = []
    while True:
        batch = typed_rows.nextRows(java_result_set, size)
        texts.append(str(batch[0]))
        if batch[2]:
            return texts


def _decoded_with_the_hook_always(java_result_set, size):
    """The rows decoded as to_list() did before #315: an object_hook on every batch, tagged or not."""
    typed_rows = jpype.JClass("com.arcadedb.python.TypedRows")
    rows = []
    while True:
        batch = typed_rows.nextRows(java_result_set, size)
        rows.extend(
            json.loads(str(batch[0]), object_hook=results._typed_hook(batch[1]))
        )
        if batch[2]:
            return rows


def _all(rows):
    return [exact(row) for row in rows]


def _java(name):
    return jpype.JClass(name)


def _java_list(*items):
    java_list = _java("java.util.ArrayList")()
    for item in items:
        java_list.add(item)
    return java_list


def _java_map(**entries):
    java_map = _java("java.util.LinkedHashMap")()
    for key, value in entries.items():
        java_map.put(key, value)
    return java_map


def _java_set(*items):
    java_set = _java("java.util.LinkedHashSet")()
    for item in items:
        java_set.add(item)
    return java_set


def _int_key_map():
    java_map = _java("java.util.HashMap")()
    java_map.put(jpype.JInt(1), "one")
    return java_map


def _tag_key_map():
    java_map = _java("java.util.HashMap")()
    java_map.put("\u0001", "tag")
    return java_map


# Every kind of value the bridge tags: the decoder must keep the hook for a batch that holds one. The
# lambdas build the Java objects when the test runs, as the JVM is not started when this module loads.
TAGGED = {
    "decimal": lambda: _java("java.math.BigDecimal")("1.25"),
    "decimal_exponent": lambda: _java("java.math.BigDecimal")("1E+3"),
    "big_integer": lambda: _java("java.math.BigInteger")(
        "123456789012345678901234567890"
    ),
    "date": lambda: _java("java.time.LocalDate").of(2020, 1, 2),
    "date_year_one": lambda: _java("java.time.LocalDate").of(1, 1, 1),
    "datetime": lambda: _java("java.time.LocalDateTime").of(
        2021, 2, 3, 4, 5, 6, 789123000
    ),
    "instant": lambda: _java("java.time.Instant").parse(
        "2021-02-03T04:05:06.789123456Z"
    ),
    "offset_datetime": lambda: _java("java.time.OffsetDateTime").parse(
        "2021-02-03T04:05:06.5+09:00"
    ),
    "util_date": lambda: _java("java.util.Date")(1612325106789),
    "set": lambda: _java_set("a", "b"),
    "nan": lambda: float("nan"),
    "inf": lambda: float("inf"),
    "ninf": lambda: float("-inf"),
    "float_array": lambda: jpype.JArray(jpype.JFloat)([0.1, 0.2, 0.3]),
    "float_array_nan": lambda: jpype.JArray(jpype.JFloat)([0.5, float("nan")]),
    "double_array": lambda: jpype.JArray(jpype.JDouble)([0.25, 0.5]),
    "map_with_an_int_key": _int_key_map,
    "map_with_the_tag_as_key": _tag_key_map,
    "decimal_in_a_list": lambda: _java_list(1, _java("java.math.BigDecimal")("2.5")),
    "date_in_a_map": lambda: _java_map(
        k=_java_list(_java("java.time.LocalDate").of(1999, 12, 31))
    ),
    "nan_in_a_map": lambda: _java_map(a=1, b=_java_list(float("nan"))),
}

# Values that hold no tag; two of them put the escape into the text anyway (a control character and the
# text of the escape in a string), so the decoder keeps the hook and must still return the plain value.
PLAIN = {
    "ints": lambda: 7,
    "long": lambda: jpype.JLong(9223372036854775807),
    "float": lambda: 0.1,
    "negative_zero": lambda: -0.0,
    "bool": lambda: True,
    "string": lambda: 'q"uote\\ \n\t é 日本 \U0001f600',
    "list": lambda: _java_list(1, "a", 2.5, True),
    "nested_map": lambda: _java_map(a=1, b=_java_map(c=_java_list(1, 2, 3))),
    "int_array": lambda: jpype.JArray(jpype.JInt)([1, 2, 3]),
    "byte_array": lambda: jpype.JArray(jpype.JByte)([0, 1, -1]),
    "control_character_in_a_string": lambda: "a\u0001b",
    "the_escape_text_in_a_string": lambda: "x\\u0001y",
}


def _rows(value, count=3):
    return [{"v": value} for _ in range(count)]


# ---------------------------------------------------------------- a batch with no tagged value


@pytest.fixture
def plain(temp_db):
    """2,100 rows (three batches) of plain values only."""
    temp_db.command("sql", "CREATE DOCUMENT TYPE Plain")
    with temp_db.transaction():
        for i in range(2100):
            temp_db.command(
                "sql",
                "INSERT INTO Plain SET k = :k, s = :s, b = :b, x = :x, z = null, li = :li, ma = :ma",
                {
                    "k": i,
                    "s": f'row {i} "q" \\ é',
                    "b": i % 2 == 0,
                    "x": i / 8,
                    "li": [i, "a", 0.5],
                    "ma": {"a": i, "b": {"c": [1, 2, 3]}},
                },
            )
    return temp_db


PLAIN_QUERY = "SELECT k, s, b, x, z, li, ma FROM Plain ORDER BY k"


def test_to_list_of_untagged_batches_uses_no_hook_and_reads_no_side_array(plain):
    result_set = plain.query("sql", PLAIN_QUERY)  # held: freeing it closes the Java one
    texts = _texts(result_set._java_result_set, BATCH)
    assert len(texts) == 3
    assert not any(ESCAPE in text for text in texts), "the premise: nothing is tagged"
    expected = _all(reference(plain, PLAIN_QUERY))

    with watching() as seen:
        rows = plain.query("sql", PLAIN_QUERY).to_list()

    assert _all(rows) == expected
    assert len(seen.loads) == 3  # one parse per batch
    assert _hooked(seen) == 0
    assert seen.hooks == []
    assert set(seen.reads) == {0, 2}, "only the text and the drained flag are read"


def test_iter_dicts_of_untagged_rows_uses_no_hook_and_reads_no_side_array(plain):
    expected = _all(reference(plain, PLAIN_QUERY))

    with watching() as seen:
        rows = list(plain.query("sql", PLAIN_QUERY).iter_dicts())

    assert _all(rows) == expected
    assert (
        len(seen.loads) == 2100 + 1
    )  # a row per crossing, and the empty batch at the end
    assert _hooked(seen) == 0
    assert seen.hooks == []
    assert set(seen.reads) == {0}, "only the text is read"


def test_an_empty_result_is_decoded_without_a_hook(plain):
    query = "SELECT k FROM Plain WHERE k < 0"
    with watching() as seen:
        assert plain.query("sql", query).to_list() == []
        assert list(plain.query("sql", query).iter_dicts()) == []
    assert _hooked(seen) == 0 and seen.hooks == []


# ---------------------------------------------------------------- a batch with a tagged value


@pytest.mark.parametrize("name", sorted(TAGGED))
def test_every_tagged_shape_keeps_the_hook(db, name):
    rows = _rows(TAGGED[name]())
    assert any(
        ESCAPE in text for text in _texts(_in_memory(rows), BATCH)
    ), "the bridge writes the tag escape for this value"
    expected = _all(r.to_dict() for r in results.ResultSet(_in_memory(rows), db))

    with watching() as seen:
        listed = results.ResultSet(_in_memory(rows), db).to_list()
    assert _all(listed) == expected
    assert len(seen.hooks) == 1 and _hooked(seen) == 1  # the one batch
    assert 1 in seen.reads

    with watching() as seen:
        iterated = list(results.ResultSet(_in_memory(rows), db).iter_dicts())
    assert _all(iterated) == expected
    assert len(seen.hooks) == 3 and _hooked(seen) == 3  # every row is tagged
    assert seen.loads[-1] == {}, "the empty batch at the end holds nothing to restore"


def test_the_decision_is_made_for_each_batch(db):
    """One tagged row among 2,100 gives one hooked batch of three for to_list(), one hooked row for iter_dicts()."""
    decimal = _java("java.math.BigDecimal")("1.5")
    rows = [{"n": decimal if i == 1500 else i} for i in range(2100)]
    expected = _all(r.to_dict() for r in results.ResultSet(_in_memory(rows), db))

    with watching() as seen:
        listed = results.ResultSet(_in_memory(rows), db).to_list()
    assert _all(listed) == expected
    assert type(listed[1500]["n"]).__name__ == "Decimal"
    assert len(seen.loads) == 3
    assert [kwargs.get("object_hook") is not None for kwargs in seen.loads] == [
        False,
        True,
        False,
    ]
    assert len(seen.hooks) == 1
    assert seen.reads.count(1) == 1

    with watching() as seen:
        iterated = list(results.ResultSet(_in_memory(rows), db).iter_dicts())
    assert _all(iterated) == expected
    assert len(seen.loads) == 2100 + 1
    assert len(seen.hooks) == 1 and _hooked(seen) == 1
    assert seen.reads.count(1) == 1


# ---------------------------------------------------------------- the rows are the ones the hook gave


def _parity(make_result_set):
    """to_list() and iter_dicts() against the always-hooked decoding of the same batches.

    ``make_result_set`` returns a new ResultSet on every call. Each one is held while it is read: a
    ResultSet that is freed closes its Java result set.
    """
    first = make_result_set()
    old = _all(_decoded_with_the_hook_always(first._java_result_set, BATCH))
    second = make_result_set()
    one_row_at_a_time = _all(_decoded_with_the_hook_always(second._java_result_set, 1))
    assert one_row_at_a_time == old
    assert _all(make_result_set().to_list()) == old
    assert _all(make_result_set().iter_dicts()) == old
    return len(old)


@pytest.mark.parametrize("name", sorted(TAGGED) + sorted(PLAIN))
def test_each_shape_decodes_as_with_the_hook_always(db, name):
    value = (TAGGED.get(name) or PLAIN[name])()
    assert _parity(lambda: results.ResultSet(_in_memory(_rows(value)), db)) == 3


def test_a_mixed_batch_decodes_as_with_the_hook_always(db):
    """Every shape in one result set: tagged and plain values side by side, and a row of its own for each."""
    values = {name: make() for name, make in {**PLAIN, **TAGGED}.items()}
    rows = [values] + [{"v": value} for value in values.values()]
    assert _parity(lambda: results.ResultSet(_in_memory(rows), db)) == len(rows)


def test_the_wide_table_decodes_as_with_the_hook_always(db):
    """Every property type, NaN and the infinities, dates at both ends of the range, a Decimal, a vector, a link."""
    assert _parity(lambda: db.query("sql", "SELECT FROM Wide")) == 7


@pytest.mark.parametrize(
    "query",
    [
        "SELECT i, s FROM Wide WHERE i IS NOT NULL",
        "SELECT s FROM Wide WHERE s IS NOT NULL",
        "SELECT x, f FROM Wide WHERE x IS NOT NULL",
        "SELECT [1, 2] AS a, {'x': {'y': 1}} AS m",
        "SELECT s AS `\u0001`, i FROM Wide LIMIT 3",
        "SELECT count(*) AS c FROM Wide",
    ],
)
def test_query_shapes_decode_as_with_the_hook_always(db, query):
    assert _parity(lambda: db.query("sql", query)) >= 1


def test_batches_across_the_boundary_decode_as_with_the_hook_always(db):
    """2,500 rows with a tagged value in the first, the middle and the last batch."""
    decimal = _java("java.math.BigDecimal")("1.5")
    date = _java("java.time.LocalDate").of(2020, 1, 2)
    tagged = {3: decimal, 1700: float("nan"), 2499: date}
    rows = [{"n": tagged.get(i, i)} for i in range(2500)]
    assert _parity(lambda: results.ResultSet(_in_memory(rows), db)) == 2500
