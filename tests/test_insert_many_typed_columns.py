"""insert_many (the JVM wheel): a chunk whose rows have the same keys and whose columns each hold one kind of value crosses as typed columns.

Before, every chunk of 10,000 rows became one JSON text (``json_bulk_dumps_tagged``), which the engine parsed again into one JSON tree and
copied key by key (``DocumentBatcher.insertManyJson``). Now a homogeneous chunk takes ``DocumentBatcher.insertManyColumns``: one numpy
buffer per int, float or bool column, one UTF-8 buffer per string column, one ``TemporalColumn`` per date or date-time column, and no text.
Every other chunk keeps the JSON text exactly as it was.

The stored data has to be the same on both routes, so each parity test stores the same rows twice, once with the typed route switched off
(``core._typed_chunk_columns`` returns None: the old route) and once as shipped, and compares every record property by property
through SQL: the property names in the order the record lists them, the Java class of every value (an Integer is not a Long), and the
value. The route that ran is read from the calls made on the bridge class. The Java types are read through the Java database
(``get_java_database``), so this file has no native counterpart (the native wheel carries homogeneous chunks through its own columnar
entry, ``arcadedb._native.bulk.insert_rows_columnar``).
"""

import datetime as dt
import decimal
import math

import jpype
import numpy as np
import pytest

from arcadedb import core
from arcadedb.exceptions import ArcadeDBError

BRIDGE = "com.arcadedb.python.DocumentBatcher"
UTC9 = dt.timezone(dt.timedelta(hours=9))


class _Spy:
    """Forwards every call of a Java class and records its name."""

    def __init__(self, target, log):
        self._target, self._log = target, log

    def __getattr__(self, name):
        attribute = getattr(self._target, name)
        if not callable(attribute):
            return attribute

        def call(*args):
            self._log.append(name)
            return attribute(*args)

        return call


@pytest.fixture
def bridge(monkeypatch):
    """The names of the DocumentBatcher calls of the test, in order."""
    log = []
    real = core._java_class

    def spy(name):
        cls = real(name)
        return _Spy(cls, log) if name == BRIDGE else cls

    monkeypatch.setattr(core, "_java_class", spy)
    return log


def _old_route(monkeypatch):
    """The route before this change: no chunk is typed, every chunk is one JSON text."""
    monkeypatch.setattr(core, "_typed_chunk_columns", lambda chunk: None)


def _describe(value):
    if value is None:
        return ("null",)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, jpype.JArray):
        return (str(value.getClass().getName()), repr(list(value)))
    if hasattr(value, "getClass"):
        return (str(value.getClass().getName()), str(value))
    return ("py:" + type(value).__name__, repr(value))


def _stored(db, type_name):
    """Every record by id: [(property, (Java class, text of the value)), ...] in the order the record lists its properties."""
    result = db.get_java_database().query(
        "sql", f"SELECT FROM {type_name} ORDER BY id"  # nosec B608 - test-owned name
    )
    records = []
    while result.hasNext():
        record = result.next()
        records.append(
            [
                (str(name), _describe(record.getProperty(name)))
                for name in record.getPropertyNames()
            ]
        )
    return records


DECLARED = {
    "i": "LONG",
    "f": "DOUBLE",
    "b": "BOOLEAN",
    "s": "STRING",
    "d": "DATE",
    "t": "DATETIME",
}


def _make(db, name, declared=False, kind="DOCUMENT", buckets=None):
    suffix = f" BUCKETS {buckets}" if buckets else ""
    db.command("sql", f"CREATE {kind} TYPE {name}{suffix}")
    db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")
    if declared:
        for prop, ty in DECLARED.items():
            db.command("sql", f"CREATE PROPERTY {name}.{prop} {ty}")


INT_EDGES = [
    0,
    1,
    -1,
    2**31 - 1,
    2**31,
    -(2**31),
    -(2**31) - 1,
    2**53 + 1,
    -(2**53) - 1,
    2**63 - 1,
    -(2**63),
]
FLOAT_EDGES = [
    0.0,
    -0.0,
    0.1,
    1.0,
    1e300,
    -1e-300,
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    float("nan"),
    float("inf"),
    float("-inf"),
]
STRINGS = [
    "",
    "a",
    'quo"te',
    "back\\slash",
    "new\nline",
    "tab\t",
    "emoji \U0001f600",
    "é ü 中",
    "\x01tag",
    "x" * 5000,
]
DATES = [
    dt.date(1970, 1, 1),
    dt.date(1969, 12, 31),
    dt.date(2026, 11, 14),
    dt.date(1, 1, 1),
    dt.date(9999, 12, 31),
    dt.date(2000, 2, 29),
]
DATETIMES = [
    dt.datetime(1970, 1, 1),
    dt.datetime(1969, 12, 31, 23, 59, 59, 999999),
    dt.datetime(2026, 11, 14, 12, 34, 56, 123456),
    dt.datetime(2000, 2, 29, 0, 0, 0, 1),
    dt.datetime(1, 1, 1),
    dt.datetime(9999, 12, 31, 23, 59, 59, 999999),
]


def _one_column(values, name="v"):
    return [{"id": i, name: v} for i, v in enumerate(values)]


def _cycle(values, n):
    return [values[i % len(values)] for i in range(n)]


def _all_kinds(nulls):
    n = 12
    rows = []
    for i in range(n):
        null = nulls and i % 3 == 1
        rows.append(
            {
                "id": i,
                "i": None if null else INT_EDGES[i % len(INT_EDGES)],
                "f": None if null else FLOAT_EDGES[i % len(FLOAT_EDGES)],
                "b": None if null else i % 2 == 0,
                "s": None if null else STRINGS[i % len(STRINGS)],
                "d": None if null else DATES[i % len(DATES)],
                "t": None if null else DATETIMES[i % len(DATETIMES)],
            }
        )
    return rows


# name -> (rows, typed): the rows the chunk holds and the route they take
CASES = {
    "ints": (_one_column(INT_EDGES), True),
    "floats_nan_inf_subnormal": (_one_column(FLOAT_EDGES), True),
    "bools": (_one_column([True, False, True, True]), True),
    "strings": (_one_column(STRINGS), True),
    "dates": (_one_column(DATES), True),
    "datetimes": (_one_column(DATETIMES), True),
    "all_kinds": (_all_kinds(False), True),
    "all_kinds_with_nulls": (_all_kinds(True), True),
    "null_in_every_row": (_one_column([None, None, None]), True),
    "null_beside_ints": (_one_column([1, None, 2**40, None]), True),
    "null_beside_floats": (_one_column([1.5, None, float("nan")]), True),
    "null_beside_strings": (_one_column(["a", None, "b"]), True),
    "single_row": (_one_column([7]), True),
    # shapes the typed route cannot carry: the chunk keeps the JSON text and the per-row path for the rows the text cannot carry
    "other_key_order": (
        [
            {"id": 0, "a": 1, "b": 2},
            {"id": 1, "b": 3, "a": 4},
            {"id": 2, "a": 5, "b": 6},
        ],
        False,
    ),
    "key_missing_in_a_row": ([{"id": 0, "a": 1}, {"id": 1}, {"id": 2, "a": 3}], False),
    "null_in_one_row_missing_in_another": (
        [{"id": 0, "a": None}, {"id": 1}, {"id": 2, "a": 2}],
        False,
    ),
    "an_extra_key_in_a_row": ([{"id": 0, "a": 1}, {"id": 1, "a": 2, "b": 3}], False),
    "int_beside_float": (_one_column([1, 2.5, 3]), False),
    "bool_beside_int": (_one_column([True, 1, False]), False),
    "str_beside_int": (_one_column(["a", 1]), False),
    "date_beside_datetime": (_one_column([DATES[0], DATETIMES[0]]), False),
    "date_inside_a_list": (_one_column([[DATES[2]], [DATES[0]]]), False),
    "datetime_inside_a_list": (_one_column([[DATETIMES[2]], [1]]), False),
    "decimal_inside_a_list": (_one_column([[decimal.Decimal("1.5")], [2]]), False),
    "decimal": (
        _one_column([decimal.Decimal("12345.6789"), decimal.Decimal("-0.50")]),
        False,
    ),
    "bytes": (_one_column([b"\x00\xff abc\x7f", b""]), False),
    "numpy_scalars": (
        _one_column([np.int64(2**40 + 5), np.int64(-3), np.int64(7)]),
        False,
    ),
    "numpy_float_scalars": (_one_column([np.float64(0.1), np.float64(2.5)]), False),
    "aware_datetime": (
        _one_column([dt.datetime(2026, 5, 6, 7, 8, 9, 654321, tzinfo=UTC9)]),
        False,
    ),
    "nested_lists_and_maps": (
        _one_column([[1, 2, [3, 4]], {"k": [1, {"z": 2}], "n": None}]),
        False,
    ),
    "tuple": (_one_column([(1, 2, 3), (4, 5, 6)]), False),
    "list_of_ints_and_a_float": (_one_column([[1, 2.5], [3, 4]]), False),
}


# every case on schemaless properties, and the rows of every kind on declared ones too (the engine converts to the declared type)
PAIRS = [(case, False) for case in CASES] + [
    ("all_kinds", True),
    ("all_kinds_with_nulls", True),
]


@pytest.mark.parametrize(
    "case, declared",
    PAIRS,
    ids=[f"{c}-{'declared' if d else 'free'}" for c, d in PAIRS],
)
def test_the_typed_route_stores_what_the_json_route_stores(
    temp_db, monkeypatch, bridge, case, declared
):
    rows, typed = CASES[case]
    _make(temp_db, "Old", declared)
    _make(temp_db, "New", declared)
    with monkeypatch.context() as old:
        _old_route(old)
        bridge.clear()
        assert temp_db.insert_many("Old", rows) == len(rows)
        old_calls = list(bridge)
    assert "insertManyColumns" not in old_calls
    bridge.clear()
    assert temp_db.insert_many("New", rows) == len(rows)
    if typed:
        assert bridge == ["insertManyColumns"], f"{case} left the typed route"
    else:
        # the chunk keeps the route it always had, call for call (the JSON text, and the per-row path for rows the text cannot carry)
        assert (
            bridge == old_calls
        ), f"{case} took the typed route, which cannot carry it"
    new, old = _stored(temp_db, "New"), _stored(temp_db, "Old")
    assert len(new) == len(rows)
    assert new == old


def test_a_whole_number_is_an_integer_when_it_fits_an_int_and_a_long_beyond(
    temp_db, bridge
):
    """The classes themselves, not only old against new: what the JSON text stored (JSONObject.parseNumber)."""
    _make(temp_db, "T")
    temp_db.insert_many(
        "T", _one_column([5, 2**31 - 1, 2**31, -(2**31), -(2**31) - 1, 2**62])
    )
    assert bridge == ["insertManyColumns"]
    classes = [dict(r)["v"][0] for r in _stored(temp_db, "T")]
    assert classes == [
        "java.lang.Integer",
        "java.lang.Integer",
        "java.lang.Long",
        "java.lang.Integer",
        "java.lang.Long",
        "java.lang.Long",
    ]


def test_insert_columns_still_stores_a_long_for_every_whole_number(temp_db):
    """insert_columns shares fillRow with the new entry point; only insertManyColumns narrows (a regression test, green before this change)."""
    _make(temp_db, "C")
    temp_db.insert_columns("C", {"id": np.arange(3, dtype=np.int64), "v": [1, 2, 3]})
    for record in _stored(temp_db, "C"):
        assert dict(record)["id"][0] == "java.lang.Long"


def test_a_chunk_that_takes_the_old_route_builds_no_bridge_array(
    temp_db, monkeypatch, bridge
):
    """A fallback chunk costs the census and nothing else: no column is built before the shape is known to qualify."""
    built = []
    real = core._typed_java_column

    def spy(*args):
        built.append(args[0])
        return real(*args)

    monkeypatch.setattr(core, "_typed_java_column", spy)
    _make(temp_db, "T")
    # the first column is fine, the second holds a Decimal: nothing is built for either
    temp_db.insert_many("T", [{"id": i, "v": decimal.Decimal(i)} for i in range(5)])
    assert built == []
    assert bridge == ["insertManyJson"]


def test_a_homogeneous_load_crosses_once_per_chunk_with_no_text_and_no_per_value_conversion(
    temp_db, monkeypatch, bridge
):
    monkeypatch.setattr(core, "_INSERT_MANY_CHUNK", 1000)

    def no_text(*args, **kwargs):
        raise AssertionError("a homogeneous chunk went through the JSON text")

    monkeypatch.setattr(core, "json_bulk_dumps_tagged", no_text)
    converted = []
    real = core.convert_python_to_java

    def counting(value):
        converted.append(type(value).__name__)
        return real(value)

    monkeypatch.setattr(core, "convert_python_to_java", counting)
    _make(temp_db, "T")
    rows = [
        {
            "id": i,
            "n": i * 3,
            "x": i * 0.5,
            "ok": i % 2 == 0,
            "name": f"row {i}",
            "day": DATES[i % len(DATES)],
            "at": DATETIMES[i % len(DATETIMES)],
        }
        for i in range(2500)
    ]
    assert temp_db.insert_many("T", rows) == 2500
    assert bridge == ["insertManyColumns"] * 3  # chunks of 1000, 1000, 500
    assert converted == [], "a value was converted one at a time"
    assert temp_db.query("sql", "SELECT count(*) AS c FROM T").first().get("c") == 2500
    record = dict(
        _stored(temp_db, "T")[1235]
    )  # 1235 % 6 == 5: the last date and date-time of the lists above
    assert record == {
        "id": ("java.lang.Integer", "1235"),
        "n": ("java.lang.Integer", "3705"),
        "x": ("java.lang.Double", "617.5"),
        "ok": ("java.lang.Boolean", "false"),
        "name": ("str", "row 1235"),
        "day": ("java.time.LocalDate", "2000-02-29"),
        "at": ("java.time.LocalDateTime", DATETIMES[5].isoformat()),
    }


def test_chunks_of_both_routes_keep_the_order_of_the_rows(temp_db, monkeypatch, bridge):
    monkeypatch.setattr(core, "_INSERT_MANY_CHUNK", 4)
    _make(temp_db, "T", buckets=1)
    rows = _one_column(list(range(12)))
    rows[5] = {"id": 5, "v": decimal.Decimal(5)}  # the middle chunk keeps the text
    assert temp_db.insert_many("T", rows) == 12
    assert bridge == ["insertManyColumns", "insertManyJson", "insertManyColumns"]
    ids = [r.get("id") for r in temp_db.query("sql", "SELECT id FROM T").to_list()]
    assert ids == list(range(12))


def test_the_commit_boundary_still_splits_a_chunk(temp_db, bridge):
    _make(temp_db, "T")
    assert temp_db.insert_many("T", _one_column(list(range(25))), commit_every=10) == 25
    assert (
        bridge == ["insertManyColumns"] * 3
    )  # 10, 10, 5: a chunk never straddles a commit
    assert len(_stored(temp_db, "T")) == 25


def test_an_empty_batch_and_a_batch_of_one(temp_db, bridge):
    _make(temp_db, "T")
    assert temp_db.insert_many("T", []) == 0
    assert bridge == []
    assert temp_db.insert_many("T", [{"id": 1, "v": 2}]) == 1
    assert bridge == ["insertManyColumns"]
    assert len(_stored(temp_db, "T")) == 1


def test_a_generator_is_read_chunk_by_chunk(temp_db, monkeypatch, bridge):
    monkeypatch.setattr(core, "_INSERT_MANY_CHUNK", 100)
    _make(temp_db, "T")
    n = temp_db.insert_many("T", ({"id": i, "v": i} for i in range(250)))
    assert n == 250
    assert bridge == ["insertManyColumns"] * 3


@pytest.mark.parametrize("kind", ["DOCUMENT", "VERTEX"])
def test_documents_and_vertices_are_the_same_on_both_routes(
    temp_db, monkeypatch, bridge, kind
):
    _make(temp_db, "Old", kind=kind)
    _make(temp_db, "New", kind=kind)
    rows = _one_column(INT_EDGES)
    with monkeypatch.context() as old:
        _old_route(old)
        temp_db.insert_many("Old", rows)
    temp_db.insert_many("New", rows)
    assert _stored(temp_db, "New") == _stored(temp_db, "Old")
    for name in ("Old", "New"):
        sql = f"SELECT FROM {name}"  # nosec B608 - test-owned name
        result = temp_db.get_java_database().query("sql", sql)
        n = 0
        while result.hasNext():
            assert result.next().isVertex() == (kind == "VERTEX")
            n += 1
        assert n == len(rows)


def test_an_edge_type_is_refused_as_before(temp_db, monkeypatch):
    temp_db.command("sql", "CREATE VERTEX TYPE V")
    temp_db.command("sql", "CREATE EDGE TYPE E")
    rows = _one_column([1, 2])
    messages = []
    for route in ("old", "new"):
        with monkeypatch.context() as m:
            if route == "old":
                _old_route(m)
            with pytest.raises(ArcadeDBError) as error:
                temp_db.insert_many("E", rows)
            messages.append(str(error.value))
    assert messages[0] == messages[1]
    assert "edge type" in messages[0]


def test_a_duplicate_key_fails_the_call_and_rolls_it_back_on_both_routes(
    temp_db, monkeypatch, bridge
):
    outcomes = []
    for route in ("old", "new"):
        name = f"T_{route}"
        temp_db.command("sql", f"CREATE DOCUMENT TYPE {name}")
        temp_db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")
        temp_db.command("sql", f"CREATE INDEX ON {name} (id) UNIQUE")
        rows = _one_column([1, 2, 3, 4]) + [
            {"id": 2, "v": 9}
        ]  # the id 2 is there twice
        with monkeypatch.context() as m:
            if route == "old":
                _old_route(m)
            bridge.clear()
            with pytest.raises(
                Exception
            ) as error:  # noqa: B017 - the class is compared below
                temp_db.insert_many(name, rows)
        outcomes.append(
            (
                type(error.value).__name__,
                temp_db.query("sql", f"SELECT count(*) AS c FROM {name}")  # nosec B608
                .first()
                .get("c"),
                list(bridge),
            )
        )
    assert outcomes[0][:2] == outcomes[1][:2]
    assert outcomes[1][1] == 0, "the call rolled back the transaction it opened"
    assert outcomes[0][2] == ["insertManyJson"]
    assert outcomes[1][2] == ["insertManyColumns"]


@pytest.mark.parametrize(
    "values",
    [[1, 2**64], [-(2**63) - 1, 1], ["ok", "a\ud800b"]],
    ids=["beyond_int64", "below_int64", "lone_surrogate"],
)
def test_a_value_no_route_can_store_fails_as_before(
    temp_db, monkeypatch, bridge, values
):
    outcomes = []
    for route in ("old", "new"):
        name = f"T_{route}"
        _make(temp_db, name)
        with monkeypatch.context() as m:
            if route == "old":
                _old_route(m)
            bridge.clear()
            try:
                temp_db.insert_many(name, _one_column(values))
                raised = None
            except Exception as error:  # noqa: BLE001 - the same class is the claim
                raised = type(error).__name__
        outcomes.append(
            (raised, len(_stored(temp_db, name)), "insertManyColumns" in bridge)
        )
    assert outcomes[0][:2] == outcomes[1][:2]
    assert (
        outcomes[1][2] is False
    ), "a value the text refuses must not take the typed route"


@pytest.mark.parametrize("case", ["ints", "all_kinds_with_nulls", "decimal"])
def test_parallel_mode_stores_what_the_json_route_stores(
    temp_db, monkeypatch, bridge, case
):
    rows, typed = CASES[case]
    _make(temp_db, "Old")
    _make(temp_db, "New")
    with monkeypatch.context() as old:
        _old_route(old)
        bridge.clear()
        assert temp_db.insert_many("Old", rows, parallel=True) == len(rows)
        assert "insertManyJsonParallel" in bridge
    bridge.clear()
    assert temp_db.insert_many("New", rows, parallel=True) == len(rows)
    assert bridge[0] == (
        "insertManyColumnsParallel" if typed else "insertManyJsonParallel"
    )
    assert _stored(temp_db, "New") == _stored(temp_db, "Old")


def test_a_caller_transaction_is_left_to_the_caller(temp_db, bridge):
    _make(temp_db, "T")
    with pytest.raises(RuntimeError):
        with temp_db.transaction():
            temp_db.insert_many("T", _one_column(list(range(30))), commit_every=10)
            raise RuntimeError("roll back")
    assert bridge == ["insertManyColumns"]
    assert temp_db.query("sql", "SELECT count(*) AS c FROM T").first().get("c") == 0


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3], ("int", False)),
        ([1, None], ("int", True)),
        ([2**63], None),
        ([-(2**63) - 1, None], None),
        ([1, 2.0], None),
        ([True, False], ("bool", False)),
        ([True, 1], None),
        ([None, None], ("null", True)),
        (["a", None], ("str", True)),
        ([1.5, math.nan], ("float", False)),
        ([dt.date(2020, 1, 2)], ("date", False)),
        ([dt.datetime(2020, 1, 2)], ("datetime", False)),
        ([dt.datetime(2020, 1, 2), None], ("datetime", True)),
        ([dt.datetime(2020, 1, 2, tzinfo=dt.timezone.utc)], None),
        ([dt.date(2020, 1, 2), dt.datetime(2020, 1, 2)], None),
        ([decimal.Decimal(1)], None),
        ([b"x"], None),
        ([[1, 2]], None),
        ([{"a": 1}], None),
        ([np.int64(1)], None),
        ([np.float64(1)], None),
    ],
)
def test_which_columns_qualify(values, expected):
    assert core._typed_kind(values) == expected


@pytest.mark.parametrize(
    "chunk",
    [
        [],
        [{}, {}],
        [{1: 1}],
        [{"a": 1}, [("a", 2)]],
        [{"a": 1, "b": 2}, {"b": 2, "a": 1}],
        [{"a": 1}, {"a": 1, "b": 2}],
        [{"a": 1}, {"b": 1}],
        [{"a": 1}, {"a": 2.5}],
        [{"a": [1]}, {"a": [2]}],
    ],
    ids=[
        "empty",
        "no_keys",
        "non_str_key",
        "not_a_dict",
        "key_order",
        "missing_key",
        "other_key",
        "mixed_kind",
        "list",
    ],
)
def test_which_chunks_qualify(chunk):
    assert core._typed_chunk_columns(chunk) is None
