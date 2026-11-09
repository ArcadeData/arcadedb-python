"""The native columnar readers give the same table as the rules they replaced (refs #312).

``arcadedb._native.columns`` infers an Arrow column with ``pa.array(values)`` first and falls back to the per-value
rules (``_kind``) when inference cannot type the column or types it differently; ``to_columns`` calls ``_kind`` once
per column; ``_names`` skips the per-key step for a row whose keys are known; ``NativeResultSet._drain_dicts`` takes
the rows of a reply without one tuple per row. None of that may change a value, a dtype or a column order.

The reference functions below are the code as it was before: the rules of ``_kind``, ``_names``, ``to_columns`` and
``to_arrow``, copied. ``_ms`` and ``_float_vectors`` were not changed and are shared with the module. Each case builds
rows, runs both and compares the outcome, an exception included (a column that raised before raises the same class now).
Pure Python on the rows: no database is needed except for the drain test at the end.

conftest.py collects this file on the native backend only (the module lives in the native package).
"""

import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

from arcadedb._native import columns as new
from arcadedb._native.columns import _float_vectors, _ms

# --- the reference: the rules before this change --------------------------------------------------------------------------


def _names_ref(rows, columns):
    if columns is not None:
        return list(columns)
    seen = {}
    for row in rows:
        for k in row:
            seen.setdefault(k, None)
    return list(seen)


def _kind_ref(values):
    kinds = set()
    for v in values:
        if v is None:
            continue
        t = type(v)
        if t is bool:
            kinds.add("b")
        elif t is int:
            kinds.add("i")
        elif t is float:
            kinds.add("f")
        elif t is datetime or t is date:
            kinds.add("dt")
        elif t is Decimal:
            kinds.add("dec")
        else:
            kinds.add("obj")
    if not kinds:
        return "obj"
    if len(kinds) == 1:
        return kinds.pop()
    if kinds == {"i", "f"}:
        return "f"
    return "obj"


def _to_columns_ref(rows, columns=None):
    out = {}
    for name in _names_ref(rows, columns):
        values = [row.get(name) for row in rows]
        if _kind_ref(values) == "obj":
            vectors = _float_vectors(values)
            if vectors is not None:
                out[name] = vectors
                continue
        has_null = any(v is None for v in values)
        kind = _kind_ref(values)
        if kind == "i":
            out[name] = (
                np.array([np.nan if v is None else v for v in values], dtype=np.float64)
                if has_null
                else np.array(values, dtype=np.int64)
            )
        elif kind == "f":
            out[name] = np.array(
                [np.nan if v is None else v for v in values], dtype=np.float64
            )
        elif kind == "b":
            out[name] = values if has_null else np.array(values, dtype=bool)
        elif kind == "dt":
            out[name] = np.array(
                [
                    (
                        np.datetime64("NaT", "ms")
                        if v is None
                        else np.datetime64(_ms(v), "ms")
                    )
                    for v in values
                ],
                dtype="datetime64[ms]",
            )
        elif kind == "dec":
            arr = np.empty(len(values), dtype=object)
            arr[:] = values
            out[name] = arr
        else:
            out[name] = values
    return out


def _to_arrow_ref(rows, columns=None):
    arrays = {}
    for name in _names_ref(rows, columns):
        values = [row.get(name) for row in rows]
        kind = _kind_ref(values)
        if kind == "i":
            arrays[name] = pa.array(values, type=pa.int64())
        elif kind == "f":
            try:
                arrays[name] = pa.array(values, type=pa.float64())
            except pa.ArrowInvalid:
                arrays[name] = pa.array(
                    [None if v is None else str(v) for v in values], type=pa.string()
                )
        elif kind == "b":
            arrays[name] = pa.array(values, type=pa.bool_())
        elif kind == "dt":
            arrays[name] = pa.array(
                [None if v is None else _ms(v) for v in values], type=pa.timestamp("ms")
            )
        elif kind == "dec":
            try:
                arrays[name] = pa.array(values)
            except pa.ArrowInvalid:
                arrays[name] = pa.array(
                    [None if v is None else str(v) for v in values], type=pa.string()
                )
        else:
            try:
                arrays[name] = pa.array([None if v is None else v for v in values])
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                arrays[name] = pa.array(
                    [None if v is None else str(v) for v in values], type=pa.string()
                )
    return pa.table(arrays)


# --- the inputs -----------------------------------------------------------------------------------------------------------

BIG = 2**53 + 1
STAMP = datetime(2026, 1, 2, 3, 4, 5, 678901)
AWARE = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone(timedelta(hours=9)))
WIDE40 = Decimal("1234567890123456789012345678901234567890.5")  # decimal256
WIDE80 = Decimal("1" * 80)  # wider than decimal256: exact text

# one column per case, each a list of values that go in rows as {"c": value}
COLUMNS = {
    "ints": [1, 2, 3],
    "ints-null": [1, None, 3],
    "ints-int64-edges": [-(2**63), 2**63 - 1, 0],
    "int-above-int64": [2**63, 1],
    "int-above-uint64": [2**64, 1],
    "int-below-int64": [-(2**63) - 1, 1],
    "floats": [0.5, 1.5, float("nan")],
    "floats-null": [0.5, None, float("inf")],
    "ints-and-floats": [1, 2.5, None, 3],
    "int-beyond-2-53-and-floats": [BIG, 0.5],
    "big-int-beyond-2-53-and-floats": [2**60, 1.5, None],
    "int-above-int64-and-floats": [2**63, 0.5],
    "int-above-uint64-and-floats": [2**64, 1.5, None],
    "int-below-int64-and-floats": [-(2**63) - 1, 0.5],
    "int-above-int64-and-string": [2**63, "x"],
    "bools": [True, False, True],
    "bools-null": [True, None, False],
    "bool-and-int": [True, 1, 0],
    "bool-and-float": [True, 0.5],
    "strings": ["a", "bb", ""],
    "strings-null": ["a", None, "c"],
    "string-and-int": ["seven", 1],
    "all-null": [None, None, None],
    "single-null": [None],
    "datetimes": [STAMP, datetime(1969, 12, 31, 23, 59, 59, 999000)],
    "datetimes-null": [STAMP, None],
    "dates": [date(2026, 1, 2), date(1969, 12, 31)],
    "date-and-datetime": [date(2026, 1, 2), STAMP, None],
    "aware-datetime": [AWARE, STAMP],
    "datetime-and-string": [STAMP, "x"],
    "decimals": [Decimal("1.50"), Decimal("-0.25")],
    "decimals-null": [Decimal("2"), None, Decimal("30")],
    "decimal-40-digits": [WIDE40, Decimal("1")],
    "decimal-80-digits": [WIDE80, Decimal("1")],
    "decimal-and-int": [Decimal("1.5"), 2],
    "vectors": [[0.5, 1.5], [0.25, 2.0]],
    "vectors-null": [[0.5, 1.5], None, [0.25, 2.0]],
    "vector-not-float32": [[0.1, 0.2], [0.3, 0.4]],
    "vector-ragged": [[0.5], [0.5, 1.5]],
    "lists-of-ints": [[1, 2], [3]],
    "lists-mixed": [[1, "a", 2.5], ["x"]],
    "empty-lists": [[], []],
    "dicts": [{"a": 1}, {"b": [1.5, None]}],
    "nested-float-lists": [[[0.5, 1.5]], [[2.5]]],
    "list-of-datetimes": [[STAMP], [None]],
    "bytes-like": [b"ab", b"c"],
}


def _rows(values, name="c"):
    return [{name: v} for v in values]


def _wide_null_heavy(seed=7, count=1000):
    """Many rows, ragged keys, about nine values in ten missing or None, three vectors, strings, ints and floats."""
    rng = random.Random(seed)  # nosec B311 - fixed test data, not a secret
    rows = []
    for i in range(count):
        row = {}
        if rng.random() < 0.9:
            row["i"] = rng.randrange(-1000, 1000) if rng.random() < 0.9 else None
        if rng.random() < 0.1:
            row["f"] = rng.random() * 10
        if rng.random() < 0.1:
            row["s"] = f"s{i}" if rng.random() < 0.8 else None
        if rng.random() < 0.1:
            row["late-" + str(i % 3)] = i
        if rng.random() < 0.1:
            row["v"] = [float(np.float32(rng.random())) for _ in range(4)]
        if rng.random() < 0.05:
            row["m"] = rng.choice([1, 2.5, "x", None, True])
        rows.append(row)
    return rows


# --- the comparison -------------------------------------------------------------------------------------------------------


def _outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the class of the failure is part of the result
        return ("raises", type(exc))


def _same_table(a, b):
    assert a.schema == b.schema, (a.schema, b.schema)
    assert a.column_names == b.column_names
    # not Table.equals: it is False for a NaN next to the same NaN. repr counts NaN, -0.0 and the type of every value
    assert repr(a.to_pydict()) == repr(b.to_pydict())


def _same_column(a, b, name):
    assert type(a) is type(b), (name, type(a), type(b))
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype)
        assert repr(a.tolist()) == repr(b.tolist()), name
    else:
        assert repr(a) == repr(b), name


def _check_arrow(rows, columns=None):
    want = _outcome(_to_arrow_ref, rows, columns)
    have = _outcome(new.to_arrow, rows, columns)
    assert have[0] == want[0], (have, want)
    if want[0] == "ok":
        _same_table(have[1], want[1])
    else:
        assert have[1] is want[1]


def _check_columns(rows, columns=None):
    want = _outcome(_to_columns_ref, rows, columns)
    have = _outcome(new.to_columns, rows, columns)
    assert have[0] == want[0], (have, want)
    if want[0] == "ok":
        assert list(have[1]) == list(want[1])
        for name in want[1]:
            _same_column(have[1][name], want[1][name], name)
    else:
        assert have[1] is want[1]


@pytest.mark.parametrize("case", sorted(COLUMNS))
def test_a_single_column_gives_the_same_arrow_table(case):
    _check_arrow(_rows(COLUMNS[case]))


@pytest.mark.parametrize("case", sorted(COLUMNS))
def test_a_single_column_gives_the_same_columns(case):
    _check_columns(_rows(COLUMNS[case]))


def test_a_table_of_all_the_cases_gives_the_same_arrow_table_and_columns():
    names = sorted(COLUMNS)
    count = max(len(COLUMNS[n]) for n in names)
    rows = [
        {n: COLUMNS[n][i] for n in names if i < len(COLUMNS[n])} for i in range(count)
    ]
    # a case that raises on its own raises for the table: compare the table without the columns that raise
    ok = [n for n in names if _outcome(_to_arrow_ref, _rows(COLUMNS[n]))[0] == "ok"]
    rows = [{n: r[n] for n in ok if n in r} for r in rows]
    _check_arrow(rows)
    ok_columns = [
        n for n in ok if _outcome(_to_columns_ref, _rows(COLUMNS[n]))[0] == "ok"
    ]
    _check_columns([{n: r[n] for n in ok_columns if n in r} for r in rows])


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_a_null_heavy_ragged_table_gives_the_same_tables(seed):
    rows = _wide_null_heavy(seed)
    _check_arrow(rows)
    _check_columns(rows)


def test_a_chosen_column_list_gives_the_same_tables():
    rows = _wide_null_heavy()
    for columns in (["i", "s"], ["v", "nope", "i"], []):
        _check_arrow(rows, columns)
        _check_columns(rows, columns)


@pytest.mark.parametrize("columns", [None, ["a"]])
def test_no_rows_gives_the_same_tables(columns):
    _check_arrow([], columns)
    _check_columns([], columns)


def test_rows_of_empty_dicts_give_the_same_tables():
    _check_arrow([{}, {}])
    _check_columns([{}, {}])


def test_names_are_the_keys_in_the_order_they_are_first_seen():
    rows = [{"b": 1}, {"a": 1, "b": 2}, {"b": 3}, {"c": 1, "a": 2}, {"d": 1, "b": 1}]
    assert new._names(rows, None) == _names_ref(rows, None) == ["b", "a", "c", "d"]
    rows = _wide_null_heavy()
    assert new._names(rows, None) == _names_ref(rows, None)
    assert new._names(rows, ["z", "y"]) == ["z", "y"]


def test_the_inference_only_takes_a_column_the_rules_type_the_same():
    """A temporal column and an int above int64 are left to the rules; the rest is the inferred array."""
    assert new._inferred([STAMP, None]) is None
    assert new._inferred([date(2026, 1, 2)]) is None
    assert new._inferred([2**63]) is None
    assert new._inferred([1, None, 3]).type == pa.int64()
    assert new._inferred([0.5, 1]).type == pa.float64()
    assert new._inferred([True, None]).type == pa.bool_()
    assert new._inferred(["a", None]).type == pa.string()
    assert new._inferred([None, None]).type == pa.null()


def test_to_columns_asks_for_the_kind_of_a_column_once(monkeypatch):
    calls = []
    real = new._kind

    def counting(values):
        calls.append(len(values))
        return real(values)

    monkeypatch.setattr(new, "_kind", counting)
    new.to_columns(_rows([1, None, 3]), None)
    assert calls == [3]
    calls.clear()
    new.to_columns(_rows([[0.5, 1.5], None]), None)
    assert calls == [2]


# --- the drain ------------------------------------------------------------------------------------------------------------

DRAIN_ROWS = 5000  # more than the first reply (256) and two further replies (2048 each)


@pytest.fixture
def drain_db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Drain")
    temp_db.command("sql", "CREATE PROPERTY Drain.id INTEGER")
    temp_db.insert_many(
        "Drain",
        [{"id": i, "txt": f"t{i}", "val": i / 8.0} for i in range(DRAIN_ROWS)],
    )
    return temp_db


SELECT_DRAIN = "SELECT id, txt, val FROM Drain ORDER BY id"


def test_the_drain_returns_the_rows_row_by_row_iteration_returns(drain_db):
    by_rows = [r.to_dict() for r in drain_db.query("sql", SELECT_DRAIN)]
    drained = drain_db.query("sql", SELECT_DRAIN)._drain_dicts()
    assert len(drained) == DRAIN_ROWS
    assert drained == by_rows
    assert drained[0] == {"id": 0, "txt": "t0", "val": 0.0}
    assert drained[-1] == {
        "id": DRAIN_ROWS - 1,
        "txt": f"t{DRAIN_ROWS - 1}",
        "val": (DRAIN_ROWS - 1) / 8.0,
    }


@pytest.mark.parametrize("taken", [1, 255, 256, 257, 2304, 4999])
def test_the_drain_after_some_rows_were_taken_returns_the_rest(drain_db, taken):
    rs = drain_db.query("sql", SELECT_DRAIN)
    head = [next(rs).to_dict() for _ in range(taken)]
    rest = rs._drain_dicts()
    want = [r.to_dict() for r in drain_db.query("sql", SELECT_DRAIN)]
    assert head + rest == want


def test_the_drain_of_an_empty_result_and_a_second_drain_are_empty(drain_db):
    rs = drain_db.query("sql", "SELECT id FROM Drain WHERE id < 0")
    assert rs._drain_dicts() == []
    rs = drain_db.query("sql", SELECT_DRAIN)
    assert len(rs._drain_dicts()) == DRAIN_ROWS
    assert rs._drain_dicts() == []


def test_to_arrow_and_to_columns_of_a_result_equal_the_reference_over_its_rows(
    drain_db,
):
    rows = drain_db.query("sql", SELECT_DRAIN)._drain_dicts()
    table = drain_db.query("sql", SELECT_DRAIN).to_arrow()
    _same_table(table, _to_arrow_ref(rows))
    columns = drain_db.query("sql", SELECT_DRAIN).to_columns()
    want = _to_columns_ref(rows)
    for name in want:
        _same_column(columns[name], want[name], name)
