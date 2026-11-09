"""insert_many and create_vertices validate their rows in the one json.dumps pass, not in a Python loop over the rows before it.

`json_bulk_dumps` and `json_bulk_dumps_tagged` used to walk every value of every row before json.dumps walked it again. Now json.dumps
writes the rows once and the check comes from what that pass shows (the text and the types of the rows), and a walk runs only when it
may find something. The contract is unchanged: the same text for rows that carry, and for a row that does not, the same exception type
and message as the old loop gave for the first bad row in row order. The old loop is copied here as the private reference, so these
tests fail on the old code (they spy on `_check_rows`, which only the new code has) and compare the new code with that reference.
The old messages name no row index, so the first bad row of a large batch is told from a later one by the value in its message.
"""

import datetime as dt
import decimal
import enum
import json
import math
import random

import numpy as np
import pytest

import arcadedb
from arcadedb import type_conversion as tc

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)


def _old_dumps(rows, tagged):
    """The json_bulk_dumps and json_bulk_dumps_tagged of 26.11.1: the validation loop, then json.dumps."""
    for row in rows:
        for key in row:
            if type(key) is not str:
                raise ValueError("a non-str key cannot go through the JSON bulk path")
        for value in row.values():
            kind = type(value)
            if kind is str:
                if not value.isascii() and not tc._str_survives_json(value):
                    raise ValueError(
                        "a string with a lone surrogate cannot go through the JSON bulk path"
                    )
            elif kind is int:
                if not INT64_MIN <= value <= INT64_MAX:
                    raise ValueError(
                        f"an integer beyond 64 bits cannot go through the JSON bulk path: {value}"
                    )
            elif kind is float or kind is bool or value is None:
                continue
            elif tagged and kind in tc._TAGGED_KINDS:
                continue
            elif tagged and kind is dict and tc._TAG in value:
                raise ValueError(
                    "a map keyed by the tag marker cannot go through the JSON bulk path"
                )
            elif tagged and (
                isinstance(
                    value, (dt.datetime, dt.date, decimal.Decimal, bytes, bytearray)
                )
                or isinstance(value, np.generic)
                or tc._is_pandas_null(kind)
            ):
                continue
            else:
                tc._json_bulk_check(value)
    if tagged:
        return tc._dumps_checked(rows, default=tc.bulk_tag_default)
    return tc._dumps_checked(rows)


def _outcome(call, rows):
    try:
        return ("text", call(rows))
    except (
        BaseException
    ) as e:  # noqa: BLE001 - the type and the message are what is compared
        return (type(e).__name__, str(e))


class Colour(enum.IntEnum):
    RED = 5


class Name(str):
    pass


def _scalar_rows():
    return [
        {"id": 1, "name": "alpha", "score": 0.5, "flag": True, "note": None},
        {"id": 2, "name": "", "score": -1e-30, "flag": False, "note": "x"},
    ]


CASES = {
    "scalars": _scalar_rows(),
    "empty": [],
    "empty_rows": [{}, {}],
    "bounds_fit": [{"a": INT64_MAX, "b": INT64_MIN, "c": 0}],
    "above_max": [{"a": INT64_MAX + 1}],
    "below_min": [{"a": INT64_MIN - 1}],
    "big_negative": [{"a": -(10**30)}],
    "huge_digits": [{"a": 10**5000}],
    "non_bmp": [{"a": "emoji \U0001f600 pair", "b": "\U00010000"}],
    "cjk_and_hangul": [{"a": "中文한글퟿"}],
    "nul_and_controls": [{"a": "a\x00b\x1f", "b": " "}],
    "lone_high": [{"a": "a\ud800b"}],
    "lone_low": [{"a": "\udc00"}],
    "two_lone_halves": [{"a": "𐀀"}],
    "literal_backslash_u": [{"a": "\\ud800", "b": "\\u0041"}],
    "digits_in_text": [{"a": "9999999999999999999", "b": "1234567890123456789012"}],
    "nan": [{"a": math.nan}],
    "inf": [{"a": -math.inf}],
    "subnormal": [{"a": 5e-324}],
    "small_floats": [{"a": 0.00012345678901234568, "b": 1.2345678901234567e19}],
    "int_key": [{1: "a"}],
    "bool_key": [{True: "a"}],
    "none_key": [{None: "a"}],
    "list_ok": [{"a": [1, "x", None, [2.5]]}],
    "list_big": [{"a": [1, 2**70]}],
    "list_nan": [{"a": [1.0, math.nan]}],
    "list_lone": [{"a": ["\ud800"]}],
    "list_datetime": [{"a": [dt.datetime(2026, 1, 2)]}],
    "dict_ok": [{"a": {"b": [1], "c": "d"}}],
    "dict_int_key": [{"a": {1: 2}}],
    "dict_tag_key": [{"a": {tc._TAG: 1}}],
    "tuple_value": [{"a": (1, 2)}],
    "datetime": [{"a": dt.datetime(2026, 1, 2, 3, 4, 5, 6)}],
    "aware_datetime": [
        {"a": dt.datetime(2026, 1, 2, tzinfo=dt.timezone(dt.timedelta(hours=9)))}
    ],
    "date": [{"a": dt.date(2026, 1, 2)}],
    "decimal": [{"a": decimal.Decimal("1.50")}],
    "decimal_nan": [{"a": decimal.Decimal("NaN")}],
    "bytes": [{"a": b"\x00\xffab", "b": bytearray(b"x")}],
    "numpy_scalars": [
        {
            "a": np.int64(5),
            "b": np.float64(1.5),
            "c": np.bool_(True),
            "d": np.float32(2.5),
        }
    ],
    "numpy_big": [{"a": np.uint64(2**64 - 1)}],
    "numpy_nan": [{"a": np.float64(math.nan)}],
    "numpy_datetime": [{"a": np.datetime64("2026-01-02T03:04:05")}],
    "numpy_str": [{"a": np.str_("q")}],
    "int_enum": [{"a": Colour.RED}],
    "str_subclass": [{"a": Name("s")}],
    "str_subclass_lone": [{"a": Name("\ud800")}],
    "set_value": [{"a": {1, 2}}],
    "object_value": [{"a": object()}],
    "none_row": [None],
    "list_row": [[1, 2]],
    "string_row": ["ab"],
    "mixed_good_then_bad": [{"a": 1}, {"a": 2**64}, {"a": math.nan}],
}


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_same_text_or_same_error_as_the_validation_loop(name, tagged):
    rows = CASES[name]
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    assert _outcome(new, rows) == _outcome(lambda r: _old_dumps(r, tagged), rows)


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
def test_random_rows_agree_with_the_validation_loop(tagged):
    atoms = [
        None, True, 0, -1, INT64_MAX, INT64_MIN, INT64_MAX + 1, 10**30, 1.5, 0.00012345678901234568,
        5e-324, math.nan, math.inf, "", "abc", "a\x00b", "é中", "\U0001f600", "a\ud800b", "\udc00",
        "𐀀", "9999999999999999999", [1, 2], [math.nan], {"a": 1}, {1: 2}, (1, 2),
        dt.datetime(2026, 1, 2), dt.date(2026, 1, 2), decimal.Decimal("1.5"), b"ab", np.int64(7),
        np.float64(math.nan), Colour.RED, Name("s"), {1, 2}, object(),
    ]  # fmt: skip
    rnd = random.Random(318)  # nosec B311 - a seeded corpus, not security
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    for _ in range(3000):
        rows = []
        for _ in range(rnd.choice([1, 2, 3, 8])):
            keys = rnd.sample(["a", "b", "c", "d"], rnd.randint(0, 4))
            if rnd.random() < 0.03:
                keys.append(1)
            rows.append({k: rnd.choice(atoms) for k in keys})
        assert _outcome(new, rows) == _outcome(lambda r: _old_dumps(r, tagged), rows)


def _batch(n, bad):
    """n scalar rows of the shape of a TPC-H part table; `bad` maps a row index to its replacement row."""
    rows = [
        {
            "p_partkey": i,
            "p_name": f"part {i}",
            "p_size": i % 50,
            "p_retailprice": 900.0 + (i % 1000) / 100,
            "p_comment": None if i % 7 == 0 else "carefully final deposits",
        }
        for i in range(n)
    ]
    for index, row in bad.items():
        rows[index] = row
    return rows


# kind -> (first bad row, a later bad row of another value), so the message tells which row raised
BAD_KINDS = {
    "int_beyond_64_bits": ({"p_size": 2**63 + 11}, {"p_size": 2**70 + 22}),
    "negative_beyond_64_bits": ({"p_size": INT64_MIN - 33}, {"p_size": 2**64 + 44}),
    "lone_surrogate": ({"p_name": "a\ud800"}, {"p_size": 2**65}),
    "non_str_key": ({7: "x"}, {"p_size": 2**65}),
    "nested_int": ({"p_name": [1, 2**66]}, {"p_size": 2**65 + 55}),
    "nan": ({"p_retailprice": math.nan}, {"p_size": 2**65}),
    "subnormal": ({"p_retailprice": 5e-324}, {"p_size": 2**65}),
}


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
@pytest.mark.parametrize("kind", sorted(BAD_KINDS))
def test_the_first_bad_row_of_a_large_batch_gives_the_old_error(kind, tagged):
    first, later = BAD_KINDS[kind]
    rows = _batch(20_000, {12_345: first, 19_000: later})
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    expected = _outcome(lambda r: _old_dumps(r, tagged), rows)
    assert expected[0] in ("ValueError", "TypeError"), expected
    assert _outcome(new, rows) == expected
    # the later row alone raises a message of its own. A NaN or a subnormal is refused by json.dumps, after the old loop had walked
    # every row, so the old error there is the later row's; for every other kind the first bad row's error comes first
    only_later = _batch(20_000, {19_000: later})
    assert _outcome(new, only_later) == _outcome(
        lambda r: _old_dumps(r, tagged), only_later
    )
    assert (_outcome(new, only_later) == expected) == (kind in ("nan", "subnormal"))


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
def test_the_first_bad_rows_error_carries_no_context(tagged):
    # json.dumps refuses the NaN of the second row first; the error is still the walk's for the first row, as it was when the
    # loop ran before json.dumps, and it does not show the NaN error as the context it was raised in
    rows = [{"a": 2**70}, {"b": math.nan}]
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    with pytest.raises(ValueError) as caught:
        new(rows)
    assert str(caught.value) == _outcome(lambda r: _old_dumps(r, tagged), rows)[1]
    assert caught.value.__context__ is None
    # with nothing for the walk to find, json.dumps's own error is the one raised
    with pytest.raises(ValueError, match="Out of range float"):
        new([{"b": math.nan}])


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
def test_a_clean_large_batch_is_the_plain_json_text(tagged):
    rows = _batch(20_000, {})
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    assert new(rows) == json.dumps(rows)


def test_scalar_rows_are_not_walked_when_the_text_carries_them(monkeypatch):
    # the insert_many shape of the paper's l1tpc Part load and the create_vertices shape of l2, e2 and l3d: dict rows of str, int, float,
    # bool and None. Numbers that look alike but fit: 19-digit ints below 2**63, floats whose text has a long run of digits after the
    # point, text with digits, Hangul and CJK (BMP characters that json.dumps writes as \uXXXX, some of them as \ud...).
    walks = []
    real = tc._check_rows
    monkeypatch.setattr(
        tc,
        "_check_rows",
        lambda rows, tagged: walks.append(tagged) or real(rows, tagged),
    )
    rows = _batch(2_000, {})
    rows += [
        {
            "ts": 1_700_000_000_123_456_789,
            "hi": INT64_MAX,
            "lo": INT64_MIN + 1,
            "f": 0.00012345678901234568,
        },
        {
            "f": 1.2345678901234567e19,
            "s": "id-123456789012345678",
            "han": "한글퟿中文",
            "nul": "a\x00b",
        },
        {"e": "", "n": None, "b": True, "neg": -123456789012345678},
    ]
    assert tc.json_bulk_dumps(rows) == json.dumps(rows)
    assert tc.json_bulk_dumps_tagged(rows) == json.dumps(rows)
    assert walks == []


def test_tagged_kinds_are_not_walked_either(monkeypatch):
    walks = []
    real = tc._check_rows
    monkeypatch.setattr(
        tc,
        "_check_rows",
        lambda rows, tagged: walks.append(tagged) or real(rows, tagged),
    )
    rows = [
        {"d": dt.datetime(2026, 1, 2, 3, 4, 5), "day": dt.date(2026, 1, 2), "m": decimal.Decimal("1.50"), "b": b"ab", "n": np.int64(5), "f": np.float32(2.5)}
        for _ in range(100)
    ]  # fmt: skip
    assert tc.json_bulk_dumps_tagged(rows)
    assert walks == []


@pytest.mark.parametrize(
    "row",
    [
        {"a": 2**63},
        {"a": 10**20},
        {"a": INT64_MIN - 1},
        {"a": "a\ud800"},
        {"a": "\U0001f600"},
        {7: "x"},
        {"a": [1]},
        {"a": 5e-324},
        {"a": Colour.RED},
    ],
)
def test_a_value_that_may_be_bad_runs_the_walk(monkeypatch, row):
    walks = []
    real = tc._check_rows
    monkeypatch.setattr(
        tc,
        "_check_rows",
        lambda rows, tagged: walks.append(tagged) or real(rows, tagged),
    )
    _outcome(tc.json_bulk_dumps, [{"id": 1}, row])
    assert walks, "the walk must decide a row the text cannot vouch for"


def test_the_column_screen_flags_exactly_the_ints_beyond_64_bits():
    # two regular expressions over the JSON text used to decide this; the text scan of the integer tokens cost 2.5 microseconds a row
    # (12 ms of a 5,000-row create_vertices call of the paper's l2 shape), so the columns decide it now, in C, and the text is not scanned
    rnd = random.Random(63)  # nosec B311 - a seeded corpus, not security
    values = [
        0,
        1,
        10**18,
        2**63 - 2,
        2**63 - 1,
        2**63,
        2**63 + 1,
        10**19 - 1,
        10**19,
        10**20,
        9 * 10**18,
        9_999_999_999_999_999_999,
    ]
    values += [rnd.randint(10**18, 2**65) for _ in range(5000)]  # fmt: skip
    for value in values:
        for number in (value, -value):
            outside = not INT64_MIN <= number <= INT64_MAX
            assert tc._rows_need_walk([{"a": number}], False) is outside, number
            assert tc._rows_need_walk([{"a": None}, {"a": number}], False) is outside
            assert tc._scalar_column_suspect([number, None]) is outside, number
    # digits after the point of a float and inside an exponent are no integer token, and a float column is not screened at all
    assert tc._rows_need_walk([{"f": 0.00012345678901234568}], False) is False
    assert tc._rows_need_walk([{"f": 1.2345678901234567e19}], False) is False
    assert tc._rows_need_walk([{"s": "id-123456789012345678901"}], False) is False


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
def test_a_bad_value_in_a_ragged_or_nullable_column_runs_the_walk_and_gives_the_old_error(
    tagged,
):
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    for rows in (
        [{"a": 1}, {"b": "x"}, {"a": 2**70}],  # the key is missing in the middle row
        [{"a": None}, {"a": INT64_MIN - 1}],
        [{"s": "ok"}, {"s": None}, {"s": "x\ud800"}],
        [{"a": 1}, {"a": "text"}, {"a": 2**63}],  # a column of two kinds
    ):
        assert tc._rows_need_walk(rows, tagged) is True, rows
        expected = _outcome(lambda r: _old_dumps(r, tagged), rows)
        assert expected[0] == "ValueError", expected
        assert _outcome(new, rows) == expected


def test_a_nullable_column_of_a_tagged_kind_and_the_scalar_columns_are_not_walked(
    monkeypatch,
):
    walks = []
    real = tc._check_rows
    monkeypatch.setattr(
        tc,
        "_check_rows",
        lambda rows, tagged: walks.append(tagged) or real(rows, tagged),
    )
    rows = [
        {
            "id": i,
            "born": None if i % 3 == 0 else dt.datetime(2026, 1, 2, 3, 4, i % 60),
            "name": None if i % 5 == 0 else f"n{i}",
            "w": i / 7,
        }
        for i in range(300)
    ]
    assert tc._rows_need_walk(rows, True) is False
    assert tc.json_bulk_dumps_tagged(rows)
    assert walks == []
    # a column of a tagged kind beside an int: the walk judges each value
    assert tc._rows_need_walk([{"a": dt.date(2026, 1, 2)}, {"a": 5}], True) is True


def test_screen_vertex_rows_vets_a_table_of_exact_scalars(monkeypatch):
    rows = _batch(2_000, {})
    rows += [
        {
            "ts": 1_700_000_000_123_456_789,
            "hi": INT64_MAX,
            "lo": INT64_MIN,
            "f": 0.00012345678901234568,
        },
        {"han": "한글퟿中文", "nul": "a\x00b", "e": "", "n": None, "b": True},
        {"mixed": 1},
        {"mixed": 2.5},  # a column of two kinds: judged value by value, and it passes
        {},
    ]
    assert tc.screen_vertex_rows(rows, set()) == (False, True)
    # a vetted table is written with no check of its own: neither the type census nor the walk runs, and the text is the plain text
    calls = []
    for name in ("_rows_need_walk", "_check_rows"):
        real = getattr(tc, name)
        monkeypatch.setattr(
            tc, name, lambda *a, _n=name, _r=real: calls.append(_n) or _r(*a)
        )
    assert tc.json_bulk_dumps(rows, vetted=True) == json.dumps(rows)
    assert calls == []
    plain = rows[
        :2_000
    ]  # no column of two kinds: the type census and the column screen decide, no walk
    assert tc.json_bulk_dumps(plain) == json.dumps(plain)
    assert calls == ["_rows_need_walk"], "an unvetted call still looks at its rows"


def test_screen_vertex_rows_does_not_vet_a_column_of_another_type():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    born = dt.datetime(2026, 1, 2)
    assert tc.screen_vertex_rows([{"a": 1, "d": born}] * 3, set()) == (True, False)
    assert tc.screen_vertex_rows([{"a": 1, "l": ["x", 2]}] * 3, set()) == (True, False)
    # float32 arrays of an undeclared column, and the list of a declared one, are left to split_float_vectors
    assert tc.screen_vertex_rows([{"v": row} for row in x], set()) == (False, False)
    assert tc.screen_vertex_rows([{"v": row.tolist()} for row in x], {"v"}) == (
        False,
        False,
    )
    # an empty row (None counts as one) holds nothing to judge
    assert tc.screen_vertex_rows([{"a": 1}, None, {}], set()) == (False, True)


@pytest.mark.parametrize(
    "bad", [2**63, INT64_MIN - 1, math.nan, math.inf, 5e-324, "a\ud800", {1: 2}, {1, 2}]
)
def test_screen_vertex_rows_gives_none_for_what_the_text_would_change(bad):
    rows = [{"id": i, "x": 1.5} for i in range(300)]
    assert tc.screen_vertex_rows(rows, set()) == (False, True)
    rows[150]["x"] = bad
    assert tc.screen_vertex_rows(rows, set()) is None
    assert tc.classify_vertex_rows(rows, set(), True) is None
    assert tc.screen_vertex_rows([{"id": 1}, {1: "a"}], set()) is None


# ---------------------------------------------------------------------------
# the entry points, on whichever wheel is installed


def _types(db):
    db.command("sql", "CREATE DOCUMENT TYPE Part")
    db.command("sql", "CREATE VERTEX TYPE PartV")


def _stored(db, type_name):
    query = f"SELECT FROM {type_name}"  # nosec B608 - test-controlled
    rows = db.query("sql", query).to_list()
    return sorted(
        (
            r["p_partkey"],
            r["p_name"],
            r["p_size"],
            r["p_retailprice"],
            r.get("p_comment"),
        )
        for r in rows
    )


def _expected(rows):
    return sorted(
        (
            r["p_partkey"],
            r["p_name"],
            r["p_size"],
            r["p_retailprice"],
            r.get("p_comment"),
        )
        for r in rows
    )


def _rows_with_text():
    rows = _batch(25_000, {})
    rows[10]["p_name"] = "emoji \U0001f600 \U00010000"
    rows[11]["p_name"] = ""
    rows[12]["p_name"] = "한글 中文 ퟿"
    rows[13]["p_size"] = INT64_MAX
    rows[14]["p_size"] = INT64_MIN
    return rows


def test_insert_many_stores_a_scalar_batch_as_before(temp_db_path):
    rows = _rows_with_text()
    with arcadedb.create_database(temp_db_path) as db:
        _types(db)
        assert db.insert_many("Part", rows) == len(rows)
        assert _stored(db, "Part") == _expected(rows)


def test_create_vertices_stores_a_scalar_batch_as_before(temp_db_path):
    rows = _rows_with_text()
    with arcadedb.create_database(temp_db_path) as db:
        _types(db)
        with db.graph_batch(parallel_flush=False) as batch:
            assert len(batch.create_vertices("PartV", rows)) == len(rows)
        assert _stored(db, "PartV") == _expected(rows)


@pytest.mark.parametrize("entry", ["insert_many", "create_vertices"])
@pytest.mark.parametrize("bad", [2**63 + 11, "a\ud800"], ids=["int", "surrogate"])
def test_a_bad_row_inside_a_large_batch_is_refused(temp_db_path, entry, bad):
    column = "p_size" if isinstance(bad, int) else "p_name"
    rows = _batch(25_000, {12_345: {column: bad}, 19_000: {"p_size": 2**70 + 22}})
    with arcadedb.create_database(temp_db_path) as db:
        _types(db)
        with pytest.raises(
            Exception
        ):  # noqa: B017 - the entry points raise their own types
            if entry == "insert_many":
                db.insert_many("Part", rows)
            else:
                with db.graph_batch(parallel_flush=False) as batch:
                    batch.create_vertices("PartV", rows)


def test_the_scalar_screen_flags_what_the_text_would_change_and_passes_the_rest():
    # the whole-input screen of create_vertices: conservative, so a false alarm is allowed, a miss is not
    clean = {
        "str": ["a", "é中", "\U0001f600", "", None],
        "int": [1, -1, INT64_MAX, INT64_MIN, None],
        "float": [0.5, -0.0, 0.0, 1e-300, 1.7e308, None],
        "bool": [True, False, None],
        "nones": [None, None],
    }
    for name, column in clean.items():
        assert tc._scalar_column_suspect(list(column)) is False, name
    bad = {
        "surrogate": ["ok", "a\ud800b"],
        "above_max": [1, INT64_MAX + 1],
        "below_min": [INT64_MIN - 1, 1],
        "nan": [1.5, math.nan],
        "inf": [math.inf],
        "subnormal": [1.5, 5e-324],
        "mixed": [1, "a"],
        "int_float": [1, 2.5],
    }
    for name, column in bad.items():
        assert tc._scalar_column_suspect(list(column)) is True, name
    # the screen runs over blocks: a bad value in the last block of a large column is found
    column = [{"x": i} for i in range(250_000)]
    column[249_999] = {"x": INT64_MAX + 1}
    assert tc.classify_vertex_rows(column, set(), True) is None
    assert tc.classify_vertex_rows(column, set(), False) is False
