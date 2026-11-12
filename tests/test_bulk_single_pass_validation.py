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
    # point, text with digits, Hangul and CJK (BMP characters that json.dumps writes as \uXXXX, some of them as \ud...), and non-BMP
    # characters (an emoji): json.dumps writes each as a surrogate PAIR (\ud83d\ude00), which the old text scan took for a lone surrogate
    # and sent to the walk, a false alarm; UTF-8 encodes them, so the column screen passes them and the old loop accepted them too.
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
        {
            "emoji": "\U0001f600",
            "plane1": "\U00010000x\U0010ffff",
            "mixed": "a\U0001f600b",
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


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
@pytest.mark.parametrize(
    "text",
    ["\U0001f600", "\U00010000", "\U0010ffff", "a\U0001f600b", "\ue000\U0001f600"],
)
def test_a_non_bmp_character_is_no_bad_value_and_is_not_walked(
    monkeypatch, text, tagged
):
    # the row the old text scan sent to the walk (a surrogate pair in the text) and the old loop accepted: same text, no walk
    walks = []
    real = tc._check_rows
    monkeypatch.setattr(
        tc,
        "_check_rows",
        lambda rows, tagged: walks.append(tagged) or real(rows, tagged),
    )
    rows = [{"id": 1}, {"a": text}]
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    assert _outcome(new, rows) == ("text", json.dumps(rows))
    assert _outcome(lambda r: _old_dumps(r, tagged), rows) == ("text", json.dumps(rows))
    assert walks == []
    # two lone surrogates next to each other are no pair: UTF-8 cannot encode them, the walk refuses them
    bad = [{"id": 1}, {"a": "\ud83d\ude00"}]
    got = _outcome(new, bad)
    assert got == _outcome(lambda r: _old_dumps(r, tagged), bad)
    assert got[0] == "ValueError" and walks == [tagged]


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
    assert tc.screen_vertex_rows([{"a": "\U0001f600"}], set()) == (False, True)
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
    "bad", [2**63, INT64_MIN - 1, math.nan, math.inf, 5e-324, "a\ud800", {1, 2}]
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


# ---------------------------------------------------------------------------
# #371: the step before json.dumps is no slower than the 26.11.1 loop, with numpy and without it. The screens of a float column no longer
# need numpy, the text is searched for a subnormal exponent by a substring first, the column screens run over blocks of rows, and rows with no
# list, tuple or dict are written without the encoder's cycle bookkeeping. None of it may change a text or an error.

CLEAN_FLOAT_COLUMNS = {
    "plain": [0.5, -3.25, 1 / 3, 1e5],
    "zeros": [0.0, -0.0, 0.0],
    "all_zero": [0.0] * 5,
    "tiny_normal": [1e-300, 2.2250738585072014e-308, -2.2250738585072014e-308],
    "huge": [1.7976931348623157e308, 2.5],
    "huge_negative": [-1.7976931348623157e308],
    "with_none": [None, 1.5, None],
}
BAD_FLOAT_COLUMNS = {
    "nan": [1.5, math.nan, 2.5],
    "inf": [math.inf],
    "minus_inf": [1.5, -math.inf],
    "smallest_subnormal": [1.5, 5e-324],
    "largest_subnormal": [2.2250738585072009e-308, 1.5],
    "negative_subnormal": [-5e-324, 0.0, 2.5],
    "subnormal_among_zeros": [0.0, 0.0, 5e-324, 0.0],
    "subnormal_only": [5e-324],
}


@pytest.mark.parametrize("numpy_present", [True, False], ids=["numpy", "no_numpy"])
def test_the_float_screen_gives_the_same_verdict_with_and_without_numpy(
    monkeypatch, numpy_present
):
    if not numpy_present:
        monkeypatch.setattr(tc, "_np", None)
    for name, column in CLEAN_FLOAT_COLUMNS.items():
        assert tc._scalar_column_suspect(list(column)) is False, name
        rows = [{"id": i, "x": value} for i, value in enumerate(column)]
        assert tc.screen_vertex_rows(rows, set()) == (False, True), name
    for name, column in BAD_FLOAT_COLUMNS.items():
        assert tc._scalar_column_suspect(list(column)) is True, name
        rows = [{"id": i, "x": value} for i, value in enumerate(column)]
        assert tc.screen_vertex_rows(rows, set()) is None, name
        assert tc.classify_vertex_rows(rows, set(), True) is None, name
    # two finite values whose sum overflows: the screen may flag them, and the judgement by value then passes them
    overflow = [{"x": 1.7976931348623157e308}, {"x": 1.7976931348623157e308}]
    assert tc.screen_vertex_rows(overflow, set()) == (False, True)


@pytest.mark.parametrize("numpy_present", [True, False], ids=["numpy", "no_numpy"])
def test_a_clean_float_column_is_not_judged_value_by_value(monkeypatch, numpy_present):
    # without numpy the old screen gave up on every float column and judged it by a function call per value
    if not numpy_present:
        monkeypatch.setattr(tc, "_np", None)

    def refuse(value):
        raise AssertionError("a per-value gate ran over a clean float column")

    monkeypatch.setattr(tc, "json_bulk_scalar_ok", refuse)
    monkeypatch.setattr(tc, "json_bulk_shape_ok", refuse)
    rows = [
        {"id": i, "x": i * 0.5 + 0.25, "y": 0.0 if i % 9 == 0 else i / 3.0}
        for i in range(5_000)
    ]
    assert tc.screen_vertex_rows(rows, set()) == (False, True)


def test_the_subnormal_search_runs_the_expression_only_when_the_text_has_e_minus_3(
    monkeypatch,
):
    calls = []
    real = tc._SUBNORMAL_EXPONENT

    class Spy:
        def search(self, text):
            calls.append(text[:20])
            return real.search(text)

    monkeypatch.setattr(tc, "_SUBNORMAL_EXPONENT", Spy())
    walks = []
    real_has = tc._has_subnormal
    monkeypatch.setattr(tc, "_has_subnormal", lambda v: walks.append(1) or real_has(v))
    plain = _batch(200, {})
    assert tc.json_bulk_dumps(plain) == json.dumps(plain)
    assert calls == [], "a text without 'e-3' is not searched by the expression"
    # ordinary text with the substring reaches the expression and is turned back by it: no walk over the rows
    named = [{"id": i, "name": f"node-3{i}", "w": 1.5} for i in range(50)]
    assert tc.json_bulk_dumps(named) == json.dumps(named)
    assert len(calls) == 1 and walks == []
    # a subnormal is still refused, with the message it always had
    got = _outcome(tc.json_bulk_dumps, [{"id": 1}, {"x": 5e-324}])
    assert got == _outcome(lambda r: _old_dumps(r, False), [{"id": 1}, {"x": 5e-324}])
    assert got[0] == "ValueError" and "subnormal" in got[1]


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
@pytest.mark.parametrize(
    "column, bad",
    [
        ("id", 2**63),
        ("id", INT64_MIN - 1),
        ("s", "a\ud800b"),
        ("x", 5e-324),
        ("x", math.nan),
        ("x", {1, 2}),
        ("x", [2**70]),
    ],
    ids=["above", "below", "surrogate", "subnormal", "nan", "set", "list_big"],
)
@pytest.mark.parametrize("where", [0, 12, 13, 51, 99], ids=lambda w: f"row{w}")
def test_a_bad_value_is_found_in_any_block_of_the_column_screens(
    monkeypatch, tagged, column, bad, where
):
    # 40 values a block of 3 columns is 13 rows: the rows 12 and 13 are the last of one block and the first of the next
    monkeypatch.setattr(tc, "_SCREEN_CELLS", 40)
    monkeypatch.setattr(tc, "_SCREEN_MIN_ROWS", 4)
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    rows = [{"id": i, "x": i + 0.5, "s": f"n{i}"} for i in range(100)]
    assert _outcome(new, rows) == ("text", json.dumps(rows))
    rows[where][column] = bad
    got = _outcome(new, rows)
    assert got == _outcome(lambda r: _old_dumps(r, tagged), rows)
    assert got[0] in ("ValueError", "TypeError"), got


def test_the_column_screens_over_blocks_cover_ragged_rows_none_rows_and_late_keys(
    monkeypatch,
):
    monkeypatch.setattr(tc, "_SCREEN_CELLS", 40)
    monkeypatch.setattr(tc, "_SCREEN_MIN_ROWS", 4)
    rows = [{"id": i, "x": i * 0.25} for i in range(60)]
    rows[0] = None
    rows[1] = {}
    assert tc.screen_vertex_rows(rows, set()) == (False, True)
    # a key that first appears in the last block, a non-str key in the last block, a bad int in a column of a late key
    late = rows + [{"id": 60, "late": 7}, {"id": 61, "late": 2**63}]
    assert tc.screen_vertex_rows(late, set()) is None
    assert tc.screen_vertex_rows(rows + [{"id": 60, 5: "x"}], set()) is None
    assert tc.screen_vertex_rows(rows + [{"id": 60, "late": "a\ud800"}], set()) is None
    clean_late = rows + [{"id": 60, "late": 7}, {"id": 61, "late": -7}]
    assert tc.screen_vertex_rows(clean_late, set()) == (False, True)
    # the same rows through json_bulk_dumps: text equal to json.dumps, or the old error
    base = [row for row in rows if row]
    ragged = base + [{"id": 60, "late": 7}, {"id": 61, "late": -7}]
    assert tc.json_bulk_dumps(ragged) == json.dumps(ragged)
    for bad in (
        base + [{"id": 60, "late": 7}, {"id": 61, "late": 2**63}],
        base + [{5: "x"}],
    ):
        got = _outcome(tc.json_bulk_dumps, bad)
        assert got == _outcome(lambda r: _old_dumps(r, False), bad)
        assert got[0] == "ValueError"


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
def test_rows_without_a_container_are_written_without_the_cycle_check(
    monkeypatch, tagged
):
    seen = []
    real = json.dumps

    def spy(obj, *args, **kwargs):
        seen.append(kwargs.get("check_circular", True))
        return real(obj, *args, **kwargs)

    monkeypatch.setattr(tc.json, "dumps", spy)
    new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
    scalars = _batch(300, {})
    assert new(scalars) == real(scalars)
    assert seen == [False]
    seen.clear()
    assert tc.json_bulk_dumps(scalars, vetted=True) == real(scalars)
    assert seen == [False]
    seen.clear()
    # a list, a tuple or a dict is a container that may hold itself: the encoder keeps its check, and the walk comes first
    nested = [{"id": 1, "l": [1, 2, [3]]}, {"id": 2, "d": {"k": [1]}}]
    assert new(nested) == real(nested)
    assert seen == [True]


@pytest.mark.parametrize("tagged", [False, True], ids=["untagged", "tagged"])
def test_a_row_that_holds_itself_still_raises_what_it_raised(tagged):
    cycle = []
    cycle.append(cycle)
    for rows in ([{"id": 1, "l": cycle}], [{"id": 1, "d": {"again": cycle}}]):
        new = tc.json_bulk_dumps_tagged if tagged else tc.json_bulk_dumps
        got = _outcome(new, rows)
        expected = _outcome(lambda r: _old_dumps(r, tagged), rows)
        assert got[0] == expected[0] == "RecursionError"


def _digest(rows):
    import hashlib

    return hashlib.sha256(
        "\n".join(repr(row) for row in sorted(rows)).encode("utf-8")
    ).hexdigest()


def _mixed_rows(n, subnormal_at=None):
    # every kind of scalar the screens judge, in a table that spans several blocks of the column screens
    rows = []
    for i in range(n):
        rows.append(
            {
                "k": i,
                "x": [0.5, 0.0, 1e-300, 1.7e308, i / 7.0][i % 5],
                "n": [INT64_MAX, INT64_MIN, 0, 1_700_000_000_123_456_789, -i][i % 5],
                "s": ["plain", "한글 中文", "\U0001f600 pair", "", f"user-{i}"][i % 5],
                "b": i % 2 == 0,
                "z": None if i % 4 == 0 else "value",
            }
        )
    if subnormal_at is not None:
        rows[subnormal_at]["x"] = 5e-324
    return rows


def _stored_digest(db, type_name):
    query = f"SELECT FROM {type_name}"  # nosec B608 - test-controlled
    got = db.query("sql", query).to_list()
    return len(got), _digest(
        [(r["k"], r["x"], r["n"], r["s"], r["b"], r.get("z")) for r in got]
    )


@pytest.mark.parametrize("subnormal_at", [None, 7_000], ids=["clean", "one_subnormal"])
@pytest.mark.parametrize("entry", ["insert_many", "create_vertices"])
def test_a_scalar_batch_keeps_its_count_and_digest_and_survives_a_reopen(
    temp_db_path, entry, subnormal_at
):
    rows = _mixed_rows(12_000, subnormal_at)
    want = (len(rows), _digest([(r["k"], r["x"], r["n"], r["s"], r["b"], r["z"]) for r in rows]))  # fmt: skip
    type_name = "Part" if entry == "insert_many" else "PartV"
    with arcadedb.create_database(temp_db_path) as db:
        _types(db)
        if entry == "insert_many":
            assert db.insert_many("Part", rows) == len(rows)
        else:
            with db.graph_batch(parallel_flush=False) as batch:
                assert len(batch.create_vertices("PartV", rows)) == len(rows)
        assert _stored_digest(db, type_name) == want
    with arcadedb.open_database(temp_db_path) as db:
        assert _stored_digest(db, type_name) == want
