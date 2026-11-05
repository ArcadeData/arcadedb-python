"""A query vector passed as a Python list or a float64 numpy array crosses into the engine as one buffer, and no result changes.

The cost was in the conversion, not in the query: a 1536-number list became 1536 boxed objects on the JVM build (505 us) or
34 KB of JSON number text on the native build (611 us) before a 0.3 to 0.75 ms query started. A float32 array already crossed as
one buffer (7 us). The two builds now send an all-float list (and, on the JVM build, a float64 array) as one buffer of the same
numbers, and the engine reads the same ``List<Double>`` it read before. Everything that is not unambiguously such a list keeps its
old path: an ``int`` stays an int, a short list stays a short list, a number float32 would round stays exact.

The codec tests state the bytes the native build hands over; the statement tests state that every use of a list parameter
(``IN``, a list property, equality, ``vectorNeighbors``) gives the result it gave. ``test_query_vector_params_jvm.py`` states the
Java objects of the JVM build.
"""

import json
import math

import numpy as np
import pytest

from arcadedb._native.codec import encode_params

DIM = 96
BASE = 120
NN = "SELECT id FROM (SELECT expand(vectorNeighbors('Vec[embedding]', {q}, 10)))"


def _f32_exact_list(seed, n=DIM):
    """The list a float32 model hands over: ``ndarray.tolist()`` of float32 numbers, so float32 holds every one exactly."""
    return np.random.default_rng(seed).random(n).astype(np.float32).tolist()


def _f64_list(seed, n=DIM):
    """Doubles float32 would round (``random()`` is a 53-bit fraction)."""
    return np.random.default_rng(seed).random(n).tolist()


def _repr(values):
    """Value and type of every element: ``0.0 == -0.0`` and ``1 == 1.0`` in Python, so == alone cannot see a change."""
    return [(type(v).__name__, repr(v)) for v in values]


# ---------------------------------------------------------------------------
# the bytes the native build hands over (pure functions of the parameters: they run on both builds)


class TestNativeCodec:
    def test_an_exact_float_list_goes_in_the_float32_side_buffer(self):
        values = _f32_exact_list(1)
        text, buf, count = encode_params(([values],), binary=True)
        assert json.loads(text) == {"pos": [{"\u0001": ["BL", 0, DIM]}]}
        assert count == DIM and buf == np.asarray(values, dtype=np.float32).tobytes()
        assert len(text) + len(buf) < 500  # was 1536 numbers of text

    def test_a_named_parameter_and_a_nested_one_too(self):
        values = _f32_exact_list(2)
        text, buf, count = encode_params(({"q": values, "k": 5},), binary=True)
        assert json.loads(text) == {"named": {"q": {"\u0001": ["BL", 0, DIM]}, "k": 5}}
        assert count == DIM and buf is not None
        text, buf, count = encode_params(({"o": {"q": values}},), binary=True)
        assert json.loads(text)["named"]["o"]["q"] == {"\u0001": ["BL", 0, DIM]}

    def test_offsets_count_every_array_in_the_call(self):
        first, second = _f32_exact_list(3), _f32_exact_list(4, n=70)
        array = np.arange(10, dtype=np.float32)
        text, buf, count = encode_params((first, array, second), binary=True)
        assert json.loads(text)["pos"] == [
            {"\u0001": ["BL", 0, DIM]},
            {"\u0001": ["B", DIM, 10]},
            {"\u0001": ["BL", DIM + 10, 70]},
        ]
        assert count == DIM + 10 + 70
        assert (
            buf == np.concatenate([first, array, second]).astype(np.float32).tobytes()
        )

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(_f64_list(5), id="doubles-float32-would-round"),
            pytest.param([0.1] * DIM, id="point-one"),
            pytest.param(_f32_exact_list(6)[:-1] + [float("nan")], id="nan"),
            pytest.param(_f32_exact_list(6)[:-1] + [float("inf")], id="infinity"),
            pytest.param(_f32_exact_list(6)[:-1] + [1e300], id="beyond-float32"),
            pytest.param(_f32_exact_list(6)[:-1] + [5e-324], id="subnormal-double"),
            pytest.param(_f32_exact_list(7, n=63), id="short"),
            pytest.param(_f32_exact_list(7)[:-1] + [1], id="an-int-among-floats"),
            pytest.param(_f32_exact_list(7)[:-1] + [True], id="a-bool-among-floats"),
            pytest.param(_f32_exact_list(7)[:-1] + [None], id="a-null-among-floats"),
            pytest.param(_f32_exact_list(7)[:-1] + ["a"], id="a-str-among-floats"),
            pytest.param(list(range(DIM)), id="ints"),
            pytest.param(list(np.float32(1) * np.arange(DIM)), id="numpy-scalars"),
        ],
    )
    def test_anything_else_stays_json_text_as_before(self, values):
        text, buf, count = encode_params(([values],), binary=True)
        assert buf is None and count == 0
        # the same text a plain json.dumps writes: no value changes on its way
        assert text == json.dumps({"pos": [values]}, default=_numpy_default)

    def test_the_lists_inside_a_list_cross_on_their_own(self):
        inner = _f32_exact_list(12)
        text, buf, count = encode_params(([[inner, [1, 2], inner]],), binary=True)
        assert json.loads(text)["pos"] == [
            [{"\u0001": ["BL", 0, DIM]}, [1, 2], {"\u0001": ["BL", DIM, DIM]}]
        ]
        assert count == 2 * DIM

    def test_a_tuple_stays_json_text(self):
        text, buf, _ = encode_params(([tuple(_f32_exact_list(8))],), binary=True)
        assert buf is None

    def test_without_the_side_buffer_the_text_is_unchanged(self):
        values = _f32_exact_list(9)
        assert encode_params(([values],)) == json.dumps({"pos": [values]})
        assert encode_params(({"q": values},)) == json.dumps({"named": {"q": values}})

    def test_the_buffer_holds_back_exactly_the_list(self):
        values = _f32_exact_list(10)
        _, buf, count = encode_params(([values],), binary=True)
        back = np.frombuffer(buf, dtype=np.float32, count=count).astype(np.float64)
        assert (
            back.tolist() == values
        )  # float32 -> double is the identity on these numbers

    def test_negative_zero_keeps_its_sign(self):
        values = _f32_exact_list(11)[:-1] + [-0.0]
        _, buf, count = encode_params(([values],), binary=True)
        back = np.frombuffer(buf, dtype=np.float32, count=count).tolist()
        assert math.copysign(1.0, back[-1]) == -1.0


def _numpy_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


# ---------------------------------------------------------------------------
# what a statement does with the list: every use gives the result it gave


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE Vec")
    temp_db.command("sql", "CREATE PROPERTY Vec.embedding ARRAY_OF_FLOATS")
    temp_db.command("sql", "CREATE DOCUMENT TYPE Item")
    with temp_db.transaction():
        for i in range(BASE):
            temp_db.command(
                "sql",
                "CREATE VERTEX Vec SET id = ?, embedding = ?",
                i,
                np.random.default_rng(1000 + i).random(DIM).astype(np.float32),
            )
        temp_db.command("sql", "INSERT INTO Item SET k = 'a', x = 0.25")
        temp_db.command("sql", "INSERT INTO Item SET k = 'b', x = 0.1")
        temp_db.command("sql", "INSERT INTO Item SET k = 'c', x = 0.3")
        temp_db.command("sql", "INSERT INTO Item SET k = 'd', x = 7")
    temp_db.command(
        "sql",
        'CREATE INDEX ON Vec (embedding) LSM_VECTOR METADATA {"dimensions": %d, "similarity": "EUCLIDEAN"}'
        % DIM,
    )
    return temp_db


def _ids(db, sql, *args):
    return [r["id"] for r in db.query("sql", sql, *args).to_list()]


class TestStatements:
    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(_f32_exact_list, id="float32-exact"),
            pytest.param(_f64_list, id="doubles"),
        ],
    )
    @pytest.mark.parametrize("n", [3, 40, DIM, 500])
    def test_a_float_list_comes_back_value_for_value(self, db, make, n):
        values = make(21, n=n)
        for params in ({"v": values},):
            row = db.query("sql", "SELECT :v AS v", params).to_list()[0]
            assert _repr(row["v"]) == _repr(values)
        row = db.query("sql", "SELECT ? AS v", [values]).to_list()[0]
        assert _repr(row["v"]) == _repr(values)

    def test_the_awkward_numbers_come_back_exact(self, db):
        values = _f32_exact_list(22)[:-4] + [-0.0, 1e300, 5e-324, 0.1]
        row = db.query("sql", "SELECT :v AS v", {"v": values}).to_list()[0]
        assert _repr(row["v"]) == _repr(values)

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(list(range(DIM)), id="ints"),
            pytest.param([i + 0.5 if i % 2 else i for i in range(DIM)], id="mixed"),
            pytest.param([i % 2 == 0 for i in range(DIM)], id="bools"),
            pytest.param([None if i % 3 else 0.5 for i in range(DIM)], id="nulls"),
            pytest.param([str(i) for i in range(DIM)], id="strings"),
        ],
    )
    def test_a_list_that_is_not_all_floats_keeps_every_type(self, db, values):
        row = db.query("sql", "SELECT :v AS v", {"v": values}).to_list()[0]
        assert _repr(row["v"]) == _repr(values)

    def test_a_nested_list_keeps_its_shape(self, db):
        values = [_f32_exact_list(23), _f64_list(24), [1, 2, 3]]
        row = db.query("sql", "SELECT :v AS v", {"v": values}).to_list()[0]
        assert [_repr(v) for v in row["v"]] == [_repr(v) for v in values]

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(_f32_exact_list, id="float32-exact"),
            pytest.param(_f64_list, id="doubles"),
        ],
    )
    def test_in_matches_the_numbers_it_matched(self, db, make):
        numbers = make(25, n=100)
        numbers[0], numbers[1] = 0.25, 0.1  # the stored items a and b
        for param in ({"xs": numbers},):
            rows = db.query(
                "sql", "SELECT k FROM Item WHERE x IN :xs ORDER BY k", param
            )
            assert [r["k"] for r in rows.to_list()] == ["a", "b"]
        # the int the list does not hold is not found by a float list
        rows = db.query("sql", "SELECT k FROM Item WHERE x IN :xs", {"xs": numbers})
        assert "d" not in [r["k"] for r in rows.to_list()]

    def test_an_int_list_still_finds_an_int(self, db):
        rows = db.query(
            "sql", "SELECT k FROM Item WHERE x IN :xs", {"xs": list(range(100))}
        )
        assert [r["k"] for r in rows.to_list()] == ["d"]

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(_f32_exact_list, id="float32-exact"),
            pytest.param(_f64_list, id="doubles"),
        ],
    )
    def test_a_list_property_stores_and_equals_the_list(self, db, make):
        values = make(26)
        with db.transaction():
            db.command("sql", "INSERT INTO Item SET k = 'p', vals = :v", {"v": values})
            db.command("sql", "INSERT INTO Item SET k = 'q', vals = ?", [values])
        for key in ("p", "q"):
            stored = db.query("sql", "SELECT vals FROM Item WHERE k = ?", key).to_list()
            assert _repr(stored[0]["vals"]) == _repr(values)
        found = db.query(
            "sql", "SELECT k FROM Item WHERE vals = :v ORDER BY k", {"v": values}
        )
        assert [r["k"] for r in found.to_list()] == ["p", "q"]

    def test_contains_finds_a_number_of_the_list(self, db):
        values = _f64_list(27)
        with db.transaction():
            db.command("sql", "INSERT INTO Item SET k = 'p', vals = :v", {"v": values})
        rows = db.query(
            "sql", "SELECT k FROM Item WHERE vals CONTAINS :x", {"x": values[7]}
        ).to_list()
        assert [r["k"] for r in rows] == ["p"]

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(_f32_exact_list, id="float32-exact"),
            pytest.param(_f64_list, id="doubles"),
        ],
    )
    def test_vector_neighbors_agree_across_every_input_form(self, db, make):
        values = make(28)
        sql = "SELECT id FROM (SELECT expand(vectorNeighbors('Vec[embedding]', ?, 10)))"
        named = (
            "SELECT id FROM (SELECT expand(vectorNeighbors('Vec[embedding]', :q, 10)))"
        )
        want = _ids(db, sql, np.asarray(values, dtype=np.float32))
        assert len(want) == 10
        assert _ids(db, sql, [values]) == want  # a list
        assert (
            _ids(db, sql, np.asarray(values, dtype=np.float64)) == want
        )  # float64 array
        assert _ids(db, named, {"q": values}) == want
        assert _ids(db, named, {"q": np.asarray(values, dtype=np.float64)}) == want
        assert _ids(db, named, {"q": np.asarray(values, dtype=np.float32)}) == want
        assert _ids(db, NN.format(q=values)) == want  # the literal text, no parameter
