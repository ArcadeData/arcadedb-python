"""The forms a ``find_nearest`` query can take, on both backends: every numeric form answers the same neighbours at the same
distances, and a query that is not a numeric vector of the index's dimension is still refused.

The native backend hands the library a float32 array for a numeric query (``test_native_find_nearest_params.py`` pins how it
travels); these tests are the contract that change must keep, and they pass unchanged on the code before it.
"""

import numpy as np
import pytest

import arcadedb
from arcadedb import ArcadeDBError

DIM = 768
N = 60


@pytest.fixture
def index(temp_db_path):
    db = arcadedb.create_database(temp_db_path)
    db.command("sql", "CREATE VERTEX TYPE Doc")
    db.command("sql", "CREATE PROPERTY Doc.id INTEGER")
    db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    vectors = np.random.default_rng(7).standard_normal((N, DIM)).astype(np.float32)
    with db.transaction():
        for i in range(N):
            db.command(
                "sql", "INSERT INTO Doc SET id = ?, embedding = ?", i, vectors[i]
            )
    idx = db.create_vector_index("Doc", "embedding", dimensions=DIM)
    idx.vectors = vectors
    yield idx
    db.drop()


def _answer(hits):
    return [(record.get("id"), distance) for record, distance in hits]


def test_every_form_of_the_query_returns_the_same_neighbours_at_the_same_distances(
    index,
):
    query = index.vectors[11]
    expected = _answer(index.find_nearest(query.tolist(), k=10))
    assert len(expected) == 10
    assert expected[0][0] == 11
    forms = {
        "float32 array": query,
        "float64 array": query.astype(np.float64),
        "tuple": tuple(query.tolist()),
        "list of numpy floats": list(query),
        "generator": (float(x) for x in query),
        "non-contiguous view": np.repeat(query, 2)[::2],
    }
    for name, form in forms.items():
        assert _answer(index.find_nearest(form, k=10)) == expected, name


def test_a_float64_query_that_float32_would_round_answers_like_its_float32_form(
    index,
):
    # not float32-exact: the values float32 rounds are the ones that left the native list path as JSON text
    query = np.random.default_rng(3).standard_normal(DIM)
    assert not np.array_equal(query.astype(np.float32).astype(np.float64), query)
    expected = _answer(index.find_nearest(query.astype(np.float32), k=10))
    assert _answer(index.find_nearest(query, k=10)) == expected
    assert _answer(index.find_nearest(query.tolist(), k=10)) == expected


def test_an_integer_array_is_a_numeric_query(index):
    ints = np.arange(DIM) % 5
    expected = _answer(index.find_nearest(ints.tolist(), k=5))
    assert _answer(index.find_nearest(ints, k=5)) == expected


def test_find_nearest_by_key_agrees_with_the_array_query(index):
    by_key = _answer(index.find_nearest_by_key(4, k=6))
    assert by_key == _answer(index.find_nearest(index.vectors[4], k=6))


def test_filter_and_ef_search_options_still_apply_to_an_array_query(index):
    everything = index.find_nearest(index.vectors[2], k=10)
    allowed = [record.get_rid() for record, _ in everything[:3]]
    hits = index.find_nearest(index.vectors[2], k=10, allowed_rids=allowed)
    assert {record.get_rid() for record, _ in hits} == set(allowed)
    assert _answer(index.find_nearest(index.vectors[2], k=10, ef_search=64))


def test_a_query_of_the_wrong_dimension_is_refused_in_both_forms(index):
    with pytest.raises(ArcadeDBError):
        index.find_nearest(np.zeros(5, dtype=np.float32), k=3)
    with pytest.raises(ArcadeDBError):
        index.find_nearest([0.0] * 5, k=3)


@pytest.mark.parametrize(
    "bad", [[None] * DIM, ["x"] * DIM], ids=["None entries", "text entries"]
)
def test_a_query_that_is_not_numeric_is_refused(index, bad):
    # numpy would turn None into nan; the query must reach the engine as it was given
    with pytest.raises(ArcadeDBError):
        index.find_nearest(bad, k=3)
