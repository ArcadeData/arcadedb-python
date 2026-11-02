"""``find_nearest`` hands the native library its query vector as one float32 array, not as a Python list of floats.

``find_nearest`` used to turn every query into a list with ``tolist()``. The codec then checked the list for float32-exactness
(a pass of numpy over the list, ``_exact_float_list``) and sent it in the side buffer as tag ``BL``, or, for a float64 query whose
values float32 rounds, printed one float repr per element as JSON text (611 us at 1536 dimensions). A numpy array travels through
the float32 side buffer directly (tag ``B``: a pointer and a count, read by the library straight from the C buffer). Whether
that is faster is a timed A/B, not these tests; they count what the codec does and where the floats travel.

The answers are pinned for both backends in ``test_find_nearest_query_forms.py``. This file needs the native library, so
conftest.py does not collect it on the JVM backend.
"""

import arcadedb
import numpy as np
import pytest
from arcadedb._native import codec
from arcadedb._native import database as native_database

DIM = 768
N = 40


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


def _search_encodings(monkeypatch):
    """Records what ``encode_params`` returns for every statement; the search is the one carrying DIM floats beside its text."""
    seen = []
    real = native_database.encode_params

    def recording(args, binary=False):
        encoded = real(args, binary)
        seen.append(encoded)
        return encoded

    monkeypatch.setattr(native_database, "encode_params", recording)
    return seen


def _searches(seen):
    return [text for text, _, floats in seen if floats == DIM]


def test_an_array_query_travels_in_the_float32_side_buffer(index, monkeypatch):
    seen = _search_encodings(monkeypatch)
    index.find_nearest(index.vectors[3], k=5)
    search = _searches(seen)
    assert len(search) == 1
    assert '"B"' in search[0]
    assert '"BL"' not in search[0]


@pytest.mark.parametrize(
    "form",
    ["float64 array", "float64 list", "float32 list"],
)
def test_every_numeric_form_travels_in_the_side_buffer_as_an_array(
    index, monkeypatch, form
):
    exact = index.vectors[3]
    rounding = np.random.default_rng(5).standard_normal(DIM)  # float32 would round it
    query = {
        "float64 array": rounding,
        "float64 list": rounding.tolist(),
        "float32 list": exact.tolist(),
    }[form]
    seen = _search_encodings(monkeypatch)
    index.find_nearest(query, k=5)
    search = _searches(seen)
    # before: the float64 forms were JSON text (no side buffer at all), the float32 list tag BL
    assert len(search) == 1
    assert '"B"' in search[0]
    assert '"BL"' not in search[0]


def test_a_numeric_query_is_not_walked_for_float32_exactness(index, monkeypatch):
    calls = []
    real = codec._exact_float_list

    def counting(value):
        calls.append(len(value))
        return real(value)

    monkeypatch.setattr(codec, "_exact_float_list", counting)
    index.find_nearest(index.vectors[3], k=5)
    index.find_nearest(index.vectors[3].tolist(), k=5)
    index.find_nearest(index.vectors[3].astype(np.float64), k=5)
    assert [n for n in calls if n >= DIM] == []


def test_a_query_that_is_not_numeric_stays_a_list(index, monkeypatch):
    seen = _search_encodings(monkeypatch)
    with pytest.raises(arcadedb.ArcadeDBError):
        index.find_nearest([None] * DIM, k=3)
    assert _searches(seen) == []
