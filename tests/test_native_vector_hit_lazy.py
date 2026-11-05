"""Native wheel: ``find_nearest`` keeps its default (hits carry the vector, no extra lookup) and, asked for ``include_vector=False``,
builds a hit without the indexed vector and fills it on first use, with one lookup by RID.

The values themselves are checked on both wheels in ``test_vector_hit_records.py``; this file counts what is shipped and looked up,
and which statement the search runs.
"""

import numpy as np
import pytest

from arcadedb.exceptions import ArcadeDBError

DIM = 8


@pytest.fixture
def lazy_db(temp_db, monkeypatch):
    temp_db.command("sql", "CREATE VERTEX TYPE Doc BUCKETS 4")
    temp_db.command("sql", "CREATE PROPERTY Doc.id INTEGER")
    temp_db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    rng = np.random.default_rng(5)
    vecs = rng.standard_normal((40, DIM)).astype(np.float32)
    with temp_db.transaction():
        for i in range(40):
            temp_db.command(
                "sql",
                "INSERT INTO Doc SET id = ?, name = ?, embedding = ?",
                i,
                f"n{i}",
                vecs[i].tolist(),
            )
    index = temp_db.create_vector_index(
        "Doc", "embedding", dimensions=DIM, id_property="id"
    )
    lookups = []
    original = type(temp_db).lookup_by_rid

    def counting(self, rid):
        lookups.append(str(rid))
        return original(self, rid)

    monkeypatch.setattr(type(temp_db), "lookup_by_rid", counting)
    statements = []
    original_query = type(temp_db).query

    def recording(self, language, text, *args, **kwargs):
        statements.append(text)
        return original_query(self, language, text, *args, **kwargs)

    monkeypatch.setattr(type(temp_db), "query", recording)
    return temp_db, index, vecs, lookups, statements


def _search_statements(statements):
    return [s for s in statements if "vectorNeighbors" in s]


def test_the_default_search_ships_the_vector_and_looks_up_nothing(lazy_db):
    db, index, vecs, lookups, statements = lazy_db
    hits = index.find_nearest(vecs[2], k=6)
    assert len(hits) == 6
    for record, _ in hits:
        assert "embedding" in record._props
        assert record._lazy is None
        assert len(record.get("embedding")) == DIM
        assert record.has_property("embedding")
        assert "embedding" in record.to_dict()
    assert lookups == []
    (text,) = _search_statements(statements)
    assert "!" not in text and "SELECT *" not in text


def test_asking_for_the_default_in_words_is_the_default(lazy_db):
    db, index, vecs, lookups, statements = lazy_db
    hits = index.find_nearest(vecs[2], k=3, include_vector=True)
    assert all("embedding" in record._props for record, _ in hits)
    assert lookups == []
    (text,) = _search_statements(statements)
    assert "!" not in text


def test_an_opt_in_search_ships_no_vector_and_looks_up_nothing(lazy_db):
    db, index, vecs, lookups, statements = lazy_db
    hits = index.find_nearest(vecs[2], k=6, include_vector=False)
    assert len(hits) == 6
    for record, _ in hits:
        assert "embedding" not in record._props
        assert record.get("id") is not None and record.get("name")
        assert record.get_rid() and record.get_type_name() == "Doc"
    assert lookups == []
    (text,) = _search_statements(statements)
    assert "!`embedding`" in text


def test_the_vector_is_one_lookup_per_record_however_it_is_reached(lazy_db):
    db, index, vecs, lookups, statements = lazy_db
    hits = index.find_nearest(vecs[2], k=4, include_vector=False)
    first, second, third, fourth = (r for r, _ in hits)
    assert len(first.get("embedding")) == DIM
    assert len(first.get("embedding")) == DIM
    assert first.has_property("embedding")
    assert "embedding" in first.to_dict()
    assert "embedding" in first.get_property_names()
    assert len(lookups) == 1
    assert second.has_property("embedding")
    assert len(lookups) == 2
    assert "embedding" in third.to_dict()
    assert "embedding" in fourth.get_property_names()
    assert len(lookups) == 4
    assert sorted(lookups) == sorted(r.get_rid() for r, _ in hits)


def test_the_loaded_vector_is_the_one_the_default_search_ships(lazy_db):
    db, index, vecs, lookups, statements = lazy_db
    full = {
        r.get_rid(): list(r.get("embedding"))
        for r, _ in index.find_nearest(vecs[9], k=5)
    }
    assert lookups == []
    lean = index.find_nearest(vecs[9], k=5, include_vector=False)
    assert lookups == []
    for record, _ in lean:
        assert list(record.get("embedding")) == full[record.get_rid()]
    assert len(lookups) == 5


def test_the_vector_of_a_closed_database_is_not_looked_up(lazy_db):
    db, index, vecs, lookups, statements = lazy_db
    record, _ = index.find_nearest(vecs[2], k=1, include_vector=False)[0]
    db.close()
    with pytest.raises(ArcadeDBError, match="closed"):
        record.get("embedding")
    assert lookups == []
