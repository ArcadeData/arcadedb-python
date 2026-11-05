"""The records a ``find_nearest`` hit holds, on both backends: the same values as a lookup by RID, whichever way it is asked.

``find_nearest(..., include_vector=False)`` builds a hit without the indexed vector on the native wheel and fills it on first
use (``test_native_vector_hit_lazy.py`` counts that); the JVM wheel accepts the keyword and returns whole records. These tests are
the contract both builds keep: type, property names and their order, every value, the vector included, and a record that can be
modified and saved after a search. The default call (``include_vector=True``) is held to the same contract.
"""

import numpy as np
import pytest

import arcadedb

DIM = 8
N_DOC = 48
N_SUB = 16
BOTH = pytest.mark.parametrize("include_vector", [True, False])


def _make(temp_db, extra_properties, extra_columns, extra_values):
    temp_db.command("sql", "CREATE VERTEX TYPE Doc BUCKETS 4")
    temp_db.command("sql", "CREATE VERTEX TYPE SubDoc EXTENDS Doc")
    temp_db.command("sql", "CREATE PROPERTY Doc.id INTEGER")
    temp_db.command("sql", "CREATE PROPERTY Doc.label STRING")
    for statement in extra_properties:
        temp_db.command("sql", statement)
    temp_db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    rng = np.random.default_rng(11)
    vecs = rng.standard_normal((N_DOC + N_SUB, DIM)).astype(np.float32)
    with temp_db.transaction():
        for i in range(N_DOC + N_SUB):
            type_name = "Doc" if i < N_DOC else "SubDoc"
            temp_db.command(
                "sql",
                f"INSERT INTO {type_name} SET id = ?, label = ?{extra_columns}, embedding = ?",
                i,
                f"label-{i}",
                *extra_values(i),
                vecs[i].tolist(),
            )
    index = temp_db.create_vector_index(
        "Doc", "embedding", dimensions=DIM, id_property="id"
    )
    return temp_db, index, vecs


@pytest.fixture
def hit_db(temp_db):
    """Four buckets and a subtype."""
    return _make(temp_db, [], "", lambda i: ())


@pytest.fixture
def clash_db(temp_db):
    """As ``hit_db``, plus properties called ``distance`` and ``record`` (the names of the search row's own columns)."""
    return _make(
        temp_db,
        [
            "CREATE PROPERTY Doc.distance DOUBLE",
            "CREATE PROPERTY Doc.record STRING",
        ],
        ", distance = ?, record = ?",
        lambda i: (1000.5 + i, f"record-{i}"),
    )


def _assert_same_distances(got, expected):
    """Exactly on the native build; to a few float32 ulps on the JVM build, whose repeated calls can differ in the last float32 bit
    (the rule of ``_assert_same`` in ``test_find_nearest_query_forms.py``)."""
    if arcadedb.__backend__ == "native":
        assert got == expected
    else:
        assert got == pytest.approx(expected, rel=1e-6, abs=1e-9)


def _same_record(hit, ref):
    assert hit.get_rid() == ref.get_rid()
    assert hit.get_type_name() == ref.get_type_name()
    assert type(hit).__name__ == type(ref).__name__
    assert list(hit.get_property_names()) == list(ref.get_property_names())
    assert list(hit.to_dict()) == list(ref.to_dict())
    for name in ref.get_property_names():
        assert hit.has_property(name)
        assert hit.get(name) == ref.get(name), name


@BOTH
def test_hits_equal_a_lookup_by_rid(hit_db, include_vector):
    db, index, vecs = hit_db
    seen_types = set()
    for q in (0, 5, 40, 50, 60):
        hits = index.find_nearest(vecs[q], k=7, include_vector=include_vector)
        assert len(hits) == 7
        assert [d for _, d in hits] == sorted(d for _, d in hits)
        for record, score in hits:
            ref = db.lookup_by_rid(record.get_rid())
            seen_types.add(record.get_type_name())
            _same_record(record, ref)
    assert seen_types == {"Doc", "SubDoc"}


def test_the_keyword_changes_no_answer(hit_db):
    db, index, vecs = hit_db
    for q in (1, 20, 55):
        full = index.find_nearest(vecs[q], k=9)
        lean = index.find_nearest(vecs[q], k=9, include_vector=False)
        assert [r.get_rid() for r, _ in full] == [r.get_rid() for r, _ in lean]
        _assert_same_distances([d for _, d in full], [d for _, d in lean])
        assert [r.get("id") for r, _ in full] == [r.get("id") for r, _ in lean]
        for (a, _), (b, _) in zip(full, lean):
            assert list(a.get("embedding")) == list(b.get("embedding"))


def test_the_keyword_is_keyword_only_and_a_boolean_default(hit_db):
    db, index, vecs = hit_db
    with pytest.raises(TypeError):
        index.find_nearest(vecs[0], 3, None, None, False)
    assert index.find_nearest(vecs[0], k=3, include_vector=True)


def test_hits_without_the_vector_equal_a_lookup_when_properties_are_named_like_columns(
    clash_db,
):
    db, index, vecs = clash_db
    for q in (0, 5, 40, 50):
        for record, score in index.find_nearest(vecs[q], k=7, include_vector=False):
            ref = db.lookup_by_rid(record.get_rid())
            _same_record(record, ref)
            # the record's own distance property is not the search score
            assert record.get("distance") == pytest.approx(
                1000.5 + record.get("id"), rel=0, abs=1e-9
            )
            assert record.get("record") == f"record-{record.get('id')}"


@pytest.mark.parametrize("first", ["vector", "has", "names", "dict", "other"])
def test_first_access_does_not_change_what_the_record_holds(hit_db, first):
    db, index, vecs = hit_db
    record, _ = index.find_nearest(vecs[3], k=1, include_vector=False)[0]
    ref = db.lookup_by_rid(record.get_rid())
    if first == "vector":
        assert list(record.get("embedding")) == list(ref.get("embedding"))
    elif first == "has":
        assert record.has_property("embedding")
    elif first == "names":
        assert list(record.get_property_names()) == list(ref.get_property_names())
    elif first == "dict":
        assert list(record.to_dict()) == list(ref.to_dict())
    else:
        assert record.get("id") == ref.get("id")
    _same_record(record, ref)


@BOTH
def test_a_hit_can_be_modified_and_saved(hit_db, include_vector):
    db, index, vecs = hit_db
    record, _ = index.find_nearest(vecs[7], k=1, include_vector=include_vector)[0]
    rid = record.get_rid()
    before = list(db.lookup_by_rid(rid).get("embedding"))
    with db.transaction():
        record.modify().set("label", "changed").save()
    after = db.lookup_by_rid(rid)
    assert after.get("label") == "changed"
    assert list(after.get("embedding")) == before
    # the search still finds it, with the new label
    again, _ = index.find_nearest(vecs[7], k=1, include_vector=include_vector)[0]
    assert again.get_rid() == rid
    assert again.get("label") == "changed"


@BOTH
def test_options_and_by_key_hits_are_full_records(hit_db, include_vector):
    db, index, vecs = hit_db
    wanted = [r.get_rid() for r, _ in index.find_nearest(vecs[1], k=12)][3:9]
    many = [
        index.find_nearest(vecs[1], k=4, ef_search=64, include_vector=include_vector),
        index.find_nearest(
            vecs[1], k=4, allowed_rids=wanted, include_vector=include_vector
        ),
        index.find_nearest(
            vecs[1],
            k=4,
            ef_search=64,
            allowed_rids=wanted,
            include_vector=include_vector,
        ),
        index.find_nearest_by_key(1, k=4),
    ]
    for hits in many:
        assert len(hits) == 4
        for record, _ in hits:
            _same_record(record, db.lookup_by_rid(record.get_rid()))
    filtered = index.find_nearest(
        vecs[1], k=4, allowed_rids=wanted, include_vector=include_vector
    )
    assert {r.get_rid() for r, _ in filtered} <= set(wanted)
