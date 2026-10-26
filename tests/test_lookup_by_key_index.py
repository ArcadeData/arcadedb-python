"""lookup_by_key() needs an index on exactly those properties, on both backends.

The engine's lookupByKey answers from an index only. It raises when the type has none on the given property list (in that order), when
the one it has is a FULL_TEXT or vector index, or when the type does not exist. The JVM backend gets that from the engine. The
native backend used to run a ``SELECT ... WHERE`` instead, which scans the type without an index: the same call returned a record on
one wheel and raised on the other. Both raise the same message now, so a program does not depend on which wheel runs it.
"""

import pytest

from arcadedb import ArcadeDBError


@pytest.fixture
def docs(temp_db):
    """Type D with an unindexed integer ``k``, two more properties ``a`` and ``b``, and a text property ``t``; one record."""
    temp_db.command("sql", "CREATE DOCUMENT TYPE D")
    for prop, kind in (("k", "INTEGER"), ("a", "INTEGER"), ("b", "INTEGER")):
        temp_db.command("sql", f"CREATE PROPERTY D.{prop} {kind}")
    temp_db.command("sql", "CREATE PROPERTY D.t STRING")
    with temp_db.transaction():
        temp_db.command(
            "sql", "INSERT INTO D SET k = 1, a = 10, b = 20, t = 'hello world'"
        )
    return temp_db


def test_lookup_on_an_unindexed_key_raises(docs):
    with pytest.raises(
        ArcadeDBError,
        match=r"Failed to lookup by key in 'D'.*No index has been created on type 'D' properties \[k\]",
    ):
        docs.lookup_by_key("D", ["k"], [1])


def test_lookup_on_an_unindexed_key_raises_also_without_a_match(docs):
    with pytest.raises(ArcadeDBError, match="No index has been created"):
        docs.lookup_by_key("D", ["k"], [99])


@pytest.mark.parametrize(
    "index_kind", ["UNIQUE", "NOTUNIQUE", "UNIQUE_HASH", "NOTUNIQUE_HASH"]
)
def test_lookup_works_once_the_key_is_indexed(docs, index_kind):
    with pytest.raises(ArcadeDBError, match="No index has been created"):
        docs.lookup_by_key("D", ["k"], [1])
    docs.command("sql", f"CREATE INDEX ON D (k) {index_kind}")
    assert docs.lookup_by_key("D", ["k"], [1]).get("a") == 10
    assert docs.lookup_by_key("D", ["k"], [99]) is None


def test_lookup_raises_again_after_the_index_is_dropped(docs):
    docs.command("sql", "CREATE INDEX ON D (k) UNIQUE")
    assert docs.lookup_by_key("D", ["k"], [1]) is not None
    docs.command("sql", "DROP INDEX `D[k]`")
    with pytest.raises(ArcadeDBError, match="No index has been created"):
        docs.lookup_by_key("D", ["k"], [1])


def test_an_index_on_another_property_does_not_serve_the_lookup(docs):
    docs.command("sql", "CREATE INDEX ON D (a) UNIQUE")
    with pytest.raises(ArcadeDBError, match=r"properties \[k\]"):
        docs.lookup_by_key("D", ["k"], [1])


def test_a_composite_index_serves_exactly_its_property_list(docs):
    docs.command("sql", "CREATE INDEX ON D (a, b) NOTUNIQUE")
    assert docs.lookup_by_key("D", ["a", "b"], [10, 20]).get("k") == 1
    assert docs.lookup_by_key("D", ["a", "b"], [10, 99]) is None
    # a prefix of the index, and the same properties in the other order, are not that index
    with pytest.raises(ArcadeDBError, match=r"properties \[a\]"):
        docs.lookup_by_key("D", ["a"], [10])
    with pytest.raises(ArcadeDBError, match=r"properties \[b, a\]"):
        docs.lookup_by_key("D", ["b", "a"], [20, 10])


def test_a_full_text_index_does_not_serve_an_exact_key_lookup(docs):
    docs.command("sql", "CREATE INDEX ON D (t) FULL_TEXT")
    with pytest.raises(
        ArcadeDBError,
        match=r"No key index has been created on type 'D' properties \[t\].*FULL_TEXT",
    ):
        docs.lookup_by_key("D", ["t"], ["hello"])


def test_a_subtype_is_served_by_the_index_of_any_supertype(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE A")
    temp_db.command("sql", "CREATE PROPERTY A.k INTEGER")
    temp_db.command("sql", "CREATE DOCUMENT TYPE B EXTENDS A")
    temp_db.command("sql", "CREATE DOCUMENT TYPE C EXTENDS B")
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO C SET k = 3, name = 'deep'")
    with pytest.raises(ArcadeDBError, match="No index has been created"):
        temp_db.lookup_by_key("C", ["k"], [3])
    temp_db.command("sql", "CREATE INDEX ON A (k) UNIQUE")
    assert temp_db.lookup_by_key("C", ["k"], [3]).get("name") == "deep"
    assert temp_db.lookup_by_key("B", ["k"], [3]).get("name") == "deep"
    assert temp_db.lookup_by_key("B", ["k"], [4]) is None


def test_a_supertype_is_not_served_by_the_index_of_a_subtype(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE A")
    temp_db.command("sql", "CREATE PROPERTY A.k INTEGER")
    temp_db.command("sql", "CREATE DOCUMENT TYPE B EXTENDS A")
    temp_db.command("sql", "CREATE INDEX ON B (k) UNIQUE")
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO B SET k = 5, name = 'sub'")
    assert temp_db.lookup_by_key("B", ["k"], [5]).get("name") == "sub"
    with pytest.raises(ArcadeDBError, match="No index has been created"):
        temp_db.lookup_by_key("A", ["k"], [5])


@pytest.mark.parametrize("kind", ["VERTEX", "EDGE"])
def test_vertex_and_edge_types_follow_the_same_rule(temp_db, kind):
    temp_db.command("sql", f"CREATE {kind} TYPE G")
    temp_db.command("sql", "CREATE PROPERTY G.k INTEGER")
    with pytest.raises(ArcadeDBError, match="No index has been created"):
        temp_db.lookup_by_key("G", ["k"], [1])
    temp_db.command("sql", "CREATE INDEX ON G (k) UNIQUE")
    assert temp_db.lookup_by_key("G", ["k"], [1]) is None


def test_lookup_on_an_unknown_type_names_the_type_not_the_index(docs):
    with pytest.raises(ArcadeDBError, match="'Nope' was not found"):
        docs.lookup_by_key("Nope", ["k"], [1])
