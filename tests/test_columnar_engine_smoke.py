"""Columnar faceting in the engine: the ``FACETS`` clause and ``CREATE INDEX ... COLUMNAR`` (ArcadeDB 26.12.1).

Both are new in engine 26.12.1 (ArcadeData/arcadedb pull requests #11338 to #11346). The engine of a lane is whatever the
lane links: Maven Central's 26.11.1 for the native pull request gate, the ``latest`` image for the JVM lane, a
``26.12.1-SNAPSHOT`` for a manual native run. The version is read through the test's database inside each test
(``SELECT version()``, so the same check runs on the JVM and the native build), and both engines are asserted on: an older
engine must refuse the new statements with its own error, a newer one must run them. The repository accepts no skips
(scripts/check_test_skips.py).

This is a smoke test: it checks that the rows of a ``FACETS`` query come back and that the statements do not raise. The facet
buckets themselves (``ResultSet.getFacets()``) have no Python API yet; the engine's own tests (``query/facet``) cover them.
On the native build it is also the first proof that the image knows the classes of the new parser and index code.
"""

import re

import pytest

from arcadedb import create_database

FACETS_SINCE = (26, 12, 1)

FACETS_QUERY = (
    "SELECT id, category FROM FacetDoc WHERE score >= 20 ORDER BY id "
    "FACETS category TERMS TOP 5, score TERMS LIMIT 3"
)


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db = create_database(str(tmp_path / "test_columnar_engine_smoke_db"))
    yield db
    db.drop()


def _engine_version(db):
    """The engine's x.y.z through SQL `SELECT version()`; a SNAPSHOT or a build suffix is dropped."""
    raw = str(db.query("sql", "SELECT version() AS v").first().get("v"))
    parts = [
        int(p) if p.isdigit() else 0
        for p in raw.split()[0].split("-")[0].split(".")[:3]
    ]
    return tuple(parts + [0] * (3 - len(parts)))


def _make_docs(db):
    db.command("sql", "CREATE DOCUMENT TYPE FacetDoc")
    db.command("sql", "CREATE PROPERTY FacetDoc.id INTEGER")
    db.command("sql", "CREATE PROPERTY FacetDoc.category STRING")
    db.command("sql", "CREATE PROPERTY FacetDoc.score INTEGER")
    categories = ["a", "b", "a", "c", "a"]
    with db.transaction():
        for i, category in enumerate(categories):
            db.command(
                "sql",
                "INSERT INTO FacetDoc SET id = ?, category = ?, score = ?",
                i,
                category,
                10 * (i + 1),
            )


class TestColumnarEngineSmoke:
    def test_facets_clause_returns_the_rows(self, test_db):
        _make_docs(test_db)

        if _engine_version(test_db) < FACETS_SINCE:
            # An engine without the clause reads FACETS as a table alias and fails on the facet that follows it.
            with pytest.raises(Exception, match="SQL syntax error"):
                test_db.query("sql", FACETS_QUERY).to_list()
            return

        test_db.command("sql", "CREATE INDEX ON FacetDoc (category) COLUMNAR")
        test_db.command("sql", "CREATE INDEX ON FacetDoc (score) COLUMNAR")

        rows = test_db.query("sql", FACETS_QUERY).to_list()

        # the rows are the page of the WHERE matches (ids 1 to 4, LIMIT 3); the facets cover all four matches
        assert [r["id"] for r in rows] == [1, 2, 3]
        assert [r["category"] for r in rows] == ["b", "a", "c"]

    def test_columnar_index_create_and_drop(self, test_db):
        _make_docs(test_db)

        if _engine_version(test_db) < FACETS_SINCE:
            with pytest.raises(
                Exception, match=re.escape("Index type 'COLUMNAR' is not supported")
            ):
                test_db.command("sql", "CREATE INDEX ON FacetDoc (category) COLUMNAR")
            return

        test_db.command("sql", "CREATE INDEX ON FacetDoc (category) COLUMNAR")
        # the rows of schema:indexes are the same on both builds; the message names them if the index is not listed as expected
        rows = test_db.query("sql", "SELECT FROM schema:indexes").to_list()
        names = sorted(
            (r.get("name"), r.get("indexType"), r.get("typeName")) for r in rows
        )
        assert any(r.get("typeName") == "FacetDoc" for r in rows), names
        assert "FacetDoc[category]" in [r.get("name") for r in rows], names
        assert test_db.schema.exists_index("FacetDoc[category]")

        test_db.command("sql", "DROP INDEX `FacetDoc[category]`")
        assert not test_db.schema.exists_index("FacetDoc[category]")
