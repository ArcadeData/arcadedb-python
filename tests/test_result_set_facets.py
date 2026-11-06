"""The facets of a result set: ``ResultSet.facets`` of a ``SELECT ... FACETS`` query (ArcadeDB 26.12.1, issue #290).

The engine computes the facets next to the rows (``ResultSet.getFacets()``); both packages expose them as a plain dict:
``meta`` (``exact``, ``patched``, ``lag_ms``) and one entry per facet alias. This file is the proof for both: it runs on
``arcadedb-jvm`` and on the native ``arcadedb``, with only the public Python API.

The engine of a lane is whatever the lane links (see tests/test_columnar_engine_smoke.py): an engine older than 26.12.1 must
refuse the ``FACETS`` clause with its own syntax error and a plain query has no facets; a newer one must return them. The version
is read through the test's database (``SELECT version()``), so the same check runs on both builds and nothing is skipped.
"""

import math
from datetime import datetime

import pytest

from arcadedb import ArcadeDBError, create_database

FACETS_SINCE = (26, 12, 1)

CATEGORIES = ["a", "b", "a", "c", "a", "b", "a", "d", "a", "c"]
# a: 5, b: 2, c: 2, d: 1
CATEGORY_COUNTS = {"a": 5, "b": 2, "c": 2, "d": 1}
# ids 0-3 in January 2024, 4-6 in February, 7-9 in April: three calendar months with rows, March empty
MONTH_OF = [1, 1, 1, 1, 2, 2, 2, 4, 4, 4]

TERMS_Q = "SELECT id, category FROM FacetDoc ORDER BY id FACETS category TERMS TOP 2"
PAGED_Q = (
    "SELECT id FROM FacetDoc ORDER BY id FACETS category TERMS TOP 10 SKIP 2 LIMIT 3"
)
RANGES_Q = (
    "SELECT id FROM FacetDoc ORDER BY id "
    "FACETS s: score RANGES ([0, 50) AS 'low', [50, *) AS 'high')"
)
RANGES_DEFAULT_Q = (
    "SELECT id FROM FacetDoc ORDER BY id "
    "FACETS s: score RANGES ([0, 30) AS 'low') DEFAULT 'rest'"
)
HISTOGRAM_Q = (
    "SELECT id FROM FacetDoc ORDER BY id "
    "FACETS h: ts HISTOGRAM CALENDAR '1M' TIMEZONE 'UTC'"
)
FILTER_Q = (
    "SELECT id, category FROM FacetDoc ORDER BY id "
    "FACETS c: category TERMS TOP 10 FILTER category = 'a', "
    "s: score RANGES ([0, 50) AS 'low', [50, *) AS 'high')"
)
STATS_Q = (
    "SELECT id FROM FacetDoc ORDER BY id FACETS s: score STATS, d: category DISTINCT"
)
RANGES_NAMED_Q = (
    "SELECT id FROM FacetDoc ORDER BY id "
    "FACETS s: score RANGES ([:lo, :hi) AS 'mid')"
)
RANGES_POSITIONAL_Q = (
    "SELECT id FROM FacetDoc ORDER BY id FACETS s: score RANGES ([?, ?) AS 'mid')"
)
INCLUDE_NAMED_Q = (
    "SELECT id FROM FacetDoc ORDER BY id FACETS c: category TERMS INCLUDE [:x, :y]"
)
INCLUDE_POSITIONAL_Q = (
    "SELECT id FROM FacetDoc ORDER BY id FACETS c: category TERMS INCLUDE [?, ?]"
)


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db = create_database(str(tmp_path / "test_result_set_facets_db"))
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


def _older_engine_refuses(db, sql, *args):
    """True when the engine predates the ``FACETS`` clause: the statement is then refused with its syntax error, and the caller
    has nothing more to check on this engine (an engine without the clause reads FACETS as a table alias).
    """
    if _engine_version(db) >= FACETS_SINCE:
        return False
    with pytest.raises(Exception, match="SQL syntax error"):
        db.query("sql", sql, *args).to_list()
    return True


def _make_docs(db):
    db.command("sql", "CREATE DOCUMENT TYPE FacetDoc")
    db.command("sql", "CREATE PROPERTY FacetDoc.id INTEGER")
    db.command("sql", "CREATE PROPERTY FacetDoc.category STRING")
    db.command("sql", "CREATE PROPERTY FacetDoc.score INTEGER")
    db.command("sql", "CREATE PROPERTY FacetDoc.ts DATETIME")
    with db.transaction():
        for i, category in enumerate(CATEGORIES):
            db.command(
                "sql",
                "INSERT INTO FacetDoc SET id = ?, category = ?, score = ?, ts = ?",
                i,
                category,
                10 * (i + 1),
                datetime(2024, MONTH_OF[i], 10 + i, 12, 0, 0),
            )


def _index_docs(db):
    """The COLUMNAR indexes, on an engine that has them (the older branch of every test returned before this)."""
    for prop in ("category", "score", "ts"):
        db.command("sql", f"CREATE INDEX ON FacetDoc ({prop}) COLUMNAR")


def _docs(db):
    _make_docs(db)
    if _engine_version(db) >= FACETS_SINCE:
        _index_docs(db)


def _counts(facet):
    """A terms facet as {value: count}, the bucket of the records without a value left out."""
    return {b["value"]: b["count"] for b in facet["buckets"] if not b.get("missing")}


def _group_by_counts(db, where=""):
    rows = db.query(
        "sql",
        f"SELECT category, count(*) AS n FROM FacetDoc {where} GROUP BY category",  # nosec B608 - test-owned text
    ).to_list()
    return {r["category"]: r["n"] for r in rows}


def _same_float(actual, expected):
    assert isinstance(actual, float), actual
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


class TestResultSetFacets:
    def test_plain_query_has_no_facets(self, test_db):
        _make_docs(test_db)

        rs = test_db.query("sql", "SELECT FROM FacetDoc ORDER BY id")
        assert rs.facets is None
        # asking did not end the rows of a statement that has no facets
        assert [r["id"] for r in rs.to_list()] == list(range(10))
        assert rs.facets is None

        rs = test_db.command("sql", "SELECT FROM FacetDoc ORDER BY id")
        assert rs.facets is None
        assert len(rs.to_list()) == 10

    def test_facets_of_a_closed_result_set_raise(self, test_db):
        _make_docs(test_db)

        rs = test_db.query("sql", "SELECT FROM FacetDoc")
        rs.close()
        with pytest.raises(ArcadeDBError, match="closed before"):
            rs.facets

    def test_older_engine_refuses_the_clause(self, test_db):
        _make_docs(test_db)

        if _engine_version(test_db) >= FACETS_SINCE:
            # the newer branch of every test below: the clause is accepted
            rs = test_db.query("sql", TERMS_Q)
            assert len(rs.to_list()) == 10
            assert rs.facets is not None
            return
        assert _older_engine_refuses(test_db, TERMS_Q)

    def test_terms(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, TERMS_Q):
            return

        rs = test_db.query("sql", TERMS_Q)
        rows = rs.to_list()  # the rows first, then the facets
        facets = rs.facets

        assert [r["id"] for r in rows] == list(range(10))
        assert list(facets) == ["meta", "category"]
        meta = facets["meta"]
        assert set(meta) == {"exact", "patched", "lag_ms"}
        assert isinstance(meta["exact"], bool)
        assert isinstance(meta["patched"], int) and meta["patched"] >= 0
        assert isinstance(meta["lag_ms"], int) and meta["lag_ms"] >= 0

        category = facets["category"]
        assert category["type"] == "terms"
        assert category["source"] in ("index", "scan")
        assert "reason" not in category
        # TOP 2: the biggest value, then one of the two with two records; the counts of the rest are other_count
        assert category["buckets"][0] == {"value": "a", "count": 5}
        assert len(category["buckets"]) == 2
        assert category["buckets"][1]["count"] == 2
        assert category["other_count"] == 3
        assert category["missing"] == 0

    def test_facets_cover_every_match_whatever_skip_and_limit_kept(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, PAGED_Q):
            return

        rs = test_db.query("sql", PAGED_Q)
        rows = rs.to_list()
        facets = rs.facets

        assert [r["id"] for r in rows] == [2, 3, 4]
        assert _counts(facets["category"]) == CATEGORY_COUNTS

    def test_ranges(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, RANGES_Q):
            return

        rs = test_db.query("sql", RANGES_Q)
        rs.to_list()
        score = rs.facets["s"]

        assert score["type"] == "range"
        assert score["source"] in ("index", "scan")
        assert "other_count" not in score  # every interval is returned
        low, high = score["buckets"]
        assert (low["key"], low["from"], low["to"], low["count"]) == ("low", 0, 50, 4)
        # an unbounded end is left out
        assert (high["key"], high["from"], high["count"]) == ("high", 50, 6)
        assert "to" not in high
        assert score["missing"] == 0

    def test_ranges_default_bucket(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, RANGES_DEFAULT_Q):
            return

        rs = test_db.query("sql", RANGES_DEFAULT_Q)
        rs.to_list()
        buckets = rs.facets["s"]["buckets"]

        assert [(b["key"], b["count"]) for b in buckets] == [("low", 2), ("rest", 8)]
        assert buckets[1]["default"] is True
        assert "default" not in buckets[0]

    def test_calendar_histogram(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, HISTOGRAM_Q):
            return

        rs = test_db.query("sql", HISTOGRAM_Q)
        rs.to_list()
        hist = rs.facets["h"]

        assert hist["type"] == "histogram"
        assert hist["source"] in ("index", "scan")
        # one bucket per month, in time order (an empty month, March, may or may not be listed)
        full = [b for b in hist["buckets"] if b["count"]]
        assert [b["count"] for b in full] == [4, 3, 3]
        assert all(b["key"] and "from" in b and "to" in b for b in full)
        assert hist["missing"] == 0

    def test_disjunctive_filter(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, FILTER_Q):
            return

        rs = test_db.query("sql", FILTER_Q)
        rows = rs.to_list()
        facets = rs.facets

        # the rows keep only what every FILTER passes
        assert [r["id"] for r in rows] == [0, 2, 4, 6, 8]
        # the facet that owns the filter ignores it (disjunctive): every category is still counted
        assert _counts(facets["c"]) == CATEGORY_COUNTS
        # the other facets count what the filter passes: the five records of category a (scores 10, 30, 50, 70, 90)
        assert [(b["key"], b["count"]) for b in facets["s"]["buckets"]] == [
            ("low", 2),
            ("high", 3),
        ]

    def test_stats_and_distinct(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, STATS_Q):
            return

        rs = test_db.query("sql", STATS_Q)
        rs.to_list()
        facets = rs.facets

        stats = facets["s"]
        assert stats["type"] == "stats"
        assert "buckets" not in stats
        assert (stats["count"], stats["missing"]) == (10, 0)
        assert (stats["min"], stats["max"], stats["sum"]) == (10, 100, 550)
        assert stats["avg"] == 55
        distinct = facets["d"]
        assert distinct["type"] == "distinct"
        assert (distinct["value"], distinct["exact"]) == (4, True)

    @pytest.mark.parametrize(
        "name, values",
        [
            ("OddNan", [1.5, math.nan]),
            ("OddPositive", [1.5, math.inf]),
            ("OddNegative", [1.5, -math.inf]),
        ],
    )
    def test_non_finite_values_are_floats(self, test_db, name, values):
        sql = f"SELECT FROM {name} FACETS s: x STATS"  # nosec B608 - test-owned name
        test_db.command("sql", f"CREATE DOCUMENT TYPE {name}")
        test_db.command("sql", f"CREATE PROPERTY {name}.x DOUBLE")
        if _older_engine_refuses(test_db, sql):
            return
        test_db.command("sql", f"CREATE INDEX ON {name} (x) COLUMNAR")
        with test_db.transaction():
            for value in values:
                test_db.command("sql", f"INSERT INTO {name} SET x = :x", {"x": value})

        rs = test_db.query("sql", sql)
        assert len(rs.to_list()) == 2
        stats = rs.facets["s"]

        assert stats["type"] == "stats" and stats["count"] == 2
        # the engine's own JSON writers turn these into null: here they are the floats a row holds
        _same_float(stats["sum"], sum(values))
        _same_float(stats["avg"], sum(values) / 2)
        if math.isnan(values[1]):
            assert stats["min"] == 1.5
            assert isinstance(stats["max"], float)
        elif values[1] > 0:
            assert (stats["min"], stats["max"]) == (1.5, math.inf)
        else:
            assert (stats["min"], stats["max"]) == (-math.inf, 1.5)

    def test_named_parameters_in_ranges_and_include(self, test_db):
        _docs(test_db)
        params = {"lo": 30, "hi": 70}
        if _older_engine_refuses(test_db, RANGES_NAMED_Q, params):
            return

        for run in (test_db.query, test_db.command):
            rs = run("sql", RANGES_NAMED_Q, params)
            assert len(rs.to_list()) == 10
            (mid,) = rs.facets["s"]["buckets"]
            # [30, 70): 30, 40, 50 and 60
            assert (mid["key"], mid["from"], mid["to"], mid["count"]) == (
                "mid",
                30,
                70,
                4,
            )

            rs = run("sql", INCLUDE_NAMED_Q, {"x": "a", "y": "c"})
            rs.to_list()
            assert _counts(rs.facets["c"]) == {"a": 5, "c": 2}

    def test_positional_parameters_in_ranges_and_include(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, RANGES_POSITIONAL_Q, 30, 70):
            return

        for run in (test_db.query, test_db.command):
            rs = run("sql", RANGES_POSITIONAL_Q, 30, 70)
            rs.to_list()
            (mid,) = rs.facets["s"]["buckets"]
            assert (mid["key"], mid["count"]) == ("mid", 4)

            rs = run("sql", INCLUDE_POSITIONAL_Q, "a", "c")
            rs.to_list()
            assert _counts(rs.facets["c"]) == {"a": 5, "c": 2}

    def test_rows_first_then_facets(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, TERMS_Q):
            return

        rs = test_db.query("sql", TERMS_Q)
        seen = [row["id"] for row in rs]  # row by row, to the end
        facets = rs.facets

        assert seen == list(range(10))
        assert facets is not None and facets["category"]["buckets"][0]["count"] == 5
        # the same dict on every call, and the rows stay read
        assert rs.facets == facets
        assert rs.to_list() == []

    def test_facets_first_end_the_rows(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, TERMS_Q):
            return

        rs = test_db.query("sql", TERMS_Q)
        facets = rs.facets  # before any row is read

        assert facets["category"]["buckets"][0] == {"value": "a", "count": 5}
        # the engine computed them over every match and ended the rows
        assert rs.to_list() == []
        assert rs.facets == facets

    def test_facets_of_a_lazy_command_cursor(self, test_db):
        _docs(test_db)
        if _older_engine_refuses(test_db, TERMS_Q):
            return

        rs = test_db.command("sql", TERMS_Q)
        facets = rs.facets
        assert facets["category"]["buckets"][0] == {"value": "a", "count": 5}
        assert rs.to_list() == []

        rs = test_db.command("sql", TERMS_Q)
        assert len(rs.to_list()) == 10
        assert rs.facets == facets

    @pytest.mark.parametrize("run", ["query", "command"])
    @pytest.mark.parametrize("rows_first", [True, False])
    def test_facets_of_a_cursor_with_rows_left(self, test_db, run, rows_first):
        """More rows than the first batch of either build reads, so the cursor is still open when the facets are asked."""
        # no ORDER BY: the rows stream, so asking for the facets first has rows left to read past
        sql = "SELECT id FROM Big FACETS c: category TERMS TOP 5"
        test_db.command("sql", "CREATE DOCUMENT TYPE Big")
        test_db.command("sql", "CREATE PROPERTY Big.id INTEGER")
        test_db.command("sql", "CREATE PROPERTY Big.category STRING")
        if _older_engine_refuses(test_db, sql):
            return
        with test_db.transaction():
            for i in range(600):
                test_db.command(
                    "sql", "INSERT INTO Big SET id = ?, category = ?", i, f"x{i % 3}"
                )
        test_db.command("sql", "CREATE INDEX ON Big (category) COLUMNAR")

        rs = getattr(test_db, run)("sql", sql)
        if rows_first:
            assert sorted(r["id"] for r in rs.to_list()) == list(range(600))
            facets = rs.facets
        else:
            facets = rs.facets
            assert rs.to_list() == []

        assert _counts(facets["c"]) == {"x0": 200, "x1": 200, "x2": 200}

    def test_facets_equal_group_by(self, test_db):
        _docs(test_db)
        sql = "SELECT id FROM FacetDoc WHERE score >= 30 ORDER BY id FACETS category TERMS TOP 10"
        if _older_engine_refuses(test_db, sql):
            return

        rs = test_db.query("sql", TERMS_Q.replace("TOP 2", "TOP 10"))
        rs.to_list()
        assert _counts(rs.facets["category"]) == _group_by_counts(test_db)

        # with a WHERE: the facets cover the matches, as GROUP BY does
        rs = test_db.query("sql", sql)
        rows = rs.to_list()
        assert len(rows) == 8
        assert _counts(rs.facets["category"]) == _group_by_counts(
            test_db, "WHERE score >= 30"
        )

    def test_unavailable_facet_is_surfaced(self, test_db):
        """A HIERARCHY facet over a type whose subtype alone has a hierarchy: the records of the parent type have none, so the facet
        is unavailable and says why, while the rows and the other facets are answered (the engine's HierarchyFacetIntegrationTest).
        """
        sql = "SELECT FROM Media FACETS t: tags HIERARCHY TOP 100, n: tags TERMS TOP 5"
        test_db.command("sql", "CREATE VERTEX TYPE FacetTag")
        test_db.command("sql", "CREATE PROPERTY FacetTag.path STRING")
        test_db.command("sql", "CREATE INDEX ON FacetTag (path) NOTUNIQUE")
        test_db.command("sql", "CREATE EDGE TYPE FacetParentOf")
        test_db.command("sql", "CREATE DOCUMENT TYPE Media")
        test_db.command("sql", "CREATE PROPERTY Media.tags LIST OF STRING")
        test_db.command("sql", "CREATE DOCUMENT TYPE Photo EXTENDS Media")
        if _older_engine_refuses(test_db, sql):
            return
        test_db.command(
            "sql",
            "CREATE INDEX ON Photo (tags BY ITEM) COLUMNAR METADATA "
            '{"hierarchy": {"vertexType": "FacetTag", "key": "path", '
            '"parentEdge": "FacetParentOf"}}',
        )
        with test_db.transaction():
            test_db.command("sql", "CREATE VERTEX FacetTag SET path = 't0'")
            test_db.command("sql", "CREATE VERTEX FacetTag SET path = 't0/t1'")
            test_db.command(
                "sql",
                "CREATE EDGE FacetParentOf FROM (SELECT FROM FacetTag WHERE path = 't0') "
                "TO (SELECT FROM FacetTag WHERE path = 't0/t1')",
            )
            for _ in range(3):
                test_db.command("sql", "INSERT INTO Media SET tags = ['t0/t1']")
                test_db.command("sql", "INSERT INTO Photo SET tags = ['t0/t1']")

        rs = test_db.query("sql", sql)
        assert len(rs.to_list()) == 6
        facets = rs.facets

        hierarchy = facets["t"]
        assert hierarchy["type"] == "hierarchy"
        assert hierarchy["source"] == "unavailable"
        assert "Media.tags" in hierarchy["reason"]
        assert "buckets" not in hierarchy
        # the other facet of the same query is answered
        assert facets["n"]["type"] == "terms"
        assert facets["n"]["source"] in ("index", "scan")
        assert _counts(facets["n"]) == {"t0/t1": 6}
