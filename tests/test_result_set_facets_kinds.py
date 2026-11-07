"""The facet kinds beyond terms and ranges, from Python: HIERARCHY, DISTINCT, STATS, AFTER and ``facet.searchValues`` (ArcadeDB 26.12.1, issue #291).

``ResultSet.facets`` (tests/test_result_set_facets.py, issue #290) already carries these kinds in the dict shape its docstring
documents; this file is the proof, for both packages and through the public API only, that every kind decodes to the same Python
values as a row holds and that its parameters, its paging cursor and the typeahead function work. It runs on ``arcadedb-jvm`` and
on the native ``arcadedb``.

The engine of a lane is whatever the lane links: an engine older than 26.12.1 must refuse the ``FACETS`` clause with its own syntax
error, a newer one must answer. The version is read through the test's database (``SELECT version()``), so the same check runs on
both builds and nothing is skipped.

What the engine takes as a parameter in a facet: the bounds of ``RANGES`` and ``HISTOGRAM``, the values of ``INCLUDE``, the cursor
of ``AFTER`` and the condition of ``FILTER`` (named ``:x`` and positional ``?``); ``TOP``, ``MIN_COUNT`` and ``PRECISION`` take an
integer literal only.

The facets cover the records of a lagging projection by patching them, so a facet query needs no wait. ``facet.searchValues`` reads
the dictionaries the projection has applied, so its tests retry until the projection caught up (a hang detector, never a latency
bound).
"""

import math
import time
from datetime import date, datetime

import pytest

from arcadedb import create_database

FACETS_SINCE = (26, 12, 1)

# a nobody-waits-forever limit for the projection to apply, and for a paging loop
CATCH_UP_SECONDS = 120.0
PAGES_MAX = 100

# ten records: ids 0-9
KIND_NAMES = [
    "apple",
    "apricot",
    "avocado",
    "banana",
    "blueberry",
    "cherry",
    "apple",
    "banana",
    "apple",
    "date",
]
# apple 3, banana 2, the other five names one each: seven distinct names
KIND_NAME_COUNTS = {
    "apple": 3,
    "apricot": 1,
    "avocado": 1,
    "banana": 2,
    "blueberry": 1,
    "cherry": 1,
    "date": 1,
}
# six records with tags, four without any
KIND_TAGS = [
    ["run", "rum"],
    ["runner"],
    ["running", "run"],
    ["rum"],
    ["run"],
    ["other"],
    None,
    None,
    None,
    None,
]
KIND_TAG_COUNTS = {"other": 1, "rum": 2, "run": 3, "runner": 1, "running": 1}

NAME_Q = "SELECT id FROM FacetKind ORDER BY id FACETS name TERMS TOP 2"

STATS_Q = (
    "SELECT id FROM FacetKind ORDER BY id "
    "FACETS s: qty STATS, r: ratio STATS, p: price STATS"
)
STATS_TEMPORAL_Q = (
    "SELECT id FROM FacetKind ORDER BY id FACETS t: ts STATS, d: day STATS"
)
DISTINCT_Q = (
    "SELECT id FROM FacetKind ORDER BY id "
    "FACETS n: name DISTINCT, q: qty DISTINCT, g: tags DISTINCT"
)
# PRECISION n keeps the exact set of a numeric column while it holds at most n values and switches to a HyperLogLog (exact False)
# at the (n + 1)th (DistinctFacetCollector: exactSize() > precisionThreshold): price has ten values
APPROXIMATE_Q = (
    "SELECT id FROM FacetKind ORDER BY id FACETS q: price DISTINCT PRECISION 5"
)
BELOW_THRESHOLD_Q = (
    "SELECT id FROM FacetKind ORDER BY id FACETS q: price DISTINCT PRECISION 10"
)
# the filter of the STATS facet narrows the rows and the DISTINCT facet, the filter of the DISTINCT facet the rows and the STATS one
STATS_FILTER_NAMED_Q = (
    "SELECT id FROM FacetKind ORDER BY id "
    "FACETS d: name DISTINCT, s: qty STATS FILTER qty >= :lo"
)
STATS_FILTER_POSITIONAL_Q = (
    "SELECT id FROM FacetKind ORDER BY id "
    "FACETS d: name DISTINCT, s: qty STATS FILTER qty >= ?"
)
DISTINCT_FILTER_NAMED_Q = (
    "SELECT id FROM FacetKind ORDER BY id "
    "FACETS d: name DISTINCT FILTER qty >= :lo, s: qty STATS"
)
DISTINCT_FILTER_POSITIONAL_Q = (
    "SELECT id FROM FacetKind ORDER BY id "
    "FACETS d: name DISTINCT FILTER qty >= ?, s: qty STATS"
)

HIERARCHY_Q = "SELECT id FROM FacetTagged ORDER BY id FACETS t: tags HIERARCHY TOP 100"
HIERARCHY_TOP_Q = (
    "SELECT id FROM FacetTagged ORDER BY id FACETS t: tags HIERARCHY TOP 2"
)
HIERARCHY_FILTER_NAMED_Q = (
    "SELECT id FROM FacetTagged ORDER BY id "
    "FACETS t: tags HIERARCHY TOP 100, f: n STATS FILTER n >= :lo"
)
HIERARCHY_FILTER_POSITIONAL_Q = (
    "SELECT id FROM FacetTagged ORDER BY id "
    "FACETS t: tags HIERARCHY TOP 100, f: n STATS FILTER n >= ?"
)

SEARCH_NAMED_Q = "SELECT expand(facet.searchValues(:t, :p, :x, :l))"
SEARCH_POSITIONAL_Q = "SELECT expand(facet.searchValues(?, ?, ?, ?))"


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db = create_database(str(tmp_path / "test_result_set_facets_kinds_db"))
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


def _make_kind(db):
    """Ten records of ``FacetKind``: a string, an integer, a float, a double, a date, a datetime and a list of strings (on six)."""
    db.command("sql", "CREATE DOCUMENT TYPE FacetKind")
    db.command("sql", "CREATE PROPERTY FacetKind.id INTEGER")
    db.command("sql", "CREATE PROPERTY FacetKind.name STRING")
    db.command("sql", "CREATE PROPERTY FacetKind.qty INTEGER")
    db.command("sql", "CREATE PROPERTY FacetKind.ratio FLOAT")
    db.command("sql", "CREATE PROPERTY FacetKind.price DOUBLE")
    db.command("sql", "CREATE PROPERTY FacetKind.day DATE")
    db.command("sql", "CREATE PROPERTY FacetKind.ts DATETIME")
    db.command("sql", "CREATE PROPERTY FacetKind.tags LIST OF STRING")
    with db.transaction():
        for i, name in enumerate(KIND_NAMES):
            sql = (
                "INSERT INTO FacetKind SET id = ?, name = ?, qty = ?, ratio = ?, "
                "price = ?, day = ?, ts = ?"
            )
            args = [
                i,
                name,
                10 * (i + 1),
                0.25
                * (i + 1),  # every value a binary fraction: sums and averages are exact
                0.5 * (i + 1),
                date(2024, 3, 1 + i),
                datetime(2024, 1, 1 + i, 12, 0, 0),
            ]
            if KIND_TAGS[i] is not None:
                sql += ", tags = ?"
                args.append(KIND_TAGS[i])
            db.command("sql", sql, *args)


def _index_kind(db):
    """The COLUMNAR indexes, on an engine that has them (the older branch of every test returned before this)."""
    for prop in ("name", "qty", "ratio", "price", "day", "ts"):
        db.command("sql", f"CREATE INDEX ON FacetKind ({prop}) COLUMNAR")
    db.command("sql", "CREATE INDEX ON FacetKind (tags BY ITEM) COLUMNAR")


def _kind(db):
    _make_kind(db)
    if _engine_version(db) >= FACETS_SINCE:
        _index_kind(db)


def _make_pg(db):
    """The records ``FacetPg`` pages over: eight values of ``name``, ``num`` (LONG) and ``stamp`` (DATETIME), the value i held by
    i % 3 + 1 records (15 in all), and three records without any of them."""
    db.command("sql", "CREATE DOCUMENT TYPE FacetPg")
    db.command("sql", "CREATE PROPERTY FacetPg.id INTEGER")
    db.command("sql", "CREATE PROPERTY FacetPg.name STRING")
    db.command("sql", "CREATE PROPERTY FacetPg.num LONG")
    db.command("sql", "CREATE PROPERTY FacetPg.stamp DATETIME")
    with db.transaction():
        row = 0
        for i in range(8):
            for _ in range(i % 3 + 1):
                db.command(
                    "sql",
                    "INSERT INTO FacetPg SET id = ?, name = ?, num = ?, stamp = ?",
                    row,
                    chr(ord("a") + i),
                    (i - 3) * 100,
                    datetime(2024, 1, 1 + i, 6, 30, 0),
                )
                row += 1
        for _ in range(3):
            db.command("sql", "INSERT INTO FacetPg SET id = ?", row)
            row += 1


def _index_pg(db):
    for prop in ("name", "num", "stamp"):
        db.command("sql", f"CREATE INDEX ON FacetPg ({prop}) COLUMNAR")


def _pg(db):
    _make_pg(db)
    if _engine_version(db) >= FACETS_SINCE:
        _index_pg(db)


def _make_tagged(db):
    """Five records whose tags live in a hierarchy of tag vertices: t0 holds t0/a (which holds t0/a/x) and t0/b."""
    db.command("sql", "CREATE VERTEX TYPE KindTag")
    db.command("sql", "CREATE PROPERTY KindTag.path STRING")
    db.command("sql", "CREATE INDEX ON KindTag (path) NOTUNIQUE")
    db.command("sql", "CREATE EDGE TYPE KindParentOf")
    db.command("sql", "CREATE DOCUMENT TYPE FacetTagged")
    db.command("sql", "CREATE PROPERTY FacetTagged.id INTEGER")
    db.command("sql", "CREATE PROPERTY FacetTagged.n INTEGER")
    db.command("sql", "CREATE PROPERTY FacetTagged.tags LIST OF STRING")


def _fill_tagged(db):
    """The hierarchy index (after the schema it names), the tag vertices and their edges, and the records."""
    db.command("sql", "CREATE INDEX ON FacetTagged (n) COLUMNAR")
    db.command(
        "sql",
        "CREATE INDEX ON FacetTagged (tags BY ITEM) COLUMNAR METADATA "
        '{"hierarchy": {"vertexType": "KindTag", "key": "path", '
        '"parentEdge": "KindParentOf"}}',
    )
    with db.transaction():
        for path in ("t0", "t0/a", "t0/a/x", "t0/b"):
            db.command("sql", "CREATE VERTEX KindTag SET path = ?", path)
        for parent, child in (("t0", "t0/a"), ("t0", "t0/b"), ("t0/a", "t0/a/x")):
            db.command(
                "sql",
                "CREATE EDGE KindParentOf FROM (SELECT FROM KindTag WHERE path = ?) "
                "TO (SELECT FROM KindTag WHERE path = ?)",
                parent,
                child,
            )
        records = [
            (1, ["t0/a/x"]),
            (2, ["t0/a"]),
            (3, ["t0/b", "t0/a/x"]),
            (4, ["orphan"]),  # a value no tag vertex has
            (5, None),  # no value at all
        ]
        for n, tags in records:
            if tags is None:
                db.command("sql", "INSERT INTO FacetTagged SET id = ?, n = ?", n, n)
            else:
                db.command(
                    "sql",
                    "INSERT INTO FacetTagged SET id = ?, n = ?, tags = ?",
                    n,
                    n,
                    tags,
                )


def _tagged(db):
    _make_tagged(db)
    if _engine_version(db) >= FACETS_SINCE:
        _fill_tagged(db)


def _facets(db, sql, *args, run="query"):
    """The rows of a facet query (read first, as the docstring of ``ResultSet.facets`` advises) and its facets."""
    rs = getattr(db, run)("sql", sql, *args)
    rows = rs.to_list()
    return rows, rs.facets


def _counts(facet):
    """A terms facet as {value: count}, the bucket of the records without a value left out."""
    return {b["value"]: b["count"] for b in facet["buckets"] if not b.get("missing")}


def _same_float(actual, expected):
    assert isinstance(actual, float), actual
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def _until(read, expected, timeout=CATCH_UP_SECONDS):
    """``read()`` until it returns ``expected`` (the projection applies in the background), the last answer after ``timeout``.

    A retry message the engine raises while a projection is not published yet is retried too, and raised when it never goes.
    """
    deadline = time.monotonic() + timeout
    answer = None
    error = None
    while True:
        try:
            answer = read()
            error = None
            if answer == expected:
                return answer
        except Exception as exc:  # the engine asks to retry while a column builds
            error = exc
        if time.monotonic() > deadline:
            if error is not None:
                raise error
            return answer
        time.sleep(0.05)


class TestStats:
    def test_columnar_indexes_are_listed(self, test_db):
        _kind(test_db)
        if _engine_version(test_db) < FACETS_SINCE:
            assert _older_engine_refuses(test_db, STATS_Q)
            return

        rows = test_db.query("sql", "SELECT FROM schema:indexes").to_list()
        names = [r.get("name") for r in rows]
        for prop in ("name", "qty", "ratio", "price", "day", "ts"):
            assert f"FacetKind[{prop}]_columnar" in names, names
        assert any(
            n.startswith("FacetKind[tags") and n.endswith("_columnar") for n in names
        ), names

    def test_numeric_stats(self, test_db):
        _kind(test_db)
        if _older_engine_refuses(test_db, STATS_Q):
            return

        rows, facets = _facets(test_db, STATS_Q)

        assert [r["id"] for r in rows] == list(range(10))
        assert list(facets) == ["meta", "s", "r", "p"]  # the order of the clause
        qty = facets["s"]
        assert qty["type"] == "stats" and qty["source"] in ("index", "scan")
        assert "buckets" not in qty and "reason" not in qty
        assert (qty["count"], qty["missing"]) == (10, 0)
        assert (qty["min"], qty["max"], qty["sum"]) == (10, 100, 550)
        assert qty["avg"] == 55
        assert isinstance(qty["min"], int) and isinstance(qty["sum"], int)
        # a FLOAT column: the extremes and the sum are floats, the average one too (the engine's own types, a Python float each)
        ratio = facets["r"]
        assert (ratio["count"], ratio["missing"]) == (10, 0)
        assert (ratio["min"], ratio["max"]) == (0.25, 2.5)
        assert (ratio["sum"], ratio["avg"]) == (13.75, 1.375)
        for member in ("min", "max", "sum", "avg"):
            assert isinstance(ratio[member], float), member
        price = facets["p"]
        assert (price["min"], price["max"]) == (0.5, 5.0)
        assert (price["sum"], price["avg"]) == (27.5, 2.75)
        for member in ("min", "max", "sum", "avg"):
            assert isinstance(price[member], float), member

    def test_temporal_stats_are_the_values_a_row_holds(self, test_db):
        _kind(test_db)
        if _older_engine_refuses(test_db, STATS_TEMPORAL_Q):
            return

        held = test_db.query("sql", "SELECT ts, day FROM FacetKind").to_list()
        stamps = [r["ts"] for r in held]
        days = [r["day"] for r in held]
        rows, facets = _facets(test_db, STATS_TEMPORAL_Q)

        assert len(rows) == 10
        for alias, values in (("t", stamps), ("d", days)):
            stats = facets[alias]
            assert stats["type"] == "stats", alias
            assert (stats["count"], stats["missing"]) == (10, 0), alias
            assert stats["min"] == min(values), alias
            assert stats["max"] == max(values), alias
            # the same Python type a row of the column holds (a date stays a date, a datetime a datetime)
            assert type(stats["min"]) is type(values[0]), alias
            assert type(stats["max"]) is type(values[0]), alias
            # there is no sum or average of dates
            assert stats["sum"] is None and stats["avg"] is None, alias

    def test_stats_of_no_match(self, test_db):
        _kind(test_db)
        sql = "SELECT id FROM FacetKind WHERE qty < 0 FACETS s: qty STATS"
        if _older_engine_refuses(test_db, sql):
            return

        rows, facets = _facets(test_db, sql)

        assert rows == []
        stats = facets["s"]
        assert stats["type"] == "stats"
        assert stats["count"] == 0
        assert stats["min"] is None and stats["max"] is None and stats["avg"] is None

    def test_stats_count_the_records_without_a_value(self, test_db):
        _pg(test_db)
        sql = "SELECT id FROM FacetPg ORDER BY id FACETS s: num STATS"
        if _older_engine_refuses(test_db, sql):
            return

        held = [
            r["num"]
            for r in test_db.query("sql", "SELECT num FROM FacetPg").to_list()
            if r.get("num") is not None
        ]
        rows, facets = _facets(test_db, sql)

        assert len(rows) == 18
        stats = facets["s"]
        assert len(held) == 15
        assert (stats["count"], stats["missing"]) == (15, 3)
        assert (stats["min"], stats["max"], stats["sum"]) == (
            min(held),
            max(held),
            sum(held),
        )

    @pytest.mark.parametrize(
        "name, values",
        [
            ("OddNan", [1.5, math.nan]),
            ("OddPositive", [1.5, math.inf]),
            ("OddNegative", [1.5, -math.inf]),
        ],
    )
    def test_non_finite_extremes_equal_sql(self, test_db, name, values):
        """MIN, MAX, SUM and AVG of a column holding NaN or an infinity: the floats the same aggregate of SQL returns."""
        sql = f"SELECT FROM {name} FACETS s: x STATS"  # nosec B608 - test-owned name
        test_db.command("sql", f"CREATE DOCUMENT TYPE {name}")
        test_db.command("sql", f"CREATE PROPERTY {name}.x DOUBLE")
        if _older_engine_refuses(test_db, sql):
            return
        test_db.command("sql", f"CREATE INDEX ON {name} (x) COLUMNAR")
        with test_db.transaction():
            for value in values:
                test_db.command("sql", f"INSERT INTO {name} SET x = :x", {"x": value})
        oracle = test_db.query(
            "sql",
            f"SELECT min(x) AS mn, max(x) AS mx, sum(x) AS sm, avg(x) AS av FROM {name}",  # nosec B608 - test-owned name
        ).first()

        _, facets = _facets(test_db, sql)
        stats = facets["s"]

        assert stats["count"] == 2
        for member, column in (
            ("min", "mn"),
            ("max", "mx"),
            ("sum", "sm"),
            ("avg", "av"),
        ):
            _same_float(stats[member], oracle.get(column))

    def test_signed_zeros_and_distinct_of_odd_doubles(self, test_db):
        """The engine's own case: NaN, both infinities and both zeros in one column. DISTINCT counts -0.0 and 0.0 as one value;
        the extremes of a negative zero come back as 0.0 (a documented limitation), and 0.0 == -0.0 in Python anyway.
        """
        sql = (
            "SELECT FROM Odd FACETS sx: x STATS, sz: z STATS, "
            "dx: x DISTINCT, dz: z DISTINCT"
        )
        test_db.command("sql", "CREATE DOCUMENT TYPE Odd")
        test_db.command("sql", "CREATE PROPERTY Odd.x DOUBLE")
        test_db.command("sql", "CREATE PROPERTY Odd.z DOUBLE")
        if _older_engine_refuses(test_db, sql):
            return
        test_db.command("sql", "CREATE INDEX ON Odd (x) COLUMNAR")
        test_db.command("sql", "CREATE INDEX ON Odd (z) COLUMNAR")
        xs = [math.nan, math.inf, -math.inf, -0.0, 0.0, 1.5, 1.5]
        zs = [-0.0, 0.0, -0.0]
        with test_db.transaction():
            for value in xs:
                test_db.command("sql", "INSERT INTO Odd SET x = :v", {"v": value})
            for value in zs:
                test_db.command("sql", "INSERT INTO Odd SET z = :v", {"v": value})
        oracle = test_db.query(
            "sql",
            "SELECT min(x) AS mn, max(x) AS mx, sum(x) AS sm, avg(x) AS av FROM Odd",
        ).first()

        rows, facets = _facets(test_db, sql)

        assert len(rows) == 10
        x = facets["sx"]
        assert (x["count"], x["missing"]) == (7, 3)
        for member, column in (
            ("min", "mn"),
            ("max", "mx"),
            ("sum", "sm"),
            ("avg", "av"),
        ):
            _same_float(x[member], oracle.get(column))
        assert x["min"] == -math.inf
        z = facets["sz"]
        assert (z["count"], z["missing"]) == (3, 7)
        assert z["min"] == 0.0 and z["max"] == 0.0 and z["sum"] == 0.0
        assert isinstance(z["min"], float) and isinstance(z["max"], float)
        # five values in x (NaN, both infinities, the zeros as one, 1.5), one in z
        assert facets["dx"]["type"] == "distinct"
        assert (facets["dx"]["value"], facets["dx"]["exact"]) == (5, True)
        assert (facets["dz"]["value"], facets["dz"]["exact"]) == (1, True)

    def test_stats_filter_parameters(self, test_db):
        """The condition of a FILTER takes named and positional parameters, through query and command."""
        _kind(test_db)
        if _older_engine_refuses(test_db, STATS_FILTER_NAMED_Q, {"lo": 60}):
            return

        # the filter of the STATS facet narrows the rows and the other facet: ids 5 to 9, whose names are cherry, apple,
        # banana, apple and date
        for run in ("query", "command"):
            for sql, args in (
                (STATS_FILTER_NAMED_Q, ({"lo": 60},)),
                (STATS_FILTER_POSITIONAL_Q, (60,)),
            ):
                rows, facets = _facets(test_db, sql, *args, run=run)
                assert [r["id"] for r in rows] == [5, 6, 7, 8, 9], (run, sql)
                assert (facets["d"]["value"], facets["d"]["exact"]) == (4, True)
                assert facets["s"]["type"] == "stats"

        # the filter of the DISTINCT facet narrows the rows and the STATS facet (a filter's own facet counts without it)
        for run in ("query", "command"):
            for sql, args in (
                (DISTINCT_FILTER_NAMED_Q, ({"lo": 60},)),
                (DISTINCT_FILTER_POSITIONAL_Q, (60,)),
            ):
                rows, facets = _facets(test_db, sql, *args, run=run)
                assert [r["id"] for r in rows] == [5, 6, 7, 8, 9], (run, sql)
                stats = facets["s"]
                assert (stats["count"], stats["min"], stats["max"]) == (5, 60, 100)
                assert (stats["sum"], stats["avg"]) == (400, 80)


class TestDistinct:
    def test_exact_distinct_equals_group_by(self, test_db):
        _kind(test_db)
        if _older_engine_refuses(test_db, DISTINCT_Q):
            return

        rows, facets = _facets(test_db, DISTINCT_Q)

        assert len(rows) == 10
        assert list(facets) == ["meta", "n", "q", "g"]
        by_name = test_db.query(
            "sql", "SELECT name FROM FacetKind GROUP BY name"
        ).to_list()
        by_qty = test_db.query(
            "sql", "SELECT qty FROM FacetKind GROUP BY qty"
        ).to_list()
        assert len(by_name) == 7 and len(by_qty) == 10
        for alias, expected in (("n", len(by_name)), ("q", len(by_qty))):
            distinct = facets[alias]
            assert distinct["type"] == "distinct", alias
            assert distinct["source"] in ("index", "scan"), alias
            assert "buckets" not in distinct, alias
            assert (distinct["value"], distinct["exact"]) == (expected, True), alias
            assert distinct["missing"] == 0, alias
        assert isinstance(facets["n"]["value"], int)
        # a list indexed BY ITEM counts its items: five of them, and four records have no list
        tags = facets["g"]
        assert tags["type"] == "distinct"
        assert (tags["value"], tags["exact"]) == (len(KIND_TAG_COUNTS), True)
        assert tags["missing"] == 4

    def test_approximate_distinct_is_flagged(self, test_db):
        _kind(test_db)
        if _older_engine_refuses(test_db, APPROXIMATE_Q):
            return

        _, facets = _facets(test_db, APPROXIMATE_Q)

        distinct = facets["q"]
        assert distinct["type"] == "distinct"
        assert distinct["exact"] is False
        assert facets["meta"]["exact"] is False
        assert abs(distinct["value"] - 10) <= 2

    def test_precision_at_or_above_the_cardinality_stays_exact(self, test_db):
        _kind(test_db)
        if _older_engine_refuses(test_db, BELOW_THRESHOLD_Q):
            return

        _, facets = _facets(test_db, BELOW_THRESHOLD_Q)

        # ten values do not exceed PRECISION 10: the exact set is kept
        distinct = facets["q"]
        assert (distinct["value"], distinct["exact"]) == (10, True)

    def test_distinct_filter_parameter_with_no_match(self, test_db):
        _kind(test_db)
        sql = "SELECT id FROM FacetKind FACETS d: name DISTINCT, s: qty STATS FILTER qty > :hi"
        if _older_engine_refuses(test_db, sql, {"hi": 1000}):
            return

        rows, facets = _facets(test_db, sql, {"hi": 1000})

        assert rows == []
        assert (facets["d"]["value"], facets["d"]["exact"]) == (0, True)


class TestHierarchy:
    def test_buckets_are_flat_with_parent_and_depth(self, test_db):
        _tagged(test_db)
        if _older_engine_refuses(test_db, HIERARCHY_Q):
            return

        rows, facets = _facets(test_db, HIERARCHY_Q)

        assert len(rows) == 5
        tags = facets["t"]
        assert tags["type"] == "hierarchy"
        assert tags["source"] in ("index", "scan"), tags.get("reason")
        assert "reason" not in tags
        by_value = {b["value"]: b for b in tags["buckets"] if not b.get("missing")}
        # every record counts once under each ancestor of its tags: two tags in one subtree are one count of the shared ancestors
        assert {v: b["count"] for v, b in by_value.items()} == {
            "t0": 3,
            "t0/a": 3,
            "t0/a/x": 2,
            "t0/b": 1,
            "orphan": 1,
        }
        # a nested hierarchy is a flat list: the place of a bucket is its parent (the value of its first parent, None for a root)
        # and its depth
        for value, parent, depth in (
            ("t0", None, 0),
            ("t0/a", "t0", 1),
            ("t0/b", "t0", 1),
            ("t0/a/x", "t0/a", 2),
        ):
            bucket = by_value[value]
            assert bucket.get("parent") == parent, value
            assert bucket["depth"] == depth, value
            assert "orphan" not in bucket, value
        # a value no tag vertex has is its own root, flagged
        orphan = by_value["orphan"]
        assert orphan["orphan"] is True
        assert orphan.get("parent") is None and orphan["depth"] == 0
        assert tags["other_count"] == 0
        assert tags["missing"] == 1

    def test_top_leaves_the_rest_in_other_count(self, test_db):
        _tagged(test_db)
        if _older_engine_refuses(test_db, HIERARCHY_TOP_Q):
            return

        _, full = _facets(test_db, HIERARCHY_Q)
        _, top = _facets(test_db, HIERARCHY_TOP_Q)

        everything = _counts(full["t"])
        returned = _counts(top["t"])
        assert len(returned) == 2
        # the two buckets with the most records: t0 and t0/a, three each
        assert returned == {"t0": 3, "t0/a": 3}
        # the counts of the nodes not returned (the counts overlap, so the sum can exceed the matches)
        assert top["t"]["other_count"] == sum(everything.values()) - sum(
            returned.values()
        )

    def test_filter_parameters_narrow_the_hierarchy(self, test_db):
        """The filter of the STATS facet narrows the rows and the hierarchy: n >= 3 keeps records 3, 4 and 5."""
        _tagged(test_db)
        if _older_engine_refuses(test_db, HIERARCHY_FILTER_NAMED_Q, {"lo": 3}):
            return

        for run in ("query", "command"):
            for sql, args in (
                (HIERARCHY_FILTER_NAMED_Q, ({"lo": 3},)),
                (HIERARCHY_FILTER_POSITIONAL_Q, (3,)),
            ):
                rows, facets = _facets(test_db, sql, *args, run=run)
                assert [r["id"] for r in rows] == [3, 4, 5], (run, sql)
                tags = facets["t"]
                assert _counts(tags) == {
                    "t0": 1,
                    "t0/a": 1,
                    "t0/a/x": 1,
                    "t0/b": 1,
                    "orphan": 1,
                }, (run, sql)
                assert tags["missing"] == 1

    def test_hierarchy_next_to_terms(self, test_db):
        _tagged(test_db)
        sql = (
            "SELECT id FROM FacetTagged ORDER BY id "
            "FACETS t: tags HIERARCHY TOP 100, g: tags TERMS TOP 10"
        )
        if _older_engine_refuses(test_db, sql):
            return

        _, facets = _facets(test_db, sql)

        assert list(facets) == ["meta", "t", "g"]
        assert facets["t"]["type"] == "hierarchy" and facets["g"]["type"] == "terms"
        # the terms facet counts the stored values only: the ancestors are not stored on any record
        assert _counts(facets["g"]) == {
            "t0/a/x": 2,
            "t0/a": 1,
            "t0/b": 1,
            "orphan": 1,
        }


class TestAfter:
    """Paging of a terms facet ordered by value: the first page asks ``ORDER BY VALUE``, every next one ``AFTER <cursor>`` with
    the cursor of the page before, until the cursor is None. The facet then holds ``after`` (the last value of a full page, as
    a row holds it), and the records without a value (``MISSING``) come once, on the last page.
    """

    def _page_all(self, db, prop, top, run="query", options="", positional=False):
        """Every page of the facet: ``([(value, count)], [missing buckets], pages)``."""
        pairs = []
        missing = []
        cursor = None
        for page in range(PAGES_MAX):
            if cursor is None:
                sql = f"SELECT id FROM FacetPg FACETS f: {prop} TERMS TOP {top}{options} ORDER BY VALUE"  # nosec B608 - test-owned text
                args = ()
            elif positional:
                sql = f"SELECT id FROM FacetPg FACETS f: {prop} TERMS TOP {top}{options} AFTER ?"  # nosec B608 - test-owned text
                args = (cursor,)
            else:
                sql = f"SELECT id FROM FacetPg FACETS f: {prop} TERMS TOP {top}{options} AFTER :c"  # nosec B608 - test-owned text
                args = ({"c": cursor},)
            _, facets = _facets(db, sql, *args, run=run)
            facet = facets["f"]
            assert facet["type"] == "terms", sql
            assert "after" in facet, sql
            page_values = [b for b in facet["buckets"] if not b.get("missing")]
            page_missing = [b for b in facet["buckets"] if b.get("missing")]
            pairs.extend((b["value"], b["count"]) for b in page_values)
            missing.extend(page_missing)
            if facet["after"] is None:
                return pairs, missing, page + 1
            # a cursor comes with a full page, and it is the last value of the page
            assert len(page_values) == top, sql
            assert facet["after"] == page_values[-1]["value"], sql
            assert not page_missing, sql  # MISSING only on the last page
            cursor = facet["after"]
        raise AssertionError(f"more than {PAGES_MAX} pages")

    def _group_by(self, db, prop):
        rows = db.query(
            "sql",
            f"SELECT {prop} AS v, count(*) AS c FROM FacetPg WHERE {prop} IS NOT NULL GROUP BY {prop} ORDER BY v",  # nosec B608 - test-owned text
        ).to_list()
        return [(r["v"], r["c"]) for r in rows]

    @pytest.mark.parametrize("run", ["query", "command"])
    @pytest.mark.parametrize("top", [1, 3, 8, 100])
    def test_paging_returns_every_value_once_in_value_order(self, test_db, run, top):
        _pg(test_db)
        probe = "SELECT id FROM FacetPg FACETS f: name TERMS TOP 3 ORDER BY VALUE"
        if _older_engine_refuses(test_db, probe):
            return

        pairs, missing, pages = self._page_all(test_db, "name", top, run=run)

        expected = self._group_by(test_db, "name")
        assert len(expected) == 8
        assert pairs == expected
        assert (
            missing == []
        )  # without MISSING the records without a value are not a bucket
        # a cursor only follows a page that a value follows: ceil(8 / top) pages
        assert pages == -(-8 // top)

    @pytest.mark.parametrize(
        "prop", ["num", "stamp"], ids=["long values", "datetime values"]
    )
    def test_paging_with_numeric_and_datetime_cursors(self, test_db, prop):
        """The cursor of a LONG or a DATETIME facet is the Python value of the last bucket, sent back as it came."""
        _pg(test_db)
        probe = "SELECT id FROM FacetPg FACETS f: num TERMS TOP 3 ORDER BY VALUE"
        if _older_engine_refuses(test_db, probe):
            return

        pairs, _, _ = self._page_all(test_db, prop, 3)

        assert pairs == self._group_by(test_db, prop)
        assert len(pairs) == 8
        sample = pairs[0][0]
        assert isinstance(sample, int if prop == "num" else datetime)

    def test_missing_is_returned_once_on_the_last_page(self, test_db):
        _pg(test_db)
        probe = (
            "SELECT id FROM FacetPg FACETS f: name TERMS TOP 3 MISSING ORDER BY VALUE"
        )
        if _older_engine_refuses(test_db, probe):
            return

        pairs, missing, pages = self._page_all(test_db, "name", 3, options=" MISSING")

        assert pairs == self._group_by(test_db, "name")
        assert pages == 3
        # once, outside TOP, with the records that have no name
        assert missing == [{"value": None, "missing": True, "count": 3}]

    def test_the_cursor_can_be_positional(self, test_db):
        _pg(test_db)
        probe = "SELECT id FROM FacetPg FACETS f: name TERMS TOP 3 AFTER ?"
        if _older_engine_refuses(test_db, probe, "a"):
            return

        pairs, _, _ = self._page_all(test_db, "name", 3, positional=True)

        assert pairs == self._group_by(test_db, "name")

    def test_a_literal_cursor_and_a_cursor_that_is_no_value(self, test_db):
        _pg(test_db)
        sql = "SELECT id FROM FacetPg FACETS f: name TERMS TOP 3 AFTER 'bz'"
        if _older_engine_refuses(test_db, sql):
            return

        # a cursor between two values: the page starts at the next one
        _, facets = _facets(test_db, sql)
        assert [b["value"] for b in facets["f"]["buckets"]] == ["c", "d", "e"]
        assert facets["f"]["after"] == "e"
        # a cursor past the last value: no bucket, no further page, and the key is there
        _, facets = _facets(test_db, sql.replace("'bz'", "'zz'"))
        assert facets["f"]["buckets"] == []
        assert "after" in facets["f"] and facets["f"]["after"] is None
        assert facets["f"]["missing"] == 3

    def test_last_page_and_other_count(self, test_db):
        _pg(test_db)
        sql = "SELECT id FROM FacetPg FACETS f: name TERMS TOP 3 AFTER 'e'"
        if _older_engine_refuses(test_db, sql):
            return

        _, facets = _facets(test_db, sql)

        last = facets["f"]
        assert [b["value"] for b in last["buckets"]] == ["f", "g", "h"]
        assert last["after"] is None  # no value follows
        # every matched value outside the page, the ones before the cursor too
        total = sum(c for _, c in self._group_by(test_db, "name"))
        assert total == 15
        assert last["other_count"] == total - sum(b["count"] for b in last["buckets"])

    def test_only_a_facet_ordered_by_value_pages(self, test_db):
        _pg(test_db)
        sql = "SELECT id FROM FacetPg FACETS f: name TERMS TOP 3"
        if _older_engine_refuses(test_db, sql):
            return

        _, facets = _facets(test_db, sql)

        assert "after" not in facets["f"]

    def test_after_with_order_by_count_is_refused(self, test_db):
        _pg(test_db)
        sql = "SELECT id FROM FacetPg FACETS f: name TERMS ORDER BY COUNT AFTER 'a'"
        if _older_engine_refuses(test_db, sql):
            return

        with pytest.raises(Exception, match="AFTER pages in value order"):
            test_db.query("sql", sql).to_list()


class TestSearchValues:
    """``facet.searchValues(type, property, prefix, limit)``: the values of a COLUMNAR string property (or a list of strings
    indexed BY ITEM) that start with a prefix, in value order, as rows ``{value, count}`` with the number of live records that
    carry the value; a row has ``exact: False`` only when the count could not be made exact.
    """

    @staticmethod
    def _search(db, type_name, prop, prefix, limit, *, named=True):
        if named:
            rows = db.query(
                "sql",
                SEARCH_NAMED_Q,
                {"t": type_name, "p": prop, "x": prefix, "l": limit},
            ).to_list()
        else:
            rows = db.query(
                "sql", SEARCH_POSITIONAL_Q, type_name, prop, prefix, limit
            ).to_list()
        return [(r["value"], r["count"]) for r in rows]

    def _ready(self, db):
        _kind(db)
        return not _older_engine_refuses(db, NAME_Q)

    def test_prefix_in_value_order_with_counts(self, test_db):
        if not self._ready(test_db):
            return

        def search(prefix, limit=20):
            return lambda: self._search(test_db, "FacetKind", "tags", prefix, limit)

        assert _until(search("run"), [("run", 3), ("runner", 1), ("running", 1)]) == [
            ("run", 3),
            ("runner", 1),
            ("running", 1),
        ]
        every = [(v, KIND_TAG_COUNTS[v]) for v in sorted(KIND_TAG_COUNTS)]
        # the empty prefix matches every value, the first ones in value order under a limit
        assert _until(search(""), every) == every
        assert self._search(test_db, "FacetKind", "tags", "", 2) == every[:2]
        assert self._search(test_db, "FacetKind", "tags", "ru", 20) == [
            ("rum", 2),
            ("run", 3),
            ("runner", 1),
            ("running", 1),
        ]
        assert self._search(test_db, "FacetKind", "tags", "ru", 2) == [
            ("rum", 2),
            ("run", 3),
        ]

    def test_counts_equal_group_by(self, test_db):
        if not self._ready(test_db):
            return

        by_name = {
            r["name"]: r["n"]
            for r in test_db.query(
                "sql", "SELECT name, count(*) AS n FROM FacetKind GROUP BY name"
            ).to_list()
        }
        assert by_name == KIND_NAME_COUNTS
        expected = [("banana", 2), ("blueberry", 1)]

        found = _until(
            lambda: self._search(test_db, "FacetKind", "name", "b", 10), expected
        )

        assert found == expected
        for value, count in self._search(test_db, "FacetKind", "name", "", 100):
            assert count == by_name[value]

    def test_a_prefix_with_no_match_gives_no_rows(self, test_db):
        if not self._ready(test_db):
            return
        # once the projection has the values (a prefix that matches nothing is an empty answer from the start, so wait on one)
        _until(
            lambda: self._search(test_db, "FacetKind", "tags", "run", 20),
            [("run", 3), ("runner", 1), ("running", 1)],
        )

        assert self._search(test_db, "FacetKind", "tags", "zzz", 20) == []
        assert self._search(test_db, "FacetKind", "name", "zzz", 20) == []
        # a prefix is literal, never a pattern
        assert self._search(test_db, "FacetKind", "tags", "r.*", 20) == []
        assert self._search(test_db, "FacetKind", "tags", "run", 0) == []

    def test_named_and_positional_parameters_and_literals(self, test_db):
        if not self._ready(test_db):
            return
        expected = [("run", 3), ("runner", 1), ("running", 1)]

        named = _until(
            lambda: self._search(test_db, "FacetKind", "tags", "run", 20), expected
        )
        positional = self._search(test_db, "FacetKind", "tags", "run", 20, named=False)
        literal = [
            (r["value"], r["count"])
            for r in test_db.query(
                "sql",
                "SELECT expand(facet.searchValues('FacetKind', 'tags', 'run', 20))",
            ).to_list()
        ]

        assert named == positional == literal == expected
        # a row is the two members, and `exact` only when a count is not exact
        row = test_db.query(
            "sql", SEARCH_POSITIONAL_Q, "FacetKind", "tags", "run", 1
        ).to_list()
        assert set(row[0]) - {"exact"} == {"value", "count"}
        assert isinstance(row[0]["count"], int)

    def test_a_deleted_record_does_not_suggest_its_value(self, test_db):
        if not self._ready(test_db):
            return
        with test_db.transaction():
            test_db.command(
                "sql",
                "INSERT INTO FacetKind SET id = 99, name = 'zed', tags = ['runaway']",
            )
        _until(
            lambda: self._search(test_db, "FacetKind", "tags", "runa", 20),
            [("runaway", 1)],
        )
        with test_db.transaction():
            test_db.command("sql", "DELETE FROM FacetKind WHERE id = 99")
        expected = [("run", 3), ("runner", 1), ("running", 1)]

        found = _until(
            lambda: self._search(test_db, "FacetKind", "tags", "run", 20), expected
        )

        # the dictionary keeps the value until compaction, the count does not: a value no live record holds is dropped
        assert found == expected
        assert self._search(test_db, "FacetKind", "tags", "runa", 20) == []

    def test_errors_reach_the_caller(self, test_db):
        if not self._ready(test_db):
            return

        with pytest.raises(Exception, match="no COLUMNAR index on FacetKind.id"):
            self._search(test_db, "FacetKind", "id", "1", 5)
        with pytest.raises(Exception, match="Nope"):
            self._search(test_db, "Nope", "tags", "run", 5)
        with pytest.raises(Exception, match="limit must be >= 0"):
            self._search(test_db, "FacetKind", "tags", "run", -1)
