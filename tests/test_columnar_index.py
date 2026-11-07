"""The COLUMNAR index on both builds: create, ingest, facets and GROUP BY, close and reopen, drop, and the settings (issue #292).

``CREATE INDEX ... COLUMNAR`` (ArcadeDB 26.12.1) keeps a column projection of a type next to its records. The projection is
applied, flushed and merged by threads of the engine, and ``FACETS`` queries and ``GROUP BY`` read it. This file runs on
``arcadedb-jvm`` and on the native ``arcadedb`` with only the public Python API. On the native build it is the proof that those
threads and the index code run inside the GraalVM shared library: the engine's classes that the index and the facets touch by
reflection are listed in native/src/main/resources/META-INF/native-image/org.humemai/arcadedb-native-lib/reachability-metadata.json
(the audit is in the pull request of issue #292).

The engine of a lane is whatever the lane links (see tests/test_columnar_engine_smoke.py): an engine older than 26.12.1 must
refuse the index and the settings with its own error, a newer one must run them. The version is read through the test's database
(``SELECT version()``), so the same check runs on both builds and nothing is skipped.

What the tests state:

* the facets of a type with a COLUMNAR index equal what Python counts over the rows it inserted, for a single-valued property, a
  numeric one and a multi-valued one (``BY ITEM``), and ``GROUP BY`` gives the same counts, sums and extremes;
* a clean close and an unclean one (a child process that ends without closing the database, so ``database.lck`` is left behind and
  the next open replays the write-ahead log) both leave the facets exact, and the projection serves them again by itself;
* ``arcadedb.columnar.groupByAcceleration`` is a database setting that ``ALTER DATABASE`` changes at once and stores with the
  database, ``PROFILE`` shows which path ran, and both paths return the same rows;
* the settings with the scope of the JVM or of a server are not database settings: ``ALTER DATABASE`` refuses them.

Waiting for the projection polls a state with a deadline that is a hang detector, never a latency bound.
"""

import glob
import json
import os
import re
import subprocess  # nosec B404 - test-controlled child process, no untrusted input
import sys
import textwrap
import time
from collections import Counter

import pytest

from arcadedb import create_database, open_database

FACETS_SINCE = (26, 12, 1)

CATEGORIES = ["red", "green", "blue", "red", "green", "red"]
TAG_SETS = [["x", "y"], ["y"], ["z", "x", "x"], ["x"], ["y", "z"]]

SCHEMA_SQL = [
    "CREATE DOCUMENT TYPE Item",
    "CREATE PROPERTY Item.id INTEGER",
    "CREATE PROPERTY Item.category STRING",
    "CREATE PROPERTY Item.score INTEGER",
    "CREATE PROPERTY Item.tags LIST OF STRING",
]
INDEX_SQL = [
    "CREATE INDEX ON Item (category) COLUMNAR",
    "CREATE INDEX ON Item (score) COLUMNAR",
    "CREATE INDEX ON Item (tags BY ITEM) COLUMNAR",
]

FACETS_Q = (
    "SELECT id FROM Item ORDER BY id "
    "FACETS category TERMS TOP 10, tags TERMS TOP 10, s: score STATS"
)
GROUP_BY_Q = (
    "SELECT category, count(*) AS n, sum(score) AS total, min(score) AS lo, "
    "max(score) AS hi FROM Item GROUP BY category ORDER BY category"
)

GROUP_BY_KEY = "arcadedb.columnar.groupByAcceleration"
# the settings of a database, by scope (GlobalConfiguration 26.12.1): these are DATABASE scope
DATABASE_SETTINGS = [
    "arcadedb.columnar.flushInterval",
    "arcadedb.columnar.flushMaxRows",
    "arcadedb.columnar.flushMinRows",
    "arcadedb.columnar.flushMaxAge",
    "arcadedb.columnar.mergeDeadRatio",
    "arcadedb.columnar.mergeMaxSegments",
    "arcadedb.columnar.maxCardinality",
    "arcadedb.columnar.segmentBlockRows",
    "arcadedb.columnar.applyInterval",
    "arcadedb.columnar.applyMaxLag",
    "arcadedb.columnar.dictionaryCompactRatio",
    "arcadedb.columnar.maxFallbackScans",
    GROUP_BY_KEY,
]
JVM_SETTINGS = [
    "arcadedb.columnar.applyThreads",
    "arcadedb.columnar.maintenanceThreads",
    "arcadedb.columnar.rebuildConcurrency",
    "arcadedb.columnar.queryMemoryBudget",
]
SERVER_SETTINGS = [
    "arcadedb.columnar.patchBudget",
    "arcadedb.columnar.patchBudgetRatio",
    "arcadedb.columnar.queryBudgetWait",
    "arcadedb.columnar.parallelCountTasks",
    "arcadedb.columnar.parallelMinMatches",
    "arcadedb.columnar.topMax",
]


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db = create_database(str(tmp_path / "test_columnar_index_db"))
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


def _is_older_engine(path):
    """True when the engine predates the COLUMNAR index. Read through a database of its own, so a test can decide before it
    starts a child process or reopens its database."""
    probe = create_database(path)
    try:
        return _engine_version(probe) < FACETS_SINCE
    finally:
        probe.drop()


def _make_rows(start, count):
    return [
        {
            "id": i,
            "category": CATEGORIES[i % len(CATEGORIES)],
            "score": (i * 7) % 50 + 1,
            "tags": TAG_SETS[i % len(TAG_SETS)],
        }
        for i in range(start, start + count)
    ]


def _insert_statement(row):
    """The INSERT of one row and its positional arguments; the list is a literal, so the same text runs on both builds."""
    tags = "[" + ", ".join(f"'{t}'" for t in row["tags"]) + "]"
    sql = f"INSERT INTO Item SET id = ?, category = ?, score = ?, tags = {tags}"  # nosec B608 - test-owned text
    return sql, [row["id"], row["category"], row["score"]]


def _insert(db, rows):
    with db.transaction():
        for row in rows:
            sql, args = _insert_statement(row)
            db.command("sql", sql, *args)


def _create_schema(db):
    for sql in SCHEMA_SQL:
        db.command("sql", sql)


def _create_indexes(db):
    for sql in INDEX_SQL:
        db.command("sql", sql)


def _refused_by_older_engine(db):
    """True when the engine predates the COLUMNAR index: the statement is then refused with the engine's error, and the caller
    has nothing more to check on this engine."""
    if _engine_version(db) >= FACETS_SINCE:
        return False
    with pytest.raises(
        Exception, match=re.escape("Index type 'COLUMNAR' is not supported")
    ):
        db.command("sql", INDEX_SQL[0])
    return True


def _columnar_names(db):
    """The names of the type-level COLUMNAR indexes of Item (the per-bucket rows are named `Item_0_<id>`)."""
    rows = db.query("sql", "SELECT FROM schema:indexes").to_list()
    return sorted(
        r.get("name") for r in rows if str(r.get("name")).endswith("_columnar")
    )


def _wait_for(predicate, what, timeout=120.0):
    """The value of `predicate()` once it is truthy. The timeout is a hang detector."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out after {timeout}s waiting for {what}")
        time.sleep(0.1)


def _counts(facet):
    """A terms facet as {value: count}, the bucket of the records without a value left out."""
    return {b["value"]: b["count"] for b in facet["buckets"] if not b.get("missing")}


def _expected_group_by(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(row["category"], [])
        group.append(row["score"])
    return [
        {"category": c, "n": len(s), "total": sum(s), "lo": min(s), "hi": max(s)}
        for c, s in sorted(groups.items())
    ]


def _check_facets(db, rows, sources=("index",)):
    """The facets of FACETS_Q equal what Python counts over `rows`; returns the sources the facets were answered from."""
    rs = db.query("sql", FACETS_Q)
    got = rs.to_list()
    facets = rs.facets
    assert [r["id"] for r in got] == [r["id"] for r in rows]
    assert facets["meta"]["exact"] is True, facets["meta"]

    category, tags, score = facets["category"], facets["tags"], facets["s"]
    assert category["source"] in sources, category
    assert tags["source"] in sources, tags
    assert score["source"] in sources, score
    assert _counts(category) == dict(Counter(r["category"] for r in rows))
    # a record counts once for a tag it holds twice
    assert _counts(tags) == dict(Counter(t for r in rows for t in set(r["tags"])))
    scores = [r["score"] for r in rows]
    assert score["count"] == len(rows)
    assert (score["min"], score["max"], score["sum"]) == (
        min(scores),
        max(scores),
        sum(scores),
    )
    return {category["source"], tags["source"], score["source"]}


def _facet_sources(db):
    """Where the facets of FACETS_Q are answered from right now, for a test that waits for the projection to serve them."""
    rs = db.query("sql", FACETS_Q)
    rs.to_list()
    facets = rs.facets
    return {facets[alias]["source"] for alias in ("category", "tags", "s")}


def _check_group_by(db, rows):
    assert db.query("sql", GROUP_BY_Q).to_list() == _expected_group_by(rows)


def _plan(db, sql):
    """The plan of a PROFILE run of `sql`: it names the path the GROUP BY took (`path: columnar` or `path: normal (reason: ...)`)."""
    return str(db.query("sql", f"PROFILE {sql}").first().get("executionPlanAsString"))


def _settings(db):
    row = db.query("sql", "SELECT settings FROM schema:database").first()
    return {s["key"]: s for s in row.get("settings")}


def _text(value):
    return str(value).lower()


def _alter(db, key, value):
    db.command("sql", f"ALTER DATABASE `{key}` {value}")


def _segment_rows(db_path):
    """The rows the published segments of every bucket hold, read from the manifests the engine writes under `columnar/`."""
    total = 0
    for manifest in glob.glob(os.path.join(db_path, "columnar", "b*", "manifest.json")):
        try:
            with open(manifest, encoding="utf-8") as handle:
                total += sum(s["rows"] for s in json.load(handle)["segments"])
        except (OSError, ValueError):
            pass  # replaced while it was read: the next poll sees the new one
    return total


class TestColumnarIndexLifecycle:
    def test_create_ingest_query_drop(self, test_db):
        _create_schema(test_db)
        if _refused_by_older_engine(test_db):
            return

        _create_indexes(test_db)
        names = _columnar_names(test_db)
        assert "Item[category]_columnar" in names, names
        assert "Item[score]_columnar" in names, names
        assert len(names) == 3 and any("tags" in n for n in names), names
        assert test_db.schema.exists_index("Item[category]_columnar")

        rows = _make_rows(0, 60)
        for start in range(0, 60, 20):
            _insert(test_db, rows[start : start + 20])

        assert _check_facets(test_db, rows) == {"index"}
        _check_group_by(test_db, rows)
        _wait_for(
            lambda: "path: columnar" in _plan(test_db, GROUP_BY_Q),
            "GROUP BY to be answered from the projection",
        )
        _check_group_by(test_db, rows)

        # updates and deletes reach the projection too
        with test_db.transaction():
            test_db.command("sql", "UPDATE Item SET category = 'blue' WHERE id < 10")
            test_db.command("sql", "DELETE FROM Item WHERE id >= 50")
        rows = [dict(r, category="blue") if r["id"] < 10 else r for r in rows[:50]]
        assert _check_facets(test_db, rows) == {"index"}
        _check_group_by(test_db, rows)

        # dropped: the same query is answered by a scan, and GROUP BY by the normal plan
        for name in names:
            test_db.command("sql", f"DROP INDEX `{name}`")
        assert _columnar_names(test_db) == []
        assert not test_db.schema.exists_index("Item[category]_columnar")
        assert _check_facets(test_db, rows, sources=("scan",)) == {"scan"}
        _check_group_by(test_db, rows)
        assert "path: columnar" not in _plan(test_db, GROUP_BY_Q)

    def test_index_created_on_existing_records_is_rebuilt_by_the_engine(self, test_db):
        _create_schema(test_db)
        if _refused_by_older_engine(test_db):
            return

        rows = _make_rows(0, 80)
        _insert(test_db, rows)
        _create_indexes(test_db)

        # the maintenance pool builds the projection of the records that were there: until then a scan answers, exactly
        assert _check_facets(test_db, rows, sources=("index", "scan"))
        _wait_for(
            lambda: _facet_sources(test_db) == {"index"},
            "the projection of the existing records",
        )
        assert _check_facets(test_db, rows) == {"index"}
        _check_group_by(test_db, rows)

    def test_flush_and_merge_run_in_the_background(self, test_db):
        """The flush and merge threads: with a short flush interval and few segments allowed, a stream of small commits ends up
        in published segments (the manifests under columnar/), merged, and the facets stay exact.
        """
        _create_schema(test_db)
        if _refused_by_older_engine(test_db):
            return

        _alter(test_db, "arcadedb.columnar.flushInterval", 20)
        _alter(test_db, "arcadedb.columnar.applyInterval", 10)
        _alter(test_db, "arcadedb.columnar.mergeMaxSegments", 2)
        assert (
            str(_settings(test_db)["arcadedb.columnar.flushInterval"]["value"]) == "20"
        )
        _create_indexes(test_db)

        rows = _make_rows(0, 120)
        for start in range(0, 120, 10):
            _insert(test_db, rows[start : start + 10])
            time.sleep(0.05)  # lets a flush land between commits

        path = test_db.get_database_path()
        _wait_for(
            lambda: _segment_rows(path) >= len(rows),
            "the flush to publish every row in a segment",
        )
        assert _check_facets(test_db, rows) == {"index"}
        _check_group_by(test_db, rows)

    def test_clean_close_and_reopen(self, tmp_path):
        path = str(tmp_path / "clean_reopen_db")
        if _is_older_engine(str(tmp_path / "version_probe_db")):
            with create_database(path) as db:
                _create_schema(db)
                assert _refused_by_older_engine(db)
            return

        rows = _make_rows(0, 70)
        with create_database(path) as db:
            _create_schema(db)
            _create_indexes(db)
            _insert(db, rows)
            assert _check_facets(db, rows) == {"index"}

        with open_database(path) as db:
            assert len(_columnar_names(db)) == 3
            # exact at once, from the projection or from a scan, and served from the projection again by itself
            assert _check_facets(db, rows, sources=("index", "scan"))
            _wait_for(
                lambda: _facet_sources(db) == {"index"},
                "the projection after the reopen",
            )
            assert _check_facets(db, rows) == {"index"}
            _check_group_by(db, rows)

            # the reopened projection takes new commits
            more = _make_rows(70, 30)
            _insert(db, more)
            rows = rows + more
            assert _check_facets(db, rows) == {"index"}

        with open_database(path) as db:
            _wait_for(
                lambda: _facet_sources(db) == {"index"},
                "the projection after the second reopen",
            )
            assert _check_facets(db, rows) == {"index"}
            # dropped after a reopen: the index and its projection are gone
            for name in _columnar_names(db):
                db.command("sql", f"DROP INDEX `{name}`")
            assert _columnar_names(db) == []
            assert _check_facets(db, rows, sources=("scan",)) == {"scan"}

        with open_database(path) as db:
            assert _columnar_names(db) == []
            assert _check_facets(db, rows, sources=("scan",)) == {"scan"}
            _check_group_by(db, rows)


_CHILD = textwrap.dedent("""
    import glob, json, os, sys, time

    import arcadedb

    path, spec_file = sys.argv[1], sys.argv[2]
    with open(spec_file, encoding="utf-8") as handle:
        spec = json.load(handle)

    db = arcadedb.create_database(path)
    for sql in spec["schema"]:
        db.command("sql", sql)
    for index, batch in enumerate(spec["batches"]):
        if index == spec["wait_before_batch"]:
            # the earlier batches must be in published segments before the rest is written
            deadline = time.monotonic() + 120
            while True:
                total = 0
                for manifest in glob.glob(os.path.join(path, "columnar", "b*", "manifest.json")):
                    try:
                        with open(manifest, encoding="utf-8") as handle:
                            total += sum(s["rows"] for s in json.load(handle)["segments"])
                    except (OSError, ValueError):
                        pass
                if total >= spec["wait_rows"]:
                    break
                if time.monotonic() > deadline:
                    sys.exit("the projection never flushed")
                time.sleep(0.1)
        with db.transaction():
            for sql, args in batch:
                db.command("sql", sql, *args)
    print("READY", flush=True)
    # no close: the process ends at once, as a crash of the process does. What was committed is in the write-ahead log.
    os._exit(0)
    """)


class TestColumnarIndexUncleanClose:
    """The process that wrote the database ends without closing it. Python cannot make the engine's own kill (the Java
    ``DatabaseInternal.kill()`` of the engine's crash tests) on the native build, and a JVM-only hook would not test both
    packages, so the writer is a child process that calls ``os._exit``: the same state the engine's recovery reads, a database
    with its lock file left behind and committed pages not written to the data files (a crash of the process, not a power loss).
    """

    @pytest.mark.parametrize(
        "flushed_first", [False, True], ids=["unflushed", "flushed"]
    )
    def test_projection_after_an_unclean_close(self, tmp_path, flushed_first):
        path = str(tmp_path / "unclean_db")
        if _is_older_engine(str(tmp_path / "version_probe_db")):
            with create_database(path) as db:
                _create_schema(db)
                assert _refused_by_older_engine(db)
            return

        first, second = _make_rows(0, 60), _make_rows(60, 20)
        spec = {
            "schema": SCHEMA_SQL + INDEX_SQL,
            "batches": [
                [list(_insert_statement(r)) for r in first],
                [list(_insert_statement(r)) for r in second],
            ],
            # flushed: the first batch is waited for in published segments, so the recovery has the second one to replay on top of them
            "wait_before_batch": 1 if flushed_first else -1,
            "wait_rows": len(first),
        }
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")

        proc = subprocess.run(  # nosec B603 - fixed argv, no shell, test-owned
            [sys.executable, "-c", _CHILD, path, str(spec_file)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert "READY" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"
        # a clean close deletes the lock file (tests/test_concurrency.py): one left behind marks an unclean close
        assert os.path.exists(os.path.join(path, "database.lck"))

        rows = first + second
        with open_database(path) as db:
            assert len(_columnar_names(db)) == 3
            # every committed row is there, and the facets are exact at once: from the projection, patched, or from a scan
            assert db.query("sql", "SELECT count(*) AS n FROM Item").first().get(
                "n"
            ) == len(rows)
            assert _check_facets(db, rows, sources=("index", "scan"))
            _check_group_by(db, rows)
            # the projection catches up or is rebuilt by the engine, with nobody asking
            _wait_for(
                lambda: _facet_sources(db) == {"index"},
                "the projection after the unclean close",
            )
            assert _check_facets(db, rows) == {"index"}
            more = _make_rows(80, 20)
            _insert(db, more)
            rows = rows + more
            assert _check_facets(db, rows) == {"index"}

        with open_database(path) as db:
            _wait_for(
                lambda: _facet_sources(db) == {"index"},
                "the projection after the clean close that followed",
            )
            assert _check_facets(db, rows) == {"index"}
            _check_group_by(db, rows)


class TestColumnarSettings:
    def test_database_settings_are_listed_with_their_defaults(self, test_db):
        if _engine_version(test_db) < FACETS_SINCE:
            with pytest.raises(Exception, match="not found"):
                _alter(test_db, GROUP_BY_KEY, "false")
            return

        settings = _settings(test_db)
        for key in DATABASE_SETTINGS:
            assert key in settings, key
        for key in JVM_SETTINGS + SERVER_SETTINGS:
            assert key not in settings, key
        assert _text(settings[GROUP_BY_KEY]["default"]) == "true"
        assert _text(settings[GROUP_BY_KEY]["value"]) == "true"

    def test_settings_of_the_jvm_and_of_a_server_are_not_database_settings(
        self, test_db
    ):
        if _engine_version(test_db) < FACETS_SINCE:
            with pytest.raises(Exception, match="not found"):
                _alter(test_db, JVM_SETTINGS[0], 4)
            return

        with pytest.raises(Exception) as jvm:
            _alter(test_db, "arcadedb.columnar.applyThreads", 4)
        assert "JVM scope" in str(jvm.value)
        assert "cannot be set per database" in str(jvm.value)

        with pytest.raises(Exception) as server:
            _alter(test_db, "arcadedb.columnar.patchBudget", 500)
        assert "SERVER scope" in str(server.value)
        assert "cannot be set per database" in str(server.value)

    def test_group_by_gives_the_same_rows_with_and_without_the_acceleration(
        self, test_db
    ):
        _create_schema(test_db)
        if _refused_by_older_engine(test_db):
            return

        _create_indexes(test_db)
        rows = _make_rows(0, 90)
        _insert(test_db, rows)
        expected = _expected_group_by(rows)

        accelerated = test_db.query("sql", GROUP_BY_Q).to_list()
        assert accelerated == expected
        _wait_for(
            lambda: "path: columnar" in _plan(test_db, GROUP_BY_Q),
            "GROUP BY to be answered from the projection",
        )

        _alter(test_db, GROUP_BY_KEY, "false")
        assert _text(_settings(test_db)[GROUP_BY_KEY]["value"]) == "false"
        plan = _plan(test_db, GROUP_BY_Q)
        assert "path: columnar" not in plan
        assert f"disabled by {GROUP_BY_KEY}" in plan
        assert test_db.query("sql", GROUP_BY_Q).to_list() == accelerated

        _alter(test_db, GROUP_BY_KEY, "true")
        assert _text(_settings(test_db)[GROUP_BY_KEY]["value"]) == "true"
        assert "path: columnar" in _plan(test_db, GROUP_BY_Q)
        assert test_db.query("sql", GROUP_BY_Q).to_list() == accelerated

    def test_the_settings_are_stored_with_the_database(self, tmp_path):
        path = str(tmp_path / "settings_db")
        if _is_older_engine(str(tmp_path / "version_probe_db")):
            with create_database(path) as db:
                with pytest.raises(Exception, match="not found"):
                    _alter(db, GROUP_BY_KEY, "false")
            return

        rows = _make_rows(0, 40)
        with create_database(path) as db:
            _create_schema(db)
            _create_indexes(db)
            _insert(db, rows)
            _alter(db, GROUP_BY_KEY, "false")
            _alter(db, "arcadedb.columnar.flushInterval", 50)
            _alter(db, "arcadedb.columnar.mergeDeadRatio", 0.5)

        with open_database(path) as db:
            settings = _settings(db)
            assert _text(settings[GROUP_BY_KEY]["value"]) == "false"
            assert str(settings["arcadedb.columnar.flushInterval"]["value"]) == "50"
            assert float(settings["arcadedb.columnar.mergeDeadRatio"]["value"]) == 0.5
            assert f"disabled by {GROUP_BY_KEY}" in _plan(db, GROUP_BY_Q)
            _check_group_by(db, rows)
            _alter(db, GROUP_BY_KEY, "true")
            _wait_for(
                lambda: "path: columnar" in _plan(db, GROUP_BY_Q),
                "GROUP BY to be answered from the projection after the reopen",
            )
            _check_group_by(db, rows)
