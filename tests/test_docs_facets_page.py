"""Every Python block of the Antora page ``docs/modules/ROOT/pages/how-to/facets.adoc`` runs, and prints what the page says.

The blocks of the page are chained on purpose: they run in the order of the page, in one namespace, against one fresh temporary
database that the namespace holds as ``db`` (the page says that each block assumes an open ``db``). A block that is followed by a
``[source,text]`` block must print exactly that text. The test uses no Java object, so it runs on the JVM build and the native
build alike.

Facets are new in engine 26.12.1 (see tests/test_columnar_engine_smoke.py). The engine of a lane is whatever the lane links: the
JVM lane runs the ``latest`` image, the native pull request gate links 26.11.1. A block of the page that creates a COLUMNAR index
or runs a ``FACETS`` query carries ``role=needs-engine`` and the title "Needs engine 26.12.1 or later". On a newer engine such a
block runs and prints what the page shows. On an older engine the same block must be refused with the engine's own error, which
is what the page documents, and the test asserts that refusal instead of the output. Nothing is skipped (the repository accepts no
skips, scripts/check_test_skips.py). The version is read through the test's database (``SELECT version()``), so the same check
runs on both builds.

The limits of a faceted query came into 26.12.1 after its first snapshots, so a block that states one carries
``role=needs-engine-limits``. It runs on an engine that has the settings (``arcadedb.columnar.hierarchyMaxNodes`` is listed by
``SELECT settings FROM schema:database``, which a 26.12.1 release must have). On any other engine the test asserts that the engine
does not know the setting.
"""

import contextlib
import io
import json
import os
import re
import subprocess  # nosec B404 - runs this interpreter on a script of this file
import sys
from pathlib import Path

import pytest

FACETS_SINCE = (26, 12, 1)

PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "modules"
    / "ROOT"
    / "pages"
    / "how-to"
    / "facets.adoc"
)

# a Python listing (its header may name a role, and a title line may come before it), then (optionally, directly after it) the
# text listing with its output
BLOCK_RE = re.compile(
    r"(?:^(?P<title>\.[^\n]*)\n)?"
    r"\[source,python(?P<attrs>[^\]\n]*)\]\n----\n(?P<code>.*?)\n----\n"
    r"(?:\n\[source,text\]\n----\n(?P<out>.*?)\n----\n)?",
    re.DOTALL | re.MULTILINE,
)

NEEDS_TITLE = ".Needs engine 26.12.1 or later"

# what an engine before 26.12.1 answers: a FACETS clause is a syntax error, a COLUMNAR index is an unknown index type
REFUSAL = r"SQL syntax error|Index type 'COLUMNAR' is not supported|COLUMNAR"


def _blocks():
    """The Python blocks as (code, expected output or None, needs the new engine, needs the limits)."""
    text = PAGE.read_text(encoding="utf-8")
    found = []
    for m in BLOCK_RE.finditer(text):
        needs = "role=needs-engine" in m.group("attrs")
        # the title and the role say the same thing, in both directions
        assert needs == (m.group("title") == NEEDS_TITLE), m.group("code")[:80]
        needs_limits = "role=needs-engine-limits" in m.group("attrs")
        found.append((m.group("code"), m.group("out"), needs, needs_limits))
    # no Python listing may escape the extraction (a different spelling of the block header would run nowhere)
    assert text.count("[source,python") == len(found)
    return found


def _engine_version(db):
    """The engine's x.y.z through SQL `SELECT version()`; a SNAPSHOT or a build suffix is dropped."""
    raw = str(db.query("sql", "SELECT version() AS v").first().get("v"))
    parts = [
        int(p) if p.isdigit() else 0
        for p in raw.split()[0].split("-")[0].split(".")[:3]
    ]
    return tuple(parts + [0] * (3 - len(parts)))


LIMIT_PROBE = "arcadedb.columnar.hierarchyMaxNodes"


def _has_limits(db):
    """Whether the engine has the limits of a faceted query. A 26.12.1 snapshot may lack them; a release may not."""
    row = db.query("sql", "SELECT settings FROM schema:database").first()
    present = LIMIT_PROBE in {s["key"] for s in row.get("settings")}
    version = str(db.query("sql", "SELECT version() AS v").first().get("v")).split()[0]
    if _engine_version(db) >= FACETS_SINCE and "SNAPSHOT" not in version:
        assert present, version
    return present


def test_the_page_has_its_blocks():
    blocks = _blocks()
    assert len(blocks) >= 18
    assert sum(1 for _, out, _, _ in blocks if out is not None) >= 14
    assert sum(1 for *_, needs_limits in blocks if needs_limits) == 2
    needing = [code for code, _, needs, _ in blocks if needs]
    assert len(needing) >= 12
    # every block that needs the new engine says so: its first engine call is a COLUMNAR index or a FACETS query
    for code in needing:
        assert "COLUMNAR" in code or "FACETS" in code, code[:80]
    # and no other block does: they all run on an engine without facets
    for code, _, needs, _ in blocks:
        if not needs:
            assert "FACETS" not in code and "COLUMNAR" not in code, code[:80]


def test_every_block_runs_in_order_and_prints_what_the_page_says(temp_db):
    namespace = {"db": temp_db, "__name__": "facets_page"}
    newer = _engine_version(temp_db) >= FACETS_SINCE
    limits = _has_limits(temp_db)
    if not limits:
        with pytest.raises(Exception, match="not found"):
            temp_db.command("sql", f"ALTER DATABASE `{LIMIT_PROBE}` 3")
    refused = 0
    for number, (code, expected, needs, needs_limits) in enumerate(_blocks(), start=1):
        compiled = compile(code, f"facets.adoc python block {number}", "exec")
        captured = io.StringIO()
        if needs_limits and not limits:
            continue
        if needs and not newer:
            # an older engine: the block is refused with the engine's own error, before it prints anything
            with pytest.raises(Exception, match=REFUSAL):
                with contextlib.redirect_stdout(captured):
                    exec(
                        compiled, namespace
                    )  # nosec B102 - the page of this repository
            assert captured.getvalue() == "", f"block {number} printed before it failed"
            refused += 1
            continue
        with contextlib.redirect_stdout(captured):
            exec(compiled, namespace)  # nosec B102 - the page of this repository
        if expected is not None:
            assert (
                captured.getvalue().strip() == expected.strip()
            ), f"block {number} printed other text than the page shows:\n{code}"
        else:
            assert (
                captured.getvalue() == ""
            ), f"block {number} prints, the page shows no output"

    # the page's own version check agrees with the test's, on the engine of this lane
    assert namespace["has_facets"] is newer
    assert (refused == 0) == newer


# four buckets for `a` and two for `b`: with a limit of 3 the first facet does not fit and the second does
BUCKET_SCRIPT = """
import json, sys
import arcadedb

with arcadedb.create_database(sys.argv[1]) as db:
    db.command("sql", "CREATE DOCUMENT TYPE D")
    db.command("sql", "CREATE PROPERTY D.a STRING")
    db.command("sql", "CREATE PROPERTY D.b STRING")
    db.command("sql", "CREATE INDEX ON D (a) COLUMNAR")
    db.command("sql", "CREATE INDEX ON D (b) COLUMNAR")
    with db.transaction():
        for i in range(8):
            db.command("sql", "INSERT INTO D SET a = ?, b = ?", f"a{i % 4}", f"b{i % 2}")
    rs = db.query("sql", "SELECT FROM D FACETS wide: a TERMS TOP 10, narrow: b TERMS TOP 10")
    rs.to_list()
    facets = rs.facets
    print(json.dumps({k: [facets[k]["source"], facets[k].get("reason")] for k in ("wide", "narrow")}))
"""


def test_a_facet_above_the_bucket_limit_is_unavailable(temp_db, tmp_path):
    """`arcadedb.columnar.maxResponseBuckets` has server scope, so it is set before the engine of a child process starts."""
    if not _has_limits(temp_db):
        with pytest.raises(Exception, match="not found"):
            temp_db.command("sql", f"ALTER DATABASE `{LIMIT_PROBE}` 3")
        return
    option = "-Darcadedb.columnar.maxResponseBuckets=3"
    env = dict(os.environ)
    for name in ("ARCADEDB_JVM_ARGS", "ARCADEDB_NATIVE_ISOLATE_ARGS"):
        env[name] = f"{env.get(name, '')} {option}".strip()
    result = subprocess.run(  # nosec B603 - this interpreter and the script above
        [sys.executable, "-c", BUCKET_SCRIPT, str(tmp_path / "bucket_limit_db")],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    answer = json.loads(result.stdout.strip().splitlines()[-1])
    assert answer["wide"] == ["unavailable", "bucket limit"]
    assert answer["narrow"][0] in ("index", "scan")
