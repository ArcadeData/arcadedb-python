#!/usr/bin/env python3
"""Smoke test of an installed native `arcadedb` wheel: import it, open a database, run one SQL and one openCypher query.

test-native.yml runs it inside the manylinux_2_28 container (glibc 2.28), the oldest system the Linux wheels are tagged for.
"""

import os
import tempfile

import arcadedb

print(f"arcadedb {arcadedb.__version__}, backend {arcadedb.__backend__}, engine {arcadedb.ENGINE_VERSION}")
assert arcadedb.__backend__ == "native", arcadedb.__backend__
with tempfile.TemporaryDirectory() as d:
    with arcadedb.create_database(os.path.join(d, "smoke")) as db:
        assert db.query("sql", "SELECT 1 AS one").first().get("one") == 1
        with db.transaction():
            db.command("sql", "CREATE VERTEX TYPE P")
            db.command("sql", "CREATE VERTEX P SET name = 'Ada'")
        assert db.query("opencypher", "MATCH (n:P) RETURN n.name AS name").to_list()[0]["name"] == "Ada"
print("smoke ok")
