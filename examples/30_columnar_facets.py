#!/usr/bin/env python3
"""Example 30: Faceted Queries with a COLUMNAR Index.

A product list with the numbers of a search sidebar next to it: the rows of a
query (a page of five), and the counts of the same query for each category,
each price range and each month, over ALL the matches, not only the five rows.
The counts come from the ``FACETS`` clause of a ``SELECT``, and Python reads
them from ``result_set.facets``, a plain dict, the same on ``arcadedb`` and on
``arcadedb-jvm``.

Workflow covered:
- create a document type, load it, then ``CREATE INDEX ... COLUMNAR`` on the
  properties that the facets count
- one query with a ``TERMS`` facet (the 3 biggest categories) and a ``RANGES``
  facet (price bands), with a page of rows: ``LIMIT 5``
- one query with ``STATS`` (count, min, max, sum, average of the price) and
  ``DISTINCT`` (number of brands)
- one query with a calendar ``HISTOGRAM`` (one bucket for each month)
- every count is checked against the same count made in Python over the rows
  that were inserted, so the example fails when a facet is wrong

Requirements:
- ArcadeDB engine 26.12.1 or later. On an older engine the example prints one
  line and exits with status 0: the engine refuses the ``FACETS`` clause and the
  ``COLUMNAR`` index type there.

Notes:
- Read the rows first and ``result_set.facets`` after, as the example does.
- No data download: the products are generated from the row number.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import arcadedb

FACETS_SINCE = (26, 12, 1)

CATEGORIES = ["laptop", "phone", "tablet", "monitor", "keyboard", "mouse"]
BRANDS = ["Acme", "Globex", "Initech", "Umbrella", "Hooli"]
MIN_PRICE = 20  # the WHERE of every query below

TERMS_AND_RANGES = (
    f"SELECT id, name, price FROM Product WHERE price >= {MIN_PRICE} ORDER BY id "
    "FACETS p: price RANGES ([0, 100) AS 'low', [100, 300) AS 'mid', [300, *) AS 'high'), "
    "c: category TERMS TOP 3 LIMIT 5"
)
STATS_AND_DISTINCT = (
    f"SELECT id FROM Product WHERE price >= {MIN_PRICE} ORDER BY id "
    "FACETS s: price STATS, d: brand DISTINCT"
)
MONTHLY = (
    f"SELECT id FROM Product WHERE price >= {MIN_PRICE} ORDER BY id "
    "FACETS h: created HISTOGRAM CALENDAR '1M' TIMEZONE 'UTC'"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rows", type=int, default=1200, help="number of products")
    p.add_argument("--db-path", default="./my_test_databases/columnar_facets")
    return p.parse_args()


def engine_version(db) -> tuple[str, tuple[int, int, int]]:
    """The engine's version text and its x.y.z (a SNAPSHOT or a build suffix is dropped)."""
    raw = str(db.query("sql", "SELECT version() AS v").first().get("v"))
    parts = [
        int(p) if p.isdigit() else 0
        for p in raw.split()[0].split("-")[0].split(".")[:3]
    ]
    return raw, tuple(parts + [0] * (3 - len(parts)))


def make_products(count: int) -> list[dict]:
    """Deterministic products: nothing is random, so every run has the same counts."""
    return [
        {
            "id": i,
            "name": f"item-{i:04d}",
            "category": CATEGORIES[(i + i // 4) % len(CATEGORIES)],
            "brand": BRANDS[(i * 3 + i // 7) % len(BRANDS)],
            "price": 5 + (i * 37) % 495,
            # noon on days 1 to 27, so no time zone moves a record to another month
            "created": datetime(2024, 1 + i % 12, 1 + (i * 5) % 27, 12, 0, 0),
        }
        for i in range(count)
    ]


def load(db, products: list[dict]) -> None:
    for ddl in (
        "CREATE DOCUMENT TYPE Product",
        "CREATE PROPERTY Product.id INTEGER",
        "CREATE PROPERTY Product.name STRING",
        "CREATE PROPERTY Product.category STRING",
        "CREATE PROPERTY Product.brand STRING",
        "CREATE PROPERTY Product.price INTEGER",
        "CREATE PROPERTY Product.created DATETIME",
    ):
        db.command("sql", ddl)
    with db.transaction():
        for p in products:
            db.command(
                "sql",
                "INSERT INTO Product SET id = ?, name = ?, category = ?, brand = ?, "
                "price = ?, created = ?",
                p["id"],
                p["name"],
                p["category"],
                p["brand"],
                p["price"],
                p["created"],
            )
    # the index is built over the records that exist, and kept up to date after
    for prop in ("category", "brand", "price", "created"):
        db.command("sql", f"CREATE INDEX ON Product ({prop}) COLUMNAR")


def show_terms_and_ranges(db, matches: list[dict]) -> None:
    rs = db.query("sql", TERMS_AND_RANGES)
    rows = rs.to_list()  # the rows first, then the facets
    facets = rs.facets
    print(f"\nrows (page of {len(rows)} of {len(matches)} matches):")
    for r in rows:
        print(f"  {r['id']:>4}  {r['name']}  {r['price']:>4}")
    assert [r["id"] for r in rows] == [p["id"] for p in matches[:5]]

    print(f"\nfacets, over all {len(matches)} matches, not the {len(rows)} rows:")
    categories = facets["c"]
    print(f"  category ({categories['type']}, from {categories['source']}):")
    for b in categories["buckets"]:
        print(f"    {b['value']:<10} {b['count']:>5}")
    print(f"    {'(other)':<10} {categories['other_count']:>5}")
    by_category = Counter(p["category"] for p in matches)
    assert len(categories["buckets"]) == 3
    for b in categories["buckets"]:
        assert b["count"] == by_category[b["value"]], b
    # ties of the same count may come in either order, so compare the counts, not the names
    assert [b["count"] for b in categories["buckets"]] == sorted(
        by_category.values(), reverse=True
    )[:3]
    assert categories["other_count"] == len(matches) - sum(
        b["count"] for b in categories["buckets"]
    )

    prices = facets["p"]
    print(f"  price ({prices['type']}):")
    for b in prices["buckets"]:
        print(f"    {b['key']:<10} {b['count']:>5}")
    expected = Counter(
        "low" if p["price"] < 100 else "mid" if p["price"] < 300 else "high"
        for p in matches
    )
    assert {b["key"]: b["count"] for b in prices["buckets"]} == dict(expected)
    meta = facets["meta"]
    print(f"  meta: exact={meta['exact']} patched={meta['patched']}")


def show_stats_and_distinct(db, matches: list[dict]) -> None:
    rs = db.query("sql", STATS_AND_DISTINCT)
    assert len(rs.to_list()) == len(matches)
    facets = rs.facets
    stats, brands = facets["s"], facets["d"]
    print("\nstats and distinct:")
    print(
        f"  price: count={stats['count']} min={stats['min']} max={stats['max']} "
        f"sum={stats['sum']} avg={stats['avg']:.2f}"
    )
    print(f"  brands: {brands['value']} (exact: {brands['exact']})")
    prices = [p["price"] for p in matches]
    assert (stats["count"], stats["missing"]) == (len(prices), 0)
    assert (stats["min"], stats["max"], stats["sum"]) == (
        min(prices),
        max(prices),
        sum(prices),
    )
    assert abs(stats["avg"] - sum(prices) / len(prices)) < 1e-9
    assert (brands["value"], brands["exact"]) == (
        len({p["brand"] for p in matches}),
        True,
    )


def show_monthly(db, matches: list[dict]) -> None:
    rs = db.query("sql", MONTHLY)
    assert len(rs.to_list()) == len(matches)
    hist = rs.facets["h"]
    print("\nproducts for each month of 2024:")
    full = [
        b for b in hist["buckets"] if b["count"]
    ]  # a month with no record may be listed
    for b in full:
        print(f"  {b['key']:<10} {b['count']:>5}")
    by_month = Counter(p["created"].month for p in matches)
    assert [b["count"] for b in full] == [by_month[m] for m in sorted(by_month)]


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with arcadedb.create_database(str(db_path)) as db:
        raw_version, version = engine_version(db)
        if version < FACETS_SINCE:
            print(
                f"Engine {raw_version} has no FACETS clause and no COLUMNAR index "
                "(they need engine 26.12.1 or later): nothing to run."
            )
            return
        print(f"engine: arcadedb {arcadedb.__version__}, engine version {raw_version}")

        products = make_products(args.rows)
        load(db, products)
        matches = [p for p in products if p["price"] >= MIN_PRICE]
        print(
            f"loaded {len(products)} products, {len(matches)} with price >= {MIN_PRICE}"
        )

        show_terms_and_ranges(db, matches)
        show_stats_and_distinct(db, matches)
        show_monthly(db, matches)
        print("\nall facet counts match the counts made in Python")


if __name__ == "__main__":
    main()
