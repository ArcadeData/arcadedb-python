#!/usr/bin/env python3
"""Example 29: Hybrid Agent-Memory Retrieval in One SQL Statement.

For an agent-memory or GraphRAG builder who today glues a vector store, a
keyword index and a graph store together. This example does the retrieval in
ONE ArcadeDB statement: the nearest facts by embedding, restricted to one
tenant BEFORE the top-k is taken, fused with the full-text matches of the same
tenant by reciprocal rank fusion, and then expanded by graph hops to the facts
that share a related entity.

Everything goes through db.query / db.command; there is no Python wrapper.
The memory is synthetic (fixed-seed numpy embeddings, templated sentences);
no download.

The pre-filter is the point. Taking the global top-k first and filtering by
tenant afterwards returns fewer rows than asked, or none, whenever the nearest
neighbours belong to another tenant (this is what Cypher's
``db.index.vector.queryNodes(index, k, vector)`` does without its fourth
argument, the ``{where, filter}`` options map of ArcadeDB 26.11.1: a ``WHERE``
after ``YIELD`` runs on its k results; ArcadeData/arcadedb#10685).
``vector.neighbors`` takes the restriction as an option, so the index search
itself only considers the tenant's vectors:

* ``{ where: { tenant: :tenant } }``: tested on each candidate as the search
  meets it. Needs ArcadeDB 26.11.1 (ArcadeData/arcadedb#10169, #10185). This
  is the form the statement below uses.
* ``{ filter: (SELECT @rid FROM Fact WHERE tenant = :tenant) }``: a list of
  RIDs. Works on 26.10.2 as well. ``--form filter`` runs the example with it,
  and the example checks that both forms return the same rows when the engine
  has ``where``.

The index types are created with SQL and are always the same: an ``LSM_VECTOR``
index on ``Fact.embedding`` and a ``FULL_TEXT`` index on ``Fact.text``, plus
a ``NOTUNIQUE`` index on ``Fact.tenant``: from 26.11.1 a selective ``where`` on
an indexed property takes the property index instead of loading candidates.
``vector.fuse`` and the ``SEARCH_INDEX`` source are used as documented in the
ArcadeDB vector-search concept page ("Hybrid Search with vector.fuse").

Note on recall: at the index defaults the search beam narrows once an index
holds 10,000 vectors or more (ArcadeData/arcadedb#10704); this memory has
about 900 vectors, far below that, so the defaults are fine here. For a larger
memory pass ``efSearch`` in the options map: ``{ where: ..., efSearch: 100 }``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np

import arcadedb

SEED = 20261024
DIM = 16
FACTS_PER_TENANT = 150
TENANTS = ["acme", "globex", "initech", "umbrella", "hooli", "stark"]
REQUESTED_TENANT = "acme"
K = 10  # neighbours the vector side asks for; also the size of the full-text side
TOP = 5  # rows of the fused answer

# topic -> (entities, sentence templates)
TOPICS = {
    "coffee": (
        ["espresso", "grinder", "barista", "beans"],
        [
            "drinks {a} every morning and keeps a {b} at home",
            "asked the team to order more {a} because the {b} ran out",
        ],
    ),
    "running": (
        ["marathon", "shoes", "pace", "trail"],
        [
            "trains for the {a} and tracks the {b} on every run",
            "bought new {a} after the {b} race last spring",
        ],
    ),
    "travel": (
        ["lisbon", "flight", "hotel", "passport"],
        [
            "plans a trip to {a} and booked the {b} already",
            "lost the {a} on the way and had to rebook the {b}",
        ],
    ),
    "music": (
        ["guitar", "piano", "concert", "vinyl"],
        [
            "practises the {a} before the {b} on Friday",
            "collects {a} records and goes to every {b} in town",
        ],
    ),
}

# The one statement. The vector side is pre-filtered by tenant (the `where` option,
# 26.11.1), the full-text side is a subquery on the same tenant, `vector.fuse` merges
# the two ranked lists by reciprocal rank fusion, and the outer SELECT hops through the
# graph from every fused fact:
#   related_entities: Fact -MENTIONS-> Entity -RELATES-> Entity (two hops)
#   related_fids:     ... -MENTIONS<- Fact, without the fact itself (three hops, a
#                     sample of 3, in no particular order)
# `[tenant = :tenant]` keeps a hop from leaving the tenant: the memory holds one stray
# RELATES edge from an acme entity to a globex entity, and without it the hops of the
# acme facts would return globex facts.
# One statement was possible; nothing is split into several queries.
_HYBRID_TEMPLATE = """
SELECT record.fid AS fid,
       record.text AS fact,
       score,
       record.out('MENTIONS').out('RELATES')[tenant = :tenant].name
         AS related_entities,
       difference(
         record.out('MENTIONS').out('RELATES').in('MENTIONS')[tenant = :tenant].fid,
         [record.fid]
       )[0..3] AS related_fids
FROM (
  SELECT expand(`vector.fuse`(
    `vector.neighbors`('Fact[embedding]', :qvec, :k, { %(restrict)s }),
    (SELECT @rid, $score FROM Fact
      WHERE tenant = :tenant AND SEARCH_INDEX('Fact[text]', :keywords) = true
      ORDER BY $score DESC LIMIT :k),
    { fusion: 'RRF' }
  ))
)
ORDER BY score DESC
LIMIT :top
"""
RESTRICT = {
    "where": "where: { tenant: :tenant }",  # 26.11.1
    "filter": "filter: (SELECT @rid FROM Fact WHERE tenant = :tenant)",  # 26.10.2 too
}
HYBRID_SQL = {form: _HYBRID_TEMPLATE % {"restrict": r} for form, r in RESTRICT.items()}


def neighbors_sql(restrict: str | None) -> str:
    """The vector side alone, with (restrict) or without a restriction."""
    opts = f", {{ {restrict} }}" if restrict else ""
    return (
        "SELECT record.fid AS fid, record.tenant AS tenant, distance FROM ("
        f"SELECT expand(`vector.neighbors`('Fact[embedding]', :qvec, :k{opts})))"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db-path", default="./my_test_databases/hybrid_memory")
    p.add_argument(
        "--form",
        choices=sorted(RESTRICT),
        default="where",
        help="how the vector side is pre-filtered: where (26.11.1) or filter "
        "(a RID subquery, 26.10.2 and later)",
    )
    return p.parse_args()


def build(db) -> dict:
    """Create the schema and a synthetic multi-tenant memory. Returns the rng-made
    topic centroids so the caller can write a query vector in the same space."""
    rng = np.random.default_rng(SEED)
    centroids = {t: _unit(rng.standard_normal(DIM)) for t in TOPICS}
    for ddl in (
        "CREATE VERTEX TYPE Fact",
        "CREATE PROPERTY Fact.fid INTEGER",
        "CREATE PROPERTY Fact.tenant STRING",
        "CREATE PROPERTY Fact.text STRING",
        "CREATE PROPERTY Fact.embedding ARRAY_OF_FLOATS",
        "CREATE VERTEX TYPE Entity",
        "CREATE PROPERTY Entity.eid INTEGER",
        "CREATE PROPERTY Entity.tenant STRING",
        "CREATE PROPERTY Entity.name STRING",
        "CREATE EDGE TYPE MENTIONS",
        "CREATE EDGE TYPE RELATES",
        "CREATE INDEX ON Fact (fid) UNIQUE_HASH",
        "CREATE INDEX ON Entity (eid) UNIQUE_HASH",
        "CREATE INDEX ON Fact (text) FULL_TEXT",
        "CREATE INDEX ON Fact (tenant) NOTUNIQUE",
        'CREATE INDEX ON Fact (embedding) LSM_VECTOR METADATA {"dimensions": %d, '
        '"similarity": "COSINE"}' % DIM,
    ):
        db.command("sql", ddl)

    eid = fid = 0
    entity_ids = {}  # (tenant, topic, name) -> eid
    entity_edges = []
    with db.transaction():
        for tenant in TENANTS:
            for topic, (names, _) in TOPICS.items():
                for name in names:
                    db.command(
                        "sql",
                        "CREATE VERTEX Entity SET eid = ?, tenant = ?, name = ?",
                        eid,
                        tenant,
                        name,
                    )
                    entity_ids[(tenant, topic, name)] = eid
                    eid += 1
            # RELATES: each entity points at the next one of its topic, and the
            # last entity of a topic points at the first of the next topic.
            topics = list(TOPICS)
            for ti, topic in enumerate(topics):
                names = TOPICS[topic][0]
                for ni, name in enumerate(names):
                    if ni + 1 < len(names):
                        nxt = (tenant, topic, names[ni + 1])
                    else:
                        other = topics[(ti + 1) % len(topics)]
                        nxt = (tenant, other, TOPICS[other][0][0])
                    entity_edges.append(
                        (entity_ids[(tenant, topic, name)], entity_ids[nxt])
                    )
        # the stray edge that crosses tenants (see HYBRID_SQL)
        entity_edges.append(
            (
                entity_ids[("acme", "coffee", "grinder")],
                entity_ids[("globex", "music", "guitar")],
            )
        )
        for src, dst in entity_edges:
            db.command(
                "sql",
                "CREATE EDGE RELATES FROM (SELECT FROM Entity WHERE eid = ?) "
                "TO (SELECT FROM Entity WHERE eid = ?)",
                src,
                dst,
            )

    mention_edges = []
    with db.transaction():
        for tenant in TENANTS:
            # acme's facts are looser around the topic than everyone else's, so
            # a query near a topic has other tenants' facts as its nearest
            # neighbours: the case where filtering after the top-k loses rows.
            noise = 0.45 if tenant == REQUESTED_TENANT else 0.15
            for i in range(FACTS_PER_TENANT):
                topic = list(TOPICS)[int(rng.integers(len(TOPICS)))]
                names, templates = TOPICS[topic]
                a, b = rng.choice(names, size=2, replace=False)
                template = templates[int(rng.integers(len(templates)))]
                text = f"{tenant} user " + template.format(a=a, b=b)
                emb = _unit(centroids[topic] + noise * rng.standard_normal(DIM))
                db.command(
                    "sql",
                    "CREATE VERTEX Fact SET fid = ?, tenant = ?, text = ?, embedding = ?",
                    fid,
                    tenant,
                    text,
                    arcadedb.to_java_float_array(emb.astype(np.float32).tolist()),
                )
                for name in (a, b):
                    mention_edges.append((fid, entity_ids[(tenant, topic, str(name))]))
                fid += 1
        for f, e in mention_edges:
            db.command(
                "sql",
                "CREATE EDGE MENTIONS FROM (SELECT FROM Fact WHERE fid = ?) "
                "TO (SELECT FROM Entity WHERE eid = ?)",
                f,
                e,
            )
    return centroids


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def query_vector(centroids: dict) -> list:
    """The embedding of 'what coffee does the user like': near the coffee topic."""
    rng = np.random.default_rng(SEED + 1)
    v = _unit(centroids["coffee"] + 0.05 * rng.standard_normal(DIM))
    return v.astype(np.float32).tolist()


def run(db, centroids: dict, form: str) -> dict:
    """Run the statements for REQUESTED_TENANT and return everything the checks need."""
    params = {
        "qvec": arcadedb.to_java_float_array(query_vector(centroids)),
        "tenant": REQUESTED_TENANT,
        "keywords": "espresso grinder",
        "k": K,
        "top": TOP,
    }

    def q(sql, **extra):
        every = {**params, **extra}
        used = {n: every[n] for n in re.findall(r":(\w+)", sql) if n in every}
        return db.query("sql", sql, used).to_list()

    # What goes wrong without a pre-filter: the global top-K, then the tenant test.
    post = q("SELECT fid FROM (" + neighbors_sql(None) + ") WHERE tenant = :tenant")
    pre = q(neighbors_sql(RESTRICT[form]))
    hybrid = q(HYBRID_SQL[form])
    # The same hops without the tenant guard, to show what the guard removes.
    unguarded = q(
        "SELECT out('MENTIONS').out('RELATES').in('MENTIONS').fid AS fids "
        "FROM Fact WHERE fid = :fid",
        fid=hybrid[0]["fid"],
    )[0]["fids"]
    return {
        "post_filtered": post,
        "pre_filtered": pre,
        "hybrid": hybrid,
        "unguarded_fids": unguarded,
    }


def tenants_of(db, fids) -> set:
    """Tenants of the given fact ids (empty input: empty set)."""
    fids = list(fids)
    if not fids:
        return set()
    rows = db.query("sql", "SELECT tenant FROM Fact WHERE fid IN :fids", {"fids": fids})
    return {r["tenant"] for r in rows.to_list()}


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with arcadedb.create_database(str(db_path)) as db:
        centroids = build(db)
        counts = {
            t: db.query("sql", f"SELECT count(*) AS n FROM {t}").to_list()[0]["n"]
            for t in ("Fact", "Entity", "MENTIONS", "RELATES")
        }
        print(f"memory: {counts}, tenants: {len(TENANTS)}")
        print(f"engine: arcadedb {arcadedb.__version__}, form: {args.form}")
        out = run(db, centroids, args.form)

        print(f"\nrequested tenant: {REQUESTED_TENANT}, k = {K}")
        print(
            f"global top-{K}, then WHERE tenant: {len(out['post_filtered'])} rows "
            "(the nearest neighbours belong to other tenants)"
        )
        print(f"pre-filtered top-{K} ({args.form}): {len(out['pre_filtered'])} rows")

        print(f"\nhybrid result (vector + full-text + graph), top {TOP}:")
        for r in out["hybrid"]:
            print(f"  fid {r['fid']:>3}  rrf {r['score']:.4f}  {r['fact']}")
            print(f"           related entities: {r['related_entities']}")
            print(f"           related facts:    {r['related_fids']}")

        # Checks: every row, and every fact reached by a hop, is the tenant's.
        shown = [r["fid"] for r in out["hybrid"]]
        related = [f for r in out["hybrid"] for f in r["related_fids"] or []]
        checks = {
            "pre-filtered rows are all the tenant's": tenants_of(
                db, [r["fid"] for r in out["pre_filtered"]]
            ),
            "hybrid rows are all the tenant's": tenants_of(db, shown),
            "hop results are all the tenant's": tenants_of(db, related),
        }
        for label, found in checks.items():
            assert found == {REQUESTED_TENANT}, (label, found)
            print(f"check ok: {label} {sorted(found)}")
        assert len(out["pre_filtered"]) == K and len(out["hybrid"]) == TOP
        leaked = tenants_of(db, out["unguarded_fids"]) - {REQUESTED_TENANT}
        print(
            f"without the [tenant = :tenant] hop guard the hops also reach: {sorted(leaked)}"
        )


if __name__ == "__main__":
    main()
