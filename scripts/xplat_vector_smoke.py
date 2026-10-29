#!/usr/bin/env python3
"""Rough vector-search smoke for one installed `arcadedb` wheel (native or JVM).

One run (the default) builds an LSM_VECTOR index (COSINE) over random float32 vectors, warms up, times
top-10 queries one by one and checks recall against an exact numpy search. It prints one line:

    RESULT {json}

`--summarize FILE...` reads the RESULT lines of earlier runs (text files) and prints a table per dimension
plus `RATIO <platform> dim=<d> native/jvm p50=<x>`, the ratio of the medians over the rounds.

This is an indicative factor from a shared CI runner, not a timed claim. Query time includes the binding's
own cost (JPype for the JVM wheel, ctypes for the native wheel), the same for every query.
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time

TYPE = "V"
PROP = "emb"
K = 10


def provider_info(arcadedb):
    """The jvector provider of this process, as far as the build exposes it."""
    info = {
        "env_ARCADEDB_VECTOR_PROVIDER": os.environ.get(
            "ARCADEDB_VECTOR_PROVIDER", "(unset)"
        )
    }
    try:
        if arcadedb.__backend__ == "native":
            from arcadedb._native import lib

            d = lib.diag()
            info["provider"] = d.get("provider")
            info["vector_module"] = d.get("vector_module")
        else:
            import jpype

            cls = jpype.JClass("io.github.jbellis.jvector.vector.VectorizationProvider")
            info["provider"] = str(cls.getInstance().getClass().getName())
    except Exception as e:  # diagnostics only, never fail the run
        info["provider_error"] = f"{type(e).__name__}: {e}"[:200]
    return info


def run(args):
    import numpy as np

    import arcadedb

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.rows, args.dim)).astype(np.float32)
    nq = max(args.queries, args.recall_queries)
    q = rng.standard_normal((nq, args.dim)).astype(np.float32)

    # exact top-K by cosine on the first recall_queries queries
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    qn = q[: args.recall_queries] / np.linalg.norm(
        q[: args.recall_queries], axis=1, keepdims=True
    )
    exact = np.argsort(-(qn @ xn.T), axis=1)[:, :K]

    workdir = tempfile.mkdtemp(prefix="xplat-vec-")
    db = None
    try:
        db = arcadedb.create_database(os.path.join(workdir, "db"))
        db.command("sql", f"CREATE DOCUMENT TYPE {TYPE}")
        db.command("sql", f"CREATE PROPERTY {TYPE}.id INTEGER")
        db.command("sql", f"CREATE PROPERTY {TYPE}.{PROP} ARRAY_OF_FLOATS")
        db.insert_many(TYPE, ({"id": i, PROP: x[i]} for i in range(args.rows)))

        t0 = time.perf_counter()
        db.command(
            "sql",
            f"CREATE INDEX ON {TYPE} ({PROP}) LSM_VECTOR METADATA "
            + '{"dimensions": %d, "similarity": "COSINE"}' % args.dim,
        )
        index = db.schema.get_vector_index(TYPE, PROP)
        if hasattr(index, "build_graph_now"):
            index.build_graph_now()
        build_s = time.perf_counter() - t0

        sql = f"SELECT id FROM (SELECT expand(vectorNeighbors('{TYPE}[{PROP}]', ?, ?)))"

        def search(vec):
            rows = db.query("sql", sql, arcadedb.to_java_float_array(vec), K).to_list()
            return [int(r["id"]) for r in rows]

        for i in range(args.warmup):
            search(q[i % nq])
        lat = []
        for i in range(args.queries):
            t = time.perf_counter_ns()
            search(q[i % nq])
            lat.append((time.perf_counter_ns() - t) / 1e6)

        hit = 0
        for i in range(args.recall_queries):
            hit += len(set(search(q[i])) & set(int(v) for v in exact[i]))
        recall = hit / (args.recall_queries * K)
    finally:
        try:
            if db is not None:
                db.close()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    lat.sort()
    result = {
        "platform": args.platform or f"{platform.system()}-{platform.machine()}",
        "backend": arcadedb.__backend__,
        "version": arcadedb.__version__,
        "round": args.round,
        "dim": args.dim,
        "rows": args.rows,
        "queries": args.queries,
        "build_s": round(build_s, 3),
        "p50_ms": round(statistics.median(lat), 4),
        "mean_ms": round(statistics.fmean(lat), 4),
        "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 4),
        "recall_at_10": round(recall, 4),
    }
    result.update(provider_info(arcadedb))
    # the engine logs to stdout without a final newline, so start on a fresh line
    print("\nRESULT " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def summarize(files):
    results = []
    for name in files:
        with open(name, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                i = line.find("RESULT {")
                if i >= 0:
                    results.append(json.loads(line[i + len("RESULT ") :]))
    if not results:
        print("no RESULT lines found")
        return 1
    groups = {}
    for r in results:
        groups.setdefault((r["platform"], r["dim"], r["backend"]), []).append(r)
    print(
        "platform dim backend n build_s(med) p50_ms(med) mean_ms(med) recall(min) provider"
    )
    for (plat, dim, backend), rs in sorted(groups.items()):
        print(
            plat,
            dim,
            backend,
            len(rs),
            round(statistics.median(r["build_s"] for r in rs), 3),
            round(statistics.median(r["p50_ms"] for r in rs), 4),
            round(statistics.median(r["mean_ms"] for r in rs), 4),
            min(r["recall_at_10"] for r in rs),
            rs[0].get("provider"),
        )
    for plat, dim in sorted({(k[0], k[1]) for k in groups}):
        nat = groups.get((plat, dim, "native"))
        jvm = groups.get((plat, dim, "jvm"))
        if not nat or not jvm:
            print(
                f"RATIO {plat} dim={dim} native/jvm p50=missing (native={bool(nat)} jvm={bool(jvm)})"
            )
            continue
        n = statistics.median(r["p50_ms"] for r in nat)
        j = statistics.median(r["p50_ms"] for r in jvm)
        nb = statistics.median(r["build_s"] for r in nat)
        jb = statistics.median(r["build_s"] for r in jvm)
        print(
            f"RATIO {plat} dim={dim} native/jvm p50={n / j:.2f} (build={nb / jb:.2f})"
        )
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--recall-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--platform", default="")
    ap.add_argument("--summarize", nargs="+", metavar="FILE")
    args = ap.parse_args()
    if args.summarize:
        return summarize(args.summarize)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
