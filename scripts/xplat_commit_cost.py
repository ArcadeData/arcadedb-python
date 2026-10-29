#!/usr/bin/env python3
"""Rough commit-cost probe for one installed `arcadedb` wheel (native or JVM).

For each WAL flush level (arcadedb.txWalFlush = 0 no flush, 1 flush without metadata, 2 full flush) it creates
a fresh database, sets the level, runs N single-row transactions and prints one line per level:

    COMMIT <platform> backend=<b> walflush=<n> p50_ms=<x> commit_only_p50_ms=<x> mean_ms=<x> p95_ms=<x> n=<N> readback=<...>

The timed unit is one transaction (begin, insert, commit); commit_only_p50_ms is the commit() call alone. On a platform whose WAL flush reaches stable storage, level 1 and 2 cost
the device's flush latency per commit; a level 2 that is about as cheap as level 0 means the flush does not wait
for the device. A shared CI runner is noisy: this is a fingerprint, not a timed claim.
"""

import argparse
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time

TYPE = "C"


def readback(db, arcadedb):
    """What the engine reports for the level, as far as this binding can read it (diagnostics only)."""
    out = []
    try:
        rows = db.query("sql", "SELECT FROM schema:database").to_list()
        for r in rows:
            s = str(r)
            i = s.find("txWalFlush")
            if i >= 0:
                out.append("schema=" + s[i : i + 40].replace(" ", ""))
    except Exception as e:
        out.append(f"schema_err={type(e).__name__}")
    try:
        if arcadedb.__backend__ != "native":
            out.append("java=" + str(db._java_db.getWALFlushSetting()))
    except Exception as e:
        out.append(f"java_err={type(e).__name__}")
    return ",".join(out) or "n/a"


def one_level(arcadedb, level, n, warmup, workdir):
    path = os.path.join(workdir, f"db{level}")
    db = arcadedb.create_database(path)
    try:
        db.command("sql", f"CREATE DOCUMENT TYPE {TYPE}")
        db.command("sql", f"CREATE PROPERTY {TYPE}.id INTEGER")
        # the SQL way, as asked, and the binding's documented call: both name the same level
        db.command("sql", f"ALTER DATABASE `arcadedb.txWalFlush` {level}")
        db.set_wal_flush(("no", "yes_nometadata", "yes_full")[level])
        rb = readback(db, arcadedb)
        lat = []
        commit_only = []
        for i in range(warmup + n):
            t0 = time.perf_counter_ns()
            db.begin()
            db.command("sql", f"INSERT INTO {TYPE} SET id = {i}")
            t1 = time.perf_counter_ns()
            db.commit()
            t2 = time.perf_counter_ns()
            if i >= warmup:
                # the whole single-row transaction: where the commit's work lands is the binding's and the
                # engine's business (an insert may stage its writes until commit), so the total is the figure
                lat.append((t2 - t0) / 1e6)
                commit_only.append((t2 - t1) / 1e6)
        count = db.query("sql", f"SELECT count(*) AS c FROM {TYPE}").to_list()[0]["c"]
        if int(count) != warmup + n:
            raise RuntimeError(f"expected {warmup + n} rows, found {count}")
    finally:
        db.close()
    lat.sort()
    return {
        "p50": statistics.median(lat),
        "mean": statistics.fmean(lat),
        "p95": lat[int(0.95 * (len(lat) - 1))],
        "commit_p50": statistics.median(commit_only),
        "readback": rb,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=500, help="timed transactions per level")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--platform", default="")
    ap.add_argument("--levels", default="0,1,2")
    args = ap.parse_args()

    import arcadedb

    plat = args.platform or f"{platform.system()}-{platform.machine()}"
    workdir = tempfile.mkdtemp(prefix="xplat-commit-")
    rc = 0
    try:
        for level in (int(v) for v in args.levels.split(",")):
            try:
                r = one_level(arcadedb, level, args.n, args.warmup, workdir)
            except Exception as e:
                print(
                    f"\nCOMMIT-FAILED {plat} backend={arcadedb.__backend__} walflush={level} "
                    f"{type(e).__name__}: {str(e)[:300]}",
                    flush=True,
                )
                rc = 1
                continue
            # the engine logs to stdout without a final newline, so start on a fresh line
            print(
                f"\nCOMMIT {plat} backend={arcadedb.__backend__} walflush={level} "
                f"p50_ms={r['p50']:.4f} commit_only_p50_ms={r['commit_p50']:.4f} mean_ms={r['mean']:.4f} p95_ms={r['p95']:.4f} "
                f"n={args.n} readback={r['readback']}",
                flush=True,
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
