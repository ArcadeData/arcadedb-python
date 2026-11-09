"""Ingest integrity (ArcadeData/arcadedb-python#336): no ingest path may lose, duplicate or silently alter a row, and a test must be
able to catch it.

``tests/ingest_integrity.py`` holds the one check. After every ingest call it compares the stored type with the rows that were sent:
the record count (read three ways), an order-independent digest of every column (read two ways), the key of every row exactly once,
and all of it again after the database is closed straight after the call and opened from disk. Both wheels run this file: the API is
the same, only the seams in ``test_native_ingest_integrity_faults.py`` and ``test_jvm_ingest_integrity_faults.py`` differ.

* ``test_every_size_at_every_boundary`` loads each ingest path (``insert_many`` from a list and from a generator, ``insert_columns``,
  ``insert_arrow``, ``insert_dataframe``, ``graph_batch().create_vertices``, ``create_record`` and ``command`` on the async executor,
  ``append_samples``; each sequential and, where it exists, with ``parallel=True``) at 0, 1 and 2 rows and one either side of every
  multiple of the chunk (7 rows) and of ``commit_every`` / ``batch_size`` (5 rows), and at 255, 256 and 257 rows (the first reply of
  the native read path). The chunk sizes are made small so the boundaries are cheap; one load at the real 100,000 chunk is separate.
* the edge, thread, failure, async and ``close()`` tests below check the cases the issue lists, with the documented outcome exact.
* ``test_mutation_*`` run the same code with a rows-dropped (or duplicated, or altered, or uncommitted) fault in the path and assert that the check
  reports it, so each family is shown able to fail.

``wait_completion()`` can return while the executor still reports itself as processing (#210), so no test here asserts idleness:
they assert the stored data.
"""

import dataclasses
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import arcadedb
from arcadedb.exceptions import ArcadeDBError
from tests.ingest_integrity import (
    CHUNK,
    COMMIT,
    FOUND,
    MUTANTS,
    NO_COMMIT,
    IntegrityError,
    problems_for_edges,
    problems_for_type,
    verify,
)

FIRST_REPLY = 256  # rows in the first reply of the native read path
MUTATION_SIZE = 25

SCHEMA = {"k": "int", "val": "float", "flag": "bool", "txt": "str"}
PROPS = {"k": "LONG", "val": "DOUBLE", "flag": "BOOLEAN", "txt": "STRING"}
TS_SCHEMA = {"host": "str", "val": "float"}


def ladder(units, extra=()):
    sizes = {0, 1, 2, *extra}
    for unit in units:
        for k in (1, 2, 3):
            sizes.update((k * unit - 1, k * unit, k * unit + 1))
    return sorted(sizes)


SIZES = ladder((CHUNK, COMMIT), extra=(FIRST_REPLY - 1, FIRST_REPLY, FIRST_REPLY + 1))


def make_rows(n, start=0):
    return [
        {
            "k": i,
            "val": (i * 37 % 1009) / 7.0 + i,
            "flag": i % 3 == 0,
            "txt": None if i % 7 == 3 else "row-%d-%s" % (i, "é" * (i % 4)),
        }
        for i in range(start, start + n)
    ]


def make_ts_rows(n):
    return [{"host": "h%d" % (i % 3), "val": float(i)} for i in range(n)]


def make_type(db, name, kind="DOCUMENT", buckets=None, unique=False):
    ddl = f"CREATE {kind} TYPE {name}"
    if buckets:
        ddl += f" BUCKETS {buckets}"
    db.command("sql", ddl)
    for prop, declared in PROPS.items():
        db.command("sql", f"CREATE PROPERTY {name}.{prop} {declared}")
    if unique:
        db.command("sql", f"CREATE INDEX ON {name} (k) UNIQUE")


def make_ts_type(db, name):
    db.command(
        "sql",
        f"CREATE TIMESERIES TYPE {name} TIMESTAMP ts TAGS (host STRING) FIELDS (val DOUBLE) SHARDS 2",
    )


# -- the ingest paths ---------------------------------------------------------------------------------------------------------


def _numpy_columns(rows):
    return {
        "k": np.array([r["k"] for r in rows], dtype=np.int64),
        "val": np.array([r["val"] for r in rows], dtype=np.float64),
        "flag": np.array([r["flag"] for r in rows], dtype=np.bool_),
        "txt": [r["txt"] for r in rows],
    }


def _list_columns(rows):
    return {n: [r[n] for r in rows] for n in SCHEMA}


ARROW_SCHEMA = pa.schema(
    [
        ("k", pa.int64()),
        ("val", pa.float64()),
        ("flag", pa.bool_()),
        ("txt", pa.string()),
    ]
)


def _arrow(rows):
    return pa.Table.from_pylist(list(rows), schema=ARROW_SCHEMA)


def _frame(rows):
    return pd.DataFrame(
        {
            "k": np.array([r["k"] for r in rows], dtype=np.int64),
            "val": np.array([r["val"] for r in rows], dtype=np.float64),
            "flag": np.array([r["flag"] for r in rows], dtype=np.bool_),
            "txt": pd.array([r["txt"] for r in rows], dtype="string"),
        }
    )


def _doc_of(db, name, row):
    doc = db.new_document(name)
    for key, value in row.items():
        if value is not None:
            doc.set(key, value)
    return doc


@dataclasses.dataclass
class Ctx:
    """The async executor of a database (there is one per database) and what its executor-wide handler was told."""

    ex: Any
    errors: list


def _executor(level=None):
    def setup(db):
        ex = db.async_executor()
        if level:
            ex.set_parallel_level(level)
        ex.set_commit_every(COMMIT)
        errors = []
        ex.on_error(errors.append)
        return Ctx(ex, errors)

    return setup


def _no_setup(db):
    return None


def _teardown(ctx):
    if ctx is not None:
        ctx.ex.wait_completion()
        ctx.ex.close()


def _ing_many(ctx, db, name, rows):
    return db.insert_many(name, list(rows), commit_every=COMMIT)


def _ing_many_gen(ctx, db, name, rows):
    return db.insert_many(name, (r for r in rows), commit_every=COMMIT)


def _ing_many_par(ctx, db, name, rows):
    return db.insert_many(name, list(rows), parallel=True)


def _ing_cols(ctx, db, name, rows):
    return db.insert_columns(name, _numpy_columns(rows), commit_every=COMMIT)


def _ing_cols_par(ctx, db, name, rows):
    return db.insert_columns(name, _numpy_columns(rows), parallel=True)


def _ing_arrow(ctx, db, name, rows):
    return db.insert_arrow(name, _arrow(rows), batch_size=COMMIT)


def _ing_arrow_par(ctx, db, name, rows):
    return db.insert_arrow(name, _arrow(rows), batch_size=COMMIT, parallel=True)


def _ing_df(ctx, db, name, rows):
    return db.insert_dataframe(name, _frame(rows), batch_size=COMMIT)


def _ing_df_par(ctx, db, name, rows):
    return db.insert_dataframe(name, _frame(rows), batch_size=COMMIT, parallel=True)


def _ing_async_create(ctx, db, name, rows):
    for row in rows:
        ctx.ex.create_record(_doc_of(db, name, row))
    ctx.ex.wait_completion()
    return len(rows)


def _ing_async_command(ctx, db, name, rows):
    head = "INSERT INTO " + name
    with_text = head + " SET k = :k, val = :val, flag = :flag, txt = :txt"
    without_text = head + " SET k = :k, val = :val, flag = :flag"
    for row in rows:
        if row["txt"] is None:
            ctx.ex.command(
                "sql", without_text, k=row["k"], val=row["val"], flag=row["flag"]
            )
        else:
            ctx.ex.command("sql", with_text, **row)
    ctx.ex.wait_completion()
    return len(rows)


def _ing_vertices(ctx, db, name, rows):
    with db.graph_batch(commit_every=CHUNK, batch_size=CHUNK) as batch:
        rids = batch.create_vertices(name, [dict(r) for r in rows])
    if len(rids) != len(rows) or len(set(rids)) != len(rids):
        raise AssertionError(
            f"create_vertices returned {len(rids)} RIDs ({len(set(rids))} distinct) for {len(rows)} rows"
        )
    return len(rids)


def _ing_timeseries(ctx, db, name, rows):
    n = len(rows)
    ts = (np.arange(n, dtype=np.int64) + 1) * 1000
    ctx.ex.append_samples(
        name,
        ts,
        [r["host"] for r in rows],
        np.array([r["val"] for r in rows], dtype=np.float64),
    )
    ctx.ex.wait_completion()
    return n


@dataclasses.dataclass(frozen=True)
class Family:
    id: str
    create: Callable
    ingest: Callable
    setup: Callable = _no_setup
    schema: Optional[dict] = None
    rows: Callable = make_rows
    key: str = "k"
    min_n: int = 0
    sync_tx: bool = (
        False  # joins a transaction the caller opened (the durability mutant needs one)
    )

    def __post_init__(self):
        if self.schema is None:
            object.__setattr__(self, "schema", SCHEMA)


def _doc(buckets=None):
    return lambda db, name: make_type(db, name, buckets=buckets)


def _vertex(db, name):
    make_type(db, name, kind="VERTEX")


FAMILIES = [
    Family("many", _doc(), _ing_many, sync_tx=True),
    Family("many_gen", _doc(), _ing_many_gen, sync_tx=True),
    Family("many_par", _doc(4), _ing_many_par, setup=_executor()),
    Family("cols", _doc(), _ing_cols, sync_tx=True),
    Family("cols_par", _doc(4), _ing_cols_par, setup=_executor()),
    Family("arrow", _doc(), _ing_arrow, sync_tx=True),
    Family("arrow_par", _doc(4), _ing_arrow_par, setup=_executor()),
    Family("arrow_vertex", _vertex, _ing_arrow),
    Family("df", _doc(), _ing_df, sync_tx=True),
    Family("df_par", _doc(4), _ing_df_par, setup=_executor()),
    Family("async_create", _doc(), _ing_async_create, setup=_executor(1)),
    Family("async_create_par", _doc(4), _ing_async_create, setup=_executor(4)),
    Family("async_command", _doc(), _ing_async_command, setup=_executor(1)),
    Family("vertices", _vertex, _ing_vertices),
    Family(
        "timeseries",
        lambda db, name: make_ts_type(db, name),
        _ing_timeseries,
        setup=_executor(),
        schema=TS_SCHEMA,
        rows=make_ts_rows,
        key="val",
        min_n=1,
    ),
]
BY_ID = {f.id: f for f in FAMILIES}


def run_cases(store, fam, sizes, mutant=None):
    """Load one type per size through ``fam``, check each after the call, close and reopen, check each again.

    ``mutant`` names a fault: one of ``MUTANTS`` rewrites the rows the path is given (the check still compares with the rows that
    were generated), ``NO_COMMIT`` loads inside a transaction nobody commits. Raises ``IntegrityError`` with every problem found.
    """
    problems, cases = [], []
    db = store.db
    ctx = fam.setup(db)
    try:
        for n in sizes:
            if n < fam.min_n:
                continue
            name = f"{fam.id}_{n}"
            rows = fam.rows(n)
            sent = MUTANTS[mutant](rows) if mutant in MUTANTS else rows
            fam.create(db, name)
            if mutant == NO_COMMIT:
                db.begin()
            try:
                got = fam.ingest(ctx, db, name, sent)
                if got != len(sent):
                    problems.append(
                        f"{FOUND} n={n}: the call returned {got}, {len(sent)} rows were given"
                    )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - reported with the size, the other sizes still run
                problems.append(f"RAISED n={n}: {type(exc).__name__}: {exc}")
                if db.is_transaction_active() and mutant != NO_COMMIT:
                    db.rollback()
            if ctx is not None and ctx.errors:
                problems.append(
                    f"{FOUND} n={n}: the executor's error handler was told {len(ctx.errors)} time(s): {ctx.errors[:2]}"
                )
                ctx.errors.clear()
            problems += problems_for_type(
                db, name, rows, fam.schema, fam.key, f"n={n} after the call"
            )
            cases.append((name, n, rows))
    finally:
        _teardown(ctx)
    store.reopen()
    for name, n, rows in cases:
        problems += problems_for_type(
            store.db, name, rows, fam.schema, fam.key, f"n={n} after reopen"
        )
    if problems:
        raise IntegrityError(f"{fam.id}: " + "\n".join(problems))


# -- sizes at every boundary --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.id)
def test_every_size_at_every_boundary(store, small_chunks, fam):
    run_cases(store, fam, SIZES)


def test_one_load_across_the_real_100000_chunk(store):
    db = store.db
    make_type(db, "Big", kind="VERTEX")
    n = 100_001
    rows = make_rows(n)
    with db.graph_batch() as batch:
        rids = batch.create_vertices("Big", rows)
    assert len(rids) == n and len(set(rids)) == n
    problems = problems_for_type(db, "Big", rows, SCHEMA, "k", "after the call")
    store.reopen()
    problems += problems_for_type(store.db, "Big", rows, SCHEMA, "k", "after reopen")
    assert not problems, "\n".join(problems)


# -- reading back at every batch size (the read path found the loss of #331) ----------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 7, 255, 256, 257, 600, 25_000])
def test_reading_back_every_row_at_every_batch_size(store, batch_size):
    db = store.db
    make_type(db, "Wide")
    rows = make_rows(600)
    assert db.insert_many("Wide", rows) == 600
    names = list(SCHEMA)
    select = "SELECT FROM Wide"
    cols = db.query("sql", select).to_columns(batch_size=batch_size, columns=names)
    from_columns = [dict(zip(names, vals)) for vals in zip(*(cols[n] for n in names))]
    table = db.query("sql", select).to_arrow(batch_size=batch_size, columns=names)
    problems = verify(
        f"batch_size={batch_size}",
        rows,
        SCHEMA,
        "k",
        {"to_columns rows": len(from_columns), "to_arrow rows": table.num_rows},
        {"to_columns()": from_columns, "to_arrow()": table.to_pylist()},
    )
    assert not problems, "\n".join(problems)


# -- edges: count and endpoint pairs ------------------------------------------------------------------------------------------

EDGE_SIZES = ladder((CHUNK, COMMIT))
VERTICES = 11


def _endpoints(m):
    return [(i % VERTICES, (3 * i + 1) % VERTICES) for i in range(m)]


def _load_edges(db, rids, etype, pairs, weights):
    batch = db.graph_batch(commit_every=CHUNK, batch_size=CHUNK)
    try:
        batch.new_edges(
            [rids[a] for a, _ in pairs],
            etype,
            [rids[b] for _, b in pairs],
            properties=None if weights is None else [{"w": w} for w in weights],
        )
    finally:
        batch.close()


def _vertex_rids(db):
    make_type(db, "Node", kind="VERTEX")
    with db.graph_batch(commit_every=CHUNK, batch_size=CHUNK) as batch:
        return batch.create_vertices("Node", make_rows(VERTICES))


@pytest.mark.parametrize("with_weights", [False, True], ids=["plain", "weighted"])
def test_new_edges_store_every_edge_between_its_endpoints(
    store, small_chunks, with_weights
):
    db = store.db
    rids = _vertex_rids(db)
    problems, cases = [], []
    for m in EDGE_SIZES:
        etype = f"Link{m}"
        db.command("sql", f"CREATE EDGE TYPE {etype}")
        pairs = _endpoints(m)
        weights = [i + 0.25 for i in range(m)] if with_weights else None
        _load_edges(db, rids, etype, pairs, weights)
        want = [
            {"a": a, "b": b, "w": None if weights is None else weights[i]}
            for i, (a, b) in enumerate(pairs)
        ]
        problems += problems_for_edges(db, etype, want, f"m={m} after close()")
        cases.append((etype, m, want))
    store.reopen()
    for etype, m, want in cases:
        problems += problems_for_edges(store.db, etype, want, f"m={m} after reopen")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "fault", ["drop_the_last_edge", "swap_two_destinations", "store_one_edge_twice"]
)
def test_mutation_edges(store, small_chunks, fault):
    db = store.db
    rids = _vertex_rids(db)
    db.command("sql", "CREATE EDGE TYPE Mut")
    m = MUTATION_SIZE
    pairs = _endpoints(m)
    weights = [i + 0.25 for i in range(m)]
    sent = list(pairs)
    sent_weights = list(weights)
    if fault == "drop_the_last_edge":
        sent, sent_weights = sent[:-1], sent_weights[:-1]
    elif fault == "store_one_edge_twice":
        sent.append(sent[4])
        sent_weights.append(sent_weights[4])
    else:
        (a1, b1), (a2, b2) = sent[1], sent[8]
        sent[1], sent[8] = (a1, b2), (a2, b1)
    _load_edges(db, rids, "Mut", sent, sent_weights)
    want = [{"a": a, "b": b, "w": weights[i]} for i, (a, b) in enumerate(pairs)]
    problems = problems_for_edges(db, "Mut", want, "after close()")
    assert problems and all(FOUND in p for p in problems), problems


# -- the mutation check: every family, every fault ---------------------------------------------------------------------------


def _mutation_cases():
    for fam in FAMILIES:
        for mutant in MUTANTS:
            yield pytest.param(fam, mutant, id=f"{fam.id}-{mutant}")
        if fam.sync_tx:
            yield pytest.param(fam, NO_COMMIT, id=f"{fam.id}-{NO_COMMIT}")


@pytest.mark.parametrize("fam, mutant", list(_mutation_cases()))
def test_mutation_the_check_reports_the_fault(store, small_chunks, fam, mutant):
    with pytest.raises(IntegrityError) as found:
        run_cases(store, fam, [MUTATION_SIZE], mutant=mutant)
    text = str(found.value)
    assert FOUND in text, text
    if mutant == NO_COMMIT:
        # the rows are visible in this session, so only the reopen can tell
        assert "after the call" not in text, text
        assert "after reopen" in text, text
    else:
        assert "after the call" in text and "after reopen" in text, text


def _chunk_seam():
    """The function that stores one chunk of insert_many on this wheel: (class, name, arguments before the chunk)."""
    if arcadedb.__backend__ == "native":
        from arcadedb._native import database

        return database.NativeDatabase, "_insert_chunk"
    from arcadedb import core

    return core.Database, "_insert_many_chunk"


@pytest.mark.parametrize(
    "fault",
    ["drop_the_last_row_of_a_chunk", "skip_a_chunk", "store_a_row_twice"],
)
def test_mutation_inside_insert_many(store, small_chunks, monkeypatch, fault):
    owner, name = _chunk_seam()
    real = getattr(owner, name)
    seen = []

    def faulty(self, type_name, *args):
        *lead, chunk = args
        seen.append(len(chunk))
        if len(seen) == 2:
            if fault == "skip_a_chunk":
                return None
            if fault == "drop_the_last_row_of_a_chunk":
                chunk = chunk[:-1]
            else:
                chunk = chunk + [chunk[-1]]
        return real(self, type_name, *lead, chunk)

    monkeypatch.setattr(owner, name, faulty)
    with pytest.raises(IntegrityError) as found:
        run_cases(store, BY_ID["many"], [MUTATION_SIZE])
    assert FOUND in str(found.value), found.value
    assert len(seen) >= 3, f"the fault was not in the path: {seen}"


def test_mutation_an_async_load_that_returns_before_the_wait(store):
    """The executor is parked inside a callback, so the queued records cannot have run when the check looks."""
    db = store.db
    make_type(db, "Early")
    rows = make_rows(MUTATION_SIZE)
    ex = db.async_executor().set_parallel_level(1).set_commit_every(COMMIT)
    started, release = threading.Event(), threading.Event()

    def park(_row):
        started.set()
        release.wait(60)

    ex.query("sql", "SELECT 1 AS one", park)
    assert started.wait(60), "the parking callback never ran"
    try:
        for row in rows:
            ex.create_record(_doc_of(db, "Early", row))
        # no wait_completion(): this is the fault
        problems = problems_for_type(db, "Early", rows, SCHEMA, "k", "before the wait")
    finally:
        release.set()
        ex.wait_completion()
        ex.close()
    assert problems and all(FOUND in p for p in problems), problems
    # and once the wait is done the same check passes, so the fault was the missing wait
    assert not problems_for_type(db, "Early", rows, SCHEMA, "k", "after the wait")


# -- several writer threads ---------------------------------------------------------------------------------------------------


def _run_threads(jobs):
    errors = []

    def run(job):
        try:
            job()
        except BaseException as exc:  # noqa: BLE001 - handed to the test thread
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(job,)) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(180)
    assert not [t for t in threads if t.is_alive()], "a writer thread never finished"
    return errors


class Reader:
    """Counts the types in a loop while writers run; the counts of one type may only grow."""

    def __init__(self, db, names):
        self.db = db
        self.series = {n: [] for n in names}
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run)

    def count(self, name):
        sql = "SELECT count(*) AS n FROM " + name  # nosec B608
        return int(self.db.query("sql", sql).to_list()[0]["n"])

    def run(self):
        try:
            while not self.stop.is_set():
                for name, values in self.series.items():
                    values.append(self.count(name))
                time.sleep(0.002)
        except BaseException as exc:  # noqa: BLE001 - handed to the test thread
            self.error = exc

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.stop.set()
        self.thread.join(60)

    def problems(self, totals):
        found = []
        if self.error is not None:
            found.append(f"RAISED the reader thread: {self.error!r}")
        for name, values in self.series.items():
            if any(b < a for a, b in zip(values, values[1:])):
                found.append(f"{FOUND} the count of {name} went down: {values}")
            if values and values[-1] > totals[name]:
                found.append(
                    f"{FOUND} the count of {name} reached {values[-1]}, above the {totals[name]} sent"
                )
            if self.count(name) != totals[name]:
                found.append(
                    f"{FOUND} the final count of {name} is {self.count(name)}, {totals[name]} were sent"
                )
        return found


def test_writer_threads_on_one_type(store, small_chunks):
    db = store.db
    make_type(db, "OneType", buckets=4)
    writers, per = 4, 60
    parts = [make_rows(per, start=w * per) for w in range(writers)]

    def job(rows):
        return lambda: db.run_in_transaction(
            lambda: db.insert_many("OneType", rows), retries=200, backoff_s=0.002
        )

    with Reader(db, ["OneType"]) as reader:
        errors = _run_threads([job(rows) for rows in parts])
    assert not errors, errors
    everything = [r for rows in parts for r in rows]
    problems = reader.problems({"OneType": len(everything)})
    problems += problems_for_type(db, "OneType", everything, SCHEMA, "k", "after join")
    store.reopen()
    problems += problems_for_type(
        store.db, "OneType", everything, SCHEMA, "k", "after reopen"
    )
    assert not problems, "\n".join(problems)


def test_writer_threads_on_different_types_and_paths(store, small_chunks):
    db = store.db
    paths = [BY_ID["many"], BY_ID["cols"], BY_ID["arrow"], BY_ID["df"]]
    names = [f"Thread{i}" for i in range(len(paths))]
    sent = {name: make_rows(90, start=1000 * i) for i, name in enumerate(names)}
    for name in names:
        make_type(db, name)

    def job(fam, name):
        return lambda: fam.ingest(None, db, name, sent[name])

    with Reader(db, names) as reader:
        errors = _run_threads([job(f, n) for f, n in zip(paths, names)])
    assert not errors, errors
    problems = reader.problems({n: len(sent[n]) for n in names})
    for name in names:
        problems += problems_for_type(db, name, sent[name], SCHEMA, "k", "after join")
    store.reopen()
    for name in names:
        problems += problems_for_type(
            store.db, name, sent[name], SCHEMA, "k", "after reopen"
        )
    assert not problems, "\n".join(problems)


# -- a bad row in the middle of a batch ---------------------------------------------------------------------------------------

BAD_ROWS = {
    "duplicate_key": lambda rows, p: {**rows[p], "k": rows[0]["k"]},
    "wrong_declared_type": lambda rows, p: {**rows[p], "val": "not a number"},
    "refused_value": lambda rows, p: {**rows[p], "extra": 2**70},
}
POSITIONS = [1, 4, 5, 7, 13]  # 14 rows, commit_every 5: batches [0-4] [5-9] [10-13]

FAIL_CALLS = {
    "insert_many": lambda db, name, sent: db.insert_many(
        name, sent, commit_every=COMMIT
    ),
    "insert_columns": lambda db, name, sent: db.insert_columns(
        name, _list_columns(sent), commit_every=COMMIT
    ),
}


def _bad_cases():
    for path in FAIL_CALLS:
        for bad in BAD_ROWS:
            if bad == "refused_value" and path != "insert_many":
                continue  # the row's extra property has no column to go into
            for p in POSITIONS:
                yield pytest.param(path, bad, p, id=f"{path}-{bad}-row{p}")


@pytest.mark.parametrize("path, bad, p", list(_bad_cases()))
def test_a_bad_row_keeps_the_committed_batches_and_nothing_else(
    store, small_chunks, path, bad, p
):
    """Documented: rows already committed stay committed, the batch that holds the bad row is rolled back, the error is raised."""
    db = store.db
    make_type(db, "Bad", unique=True)
    rows = make_rows(14)
    sent = list(rows)
    sent[p] = BAD_ROWS[bad](rows, p)
    with pytest.raises((ArcadeDBError, OverflowError)):
        FAIL_CALLS[path](db, "Bad", sent)
    assert not db.is_transaction_active()
    kept = rows[: (p // COMMIT) * COMMIT]
    problems = problems_for_type(db, "Bad", kept, SCHEMA, "k", "after the failed call")
    store.reopen()
    problems += problems_for_type(store.db, "Bad", kept, SCHEMA, "k", "after reopen")
    assert not problems, "\n".join(problems)


ARROW_CALLS = {
    "insert_arrow": lambda db, name, rows: db.insert_arrow(
        name, _arrow(rows), batch_size=COMMIT
    ),
    "insert_dataframe": lambda db, name, rows: db.insert_dataframe(
        name, _frame(rows), batch_size=COMMIT
    ),
}


@pytest.mark.parametrize("path", list(ARROW_CALLS))
@pytest.mark.parametrize("p", POSITIONS)
def test_arrow_and_dataframe_batches_before_a_duplicate_stay_committed(store, path, p):
    db = store.db
    make_type(db, "ArrowDup", unique=True)
    rows = make_rows(14)
    sent = list(rows)
    sent[p] = BAD_ROWS["duplicate_key"](rows, p)
    with pytest.raises(ArcadeDBError) as err:
        ARROW_CALLS[path](db, "ArrowDup", sent)
    committed = (p // COMMIT) * COMMIT
    if committed:
        assert (
            f"the {committed} row(s) of the batches before this one were committed"
            in str(err.value)
        ), err.value
    else:
        assert "batches before this one" not in str(err.value), err.value
    kept = rows[:committed]
    problems = problems_for_type(db, "ArrowDup", kept, SCHEMA, "k", "after the call")
    store.reopen()
    problems += problems_for_type(
        store.db, "ArrowDup", kept, SCHEMA, "k", "after reopen"
    )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", list(ARROW_CALLS))
def test_a_column_the_declared_type_refuses_stores_nothing(store, path):
    """Documented: a TypeError before any row is written."""
    db = store.db
    make_type(db, "Refused")
    rows = make_rows(12)
    text_rows = [{**r, "val": str(r["val"])} for r in rows]
    schema = pa.schema(
        [
            ("k", pa.int64()),
            ("val", pa.string()),
            ("flag", pa.bool_()),
            ("txt", pa.string()),
        ]
    )
    if path == "insert_arrow":
        with pytest.raises(TypeError):
            db.insert_arrow(
                "Refused", pa.Table.from_pylist(text_rows, schema=schema), batch_size=3
            )
    else:
        frame = _frame(rows)
        frame["val"] = pd.array([r["val"] for r in text_rows], dtype="string")
        with pytest.raises(TypeError):
            db.insert_dataframe("Refused", frame, batch_size=3)
    problems = problems_for_type(db, "Refused", [], SCHEMA, "k", "after the call")
    store.reopen()
    problems += problems_for_type(store.db, "Refused", [], SCHEMA, "k", "after reopen")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("p", POSITIONS)
def test_create_vertices_chunks_before_a_duplicate_stay_stored(store, small_chunks, p):
    db = store.db
    make_type(db, "BadV", kind="VERTEX", unique=True)
    rows = make_rows(14)
    sent = [dict(r) for r in rows]
    sent[p] = BAD_ROWS["duplicate_key"](rows, p)
    batch = db.graph_batch(commit_every=CHUNK, batch_size=CHUNK)
    try:
        with pytest.raises(ArcadeDBError):
            batch.create_vertices("BadV", sent)
    finally:
        batch.close()
    assert not db.is_transaction_active()
    kept = rows[: (p // CHUNK) * CHUNK]
    problems = problems_for_type(db, "BadV", kept, SCHEMA, "k", "after the call")
    store.reopen()
    problems += problems_for_type(store.db, "BadV", kept, SCHEMA, "k", "after reopen")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", ["insert_many", "insert_columns"])
def test_a_parallel_duplicate_is_reported_and_nothing_is_stored_twice(
    store, small_chunks, path
):
    """Documented for parallel=True: the failure is raised after the load, records other than the failed ones may be stored."""
    db = store.db
    make_type(db, "ParDup", buckets=4, unique=True)
    db.async_executor().set_commit_every(COMMIT)
    rows = make_rows(40)
    sent = list(rows)
    sent[30] = BAD_ROWS["duplicate_key"](rows, 30)
    with pytest.raises(ArcadeDBError, match="failed record"):
        if path == "insert_many":
            db.insert_many("ParDup", sent, parallel=True)
        else:
            db.insert_columns("ParDup", _numpy_columns(sent), parallel=True)

    def keys(database):
        stored = database.query("sql", "SELECT k FROM ParDup").to_list()
        return [int(r["k"]) for r in stored]

    before = keys(db)
    assert len(before) == len(set(before)), "a key was stored twice"
    assert set(before) <= {r["k"] for r in rows}, "a key was stored that was never sent"
    assert db.count_type("ParDup") == len(before)
    store.reopen()
    after = keys(store.db)
    assert sorted(after) == sorted(before), "the stored keys changed across a reopen"


# -- async: a rejected row and a failing callback must reach the caller ------------------------------------------------------


def test_async_rejected_rows_reach_their_error_callbacks(store):
    db = store.db
    make_type(db, "AsyncDup", unique=True)
    rows = make_rows(12)
    sent = list(rows)
    sent[5] = {**rows[5], "k": rows[0]["k"]}
    sent[9] = {**rows[9], "k": rows[2]["k"]}
    rejected, reported = [], []
    ex = db.async_executor().set_parallel_level(1).set_commit_every(1)
    ex.on_error(reported.append)
    for i, row in enumerate(sent):
        ex.create_record(
            _doc_of(db, "AsyncDup", row),
            error_callback=lambda _e, i=i: rejected.append(i),
        )
    ex.wait_completion()
    ex.close()
    assert sorted(rejected) == [5, 9]
    assert len(reported) == 2, reported
    kept = [r for i, r in enumerate(sent) if i not in (5, 9)]
    problems = problems_for_type(db, "AsyncDup", kept, SCHEMA, "k", "after wait")
    store.reopen()
    problems += problems_for_type(
        store.db, "AsyncDup", kept, SCHEMA, "k", "after reopen"
    )
    assert not problems, "\n".join(problems)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_a_callback_that_raises_loses_no_record_without_a_report(store):
    """A record is stored, or its own error callback was told: nothing is dropped silently when a per-record callback raises."""
    db = store.db
    make_type(db, "CbRaise")
    rows = make_rows(20)
    raised, reported = [], []

    def callback(i):
        def run(_record):
            if i % 4 == 1:
                raised.append(i)
                raise RuntimeError(f"callback {i}")

        return run

    ex = db.async_executor().set_parallel_level(1).set_commit_every(3)
    for i, row in enumerate(rows):
        ex.create_record(
            _doc_of(db, "CbRaise", row),
            callback=callback(i),
            error_callback=lambda _e, i=i: reported.append(i),
        )
    ex.wait_completion(60_000)
    ex.close()
    assert sorted(raised) == [1, 5, 9, 13, 17], "the callbacks did not all run"
    kept = [r for i, r in enumerate(rows) if i not in set(reported)]
    problems = problems_for_type(
        db,
        "CbRaise",
        kept,
        SCHEMA,
        "k",
        f"after wait, errors reported for {sorted(reported)}",
    )
    store.reopen()
    problems += problems_for_type(
        store.db, "CbRaise", kept, SCHEMA, "k", "after reopen"
    )
    assert not problems, "\n".join(problems)


# -- close() straight after the last call -------------------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["executor_close", "database_close"])
def test_close_straight_after_async_submissions_stores_all_or_raises(store, how):
    """Either every record is stored or an error is raised: never a silent partial."""
    db = store.db
    make_type(db, "Straight")
    n = 200
    rows = make_rows(n)
    ex = db.async_executor().set_parallel_level(1).set_commit_every(10)
    for row in rows:
        ex.create_record(_doc_of(db, "Straight", row))
    try:
        if how == "executor_close":
            ex.close()
        db.close()
    except Exception:  # noqa: BLE001 - an error was raised: not a silent partial
        return
    store.db = arcadedb.open_database(store.path)
    problems = problems_for_type(
        store.db, "Straight", rows, SCHEMA, "k", "after reopen"
    )
    assert not problems, "\n".join(problems)


def test_close_straight_after_the_last_edges_stores_them_all(store, small_chunks):
    db = store.db
    rids = _vertex_rids(db)
    db.command("sql", "CREATE EDGE TYPE Late")
    pairs = _endpoints(40)
    weights = [i + 0.25 for i in range(40)]
    batch = db.graph_batch(commit_every=CHUNK, batch_size=CHUNK)
    batch.new_edges(
        [rids[a] for a, _ in pairs],
        "Late",
        [rids[b] for _, b in pairs],
        properties=[{"w": w} for w in weights],
    )
    batch.close()
    store.reopen()
    want = [{"a": a, "b": b, "w": weights[i]} for i, (a, b) in enumerate(pairs)]
    problems = problems_for_edges(store.db, "Late", want, "after reopen")
    assert not problems, "\n".join(problems)
