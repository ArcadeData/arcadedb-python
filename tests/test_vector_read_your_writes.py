"""A vector search finds every vector whose transaction committed before the search started (ArcadeData/arcadedb-python#306).

An agent that remembers things writes a few vectors per turn and searches right after, from one asyncio task or from several. The
engine buffers the writes of a built index and rebuilds the graph in the background, so this pins what the agent may rely on: an
insert in one call and a search in the next call finds the vector, in the same task and in another one, also while the graph rebuilds.

The calls are the ones the concurrency page tells asyncio code to use: each blocking call runs in a worker thread through
``asyncio.to_thread``, so one task's insert and its next search can run on different threads. The corpus starts at BASE vectors, so the
rebuild threshold is max(100, 20 percent of the graph) = 100 pending changes, and the 200 inserts of a test cross it: the searches meet the
delta buffer first and the rebuilt graph later. ``ef_search`` is as large as the corpus on purpose, so a miss cannot be the approximate
walk of a large graph: every vector is the nearest neighbour of itself, at distance 0, and a different result means the write was not seen.
The same two tests run again through ``arcadedb.aio.AsyncDatabase`` (the facade over those calls), where an insert is a ``run_in_transaction`` and
a search is ``find_nearest``. The file uses no Java object, so it runs on the JVM build and on the native build alike.
"""

import asyncio
import math
import random

import numpy as np
import pytest

import arcadedb
from arcadedb.aio import AsyncDatabase

DIM = 8
BASE = 300
WRITERS = 4
PER_WRITER = 50
TOTAL = WRITERS * PER_WRITER
READERS = 3
EF_SEARCH = 400  # at least the number of vectors the corpus reaches (BASE + TOTAL = 500 is walked almost whole)


def _vec(i):
    """A deterministic unit vector whose direction is unique to `i` (see tests/test_vector_delta_visibility.py for why a unit vector)."""
    rnd = random.Random(i)  # nosec B311 - deterministic test corpus, not security
    v = [rnd.uniform(-1.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return np.asarray([x / norm for x in v], dtype=np.float32)


def _insert(db, i):
    """One committed insert, retried on a conflict with another writer (what the concurrency page recommends)."""

    def write():
        db.command("sql", "INSERT INTO Doc SET vid = ?, embedding = ?", i, _vec(i))

    db.run_in_transaction(write, retries=60)


def _nearest_ids(index, i):
    hits = index.find_nearest(_vec(i), k=5, ef_search=EF_SEARCH)
    return [record.get("vid") for record, _distance in hits]


def _count(db):
    return db.query("sql", "SELECT count(*) AS n FROM Doc").first().get("n")


@pytest.fixture
def corpus(tmp_path):
    db = arcadedb.create_database(str(tmp_path / "read_your_writes"))
    db.command("sql", "CREATE VERTEX TYPE Doc")
    db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    with db.transaction():
        for i in range(BASE):
            db.command("sql", "INSERT INTO Doc SET vid = ?, embedding = ?", i, _vec(i))
    index = db.create_vector_index("Doc", "embedding", dimensions=DIM)
    yield db, index
    db.drop()


def test_a_task_finds_the_vector_it_committed_in_the_call_before(corpus):
    db, index = corpus
    missed = []

    async def agent(n):
        for step in range(PER_WRITER):
            i = BASE + n * PER_WRITER + step
            await asyncio.to_thread(_insert, db, i)
            ids = await asyncio.to_thread(_nearest_ids, index, i)
            if ids[:1] != [i]:
                missed.append((i, ids))

    async def main():
        await asyncio.gather(*(agent(n) for n in range(WRITERS)))

    asyncio.run(main())
    assert (
        missed == []
    ), f"{len(missed)} of {TOTAL} committed vectors were not found by the next search: {missed[:5]}"
    assert _count(db) == BASE + TOTAL


def test_another_task_finds_the_vector_once_the_writing_task_has_committed_it(corpus):
    db, index = corpus
    missed = []

    async def writer(queue):
        for step in range(TOTAL):
            i = BASE + step
            await asyncio.to_thread(_insert, db, i)
            await queue.put(i)  # announced only after the commit returned
        for _ in range(READERS):
            await queue.put(None)

    async def reader(queue):
        while (i := await queue.get()) is not None:
            ids = await asyncio.to_thread(_nearest_ids, index, i)
            if ids[:1] != [i]:
                missed.append((i, ids))

    async def main():
        queue = asyncio.Queue()
        await asyncio.gather(writer(queue), *(reader(queue) for _ in range(READERS)))

    asyncio.run(main())
    assert (
        missed == []
    ), f"{len(missed)} of {TOTAL} committed vectors were not found by another task: {missed[:5]}"
    assert _count(db) == BASE + TOTAL


def _write(db, i):
    db.command("sql", "INSERT INTO Doc SET vid = ?, embedding = ?", i, _vec(i))


async def _facade_insert(adb, db, i):
    await adb.run_in_transaction(lambda: _write(db, i), retries=60)


async def _facade_nearest_ids(adb, index, i):
    hits = await adb.find_nearest(index, _vec(i), k=5, ef_search=EF_SEARCH)
    return [record.get("vid") for record, _distance in hits]


def test_through_the_facade_a_task_finds_the_vector_it_committed_in_the_call_before(
    corpus,
):
    db, index = corpus
    missed = []

    async def agent(adb, n):
        for step in range(PER_WRITER):
            i = BASE + n * PER_WRITER + step
            await _facade_insert(adb, db, i)
            ids = await _facade_nearest_ids(adb, index, i)
            if ids[:1] != [i]:
                missed.append((i, ids))

    async def main():
        async with AsyncDatabase(db, max_workers=8) as adb:
            await asyncio.gather(*(agent(adb, n) for n in range(WRITERS)))

    asyncio.run(main())
    assert (
        missed == []
    ), f"{len(missed)} of {TOTAL} committed vectors were not found by the next search: {missed[:5]}"
    assert _count(db) == BASE + TOTAL


def test_through_the_facade_another_task_finds_the_vector_once_the_writing_task_has_committed_it(
    corpus,
):
    db, index = corpus
    missed = []

    async def writer(adb, queue):
        for step in range(TOTAL):
            i = BASE + step
            await _facade_insert(adb, db, i)
            await queue.put(i)  # announced only after the commit returned
        for _ in range(READERS):
            await queue.put(None)

    async def reader(adb, queue):
        while (i := await queue.get()) is not None:
            ids = await _facade_nearest_ids(adb, index, i)
            if ids[:1] != [i]:
                missed.append((i, ids))

    async def main():
        async with AsyncDatabase(db, max_workers=8) as adb:
            queue = asyncio.Queue()
            await asyncio.gather(
                writer(adb, queue), *(reader(adb, queue) for _ in range(READERS))
            )

    asyncio.run(main())
    assert (
        missed == []
    ), f"{len(missed)} of {TOTAL} committed vectors were not found by another task: {missed[:5]}"
    assert _count(db) == BASE + TOTAL
