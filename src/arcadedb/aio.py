"""asyncio facade over the blocking calls of an open :class:`~arcadedb.Database` (``arcadedb.aio``).

An agent loop written with ``async def`` can call the bindings through :class:`AsyncDatabase` instead of wrapping every call in
``asyncio.to_thread`` itself. Each call runs the ordinary blocking call on a worker thread of the facade's own pool, so the event loop
keeps running while the engine works. It is a thin layer: no engine change, the same calls, the same errors.

Three rules shape it:

* **A transaction belongs to one thread.** The engine ties a transaction to the thread that began it, and two awaits of one task can
  run on two different workers, so there is no ``async with adb.transaction()``. :meth:`AsyncDatabase.run_in_transaction` takes a plain
  function and runs it, begin to commit and every retry, on one worker.
* **One pool, bounded.** The facade owns a :class:`concurrent.futures.ThreadPoolExecutor` of ``max_workers`` threads (default 8) and
  never takes the loop's default executor. One ``Database`` shared by several threads is supported: see the concurrency page.
* **Cancellation is not exposed.** Cancelling the task that awaits a call abandons the result. A call that already started on a worker
  finishes anyway (a transaction function still commits or rolls back on its own thread). A call that still waits for a worker does
  not run. :meth:`AsyncDatabase.close` waits for the calls that were submitted.

``AsyncDatabase`` and :class:`~arcadedb.AsyncExecutor` (``db.async_executor()``) are different things. ``AsyncExecutor`` is the engine's
own asynchronous API: statements are queued on the engine's threads, the call returns at once, the result arrives in a callback, and it
is meant for fire-and-forget bulk writes. ``AsyncDatabase`` is for ``async def`` code: every call is awaited, returns its own result or
raises in the awaiting task, and runs through the normal blocking API on a Python thread pool.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

__all__ = ["AsyncDatabase"]

_DEFAULT_MAX_WORKERS = 8


class AsyncDatabase:
    """Await the calls of an open ``Database`` from ``asyncio`` code.

    Args:
        db: An open :class:`~arcadedb.Database`. The facade does not close it; closing the facade does not either.
        max_workers: The number of worker threads, so the most calls that run at once. Default 8. It is a starting point, not a
            measured optimum: writers that touch the same records conflict, and ``run_in_transaction`` retries them.

    Use it as ``async with AsyncDatabase(db) as adb:`` or call ``await adb.close()`` yourself.
    """

    def __init__(self, db, max_workers: int = _DEFAULT_MAX_WORKERS):
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers < 1
        ):
            raise ValueError(f"max_workers must be an int >= 1, got {max_workers!r}")
        self._db = db
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="arcadedb-aio"
        )
        self._closed = False

    @property
    def database(self):
        """The wrapped ``Database``."""
        return self._db

    @property
    def max_workers(self) -> int:
        """The size of the worker pool."""
        return self._max_workers

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` was called."""
        return self._closed

    async def __aenter__(self) -> "AsyncDatabase":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` on a worker thread and return its result.

        This is the general call: every other method is this one with a database call in ``fn``. Use it for anything the facade does not
        wrap, for example ``await adb.run(lambda: db.query("sql", "SELECT count(*) AS n FROM Note").first().get("n"))``. The context
        variables of the calling task are visible to ``fn``, as with ``asyncio.to_thread``. An exception of ``fn`` is raised here.

        Raises:
            RuntimeError: The facade was closed.
        """
        if self._closed:
            raise RuntimeError("this AsyncDatabase is closed")
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        call = functools.partial(context.run, fn, *args, **kwargs)
        return await loop.run_in_executor(self._executor, call)

    async def query(
        self, language: str, command: str, *args: Any
    ) -> List[Dict[str, Any]]:
        """``db.query(language, command, *args)``, with the rows read on the worker: a list of ``dict`` (``ResultSet.to_list()``).

        The parameters bind as in ``db.query``: positional ``?`` from the extra arguments, or by name from one ``dict``.
        """
        return await self.run(self._query, language, command, *args)

    async def command(
        self, language: str, command: str, *args: Any
    ) -> Optional[List[Dict[str, Any]]]:
        """``db.command(language, command, *args)``, with the rows read on the worker: a list of ``dict``, or ``None`` when the
        engine returns no result set.

        An ``INSERT``, ``UPDATE`` or ``DELETE`` needs a transaction, and a transaction cannot span awaits: put writes in
        :meth:`run_in_transaction`, or switch on ``db.set_auto_transaction(True)``. Do not send ``BEGIN`` or ``COMMIT`` through this
        method: the next call may run on another worker.
        """
        return await self.run(self._command, language, command, *args)

    async def run_in_transaction(
        self, fn: Callable[[], Any], retries: int = 12, backoff_s: float = 0.005
    ) -> Any:
        """``db.run_in_transaction(fn, retries, backoff_s)``: ``fn`` runs in one transaction, on one worker, from begin to commit.

        ``fn`` is a plain function with no arguments that makes blocking calls on the ``Database`` (not awaits, and not this facade).
        A conflict with another writer rolls the transaction back and runs ``fn`` again, up to ``retries`` times, on the same worker.
        An exception of ``fn`` rolls the transaction back and is raised here. Returns what ``fn`` returns.
        """
        return await self.run(
            self._db.run_in_transaction, fn, retries=retries, backoff_s=backoff_s
        )

    async def find_nearest(
        self,
        index,
        query_vector,
        k: int = 10,
        ef_search: Optional[int] = None,
        allowed_rids=None,
        *,
        include_vector: bool = True,
    ):
        """``index.find_nearest(...)`` on a worker: a list of ``(record, distance)``, nearest first.

        ``index`` is the ``VectorIndex`` that ``db.create_vector_index(...)`` or ``db.schema.get_vector_index(...)`` returns. A search
        finds every vector whose transaction committed before the search started.
        """
        return await self.run(
            index.find_nearest,
            query_vector,
            k=k,
            ef_search=ef_search,
            allowed_rids=allowed_rids,
            include_vector=include_vector,
        )

    async def close(self) -> None:
        """Refuse new calls and wait until every call that was submitted has finished, the queued ones included.

        The wait does not block the event loop. Calling it again waits for the same calls. It does not close the ``Database``.
        """
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, True)

    def _query(self, language: str, command: str, *args: Any):
        return self._db.query(language, command, *args).to_list()

    def _command(self, language: str, command: str, *args: Any):
        result = self._db.command(language, command, *args)
        return None if result is None else result.to_list()
