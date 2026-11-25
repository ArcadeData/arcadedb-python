"""Open a database when it exists and create it when it does not: the one routine behind
``arcadedb.open_or_create_database`` and ``DatabaseFactory.open_or_create``, for both builds.

``factory`` is a ``DatabaseFactory`` (the JVM build) or a ``NativeDatabaseFactory`` (the native build): both have ``exists()``,
``create()`` and ``open(read_only=)`` and raise ``ArcadeDBError``. The rules are the ones on the page "Databases and queries".
"""

from __future__ import annotations

import threading
import time

from .exceptions import ArcadeDBError

# One lock for the process, so two threads that open or create the same path do not both see "missing" and both create.
_LOCK = threading.Lock()

# After the second create failed, how long to wait for another process to finish creating the same database, and how often to look.
_LOST_RACE_WAIT_S = 0.25
_LOST_RACE_STEP_S = 0.01


def open_or_create(factory, read_only: bool = False):
    """Open the database of ``factory`` when it exists, create it when it does not, and return the open database.

    ``read_only=True`` never creates: there is nothing to open read-only in a path that holds no database, so a missing database
    raises the same ``ArcadeDBError`` as ``open_database``.

    Two processes that call this on the same new path at the same time: one of them creates the database. The other one opens it,
    or, because the engine allows one process per database, raises the engine's "locked by another process" ``ArcadeDBError`` while
    the first process still has the database open, which includes the time in which it is still creating it.
    """
    with _LOCK:
        if read_only:
            return factory.open(read_only=True)
        if factory.exists():
            return factory.open()
        try:
            return factory.create()
        except ArcadeDBError:
            # Create once more (#492). The engine makes the database directory with exists() and then mkdirs(), so a process that
            # another one overtakes between the two gets "Cannot create directory" for a directory that is there now, and the other
            # process may need longer than any short wait to write the database. The second create finds the directory, and the
            # engine's database lock decides: it creates, or it raises "locked by another process" or "already exists".
            pass
        try:
            return factory.create()
        except ArcadeDBError:
            # The database may have appeared in the meantime (another process won): then open it.
            # Any other failure (a path that is a file, no permission) is still missing a database: raise the create error.
            deadline = time.monotonic() + _LOST_RACE_WAIT_S
            while not factory.exists():
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_LOST_RACE_STEP_S)
            return factory.open()
