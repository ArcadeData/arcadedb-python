"""Ctrl-C inside a Python function or callback that the engine called, on the native backend.

A ``KeyboardInterrupt`` cannot cross the engine as it is: it ends the statement as an engine error. It is remembered per
thread when the upcall catches it (``note``), and the place that turns the statement's failure into an ``ArcadeDBError``
raises the ``KeyboardInterrupt`` instead (chained from that error, ``convert``), so Ctrl-C reaches the caller as a
``KeyboardInterrupt`` wherever the signal landed: in the function, in the engine between two calls, or on the way into the
callback (the same rule as the JPype backend, see ``arcadedb.callbacks``).
"""

import threading

_PENDING = threading.local()


def note(error: BaseException) -> None:
    """Remember a ``KeyboardInterrupt`` raised in a callback on this thread."""
    if isinstance(error, KeyboardInterrupt):
        _PENDING.interrupt = error


def clear() -> None:
    """Forget an interrupt an earlier statement left (called when a statement starts)."""
    _PENDING.interrupt = None


def convert(error: BaseException) -> BaseException:
    """The ``KeyboardInterrupt`` to raise in place of ``error`` if a callback left one on this thread, else ``error``."""
    interrupt = getattr(_PENDING, "interrupt", None)
    if interrupt is None:
        return error
    _PENDING.interrupt = None
    interrupt.__cause__ = error
    return interrupt
