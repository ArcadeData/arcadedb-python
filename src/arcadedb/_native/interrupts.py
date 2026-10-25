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


# What the engine says when a callback returned no reply (NativeExtras.answer): the callback ended before it could write one.
# That happens when a Ctrl-C is raised as Python enters the callback, before its code (and its try) runs.
_NO_REPLY = "was interrupted before Python returned a value"


def convert(error: BaseException) -> BaseException:
    """The ``KeyboardInterrupt`` to raise in place of ``error``: one a callback left on this thread, or the Ctrl-C that
    ended a callback before it could reply. Else ``error`` itself."""
    interrupt = getattr(_PENDING, "interrupt", None)
    if interrupt is None:
        if _NO_REPLY not in str(error):
            return error
        interrupt = KeyboardInterrupt("interrupted while a Python function was called")
    _PENDING.interrupt = None
    interrupt.__cause__ = error
    return interrupt
