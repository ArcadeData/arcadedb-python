"""
ArcadeDB Python Bindings - Transaction Management

Transaction context manager and related utilities.
"""

from types import TracebackType
from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from .core import Database


class TransactionContext:
    """Context manager for ArcadeDB transactions."""

    def __init__(self, database: "Database"):
        self.database = database
        self.started = False

    def __enter__(self) -> "TransactionContext":
        # The levels the caller already holds are not ours to roll back: begin() inside a caller's transaction pushes a
        # nested level, so ownership is decided by the stack before the call, not by whether begin() returned (#9970)
        mark = self.database._transaction_stack_mark()
        try:
            # begin() inside the try: an interrupt landing right after it must still reach the rollback,
            # because an exception out of __enter__ means __exit__ never runs
            self.database.begin()
            self.started = True
            return self
        except BaseException:
            self.started = False
            try:
                self.database._end_transactions_opened_since(mark)
            except Exception:  # nosec B110 - best-effort rollback
                pass
            raise

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.started:
            if exc_type is None:
                self.database.commit()
            else:
                self.database.rollback()
