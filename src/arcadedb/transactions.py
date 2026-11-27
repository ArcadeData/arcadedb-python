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
        self._witness = None

    def __enter__(self) -> "TransactionContext":
        # The levels the caller already holds are not ours to roll back: begin() inside a caller's transaction pushes a
        # nested level, so ownership is decided by the stack before the call, not by whether begin() returned (#9970)
        mark, commits = self.database._transaction_marks()
        try:
            # begin() inside the try: an interrupt landing right after it must still reach the rollback,
            # because an exception out of __enter__ means __exit__ never runs
            self.database.begin()
            self.started = True
            # what tells at the end whether the block committed part of its work (#504)
            self._witness = self.database._commit_witness(mark, commits)
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
        if not self.started:
            return
        if exc_type is None:
            try:
                self.database.commit()
            except Exception as e:
                self._refuse_a_rerun(e)
                raise
        else:
            self.database.rollback()
            if isinstance(exc_val, Exception):
                self._refuse_a_rerun(exc_val)

    def _refuse_a_rerun(self, failure: Exception) -> None:
        """Raise ``PartialCommitError`` instead of ``failure`` when it is a conflict and part of the block is committed.

        A conflict tells the caller to run the whole transaction again, which would apply the committed part twice (#504).
        """
        refusal = self.database._partial_commit_refusal(
            failure, self._witness, "Transaction block"
        )
        if refusal is not None:
            raise refusal
