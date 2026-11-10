"""The drain of a native result set (``NativeResultSet._drain_dicts``) returns the rows row-by-row iteration returns (refs #321).

``_drain_dicts`` takes the rows of a reply without building one tuple per row (``_fill(into)``). That may not change a row or
their order, also after some rows were taken one by one, or on an empty or an already drained result.

This file was ``arcadedb._native.columns``'s reference test (#321); that module is gone since the column frames (#331), whose own
reference test is ``test_native_column_frames_reference.py``. Only the tests of the drain remain here.

conftest.py collects this file on the native backend only.
"""

import pytest

# --- the drain ------------------------------------------------------------------------------------------------------------

DRAIN_ROWS = 5000  # more than the first reply (256) and two further replies (2048 each)


@pytest.fixture
def drain_db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Drain")
    temp_db.command("sql", "CREATE PROPERTY Drain.id INTEGER")
    temp_db.insert_many(
        "Drain",
        [{"id": i, "txt": f"t{i}", "val": i / 8.0} for i in range(DRAIN_ROWS)],
    )
    return temp_db


SELECT_DRAIN = "SELECT id, txt, val FROM Drain ORDER BY id"


def test_the_drain_returns_the_rows_row_by_row_iteration_returns(drain_db):
    by_rows = [r.to_dict() for r in drain_db.query("sql", SELECT_DRAIN)]
    drained = drain_db.query("sql", SELECT_DRAIN)._drain_dicts()
    assert len(drained) == DRAIN_ROWS
    assert drained == by_rows
    assert drained[0] == {"id": 0, "txt": "t0", "val": 0.0}
    assert drained[-1] == {
        "id": DRAIN_ROWS - 1,
        "txt": f"t{DRAIN_ROWS - 1}",
        "val": (DRAIN_ROWS - 1) / 8.0,
    }


@pytest.mark.parametrize("taken", [1, 255, 256, 257, 2304, 4999])
def test_the_drain_after_some_rows_were_taken_returns_the_rest(drain_db, taken):
    rs = drain_db.query("sql", SELECT_DRAIN)
    head = [next(rs).to_dict() for _ in range(taken)]
    rest = rs._drain_dicts()
    want = [r.to_dict() for r in drain_db.query("sql", SELECT_DRAIN)]
    assert head + rest == want


def test_the_drain_of_an_empty_result_and_a_second_drain_are_empty(drain_db):
    rs = drain_db.query("sql", "SELECT id FROM Drain WHERE id < 0")
    assert rs._drain_dicts() == []
    rs = drain_db.query("sql", SELECT_DRAIN)
    assert len(rs._drain_dicts()) == DRAIN_ROWS
    assert rs._drain_dicts() == []
