"""Review of the temporal columnar kinds (#333): a value the image cannot take is refused in Python, before the crossing, and never wraps.

``T`` and ``D`` columns carry integers that the image turns into ``LocalDateTime`` and ``LocalDate``. A ``date64`` past int32 days used
to wrap in the int32 cast and a ``timestamp[s]`` past the years of ``java.time`` threw ``DateTimeException`` in the middle of a batch. Now
``columnar.pack`` returns None for both, so the call takes the row path, which raises what it raised before (nothing stored, no
transaction left open on the synchronous call; ``may have been stored`` for a parallel call, which has handed rows over by then). Also
pinned: a chunk with a date-time column and a float column holding NaN goes columnar now and stores the same NaN that ``INSERT INTO t
CONTENT :row`` stores (both end in ``MutableDocument.putProperty``). Needs the native library: conftest.py collects this file on the
native backend only.
"""

import datetime as dt
import math

import pytest

from arcadedb._native import columnar, lib  # noqa: E402
from arcadedb.exceptions import ArcadeDBError

pa = pytest.importorskip("pyarrow")

COLP = "arcadedb_n_insert_colp"
DAY_MS = 86_400_000
LDT_MIN_SECONDS, LDT_MAX_SECONDS = -31_557_014_167_219_200, 31_556_889_864_403_199
NO_TYPES = columnar.Declared(known={})


class Spy:
    def __init__(self, monkeypatch):
        self.calls = []
        real = lib.call

        def call(name, *args):
            self.calls.append(name)
            return real(name, *args)

        monkeypatch.setattr(lib, "call", call)


def _count(db, name):
    sql = f"SELECT count(*) AS n FROM {name}"  # nosec B608 - test-owned type name
    return db.query("sql", sql).first().get("n")


def _kinds(columns):
    packed = columnar.pack(columns, NO_TYPES)
    return None if packed is None else [c.kind for c in packed[1]]


def test_a_date64_past_int32_days_is_not_laid_out_and_never_wraps():
    wraps = pa.array([DAY_MS * (2**31 + 5)], pa.int64()).cast(pa.date64())
    assert columnar.pack({"d": wraps}, NO_TYPES) is None
    below = pa.array([DAY_MS * (-(2**31) - 1)], pa.int64()).cast(pa.date64())
    assert columnar.pack({"d": below}, NO_TYPES) is None
    one_bad = pa.array([0, DAY_MS, DAY_MS * 2**40], pa.int64()).cast(pa.date64())
    assert columnar.pack({"d": one_bad}, NO_TYPES) is None  # one value is enough


def test_a_date64_at_the_int32_limits_stays_columnar_and_exact():
    edge = pa.array(
        [DAY_MS * (2**31 - 1), DAY_MS * -(2**31), None], pa.int64()
    ).cast(pa.date64())
    rows, packed = columnar.pack({"d": edge}, NO_TYPES)
    assert rows == 3 and packed[0].kind == "D"
    import ctypes

    import numpy as np

    addr, length = packed[0].data
    days = np.frombuffer(ctypes.string_at(addr, length), dtype=np.int32)
    assert days.tolist() == [2**31 - 1, -(2**31), 0]


def test_a_timestamp_past_the_years_of_java_time_is_not_laid_out():
    assert (
        columnar.pack({"t": pa.array([0, 2**62], pa.timestamp("s"))}, NO_TYPES) is None
    )
    over = pa.array([LDT_MAX_SECONDS + 1], pa.timestamp("s"))
    assert columnar.pack({"t": over}, NO_TYPES) is None
    under = pa.array([LDT_MIN_SECONDS - 1], pa.timestamp("s"))
    assert columnar.pack({"t": under}, NO_TYPES) is None


def test_a_timestamp_at_the_limits_stays_columnar():
    edge = pa.array([LDT_MAX_SECONDS, LDT_MIN_SECONDS, None], pa.timestamp("s"))
    assert _kinds({"t": edge}) == ["T"]
    # an int64 of milliseconds, microseconds or nanoseconds is always inside the years of java.time
    for unit in ("ms", "us", "ns"):
        big = pa.array([2**63 - 1, -(2**63)], pa.timestamp(unit))
        assert _kinds({"t": big}) == ["T"], unit


def _type(db, name):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")


def test_a_timestamp_the_image_cannot_take_stores_nothing_and_leaves_no_transaction(
    temp_db,
):
    _type(temp_db, "Col")
    table = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "t": pa.array([0, 2**62], pa.timestamp("s")),
        }
    )
    with pytest.raises(ArcadeDBError):
        temp_db.insert_arrow("Col", table)
    assert not temp_db.is_transaction_active()
    assert _count(temp_db, "Col") == 0


def test_a_date_the_image_cannot_take_stores_nothing_and_leaves_no_transaction(temp_db):
    _type(temp_db, "Col")
    wraps = pa.array([0, DAY_MS * (2**31 + 5)], pa.int64()).cast(pa.date64())
    table = pa.table({"id": pa.array([1, 2], pa.int64()), "d": wraps})
    with pytest.raises(ArcadeDBError):
        temp_db.insert_arrow("Col", table)
    assert not temp_db.is_transaction_active()
    assert _count(temp_db, "Col") == 0


def test_a_parallel_load_of_such_a_timestamp_says_that_rows_may_have_been_stored(
    temp_db,
):
    _type(temp_db, "Col")
    table = pa.table(
        {
            "id": pa.array([1, 2], pa.int64()),
            "t": pa.array([0, 2**62], pa.timestamp("s")),
        }
    )
    with pytest.raises(ArcadeDBError, match="may have been stored"):
        temp_db.insert_arrow("Col", table, parallel=True)
    temp_db.async_executor().wait_completion()
    assert (
        _count(temp_db, "Col") <= 1
    )  # the row before the bad one may have been handed over


def test_a_datetime_chunk_with_nan_stores_the_nan_the_statement_path_stores(
    temp_db, monkeypatch
):
    for name in ("Col", "Ref"):
        _type(temp_db, name)
        temp_db.command("sql", f"CREATE PROPERTY {name}.ts DATETIME_MICROS")
        temp_db.command("sql", f"CREATE PROPERTY {name}.val DOUBLE")
    rows = [
        {
            "id": i,
            "ts": dt.datetime(2020, 1, 1, 0, 0, i),
            "val": float("nan") if i % 2 else 1.5 * i,
        }
        for i in range(10)
    ]
    spy = Spy(monkeypatch)
    temp_db.insert_many("Col", rows)
    assert (
        spy.calls.count(COLP) == 1
    )  # the date-time column no longer forces the JSON path
    sql = "INSERT INTO `Ref` CONTENT :row"
    temp_db.begin()
    temp_db._command_many("sql", sql, [{"row": row} for row in rows])
    temp_db.commit()

    def stored(name):
        select = f"SELECT FROM {name} ORDER BY id"  # nosec B608 - test-owned type name
        return [
            (r["id"], type(r["val"]).__name__, repr(r["val"]), repr(r["ts"]))
            for r in temp_db.query("sql", select).to_list()
        ]

    got, want = stored("Col"), stored("Ref")
    assert got == want
    assert all(math.isnan(float(v[2])) for v in got if v[0] % 2)
