"""A column of date-times, dates, Decimals or bytes stores the same values on both wheels (ArcadeData/arcadedb-python #316).

On the native wheel such a column used to move the whole batch off the columnar entry; now ``insert_columns`` and ``insert_many`` send
it as a column (``tests/test_native_columnar_temporal.py`` proves the entry and compares with the row path). Here the expected values
are literals, so the same file on the JVM wheel and on the native wheel says that both store them: a naive date-time as it is, a
time-zone-aware one as its UTC wall clock, a date-time before 1970, a null, a date, a Decimal and bytes.
"""

import datetime as dt
import decimal

import pytest

UTC9 = dt.timezone(dt.timedelta(hours=9))
TS = [
    dt.datetime(2020, 1, 2, 3, 4, 5, 123456),
    dt.datetime(1969, 12, 31, 23, 59, 59, 999999),
    dt.datetime(1900, 1, 1, 0, 0, 0, 1),
    dt.datetime(2400, 2, 29, 23, 59, 59),
    None,
]
TS_STORED = TS  # a naive date-time is the wall clock
AWARE = [
    dt.datetime(2020, 1, 2, 12, 0, 0, 5, tzinfo=UTC9),
    dt.datetime(1969, 12, 31, 23, 0, 0, tzinfo=dt.timezone.utc),
    dt.datetime(2020, 6, 1, 0, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-7))),
    None,
    dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc),
]
AWARE_STORED = [
    dt.datetime(2020, 1, 2, 3, 0, 0, 5),
    dt.datetime(1969, 12, 31, 23, 0, 0),
    dt.datetime(2020, 6, 1, 7, 0, 0),
    None,
    dt.datetime(2000, 1, 1),
]
DAYS = [
    dt.date(2020, 2, 29),
    dt.date(1960, 1, 1),
    None,
    dt.date(1970, 1, 1),
    dt.date(1969, 12, 31),
]
AMOUNTS = [
    decimal.Decimal("12345.6789"),
    decimal.Decimal("-0.50"),
    decimal.Decimal("1E+3"),
    None,
    decimal.Decimal("0"),
]
BLOBS = [b"\x00\xff abc\x7f", b"", None, bytearray(b"ab\x80"), b"z"]


def _type(db, name):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")
    db.command("sql", f"CREATE PROPERTY {name}.ts DATETIME_MICROS")
    db.command("sql", f"CREATE PROPERTY {name}.day DATE")
    db.command("sql", f"CREATE PROPERTY {name}.amount DECIMAL")
    db.command("sql", f"CREATE PROPERTY {name}.blob BINARY")


def _stored(db, name, column):
    sql = f"SELECT id, {column} AS v FROM {name} ORDER BY id"  # nosec B608 - test-owned names
    return [r.get("v") for r in db.query("sql", sql).to_list()]


def _bytes(values):
    # a BINARY property reads back as the values of a Java byte[] (signed, -128..127) or as bytes; & 0xFF takes both to the same bytes
    return [None if v is None else bytes(b & 0xFF for b in v) for v in values]


@pytest.mark.parametrize("path", ["insert_columns", "insert_many"])
def test_naive_and_aware_datetimes_dates_decimals_and_bytes_are_stored_the_same(
    temp_db, path
):
    _type(temp_db, "Ev")
    n = len(TS)
    columns = {
        "id": list(range(n)),
        "ts": TS,
        "day": DAYS,
        "amount": AMOUNTS,
        "blob": BLOBS,
    }
    if path == "insert_columns":
        assert temp_db.insert_columns("Ev", columns) == n
    else:
        rows = [dict(zip(columns, vals)) for vals in zip(*columns.values())]
        assert temp_db.insert_many("Ev", rows) == n
    assert _stored(temp_db, "Ev", "ts") == TS_STORED
    assert _stored(temp_db, "Ev", "day") == DAYS
    assert _stored(temp_db, "Ev", "amount") == AMOUNTS
    assert _bytes(_stored(temp_db, "Ev", "blob")) == _bytes(BLOBS)


def test_time_zone_aware_datetimes_are_stored_as_their_utc_wall_clock(temp_db):
    _type(temp_db, "Zoned")
    n = len(AWARE)
    assert temp_db.insert_columns("Zoned", {"id": list(range(n)), "ts": AWARE}) == n
    assert _stored(temp_db, "Zoned", "ts") == AWARE_STORED


def test_a_parallel_load_of_datetimes_stores_the_same_values(temp_db):
    _type(temp_db, "Par")
    n = len(TS)
    assert (
        temp_db.insert_columns(
            "Par", {"id": list(range(n)), "ts": TS, "day": DAYS}, parallel=True
        )
        == n
    )
    assert _stored(temp_db, "Par", "ts") == TS_STORED
    assert _stored(temp_db, "Par", "day") == DAYS
