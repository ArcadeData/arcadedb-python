"""Native: a column of date-times, dates, Decimals or bytes stays on the columnar entry and stores what the row path stores (#316).

Before, one such column made ``columnar.pack`` return None, so every column of the batch went through one dict per row and a JSON text
with tagged values. Now the kinds T, D, n and x of ``arcadedb_n_insert_colp`` carry them and the image builds the same objects
(``LocalDateTime``, ``LocalDate``, ``BigDecimal``, ``byte[]``) that ``Json.decodeTag`` builds for the tagged values. Each test loads the
same data through the row path (the private reference: the columnar path switched off, so the chunk goes as tagged JSON) and through
the columns, compares every stored value and its type, and spies on the entry points so a silent fall back cannot pass. Needs the
native library, so conftest.py collects this file on the native backend only.
"""

import datetime as dt
import decimal

import pytest

from arcadedb._native import bulk, columnar, lib  # noqa: E402

COLP, ROWS, MANY = (
    "arcadedb_n_insert_colp",
    "arcadedb_n_insert_rows",
    "arcadedb_n_exec_many",
)
UTC9 = dt.timezone(dt.timedelta(hours=9))


class Spy:
    def __init__(self, monkeypatch):
        self.calls = []
        real = lib.call

        def call(name, *args):
            self.calls.append(name)
            return real(name, *args)

        monkeypatch.setattr(lib, "call", call)

    def count(self, name):
        return self.calls.count(name)


@pytest.fixture
def spy(monkeypatch):
    return Spy(monkeypatch)


def _type(db, name, declared=True):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")
    if declared:
        for prop, ty in (
            ("ts", "DATETIME_MICROS"),
            ("ns", "DATETIME_NANOS"),
            ("day", "DATE"),
            ("amount", "DECIMAL"),
            ("blob", "BINARY"),
        ):
            db.command("sql", f"CREATE PROPERTY {name}.{prop} {ty}")


def _stored(db, name):
    """Every record by id as {property: (python type name, repr)}, so a changed type is as visible as a changed value."""
    sql = f"SELECT FROM {name} ORDER BY id"  # nosec B608 - test-owned type name
    return [
        {
            k: (type(v).__name__, repr(v))
            for k, v in sorted(rec.items())
            if not k.startswith("@") and k != "rid"
        }
        for rec in db.query("sql", sql).to_list()
    ]


def _rows(columns):
    return [dict(zip(columns, vals)) for vals in zip(*columns.values())]


def _reference(db, name, rows, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(bulk, "_ENABLED", False)
        db.insert_many(name, rows)


def _columns(n=60):
    base = dt.datetime(1969, 12, 31, 23, 59, 59, 999999)
    return {
        "id": list(range(n)),
        "ts": [
            (
                None
                if i % 11 == 0
                else (base + dt.timedelta(seconds=i * 86_399, microseconds=i * 7))
            )
            for i in range(n)
        ],
        "day": [
            None if i % 13 == 0 else dt.date(1960, 1, 1) + dt.timedelta(days=i * 401)
            for i in range(n)
        ],
        "amount": [
            (
                None
                if i % 9 == 0
                else decimal.Decimal(f"{i * 1234567}.{i % 100:02d}")
                * (-1 if i % 2 else 1)
            )
            for i in range(n)
        ],
        "blob": [
            None if i % 8 == 0 else bytes(range(i % 256)) + b"\x00\xff"
            for i in range(n)
        ],
    }


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "schemaless"])
def test_a_batch_with_temporal_decimal_and_bytes_columns_stays_columnar(
    temp_db, spy, monkeypatch, declared
):
    for name in ("Ref", "ByRows", "ByColumns"):
        _type(temp_db, name, declared)
    columns = _columns()
    _reference(temp_db, "Ref", _rows(columns), monkeypatch)
    assert (
        spy.count(COLP) == 0 and spy.count(ROWS) == 1
    )  # the reference is the JSON path
    spy.calls.clear()
    assert temp_db.insert_many("ByRows", _rows(columns)) == 60
    assert spy.count(COLP) == 1 and spy.count(ROWS) == 0 and spy.count(MANY) == 0
    spy.calls.clear()
    assert temp_db.insert_columns("ByColumns", columns) == 60
    assert spy.count(COLP) == 1 and spy.count(ROWS) == 0
    expected = _stored(temp_db, "Ref")
    assert len(expected) == 60 and expected[1]["ts"][0] == "datetime"
    assert _stored(temp_db, "ByRows") == expected
    assert _stored(temp_db, "ByColumns") == expected


def test_time_zone_aware_and_naive_datetimes_follow_the_json_rule(
    temp_db, spy, monkeypatch
):
    """An aware date-time is its UTC wall clock, a naive one is taken as it is (_tag_datetime): the column does not decide a zone."""
    _type(temp_db, "Ref")
    _type(temp_db, "Col")
    values = [
        dt.datetime(2020, 1, 2, 12, 0, 0, 5, tzinfo=UTC9),
        dt.datetime(1969, 12, 31, 23, 0, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2020, 6, 1, 0, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-7))),
    ]
    rows = [{"id": i, "ts": v} for i, v in enumerate(values)]
    _reference(temp_db, "Ref", rows, monkeypatch)
    spy.calls.clear()
    temp_db.insert_many("Col", rows)
    assert spy.count(COLP) == 1
    assert _stored(temp_db, "Col") == _stored(temp_db, "Ref")


def test_pandas_timestamps_keep_microseconds_as_the_json_path_does(
    temp_db, spy, monkeypatch
):
    import pandas as pd

    _type(temp_db, "Ref")
    _type(temp_db, "Col")
    stamps = [
        pd.Timestamp("2026-05-06 07:08:09.123456789"),
        pd.Timestamp("1960-05-06 07:08:09.123456", tz="Asia/Seoul"),
    ]
    rows = [{"id": i, "ts": v} for i, v in enumerate(stamps)]
    _reference(temp_db, "Ref", rows, monkeypatch)
    spy.calls.clear()
    temp_db.insert_many("Col", rows)
    assert spy.count(COLP) == 1 and spy.count(ROWS) == 0
    assert _stored(temp_db, "Col") == _stored(temp_db, "Ref")


@pytest.mark.parametrize(
    "column",
    [
        [dt.datetime(2020, 1, 1), dt.date(2020, 1, 1)],  # a date-time and a date
        [decimal.Decimal("1.5"), decimal.Decimal("NaN")],  # a non-finite Decimal
        [b"a", "b"],  # bytes and text
    ],
    ids=["datetime-and-date", "nan-decimal", "bytes-and-text"],
)
def test_a_column_the_buffers_cannot_carry_unchanged_still_falls_back(
    temp_db, spy, monkeypatch, column
):
    _type(temp_db, "Ref", declared=False)
    _type(temp_db, "Col", declared=False)
    rows = [{"id": i, "x": v} for i, v in enumerate(column)]
    try:
        _reference(temp_db, "Ref", rows, monkeypatch)
    except (
        Exception
    ):  # the statement path may refuse it; then the new path must refuse it the same way
        with pytest.raises(Exception):
            temp_db.insert_many("Col", rows)
        assert spy.count(COLP) == 0
        return
    spy.calls.clear()
    temp_db.insert_many("Col", rows)
    assert spy.count(COLP) == 0
    assert _stored(temp_db, "Col") == _stored(temp_db, "Ref")


def _arrow_table(pa):
    import numpy as np

    n = 40
    ids = np.arange(n, dtype=np.int64)
    epoch_us = (ids - 20) * 1_000_003_000 + 7  # microseconds, before and after 1970
    mask = ids % 6 == 0

    def stamps(unit, tz=None):
        raw = {
            "s": epoch_us // 1_000_000,
            "ms": epoch_us // 1_000,
            "us": epoch_us,
            "ns": epoch_us * 1_000,
        }[unit]
        return pa.array(raw.astype(np.int64), mask=mask).cast(pa.timestamp(unit, tz=tz))

    return pa.table(
        {
            "id": ids,
            "ts": stamps("us"),
            "ns": stamps("ns"),
            "sec": stamps("s"),
            "zoned": stamps("ms", tz="Asia/Seoul"),
            "day": pa.array(
                ((ids - 10) * 397).astype(np.int32), mask=ids % 5 == 0
            ).cast(pa.date32()),
            "amount": pa.array(
                [
                    None if i % 7 == 0 else decimal.Decimal(f"{i}.{i % 100:02d}")
                    for i in range(n)
                ],
                type=pa.decimal128(20, 2),
            ),
            "blob": pa.array(
                [None if i % 4 == 0 else bytes(range(i)) for i in range(n)]
            ),
        }
    )


def test_arrow_timestamps_dates_decimals_and_binary_stay_columnar(
    temp_db, spy, monkeypatch
):
    import pyarrow as pa

    for name in ("Ref", "Col"):
        _type(temp_db, name)
        for prop, ty in (("sec", "DATETIME_SECOND"), ("zoned", "DATETIME")):
            temp_db.command("sql", f"CREATE PROPERTY {name}.{prop} {ty}")
    table = _arrow_table(pa)
    with (
        monkeypatch.context() as m
    ):  # the reference: tagged values through the JSON path
        m.setattr(bulk, "_ENABLED", False)
        assert temp_db.insert_arrow("Ref", table) == 40
    assert spy.count(COLP) == 0
    spy.calls.clear()
    assert temp_db.insert_arrow("Col", table) == 40
    assert spy.count(COLP) == 1 and spy.count(ROWS) == 0 and spy.count(MANY) == 0
    expected = _stored(temp_db, "Ref")
    assert len(expected) == 40 and expected[1]["ns"][0] == "datetime"
    assert _stored(temp_db, "Col") == expected


def test_an_arrow_batch_with_one_list_column_falls_back_with_the_same_values(
    temp_db, spy, monkeypatch
):
    """A list<string> column cannot cross as a buffer: the whole batch takes the row path, and the temporal columns reach it as tagged values."""
    import pyarrow as pa

    for name in ("Ref", "Col"):
        _type(temp_db, name)
        for prop, ty in (("sec", "DATETIME_SECOND"), ("zoned", "DATETIME")):
            temp_db.command("sql", f"CREATE PROPERTY {name}.{prop} {ty}")
    table = _arrow_table(pa).append_column(
        "tags", pa.array([["a", "b"]] * 40, type=pa.list_(pa.string()))
    )
    with monkeypatch.context() as m:
        m.setattr(bulk, "_ENABLED", False)
        temp_db.insert_arrow("Ref", table)
    spy.calls.clear()
    temp_db.insert_arrow("Col", table)
    assert spy.count(COLP) == 0
    assert _stored(temp_db, "Col") == _stored(temp_db, "Ref")


def test_a_parallel_load_of_datetimes_is_one_columnar_crossing(temp_db, spy):
    _type(temp_db, "Par")
    columns = _columns(300)
    assert temp_db.insert_columns("Par", columns, parallel=True) == 300
    assert spy.count(COLP) == 1 and spy.count(ROWS) == 0
    assert temp_db.query("sql", "SELECT count(*) AS n FROM Par").first().get("n") == 300


def test_the_image_refuses_a_wrong_ticks_per_second_and_writes_nothing(temp_db):
    import json

    _type(temp_db, "Col")
    rows, packed = columnar.pack(
        {"id": [1, 2], "ts": [dt.datetime(2020, 1, 1)] * 2}, columnar.Declared(known={})
    )
    assert packed[1].kind == "T" and packed[1].dim == 1_000_000
    packed[1].dim = 7
    spec = columnar.spec_of(packed)
    reply = json.loads(
        lib.call(
            COLP,
            temp_db._h,
            lib.b("Col"),
            lib.b(columnar.header_of(rows, 10_000, packed)),
            spec,
            len(spec),
        )
    )
    assert not reply["ok"] and "ticks per second" in reply["error"]
    assert temp_db.query("sql", "SELECT count(*) AS n FROM Col").first().get("n") == 0
