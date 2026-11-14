"""insert_arrow and insert_dataframe with timestamp and date columns on the JVM wheel: a column crosses as one array of epoch ticks per batch,
not as one JPype call per cell, and what is stored is unchanged.

``_column_values`` turned every timestamp or date column into a Python list of ``LocalDateTime`` or ``LocalDate``, one
``ofEpochSecond`` or ``ofEpochDay`` call per cell, and ``_column_to_java`` then converted every element again with
``convert_python_to_java``: a million-row timestamp column was two million JPype calls. Now the column is the integer epoch values in ONE
``long[]`` (plus one ``boolean[]`` when a row is null) wrapped in the bridge's ``TemporalColumn``, built with one call per column and batch, and
``DocumentBatcher`` builds each ``LocalDateTime`` or ``LocalDate`` as it fills the document of that row.

Two tests fail on the code before (they count the per-cell ``java.time`` calls, the per-element conversions and the columns that reach the bridge
as a ``TemporalColumn``); the others pin what the old route stored, so the new one must store the same: the bridge column against the per-cell
reading it replaced (every unit, negative ticks, nulls) and the refusal of a bad unit and of a timestamp past the years of ``java.time``
before a row is written. The round trips of every Arrow temporal type through a real load, with the stored wall clock to the nanosecond,
are in ``test_dataframe_ingest.py::TestTemporalColumns``, which runs on both wheels. The JVM build only: it counts JPype calls and builds
bridge objects, so conftest.py does not collect this file on the native wheel.
"""

from collections import Counter

import jpype
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from arcadedb import core, type_conversion
from arcadedb.jvm import start_jvm

TEMPORAL_COLUMN = "com.arcadedb.python.TemporalColumn"
UNITS = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
TEMPORAL_COLUMNS = 7  # the five timestamp columns and the two date columns of _table()


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    """The tests that build a bridge object without a database need the JVM too (a no-op when it already runs)."""
    start_jvm()


def _column_class():
    return jpype.JClass(TEMPORAL_COLUMN)


def _table(n, start=0):
    """n rows: a timestamp column of every unit, one with a zone, a date32 and a date64, a null in every seventh row."""
    idx = np.arange(start, start + n, dtype=np.int64)
    null = idx % 7 == 3
    stamp = idx * 1_000 - 5_000_000
    columns = {"idx": pa.array(idx)}
    for unit in UNITS:
        columns[f"t_{unit}"] = pa.array(stamp, mask=null).cast(pa.timestamp(unit))
    columns["zoned"] = pa.array(stamp, mask=null).cast(
        pa.timestamp("us", tz="Asia/Karachi")
    )
    columns["d32"] = pa.array(idx - 300, mask=null).cast(pa.int32()).cast(pa.date32())
    columns["d64"] = pa.array((idx - 300) * 86_400_000, mask=null).cast(pa.date64())
    return pa.table(columns)


def _make(db, name="Stamp"):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.idx LONG")


def _count(db, name):
    sql = f"SELECT count(*) AS c FROM {name}"  # nosec B608 - test-owned name
    return db.query("sql", sql).first().get("c")


@pytest.fixture
def crossings(monkeypatch):
    """What one load does on the JPype side: the ``java.time`` calls Python makes, the per-value conversions, and the type and length of every column handed to the bridge."""
    # the type conversion caches the classes it fetched first: fetch them now, so the counting proxy below never lands in that cache
    type_conversion._get_java_time_types()
    type_conversion._get_java_python_types()
    log = {"java_time": [], "convert": [], "columns": []}
    real_convert = core.convert_python_to_java
    real_column = core._column_to_java
    real_jclass = jpype.JClass

    class _Counting:
        def __init__(self, name):
            self._name = name
            self._target = real_jclass(name)

        def __getattr__(self, attr):
            member = getattr(self._target, attr)
            if not callable(member):
                return member

            def call(*args):
                log["java_time"].append(f"{self._name}.{attr}")
                return member(*args)

            return call

    def convert(value):
        log["convert"].append(type(value).__name__)
        return real_convert(value)

    def column_to_java(name, values):
        array, n = real_column(name, values)
        log["columns"].append((type(array).__name__.rsplit(".", 1)[-1], n))
        return array, n

    def jclass(name, *args, **kwargs):
        if isinstance(name, str) and name.startswith("java.time."):
            return _Counting(name)
        return real_jclass(name, *args, **kwargs)

    monkeypatch.setattr(core, "convert_python_to_java", convert)
    monkeypatch.setattr(core, "_column_to_java", column_to_java)
    monkeypatch.setattr(jpype, "JClass", jclass)
    return log


def _temporal_columns(log):
    """The lengths of the columns that reached the bridge as a TemporalColumn."""
    return Counter(n for kind, n in log["columns"] if kind == "TemporalColumn")


# ---- the tests that fail on the code before the fix: they count what the load does, never how long it takes


def test_a_temporal_column_makes_no_java_time_call_and_no_conversion_per_cell(
    temp_db, crossings
):
    """Before: one ``ofEpochSecond`` or ``ofEpochDay`` and one ``convert_python_to_java`` per cell. After: neither, for any of the seven columns."""
    n = 600
    _make(temp_db)
    assert temp_db.insert_arrow("Stamp", _table(n), batch_size=n) == n
    assert crossings["java_time"] == []
    assert crossings["convert"] == []
    assert _temporal_columns(crossings) == {n: TEMPORAL_COLUMNS}


def test_each_temporal_column_crosses_as_one_temporal_column_per_batch(
    temp_db, crossings
):
    """1,300 rows in batches of 500 (500, 500, 300): each of the seven columns is ONE bridge column of the batch's rows, three times."""
    _make(temp_db)
    assert temp_db.insert_arrow("Stamp", _table(1_300), batch_size=500) == 1_300
    assert _temporal_columns(crossings) == {
        500: 2 * TEMPORAL_COLUMNS,
        300: TEMPORAL_COLUMNS,
    }


def test_a_dataframe_and_a_parallel_load_take_the_same_route(temp_db, crossings):
    """``insert_dataframe`` reads the frame as an Arrow table and ``parallel=True`` fills the same columns: no per-cell call on either."""
    _make(temp_db)
    frame = _table(400).to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert temp_db.insert_dataframe("Stamp", frame, batch_size=200) == 400
    _make(temp_db, "Par")
    assert (
        temp_db.insert_arrow("Par", _table(400), batch_size=200, parallel=True) == 400
    )
    assert crossings["java_time"] == [] and crossings["convert"] == []
    assert _temporal_columns(crossings) == {200: 4 * TEMPORAL_COLUMNS}


# ---- what the bridge column holds


def _reference(ticks, per_second):
    """The per-cell reading the old route made: ``LocalDateTime.ofEpochSecond`` of the floor quotient, the remainder in nanoseconds."""
    local_datetime = jpype.JClass("java.time.LocalDateTime")
    utc = jpype.JClass("java.time.ZoneOffset").UTC
    seconds, rest = np.divmod(np.asarray(ticks, dtype=np.int64), per_second)
    nanos = rest * (1_000_000_000 // per_second)
    return [
        local_datetime.ofEpochSecond(int(s), int(n), utc)
        for s, n in zip(seconds, nanos)
    ]


@pytest.mark.parametrize("unit", list(UNITS))
def test_a_date_time_column_reads_as_the_per_cell_reading_did(unit):
    per_second = UNITS[unit]
    limit = min(100_000_000_000 * per_second, 2**62)  # about the year 5000
    ticks = np.random.default_rng(7).integers(-limit, limit, size=2_000)
    ticks[:4] = [
        -1,
        0,
        1,
        -per_second,
    ]  # one tick before the epoch is a floor, not a truncation
    nulls = np.zeros(len(ticks), dtype=np.bool_)
    nulls[5::11] = True
    column = _column_class().dateTimes(
        jpype.JArray(jpype.JLong)(ticks),
        per_second,
        jpype.JArray(jpype.JBoolean)(nulls),
    )
    assert int(column.rows()) == len(ticks)
    expected = _reference(ticks, per_second)
    for i, want in enumerate(expected):
        got = column.get(i)
        if nulls[i]:
            assert got is None
        else:
            assert bool(got.equals(want)), (i, int(ticks[i]), str(got), str(want))


def test_a_nanosecond_tick_keeps_its_nanosecond_and_a_null_flag_is_optional():
    ns = 1_577_934_245_123_456_789  # 2020-01-02 03:04:05.123456789
    column = _column_class().dateTimes(
        jpype.JArray(jpype.JLong)([ns, -1]), 1_000_000_000, None
    )
    assert str(column.get(0)) == "2020-01-02T03:04:05.123456789"
    assert str(column.get(1)) == "1969-12-31T23:59:59.999999999"


@pytest.mark.parametrize(
    "per_day, days",
    [(1, [-1, 0, 1, 18_322, -135_140]), (86_400_000, [-86_400_000, 0, 86_400_000])],
    ids=["date32", "date64"],
)
def test_a_date_column_reads_the_day(per_day, days):
    column = _column_class().dates(jpype.JArray(jpype.JLong)(days), per_day, None)
    local_date = jpype.JClass("java.time.LocalDate")
    for i, value in enumerate(days):
        assert bool(column.get(i).equals(local_date.ofEpochDay(value // per_day)))


def test_a_date64_value_is_milliseconds_divided_with_a_floor():
    """The numpy ``//`` the old route used: a millisecond before midnight is still the day before."""
    column = _column_class().dates(
        jpype.JArray(jpype.JLong)([86_400_000 - 1, -1]), 86_400_000, None
    )
    assert str(column.get(0)) == "1970-01-01" and str(column.get(1)) == "1969-12-31"


def test_the_bridge_refuses_what_it_cannot_read():
    longs = jpype.JArray(jpype.JLong)([1, 2])
    for per_second in (
        0,
        -1,
        7,
        3_000_000_000,
    ):  # none divides a second into whole nanoseconds
        with pytest.raises(jpype.JException):
            _column_class().dateTimes(longs, per_second, None)
    with pytest.raises(jpype.JException):
        _column_class().dates(longs, 0, None)
    with pytest.raises(jpype.JException):  # one null flag for two rows
        _column_class().dateTimes(longs, 1_000, jpype.JArray(jpype.JBoolean)([False]))


def test_column_to_java_hands_a_temporal_column_over_as_it_is():
    column = _column_class().dates(jpype.JArray(jpype.JLong)([0, 1, 2]), 1, None)
    array, n = core._column_to_java("day", column)
    assert n == 3 and array is column


def test_a_timestamp_past_the_years_of_java_time_is_refused_before_a_row_is_written(
    temp_db,
):
    """``timestamp[s]`` holds 2**62 seconds, which no ``LocalDateTime`` does: the call raises what it raised before, with the batch unwritten."""
    _make(temp_db)
    table = pa.table(
        {
            "idx": pa.array([0, 1], pa.int64()),
            "t": pa.array([0, 2**62], pa.int64()).cast(pa.timestamp("s")),
        }
    )
    with pytest.raises(jpype.JException):
        temp_db.insert_arrow("Stamp", table)
    assert _count(temp_db, "Stamp") == 0
    assert not temp_db.is_transaction_active()
