"""What loading and reading columns does, as the how-to page ``docs/modules/ROOT/pages/how-to/dataframes-and-columns.adoc`` says it
(ArcadeData/arcadedb-python#267).

The page covers ``insert_dataframe``, ``insert_arrow``, ``insert_columns`` and the readers ``to_columns``, ``to_arrow``, ``to_dataframe`` and
``to_polars``. Each statement of it is pinned here first: how a frame becomes records (the Arrow type to the stored type, nulls, NaN), what a
declared property accepts and refuses, how many rows are converted at a time, what a failed load leaves behind, what ``insert_columns`` takes
and refuses, and what the readers return for nulls, dates, ``batch_size`` and a missing library. ``test_dataframe_ingest.py``,
``test_insert_columns.py``, ``test_resultset_arrow.py`` and ``test_columnar_readers.py`` pin the same calls in more depth; the tests here are
the ones the page quotes, so that a change to a sentence of the page has one place to fail. They use no Java object, so they run on the JVM
build and the native build alike (the two branches that differ test the backend they run on).
"""

import datetime as dt
import decimal
import sys

import pytest

import arcadedb
from arcadedb import ArcadeDBError

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")


def _rows(db, sql):
    return db.query("sql", sql).to_list()


def _count(db, type_name) -> int:
    query = f"SELECT count(*) AS n FROM {type_name}"  # nosec B608 - test-controlled
    return int(_rows(db, query)[0]["n"])


# -------------------------------------------------------------------------------------------------- loading a frame


def test_a_pandas_frame_loads_and_nan_is_a_missing_value(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Trip")
    frame = pd.DataFrame(
        {
            "id": np.array([1, 2, 3], dtype=np.int64),
            "fare": [12.5, float("nan"), 7.25],
            "city": ["Oslo", "Rome", None],
            "paid": [True, False, True],
            "at": pd.to_datetime(
                ["2026-01-02 03:04:05", "2026-01-03 00:00:00", "2026-01-04 12:00:00"]
            ),
        }
    )
    assert temp_db.insert_dataframe("Trip", frame) == 3
    first = _rows(temp_db, "SELECT FROM Trip WHERE id = 1")[0]
    assert first["fare"] == 12.5 and isinstance(first["fare"], float)
    assert first["city"] == "Oslo" and first["paid"] is True
    assert first["at"] == dt.datetime(2026, 1, 2, 3, 4, 5)
    second = _rows(temp_db, "SELECT FROM Trip WHERE id = 2")[0]
    assert second.get("fare") is None and second["paid"] is False
    third = _rows(temp_db, "SELECT FROM Trip WHERE id = 3")[0]
    assert third.get("city") is None


def test_a_polars_nan_stays_a_nan_and_a_polars_null_is_missing(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Pl")
    frame = pl.DataFrame({"k": [0, 1, 2], "x": [1.5, float("nan"), None]})
    assert temp_db.insert_dataframe("Pl", frame) == 3
    nan_row = _rows(temp_db, "SELECT FROM Pl WHERE k = 1")[0]
    assert nan_row["x"] != nan_row["x"]
    assert _rows(temp_db, "SELECT FROM Pl WHERE k = 2")[0].get("x") is None


def test_the_pandas_index_is_not_loaded_until_it_is_a_column(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Indexed")
    frame = pd.DataFrame({"v": [1, 2]}, index=pd.Index(["x", "y"], name="name"))
    temp_db.insert_dataframe("Indexed", frame)
    assert _count(temp_db, "Indexed") == 2
    assert all(r.get("name") is None for r in _rows(temp_db, "SELECT FROM Indexed"))
    temp_db.insert_dataframe("Indexed", frame.reset_index())
    assert _count(temp_db, "Indexed WHERE name = 'y'") == 1


@pytest.mark.parametrize(
    "column, expected",
    [
        (pa.array([1, 2], pa.int32()), [1, 2]),
        (pa.array([0.5, 1.5], pa.float32()), [0.5, 1.5]),
        (pa.array([True, False], pa.bool_()), [True, False]),
        (pa.array(["a", "b"], pa.string()), ["a", "b"]),
        (
            pa.array([dt.datetime(2026, 1, 2, 3, 4, 5)] * 2, pa.timestamp("us")),
            [dt.datetime(2026, 1, 2, 3, 4, 5)] * 2,
        ),
        (
            pa.array(
                [
                    dt.datetime(
                        2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone(dt.timedelta(hours=2))
                    )
                ]
                * 2,
                pa.timestamp("us", tz="UTC"),
            ),
            [dt.datetime(2026, 1, 2, 1, 4, 5)] * 2,
        ),
        (pa.array([dt.date(2026, 1, 2)] * 2, pa.date32()), [dt.date(2026, 1, 2)] * 2),
        (
            pa.array([decimal.Decimal("1.25")] * 2, pa.decimal128(5, 2)),
            [decimal.Decimal("1.25")] * 2,
        ),
        (pa.array([[1, 2], [3]], pa.list_(pa.int64())), [[1, 2], [3]]),
        (
            pa.array([{"a": 1}, {"a": 2}], pa.struct([("a", pa.int64())])),
            [{"a": 1}, {"a": 2}],
        ),
    ],
)
def test_the_arrow_types_of_the_mapping_table_read_back(temp_db, column, expected):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Mapped")
    table = pa.table({"i": pa.array([0, 1], pa.int64()), "v": column})
    assert temp_db.insert_arrow("Mapped", table) == 2
    got = [r.get("v") for r in _rows(temp_db, "SELECT FROM Mapped ORDER BY i")]
    assert got == expected


def test_the_declared_type_table_of_the_page():
    from arcadedb.dataframe import _ALLOWED

    integers = {
        "BYTE",
        "SHORT",
        "INTEGER",
        "LONG",
        "FLOAT",
        "DOUBLE",
        "DECIMAL",
        "STRING",
    }
    datetimes = {"DATETIME", "DATETIME_SECOND", "DATETIME_MICROS", "DATETIME_NANOS"}
    assert _ALLOWED["integer"] == integers
    assert _ALLOWED["floating"] == {"FLOAT", "DOUBLE", "DECIMAL", "STRING"}
    assert "INTEGER" not in _ALLOWED["floating"]  # a float would lose its fraction
    assert _ALLOWED["bool"] == {"BOOLEAN", "STRING"}
    assert _ALLOWED["string"] == {"STRING", "LINK"}
    assert _ALLOWED["timestamp"] == datetimes | {"STRING"}
    assert _ALLOWED["date"] == {"DATE"} | datetimes | {"STRING"}


def test_a_uint64_value_past_the_signed_range_is_refused(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Big")
    with pytest.raises(ValueError):
        temp_db.insert_arrow("Big", pa.table({"v": pa.array([2**63], pa.uint64())}))
    assert _count(temp_db, "Big") == 0


def test_a_type_with_no_mapping_is_refused_before_a_row_is_written(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE NoMap")
    with pytest.raises(TypeError, match="no ArcadeDB mapping"):
        temp_db.insert_arrow("NoMap", pa.table({"t": pa.array([1], pa.time64("us"))}))
    assert _count(temp_db, "NoMap") == 0


def test_arrow_table_record_batch_and_reader_all_load(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Shapes")
    table = pa.table({"v": [1, 2, 3]})
    assert temp_db.insert_arrow("Shapes", table) == 3
    assert temp_db.insert_arrow("Shapes", table.to_batches()[0]) == 3
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    assert temp_db.insert_arrow("Shapes", reader) == 3
    assert _count(temp_db, "Shapes") == 9


# -------------------------------------------------------------------------------------------------- declared types


def test_a_declared_property_converts_what_it_can_and_refuses_the_rest(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Typed")
    temp_db.command("sql", "CREATE PROPERTY Typed.qty INTEGER")
    temp_db.command("sql", "CREATE PROPERTY Typed.label STRING")
    ok = pd.DataFrame(
        {"qty": np.array([1, 2], dtype=np.int64), "label": np.array([7, 8])}
    )
    assert temp_db.insert_dataframe("Typed", ok) == 2
    row = _rows(temp_db, "SELECT FROM Typed WHERE qty = 1")[0]
    assert row["qty"] == 1 and row["label"] == "7"  # an integer column into STRING

    with pytest.raises(TypeError, match=r"Typed\.qty. is declared INTEGER"):
        temp_db.insert_dataframe("Typed", pd.DataFrame({"qty": [1.5, 2.5]}))
    assert _count(temp_db, "Typed") == 2  # nothing of the refused frame was written

    fixed = pd.DataFrame({"qty": [1.5, 2.5]}).astype("int64")
    assert temp_db.insert_dataframe("Typed", fixed) == 2


def test_a_value_the_declared_type_cannot_hold_fails_the_load_and_rolls_it_back(
    temp_db,
):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Small")
    temp_db.command("sql", "CREATE PROPERTY Small.v INTEGER")
    with pytest.raises(ArcadeDBError):
        temp_db.insert_dataframe("Small", pd.DataFrame({"v": [1, 2**40]}))
    assert _count(temp_db, "Small") == 0


def test_an_undeclared_column_is_stored_as_its_own_type(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Loose")
    temp_db.command("sql", "CREATE PROPERTY Loose.a INTEGER")
    temp_db.insert_dataframe("Loose", pd.DataFrame({"a": [1], "b": [2.5]}))
    row = _rows(temp_db, "SELECT FROM Loose")[0]
    assert row["a"] == 1 and row["b"] == 2.5 and isinstance(row["b"], float)


def test_vertex_types_get_vertices_and_edge_types_are_refused(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE City")
    temp_db.command("sql", "CREATE EDGE TYPE Road")
    assert temp_db.insert_dataframe("City", pd.DataFrame({"name": ["a", "b"]})) == 2
    assert _count(temp_db, "City") == 2
    with pytest.raises(ArcadeDBError, match="edge type"):
        temp_db.insert_dataframe("Road", pd.DataFrame({"a": [1]}))
    with pytest.raises(ArcadeDBError, match="does not exist"):
        temp_db.insert_dataframe("Nope", pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="vertex type"):
        temp_db.insert_dataframe("City", pd.DataFrame({"a": [1]}), parallel=True)


# -------------------------------------------------------------------------------------------------- batches and failures


def test_rows_are_converted_one_batch_at_a_time(temp_db, monkeypatch):
    from arcadedb import dataframe

    temp_db.command("sql", "CREATE DOCUMENT TYPE Batched")
    sizes = []
    events = []
    real = dataframe._column_values
    real_table = dataframe.to_arrow_table

    def table_spy(data, **kwargs):
        table = real_table(data, **kwargs)
        events.append(f"table:{table.num_rows}")
        return table

    monkeypatch.setattr(dataframe, "to_arrow_table", table_spy)

    def spy(name, chunk, *args, **kwargs):
        if name == "v":
            sizes.append(len(chunk))
            events.append(f"batch:{len(chunk)}")
        return real(name, chunk, *args, **kwargs)

    monkeypatch.setattr(dataframe, "_column_values", spy)
    assert (
        temp_db.insert_dataframe(
            "Batched", pd.DataFrame({"v": range(10)}), batch_size=4
        )
        == 10
    )
    assert sizes == [4, 4, 2]
    # the whole frame is one Arrow table before the first batch is converted
    assert events == ["table:10", "batch:4", "batch:4", "batch:2"]

    # a table that already comes in chunks keeps them: a chunk is never merged with the next one
    sizes.clear()
    chunked = pa.concat_tables([pa.table({"v": [0, 1, 2]}), pa.table({"v": [3, 4]})])
    assert temp_db.insert_arrow("Batched", chunked, batch_size=100) == 5
    assert sizes == [3, 2]


def _unique_id_type(db, name):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    db.command("sql", f"CREATE INDEX ON {name} (id) UNIQUE")


def test_a_failed_batch_is_rolled_back_and_the_earlier_ones_stay(temp_db):
    _unique_id_type(temp_db, "Dup")
    frame = pd.DataFrame({"id": list(range(10)) + [3]})  # the 11th row repeats id 3
    with pytest.raises(ArcadeDBError) as error:
        temp_db.insert_dataframe("Dup", frame, batch_size=10)
    assert "10 row(s) of the batches before this one were committed" in str(error.value)
    assert _count(temp_db, "Dup") == 10


def test_a_load_inside_your_transaction_joins_it(temp_db):
    _unique_id_type(temp_db, "Mine")
    temp_db.begin()
    try:
        assert (
            temp_db.insert_dataframe(
                "Mine", pd.DataFrame({"id": [1, 2, 3]}), batch_size=2
            )
            == 3
        )
        assert temp_db.is_transaction_active() is True
        assert _count(temp_db, "Mine") == 3
    finally:
        temp_db.rollback()
    assert _count(temp_db, "Mine") == 0


def test_parallel_loads_documents_through_the_async_writers(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Par")
    assert (
        temp_db.insert_dataframe(
            "Par", pd.DataFrame({"id": range(500)}), batch_size=200, parallel=True
        )
        == 500
    )
    assert _count(temp_db, "Par") == 500


# -------------------------------------------------------------------------------------------------- insert_columns


def _columns_type(db, name="Col"):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    db.command("sql", f"CREATE PROPERTY {name}.qty INTEGER")
    db.command("sql", f"CREATE PROPERTY {name}.label STRING")


def test_insert_columns_takes_arrays_lists_and_none(temp_db):
    _columns_type(temp_db)
    written = temp_db.insert_columns(
        "Col",
        {
            "id": np.arange(3, dtype=np.int64),
            "qty": np.array([10, 20, 30], dtype=np.int32),
            "label": ["a", None, "c"],
            "score": np.array([0.5, float("nan"), 2.5]),
        },
    )
    assert written == 3
    rows = {r["id"]: r for r in _rows(temp_db, "SELECT FROM Col")}
    assert rows[0]["label"] == "a" and rows[0]["qty"] == 10
    assert rows[1].get("label") is None  # None is a missing value
    nan = rows[1]["score"]
    assert nan != nan  # a NaN in a numpy float column stays a NaN, it is not a null


def test_insert_columns_refuses_datetime64_and_takes_datetime_objects(temp_db):
    _columns_type(temp_db)
    with pytest.raises(TypeError, match="insert_many"):
        temp_db.insert_columns(
            "Col",
            {
                "id": np.arange(2),
                "when": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
            },
        )
    assert _count(temp_db, "Col") == 0

    stamps = [dt.datetime(2026, 1, 2, 3, 4, 5), dt.datetime(2026, 1, 3, 0, 0, 0)]
    assert temp_db.insert_columns("Col", {"id": np.arange(2), "when": stamps}) == 2
    got = [r["when"] for r in _rows(temp_db, "SELECT FROM Col ORDER BY id")]
    assert got == stamps

    # the same dates in a frame go through insert_dataframe, which reads them as Arrow timestamps
    temp_db.command("sql", "CREATE DOCUMENT TYPE Dated")
    frame = pd.DataFrame(
        {"id": [0, 1], "when": pd.to_datetime(["2026-01-02", "2026-01-03"])}
    )
    assert temp_db.insert_dataframe("Dated", frame) == 2
    assert [r["when"] for r in _rows(temp_db, "SELECT FROM Dated ORDER BY id")] == [
        dt.datetime(2026, 1, 2),
        dt.datetime(2026, 1, 3),
    ]


def test_insert_columns_refuses_bad_input_before_it_writes(temp_db):
    _columns_type(temp_db)
    with pytest.raises(ValueError, match="differ in length"):
        temp_db.insert_columns("Col", {"id": np.arange(3), "qty": np.arange(2)})
    with pytest.raises(ValueError, match="at least one column"):
        temp_db.insert_columns("Col", {})
    with pytest.raises(TypeError, match="not a sequence of values"):
        temp_db.insert_columns("Col", {"label": "abc"})
    with pytest.raises(ValueError, match="64-bit signed"):
        temp_db.insert_columns("Col", {"id": np.array([1, 2**63 + 5], dtype=np.uint64)})
    with pytest.raises(ValueError, match="one-dimensional"):
        temp_db.insert_columns("Col", {"id": np.zeros((2, 2), dtype=np.int64)})
    assert _count(temp_db, "Col") == 0


def test_insert_columns_does_not_check_a_declared_type_the_way_arrow_does(temp_db):
    """insert_dataframe refuses a float column for an INTEGER property (see test_a_declared_property_converts_what_it_can_and_refuses_the_rest);
    insert_columns hands the values to the engine, which converts them (a float to an INTEGER drops the fraction).
    """
    _columns_type(temp_db)
    temp_db.insert_columns("Col", {"id": np.arange(2), "qty": np.array([1.5, 2.5])})
    assert [r["qty"] for r in _rows(temp_db, "SELECT FROM Col ORDER BY id")] == [1, 2]


def test_insert_columns_commits_every_batch_it_finished(temp_db):
    _unique_id_type(temp_db, "Chunked")
    ids = np.arange(30, dtype=np.int64)
    ids[25] = ids[3]  # a duplicate key in the third batch of ten
    with pytest.raises(ArcadeDBError):
        temp_db.insert_columns("Chunked", {"id": ids}, commit_every=10)
    assert (
        _count(temp_db, "Chunked") == 20
    )  # two batches committed, the third rolled back

    assert temp_db.insert_columns("Chunked", {"id": np.arange(100, 105)}) == 5


def test_insert_columns_inside_your_transaction_commits_nothing(temp_db):
    _columns_type(temp_db)
    temp_db.begin()
    try:
        assert (
            temp_db.insert_columns("Col", {"id": np.arange(20)}, commit_every=5) == 20
        )
        assert temp_db.is_transaction_active() is True
    finally:
        temp_db.rollback()
    assert _count(temp_db, "Col") == 0


def test_a_float32_matrix_is_a_column_of_vectors(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Vec")
    temp_db.command("sql", "CREATE PROPERTY Vec.v ARRAY_OF_FLOATS")
    matrix = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    assert temp_db.insert_columns("Vec", {"id": np.arange(2), "v": matrix}) == 2
    rows = _rows(temp_db, "SELECT FROM Vec ORDER BY id")
    assert [list(r["v"]) for r in rows] == [[0.5, 1.5], [2.5, 3.5]]


def test_insert_columns_takes_a_pandas_frame(temp_db):
    _columns_type(temp_db)
    frame = pd.DataFrame({"id": [0, 1], "label": ["a", "b"]})
    assert temp_db.insert_columns("Col", frame) == 2
    assert [r["label"] for r in _rows(temp_db, "SELECT FROM Col ORDER BY id")] == [
        "a",
        "b",
    ]


# -------------------------------------------------------------------------------------------------- reading columns


@pytest.fixture
def readings(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Rec")
    stamp = dt.datetime(2026, 10, 3, 12, 34, 56, 123456)
    with temp_db.transaction():
        temp_db.command(
            "sql",
            "INSERT INTO Rec SET k = 'a', n = 7, ok = true, s = 'x', x = 1.5, t = :t",
            {"t": stamp},
        )
        temp_db.command("sql", "INSERT INTO Rec SET k = 'b'")
    return temp_db


QUERY = "SELECT k, n, ok, s, x, t FROM Rec ORDER BY k"


def test_to_columns_follows_the_pandas_conventions_for_missing_values(readings):
    columns = readings.query("sql", QUERY).to_columns()
    assert set(columns) == {"k", "n", "ok", "s", "x", "t"}
    assert columns["k"] == ["a", "b"]
    assert (
        columns["n"].dtype == np.float64
    )  # an integer column with a gap is a float column
    assert columns["n"][0] == 7.0 and np.isnan(columns["n"][1])
    assert columns["ok"] == [True, None]  # a boolean column with a gap is a list
    assert columns["s"] == ["x", None]
    assert columns["x"].dtype == np.float64 and np.isnan(columns["x"][1])
    assert columns["t"].dtype == np.dtype("datetime64[ms]") and np.isnat(
        columns["t"][1]
    )
    assert columns["t"][0] == np.datetime64("2026-10-03T12:34:56.123", "ms")


def test_to_columns_without_a_gap_is_one_typed_array(readings):
    columns = readings.query("sql", "SELECT k, n FROM Rec WHERE k = 'a'").to_columns()
    assert columns["n"].dtype == np.int64 and columns["n"].tolist() == [7]


def test_to_arrow_keeps_the_type_of_a_column_with_a_gap(readings):
    table = readings.query("sql", QUERY).to_arrow()
    assert table.schema.field("n").type == pa.int64()
    assert table.column("n").to_pylist() == [7, None]
    assert table.schema.field("ok").type == pa.bool_()
    assert table.column("ok").to_pylist() == [True, None]
    assert table.column("s").to_pylist() == ["x", None]
    assert table.schema.field("t").type == pa.timestamp("ms")
    assert table.column("t").to_pylist() == [
        dt.datetime(2026, 10, 3, 12, 34, 56, 123000),
        None,
    ]


def test_to_dataframe_and_to_polars(readings):
    frame = readings.query("sql", QUERY).to_dataframe()
    assert set(frame.columns) == {"k", "n", "ok", "s", "x", "t"}
    assert frame["t"].iloc[0].to_pydatetime() == dt.datetime(
        2026, 10, 3, 12, 34, 56, 123456
    )
    assert pd.isna(frame["t"].iloc[1]) and pd.isna(frame["n"].iloc[1])
    assert frame["n"].dtype == np.float64  # a gap in an integer column makes it float64

    polars_frame = readings.query("sql", QUERY).to_polars()
    assert set(polars_frame.columns) == {"k", "n", "ok", "s", "x", "t"}
    assert polars_frame["n"].to_list() == [7, None]  # Polars keeps the integer type
    assert polars_frame["n"].dtype == pl.Int64


def test_columns_pins_the_names_and_a_missing_one_reads_as_missing(readings):
    table = readings.query("sql", QUERY).to_arrow(columns=["n", "nope"])
    assert table.column_names == ["n", "nope"]
    assert table.column("nope").to_pylist() == [None, None]
    columns = readings.query("sql", QUERY).to_columns(columns=["k"])
    assert list(columns) == ["k"]


def test_the_columns_do_not_depend_on_batch_size(readings):
    reference = readings.query("sql", QUERY).to_arrow()
    for size in (1, 2, 25_000):
        assert readings.query("sql", QUERY).to_arrow(batch_size=size).equals(reference)
        columns = readings.query("sql", QUERY).to_columns(batch_size=size)
        assert columns["k"] == ["a", "b"] and columns["n"][0] == 7.0


def test_how_the_rows_reach_the_columns_differs_between_the_builds(
    readings, monkeypatch
):
    """The JVM build reads the result in batches of ``batch_size`` rows from the engine's column buffer. The native build reads
    every row into Python dicts first (``_drain_dicts``) and builds the columns from them, so ``batch_size`` is not used.
    """
    from arcadedb import results

    if arcadedb.__backend__ == "native":
        resultset = type(readings.query("sql", QUERY))
        drained = []
        real = resultset._drain_dicts

        def spy(self, *args, **kwargs):
            rows = real(self, *args, **kwargs)
            drained.append(len(rows))
            return rows

        monkeypatch.setattr(resultset, "_drain_dicts", spy)
        readings.query("sql", QUERY).to_columns(batch_size=1)
        assert drained == [2]  # one read of all the rows, whatever batch_size says
    else:
        batches = []
        real_bridge = results._bridge_class

        class Spy:
            def __init__(self, inner):
                self._inner = inner

            def nextColumnBatch(
                self, java_rows, batch_size, *rest
            ):  # noqa: N802 - the Java name
                batches.append(int(batch_size))
                return self._inner.nextColumnBatch(java_rows, batch_size, *rest)

        def bridge(name):
            inner = real_bridge(name)
            return Spy(inner) if name == "ColumnBatcher" else inner

        monkeypatch.setattr(results, "_bridge_class", bridge)
        readings.query("sql", QUERY).to_columns(batch_size=1)
        assert batches[:2] == [1, 1]


def test_an_empty_result_has_no_columns(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Nothing")
    assert temp_db.query("sql", "SELECT FROM Nothing").to_columns() == {}


def test_a_missing_library_returns_none_or_raises_import_error(readings, monkeypatch):
    # the result sets are made first: only the call under test runs without the library
    for_arrow = readings.query("sql", QUERY)
    for_polars = readings.query("sql", QUERY)
    for_columns = readings.query("sql", QUERY)
    for_frame = readings.query("sql", QUERY)

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    assert for_arrow.to_arrow() is None
    with pytest.raises(ImportError):
        for_polars.to_polars()
    monkeypatch.undo()

    monkeypatch.setitem(sys.modules, "numpy", None)
    assert for_columns.to_columns() is None
    monkeypatch.undo()

    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="pandas") as error:
        for_frame.to_dataframe()
    # the JVM build adds the install hint to the message, the native build raises the plain ImportError
    assert ("uv add" in str(error.value)) == (arcadedb.__backend__ != "native")


def test_a_vertex_frame_inside_your_transaction_is_refused_and_leaves_it_alone(
    temp_db,
):
    """The graph batch behind a vertex load refuses to run inside a transaction of yours (test_transactions_behavior.py)."""
    temp_db.command("sql", "CREATE VERTEX TYPE Town")
    temp_db.begin()
    try:
        with pytest.raises(ArcadeDBError):
            temp_db.insert_dataframe("Town", pd.DataFrame({"name": ["a", "b"]}))
        assert temp_db.is_transaction_active() is True
    finally:
        temp_db.rollback()
    assert _count(temp_db, "Town") == 0
