"""What importing and exporting does, as the how-to page ``docs/modules/ROOT/pages/how-to/import-export.adoc`` says it (ArcadeData/arcadedb-python#267).

Each statement of that page is pinned here first: the result of ``import_documents``, ``probe_only``, what is left in the database when an
import fails (the default ``abort`` and ``on_row_error="skip"``), an import inside the caller's transaction, the SQL ``IMPORT DATABASE`` of the
same importer (and the one failure it reports as a row instead of raising), ``export_database`` (where the file lands, the filters,
``overwrite``, the formats), the SQL ``EXPORT DATABASE`` and ``export_to_csv``. The reading of the engine source behind the transaction rules is
named next to each test. The tests use SQL and the public calls only, so they run on the JVM build and the native build alike.
"""

import csv
import os
import shutil
import uuid
from pathlib import Path

import pytest

import arcadedb
from arcadedb import ArcadeDBError


def _uri(path) -> str:
    return Path(path).resolve().as_uri()


def _count(db, type_name) -> int:
    query = f"SELECT count(*) AS n FROM {type_name}"  # nosec B608 - test-controlled
    return int(db.query("sql", query).to_list()[0]["n"])


def _people_csv(directory) -> Path:
    path = Path(directory) / "people.csv"
    path.write_text(
        "name,age,city\nAlice,30,New York\nBob,25,London\nCharlie,35,Paris\n",
        encoding="utf-8",
    )
    return path


def _duplicate_key_csv(directory) -> Path:
    """Three rows, the middle one repeats the unique key of the first."""
    path = Path(directory) / "products.csv"
    path.write_text("sku,name\nA-1,first\nA-1,duplicate\nB-2,third\n", encoding="utf-8")
    return path


@pytest.fixture
def unique_product(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Product")
    temp_db.command("sql", "CREATE PROPERTY Product.sku STRING")
    temp_db.command("sql", "CREATE INDEX ON Product (sku) UNIQUE")
    return temp_db


# -------------------------------------------------------------------------------------------------- import_documents


def test_import_documents_returns_an_import_result(temp_db, tmp_path):
    csv_path = _people_csv(tmp_path)
    result = temp_db.import_documents(csv_path, "Person")
    assert result.result == "OK"
    assert result.operation == "import documents"
    assert result.source_url.startswith("file:")
    assert isinstance(result.statistics, dict)
    assert result.get("no such statistic", 7) == 7
    assert {"result", "operation", "source_url"} <= set(result.to_dict())

    assert _count(temp_db, "Person") == 3
    row = temp_db.query("sql", "SELECT FROM Person WHERE name = 'Alice'").one()
    assert row.get("city") == "New York"


def test_import_documents_creates_a_missing_type(temp_db, tmp_path):
    csv_path = _people_csv(tmp_path)
    temp_db.import_documents(csv_path, document_type="Person")
    assert _count(temp_db, "Person") == 3


def test_probe_only_reads_the_file_and_writes_nothing(temp_db, tmp_path):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Probe")
    result = temp_db.import_documents(
        _people_csv(tmp_path), document_type="Probe", probe_only=True
    )
    assert result.result == "PROBE_ONLY"
    assert _count(temp_db, "Probe") == 0


def test_the_keyword_arguments_are_the_importer_settings_of_the_sql_statement():
    from arcadedb.importer import _build_import_settings

    settings = _build_import_settings(
        "file:///x.csv",
        document_type="Person",
        file_type="csv",
        delimiter=";",
        header="a;b",
        skip_entries=1,
        properties_include="a,b",
        commit_every=100,
        parallel=2,
        wal=False,
        verbose_level=0,
        probe_only=True,
        force_database_create=False,
        trim_text=False,
        on_row_error="skip",
        extra_settings={"typeIdProperty": "id", "ignored": None},
    )
    assert settings == {
        "documents": "file:///x.csv",
        "documentType": "Person",
        "documentsFileType": "csv",
        "documentsDelimiter": ";",
        "documentsHeader": "a;b",
        "documentsSkipEntries": 1,
        "documentPropertiesInclude": "a,b",
        "commitEvery": 100,
        "parallel": 2,
        "wal": False,
        "verboseLevel": 0,
        "probeOnly": True,
        "forceDatabaseCreate": False,
        "trimText": False,
        "onRowError": "skip",
        "typeIdProperty": "id",
    }
    # an argument left at None is not sent at all, so the engine default applies
    assert _build_import_settings(
        "file:///x.csv",
        document_type="D",
        file_type=None,
        delimiter=None,
        header=None,
        skip_entries=None,
        properties_include=None,
        commit_every=None,
        parallel=None,
        wal=None,
        verbose_level=None,
        probe_only=None,
        force_database_create=None,
        trim_text=None,
        on_row_error=None,
        extra_settings=None,
    ) == {"documents": "file:///x.csv", "documentType": "D"}


def test_extra_settings_win_over_the_arguments():
    from arcadedb.importer import _build_import_settings

    kwargs = dict(
        document_type="D",
        file_type=None,
        delimiter=None,
        header=None,
        skip_entries=None,
        properties_include=None,
        parallel=None,
        wal=None,
        verbose_level=None,
        probe_only=None,
        force_database_create=None,
        trim_text=None,
        on_row_error=None,
    )
    settings = _build_import_settings(
        "file:///x.csv", commit_every=100, extra_settings={"commitEvery": 5}, **kwargs
    )
    assert settings["commitEvery"] == 5


def test_a_setting_the_importer_cannot_read_raises(temp_db, tmp_path):
    with pytest.raises(ArcadeDBError, match="Document import failed"):
        temp_db.import_documents(
            _people_csv(tmp_path),
            document_type="Person",
            extra_settings={"commitEvery": "not a number"},
        )
    with pytest.raises(ValueError, match="Invalid on_row_error mode"):
        temp_db.import_documents(
            _people_csv(tmp_path), document_type="Person", on_row_error="ignore"
        )


# -------------------------------------------------------------------------------------------------- a failed import


def test_a_failed_import_leaves_none_of_its_rows(unique_product, tmp_path):
    """The documents of one file load in one transaction that the import owns (``CSVImporterFormat.loadDocuments``: one
    ``database.begin()``, one ``commit()`` after the last row, a rollback of the owned transaction on any failure).
    """
    csv_path = _duplicate_key_csv(tmp_path)
    with pytest.raises(ArcadeDBError, match="Document import failed"):
        unique_product.import_documents(csv_path, document_type="Product")
    assert _count(unique_product, "Product") == 0
    assert unique_product.is_transaction_active() is False

    with pytest.raises(ArcadeDBError, match="Document import failed"):
        unique_product.import_documents(
            csv_path, document_type="Product", on_row_error="abort"
        )
    assert _count(unique_product, "Product") == 0


def test_skip_keeps_the_good_rows_and_drops_the_bad_one(unique_product, tmp_path):
    unique_product.import_documents(
        _duplicate_key_csv(tmp_path), document_type="Product", on_row_error="skip"
    )
    skus = sorted(
        r.get("sku") for r in unique_product.query("sql", "SELECT sku FROM Product")
    )
    assert skus == ["A-1", "B-2"]
    first = unique_product.query("sql", "SELECT FROM Product WHERE sku = 'A-1'").one()
    assert first.get("name") == "first"  # the first row won, the duplicate was dropped


def test_skip_needs_the_transaction_for_itself(unique_product, tmp_path):
    unique_product.begin()
    try:
        with pytest.raises(ArcadeDBError):
            unique_product.import_documents(
                _duplicate_key_csv(tmp_path),
                document_type="Product",
                on_row_error="skip",
            )
    finally:
        unique_product.rollback()
    assert _count(unique_product, "Product") == 0


def test_an_import_inside_your_transaction_commits_nothing(temp_db, tmp_path):
    """``CSVImporterFormat``: a transaction that was active on entry is not the import's, so it neither commits nor rolls it back."""
    temp_db.command("sql", "CREATE DOCUMENT TYPE Person")
    temp_db.begin()
    try:
        temp_db.import_documents(_people_csv(tmp_path), document_type="Person")
        assert temp_db.is_transaction_active() is True
        assert _count(temp_db, "Person") == 3
    finally:
        temp_db.rollback()
    assert _count(temp_db, "Person") == 0


# -------------------------------------------------------------------------------------------------- IMPORT DATABASE


def test_sql_import_database_is_the_same_importer(temp_db, tmp_path):
    url = _uri(_people_csv(tmp_path))
    row = temp_db.command("sql", f"IMPORT DATABASE {url}").one()
    assert row.get("result") == "OK"
    assert row.get("operation") == "import database"
    assert _count(temp_db, "`Document`") == 3  # the default type of a file of documents

    row = temp_db.command(
        "sql",
        f"IMPORT DATABASE WITH documents = '{url}', documentType = 'Person', "
        "documentsFileType = 'csv', commitEvery = 2",
    ).one()
    assert row.get("result") == "OK"
    assert _count(temp_db, "Person") == 3


def test_sql_import_of_a_missing_file_and_of_a_bad_row_raises(unique_product, tmp_path):
    with pytest.raises(ArcadeDBError):
        unique_product.command(
            "sql", f"IMPORT DATABASE {_uri(Path(tmp_path) / 'missing.csv')}"
        )
    url = _uri(_duplicate_key_csv(tmp_path))
    with pytest.raises(ArcadeDBError):
        unique_product.command(
            "sql", f"IMPORT DATABASE WITH documents = '{url}', documentType = 'Product'"
        )
    assert _count(unique_product, "Product") == 0


def test_sql_import_reports_an_unusable_setting_as_a_row(temp_db, tmp_path):
    """``ImportDatabaseStatement`` turns an ``IllegalArgumentException`` of the importer (a ``NumberFormatException`` of a setting
    value is one) into a ``FAIL`` row with the reason, where the Python call raises (see test_a_setting_the_importer_cannot_read_raises).
    """
    url = _uri(_people_csv(tmp_path))
    row = temp_db.command(
        "sql",
        f"IMPORT DATABASE WITH documents = '{url}', documentType = 'Person', "
        "commitEvery = 'not a number'",
    ).one()
    assert row.get("result") == "FAIL"
    assert row.get("reason")


# -------------------------------------------------------------------------------------------------- export_database


@pytest.fixture
def graph_db(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE Person")
    temp_db.command("sql", "CREATE EDGE TYPE Knows")
    temp_db.command("sql", "CREATE DOCUMENT TYPE Note")
    with temp_db.transaction():
        temp_db.command("sql", "CREATE VERTEX Person SET name = 'a'")
        temp_db.command("sql", "CREATE VERTEX Person SET name = 'b'")
        temp_db.command(
            "sql",
            "CREATE EDGE Knows FROM (SELECT FROM Person WHERE name = 'a') "
            "TO (SELECT FROM Person WHERE name = 'b')",
        )
        temp_db.command("sql", "INSERT INTO Note SET text = 'n1'")
        temp_db.command("sql", "INSERT INTO Note SET text = 'n2'")
        temp_db.command("sql", "INSERT INTO Note SET text = 'n3'")
    return temp_db


def test_export_database_writes_the_absolute_path_it_is_given(graph_db, tmp_path):
    target = tmp_path / "backup.jsonl.tgz"
    stats = graph_db.export_database(str(target))
    assert target.exists() and target.stat().st_size > 0
    assert int(stats["vertices"]) == 2
    assert int(stats["edges"]) == 1
    assert int(stats["documents"]) == 3
    assert int(stats["totalRecords"]) == 6


def test_a_relative_name_lands_in_exports_below_the_working_directory(graph_db):
    name = f"adb-test-{uuid.uuid4().hex}.jsonl.tgz"
    target = Path("exports") / name
    try:
        graph_db.export_database(name)
        assert target.exists()
        # the same text with the directory in front is the same file, not exports/exports
        graph_db.export_database(f"exports/{name}", overwrite=True)
        assert not (Path("exports") / "exports").exists()
    finally:
        if target.exists():
            target.unlink()


def test_include_and_exclude_types(graph_db, tmp_path):
    only = graph_db.export_database(
        str(tmp_path / "only.jsonl.tgz"), include_types=["Person"]
    )
    assert int(only["vertices"]) == 2
    assert int(only.get("edges", 0)) == 0 and int(only.get("documents", 0)) == 0

    with_edges = graph_db.export_database(
        str(tmp_path / "with_edges.jsonl.tgz"), include_types=["Person", "Knows"]
    )
    assert int(with_edges["vertices"]) == 2 and int(with_edges["edges"]) == 1

    without = graph_db.export_database(
        str(tmp_path / "without.jsonl.tgz"), exclude_types=["Note"]
    )
    assert int(without["vertices"]) == 2 and int(without["edges"]) == 1
    assert int(without.get("documents", 0)) == 0


def test_an_existing_file_is_refused_unless_overwrite_is_true(graph_db, tmp_path):
    target = str(tmp_path / "twice.jsonl.tgz")
    graph_db.export_database(target)
    with pytest.raises(ArcadeDBError, match="overwrite=True"):
        graph_db.export_database(target)
    assert int(graph_db.export_database(target, overwrite=True)["totalRecords"]) == 6


@pytest.mark.parametrize("fmt", ["graphml", "graphson"])
def test_graphml_and_graphson_are_not_bundled(graph_db, tmp_path, fmt):
    with pytest.raises(ArcadeDBError, match="jsonl"):
        graph_db.export_database(str(tmp_path / f"x.{fmt}.tgz"), format=fmt)


def test_an_unknown_format_is_refused(graph_db, tmp_path):
    with pytest.raises(ArcadeDBError, match="Invalid export format"):
        graph_db.export_database(str(tmp_path / "x.tgz"), format="parquet")


def test_the_export_imports_into_a_new_database_with_the_sql_statement(
    graph_db, tmp_path
):
    target = tmp_path / "copy.jsonl.tgz"
    graph_db.export_database(str(target))
    with arcadedb.create_database(str(tmp_path / "copy_db")) as copy:
        row = copy.command("sql", f"IMPORT DATABASE {_uri(target)}").one()
        assert row.get("result") == "OK"
        assert _count(copy, "Person") == 2
        assert _count(copy, "Knows") == 1
        assert _count(copy, "Note") == 3
        names = copy.query(
            "sql", "SELECT out('Knows').name AS friends FROM Person WHERE name = 'a'"
        ).to_list()
        assert list(names[0]["friends"]) == ["b"]


# -------------------------------------------------------------------------------------------------- EXPORT DATABASE


def test_sql_export_database_writes_into_exports(graph_db):
    name = f"adb-test-{uuid.uuid4().hex}.jsonl.tgz"
    os.makedirs("exports", exist_ok=True)
    target = Path("exports") / name
    try:
        row = graph_db.command("sql", f"EXPORT DATABASE file://{name}").one()
        assert row.get("operation") == "export database"
        assert target.exists() and target.stat().st_size > 0
    finally:
        if target.exists():
            target.unlink()


def test_sql_export_database_takes_a_name_and_no_directory(graph_db):
    """``ExportDatabaseStatement`` refuses a name that contains ``..`` or ``File.separator``: the second is ``/`` on Linux and macOS and
    ``\\`` on Windows, where a forward slash is not refused (the first CI run of this test showed it).
    """
    with pytest.raises(ArcadeDBError):
        graph_db.command("sql", "EXPORT DATABASE file://..up.jsonl.tgz")
    if os.name != "nt":
        with pytest.raises(ArcadeDBError):
            graph_db.command("sql", "EXPORT DATABASE file://sub/dir.jsonl.tgz")
    else:
        folder = f"adb-test-{uuid.uuid4().hex}"
        os.makedirs("exports", exist_ok=True)
        try:
            graph_db.command("sql", f"EXPORT DATABASE file://{folder}/dir.jsonl.tgz")
        finally:
            shutil.rmtree(Path("exports") / folder, ignore_errors=True)


def test_sql_export_database_rolls_back_your_open_transaction(graph_db):
    """``ExportDatabaseStatement`` calls ``rollbackAllNested()`` when a transaction is active."""
    name = f"adb-test-{uuid.uuid4().hex}.jsonl.tgz"
    os.makedirs("exports", exist_ok=True)
    target = Path("exports") / name
    try:
        graph_db.begin()
        graph_db.command("sql", "INSERT INTO Note SET text = 'pending'")
        graph_db.command("sql", f"EXPORT DATABASE file://{name}")
        assert graph_db.is_transaction_active() is False
    finally:
        graph_db.rollback()
        if target.exists():
            target.unlink()
    assert _count(graph_db, "Note") == 3


# -------------------------------------------------------------------------------------------------- export_to_csv


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


@pytest.fixture
def people_db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Person")
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO Person SET name = 'Alice', age = 30")
        temp_db.command("sql", "INSERT INTO Person SET name = 'Bob', age = 25")
    return temp_db


def test_export_to_csv_writes_a_header_and_the_rows(people_db, tmp_path):
    target = tmp_path / "people.csv"
    people_db.export_to_csv("SELECT name, age FROM Person ORDER BY name", str(target))
    rows = _read_csv(target)
    assert set(rows[0]) == {"name", "age"}
    by_name = {
        dict(zip(rows[0], row))["name"]: dict(zip(rows[0], row)) for row in rows[1:]
    }
    assert by_name["Alice"]["age"] == "30" and by_name["Bob"]["age"] == "25"


def test_fieldnames_choose_the_order_and_cannot_rename(people_db, tmp_path):
    target = tmp_path / "ordered.csv"
    people_db.export_to_csv(
        "SELECT name, age FROM Person ORDER BY name",
        str(target),
        fieldnames=["age", "name"],
    )
    assert _read_csv(target)[:2] == [["age", "name"], ["30", "Alice"]]

    with pytest.raises(ArcadeDBError):
        people_db.export_to_csv(
            "SELECT name, age FROM Person ORDER BY name",
            str(tmp_path / "short.csv"),
            fieldnames=["name"],
        )

    renamed = tmp_path / "renamed.csv"
    people_db.export_to_csv(
        "SELECT name AS who, age AS years FROM Person ORDER BY name",
        str(renamed),
        fieldnames=["who", "years"],
    )
    assert _read_csv(renamed)[:2] == [["who", "years"], ["Alice", "30"]]


def test_export_to_csv_takes_a_list_of_dicts_and_makes_the_folder(tmp_path):
    from arcadedb.exporter import export_to_csv

    target = tmp_path / "new" / "folder" / "rows.csv"
    export_to_csv([{"id": 1, "name": "x"}, {"id": 2, "name": "y"}], str(target))
    assert _read_csv(target) == [["id", "name"], ["1", "x"], ["2", "y"]]


def test_export_to_csv_with_no_rows_writes_the_header_only_when_it_is_given(
    people_db, tmp_path
):
    target = tmp_path / "none.csv"
    people_db.export_to_csv(
        "SELECT name, age FROM Person WHERE age > 100",
        str(target),
        fieldnames=["name", "age"],
    )
    assert _read_csv(target) == [["name", "age"]]
    bare = tmp_path / "bare.csv"
    people_db.export_to_csv("SELECT name, age FROM Person WHERE age > 100", str(bare))
    assert bare.read_text(encoding="utf-8") == ""
