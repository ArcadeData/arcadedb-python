"""A relative path names a place under Python's current directory, also after ``os.chdir``.

The engine resolves a part of the files of a relative path against the directory that was current when it started, so one
relative path could name two directories. Every test starts the engine first and changes the directory afterwards.
"""

import os

import pytest

import arcadedb
from tests.conftest import TEST_PASSWORD, free_port


@pytest.fixture
def two_dirs(tmp_path, monkeypatch):
    """Two empty directories, made after the engine has started somewhere else."""
    monkeypatch.chdir(tmp_path)
    with arcadedb.create_database(str(tmp_path / "engine_is_running")):
        pass
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    return first, second


def _same(path, expected):
    return os.path.realpath(path) == os.path.realpath(expected)


def _types(db):
    return [name for name in ("InFirst", "InSecond") if db.schema.exists_type(name)]


def _create_with_type(path, type_name):
    with arcadedb.create_database(path) as db:
        db.command("sql", f"CREATE DOCUMENT TYPE {type_name}")
        with db.transaction():
            db.command("sql", f"INSERT INTO {type_name} SET n = 1")


def test_a_relative_path_names_the_database_of_the_current_directory(
    two_dirs, monkeypatch
):
    first, second = two_dirs
    monkeypatch.chdir(first)
    _create_with_type("./db", "InFirst")
    monkeypatch.chdir(second)
    _create_with_type("./db", "InSecond")

    for directory, type_name in ((first, "InFirst"), (second, "InSecond")):
        assert (directory / "db" / "schema.json").is_file()
        monkeypatch.chdir(directory)
        with arcadedb.open_database("./db") as db:
            assert _types(db) == [type_name]
            assert db.count_type(type_name) == 1

    # still in the second directory
    with arcadedb.open_database(os.path.join("..", "first", "db")) as db:
        assert _types(db) == ["InFirst"]
    # nothing appeared in the directory that the engine saw first
    assert not (first.parent / "db").exists()


def test_exists_and_open_or_create_follow_the_current_directory(two_dirs, monkeypatch):
    first, second = two_dirs
    monkeypatch.chdir(first)
    with arcadedb.open_or_create_database("db") as db:
        db.command("sql", "CREATE DOCUMENT TYPE InFirst")
    assert arcadedb.database_exists("db")

    monkeypatch.chdir(second)
    assert not arcadedb.database_exists("db")
    with arcadedb.open_or_create_database("db") as db:
        assert _types(db) == []

    monkeypatch.chdir(first)
    with arcadedb.open_or_create_database("db") as db:
        assert _types(db) == ["InFirst"]


def test_one_relative_path_opens_two_databases_from_two_directories(
    two_dirs, monkeypatch
):
    first, second = two_dirs
    monkeypatch.chdir(first)
    with arcadedb.create_database("db") as in_first:
        in_first.command("sql", "CREATE DOCUMENT TYPE InFirst")
        monkeypatch.chdir(second)
        with arcadedb.create_database("db") as in_second:
            in_second.command("sql", "CREATE DOCUMENT TYPE InSecond")
            assert _types(in_second) == ["InSecond"]
            assert _types(in_first) == ["InFirst"]


def test_a_factory_keeps_the_directory_it_was_made_in(two_dirs, monkeypatch):
    first, second = two_dirs
    monkeypatch.chdir(first)
    factory = arcadedb.DatabaseFactory("db")
    monkeypatch.chdir(second)
    assert not factory.exists()
    factory.create().close()
    assert factory.exists()
    assert (first / "db" / "schema.json").is_file()
    assert not (second / "db").exists()


def test_the_database_path_is_absolute(two_dirs, monkeypatch):
    first, second = two_dirs
    monkeypatch.chdir(first)
    with arcadedb.create_database("./db") as db:
        monkeypatch.chdir(second)
        path = db.get_database_path()
        assert os.path.isabs(path), path
        assert _same(path, first / "db"), path


@pytest.mark.server
@pytest.mark.parametrize(
    "make", [arcadedb.ArcadeDBServer, arcadedb.create_server], ids=lambda f: f.__name__
)
def test_a_relative_server_root_follows_the_current_directory(
    two_dirs, monkeypatch, make
):
    first, second = two_dirs
    monkeypatch.chdir(first)
    server = make(
        root_path="./server_root",
        root_password=TEST_PASSWORD,
        config={"http_port": free_port()},
    )
    # the root is taken from the directory that is current when the server is made
    monkeypatch.chdir(second)
    with server:
        database = server.create_database("served")
        path = database.get_database_path()
        assert _same(path, first / "server_root" / "databases" / "served"), path
        assert (
            first / "server_root" / "databases" / "served" / "schema.json"
        ).is_file()
    assert not (second / "server_root").exists()


def test_relative_export_and_import_paths_follow_the_current_directory(
    two_dirs, monkeypatch
):
    first, second = two_dirs
    monkeypatch.chdir(second)
    (second / "rows.csv").write_text("name,n\nb,1\nc,2\n", encoding="utf-8")
    with arcadedb.create_database(str(first / "db")) as db:
        db.import_documents("rows.csv", document_type="Imported")
        assert db.count_type("Imported") == 2
        db.export_database("backup.jsonl.tgz")
    assert (second / "exports" / "backup.jsonl.tgz").is_file()
