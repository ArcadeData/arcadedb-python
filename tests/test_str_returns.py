"""Methods that document a ``str`` return one: a plain Python ``str``, not a ``java.lang.String``, on both packages.

``arcadedb-jvm`` reads these from Java getters, and JPype returns a ``java.lang.String`` for a getter result unless the binding
wraps it with ``str()``. A ``java.lang.String`` equals the same text but is not accepted by ``os.path.join`` or ``pathlib.Path``
(``TypeError: expected str, bytes or os.PathLike object, not String``, found by tests/test_columnar_index.py), so the type is
asserted with ``type(x) is str``. The native build builds its answers in Python and must give the same type. Same family as
ArcadeData/arcadedb-python#286.
"""

import os

import pytest


def _db_name(db):
    return db.get_name()


def _db_path(db):
    return db.get_database_path()


def _vector_index_name(db):
    db.command(
        "opencypher",
        "CREATE VECTOR INDEX str_return_embedding FOR (n:StrReturn) ON n.embedding "
        "OPTIONS {indexConfig: {`vector.dimensions`: 3}}",
    )
    names = db.schema.list_vector_indexes()
    assert "str_return_embedding" in names
    assert all(type(n) is str for n in names), [type(n) for n in names]
    return names[0]


def _result_rid(db):
    db.command("sql", "CREATE DOCUMENT TYPE StrReturnDoc")
    with db.transaction():
        db.command("sql", "INSERT INTO StrReturnDoc SET n = 1")
    return db.query("sql", "SELECT FROM StrReturnDoc").first().get_rid()


@pytest.mark.parametrize(
    "read",
    [_db_name, _db_path, _vector_index_name, _result_rid],
    ids=lambda f: f.__name__.lstrip("_"),
)
def test_documented_str_returns_are_python_str(temp_db, read):
    value = read(temp_db)
    assert type(value) is str, type(value)


def test_database_path_can_be_joined(temp_db):
    path = temp_db.get_database_path()
    assert os.path.isdir(path)
    assert os.path.join(path, "database.lck").startswith(path)
