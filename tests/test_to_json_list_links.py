"""``to_json_list`` returns the text of a link (``"#10:5"``) on both builds, in a column, in a list and inside a map, and a map
with keys that are not strings comes back with string keys (a JSON object has no other kind). The native build returned its ``RID``
object for a link, which ``json.dumps`` refuses (#219); the JVM build writes the text. The expected values are the JVM build's.
"""

import json

import pytest


@pytest.fixture
def linked(temp_db):
    db = temp_db
    db.command("sql", "CREATE VERTEX TYPE P")
    db.command("sql", "CREATE PROPERTY P.friend LINK")
    db.command("sql", "CREATE PROPERTY P.friends LIST OF LINK")
    db.command("sql", "CREATE DOCUMENT TYPE D")
    db.command("sql", "CREATE PROPERTY D.m MAP")
    with db.transaction():
        db.command("sql", "CREATE VERTEX P SET n = 'a'")
        db.command("sql", "CREATE VERTEX P SET n = 'b'")
        db.command(
            "sql",
            "UPDATE P SET friend = (SELECT FROM P WHERE n = 'a'), friends = (SELECT FROM P) WHERE n = 'b'",
        )
        db.command("sql", "INSERT INTO D SET m = :m", {"m": {1: "a", 2: "b"}})
    rids = [
        row["s"]
        for row in db.query(
            "sql", "SELECT @rid.asString() AS s FROM P ORDER BY n"
        ).to_list()
    ]
    assert len(rids) == 2 and all(r.startswith("#") for r in rids)
    return db, rids


def test_a_link_in_a_column_is_its_text(linked):
    db, (a, b) = linked
    got = db.query("sql", "SELECT @rid AS r FROM P ORDER BY n").to_json_list()
    assert got == [{"r": a}, {"r": b}]
    assert all(type(row["r"]) is str for row in got)
    json.dumps(got)


def test_a_link_property_and_a_list_of_links_are_texts(linked):
    db, (a, b) = linked
    got = db.query("sql", "SELECT friend, friends FROM P WHERE n = 'b'").to_json_list()
    assert got == [{"friend": a, "friends": [a, b]}]
    assert type(got[0]["friend"]) is str
    assert [type(x) for x in got[0]["friends"]] == [str, str]
    json.dumps(got)


def test_a_link_inside_a_map_is_its_text(linked):
    db, (a, b) = linked
    got = db.query("sql", "SELECT {'k': @rid} AS m FROM P WHERE n = 'b'").to_json_list()
    assert got == [{"m": {"k": b}}]
    assert type(got[0]["m"]["k"]) is str
    json.dumps(got)


def test_a_map_with_integer_keys_has_string_keys(linked):
    db, _ = linked
    got = db.query("sql", "SELECT m FROM D").to_json_list()
    assert got == [{"m": {"1": "a", "2": "b"}}]
    assert all(type(k) is str for k in got[0]["m"])
    json.dumps(got)
