"""A lone dict is the named-parameter map, so a statement with only ``?`` placeholders is refused, not bound to null.

``command("sql", "INSERT INTO T SET v = ?", {"a": 1})`` (and ``[{"a": 1}]``, ``({"a": 1},)``) hands the engine a named map. The
engine binds a positional ``?`` that has no key in the map to null and nothing fails before 26.11.1 (ArcadeData/arcadedb#11098, fixed in
26.11.1, where the engine refuses it), so the record is stored without the value. The bindings refuse that call with a TypeError on both builds before it runs. A dict that has a
``:name`` placeholder to bind, or a digit-string key (``{"0": v}`` binds positional slot 0 by key), keeps working.
"""

import pytest


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE T")
    with temp_db.transaction():
        for k in range(3):
            temp_db.command("sql", "INSERT INTO T SET k = ?, v = 0", k)
    return temp_db


def _stored(db, k):
    return db.query("sql", "SELECT FROM T WHERE k = :k", {"k": k}).to_list()


DICT_FORMS = [
    pytest.param(lambda d: (d,), id="dict"),
    pytest.param(lambda d: ([d],), id="list-of-dict"),
    pytest.param(lambda d: ((d,),), id="tuple-of-dict"),
]


@pytest.mark.parametrize("form", DICT_FORMS)
def test_command_refuses_a_lone_dict_for_a_positional_placeholder(db, form):
    with pytest.raises(TypeError, match="NAMED parameters"):
        with db.transaction():
            db.command("sql", "INSERT INTO T SET k = 9, v = ?", *form({"a": 1}))
    assert (
        _stored(db, 9) == []
    )  # nothing was inserted, in particular no record with v = null


@pytest.mark.parametrize("form", DICT_FORMS)
def test_query_refuses_a_lone_dict_for_a_positional_placeholder(db, form):
    with pytest.raises(TypeError, match="NAMED parameters"):
        db.query("sql", "SELECT FROM T WHERE k = ?", *form({"a": 1}))


def test_sqlscript_refuses_a_lone_dict_for_a_positional_placeholder(db):
    with pytest.raises(TypeError, match="NAMED parameters"):
        db.command("sqlscript", "INSERT INTO T SET k = 9, v = ?;", {"a": 1})
    assert _stored(db, 9) == []


def test_an_empty_dict_is_refused_too(db):
    with pytest.raises(TypeError, match="NAMED parameters"):
        db.query("sql", "SELECT FROM T WHERE k = ?", {})


def test_the_refusal_is_a_wrong_call_not_a_failed_statement(db):
    with pytest.raises(TypeError) as caught:
        db.query("sql", "SELECT FROM T WHERE k = ?", {"a": 1})
    message = str(caught.value)
    assert not message.startswith("Query failed")
    assert "[value]" in message and ":name" in message and "list" in message


# -- the forms that bind keep binding


def test_a_dict_binds_a_named_placeholder(db):
    with db.transaction():
        db.command("sql", "INSERT INTO T SET k = 9, v = :v", {"v": 5})
    assert [row["v"] for row in _stored(db, 9)] == [5]


def test_a_list_binds_a_positional_placeholder(db):
    with db.transaction():
        db.command("sql", "INSERT INTO T SET k = 9, v = ?", [5])
    assert [row["v"] for row in _stored(db, 9)] == [5]


def test_a_dict_with_a_question_mark_in_a_string_literal_binds_by_name(db):
    for text in (
        "SELECT FROM T WHERE k = :k AND '?' = '?'",
        'SELECT FROM T WHERE k = :k AND "?" = "?"',
    ):
        rows = db.query("sql", text, {"k": 1}).to_list()
        assert [row["k"] for row in rows] == [1], text


def test_a_dict_with_a_digit_string_key_binds_the_positional_slot(db):
    with db.transaction():
        db.command("sql", "INSERT INTO T SET k = 9, v = ?", {"0": 7})
    assert [row["v"] for row in _stored(db, 9)] == [7]


def test_the_named_map_of_opencypher_is_not_checked(db):
    rows = db.query("opencypher", "RETURN $a AS a", {"a": 1}).to_list()
    assert [row["a"] for row in rows] == [1]


# -- the scanner


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SELECT FROM T WHERE k = ?", (True, False)),
        ("SELECT FROM T WHERE k = :k", (False, True)),
        ("SELECT FROM T WHERE k = $k", (False, True)),
        ("SELECT FROM T WHERE k = $1", (False, False)),
        ("SELECT FROM T WHERE k = ? AND j = :j", (True, True)),
        ("SELECT FROM T WHERE s = '?:x$y'", (False, False)),
        ('SELECT FROM T WHERE s = "?:x$y"', (False, False)),
        ("SELECT FROM `a?b:c` WHERE k = ?", (True, False)),
        ("SELECT FROM T WHERE s = 'it''s' AND k = ?", (True, False)),
        ("SELECT FROM T WHERE s = 'a\\'?' AND k = :k", (False, True)),
        ("SELECT FROM T WHERE s = 'unterminated ?", (False, False)),
    ],
)
def test_the_scanner_skips_literals_and_backtick_identifiers(text, expected):
    from arcadedb._params import _placeholder_kinds

    assert _placeholder_kinds(text) == expected
