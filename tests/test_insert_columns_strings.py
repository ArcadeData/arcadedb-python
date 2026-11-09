"""insert_columns stores a string column exactly, on both wheels, whatever way it crosses.

The JVM wheel sends a column of nothing but str and None as one UTF-8 blob plus int32 offsets (test_insert_columns_string_blob.py);
the native wheel reads the column in place. A user must not see a difference: a character beyond the BMP, a NUL, a line separator,
an empty string and a null are stored as they were given, in one column and in a mixed column, in the synchronous and the parallel
mode, from a list, a numpy string array and an object array.
"""

import pytest

np = pytest.importorskip("numpy")

TEXTS = [
    "plain",
    "",
    "emoji \U0001f600 and \U00010000 and \U0010ffff",
    "embedded\x00nul",
    "\x00",
    "line separator ",
    "中文 한글 ퟿ é",
    "x" * 5_000,
    'tab\tnewline\nquote"backslash\\',
]


def _make_type(db, name):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    db.command("sql", f"CREATE PROPERTY {name}.label STRING")
    db.command("sql", f"CREATE PROPERTY {name}.other STRING")


def _stored(db, name):
    rows = db.query("sql", f"SELECT FROM {name} ORDER BY id").to_list()  # nosec B608
    return [(r["id"], r.get("label"), r.get("other")) for r in rows]


def _columns(n):
    label = [TEXTS[i % len(TEXTS)] for i in range(n)]
    other = [None if i % 4 == 1 else f"o{i}" for i in range(n)]
    return {"id": np.arange(n, dtype=np.int64), "label": label, "other": other}


@pytest.mark.parametrize("parallel", [False, True], ids=["sync", "parallel"])
def test_strings_are_stored_as_given(temp_db, parallel):
    _make_type(temp_db, "S")
    cols = _columns(3_000)
    assert temp_db.insert_columns("S", cols, parallel=parallel) == 3_000
    assert _stored(temp_db, "S") == [
        (i, cols["label"][i], cols["other"][i]) for i in range(3_000)
    ]


@pytest.mark.parametrize("kind", ["numpy_str", "object_array"])
def test_other_containers_of_strings_are_stored_as_given(temp_db, kind):
    _make_type(temp_db, "S")
    # a numpy "U" array drops a trailing NUL by itself, so its texts have none
    texts = [t for t in TEXTS if "\x00" not in t]
    label = [texts[i % len(texts)] for i in range(40)]
    column = np.array(label) if kind == "numpy_str" else np.array(label, dtype=object)
    expected = label
    temp_db.insert_columns("S", {"id": np.arange(40, dtype=np.int64), "label": column})
    assert [row[1] for row in _stored(temp_db, "S")] == expected


def test_a_mixed_column_keeps_every_value(temp_db):
    _make_type(temp_db, "S")
    temp_db.command("sql", "CREATE PROPERTY S.n LONG")
    labels = ["a", "emoji \U0001f600", None, "", "z"]
    temp_db.insert_columns(
        "S",
        {
            "id": np.arange(5, dtype=np.int64),
            "label": labels,
            "n": [1, None, 3, 4, 5],
        },
    )
    assert [row[1] for row in _stored(temp_db, "S")] == labels
    assert _stored(temp_db, "S")[0][0] == 0


def test_a_single_row_column(temp_db):
    _make_type(temp_db, "S")
    temp_db.insert_columns(
        "S", {"id": np.array([10], dtype=np.int64), "label": ["one"]}
    )
    assert _stored(temp_db, "S") == [(10, "one", None)]
