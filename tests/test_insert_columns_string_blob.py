"""The JVM wheel sends a column of str and None as one UTF-8 blob plus int32 offsets, not as one JPype conversion per element.

`_column_to_java` used to convert every element of a string column (`convert_python_to_java`, then an `Object[]` of Java Strings built
element by element). Now a column of nothing but `str` and `None` is one `byte[]`, one `int[]` of row + 1 offsets and, when there is
a null, one `boolean[]`, wrapped in the bridge's `StringColumn`, which `DocumentBatcher.insertColumns` decodes row by row as it builds
each document. The blob crosses through `bytes` into `JArray(JByte)`, a copy; no JPype view of Python memory is relied on. A mixed
column, an empty column, a column with a lone surrogate (no UTF-8 form) and a column too large for an int offset keep the
per-element path. These tests spy on which path ran (they fail on the code before) and compare the stored table with the one the
per-element path stores for the same columns.
"""

import jpype
import pytest

np = pytest.importorskip("numpy")

from arcadedb import core  # noqa: E402

TEXTS = [
    "plain",
    "",
    "emoji \U0001f600 and \U00010000 and \U0010ffff",
    "embedded\x00nul",
    "\x00",
    "line separator ",
    "中文 한글 ퟿ é",
    "x" * 5_000,
]


def _string_column_class():
    return jpype.JClass("com.arcadedb.python.StringColumn")


def _make_type(db, name):
    db.command("sql", f"CREATE DOCUMENT TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    db.command("sql", f"CREATE PROPERTY {name}.label STRING")


def _stored(db, name):
    rows = db.query("sql", f"SELECT FROM {name} ORDER BY id").to_list()  # nosec B608
    return [(r["id"], r.get("label")) for r in rows]


@pytest.mark.parametrize(
    "values",
    [
        ["a", "bb", ""],
        ["a", None, "c"],
        ("tuple", "of", "str"),
        np.array(["x", "yy"]),
        np.array(["x", None, "z"], dtype=object),
        TEXTS,
    ],
    ids=["list", "list_with_none", "tuple", "numpy_str", "object_array", "edge_texts"],
)
def test_a_column_of_str_and_none_is_one_string_column(values):
    array, n = core._column_to_java("label", values)
    assert n == len(values)
    assert isinstance(array, _string_column_class())


@pytest.mark.parametrize(
    "values",
    [
        ["a", 1],
        ["a", b"bytes"],
        ["a", 1.5, None],
        [1, 2, 3],
        [None, None],
        ["ok", "lone \ud800 surrogate"],
        [],
    ],
    ids=[
        "str_int",
        "str_bytes",
        "str_float",
        "ints",
        "only_none",
        "lone_surrogate",
        "empty",
    ],
)
def test_any_other_column_keeps_the_per_element_path(values):
    array, n = core._column_to_java("label", values)
    assert n == len(values)
    assert not isinstance(array, _string_column_class())


def test_a_string_column_makes_no_per_element_conversion(temp_db, monkeypatch):
    _make_type(temp_db, "S")
    calls = []
    real = core.convert_python_to_java
    monkeypatch.setattr(
        core, "convert_python_to_java", lambda v: calls.append(v) or real(v)
    )
    n = 2_000
    label = [TEXTS[i % len(TEXTS)] if i % 7 else None for i in range(n)]
    temp_db.insert_columns("S", {"id": np.arange(n, dtype=np.int64), "label": label})
    assert calls == [], "a string column was converted element by element"
    assert _stored(temp_db, "S") == [(i, label[i]) for i in range(n)]


@pytest.mark.parametrize("parallel", [False, True], ids=["sync", "parallel"])
def test_the_blob_stores_the_table_the_per_element_path_stores(
    temp_db, monkeypatch, parallel
):
    n = 4_000
    label = [
        None if i % 5 == 3 else TEXTS[i % len(TEXTS)] + str(i % 3) for i in range(n)
    ]
    cols = {"id": np.arange(n, dtype=np.int64), "label": label}
    _make_type(temp_db, "Blob")
    _make_type(temp_db, "Ref")
    temp_db.insert_columns("Blob", cols, parallel=parallel)
    with monkeypatch.context() as mp:
        mp.setattr(core, "_string_column", lambda values: None)  # the path before
        array, _n = core._column_to_java("label", label)
        assert not isinstance(array, _string_column_class())
        temp_db.insert_columns("Ref", cols, parallel=parallel)
    assert _stored(temp_db, "Blob") == _stored(temp_db, "Ref")
    assert _stored(temp_db, "Blob") == [(i, label[i]) for i in range(n)]


def test_a_lone_surrogate_column_is_handled_as_before(temp_db, monkeypatch):
    # no UTF-8 form: the column takes the per-element path whatever that path does with it, so the outcome equals the old one
    values = ["ok", "lone \ud800 surrogate", "fine"]
    cols = {"id": np.arange(3, dtype=np.int64), "label": values}
    _make_type(temp_db, "Blob")
    _make_type(temp_db, "Ref")
    outcomes = []
    for name, force_old in (("Blob", False), ("Ref", True)):
        with monkeypatch.context() as mp:
            if force_old:
                mp.setattr(core, "_string_column", lambda values: None)
            try:
                temp_db.insert_columns(name, cols)
                outcomes.append(("stored", _stored(temp_db, name)))
            except (
                Exception
            ) as e:  # noqa: BLE001 - the type and message are what is compared
                outcomes.append((type(e).__name__, str(e).replace(name, "T")))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize(
    "blob, offsets, nulls",
    [
        (b"ab", [1, 2], None),
        (b"ab", [0, 2, 1], None),
        (b"ab", [0, 3], None),
        (b"ab", [0, 1, 2], [False]),
        (b"ab", [], None),
    ],
    ids=[
        "first_not_zero",
        "decreasing",
        "past_the_end",
        "short_null_flags",
        "no_offsets",
    ],
)
def test_the_java_side_refuses_a_malformed_column(blob, offsets, nulls):
    with pytest.raises(Exception):  # noqa: B017 - a Java IllegalArgumentException
        _string_column_class()(
            jpype.JArray(jpype.JByte)(blob),
            jpype.JArray(jpype.JInt)(offsets),
            None if nulls is None else jpype.JArray(jpype.JBoolean)(nulls),
        )


def test_a_column_beyond_an_int_offset_keeps_the_per_element_path(monkeypatch):
    monkeypatch.setattr(core, "_JAVA_INT_MAX", 10)
    array, n = core._column_to_java("label", ["0123456", "0123456"])
    assert n == 2
    assert not isinstance(array, _string_column_class())
