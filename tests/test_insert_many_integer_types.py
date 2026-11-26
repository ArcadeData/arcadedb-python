"""insert_many stores a whole number as an Integer when it fits an int and as a Long beyond, on both builds (#445).

That is how the engine's own JSON reader types a number. The native build used to store every whole number of an insert_many as a
Long, so the same Python rows had other stored types on the two packages, and a JSONL export of the native database was longer. A row
that JSON text cannot carry (a NaN, say) is the exception on both builds: it keeps its whole numbers as Long. ``insert_columns`` is not
``insert_many`` and stores an int64 column as Long on both builds.

The stored type is read with the SQL method ``type()``. Nothing here depends on which build runs it.
"""

import pytest

INT_MIN, INT_MAX = -(2**31), 2**31 - 1


def _type_of(db, type_name, expressions):
    """For every record, in the order of ``id``: the stored type of each expression."""
    projection = ", ".join(f"{e}.type() AS t{n}" for n, e in enumerate(expressions))
    sql = f"SELECT id, {projection} FROM {type_name} ORDER BY id"  # nosec B608 - test-owned names
    return [
        [row.get(f"t{n}") for n in range(len(expressions))]
        for row in db.query("sql", sql).to_list()
    ]


@pytest.mark.parametrize("parallel", [False, True], ids=["transaction", "parallel"])
@pytest.mark.parametrize("count", [3, 200], ids=["few-rows", "many-rows"])
def test_whole_numbers_of_a_flat_row(temp_db, count, parallel):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Flat")
    rows = [
        {
            "id": i,
            "lowest": INT_MIN,
            "highest": INT_MAX,
            "above": INT_MAX + 1,
            "below": INT_MIN - 1,
            "gaps": None if i % 3 == 0 else i,
            "half": i / 2,
        }
        for i in range(count)
    ]
    assert temp_db.insert_many("Flat", rows, parallel=parallel) == count

    names = ["id", "lowest", "highest", "above", "below", "gaps", "half"]
    stored = _type_of(temp_db, "Flat", names)
    assert len(stored) == count
    for i, types in enumerate(stored):
        gaps = None if i % 3 == 0 else "INTEGER"
        assert types == [
            "INTEGER",
            "INTEGER",
            "INTEGER",
            "LONG",
            "LONG",
            gaps,
            "DOUBLE",
        ]


@pytest.mark.parametrize("parallel", [False, True], ids=["transaction", "parallel"])
def test_whole_numbers_inside_a_list_and_a_map(temp_db, parallel):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Nested")
    rows = [
        {"id": i, "lst": [1, 2**40], "obj": {"one": 1, "deep": [2, 2**40]}}
        for i in range(3)
    ]
    assert temp_db.insert_many("Nested", rows, parallel=parallel) == 3

    stored = _type_of(
        temp_db,
        "Nested",
        ["lst[0]", "lst[1]", "obj.one", "obj.deep[0]", "obj.deep[1]"],
    )
    assert stored == [["INTEGER", "LONG", "INTEGER", "INTEGER", "LONG"]] * 3
    read = temp_db.query("sql", "SELECT lst, obj FROM Nested WHERE id = 0").to_list()
    assert read == [{"lst": [1, 2**40], "obj": {"one": 1, "deep": [2, 2**40]}}]


@pytest.mark.parametrize("parallel", [False, True], ids=["transaction", "parallel"])
def test_a_row_json_cannot_carry_keeps_long(temp_db, parallel):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Odd")
    rows = [
        {"id": 0, "n": 5, "x": float("nan")},
        {"id": 1, "n": 5, "x": 1.0},
        {"id": 2, "n": 5, "x": float("inf")},
    ]
    assert temp_db.insert_many("Odd", rows, parallel=parallel) == 3

    assert _type_of(temp_db, "Odd", ["id", "n"]) == [
        ["LONG", "LONG"],
        ["INTEGER", "INTEGER"],
        ["LONG", "LONG"],
    ]


def test_a_declared_type_and_insert_columns_are_not_narrowed(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Declared")
    temp_db.command("sql", "CREATE PROPERTY Declared.wide LONG")
    temp_db.insert_many("Declared", [{"id": i, "wide": i} for i in range(100)])
    assert _type_of(temp_db, "Declared", ["wide"]) == [["LONG"]] * 100

    temp_db.command("sql", "CREATE DOCUMENT TYPE Columns")
    temp_db.insert_columns("Columns", {"id": [0, 1], "n": [5, 6]})
    assert _type_of(temp_db, "Columns", ["id", "n"]) == [["LONG", "LONG"]] * 2
