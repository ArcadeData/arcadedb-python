"""
Tests for NumPy support in ArcadeDB Python bindings.
"""

import pytest

import arcadedb

np = pytest.importorskip("numpy")


def test_numpy_array_conversion_in_command(temp_db):
    """Test automatic conversion of NumPy arrays in db.command()."""
    db = temp_db

    db.command("sql", "CREATE VERTEX TYPE VectorData")
    db.command("sql", "CREATE PROPERTY VectorData.vector ARRAY_OF_FLOATS")

    # Create a NumPy array
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Insert using command with NumPy array
    with db.transaction():
        db.command("sql", "INSERT INTO VectorData SET vector = ?", vec)

    # Verify
    result = db.query("sql", "SELECT FROM VectorData").first()
    stored_vec = result.get("vector")

    # Check values (approximate for float)
    assert len(stored_vec) == 3
    assert abs(stored_vec[0] - 0.1) < 0.0001
    assert abs(stored_vec[1] - 0.2) < 0.0001
    assert abs(stored_vec[2] - 0.3) < 0.0001


def test_numpy_array_conversion_in_query(temp_db):
    """Test automatic conversion of NumPy arrays in db.query()."""
    db = temp_db

    db.command("sql", "CREATE VERTEX TYPE VectorData")
    db.command("sql", "CREATE PROPERTY VectorData.vector ARRAY_OF_FLOATS")

    # Insert data manually first. The vector goes in as ONE array parameter:
    # a bare Python list as the only argument is the positional-parameter
    # array itself, so it would bind vector = 0.1.
    with db.transaction():
        db.command(
            "sql",
            "INSERT INTO VectorData SET vector = ?",
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
    stored = db.query("sql", "SELECT vector FROM VectorData").first().get("vector")
    assert np.allclose(stored, [0.1, 0.2, 0.3])

    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Ensure the call succeeds with NumPy array as parameter
    try:
        db.query("sql", "SELECT FROM VectorData WHERE vector = ?", vec)
    except Exception as e:
        pytest.fail(f"Query with NumPy array failed: {e}")


def test_numpy_array_conversion_in_transaction(temp_db):
    """Test NumPy array conversion in regular transactions (no batch context)."""
    db = temp_db

    db.command("sql", "CREATE VERTEX TYPE VectorData")
    db.command("sql", "CREATE PROPERTY VectorData.vector ARRAY_OF_FLOATS")
    db.command("sql", "CREATE DOCUMENT TYPE DocData")
    db.command("sql", "CREATE PROPERTY DocData.embedding ARRAY_OF_FLOATS")

    vec1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vec2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    vec1_java = arcadedb.to_java_float_array(vec1)
    vec2_java = arcadedb.to_java_float_array(vec2)

    with db.transaction():
        db.command("sql", "INSERT INTO VectorData SET vector = ?", vec1_java)
        db.command("sql", "INSERT INTO DocData SET embedding = ?", vec2_java)

    # Verify Vertex
    result = db.query("sql", "SELECT FROM VectorData").first()
    stored_vec = result.get("vector")
    assert len(stored_vec) == 3
    assert abs(stored_vec[0] - 0.1) < 0.0001

    # Verify Document
    result = db.query("sql", "SELECT FROM DocData").first()
    stored_emb = result.get("embedding")
    assert len(stored_emb) == 3
    assert abs(stored_emb[0] - 0.4) < 0.0001


def _param(db, value):
    """The one parameter `value` as the engine hands it back."""
    return db.query("sql", "SELECT ? AS v", [value]).first().get("v")


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8])
def test_integer_ndarray_param_binds_as_the_list_does(temp_db, dtype):
    """A positional integer array keeps its integers; the JVM build once sent every ndarray as a float[]."""
    values = [1, 2, 3, 100]
    bound = _param(temp_db, np.array(values, dtype=dtype))
    assert list(bound) == values
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in bound)
    assert list(bound) == list(_param(temp_db, values))


def test_int64_ndarray_param_is_exact(temp_db):
    """2**53 + 1 does not survive a float; as a long it does."""
    values = [2**53 + 1, -(2**40), 7]
    bound = _param(temp_db, np.array(values, dtype=np.int64))
    assert list(bound) == values


def test_bool_ndarray_param_stays_boolean(temp_db):
    bound = _param(temp_db, np.array([True, False, True]))
    assert list(bound) == [True, False, True]
    assert all(isinstance(x, bool) for x in bound)


def test_int_ndarray_stored_in_array_of_integers_and_longs(temp_db):
    """int32 and int64 arrays bound as parameters land in ARRAY_OF_INTEGERS / ARRAY_OF_LONGS and read back as ints."""
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE IntArrays")
    db.command("sql", "CREATE PROPERTY IntArrays.small ARRAY_OF_INTEGERS")
    db.command("sql", "CREATE PROPERTY IntArrays.big ARRAY_OF_LONGS")
    small = np.array([1, -2, 3], dtype=np.int32)
    big = np.array([2**53 + 1, -(2**40), 5], dtype=np.int64)
    with db.transaction():
        db.command("sql", "INSERT INTO IntArrays SET small = ?, big = ?", small, big)
    row = db.query("sql", "SELECT small, big FROM IntArrays").first()
    assert list(row.get("small")) == [1, -2, 3]
    assert list(row.get("big")) == [2**53 + 1, -(2**40), 5]
    assert all(isinstance(x, int) for x in row.get("small"))
    assert all(isinstance(x, int) for x in row.get("big"))


def test_int_ndarray_as_named_param_and_document_property(temp_db):
    """The named-parameter and document-property paths already keep the integers; pin them beside the positional one."""
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE IntDoc")
    db.command("sql", "CREATE PROPERTY IntDoc.small ARRAY_OF_INTEGERS")
    db.command("sql", "CREATE PROPERTY IntDoc.big ARRAY_OF_LONGS")
    small = np.array([4, 5, 6], dtype=np.int32)
    big = np.array([2**53 + 1, 8], dtype=np.int64)
    with db.transaction():
        db.command(
            "sql",
            "INSERT INTO IntDoc SET small = :s, big = :b",
            {"s": small, "b": big},
        )
        doc = db.new_document("IntDoc")
        doc.set("small", small).set("big", big).save()
    for row in db.query("sql", "SELECT small, big FROM IntDoc"):
        assert list(row.get("small")) == [4, 5, 6]
        assert list(row.get("big")) == [2**53 + 1, 8]


# A float64 array is a value, not a query vector, on every path but a positional ``?``: it keeps its doubles. The native build sent
# it through the float32 side buffer on the named, CONTENT and record paths, so 0.1 was stored as 0.10000000149011612.


def _write_named(db, t, arr):
    with db.transaction():
        sql = f"INSERT INTO {t} SET v = :v"  # nosec B608 - test-owned
        db.command("sql", sql, {"v": arr})


def _write_content_named(db, t, arr):
    with db.transaction():
        sql = f"INSERT INTO {t} CONTENT :row"  # nosec B608 - test-owned
        db.command("sql", sql, {"row": {"v": arr}})


def _write_record(db, t, arr):
    with db.transaction():
        db.new_document(t).set("v", arr).save()


def _stored(db, t):
    """The one stored value of ``v`` read two ways: the result row and a lookup of the record by its rid."""
    sql = f"SELECT @rid AS rid, v FROM {t}"  # nosec B608 - test-owned
    row = db.query("sql", sql).first()
    by_rid = db.lookup_by_rid(str(row.get("rid"))).get("v")
    return list(row.get("v")), list(by_rid)


@pytest.mark.parametrize("write", [_write_named, _write_content_named, _write_record])
def test_float64_array_keeps_its_doubles(temp_db, write):
    temp_db.command("sql", "CREATE DOCUMENT TYPE F64")
    write(temp_db, "F64", np.array([0.1, 0.2, 0.3], dtype=np.float64))
    from_row, from_rid = _stored(temp_db, "F64")
    assert from_row == [0.1, 0.2, 0.3]
    assert from_rid == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("write", [_write_named, _write_content_named, _write_record])
def test_float32_array_reads_back_as_its_float32_values(temp_db, write):
    temp_db.command("sql", "CREATE DOCUMENT TYPE F32")
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    write(temp_db, "F32", arr)
    expected = arr.astype(
        np.float64
    ).tolist()  # float32 widened: 0.10000000149011612, ...
    from_row, from_rid = _stored(temp_db, "F32")
    assert from_row == expected
    assert from_rid == expected


def test_float64_array_as_a_lone_positional_parameter_is_a_query_vector(temp_db):
    """The documented rule (databases-and-queries, "A query vector"): a float array among the positional parameters is sent as
    float32 on both builds. Pinned beside the named case so a change of either side is a decision.
    """
    temp_db.command("sql", "CREATE DOCUMENT TYPE F64Pos")
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO F64Pos SET v = ?", arr)
    from_row, _ = _stored(temp_db, "F64Pos")
    assert from_row == arr.astype(np.float32).astype(np.float64).tolist()
