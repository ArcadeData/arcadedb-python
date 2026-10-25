"""Vectors across the native boundary on every batch of a result: arcadedb_n_exec returns the first 256 rows, arcadedb_n_next (and
arcadedb_n_next_bin, which carries float arrays of 8 or more elements in a float32 tail instead of as number text) the following
2048 at a time. Every vector has to come back bit for bit as it went in, on the first batch and on the later ones, whichever way the
native library carries it. The tests use the public API only, so they also pin the result of the number-text path
(ARCADEDB_NATIVE_NEXT_BIN=0).
"""

import struct


def f32(x):
    """The double a float32 value widens to (what the engine returns for an ARRAY_OF_FLOATS element)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _typed_db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE R")
    for p, t in (("id", "INTEGER"), ("vec", "ARRAY_OF_FLOATS")):
        temp_db.command("sql", f"CREATE PROPERTY R.{p} {t}")
    return temp_db


SPECIAL = [
    float("inf"),
    float("-inf"),
    -0.0,
    0.0,
    1e-45,
    3.4028235e38,
    0.1,
    -123456.789,
]  # 8 values: a float32 tail on the binary path


def _bits(x):
    """The 8 bytes of a double as hex (any NaN is 'nan'): bit-exact comparison, so -0.0 differs from 0.0."""
    return "nan" if x != x else struct.pack("<d", x).hex()


def _vector_rows(n):
    rows, want = [], {}
    for i in range(n):
        kind = i % 6
        if kind == 0:
            vec = [f32(((i * 7 + k) % 1000) / 37.0 - 5.0) for k in range(32)]
        elif kind == 1:
            vec = [f32(x) for x in SPECIAL]
        elif kind == 2:
            vec = []
        elif kind == 3:
            vec = [
                f32(0.1),
                f32(-2.5),
                f32(1e-45),
            ]  # fewer than 8 floats: number text on every path
        elif kind == 4:
            vec = [
                f32((i + k) / 11.0) for k in range(8)
            ]  # exactly 8: the smallest vector that travels as a float32 tail
        else:
            vec = None  # no vec property at all
        row = {"id": i}
        if vec is not None:
            row["vec"] = vec
        rows.append(row)
        want[i] = vec
    return rows, want


def test_vectors_are_bit_exact_on_the_first_and_on_later_batches(temp_db):
    db = _typed_db(temp_db)
    rows, want = _vector_rows(3000)  # 256 rows in the first reply, then 2048 + 696
    db.insert_many("R", rows)
    got = db.query("sql", "SELECT id, vec FROM R ORDER BY id").to_list()
    assert [r["id"] for r in got] == list(range(3000))
    for r in got:
        expect = want[r["id"]]
        if expect is None:
            assert r.get("vec") is None, r["id"]
        else:
            assert [_bits(x) for x in r["vec"]] == [_bits(x) for x in expect], r["id"]


def test_vectors_in_a_cursor_read_row_by_row_are_bit_exact(temp_db):
    db = _typed_db(temp_db)
    rows, want = _vector_rows(700)
    db.insert_many("R", rows)
    seen = 0
    for r in db.query("sql", "SELECT id, vec FROM R ORDER BY id"):
        expect = want[r.get("id")]
        vec = r.get("vec")
        assert (
            (vec is None)
            if expect is None
            else [_bits(x) for x in vec] == [_bits(x) for x in expect]
        )
        seen += 1
    assert seen == 700
