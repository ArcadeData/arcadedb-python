"""``create_record`` on the native async executor stores what the per-record op stored, and every record keeps its own callbacks.

Records of one type go to the library as one columnar frame (``arcadedb_n_async_bin``, op ``create_records``) with the callback ids of
each record. Here the real library runs: the stored values of a batch are compared with the ones the per-record JSON op stores (a batch
of one, so one ``create_record`` op per record), the callbacks are the records' own and in the order of the calls, and a record that an
async writer refuses in the middle of a batch reaches its own error callback while the others are stored. The counts are crossings.

Needs the native library, so conftest.py does not collect this file on the JPype backend; it runs in the native parity run. The
Python side with a recorded library is in test_native_async_create_batch_unit.py.
"""

import datetime
import decimal

import pytest

from arcadedb._native import lib

np = pytest.importorskip("numpy")


class _Crossings:
    """Records every library call whose entry point is the async executor's: ``(entry point, op)`` in order."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = lib.call

        def counting(name, *args):
            if name in ("arcadedb_n_async", "arcadedb_n_async_bin"):
                self.calls.append(
                    (name, args[1].decode() if isinstance(args[1], bytes) else args[1])
                )
            return real(name, *args)

        monkeypatch.setattr(lib, "call", counting)

    def ops(self, op):
        return [c for c in self.calls if c[1] == op]


class _Stub:
    """A record of a type that does not exist (the database refuses ``new_document`` for it, the executor is asked all the same)."""

    def __init__(self, type_name, props):
        self._type = type_name
        self._props = props


def _executor(db):
    executor = db.async_executor()
    executor.set_parallel_level(
        1
    )  # one writer: the callbacks run in the order of the calls
    executor.set_commit_every(
        1
    )  # one record per transaction: a refused record takes no other with it
    return executor


def _store(db, type_name, rows, one_by_one):
    """Create the rows through the executor. ``one_by_one``: wait after each (a batch of one: the per-record op)."""
    executor = _executor(db)
    for row in rows:
        executor.create_record(_document(db, type_name, row))
        if one_by_one:
            executor.wait_completion()
    executor.wait_completion()
    executor.close()


def _document(db, type_name, props):
    doc = db.new_document(type_name)
    for key, value in props.items():
        doc.set(key, value)
    return doc


def _dump(db, type_name):
    rows = db.query(
        "sql", f"SELECT FROM {type_name} ORDER BY i"
    ).to_list()  # nosec B608 - a literal type name of this test
    rows = [
        {k: v for k, v in row.items() if not k.startswith("@")} for row in rows
    ]  # the rid and the type name differ by design
    return repr(
        rows
    )  # repr: a NaN equals a NaN, a None differs from a missing key, key order counts


_N = 12

SHAPES = {
    "ints": lambda i: {"i": i, "v": i * 7},
    "big_ints": lambda i: {"i": i, "v": 2**62 + i},
    "floats": lambda i: {"i": i, "v": i / 3},
    "nan_inf": lambda i: {
        "i": i,
        "v": [float("nan"), float("inf"), -float("inf")][i % 3],
    },
    "strings": lambda i: {"i": i, "v": f"s{i}é\U0001f600"},
    "empty_strings": lambda i: {"i": i, "v": ""},
    "bools": lambda i: {"i": i, "v": bool(i % 2)},
    "bytes": lambda i: {"i": i, "v": bytes([i, 0, 255])},
    "datetimes": lambda i: {"i": i, "v": datetime.datetime(2020, 1, 1, 0, 0, i, 1)},
    "dates": lambda i: {"i": i, "v": datetime.date(2020, 1, 1 + i)},
    "decimals": lambda i: {"i": i, "v": decimal.Decimal(i) / decimal.Decimal(7)},
    "nulls": lambda i: {"i": i, "v": None},
    "some_nulls": lambda i: {"i": i, "v": None if i % 2 else i},
    "numpy_scalars": lambda i: {"i": i, "v": np.int64(i), "w": np.float64(i) / 3},
    "nested": lambda i: {"i": i, "v": [i, [i, {"k": i}]], "m": {"a": [i]}},
    "key_order": lambda i: {"z": 1, "i": i, "a": 2},
    "keys_differ": lambda i: {"i": i, "v": i} if i % 2 else {"i": i},
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_a_batch_stores_what_the_per_record_op_stores(temp_db, monkeypatch, shape):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Old")
    temp_db.command("sql", "CREATE DOCUMENT TYPE New")
    rows = [SHAPES[shape](i) for i in range(_N)]
    crossings = _Crossings(monkeypatch)
    _store(temp_db, "Old", rows, one_by_one=True)
    old = crossings.calls[:]
    crossings.calls.clear()
    _store(temp_db, "New", rows, one_by_one=False)

    assert _dump(temp_db, "New") == _dump(temp_db, "Old")
    assert (
        len([c for c in old if c[1] == "create_record"]) == _N
    )  # the control: one op per record
    batched = crossings.ops("create_records")
    if shape in (
        "nested",
        "keys_differ",
    ):  # no frame for these: the per-record op, as before
        assert batched == [] and len(crossings.ops("create_record")) == _N
    else:
        assert len(batched) == 1 and crossings.ops("create_record") == []


def test_empty_and_single_batches(temp_db, monkeypatch):
    temp_db.command("sql", "CREATE DOCUMENT TYPE One")
    crossings = _Crossings(monkeypatch)
    executor = _executor(temp_db)
    executor.wait_completion()  # nothing was created: nothing is handed over
    assert (
        crossings.ops("create_records") == [] and crossings.ops("create_record") == []
    )
    executor.create_record(_document(temp_db, "One", {"i": 1, "v": "x"}))
    executor.wait_completion()
    executor.close()
    assert (
        crossings.ops("create_records") == []
        and len(crossings.ops("create_record")) == 1
    )
    assert temp_db.query("sql", "SELECT FROM One").to_list()[0]["v"] == "x"


def test_every_record_gets_its_own_callback_in_the_order_of_the_calls(
    temp_db, monkeypatch
):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Item")
    temp_db.command("sql", "CREATE VERTEX TYPE Node")
    crossings = _Crossings(monkeypatch)
    executor = _executor(temp_db)
    seen = []
    for type_name, make in (
        ("Item", temp_db.new_document),
        ("Node", temp_db.new_vertex),
    ):
        for i in range(50):
            doc = make(type_name)
            doc.set("i", i)
            executor.create_record(
                doc,
                callback=lambda record, i=i, t=type_name: seen.append(
                    (t, i, record.get("i"), record.get_rid())
                ),
                error_callback=lambda e, i=i: seen.append(("error", i, str(e))),
            )
    executor.wait_completion()
    executor.close()

    assert len(crossings.ops("create_records")) == 2  # one frame per type, not 100 ops
    assert [(t, i, v) for t, i, v, _ in seen] == [
        (t, i, i) for t in ("Item", "Node") for i in range(50)
    ]
    assert all(
        str(rid).startswith("#") for _, _, _, rid in seen
    )  # the stored record: it has its RID
    assert temp_db.count_type("Item") == 50 and temp_db.count_type("Node") == 50


def test_a_record_the_writer_refuses_in_the_middle_reaches_its_own_error_callback(
    temp_db, monkeypatch
):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Uniq")
    temp_db.command("sql", "CREATE PROPERTY Uniq.k INTEGER")
    temp_db.command("sql", "CREATE INDEX ON Uniq (k) UNIQUE")
    keys = [
        1,
        2,
        3,
        4,
        2,
        5,
        6,
        3,
        7,
    ]  # records 4 and 7 repeat an earlier key: the writer refuses them
    results = {}
    for batched in (False, True):
        temp_db.command("sql", "DELETE FROM Uniq")
        seen = {"ok": [], "err": []}
        executor = _executor(temp_db)
        for index, key in enumerate(keys):
            executor.create_record(
                _document(temp_db, "Uniq", {"k": key, "n": index}),
                callback=lambda record, index=index: seen["ok"].append(index),
                error_callback=lambda e, index=index: seen["err"].append(
                    (index, type(e).__name__)
                ),
            )
            if not batched:
                executor.wait_completion()
        executor.wait_completion()
        executor.close()
        stored = sorted(
            (r["k"], r["n"]) for r in temp_db.query("sql", "SELECT FROM Uniq").to_list()
        )
        results[batched] = (sorted(seen["ok"]), sorted(seen["err"]), stored)
    assert results[True] == results[False]
    ok, err, stored = results[True]
    assert [index for index, _ in err] == [
        4,
        7,
    ]  # exactly the refused records, each on its own callback
    assert ok == [0, 1, 2, 3, 5, 6, 8] and len(stored) == 7


def test_the_executor_keeps_working_after_a_refused_batch(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Item")
    executor = _executor(temp_db)
    errors = []
    for i in range(5):
        executor.create_record(
            _Stub("NoSuchType", {"i": i}), error_callback=errors.append
        )
    with pytest.raises(Exception, match="NoSuchType"):
        executor.wait_completion()  # the batch of a type that does not exist is refused before any record is handed over
    for i in range(5):
        executor.create_record(_document(temp_db, "Item", {"i": i}))
    executor.wait_completion()
    executor.close()
    assert temp_db.count_type("Item") == 5
