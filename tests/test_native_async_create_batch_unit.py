"""The native async executor hands ``create_record`` calls to the library in batches, and every record keeps its own callbacks.

``NativeAsyncExecutor.create_record`` used to send one JSON op (``{type, kind, props, cb, err}``) per record: one crossing, one
``json.dumps`` and one JSON parse in the image per record. It now buffers the records of one type and kind and hands a batch over
as one columnar frame (``arcadedb_n_async_bin`` with the ``create_records`` op, the layout of ``arcadedb_n_insert_colp``) plus the
callback ids of each record. These tests pin what must not change: nothing is ever lost or reordered (every other call on the
executor, ``wait_completion``, ``close`` and ``kill`` hand the buffer over first), a record changed after ``create_record`` returned
is stored as it was at the call, a shape the frame cannot carry keeps the per-record op, and a record that the image refuses in the
middle of a batch reaches its own error callback while the records around it still go in, in order.

No JVM and no native library: the library call is replaced by a recorder, so the counts are crossings, not seconds. The stored
values and the callbacks of the real library are in test_native_async_create_batch.py.
"""

import json

import pytest

from arcadedb import ArcadeDBError
from arcadedb._native import bulk, columnar, extras

np = pytest.importorskip("numpy")

BATCH = getattr(extras, "_ASYNC_BATCH", 1000)
REFUSAL = {"cls": "DuplicatedKeyException", "error": "duplicate", "chain": []}


class _Doc:
    """What the executor needs of a document: its type and its properties."""

    def __init__(self, type_name, props):
        self._type = type_name
        self._props = props


class _Db:
    _h = 7

    def _discard_async_executor(self, executor):
        pass

    def query(
        self, *args, **kwargs
    ):  # the schema read of columnar.Declared: no engine here, so no declared types
        raise RuntimeError("no engine")


class _Library:
    """Stands for ``extras._call``: records every crossing in order and answers like the image."""

    def __init__(self):
        self.log = []  # (kind, op, args or header, n_spec)
        self.refusals = (
            []
        )  # replies for the next create_records calls, None for a plain success
        self.raise_on_batch = None

    def __call__(self, prefix, name, handle, op, text, *rest):
        args = json.loads(text)
        op = op.decode() if isinstance(op, bytes) else op
        if name == "arcadedb_n_async_bin":
            self.log.append(("bin", op, args, rest[1]))
            if self.raise_on_batch is not None:
                raise self.raise_on_batch
            reply = self.refusals.pop(0) if self.refusals else None
            return {"ok": True, "n": args["n"]} if reply is None else reply
        self.log.append(("op", op, args, 0))
        return {
            "ok": True,
            "parallel": 1,
            "back_pressure": 0,
            "commit_every": 1,
            "wal": True,
            "sync": "NO",
            "threads": 1,
            "processing": False,
            "done": True,
        }

    def batches(self):
        return [entry for entry in self.log if entry[0] == "bin"]

    def per_record(self):
        return [entry for entry in self.log if entry[1] == "create_record"]

    def order(self):
        return [(entry[1], entry[2].get("n")) for entry in self.log]


@pytest.fixture
def library(monkeypatch):
    fake = _Library()
    monkeypatch.setattr(extras, "_call", fake)
    monkeypatch.setattr(extras, "_ensure_upcall", lambda: None)
    monkeypatch.setattr(bulk, "_usable", lambda: True)
    return fake


@pytest.fixture
def executor(library):
    return extras.NativeAsyncExecutor(_Db())


def _size():
    with extras._callbacks_lock:
        return len(extras._callbacks)


def _rows(n, type_name="Item"):
    return [_Doc(type_name, {"id": i, "name": f"n{i}"}) for i in range(n)]


def test_records_of_one_type_cross_in_one_batch(library, executor):
    seen = []
    for doc in _rows(250):
        executor.create_record(
            doc, callback=seen.append, error_callback=lambda e: seen.append(e)
        )
    executor.wait_completion()

    assert len(library.batches()) == 1
    assert library.per_record() == []
    _, op, header, n_spec = library.batches()[0]
    assert op == "create_records"
    assert header["type"] == "Item" and header["kind"] == "d" and header["n"] == 250
    assert [c[0] for c in header["cols"]] == ["id", "name"]
    assert len(header["cbs"]) == 250 and len(header["errs"]) == 250
    assert all(header["cbs"]) and all(header["errs"])
    assert (
        len(set(header["cbs"])) == 250
    )  # one callback id per record: each record keeps its own
    assert n_spec == 6 * len(header["cols"])


def test_a_record_without_callbacks_sends_zero_ids(library, executor):
    for doc in _rows(5):
        executor.create_record(doc)
    executor.wait_completion()
    ((_, _, header, _),) = library.batches()
    assert header["cbs"] == [0] * 5 and header["errs"] == [0] * 5


def test_the_buffer_is_bounded(library, executor):
    for doc in _rows(2 * BATCH + 10):
        executor.create_record(doc)
    # the full batches went over while the records were being added, the rest waits for the next flush point
    assert [b[2]["n"] for b in library.batches()] == [BATCH] * 2
    executor.wait_completion()
    assert [b[2]["n"] for b in library.batches()] == [BATCH] * 2 + [10]


@pytest.mark.parametrize(
    "later",
    [
        lambda e: e.command("sql", "SELECT 1", lambda row: None),
        lambda e: e.wait_completion(),
        lambda e: e.is_processing(),
        lambda e: e.close(),
        lambda e: e.kill(),
        lambda e: e.get_parallel_level(),
    ],
    ids=["command", "wait_completion", "is_processing", "close", "kill", "a_getter"],
)
def test_every_other_call_hands_the_buffer_over_first(library, executor, later):
    for doc in _rows(3):
        executor.create_record(doc)
    assert library.log == []  # still buffered
    later(executor)
    ops = library.order()
    assert ops[0] == (
        "create_records",
        3,
    )  # first, so the engine queue keeps the order of the calls
    assert len(ops) >= 2


def test_records_after_another_call_form_their_own_batch(library, executor):
    for doc in _rows(3):
        executor.create_record(doc)
    executor.command("sql", "SELECT 1")
    for doc in _rows(2):
        executor.create_record(doc)
    executor.wait_completion()
    assert library.order()[:3] == [
        ("create_records", 3),
        ("command", None),
        ("create_records", 2),
    ]


def test_a_change_of_type_or_kind_hands_the_buffer_over(library, executor):
    for doc in _rows(2, "A") + _rows(2, "B"):
        executor.create_record(doc)
    vertex = _Doc("B", {"id": 9, "name": "v"})
    vertex.get_out_edges = lambda: None  # what a vertex has and a document has not
    executor.create_record(vertex)
    executor.create_record(_Doc("B", {"id": 10, "name": "x"}))
    executor.wait_completion()
    batches = [(b[2]["type"], b[2]["kind"], b[2]["n"]) for b in library.batches()]
    # a batch of one goes as the per-record op (a frame of one is not cheaper), so only the two pairs are frames
    assert batches == [("A", "d", 2), ("B", "d", 2)]
    singles = [(e[2]["type"], e[2]["kind"]) for e in library.per_record()]
    assert singles == [("B", "v"), ("B", "d")]
    order = [e[1] for e in library.log if e[1] in ("create_records", "create_record")]
    assert order == [
        "create_records",
        "create_records",
        "create_record",
        "create_record",
    ]


def test_a_batch_of_one_keeps_the_per_record_op(library, executor):
    executor.create_record(_Doc("Item", {"id": 1}), callback=lambda r: None)
    executor.wait_completion()
    assert library.batches() == []
    ((_, op, args, _),) = library.per_record()
    assert args["type"] == "Item" and args["props"] == {"id": 1} and args["cb"]


@pytest.mark.parametrize(
    "shape",
    [
        [{"a": 1}, {"b": 2}, {"a": 3}],  # different keys
        [
            {"a": [1, 2]},
            {"a": [3]},
            {"a": []},
        ],  # a nested value the frame has no column for
        [{"a": 1}, {"a": "x"}, {"a": 2}],  # a column of mixed types
        [{"a": 2**70}, {"a": 1}, {"a": 2}],  # beyond 64 bits
    ],
    ids=["keys", "lists", "mixed", "big"],
)
def test_a_shape_the_frame_cannot_carry_keeps_the_per_record_op(
    library, executor, shape
):
    for props in shape:
        executor.create_record(_Doc("Item", dict(props)))
    executor.wait_completion()
    assert library.batches() == []
    assert [
        e[2]["props"] for e in library.per_record()
    ] == shape  # every record, in the order given


def test_a_record_changed_after_the_call_is_stored_as_it_was(
    library, executor, monkeypatch
):
    captured = []
    real_pack = columnar.pack

    def spying_pack(columns, declared):
        captured.append({k: list(v) for k, v in columns.items()})
        return real_pack(columns, declared)

    monkeypatch.setattr(columnar, "pack", spying_pack)
    doc = _Doc("Item", {"id": 0, "name": "zero"})
    for i in range(4):
        doc._props["id"] = i  # the caller reuses one document object
        doc._props["name"] = f"n{i}"
        executor.create_record(doc)
    executor.wait_completion()
    assert captured == [{"id": [0, 1, 2, 3], "name": ["n0", "n1", "n2", "n3"]}]

    # a list that the caller mutates later is also copied at the call (the per-record JSON op copied it by encoding it)
    library.log.clear()
    tags = ["a"]
    doc = _Doc("Item", {"id": 1, "tags": tags})
    executor.create_record(doc)
    tags.append("b")
    executor.create_record(_Doc("Item", {"id": 2, "tags": ["c"]}))
    executor.wait_completion()
    assert [e[2]["props"]["tags"] for e in library.per_record()] == [["a"], ["c"]]


def test_a_record_the_image_refuses_in_the_middle_reaches_its_own_callback(
    library, executor
):
    baseline = _size()
    seen = []
    docs = _rows(6)
    for i, doc in enumerate(docs):
        executor.create_record(
            doc,
            callback=lambda r, i=i: seen.append(("ok", i)),
            error_callback=lambda e, i=i: seen.append(("err", i, str(e))),
        )
    # the image handed records 0 and 1 to the writers and refused record 2
    library.refusals.append({"ok": True, "n": 2, "err": REFUSAL})
    executor.wait_completion()

    assert [b[2]["n"] for b in library.batches()] == [
        6,
        3,
    ]  # the rest (3, 4, 5) went on in a second batch
    first, second = (b[2] for b in library.batches())
    assert second["cbs"] == first["cbs"][3:] and second["errs"] == first["errs"][3:]
    assert [s[:2] for s in seen] == [
        ("err", 2)
    ]  # exactly the refused record's own error callback ran, once
    assert "DuplicatedKeyException" in seen[0][2] and "duplicate" in seen[0][2]
    # the refused record's ids are given back, the handed records' ids are released by wait_completion
    assert all(
        cid not in extras._callbacks for cid in (first["cbs"][2], first["errs"][2])
    )
    assert _size() == baseline
    assert executor._cb_ids == []


def test_a_record_the_image_refuses_without_an_error_callback_raises_after_the_rest_is_handed(
    library, executor
):
    for doc in _rows(5):
        executor.create_record(doc)
    library.refusals.append({"ok": True, "n": 1, "err": REFUSAL})
    with pytest.raises(ArcadeDBError, match="DuplicatedKeyException: duplicate"):
        executor.wait_completion()
    # the records around the refused one were still handed over, in order
    assert [b[2]["n"] for b in library.batches()] == [5, 3]
    # and the executor works on: a later flush point does not raise the same error again
    executor.wait_completion()


def test_a_batch_the_library_refuses_gives_back_its_ids(library, executor):
    baseline = _size()
    library.raise_on_batch = ArcadeDBError("Async executor failed: type Nope not found")
    for doc in _rows(4, "Nope"):
        executor.create_record(
            doc, callback=lambda r: None, error_callback=lambda e: None
        )
    assert _size() > baseline  # the buffered records hold their ids
    with pytest.raises(ArcadeDBError, match="Nope"):
        executor.wait_completion()
    assert _size() == baseline
    assert executor._cb_ids == []


def test_wait_completion_releases_the_ids_of_the_batch(library, executor):
    baseline = _size()
    for doc in _rows(30):
        executor.create_record(
            doc, callback=lambda r: None, error_callback=lambda e: None
        )
    assert _size() == baseline + 60
    executor.wait_completion()
    assert _size() == baseline


def test_an_empty_executor_hands_nothing_over(library, executor):
    executor.wait_completion()
    executor.close()
    assert library.batches() == [] and library.per_record() == []
