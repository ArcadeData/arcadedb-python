"""The Python half of the binary approximate search of the native wheel: which queries cross by address, what the library is handed, and
how its answer is read (``arcadedb_n_vindex_bin``, ``find_nearest_approximate``).

Before, the query vector crossed as a JSON list of d numbers (``arcadedb_n_vindex``, op ``approximate``) and the hits came back as a JSON
list of ``[rid, score]``. Now a query that float32 carries goes by address (a float32 array and its count), a filter goes as int64
(bucket, position) pairs, and the library writes the hits into a buffer of int64 words that Python hands it: three words per hit, the
bucket, the position and the score as the bits of a double. Anything the entry cannot carry keeps the JSON route.

Needs no library: ``lib.vindex_bin`` is replaced by a stand-in that reads the buffers by their addresses, as the library does, so the
tests see exactly the floats and the pairs the engine would be given. The real library, and both routes against each other, are in
``test_native_vindex_bin.py`` (native backend only).
"""

import ctypes
import json
import struct

import numpy as np
import pytest

from arcadedb._native import lib, schema_objects
from arcadedb._native.schema_objects import NativeVectorIndex, _IndexShim
from arcadedb.exceptions import ArcadeDBError


def _f32(x):
    """The float32 that a C cast of the double `x` gives (what Java's floatValue() gives), by way of struct, not numpy."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


class _Db:
    _h = 11

    def __init__(self):
        self.lookups = []

    def lookup_by_rid(self, rid):
        self.lookups.append(rid)
        return rid


class _Library:
    """What the library does with the arguments of ``arcadedb_n_vindex_bin``, and what ``arcadedb_n_vindex`` gets as JSON text."""

    def __init__(self, hits, growth=0):
        self.hits = hits  # (bucket, position, score) in the order the engine finds them
        self.growth = growth  # extra hits per call: an index that outgrows every buffer
        self.bin_calls, self.json_calls = [], []

    def vindex_bin(
        self, handle, type_name, prop, k, query, dims, allowed, pairs, out, cap
    ):
        floats = list((ctypes.c_float * dims).from_address(query)) if dims else []
        pair_words = (
            (ctypes.c_int64 * (2 * pairs)).from_address(allowed) if pairs else ()
        )
        self.bin_calls.append(
            {
                "handle": handle,
                "type": type_name,
                "prop": prop,
                "k": k,
                "query": floats,
                "dims": dims,
                "allowed": [
                    (pair_words[2 * i], pair_words[2 * i + 1]) for i in range(pairs)
                ],
                "allowed_address": allowed,
                "cap": cap,
            }
        )
        hits = self.hits + [
            (9, 1000 + i, 0.5) for i in range(self.growth * len(self.bin_calls))
        ]
        if len(hits) > cap:
            return len(hits)
        words = (ctypes.c_int64 * (3 * len(hits))).from_address(out)
        for i, (bucket, position, score) in enumerate(hits):
            words[3 * i] = bucket
            words[3 * i + 1] = position
            words[3 * i + 2] = struct.unpack("<q", struct.pack("<d", score))[0]
        return len(hits)

    def call(self, name, *args):
        assert name == "arcadedb_n_vindex"
        self.json_calls.append((args[1], json.loads(args[2])))
        return json.dumps(
            {"ok": True, "hits": [[f"#{b}:{p}", s] for b, p, s in self.hits]}
        )


@pytest.fixture
def search(monkeypatch):
    """``search(query, k, allowed)`` through ``NativeVectorIndex._run_approximate`` with a stand-in library; returns (hits, library, db)."""
    hits = [(2, 7, 0.25), (1, 3, 0.125), (3, 9, 0.5), (0, 1, 0.1), (1, 4, 0.125)]
    library = _Library(hits)
    database = _Db()
    monkeypatch.setattr(lib, "VINDEX_BIN", True)
    monkeypatch.setattr(lib, "vindex_bin", library.vindex_bin)
    monkeypatch.setattr(lib, "call", library.call)
    monkeypatch.setattr(
        lib,
        "last_error",
        lambda: json.dumps(
            {
                "ok": False,
                "error": "boom",
                "cls": "java.lang.IllegalArgumentException",
                "chain": [],
            }
        ),
    )
    index = NativeVectorIndex.__new__(NativeVectorIndex)
    index._database = database
    index._shim = _IndexShim(database, "Doc", "emb", "Doc[emb]")

    def run(query, k=3, allowed=None):
        return index._run_approximate(query, k, allowed)

    run.library, run.database, run.index = library, database, index
    return run


# ---------------------------------------------------------------------------------------------------- what is read from the address

# doubles that float32 rounds, in the cases that decide a rounding: the exact ties (to even), the subnormal range, an underflow to zero
TRICKY = [
    1 + 2.0**-24,  # a tie between 1.0 and 1 + 2**-23: even is 1.0
    1 + 3 * 2.0**-24,  # a tie between 1 + 2**-23 and 1 + 2**-22: even is 1 + 2**-22
    16777217.0,  # 2**24 + 1: a tie, even is 2**24
    16777219.0,  # 2**24 + 3: a tie, even is 2**24 + 4
    1e-40,  # a float32 subnormal
    1e-46,  # rounds to 0.0
    0.1,
    -0.1,
    3.4028235e38,  # the largest float32
    -0.0,
]


def _forms():
    rng = np.random.default_rng(3)
    wide = rng.standard_normal(40)  # float64, float32 rounds every one of them
    exact = rng.standard_normal(40).astype(np.float32)
    mixed = [1, 2.5, -3, 0, 7.25, 16777217, 2**53, -(2**53)]
    return {
        "float64 array": (wide, [_f32(x) for x in wide.tolist()]),
        "float32 array": (exact, exact.tolist()),
        "float16 array": (
            wide.astype(np.float16),
            wide.astype(np.float16).astype(np.float64).tolist(),
        ),
        "float64 list": (wide.tolist(), [_f32(x) for x in wide.tolist()]),
        "float64 tuple": (tuple(wide.tolist()), [_f32(x) for x in wide.tolist()]),
        "tricky list": (TRICKY, [_f32(x) for x in TRICKY]),
        "tricky array": (np.array(TRICKY), [_f32(x) for x in TRICKY]),
        "ints and floats": (mixed, [_f32(float(x)) for x in mixed]),
        "non-contiguous float32": (
            np.arange(30, dtype=np.float32)[::3],
            np.arange(30, dtype=np.float32)[::3].tolist(),
        ),
    }


FORMS = _forms()


@pytest.mark.parametrize("form", sorted(FORMS))
def test_a_numeric_query_reaches_the_library_as_the_float32_java_would_make_of_it(
    search, form
):
    query, expected = FORMS[form]
    search(query)
    assert search.library.json_calls == []
    (call,) = search.library.bin_calls
    assert call["dims"] == len(expected)
    # bit for bit: -0.0 against 0.0 and a tie rounded the other way would both show here
    assert [struct.pack("<f", x) for x in call["query"]] == [
        struct.pack("<f", x) for x in expected
    ]


def test_the_library_is_given_the_type_the_property_the_handle_and_k(search):
    search(np.zeros(4, dtype=np.float32), k=7)
    (call,) = search.library.bin_calls
    assert (call["handle"], call["type"], call["prop"], call["k"]) == (
        11,
        b"Doc",
        b"emb",
        7,
    )


def test_a_float32_array_is_not_copied(search):
    query = np.arange(8, dtype=np.float32)
    search(query)
    assert search.library.bin_calls[0]["query"] == query.tolist()
    assert schema_objects._vindex_query(query) is query


# the shapes the binary entry does not carry: each keeps the JSON route, with the vector the old code sent
NOT_CARRIED = {
    "a generator": (lambda: (x for x in [0.5, 0.25]), [0.5, 0.25]),
    "a None entry": (lambda: [0.5, None], [0.5, None]),
    "a bool entry": (lambda: [0.5, True], [0.5, True]),
    "a NaN": (lambda: [0.5, float("nan")], [0.5, float("nan")]),
    "an infinity": (lambda: [0.5, float("inf")], [0.5, float("inf")]),
    "a double beyond float32": (lambda: [0.5, 1e39], [0.5, 1e39]),
    "an int beyond 2**53": (lambda: [0.5, 2**53 + 1], [0.5, 2**53 + 1]),
    "an empty list": (lambda: [], []),
    "an empty array": (lambda: np.zeros(0, dtype=np.float32), []),
    "an int array": (lambda: np.array([1, 2, 3]), [1, 2, 3]),
    "a 2-d array": (lambda: np.zeros((2, 2), dtype=np.float32), [[0.0, 0.0]] * 2),
    "numpy scalars in a list": (
        lambda: [np.float32(0.5), np.float32(0.25)],
        [0.5, 0.25],
    ),
    "a big-endian array": (
        lambda: np.array([0.5, 0.25], dtype=">f4"),
        [0.5, 0.25],
    ),
    "a nested list": (lambda: [[0.5], [0.25]], [[0.5], [0.25]]),
}


@pytest.mark.parametrize("shape", sorted(NOT_CARRIED))
def test_a_query_the_entry_cannot_carry_keeps_the_json_route(search, shape):
    make, sent = NOT_CARRIED[shape]
    search(make())
    assert search.library.bin_calls == []
    ((op, args),) = search.library.json_calls
    assert op == b"approximate"
    # what the old code put into the text (NaN and the infinities as the json module spells them)
    assert json.dumps(args["vector"]) == json.dumps(sent)
    assert (args["type"], args["prop"], args["k"], args["allowed"]) == (
        "Doc",
        "emb",
        3,
        None,
    )


def test_the_switch_and_a_library_without_the_entry_keep_the_json_route(
    search, monkeypatch
):
    monkeypatch.setattr(lib, "VINDEX_BIN", False)
    search(np.arange(8, dtype=np.float32))
    assert search.library.bin_calls == []
    assert len(search.library.json_calls) == 1


def test_without_numpy_the_json_route_answers(search, monkeypatch):
    monkeypatch.setattr(schema_objects, "_np", None)
    search([0.5, 0.25])
    assert search.library.bin_calls == []
    assert len(search.library.json_calls) == 1


# ---------------------------------------------------------------------------------------------------- the filter

ALLOWED = ["#3:5", "#0:0", "#12:123456789012", "#999999999:7"]


def test_a_filter_of_plain_rids_crosses_as_int64_pairs(search):
    search(np.zeros(4, dtype=np.float32), allowed=ALLOWED)
    (call,) = search.library.bin_calls
    assert call["allowed"] == [(3, 5), (0, 0), (12, 123456789012), (999999999, 7)]
    assert search.library.json_calls == []


def test_a_record_with_a_rid_is_a_filter_entry_as_its_text_is(search):
    class Record:
        def __init__(self, rid):
            self._rid = rid

        def get_rid(self):
            return self._rid

    search(np.zeros(4, dtype=np.float32), allowed=[Record("#4:8"), "#1:2"])
    assert search.library.bin_calls[0]["allowed"] == [(4, 8), (1, 2)]


def test_no_filter_and_an_empty_filter_send_no_pairs(search):
    search(np.zeros(4, dtype=np.float32), allowed=None)
    search(np.zeros(4, dtype=np.float32), allowed=[])
    for call in search.library.bin_calls:
        assert call["allowed"] == [] and call["allowed_address"] == 0


@pytest.mark.parametrize(
    "allowed",
    [
        [
            "3:5"
        ],  # no '#': the engine's RID parser accepts it, the entry does not read it
        ["#3:5", "#a:b"],
        ["#-1:-1"],
        ["#3:5;#4:6"],
        ["#1234567890:1"],  # a bucket of ten digits
        ["#1:1234567890123456789"],  # a position of 19 digits
        [""],
    ],
)
def test_a_filter_that_is_not_plain_rid_text_keeps_the_json_route(search, allowed):
    search(np.zeros(4, dtype=np.float32), allowed=allowed)
    assert search.library.bin_calls == []
    ((_, args),) = search.library.json_calls
    assert args["allowed"] == allowed


# ---------------------------------------------------------------------------------------------------- the answer


def test_the_answer_is_the_json_routes_answer_with_the_same_types_order_and_bits(
    search, monkeypatch
):
    via_bin = search(np.arange(8, dtype=np.float32), k=4)
    monkeypatch.setattr(lib, "VINDEX_BIN", False)
    via_json = search(np.arange(8, dtype=np.float32), k=4)
    assert via_bin == via_json
    assert [type(score) for _, score in via_bin] == [float] * 4
    assert [struct.pack("<d", s) for _, s in via_bin] == [
        struct.pack("<d", s) for _, s in via_json
    ]
    # ascending by score, the two equal scores in the order the engine found them, cut to k
    assert via_bin == [("#0:1", 0.1), ("#1:3", 0.125), ("#1:4", 0.125), ("#2:7", 0.25)]


def test_a_rid_is_the_text_the_engine_prints(search):
    search.library.hits = [(0, 0, 0.0), (1234, 9876543210, 1.5)]
    assert [rid for rid, _ in search(np.zeros(2, dtype=np.float32), k=2)] == [
        "#0:0",
        "#1234:9876543210",
    ]


def test_no_hits_is_an_empty_list(search):
    search.library.hits = []
    assert search(np.zeros(2, dtype=np.float32)) == []


def test_each_hit_is_looked_up_once_in_hit_order(search):
    search(np.zeros(2, dtype=np.float32), k=5)
    assert search.database.lookups == ["#2:7", "#1:3", "#3:9", "#0:1", "#1:4"]


@pytest.mark.parametrize("k", [np.int64(3), 3.0])
def test_k_is_read_as_an_int(search, k):
    search(np.zeros(2, dtype=np.float32), k=k)
    assert search.library.bin_calls[0]["k"] == 3
    assert type(search.library.bin_calls[0]["k"]) is int


# ---------------------------------------------------------------------------------------------------- the buffer


def test_the_first_call_has_room_for_eight_bucket_indexes_of_k_hits(search):
    search(np.zeros(2, dtype=np.float32), k=10)
    assert search.library.bin_calls[0]["cap"] == 80


def test_an_answer_larger_than_the_buffer_is_asked_for_again_with_room_and_the_size_is_kept(
    search,
):
    search.index._shim._vindex_parts = 1
    search.library.hits = [(b, p, p / 100) for b in range(4) for p in range(5)]  # 20
    hits = search(np.zeros(2, dtype=np.float32), k=2)
    assert [c["cap"] for c in search.library.bin_calls] == [2, 20]
    assert len(hits) == 2 and search.library.json_calls == []
    search(np.zeros(2, dtype=np.float32), k=2)
    assert [c["cap"] for c in search.library.bin_calls] == [2, 20, 20]


def test_an_index_that_outgrows_every_buffer_is_answered_by_the_json_route(search):
    search.index._shim._vindex_parts = 1
    search.library.growth = 40
    search.library.hits = [(0, i, i / 100) for i in range(5)]
    hits = search(np.zeros(2, dtype=np.float32), k=2)
    assert len(search.library.bin_calls) == 3
    assert len(search.library.json_calls) == 1
    assert [rid for rid, _ in hits] == ["#0:0", "#0:1"]


@pytest.mark.parametrize("k", [2**31, -(2**31) - 1])
def test_a_k_that_is_not_a_c_int_keeps_the_json_route(search, k):
    search(np.zeros(2, dtype=np.float32), k=k)
    assert search.library.bin_calls == []
    assert len(search.library.json_calls) == 1


def test_an_answer_buffer_of_more_than_a_gibibyte_is_not_asked_for(search):
    search(
        np.zeros(2, dtype=np.float32), k=10**7
    )  # 8 bucket indexes x 10**7 hits x 24 bytes
    assert search.library.bin_calls == []
    assert len(search.library.json_calls) == 1


def test_k_of_zero_still_has_a_buffer(search):
    search.library.hits = []
    assert search(np.zeros(2, dtype=np.float32), k=0) == []
    assert search.library.bin_calls[0]["cap"] == 8


# ---------------------------------------------------------------------------------------------------- errors


def test_an_error_of_the_library_is_the_error_the_json_route_raises(
    search, monkeypatch
):
    monkeypatch.setattr(lib, "vindex_bin", lambda *a: -1)
    with pytest.raises(ArcadeDBError) as binary:
        search(np.zeros(2, dtype=np.float32))
    reply = {
        "ok": False,
        "error": "boom",
        "cls": "java.lang.IllegalArgumentException",
        "chain": [],
    }
    monkeypatch.setattr(lib, "call", lambda *a: json.dumps(reply))
    monkeypatch.setattr(lib, "VINDEX_BIN", False)
    with pytest.raises(ArcadeDBError) as text:
        search(np.zeros(2, dtype=np.float32))
    assert str(binary.value) == str(text.value)
    assert str(binary.value).startswith("Approximate search failed: java.lang.")
