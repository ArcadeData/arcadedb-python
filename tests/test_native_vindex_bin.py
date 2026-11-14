"""Native wheel: ``find_nearest_approximate`` through ``arcadedb_n_vindex_bin`` against the JSON route it replaces.

The query vector used to cross as a JSON list of d numbers and the hits came back as a JSON list of ``[rid, score]``. Now a float32 query
crosses by address and the hits come back in one buffer. The answer must be the same list: the same records in the same order, the same
RID texts, and scores of the same type with the same bits (a score is the double that the number text of the engine's float parses to).
The JSON route stays for what the entry cannot carry, and ``lib.VINDEX_BIN`` (``ARCADEDB_NATIVE_VINDEX_BIN=0``) selects it, which is how
these tests run the old route beside the new one in one process.

Whether the new route is faster is a timed A/B, not these tests. What the Python half does with its arguments and its answer, with a
stand-in for the library, is in ``test_native_vindex_glue.py``. This file needs the native library, so conftest.py does not collect it on
the JVM backend.
"""

import struct

import numpy as np
import pytest

import arcadedb
from arcadedb._native import lib
from arcadedb.exceptions import ArcadeDBError

DIM, N, BUCKETS = (
    16,
    1200,
    4,
)  # 300 vectors a bucket: above the 64 PQ clusters, so the search is approximate


@pytest.fixture(scope="module")
def pq_index(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("vindex_bin") / "db")
    db = arcadedb.create_database(path)
    db.command("sql", f"CREATE VERTEX TYPE Doc BUCKETS {BUCKETS}")
    db.command("sql", "CREATE PROPERTY Doc.id INTEGER")
    db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    index = db.create_vector_index(
        "Doc",
        "embedding",
        dimensions=DIM,
        id_property="id",
        quantization="PRODUCT",
        pq_subspaces=4,
        pq_clusters=64,
        build_graph_now=False,
    )
    vectors = np.random.default_rng(11).standard_normal((N, DIM)).astype(np.float32)
    with db.transaction():
        for i in range(N):
            db.command(
                "sql", "INSERT INTO Doc SET id = ?, embedding = ?", i, vectors[i]
            )
    index.build_graph_now()
    index.vectors = vectors
    index.database = db
    yield index
    db.drop()


class _Calls:
    """The approximate searches that reach the library, by entry point."""

    def __init__(self, monkeypatch):
        self.binary, self.text = [], []
        real_bin, real_call = lib.vindex_bin, lib.call

        def vindex_bin(*args):
            n = real_bin(*args)
            self.binary.append(
                {
                    "k": args[3],
                    "dims": args[5],
                    "pairs": args[7],
                    "cap": args[9],
                    "n": n,
                }
            )
            return n

        def call(name, *args):
            if name == "arcadedb_n_vindex" and args[1] == b"approximate":
                self.text.append(args)
            return real_call(name, *args)

        monkeypatch.setattr(lib, "vindex_bin", vindex_bin)
        monkeypatch.setattr(lib, "call", call)


def _signature(hits):
    """What a caller can see of a result: the RID text, the record's id, the score's type and its bits, in order."""
    return [
        (r.get_rid(), r.get("id"), type(score), struct.pack("<d", score))
        for r, score in hits
    ]


def _run(monkeypatch, search, binary=True):
    """``search()`` with the library entries spied on; ``binary=False`` switches the binary entry off (the JSON route answers).
    Returns the result and the :class:`_Calls` of that one search."""
    with monkeypatch.context() as m:
        if not binary:
            m.setattr(lib, "VINDEX_BIN", False)
        calls = _Calls(m)
        return search(), calls


def _both(index, monkeypatch, query, k, allowed_rids=None):
    """The hits of one search by the binary entry and by the JSON route, and the entries each used."""

    def search():
        return index.find_nearest_approximate(query, k=k, allowed_rids=allowed_rids)

    binary, via_bin = _run(monkeypatch, search)
    text, via_text = _run(monkeypatch, search, binary=False)
    return binary, text, via_bin, via_text


@pytest.mark.parametrize("k", [3, 500])
@pytest.mark.parametrize("form", ["float32 array", "float64 array", "float64 list"])
def test_the_binary_route_answers_as_the_json_route_does(
    pq_index, monkeypatch, form, k
):
    """k of 3 is a small answer (4 bucket indexes of 3 hits, cut to 3); k of 500 asks for more than a bucket holds, so every vector comes
    back from every bucket, and the merge across the four answers is long."""
    base = pq_index.vectors[7]
    rng = np.random.default_rng(5 + k)
    wide = (
        base.astype(np.float64) + rng.standard_normal(DIM) * 1e-3
    )  # float32 rounds it
    query = {
        "float32 array": base,
        "float64 array": wide,
        "float64 list": wide.tolist(),
    }[form]
    binary, text, via_bin, via_text = _both(pq_index, monkeypatch, query, k)
    assert len(via_bin.binary) == 1 and via_bin.text == []
    assert via_text.binary == [] and len(via_text.text) == 1
    assert 0 < len(binary) <= k
    if k == 3:
        assert len(binary) == 3
    assert _signature(binary) == _signature(text)
    assert all(type(score) is float for _, score in binary)
    assert all(
        hit.get_rid().startswith("#") and ":" in hit.get_rid() for hit, _ in binary
    )


def test_the_scores_are_the_same_bits_for_queries_that_float32_rounds(
    pq_index, monkeypatch
):
    """Java narrows the double of the JSON text to float32 and the binary route narrows in numpy: a different rounding of any one element
    would move the scores in their last bits, which the comparison of the packed doubles sees.
    """
    rng = np.random.default_rng(21)
    queries = []
    for scale in (1.0, 1e-3, 1e3):
        queries.append(rng.standard_normal(DIM) * scale)
    ties = rng.standard_normal(DIM)
    ties[0], ties[1], ties[2], ties[3] = (
        1 + 2.0**-24,
        1 + 3 * 2.0**-24,
        16777217.0,
        1e-40,
    )
    queries.append(ties)
    for query in queries:
        assert not np.array_equal(query.astype(np.float32).astype(np.float64), query)
        binary, text, _, _ = _both(pq_index, monkeypatch, query, 20)
        assert len(binary) == 20
        assert _signature(binary) == _signature(text)


def test_a_filter_gives_the_same_hits_by_text_and_by_record(pq_index, monkeypatch):
    seed = pq_index.find_nearest_approximate(pq_index.vectors[3], k=120)
    records = [r for r, _ in seed]
    keep = records[::3]
    for allowed in ([r.get_rid() for r in keep], keep):
        binary, text, via_bin, via_text = _both(
            pq_index, monkeypatch, pq_index.vectors[3], 25, allowed
        )
        assert via_bin.binary[0]["pairs"] == len(keep) and via_bin.text == []
        assert len(via_text.text) == 1
        assert _signature(binary) == _signature(text)
        assert {r.get_rid() for r, _ in binary} <= {r.get_rid() for r in keep}
        assert 0 < len(binary) <= 25


def test_an_empty_filter_is_no_filter(pq_index, monkeypatch):
    binary, text, via_bin, _ = _both(pq_index, monkeypatch, pq_index.vectors[3], 10, [])
    assert via_bin.binary[0]["pairs"] == 0
    assert _signature(binary) == _signature(text) and len(binary) == 10


def test_a_buffer_too_small_for_the_answer_is_asked_for_again(pq_index, monkeypatch):
    """The bucket indexes answer up to k hits each; a buffer for one bucket's answer is too small, and the second call has room for all."""
    shim = pq_index._shim
    k = 40
    expected = _signature(pq_index.find_nearest_approximate(pq_index.vectors[9], k=k))
    monkeypatch.setattr(shim, "_vindex_parts", 1)
    calls = _Calls(monkeypatch)
    again = pq_index.find_nearest_approximate(pq_index.vectors[9], k=k)
    n = calls.binary[0]["n"]
    parts = -(-n // k)
    assert n > k and parts <= BUCKETS
    assert [c["cap"] for c in calls.binary] == [k, k * parts]
    assert calls.text == []
    assert _signature(again) == expected
    assert shim._vindex_parts == parts
    pq_index.find_nearest_approximate(pq_index.vectors[9], k=k)
    assert [c["cap"] for c in calls.binary] == [
        k,
        k * parts,
        k * parts,
    ]  # one crossing now, with the size that was found


# queries the binary entry does not carry: the engine decides what they mean, so the JSON route must still be the one that runs, with
# the outcome (hits or error) the same as with the entry switched off
NOT_CARRIED = {
    "a NaN": lambda v: np.concatenate([[np.nan], v[1:].astype(np.float64)]),
    "an infinity": lambda v: np.concatenate([[np.inf], v[1:].astype(np.float64)]),
    "beyond float32": lambda v: np.concatenate([[1e39], v[1:].astype(np.float64)]),
    "an int array": lambda v: np.arange(DIM),
    "numpy scalars": lambda v: [np.float32(x) for x in v],
    "a None entry": lambda v: [None] + v[1:].tolist(),
}


def _outcome(call):
    try:
        return ("hits", _signature(call()))
    except ArcadeDBError as e:
        return ("error", str(e))


@pytest.mark.parametrize("shape", sorted(NOT_CARRIED))
def test_a_query_the_entry_cannot_carry_keeps_the_json_route(
    pq_index, monkeypatch, shape
):
    query = NOT_CARRIED[shape](pq_index.vectors[4])

    def search():
        return _outcome(lambda: pq_index.find_nearest_approximate(query, k=5))

    outcome, via_bin = _run(monkeypatch, search)
    expected, _ = _run(monkeypatch, search, binary=False)
    assert outcome == expected
    assert via_bin.binary == [] and len(via_bin.text) == 1


def test_a_wrong_dimension_is_the_engines_answer_on_both_routes(pq_index, monkeypatch):
    """Thirteen float32 numbers cross by address; what the engine makes of a query of the wrong length is the same either way."""
    query = pq_index.vectors[4][: DIM - 3]

    def search():
        return _outcome(lambda: pq_index.find_nearest_approximate(query, k=5))

    outcome, via_bin = _run(monkeypatch, search)
    expected, _ = _run(monkeypatch, search, binary=False)
    assert outcome == expected
    assert len(via_bin.binary) == 1 and via_bin.binary[0]["dims"] == DIM - 3
    assert via_bin.text == []


def test_a_filter_that_names_no_record_of_the_index_is_answered_alike(
    pq_index, monkeypatch
):
    """A plain RID of a bucket the type does not have: the engine's answer (hits or an error) is the same on both routes."""

    def search():
        return _outcome(
            lambda: pq_index.find_nearest_approximate(
                pq_index.vectors[4], k=5, allowed_rids=["#999:0"]
            )
        )

    outcome, via_bin = _run(monkeypatch, search)
    expected, _ = _run(monkeypatch, search, binary=False)
    assert outcome == expected
    assert len(via_bin.binary) == 1 and via_bin.binary[0]["pairs"] == 1


def test_a_search_on_an_unindexed_property_raises_the_same_error(pq_index, monkeypatch):
    """The library looks the index up by type and property on both routes: the same refusal comes back."""
    from arcadedb._native.schema_objects import NativeVectorIndex, _IndexShim

    bogus = NativeVectorIndex.__new__(NativeVectorIndex)
    bogus._database = pq_index.database
    bogus._shim = _IndexShim(pq_index.database, "Doc", "id", "Doc[id]")
    monkeypatch.setattr(bogus, "get_quantization", lambda: "PRODUCT")

    def search():
        return _outcome(
            lambda: bogus.find_nearest_approximate(pq_index.vectors[4], k=5)
        )

    outcome, via_bin = _run(monkeypatch, search)
    expected, _ = _run(monkeypatch, search, binary=False)
    assert outcome[0] == "error" and "No vector index on Doc[id]" in outcome[1]
    assert outcome == expected
    assert len(via_bin.binary) == 1
