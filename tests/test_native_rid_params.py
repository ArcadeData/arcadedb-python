"""An RID built from an RID is the same id, and it travels as a parameter like that id (#265).

``find_nearest(include_vector=True)`` hands out records whose rid is already the row's ``RID``, and the native record wrappers wrap
their rid again (``RID(self._rid)`` in save, the update, delete, ``new_edge`` and the edge reads). ``RID.__init__`` kept its argument
as is, so ``.text`` was an ``RID``: the fast named-parameter path of ``codec.encode_params`` raised ``TypeError: Object of type RID
is not JSON serializable`` on save, and the slower paths sent a nested tag. The save itself is covered by
``test_vector_hit_records.py::test_a_hit_can_be_modified_and_saved``; conftest.py collects this file on the native backend only.
"""

import pytest

from arcadedb._native.codec import RID, encode_params


def test_an_rid_of_an_rid_keeps_the_text():
    inner = RID("#3:7")
    outer = RID(inner)
    assert type(outer.text) is str
    assert outer.text == "#3:7"
    assert outer == inner
    assert hash(outer) == hash(inner)
    assert str(outer) == "#3:7"
    assert repr(outer) == "RID('#3:7')"
    assert RID(RID(RID("#3:7"))).text == "#3:7"


@pytest.mark.parametrize("binary", [False, True])
def test_a_wrapped_rid_encodes_like_the_rid(binary):
    def text(args):
        out = encode_params(args, binary=binary)
        return out[0] if binary else out

    # a named map of plain values and RIDs (the fast path: save's conflict check, delete, new_edge's src and dst)
    assert text([{"rid": RID(RID("#3:7"))}]) == text([{"rid": RID("#3:7")}])
    assert text([{"src": RID(RID("#3:7")), "dst": RID("#3:8")}]) == text(
        [{"src": RID("#3:7"), "dst": RID("#3:8")}]
    )
    # an RID beside a value that is not plain (the update: rid and row) and a positional RID take the hook path
    assert text([{"rid": RID(RID("#3:7")), "row": {"a": [1, 2]}}]) == text(
        [{"rid": RID("#3:7"), "row": {"a": [1, 2]}}]
    )
    assert text([RID(RID("#3:7")), 1]) == text([RID("#3:7"), 1])
