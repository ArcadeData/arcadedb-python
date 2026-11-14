"""How the native build reads a reply out of the library (``lib._take``; ``call``, ``call_bin`` and ``exec_query`` all end there).

A reply is a malloc'd UTF-8 text, optionally followed by a float32 tail, and it must be freed exactly once. It used to be copied as
``bytes`` (``ctypes.string_at``), cut for the tail marker, decoded to ``str``, and only then parsed. It is now decoded straight from its
address in one copy, and only the float tail is ever copied as ``bytes``: the marker is looked for in the decoded text, so a reply
with no marker costs no peek at its last bytes (a foreign call fewer per reply).

The oracle is a private copy of the former reader (``_former``): every case is read by both, from two identical replies, and the text,
the floats and the number of frees must agree. The copy counts are read from a spy on ``ctypes.string_at``: the former reader called it
once with the whole reply, the new one never does for a text and asks only for the tail. The free counts are read
from a spy on ``arcadedb_free``: once, after the use, also when the decode or the tail fails. conftest.py collects this file on the
native backend only: it needs the native library.
"""

import ctypes
import json

import numpy as np
import pytest

from arcadedb._native import lib
from arcadedb.exceptions import ArcadeDBError

_string_at = ctypes.string_at


def _reply(raw: bytes, floats=None):
    """A reply in the library's own memory, as the library hands one out: the text, its NUL, then (4-byte aligned) the floats."""
    tail = b""
    if floats is not None:
        raw = raw[:-1] + b',"bin":%d}' % len(floats)
        pad = -(len(raw) + 1) % 4
        tail = b"\0" * pad + np.asarray(floats, dtype=np.float32).tobytes()
    data = raw + b"\0" + tail
    ptr = lib.alloc(len(data))
    ctypes.memmove(ptr, data, len(data))
    return ptr, len(raw)


def _former(th, ptr):
    """The reader this file replaces (``call_bin`` before the change), for the oracle."""
    try:
        raw = _string_at(ptr)
        floats = None
        if raw.endswith(b"}"):
            i = raw.rfind(b',"bin":', max(0, len(raw) - 40))
            if i >= 0:
                n = int(raw[i + 7 : -1])
                off = (len(raw) + 1 + 3) & ~3
                floats = np.frombuffer(_string_at(ptr + off, 4 * n), dtype=np.float32)
                raw = raw[:i] + b"}"
    finally:
        lib._lib.arcadedb_free(th, ptr)
    return raw.decode("utf-8"), floats


@pytest.fixture
def frees(temp_db, monkeypatch):
    """Every pointer given to ``arcadedb_free``, in order."""
    seen = []
    real = lib._lib.arcadedb_free

    def spy(th, ptr):
        seen.append(ptr)
        return real(th, ptr)

    monkeypatch.setattr(lib._lib, "arcadedb_free", spy)
    return seen


@pytest.fixture
def string_at_sizes(monkeypatch):
    """The size asked of each ``ctypes.string_at`` call (None: the whole NUL-terminated string)."""
    seen = []

    def spy(ptr, size=-1):
        seen.append(None if size == -1 else size)
        return _string_at(ptr, size)

    monkeypatch.setattr(ctypes, "string_at", spy)
    return seen


LONG = json.dumps({"ok": True, "rows": [{"t": "x" * 5000}], "meta": []}).encode()

CASES = {
    "empty object": (b"{}", None),
    "ascii": (b'{"ok":true,"rows":[{"a":1}],"drained":true}', None),
    "non-ascii": ('{"ok":true,"t":"café 日本語 \U0001f600"}'.encode(), None),
    "long text": (LONG, None),
    "tail": (
        b'{"ok":true,"rows":[{"v":["B",0,4]}],"meta":[],"drained":true}',
        [1, 2, 3, 4],
    ),
    "tail, short text": (b'{"v":1}', [0.5, 1.5]),
    "tail, non-ascii text": (
        '{"ok":true,"rows":[{"t":"日本語 café \U0001f600","v":["B",0,3]}],"drained":true}'.encode(),
        [1.5, -2.5, 3.25],
    ),
    "tail, long text": (LONG, np.arange(16, dtype=np.float32)),
    "tail after 50 two-byte characters": (
        ('{"t":"' + "\u00e9" * 50 + '"}').encode(),
        [1.0, 2.0],
    ),
    "no tail, 50 two-byte characters": (
        ('{"t":"' + "\u00e9" * 50 + '"}').encode(),
        None,
    ),
    "tail of no floats": (b'{"ok":true}', []),
    **{
        f"tail, text of {k} digits": (b'{"a":' + b"1" * k + b"}", [7.0, 8.0])
        for k in range(1, 5)  # the four paddings that align the floats to 4 bytes
    },
    "marker deep in the text": (
        b'{"a":"' + b"x" * 100 + b'","b":1,"bin":2,"c":' + b"y" * 60 + b"}",
        None,
    ),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("tail", [True, False])
def test_every_case_reads_as_the_former_reader_does(temp_db, frees, name, tail):
    raw, floats = CASES[name]
    th = lib._thread_handle()
    a, _ = _reply(raw, floats)
    b, _ = _reply(raw, floats)
    want_text, want_floats = _former(th, a)
    if not tail:  # call(): the text as it stands, tail marker included
        want_text = _string_at(b).decode("utf-8")
    del frees[:]
    got_text, got_floats = lib._take(lib._lib, th, b, tail)
    assert got_text == want_text
    assert frees == [b]  # once, and the reply that was read
    if tail and floats is not None:
        assert got_floats.dtype == np.float32
        assert np.array_equal(got_floats, want_floats)
        assert (
            json.loads(got_text) is not None
        )  # the text is a whole JSON value without the marker
    else:
        assert got_floats is None


def test_a_text_reply_is_not_copied_as_bytes(frees, string_at_sizes):
    text = lib.call("arcadedb_n_diag")
    assert json.loads(text)
    assert (
        string_at_sizes == []
    )  # the former reader called string_at once, for the whole reply
    assert len(frees) == 1


def test_a_reply_with_a_tail_copies_only_the_floats(temp_db, frees, string_at_sizes):
    ptr, _ = _reply(LONG, np.arange(16, dtype=np.float32))
    th = lib._thread_handle()
    text, floats = lib._take(lib._lib, th, ptr, True)
    assert len(text) > 5000 and floats.size == 16
    assert string_at_sizes == [
        64
    ]  # 16 floats: never the 5 kB of text, no marker window
    assert frees == [ptr]


@pytest.mark.parametrize("tail", [True, False])
def test_a_reply_with_no_marker_costs_the_read_and_no_peek(
    temp_db, frees, string_at_sizes, monkeypatch, tail
):
    """strlen, one decode and one free: the peek at the last 40 bytes that ``tail`` used to add is a foreign call that nearly
    every reply paid for and nearly none needed."""
    calls = []
    real_strlen, real_decode = lib._strlen, lib._str_from_utf8

    def strlen(p):
        calls.append("strlen")
        return real_strlen(p)

    def decode(p, n):
        calls.append("decode")
        return real_decode(p, n)

    monkeypatch.setattr(lib, "_strlen", strlen)
    monkeypatch.setattr(lib, "_str_from_utf8", decode)
    ptr, _ = _reply(CASES["ascii"][0])
    text, floats = lib._take(lib._lib, lib._thread_handle(), ptr, tail)
    assert json.loads(text)["ok"] is True and floats is None
    assert calls == ["strlen", "decode"]
    assert (
        string_at_sizes == []
    )  # the parent peeked at the last 40 bytes when tail was set
    assert frees == [ptr]


def test_a_real_query_reply_is_read_with_no_string_at(temp_db, string_at_sizes):
    temp_db.command("sql", "CREATE DOCUMENT TYPE R")
    del string_at_sizes[:]
    with temp_db.transaction():
        temp_db.command("sql", "INSERT INTO R SET a = 1, s = 'x'").close()
    assert temp_db.query("sql", "SELECT FROM R").to_list() == [{"a": 1, "s": "x"}]
    assert string_at_sizes == []


def test_a_reply_that_fails_to_decode_is_still_freed_once(temp_db, frees):
    th = lib._thread_handle()
    for raw in (b'{"a":"\xff"}', b'{"a":"\xc3"}', b'{"a":"\xed\xa0\x80"}'):
        ptr, _ = _reply(raw)
        del frees[:]
        with pytest.raises(UnicodeDecodeError):
            lib._take(lib._lib, th, ptr, False)
        assert frees == [ptr]
        ptr, _ = _reply(raw)
        del frees[:]
        with pytest.raises(UnicodeDecodeError):
            lib._take(lib._lib, th, ptr, True)
        assert frees == [ptr]


def test_a_reply_whose_tail_fails_is_still_freed_once(temp_db, frees):
    th = lib._thread_handle()
    ptr, _ = _reply(b'{"ok":true,"bin":x}')
    with pytest.raises(ValueError):
        lib._take(lib._lib, th, ptr, True)
    assert frees == [ptr]


def test_no_reply_raises_and_frees_nothing(temp_db, frees):
    with pytest.raises(ArcadeDBError):
        lib._take(lib._lib, lib._thread_handle(), None, False)
    assert frees == []


def test_the_public_readers_free_each_reply_once(temp_db, frees):
    temp_db.command("sql", "CREATE DOCUMENT TYPE R")
    del frees[:]
    text, floats = lib.call_bin("arcadedb_n_diag")
    assert json.loads(text) and floats is None
    assert len(frees) == 1
    lib.call("arcadedb_n_diag")
    assert len(frees) == 2
