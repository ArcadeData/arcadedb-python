"""codec.loads skips the per-object hook when a reply holds no tagged value (#151).

The native library writes every character below 0x20 as a \\u escape (NativeCodec.quote), so a tagged value always puts the escaped tag
in the reply text. These tests state that the fast path gives the same objects as the hook path, that it is taken only for tag-free text,
and that a string or a key merely containing U+0001 still decodes as before.
"""

import json
from datetime import date

import pytest

from arcadedb._native import codec


def _reply(rows):
    meta = [[f"#3:{i}", "T", "d"] for i in range(len(rows))]
    return json.dumps(
        {"ok": True, "rs": 1, "rows": rows, "meta": meta, "drained": True},
        separators=(",", ":"),
    )


class TestLoadsWithoutTag:
    @pytest.mark.parametrize(
        "rows",
        [
            [],
            [{"a": 1}],
            [
                {
                    "name": "x",
                    "n": 2,
                    "f": 0.5,
                    "flag": True,
                    "none": None,
                    "list": [1, 2],
                    "map": {"k": "v"},
                }
            ],
            [{"name": f"n{i}", "n": i} for i in range(1000)],
        ],
    )
    def test_a_tag_free_reply_decodes_as_with_the_hook(self, rows):
        text = _reply(rows)
        assert codec._TAG_ESCAPED not in text
        assert codec.loads(text) == json.loads(text, object_hook=codec._hook)

    def test_the_hook_is_not_called_for_a_tag_free_reply(self, monkeypatch):
        calls = []
        monkeypatch.setattr(codec, "_hook", lambda obj: calls.append(obj) or obj)
        codec.loads(_reply([{"a": 1}, {"b": 2}]))
        assert calls == []

    def test_the_float_tail_argument_does_not_change_a_tag_free_reply(self):
        text = _reply([{"a": 1.5}])
        assert codec.loads(text, floats=[0.25, 0.5]) == json.loads(text)


class TestLoadsWithTag:
    def test_a_tagged_value_is_still_restored(self):
        text = _reply([{"d": {codec.TAG: ["D", "2026-10-31"]}}])
        assert codec._TAG_ESCAPED in text
        assert codec.loads(text)["rows"][0]["d"] == date(2026, 10, 31)

    def test_a_string_holding_the_control_character_comes_back_unchanged(self):
        text = _reply([{"s": "a\u0001b"}])
        assert codec._TAG_ESCAPED in text
        assert codec.loads(text)["rows"][0]["s"] == "a\u0001b"

    def test_a_column_named_like_the_tag_is_kept(self):
        text = _reply([{codec.TAG: "plain", "x": 1}])
        assert codec.loads(text)["rows"][0] == {codec.TAG: "plain", "x": 1}
