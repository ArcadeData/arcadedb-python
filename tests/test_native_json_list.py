"""``to_json_list`` on the native build walks the rows with ``codec.json_native`` only when a reply of the result restored a tagged value.

A reply without the escaped tag decodes to ``json.loads`` types, which ``json_native`` returns unchanged, so the walk over every value of
every row was pure overhead (#219). A restored tag (DATE, DATETIME, DECIMAL, a set, an RID, a map with non-string keys) still takes it:
the rows are then the JSON-native rows they always were. The flag is per result and sticky: a result whose first reply is untagged and a
later one is tagged (or the other way round) walks. A reply whose only tags are ``["B", offset, count]`` (the float tail of vectors)
decodes to plain float lists and does not. conftest.py collects this file on the native backend only (the JVM build has no such walk);
the values the two builds return for a link are compared in test_to_json_list_links.py.
"""

import calendar
import json
import struct
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pytest

from arcadedb._native import codec, database
from arcadedb._native.codec import RID

DIM = (
    16  # 8 or more floats per vector: they travel in the float32 tail of a binary reply
)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _ms(year, month, day):
    return calendar.timegm((year, month, day, 0, 0, 0)) * 1000


@pytest.fixture
def spy(monkeypatch):
    """Counts the rows ``json_native`` is called for (the walk) and records, per decoded reply, ``(restored, has float tail)``."""
    seen = {"walk": 0, "replies": []}
    real_walk = database.json_native

    def walk(value):
        seen["walk"] += 1
        return real_walk(value)

    monkeypatch.setattr(database, "json_native", walk)
    real_loads = getattr(
        database, "loads_tagged", None
    )  # absent on a build that has no flag
    if real_loads is not None:

        def loads_tagged(text, floats=None):
            value, restored = real_loads(text, floats)
            seen["replies"].append(
                (restored, floats is not None and len(floats) > 0 and "\\u0001" in text)
            )
            return value, restored

        monkeypatch.setattr(database, "loads_tagged", loads_tagged)
    return seen


def _docs(db, rows, **properties):
    db.command("sql", "CREATE DOCUMENT TYPE Doc")
    db.command("sql", "CREATE PROPERTY Doc.id INTEGER")
    for name, type_name in properties.items():
        db.command("sql", f"CREATE PROPERTY Doc.{name} {type_name}")
    db.insert_many("Doc", rows)
    return db


class TestUntaggedResultSkipsTheWalk:
    def test_a_short_untagged_result_is_not_walked(self, temp_db, spy):
        db = _docs(
            temp_db,
            [
                {"id": i, "name": f"n{i}", "score": i / 4, "ok": i % 2 == 0}
                for i in range(5)
            ],
            name="STRING",
            score="DOUBLE",
            ok="BOOLEAN",
        )
        spy["walk"] = 0
        got = db.query(
            "sql", "SELECT id, name, score, ok FROM Doc ORDER BY id"
        ).to_json_list()
        assert spy["walk"] == 0
        assert spy["replies"][-1] == (False, False)
        assert got == [
            {"id": i, "name": f"n{i}", "score": i / 4, "ok": i % 2 == 0}
            for i in range(5)
        ]
        assert (
            got
            == db.query(
                "sql", "SELECT id, name, score, ok FROM Doc ORDER BY id"
            ).to_list()
        )

    def test_every_batch_of_a_long_untagged_result_is_untagged_and_not_walked(
        self, temp_db, spy
    ):
        db = _docs(
            temp_db, [{"id": i, "name": f"n{i}"} for i in range(3000)], name="STRING"
        )
        spy["walk"] = 0
        spy["replies"].clear()
        got = db.query("sql", "SELECT id, name FROM Doc ORDER BY id").to_json_list()
        assert (
            len(spy["replies"]) >= 3
        )  # 256, then 2048 + 696 through arcadedb_n_next_bin
        assert all(restored is False for restored, _ in spy["replies"])
        assert spy["walk"] == 0
        assert [r["id"] for r in got] == list(range(3000))

    def test_an_empty_result_is_not_walked(self, temp_db, spy):
        db = _docs(temp_db, [])
        spy["walk"] = 0
        assert db.query("sql", "SELECT FROM Doc").to_json_list() == []
        assert spy["walk"] == 0

    def test_the_rows_are_the_decoded_rows(self, temp_db):
        db = _docs(temp_db, [{"id": 1}])
        result = db.query("sql", "SELECT id FROM Doc")
        assert result.to_json_list() == [{"id": 1}]
        assert result._tagged is False


class TestTaggedResultStillWalks:
    ROW = {
        "id": 1,
        "d": date(2026, 1, 2),
        "dt": datetime(2026, 1, 2, 3, 4, 5, 678000),
        "amount": Decimal("1.50"),
    }

    def _db(self, temp_db):
        return _docs(temp_db, [self.ROW], d="DATE", dt="DATETIME", amount="DECIMAL")

    def test_a_tagged_reply_is_walked_and_converted(self, temp_db, spy):
        db = self._db(temp_db)
        spy["walk"] = 0
        got = db.query("sql", "SELECT id, d, dt, amount FROM Doc").to_json_list()
        assert spy["walk"] == 1
        assert spy["replies"][-1][0] is True
        row = got[0]
        assert row["id"] == 1
        assert type(row["d"]) is int and row["d"] == _ms(2026, 1, 2)
        assert (
            type(row["dt"]) is int
            and row["dt"] == _ms(2026, 1, 2) + (3 * 3600 + 4 * 60 + 5) * 1000 + 678
        )
        assert type(row["amount"]) is float and row["amount"] == 1.5
        json.dumps(got)

    def test_to_list_keeps_the_python_types(self, temp_db):
        db = self._db(temp_db)
        row = db.query("sql", "SELECT id, d, dt, amount FROM Doc").to_list()[0]
        assert row["d"] == date(2026, 1, 2)
        assert isinstance(row["dt"], datetime)
        assert row["amount"] == Decimal("1.50") and isinstance(row["amount"], Decimal)

    def test_the_first_reply_untagged_and_a_later_one_tagged_walks(self, temp_db, spy):
        rows = [{"id": i} for i in range(300)]
        for row in rows[270:]:
            row["d"] = date(2026, 1, 2)
        db = _docs(temp_db, rows, d="DATE")
        spy["walk"] = 0
        spy["replies"].clear()
        got = db.query("sql", "SELECT id, d FROM Doc ORDER BY id").to_json_list()
        flags = [restored for restored, _ in spy["replies"]]
        assert (
            flags[0] is False and True in flags[1:]
        )  # the premise: untagged first, tagged later
        assert spy["walk"] == 300
        assert got[0].get("d") is None
        assert got[299]["d"] == _ms(2026, 1, 2) and type(got[299]["d"]) is int
        json.dumps(got)

    def test_the_first_reply_tagged_and_a_later_one_untagged_walks(self, temp_db, spy):
        rows = [{"id": i} for i in range(300)]
        for row in rows[:10]:
            row["d"] = date(2026, 1, 2)
        db = _docs(temp_db, rows, d="DATE")
        spy["walk"] = 0
        spy["replies"].clear()
        got = db.query("sql", "SELECT id, d FROM Doc ORDER BY id").to_json_list()
        flags = [restored for restored, _ in spy["replies"]]
        assert flags[0] is True and False in flags[1:]
        assert spy["walk"] == 300
        assert got[0]["d"] == _ms(2026, 1, 2) and type(got[0]["d"]) is int
        json.dumps(got)

    def test_a_flag_read_from_a_lazy_command_result_covers_every_batch(
        self, temp_db, spy
    ):
        rows = [{"id": i} for i in range(300)]
        for row in rows[270:]:
            row["d"] = date(2026, 1, 2)
        db = _docs(temp_db, rows, d="DATE")
        spy["walk"] = 0
        got = db.command("sql", "SELECT id, d FROM Doc ORDER BY id").to_json_list()
        assert spy["walk"] == 300
        assert got[299]["d"] == _ms(2026, 1, 2)


class TestBinaryVectorReply:
    def _vectors(self, db, count):
        rows = [
            {"id": i, "vec": [f32((i * DIM + k) / 1000.0 - 3.0) for k in range(DIM)]}
            for i in range(count)
        ]
        _docs(db, rows, vec="ARRAY_OF_FLOATS")
        return rows

    def test_a_reply_with_only_float_tail_tags_is_not_walked_and_stays_plain(
        self, temp_db, spy
    ):
        rows = self._vectors(temp_db, 1000)
        spy["walk"] = 0
        spy["replies"].clear()
        got = temp_db.query("sql", "SELECT id, vec FROM Doc ORDER BY id").to_json_list()
        assert any(tail for _, tail in spy["replies"])  # the float tail was used
        assert all(restored is False for restored, _ in spy["replies"])
        assert spy["walk"] == 0
        assert [r["id"] for r in got] == [r["id"] for r in rows]
        for have, want in zip(got, rows):
            assert type(have["vec"]) is list
            assert all(type(x) is float for x in have["vec"])
            assert have["vec"] == want["vec"]
        json.dumps(got)

    def test_a_reply_with_vectors_and_a_date_walks_and_keeps_the_vectors(
        self, temp_db, spy
    ):
        rows = [
            {
                "id": i,
                "d": date(2026, 1, 2),
                "vec": [f32(i + k / 8.0) for k in range(DIM)],
            }
            for i in range(20)
        ]
        _docs(temp_db, rows, d="DATE", vec="ARRAY_OF_FLOATS")
        spy["walk"] = 0
        spy["replies"].clear()
        got = temp_db.query(
            "sql", "SELECT id, d, vec FROM Doc ORDER BY id"
        ).to_json_list()
        assert any(tail for _, tail in spy["replies"])
        assert spy["walk"] == 20
        assert all(r["d"] == _ms(2026, 1, 2) for r in got)
        assert [r["vec"] for r in got] == [r["vec"] for r in rows]
        json.dumps(got)


class TestLoadsTagged:
    def _text(self, rows):
        meta = [[f"#3:{i}", "T", "d"] for i in range(len(rows))]
        return json.dumps(
            {"ok": True, "rs": 1, "rows": rows, "meta": meta, "drained": True},
            separators=(",", ":"),
        )

    @pytest.mark.parametrize(
        "rows, restored",
        [
            ([{"a": 1, "s": "x"}], False),
            (
                [{"s": "a\u0001b"}],
                False,
            ),  # the control character in a string is not a tag
            ([{codec.TAG: "plain", "x": 1}], False),  # a column named like the tag
            ([{"d": {codec.TAG: ["D", "2026-10-31"]}}], True),
            ([{"n": {codec.TAG: ["n", "1.50"]}}], True),
            ([{"r": {codec.TAG: ["R", "#3:4"]}}], True),
            ([{"m": {codec.TAG: ["m", [[1, "a"]]]}}], True),
        ],
    )
    def test_restored_flag_and_same_value_as_loads(self, rows, restored):
        text = self._text(rows)
        value, flag = codec.loads_tagged(text)
        assert flag is restored
        assert value == codec.loads(text)

    def test_a_float_tail_tag_alone_is_not_a_restored_value(self):
        floats = np.array([0.25, 0.5, 0.75], dtype=np.float32)
        text = self._text([{"v": {codec.TAG: ["B", 0, 3]}}])
        value, flag = codec.loads_tagged(text, floats)
        assert flag is False
        assert value["rows"][0]["v"] == [0.25, 0.5, 0.75]
        assert value == codec.loads(text, floats)

    def test_a_float_tail_tag_next_to_a_date_is_restored(self):
        floats = np.array([0.25, 0.5], dtype=np.float32)
        text = self._text(
            [{"v": {codec.TAG: ["B", 0, 2]}, "d": {codec.TAG: ["D", "2026-10-31"]}}]
        )
        value, flag = codec.loads_tagged(text, floats)
        assert flag is True
        assert value == codec.loads(text, floats)


class TestJsonNativeIsJsonSafe:
    def test_an_rid_is_its_text(self):
        out = codec.json_native({"r": RID("#10:5")})
        assert type(out["r"]) is str and out["r"] == "#10:5"

    def test_an_rid_in_a_list_is_its_text(self):
        out = codec.json_native({"l": [RID("#1:0"), RID("#1:1")]})
        assert [type(x) for x in out["l"]] == [str, str]
        assert out == {"l": ["#1:0", "#1:1"]}
        json.dumps(out)

    def test_an_rid_key_of_a_map_is_its_text(self):
        out = codec.json_native({"m": {RID("#1:0"): 1, RID("#1:1"): [RID("#2:2")]}})
        assert [type(k) for k in out["m"]] == [str, str]
        assert out == {"m": {"#1:0": 1, "#1:1": ["#2:2"]}}
        assert type(out["m"]["#1:1"][0]) is str
        json.dumps(out)

    def test_other_keys_that_are_not_strings_become_the_text_the_jvm_build_writes(self):
        out = codec.json_native(
            {
                "m": {1: "a", 2.5: "b", None: "d", "k": "e"},
                "t": {True: "c"},
                "f": {False: 0},
            }
        )
        assert out == {
            "m": {"1": "a", "2.5": "b", "null": "d", "k": "e"},
            "t": {"true": "c"},
            "f": {"false": 0},
        }
        assert all(type(k) is str for m in out.values() for k in m)
        json.dumps(out)

    def test_the_other_conversions_are_unchanged(self):
        out = codec.json_native(
            {
                "d": date(2026, 1, 2),
                "n": Decimal("1.5"),
                "s": {7},
                "f": frozenset({8}),
                "l": [1, "x", None],
            }
        )
        assert out == {
            "d": _ms(2026, 1, 2),
            "n": 1.5,
            "s": [7],
            "f": [8],
            "l": [1, "x", None],
        }
        assert type(out["d"]) is int and type(out["n"]) is float


class TestToListIsUnchanged:
    def test_to_list_still_returns_rid_objects_and_non_string_keys(self, temp_db):
        db = temp_db
        db.command("sql", "CREATE VERTEX TYPE P")
        db.command("sql", "CREATE DOCUMENT TYPE D")
        db.command("sql", "CREATE PROPERTY D.m MAP")
        with db.transaction():
            db.command("sql", "CREATE VERTEX P SET n = 'a'")
            db.command("sql", "INSERT INTO D SET m = :m", {"m": {1: "a"}})
        row = db.query("sql", "SELECT @rid AS r FROM P").to_list()[0]
        assert isinstance(row["r"], RID)
        assert db.query("sql", "SELECT m FROM D").to_list() == [{"m": {1: "a"}}]
