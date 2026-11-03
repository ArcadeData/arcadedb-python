"""Native build: the UPDATE of save() merges the values that changed, not every value of the record.

The stored row equals the copy of the values as read (save() checks that before it writes), so a value that is as read is already stored
and need not travel again: for a record with a vector this was the bulk of the statement's parameters. The seam the tests read is the
parameter map handed to the engine with the statement (``NativeDatabase._exec``). What the record holds, what a fresh read returns and
the refusal of a stale save are the same as before; test_native_save_baseline.py covers the copy and the read-back.
"""

import numpy as np
import pytest

import arcadedb
from arcadedb._native.database import NativeDatabase

_VEC = [0.5 * i for i in range(64)]  # float32-exact, so a save of it is stored as sent


@pytest.fixture
def merges(monkeypatch):
    """The ``row`` parameter of every UPDATE statement handed to the engine, in order, while the fixture is active."""
    seen = []
    original = NativeDatabase._exec

    def recording(self, flags, prefix, language, command, args):
        if command.startswith("UPDATE") and args:
            seen.append(dict(args[0]["row"]))
        return original(self, flags, prefix, language, command, args)

    monkeypatch.setattr(NativeDatabase, "_exec", recording)
    return seen


def _make(db, **extra):
    db.command("sql", "CREATE DOCUMENT TYPE D")
    with db.transaction():
        doc = db.new_document("D")
        doc.set("id", 1)
        doc.set("name", "a")
        doc.set("vec", list(_VEC))
        for key, value in extra.items():
            doc.set(key, value)
        doc.save()
        return str(doc.get_identity())


def test_one_changed_property_is_the_only_one_merged(temp_db_path, merges):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _make(db)
        rec = db.lookup_by_rid(rid).modify()
        rec.set("name", "b")
        del merges[:]
        with db.transaction():
            rec.save()
        assert merges == [{"name": "b"}]
        stored = db.lookup_by_rid(rid)
        assert stored.get("name") == "b"
        assert stored.get("id") == 1
        assert stored.get("vec") == _VEC


def test_a_save_that_changes_nothing_still_runs_its_update(temp_db_path, merges):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _make(db)
        rec = db.lookup_by_rid(rid).modify()
        del merges[:]
        with db.transaction():
            rec.save()
        assert len(merges) == 1
        assert merges[0]["name"] == "a" and merges[0]["id"] == 1
        assert db.lookup_by_rid(rid).get("vec") == _VEC


def test_a_changed_vector_goes_as_the_caller_holds_it(temp_db_path, merges):
    """An ndarray is sent as the array (the codec's binary path), not as the list the copy of it is."""
    with arcadedb.create_database(temp_db_path) as db:
        rid = _make(db)
        rec = db.lookup_by_rid(rid).modify()
        new = np.asarray(_VEC, dtype=np.float32) + 1.0
        rec.set("vec", new)
        del merges[:]
        with db.transaction():
            rec.save()
        assert list(merges[0]) == ["vec"]
        assert merges[0]["vec"] is new
        assert db.lookup_by_rid(rid).get("vec") == new.tolist()


def test_a_tuple_equal_to_what_was_read_is_not_merged(temp_db_path, merges):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _make(db, tags=["x", "y"])
        rec = db.lookup_by_rid(rid).modify()
        rec.set("tags", ("x", "y"))
        rec.set("name", "c")
        del merges[:]
        with db.transaction():
            rec.save()
        assert merges == [{"name": "c"}]


def test_a_new_property_and_a_changed_one_are_merged_together(temp_db_path, merges):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _make(db)
        rec = db.lookup_by_rid(rid).modify()
        rec.set("fresh", "x")  # not read before: the update reads the stored row back
        rec.set("id", 2)
        del merges[:]
        with db.transaction():
            rec.save()
        assert merges == [{"id": 2, "fresh": "x"}]
        stored = db.lookup_by_rid(rid)
        assert (stored.get("id"), stored.get("fresh"), stored.get("name")) == (
            2,
            "x",
            "a",
        )
        # the object's copy is the stored row, so its next save is not a conflict and merges only what changes again
        rec.set("fresh", "y")
        del merges[:]
        with db.transaction():
            rec.save()
        assert merges == [{"fresh": "y"}]
        assert db.lookup_by_rid(rid).get("fresh") == "y"


def test_a_stale_save_is_still_refused_and_writes_nothing(temp_db_path, merges):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _make(db)
        mine = db.lookup_by_rid(rid).modify()
        other = db.lookup_by_rid(rid).modify()
        other.set("name", "other")
        with db.transaction():
            other.save()
        mine.set("id", 9)
        del merges[:]
        with pytest.raises(
            arcadedb.ArcadeDBError, match="ConcurrentModificationException"
        ):
            with db.transaction():
                mine.save()
        assert merges == []
        stored = db.lookup_by_rid(rid)
        assert (stored.get("id"), stored.get("name")) == (1, "other")


def test_repeated_saves_of_one_object_merge_one_change_each(temp_db_path, merges):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _make(db)
        rec = db.lookup_by_rid(rid).modify()
        del merges[:]
        with db.transaction():
            for k in range(3):
                rec.set("name", f"n{k}")
                rec.save()
        assert merges == [{"name": "n0"}, {"name": "n1"}, {"name": "n2"}]
        assert db.lookup_by_rid(rid).get("name") == "n2"
