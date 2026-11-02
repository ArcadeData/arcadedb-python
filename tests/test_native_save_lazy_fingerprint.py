"""Native build: save() of a record refuses to overwrite a record that changed after it was read, and the check costs a read nothing.

The fingerprint of the values a record held as read is taken lazily, at ``modify()``, at the first ``set()`` of a saved record or at
``save()``, never when a read decodes a row into a record. The seam is ``arcadedb._native.records._fingerprint``: these tests count its
calls. The refusal itself is also tested in test_graph_api.py, which runs on both builds.
"""

import pytest

import arcadedb
from arcadedb._native import records


@pytest.fixture
def fingerprints(monkeypatch):
    """The list of calls to the fingerprint function (it still computes the real value)."""
    calls = []
    real = records._fingerprint

    def counting(props):
        calls.append(dict(props))
        return real(props)

    monkeypatch.setattr(records, "_fingerprint", counting)
    return calls


def _create(db):
    db.command("sql", "CREATE VERTEX TYPE V")
    db.command("sql", "CREATE EDGE TYPE E")
    with db.transaction():
        a = db.new_vertex("V")
        a.set("a", 1)
        a.set("b", 1)
        a.save()
        b = db.new_vertex("V")
        b.set("a", 2)
        b.save()
        a.new_edge("E", b, w=1.5)
        return str(a.get_identity())


def _stored(db, rid):
    row = list(db.query("sql", f"SELECT a, b FROM {rid}"))[0]
    return row.get("a"), row.get("b")


def test_reads_never_compute_the_fingerprint(temp_db_path, fingerprints):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)
        fingerprints.clear()

        rows = list(db.query("sql", "SELECT FROM V"))
        records_read = [row.get_element() for row in rows]
        by_rid = db.lookup_by_rid(rid)
        vertex = records_read[0]
        edges = vertex.get_out_edges("E") + vertex.get_in_edges("E")
        ends = [edge.get_in() for edge in edges] + [edge.get_out() for edge in edges]
        cypher = [
            row.get_element() for row in db.query("opencypher", "MATCH (n:V) RETURN n")
        ]
        assert len(records_read) == 2 and by_rid is not None and ends and cypher
        assert fingerprints == []  # nothing was fingerprinted by any read path

        mutable = by_rid.modify()
        assert (
            len(fingerprints) == 1
        )  # modify() takes it, once, from the values the record holds as read
        assert fingerprints[0] == by_rid.to_dict()
        by_rid.modify()
        assert len(fingerprints) == 1  # and caches it on the record

        mutable.set("a", 10)  # the snapshot is already there: no work
        assert len(fingerprints) == 1
        with db.transaction():
            mutable.save()  # the stored values to compare with, and the stored values after the write
        assert len(fingerprints) == 3
        assert _stored(db, rid) == (10, 1)


def test_modify_then_save_after_a_concurrent_change_still_raises(
    temp_db_path, fingerprints
):
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)

        mutable = db.lookup_by_rid(rid).modify()
        with (
            db.transaction()
        ):  # another transaction changes the stored record after the read
            db.command("sql", f"UPDATE {rid} SET b = 2")
        mutable.set("a", 10)
        with pytest.raises(arcadedb.ArcadeDBError) as excinfo:
            with db.transaction():
                mutable.save()
        assert "ConcurrentModificationException" in str(excinfo.value)
        assert (
            excinfo.value.native["retry"] == "need_retry"
        )  # run_in_transaction retries it
        assert _stored(db, rid) == (1, 2)  # nothing of the refused save is stored

        # outside a transaction too, and a fresh read saves
        with pytest.raises(arcadedb.ArcadeDBError):
            mutable.save()
        assert _stored(db, rid) == (1, 2)
        fresh = db.lookup_by_rid(rid).modify()
        fresh.set("a", 10)
        fresh.save()
        assert _stored(db, rid) == (10, 2)


def test_a_change_between_the_read_and_modify_is_not_lost_when_modify_comes_late(
    temp_db_path,
):
    """The fingerprint is of the values read, not of what is stored when modify() runs."""
    with arcadedb.create_database(temp_db_path) as db:
        rid = _create(db)

        read = db.lookup_by_rid(rid)
        with db.transaction():
            db.command("sql", f"UPDATE {rid} SET b = 2")
        mutable = (
            read.modify()
        )  # late: the stored record is already newer than the values held
        mutable.set("a", 10)
        with pytest.raises(arcadedb.ArcadeDBError):
            with db.transaction():
                mutable.save()
        assert _stored(db, rid) == (1, 2)


def test_a_record_created_here_then_changed_elsewhere_is_refused(
    temp_db_path, fingerprints
):
    """A record this object created has no fingerprint until its first change: taken before that change, from the stored values."""
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE D")
        with db.transaction():
            created = db.new_document("D")
            created.set("a", 1)
            created.set("b", 1)
            created.save()
        assert (
            fingerprints == []
        )  # creating and saving a new record fingerprints nothing
        rid = str(created.get_identity())
        with db.transaction():
            db.command("sql", f"UPDATE {rid} SET b = 2")
        created.set("a", 9)
        with pytest.raises(arcadedb.ArcadeDBError):
            with db.transaction():
                created.save()
        assert _stored(db, rid) == (1, 2)
