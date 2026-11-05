"""Vertex.new_edge on a vertex object that was read before other commits.

The first edge of a vertex rewrites the vertex record (the head chunk of its edge list). A Java vertex object read before that
commit holds the old version, and the engine's update then failed with ConcurrentModificationException on the JVM build while
the native build, which goes by record id, succeeded (P-08 of the bindings discovery pass). Both builds now resolve both ends
by record id inside the active transaction, so a vertex object read once can create edges in any later transaction.
"""

import arcadedb

N_VERTICES = 6


def _make_graph(db):
    db.command("sql", "CREATE VERTEX TYPE SPerson")
    db.command("sql", "CREATE EDGE TYPE SKnows")
    with db.transaction():
        for i in range(N_VERTICES):
            db.command("sql", "CREATE VERTEX SPerson SET id = ?", i)
    rows = db.query("sql", "SELECT @rid AS r FROM SPerson ORDER BY id").to_list()
    return [db.lookup_by_rid(str(row["r"])) for row in rows]


def _edge_count(db):
    return db.query("sql", "SELECT count(*) AS c FROM SKnows").first().get("c")


def _degrees(db, i, direction):
    sql = f"SELECT {direction}('SKnows').size() AS d FROM SPerson WHERE id = ?"  # nosec B608 - direction is a fixed literal
    return db.query("sql", sql, i).first().get("d")


def test_new_edge_from_vertices_read_once(temp_db_path):
    """Source and target objects read once, one edge per transaction; the first
    edge of a vertex must not make its later edges fail."""
    with arcadedb.create_database(temp_db_path) as db:
        vs = _make_graph(db)
        pairs = [(i, (i + 1) % N_VERTICES) for i in range(N_VERTICES)]
        pairs += [(i, (i + 2) % N_VERTICES) for i in range(N_VERTICES)]
        for src, dst in pairs:
            with db.transaction():
                edge = vs[src].new_edge("SKnows", vs[dst], w=src)
                assert edge.get("w") == src

        assert _edge_count(db) == len(pairs)
        for i in range(N_VERTICES):
            assert _degrees(db, i, "out") == 2
            assert _degrees(db, i, "in") == 2


def test_new_edge_stale_target_only(temp_db_path):
    """The source is read fresh, only the target object is stale."""
    with arcadedb.create_database(temp_db_path) as db:
        vs = _make_graph(db)
        with db.transaction():
            vs[0].new_edge("SKnows", vs[1])  # rewrites the record of vertex 1

        fresh = db.lookup_by_rid(vs[2].get_rid())
        with db.transaction():
            fresh.new_edge("SKnows", vs[1])  # vs[1] is stale since the commit above

        assert _edge_count(db) == 2
        assert _degrees(db, 1, "in") == 2


def test_new_edge_stale_after_edge_made_another_way(temp_db_path):
    """An edge made by SQL in an earlier transaction also leaves the objects stale."""
    with arcadedb.create_database(temp_db_path) as db:
        vs = _make_graph(db)
        with db.transaction():
            db.command(
                "sql",
                "CREATE EDGE SKnows FROM (SELECT FROM SPerson WHERE id = 0) "
                "TO (SELECT FROM SPerson WHERE id = 1)",
            )
        with db.transaction():
            vs[0].new_edge("SKnows", vs[1])
        with db.transaction():
            vs[1].new_edge("SKnows", vs[0])

        assert _edge_count(db) == 3
        assert _degrees(db, 0, "out") == 2
        assert _degrees(db, 0, "in") == 1


def test_new_edge_two_edges_in_one_transaction_from_stale_objects(temp_db_path):
    """Several edges from the same stale objects inside one transaction."""
    with arcadedb.create_database(temp_db_path) as db:
        vs = _make_graph(db)
        with db.transaction():
            vs[0].new_edge("SKnows", vs[1])
        with db.transaction():
            vs[0].new_edge("SKnows", vs[2])
            vs[0].new_edge("SKnows", vs[1])
            vs[3].new_edge("SKnows", vs[0])

        assert _edge_count(db) == 4
        assert _degrees(db, 0, "out") == 3
        assert _degrees(db, 0, "in") == 1
