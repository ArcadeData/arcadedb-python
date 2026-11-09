"""The one check that every ingest-integrity test calls (ArcadeData/arcadedb-python#336).

An ingest call must not lose, duplicate or silently alter a row. After a call, ``problems_for_type`` checks three things on the
stored type, each read back through the public API:

1. the record count, read three ways (``SELECT count(*)``, ``Database.count_type`` and the length of the rows that come back),
2. an order-independent digest of every column: the sum and the xor of a stable 64-bit hash of each row, compared with the digest
   of the rows that were sent (two row readers, ``to_list()`` and iterating the result set, must each match it),
3. for a key column, every sent key present exactly once and no other key.

``Store.reopen`` closes the database right after the call and opens it again from disk, and the same three checks run once more
(durability). A check never looks at what the path under test says it did: the expected side is the data the test generated.

``MUTANTS`` are the faults the mutation tests inject into the rows a path is given (drop the last row, skip a middle chunk, store
one row twice, change one value, swap two values). A test family is proven able to fail when the check reports each of them.
"""

import collections
import hashlib

import arcadedb

MASK = (1 << 64) - 1

CHUNK = 7  # rows per crossing, made small so every boundary is cheap
COMMIT = 5  # commit_every and batch_size

#: every problem line carries this word, so a test can tell "the check found it" from "the call raised"
FOUND = "MISMATCH"


class IntegrityError(AssertionError):
    """The stored records are not the sent records."""


def canon(kind, value):
    """The canonical text of one cell. ``kind`` is how the property is declared, so 5 and 5.0 are the same DOUBLE."""
    if value is None:
        return "N"
    if kind == "int":
        return "i%d" % int(value)
    if kind == "float":
        number = float(value)
        return "nan" if number != number else "f" + number.hex()
    if kind == "bool":
        return "b1" if value else "b0"
    if kind == "str":
        return "s" + str(value)
    raise ValueError(f"no canonical form for kind {kind!r}")


def row_hash(schema, row):
    """A stable 64-bit hash of one row: the column names sorted, each cell length-prefixed, a missing cell the same as a null."""
    parts = []
    for name in sorted(schema):
        text = canon(schema[name], row.get(name))
        parts.append("%s:%d:%s" % (name, len(text), text))
    data = "|".join(parts).encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def digest(schema, rows):
    """``(row count, sum of the row hashes mod 2**64, xor of the row hashes)``: the same for any order of the rows."""
    count = total = mixed = 0
    for row in rows:
        h = row_hash(schema, row)
        count += 1
        total = (total + h) & MASK
        mixed ^= h
    return count, total, mixed


def _key_problems(label, view, schema, key, want_rows, rows):
    kind = schema[key]
    wanted = collections.Counter(canon(kind, r.get(key)) for r in want_rows)
    stored = collections.Counter(canon(kind, r.get(key)) for r in rows)
    missing = sorted(set(wanted) - set(stored))
    extra = sorted(set(stored) - set(wanted))
    twice = sorted(k for k, c in stored.items() if c > 1 and wanted.get(k, 0) < c)
    problems = []
    if missing:
        problems.append(
            f"{FOUND} {label}: {len(missing)} sent key(s) not stored, first {missing[:3]} ({view})"
        )
    if extra:
        problems.append(
            f"{FOUND} {label}: {len(extra)} stored key(s) were never sent, first {extra[:3]} ({view})"
        )
    if twice:
        problems.append(
            f"{FOUND} {label}: {len(twice)} key(s) stored more than once, first {twice[:3]} ({view})"
        )
    return problems


def verify(label, want_rows, schema, key, counts, readers):
    """The problems (a list of text lines, empty when none) between the sent rows and what ``counts`` and ``readers`` saw."""
    want = digest(schema, want_rows)
    problems = []
    for name, value in counts.items():
        if value != want[0]:
            problems.append(
                f"{FOUND} {label}: {name} says {value} record(s), {want[0]} were sent"
            )
    first = True
    for name, rows in readers.items():
        got = digest(schema, rows)
        if got[0] != want[0]:
            problems.append(
                f"{FOUND} {label}: reading with {name} returned {got[0]} row(s), {want[0]} were sent"
            )
        elif got != want:
            problems.append(
                f"{FOUND} {label}: reading with {name} returned {got[0]} row(s) whose digest differs from the sent data"
                f" (a stored value changed)"
            )
        if key is not None and first:
            problems += _key_problems(label, name, schema, key, want_rows, rows)
        first = False
    return problems


def type_view(db, name, schema):
    """What the public API reports for one document, vertex or time-series type: the counts and the rows two ways."""
    names = list(schema)
    select = "SELECT FROM " + name  # nosec B608
    count_sql = "SELECT count(*) AS n FROM " + name  # nosec B608
    counts = {
        "SELECT count(*)": int(db.query("sql", count_sql).to_list()[0]["n"]),
        "count_type()": int(db.count_type(name)),
    }
    listed = db.query("sql", select).to_list()
    iterated = [{n: r.get(n) for n in names} for r in db.query("sql", select)]
    return counts, {"to_list()": listed, "iterating the result": iterated}


def problems_for_type(db, name, want_rows, schema, key, label):
    counts, readers = type_view(db, name, schema)
    return verify(label, want_rows, schema, key, counts, readers)


EDGE_SCHEMA = {"a": "int", "b": "int", "w": "float"}


def edge_view(db, edge_type):
    """The counts of an edge type, and its edges as (source key, destination key, weight) rows."""
    count_sql = "SELECT count(*) AS n FROM " + edge_type  # nosec B608
    cypher_count = "MATCH ()-[r:" + edge_type + "]->() RETURN count(r) AS c"
    pairs = "MATCH (a)-[r:" + edge_type + "]->(b) RETURN a.k AS a, b.k AS b, r.w AS w"
    counts = {
        "SELECT count(*)": int(db.query("sql", count_sql).to_list()[0]["n"]),
        "Cypher count(r)": int(db.query("opencypher", cypher_count).to_list()[0]["c"]),
    }
    return counts, {"Cypher MATCH": db.query("opencypher", pairs).to_list()}


def problems_for_edges(db, edge_type, want_rows, label):
    counts, readers = edge_view(db, edge_type)
    return verify(label, want_rows, EDGE_SCHEMA, None, counts, readers)


class Store:
    """A database on disk that a test can close and open again."""

    def __init__(self, path):
        self.path = path
        self.db = arcadedb.create_database(path)

    def reopen(self):
        """Close straight after the last call and open the same files again."""
        self.db.close()
        self.db = arcadedb.open_database(self.path)

    def close(self):
        db, self.db = self.db, None
        if db is not None and db.is_open():
            db.close()


def _mutant_drop_last(rows):
    return rows[:-1]


def _mutant_drop_first(rows):
    return rows[1:]


def _mutant_skip_a_chunk(rows):
    third = max(1, len(rows) // 3)
    return rows[:third] + rows[2 * third :]


def _mutant_duplicate_one(rows):
    return rows + [dict(rows[len(rows) // 2])]


def _mutant_alter_one_value(rows):
    out = [dict(r) for r in rows]
    out[len(out) // 2]["val"] = out[len(out) // 2]["val"] + 0.5
    return out


def _mutant_swap_two_values(rows):
    out = [dict(r) for r in rows]
    a, b = 1, len(out) - 2
    out[a]["val"], out[b]["val"] = out[b]["val"], out[a]["val"]
    return out


MUTANTS = {
    "drop_the_last_row": _mutant_drop_last,
    "drop_the_first_row": _mutant_drop_first,
    "skip_a_chunk": _mutant_skip_a_chunk,
    "store_one_row_twice": _mutant_duplicate_one,
    "change_one_value": _mutant_alter_one_value,
    "swap_two_values": _mutant_swap_two_values,
}

#: the durability mutant: the rows go into a transaction nobody commits, so they are visible in this session and gone after a reopen
NO_COMMIT = "leave_the_transaction_open"
