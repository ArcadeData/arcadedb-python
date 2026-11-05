"""Crossings of the JVM build's record and row readers (one JPype call is a few microseconds of dispatch).

A record of three properties took 18 reads of Java objects to become a dict, a row 14, a call for the
property names 12, and each row of a result set read with ``for row in rs`` two calls (``hasNext()``,
``next()``). The bridge's ``ReaderCalls`` writes a record or a row as the typed JSON of ``TypedRows`` and
hands over the next row in one call. Every test here pins two things: the number of calls into the JVM
and of reads of the arrays and sets they return (counted on a proxy of the Java object or bridge class,
the technique of ``test_percall_calls.py``), and that the answer is the one the per-property path gives
(same keys in the same order, same Python types and values, same errors, same result-set state), so a
count cannot be bought with a change of behaviour.
"""

import jpype
import pytest

from arcadedb import graph, results
from arcadedb.exceptions import ArcadeDBError
from arcadedb.type_conversion import convert_java_to_python

# fixtures and helpers of the typed-row tests: the wide table with every property type (fixture ``db``)
from tests.test_typed_rows import _in_memory, db, exact  # noqa: F401


class _CountedArray:
    """A Java array that logs every element read; elements that are arrays are counted too."""

    def __init__(self, target, log, label):
        self._target = target
        self._log = log
        self._label = label

    def __len__(self):
        return len(self._target)

    def __getitem__(self, index):
        label = f"{self._label}[{index}]"
        self._log.append(label)
        return _counted(self._target[index], self._log, label)

    def __iter__(self):
        for index in range(len(self._target)):
            yield self[index]


class _CountedSet:
    """A Java collection that logs every element read."""

    def __init__(self, target, log, label):
        self._target = target
        self._log = log
        self._label = label

    def __iter__(self):
        for item in self._target:
            self._log.append(f"{self._label}.next")
            yield item


def _counted(value, log, label):
    if isinstance(value, jpype.JArray):
        return _CountedArray(value, log, label)
    if label.endswith(".getPropertyNames"):
        return _CountedSet(value, log, label)
    return value


class Counting:
    """Counts the method calls made on a Java object or a Java class, and the reads of what they return.

    Arguments that are themselves a ``Counting`` are unwrapped before the call, so the proxy of a result set
    can be handed to a bridge class as it is. Arrays a call returns, and the set ``getPropertyNames()``
    returns, log each element read.
    """

    def __init__(self, target, log, label):
        self._target = target
        self._log = log
        self._label = label

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr
        label = f"{self._label}.{name}"

        def call(*args):
            self._log.append(label)
            args = [a._target if isinstance(a, Counting) else a for a in args]
            return _counted(attr(*args), self._log, label)

        return call


@pytest.fixture
def small_db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE T")
    with temp_db.transaction():
        for key in range(5):
            temp_db.command(
                "sql", "INSERT INTO T SET k = ?, v = ?, s = ?", key, key * 10, f"s{key}"
            )
    return temp_db


@pytest.fixture
def log(small_db, monkeypatch):
    """The list every counted call is appended to; the bridge classes are counted."""
    calls = []
    for name in ("ReaderCalls", "RowAccess"):
        real = results._bridge_class(name)
        assert real is not None, f"the bridge jar must carry {name}"
        monkeypatch.setitem(results._BRIDGE_CLASSES, name, Counting(real, calls, name))
    return calls


def _row(db, query="SELECT k, v, s FROM T WHERE k = 3"):
    return db.query("sql", query).first()


def _per_property(java_row):
    """A row read the way the bridge-less path reads it: one read per name and per value."""
    return {
        str(n): convert_java_to_python(java_row.getProperty(str(n)))
        for n in java_row.getPropertyNames()
    }


def _per_property_document(java_document):
    return {
        str(n): convert_java_to_python(java_document.get(str(n)))
        for n in java_document.getPropertyNames()
    }


# ---------------------------------------------------------------- Result.to_dict


def test_result_to_dict_is_one_call_and_one_read(small_db, log):
    row = _row(small_db)
    row._java_result = Counting(row._java_result, log, "row")
    log.clear()
    assert row.to_dict() == {"k": 3, "v": 30, "s": "s3"}
    assert log == ["ReaderCalls.rowJson", "ReaderCalls.rowJson[0]"]


def test_result_to_dict_costs_less_than_the_row_access_path(small_db, log, monkeypatch):
    row = _row(small_db)
    row._java_result = Counting(row._java_result, log, "row")
    log.clear()
    new = row.to_dict()
    new_log = list(log)

    monkeypatch.setitem(results._BRIDGE_CLASSES, "ReaderCalls", None)
    row = _row(small_db)
    row._java_result = Counting(row._java_result, log, "row")
    log.clear()
    old = row.to_dict()
    old_log = list(log)

    assert new == old
    assert list(new) == list(old)
    assert old_log[0] == "RowAccess.namesAndValues"
    assert len(new_log) == 2
    assert (
        len(old_log) >= 9
    )  # the call, then the names array and each name, the values array and each value
    assert len(new_log) < len(old_log)


def test_result_to_dict_without_any_bridge_class_is_the_same(small_db, monkeypatch):
    expected = _row(small_db).to_dict()
    monkeypatch.setitem(results._BRIDGE_CLASSES, "ReaderCalls", None)
    monkeypatch.setitem(results._BRIDGE_CLASSES, "RowAccess", None)
    row = _row(small_db)
    assert row.to_dict() == expected
    assert list(row.to_dict()) == list(expected)


def test_result_to_dict_without_conversion_is_unchanged(small_db):
    row = _row(small_db)
    raw = row.to_dict(convert_types=False)
    assert {str(k): int(v) if k != "s" else str(v) for k, v in raw.items()} == {
        "k": 3,
        "v": 30,
        "s": "s3",
    }


def test_every_property_type_reads_as_the_per_property_path_reads_it(db):
    seen = 0
    for row in db.query("sql", "SELECT FROM Wide"):
        want = _per_property(row._java_result)
        assert row.get_property_names() == list(
            want
        )  # before to_dict() has cached them
        got = row.to_dict()
        assert exact(got) == exact(want)
        assert list(got) == list(want)
        seen += 1
    assert seen == 7


@pytest.mark.parametrize(
    "query",
    [
        "SELECT FROM Wide LIMIT 1",
        "SELECT FROM VT",
        "SELECT count(*) AS c, list(s) AS ls, set(s) AS ss FROM Wide",
        "SELECT [1, 2] AS a, {'x': {'y': 1}} AS m, 1 / 3.0 AS q, date('2020-01-01') AS d, 2.5 AS dec",
        "SELECT s AS `\u0001`, i FROM Wide LIMIT 3",
    ],
)
def test_other_row_shapes_read_the_same(db, query):
    rows = 0
    for row in db.query("sql", query):
        want = _per_property(row._java_result)
        assert row.get_property_names() == list(want)
        assert exact(row.to_dict()) == exact(want)
        assert list(row.to_dict()) == list(want)
        rows += 1
    assert rows >= 1


def test_values_sql_cannot_produce_read_the_same(db):
    """Java values built in the test: instants, offsets, sets, a tag-named map key, a tag-named property."""
    time = jpype.JClass("java.time.Instant")
    zoned = jpype.JClass("java.time.ZonedDateTime")
    zone = jpype.JClass("java.time.ZoneId")
    big_integer = jpype.JClass("java.math.BigInteger")
    hash_map = jpype.JClass("java.util.HashMap")
    linked_set = jpype.JClass("java.util.LinkedHashSet")
    java_date = jpype.JClass("java.util.Date")
    instant = time.parse("2021-02-03T04:05:06.789123456Z")
    tag_key = hash_map()
    tag_key.put("\u0001", "tag")
    members = linked_set()
    members.add("a")
    members.add("b")
    rows = [
        {
            "instant": instant,
            "zoned": zoned.ofInstant(instant, zone.of("Asia/Seoul")),
            "big": big_integer("123456789012345678901234567890"),
            "util_date": java_date(1612325106789),
            "set": members,
            "tag_key": tag_key,
            "ints": jpype.JArray(jpype.JInt)([1, 2]),
            "floats": jpype.JArray(jpype.JFloat)([0.5, float("nan")]),
            "emoji": "x \U0001f600  ",
        },
        {"\u0001": 1, "other": 2},
    ]
    seen = 0
    for row in results.ResultSet(_in_memory(rows), db):
        want = _per_property(row._java_result)
        assert row.get_property_names() == list(want)
        got = row.to_dict()
        assert exact(got) == exact(want)
        assert list(got) == list(want)
        seen += 1
    assert seen == 2


def test_a_value_python_cannot_hold_raises_as_before(db):
    far_future = jpype.JClass("java.time.Instant").parse("+10000-01-01T00:00:00Z")
    with pytest.raises(OverflowError):
        [r.to_dict() for r in results.ResultSet(_in_memory([{"t": far_future}]), db)]


def test_an_object_array_raises_as_before(db):
    strings = jpype.JArray(jpype.JString)(["a", "b"])
    with pytest.raises(BufferError):
        [r.to_dict() for r in results.ResultSet(_in_memory([{"a": strings}]), db)]


def test_result_to_dict_remembers_the_names(small_db, log):
    row = _row(small_db)
    row._java_result = Counting(row._java_result, log, "row")
    log.clear()
    row.to_dict()
    log.clear()
    assert row.get_property_names() == ["k", "v", "s"]
    assert row.property_names == ["k", "v", "s"]
    assert log == []


# ---------------------------------------------------------------- Document.to_dict


def test_document_to_dict_is_one_call_and_one_read(small_db, log):
    record = _row(small_db, "SELECT FROM T WHERE k = 3").get_element()
    record._java_document = Counting(record._java_document, log, "doc")
    log.clear()
    assert record.to_dict() == {"k": 3, "v": 30, "s": "s3"}
    assert log == ["ReaderCalls.documentJson", "ReaderCalls.documentJson[0]"]


def test_document_to_dict_costs_less_than_the_per_property_path(
    small_db, log, monkeypatch
):
    record = _row(small_db, "SELECT FROM T WHERE k = 3").get_element()
    record._java_document = Counting(record._java_document, log, "doc")
    log.clear()
    new = record.to_dict()
    new_log = list(log)

    monkeypatch.setitem(results._BRIDGE_CLASSES, "ReaderCalls", None)
    record = _row(small_db, "SELECT FROM T WHERE k = 3").get_element()
    record._java_document = Counting(record._java_document, log, "doc")
    log.clear()
    old = record.to_dict()
    old_log = list(log)

    assert new == old
    assert list(new) == list(old)
    assert old_log.count("doc.get") == 3
    assert "doc.getPropertyNames" in old_log
    assert len(old_log) >= 7  # getPropertyNames, each name, one get per property
    assert len(new_log) == 2


def test_document_to_dict_without_the_bridge_class_is_the_same(small_db, monkeypatch):
    expected = _row(small_db, "SELECT FROM T WHERE k = 3").get_element().to_dict()
    monkeypatch.setitem(results._BRIDGE_CLASSES, "ReaderCalls", None)
    record = _row(small_db, "SELECT FROM T WHERE k = 3").get_element()
    assert record.to_dict() == expected
    assert sorted(record.get_property_names()) == ["k", "s", "v"]


def test_every_property_type_of_a_record_reads_as_the_per_property_path_reads_it(db):
    seen = 0
    for row in db.query("sql", "SELECT FROM Wide"):
        record = row.get_element()
        want = _per_property_document(record._java_document)
        assert record.get_property_names() == list(want)
        got = record.to_dict()
        assert exact(got) == exact(want)
        assert list(got) == list(want)
        seen += 1
    assert seen == 7
    vertices = [
        r.get_element() for r in db.query("sql", "SELECT FROM VT")
    ]  # a vertex wrapper reads the same way
    assert len(vertices) == 1
    assert isinstance(vertices[0], graph.Vertex)
    assert exact(vertices[0].to_dict()) == exact(
        _per_property_document(vertices[0]._java_document)
    )


def test_a_property_named_like_the_tag_reads_as_before(small_db):
    with small_db.transaction():
        doc = small_db.new_document("T")
        doc.set("\u0001", 1).set("other", 2)
        doc.save()
        rid = str(doc.get_identity())
    record = small_db.lookup_by_rid(rid)
    want = _per_property_document(record._java_document)
    assert want == {"\u0001": 1, "other": 2}
    assert record.get_property_names() == list(want)
    assert exact(record.to_dict()) == exact(want)


def test_a_record_read_again_after_a_set_has_the_new_property(small_db):
    with small_db.transaction():
        doc = small_db.new_document("T")
        doc.set("a", 1)
        assert doc.to_dict() == {"a": 1}
        doc.set("b", 2)
        assert doc.to_dict() == {"a": 1, "b": 2}
        assert sorted(doc.get_property_names()) == ["a", "b"]


def test_a_record_after_its_database_closed_still_raises(small_db):
    record = _row(small_db, "SELECT FROM T WHERE k = 1").get_element()
    row = _row(small_db, "SELECT FROM T WHERE k = 2")
    small_db.close()
    with pytest.raises(ArcadeDBError, match="Database is closed"):
        record.to_dict()
    with pytest.raises(ArcadeDBError, match="Database is closed"):
        record.get_property_names()
    with pytest.raises(ArcadeDBError, match="Database is closed"):
        row.to_dict()
    with pytest.raises(ArcadeDBError, match="Database is closed"):
        row.get_property_names()


# ---------------------------------------------------------------- property names


def test_row_property_names_are_one_call(small_db, log):
    row = _row(small_db)
    row._java_result = Counting(row._java_result, log, "row")
    log.clear()
    assert row.get_property_names() == ["k", "v", "s"]
    assert log == ["ReaderCalls.rowNames"]
    assert row.property_names == ["k", "v", "s"]
    assert log == ["ReaderCalls.rowNames"]  # cached


def test_document_property_names_are_one_call(small_db, log):
    record = _row(small_db, "SELECT FROM T WHERE k = 3").get_element()
    record._java_document = Counting(record._java_document, log, "doc")
    log.clear()
    assert sorted(record.get_property_names()) == ["k", "s", "v"]
    assert log == ["ReaderCalls.documentNames"]
    assert sorted(record.get_property_names()) == ["k", "s", "v"]
    assert log == ["ReaderCalls.documentNames"]  # cached


def test_property_names_cost_less_than_the_walk_over_the_java_set(
    small_db, log, monkeypatch
):
    monkeypatch.setitem(results._BRIDGE_CLASSES, "ReaderCalls", None)
    row = _row(small_db)
    row._java_result = Counting(row._java_result, log, "row")
    log.clear()
    assert row.get_property_names() == ["k", "v", "s"]
    # the call, then one read per name of the Java set
    assert log == [
        "row.getPropertyNames",
        "row.getPropertyNames.next",
        "row.getPropertyNames.next",
        "row.getPropertyNames.next",
    ]


UNUSUAL_NAMES = ['a"b', "c\\d", "sp ace", "é日本", "\U0001f600", "tab\tname"]


def test_unusual_property_names_come_back_as_the_engine_has_them(small_db):
    with small_db.transaction():
        doc = small_db.new_document("T")
        for index, name in enumerate(UNUSUAL_NAMES):
            doc.set(name, index)
        doc.save()
        rid = str(doc.get_identity())
    record = small_db.lookup_by_rid(rid)
    expected = [str(n) for n in record._java_document.getPropertyNames()]
    assert sorted(expected) == sorted(UNUSUAL_NAMES)
    assert record.get_property_names() == expected
    record = small_db.lookup_by_rid(rid)
    assert list(record.to_dict()) == expected
    row = small_db.query("sql", "SELECT FROM T WHERE k IS NULL").first()
    assert row.get_property_names() == [
        str(n) for n in row._java_result.getPropertyNames()
    ]
    assert list(row.to_dict()) == row.get_property_names()


# ---------------------------------------------------------------- ResultSet iteration


def test_iteration_is_one_call_per_row(small_db, log):
    rs = small_db.query("sql", "SELECT k FROM T ORDER BY k")
    rs._java_result_set = Counting(rs._java_result_set, log, "rs")
    log.clear()
    rows = list(rs)
    calls = list(log)
    assert [r.get("k") for r in rows] == [0, 1, 2, 3, 4]
    assert calls[:6] == ["ReaderCalls.nextOrNull"] * 6  # five rows and the end
    assert set(calls[6:]) <= {"rs.close"}


def test_iteration_costs_less_than_hasnext_and_next(small_db, log, monkeypatch):
    rs = small_db.query("sql", "SELECT k FROM T ORDER BY k")
    rs._java_result_set = Counting(rs._java_result_set, log, "rs")
    log.clear()
    new_rows = list(rs)
    new_calls = len([c for c in log if c != "rs.close"])

    monkeypatch.setitem(results._BRIDGE_CLASSES, "ReaderCalls", None)
    rs = small_db.query("sql", "SELECT k FROM T ORDER BY k")
    rs._java_result_set = Counting(rs._java_result_set, log, "rs")
    log.clear()
    old_rows = list(rs)
    old_calls = len([c for c in log if c != "rs.close"])

    expected = [{"k": k} for k in range(5)]
    assert [r.to_dict() for r in new_rows] == expected
    assert [r.to_dict() for r in old_rows] == expected
    assert old_calls == 11  # hasNext and next per row, and the hasNext that ends it
    assert new_calls == 6
    assert new_calls < old_calls


def test_a_row_is_read_when_it_is_asked_for(small_db, log):
    rs = small_db.query("sql", "SELECT k FROM T ORDER BY k")
    rs._java_result_set = Counting(rs._java_result_set, log, "rs")
    log.clear()
    first = next(rs)
    calls = list(log)
    assert first.get("k") == 0
    assert calls == ["ReaderCalls.nextOrNull"]  # no row taken ahead
    assert not rs._closed and not rs._exhausted
    assert [r.get("k") for r in rs] == [1, 2, 3, 4]
    assert rs._closed and rs._exhausted
    assert list(rs) == []


def test_iteration_without_the_bridge_class_is_the_same(small_db, monkeypatch):
    monkeypatch.setitem(results._BRIDGE_CLASSES, "ReaderCalls", None)
    rs = small_db.query("sql", "SELECT k FROM T ORDER BY k")
    assert [r.get("k") for r in rs] == [0, 1, 2, 3, 4]
    assert rs._closed and rs._exhausted
    assert list(rs) == []


def test_a_set_closed_before_its_end_still_raises(small_db):
    rs = small_db.query("sql", "SELECT k FROM T ORDER BY k")
    assert next(rs).get("k") == 0
    rs.close()
    with pytest.raises(ArcadeDBError, match="closed before all its rows"):
        next(rs)


def test_an_empty_set_ends_at_once(small_db, log):
    rs = small_db.query("sql", "SELECT k FROM T WHERE k = 99")
    rs._java_result_set = Counting(rs._java_result_set, log, "rs")
    log.clear()
    assert list(rs) == []
    assert log[0] == "ReaderCalls.nextOrNull"
    assert rs._exhausted and rs._closed
