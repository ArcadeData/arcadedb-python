"""GraphBatch.create_vertices and new_edges (the JVM wheel): rows of exact scalars and float vectors cross as typed columns, not as JSON text (#388).

Before, every chunk of rows crossed as one JSON text that the engine parsed into a JSONObject per row, and the created RIDs came back as one
';'-joined text. Now rows that are all dicts with the same keys in the same order, each property one kind of int, float, bool or str (None
for a null), and the float vectors of a declared or float32 property, cross as one typed column per property
(``VertexBatcher.createVerticesColumns``, ``EdgeBatcher.newEdgesColumns``), and the RIDs of the vertices come back as one ``long[]`` of
bucket id and position pairs. Every other shape keeps the JSON text path (or the property-matrix path), which these tests also pin.

The parity tests store the same rows twice, once through the typed route and once with it switched off (``_scalar_columns`` answering None,
which is what every other shape gets), and compare what the engine holds: the property names in order, the Java class of each value and
its text. The JPype crossings are counted by wrapping the bridge classes. The JPype bridge does not exist on the native wheel, so
conftest.py does not collect this file there.
"""

import jpype
import numpy as np
import pytest

from arcadedb import graph_batch as gbm
from arcadedb import results as results_module
from arcadedb.graph_batch import GraphBatch

CHUNK = 100  # rows per crossing in these tests: 250 rows are three crossings


class _Counting:
    """Forwards every call of a Java class and records its name."""

    def __init__(self, target, log):
        self._target, self._log = target, log

    def __getattr__(self, name):
        attribute = getattr(self._target, name)
        if not callable(attribute):
            return attribute

        def call(*args):
            self._log.append(name)
            return attribute(*args)

        return call


@pytest.fixture
def bridge(monkeypatch):
    """The bridge calls of a load, by class, and the property-matrix fallbacks."""
    log = {"vertex": [], "edge": [], "matrix": []}
    real_jclass = jpype.JClass

    def jclass(name, *args, **kwargs):
        cls = real_jclass(name, *args, **kwargs)
        if name == "com.arcadedb.python.VertexBatcher":
            return _Counting(cls, log["vertex"])
        return cls

    monkeypatch.setattr(jpype, "JClass", jclass)
    real_bridge_class = results_module._bridge_class

    def bridge_class(name):
        cls = real_bridge_class(name)
        if name == "EdgeBatcher" and cls is not None:
            return _Counting(cls, log["edge"])
        return cls

    monkeypatch.setattr(results_module, "_bridge_class", bridge_class)
    real_matrix = GraphBatch._to_java_property_matrix

    def matrix(self, rows):
        log["matrix"].append(len(rows))
        return real_matrix(self, rows)

    monkeypatch.setattr(GraphBatch, "_to_java_property_matrix", matrix)
    monkeypatch.setattr(GraphBatch, "_BULK_CHUNK", CHUNK)
    return log


@pytest.fixture
def db(temp_db):
    for name in ("A", "B", "VA", "VB", "VC"):
        temp_db.command("sql", f"CREATE VERTEX TYPE {name}")
    for name in ("EA", "EB"):
        temp_db.command("sql", f"CREATE EDGE TYPE {name}")
    for name in ("A", "B"):
        temp_db.command("sql", f"CREATE PROPERTY {name}.embedding ARRAY_OF_FLOATS")
    return temp_db


def _plain(value):
    """A stored value as (Java class, text), or the list of the numbers of a float[]; None stays None."""
    if value is None:
        return None
    get_class = getattr(value, "getClass", None)
    if get_class is None:  # JPype handed over a Python value
        return (type(value).__name__, repr(value))
    name = str(get_class().getName())
    if name.startswith("["):
        return (name, [float(x) for x in value])
    return (name, str(value))


def _stored(db, type_name):
    """Every record of the type as its (property name, value) pairs in the order the engine holds them, sorted by id."""
    iterator = db._java_db.iterateType(type_name, False)
    rows = []
    while iterator.hasNext():
        record = iterator.next().asDocument()
        rows.append(
            [(str(n), _plain(record.get(n))) for n in record.getPropertyNames()]
        )
    rows.sort(key=lambda pairs: int(dict(pairs).get("id", (None, "-1"))[1]))
    return rows


INTS = [
    0,
    1,
    -1,
    2**31 - 1,
    2**31,
    -(2**31),
    -(2**31) - 1,
    2**53 + 1,
    2**63 - 1,
    -(2**63),
]
_RNG = np.random.default_rng(11)
FLOATS = (
    [
        0.0,
        -0.0,
        1.5,
        0.1,
        1 / 3,
        1e300,
        -1e-300,
        2.2250738585072014e-308,
        1e16,
        123456789.123456789,
    ]
    + [float(_RNG.uniform(-1e6, 1e6)) for _ in range(10)]
    + [float(_RNG.random() ** 7) for _ in range(10)]
)
STRS = [
    "",
    "a",
    "héllo",
    "\U0001f600 emoji",
    "x" * 1000,
    'quote"s \\ back',
    "nul\x00mid",
    "line\nbreak",
]


def _scalar_rows(n=40):
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": i,
                "i": INTS[i % len(INTS)],
                "f": FLOATS[i % len(FLOATS)],
                "s": STRS[i % len(STRS)],
                "b": i % 3 == 0,
                "ni": None if i % 4 == 0 else i * 7,
                "nf": None if i % 5 == 0 else i / 8,
                "ns": None if i % 6 == 0 else f"v{i}",
                "nb": None if i % 7 == 0 else i % 2 == 0,
                "none": None,
            }
        )
    return rows


def _load(db, type_name, rows, **options):
    with db.graph_batch(parallel_flush=False, **options) as gb:
        return gb.create_vertices(type_name, rows)


# ---------------------------------------------------------------------------
# crossings: no JSON text for the shapes the typed route carries


def test_scalar_rows_cross_as_typed_columns_one_call_per_chunk(db, bridge):
    rows = [
        {"id": i, "name": f"n{i}", "age": i % 90, "score": i / 4} for i in range(250)
    ]
    rids = _load(db, "A", rows)
    assert len(rids) == 250
    assert bridge["vertex"] == ["createVerticesColumns"] * 3  # 100, 100, 50
    assert bridge["matrix"] == []


def test_scalar_rows_with_a_vector_cross_as_typed_columns(db, bridge):
    x = np.random.default_rng(3).standard_normal((250, 8)).astype(np.float32)
    rows = [
        {"id": i, "views": i % 9, "embedding": jpype.JArray(jpype.JFloat)(x[i])}
        for i in range(250)
    ]
    assert len(_load(db, "A", rows)) == 250
    assert bridge["vertex"] == ["createVerticesColumns"] * 3
    assert bridge["matrix"] == []


def test_an_empty_batch_and_a_batch_of_one(db, bridge):
    assert _load(db, "A", []) == []
    assert bridge["vertex"] == []
    one = _load(db, "A", [{"id": 1, "x": "y"}])
    assert len(one) == 1 and one[0].startswith("#")
    assert bridge["vertex"] == ["createVerticesColumns"]


def test_rows_without_properties_cross_as_zero_columns(db, bridge):
    rids = _load(db, "A", [{} for _ in range(7)])
    assert len(rids) == 7 and len(set(rids)) == 7
    assert bridge["vertex"] == ["createVerticesColumns"]


# ---------------------------------------------------------------------------
# the RIDs that come back


def test_the_rids_are_the_texts_in_row_order(db, bridge):
    rows = [{"id": i} for i in range(250)]
    rids = _load(db, "A", rows)
    assert bridge["vertex"] == ["createVerticesColumns"] * 3
    assert all(type(rid) is str and rid.startswith("#") for rid in rids)
    assert len(set(rids)) == 250
    # the engine's own lookup of each returned text: row i must be the vertex that was created from rows[i]
    rid_class = jpype.JClass("com.arcadedb.database.RID")
    ids = [
        int(str(db._java_db.lookupByRID(rid_class(rid), True).asDocument().get("id")))
        for rid in rids
    ]
    assert ids == list(range(250))


def test_the_rids_equal_those_of_the_old_route(db, bridge, monkeypatch):
    rows = [{"id": i, "s": f"v{i}"} for i in range(250)]
    typed = _load(db, "A", rows)
    monkeypatch.setattr(gbm, "_scalar_columns", lambda rows, reserved=(): None)
    old = _load(db, "B", rows)
    assert (
        bridge["vertex"] == ["createVerticesColumns"] * 3 + ["createVerticesJson"] * 3
    )
    assert len(typed) == len(old) == 250 and len(set(typed)) == len(set(old)) == 250
    assert all(type(rid) is str and rid.startswith("#") for rid in typed + old)


# ---------------------------------------------------------------------------
# parity: what the engine holds is what the old route stored


def test_every_scalar_kind_is_stored_as_the_old_route_stores_it(
    db, bridge, monkeypatch
):
    rows = _scalar_rows()
    _load(db, "A", rows)
    assert bridge["vertex"] == ["createVerticesColumns"]
    monkeypatch.setattr(gbm, "_scalar_columns", lambda rows, reserved=(): None)
    _load(db, "B", rows)
    assert bridge["vertex"] == ["createVerticesColumns", "createVerticesJson"]
    typed, old = _stored(db, "A"), _stored(db, "B")
    assert len(typed) == len(old) == len(rows)
    assert typed == old


def test_a_float_vector_among_scalars_is_stored_as_the_old_route_stores_it(
    db, bridge, monkeypatch
):
    x = np.random.default_rng(5).standard_normal((60, 8)).astype(np.float32)
    rows = [
        {
            "id": i,
            "views": i % 11,
            "embedding": jpype.JArray(jpype.JFloat)(x[i]),
            "note": None if i % 3 else "n",
        }
        for i in range(60)
    ]
    _load(db, "A", rows)
    monkeypatch.setattr(gbm, "_scalar_columns", lambda rows, reserved=(): None)
    _load(db, "B", rows)
    assert bridge["vertex"] == ["createVerticesColumns", "createVerticesJson"]
    typed, old = _stored(db, "A"), _stored(db, "B")
    assert len(typed) == 60 and typed == old
    assert dict(typed[3])["embedding"][1] == [float(v) for v in x[3]]


def test_a_batch_of_one_and_a_load_over_the_chunk_boundary_agree(
    db, bridge, monkeypatch
):
    rows = _scalar_rows(250)
    _load(db, "A", rows)
    _load(db, "VA", rows[:1])
    monkeypatch.setattr(gbm, "_scalar_columns", lambda rows, reserved=(): None)
    _load(db, "B", rows)
    _load(db, "VB", rows[:1])
    assert _stored(db, "A") == _stored(db, "B")
    assert _stored(db, "VA") == _stored(db, "VB")


def test_a_null_and_a_missing_key_stay_apart(db, bridge):
    # rows with other keys keep the JSON text path: a missing key is no property, a None is a null, on both routes
    rows = [{"id": 0, "x": None}, {"id": 1}, {"id": 2, "x": 5}]
    _load(db, "A", rows)
    assert bridge["vertex"] == ["createVerticesJson"]
    stored = _stored(db, "A")
    assert len(stored) == 3
    assert "x" not in dict(stored[1])
    assert dict(stored[2])["x"][1] == "5"


# ---------------------------------------------------------------------------
# the shapes the typed route keeps off, and where they go instead


@pytest.mark.parametrize(
    "rows, route",
    [
        ([{"id": 0, "a": 1}, {"id": 1}], "createVerticesJson"),  # rows with other keys
        (
            [{"id": 0, "a": 1, "b": 2}, {"id": 1, "b": 2, "a": 1}],
            "createVerticesJson",
        ),  # another key order
        ([{"id": 0}, None], "createVerticesJson"),  # a row that is None
        ([{"id": i, "l": ["a", i]} for i in range(3)], "createVerticesJson"),  # a list
        ([{"id": i, "m": {"k": i}} for i in range(3)], "createVerticesJson"),  # a dict
        (
            [{"id": 0, "x": 1}, {"id": 1, "x": 2.5}],
            "createVerticesJson",
        ),  # ints beside floats
        (
            [{"id": 0, "x": 1}, {"id": 1, "x": True}],
            "createVerticesJson",
        ),  # an int beside a bool
        (
            [{"id": 0, "x": np.int64(4)}],
            None,
        ),  # a numpy scalar: the property-matrix path
        ([{"id": 0, "x": float("nan")}], None),  # a NaN
        ([{"id": 0, "x": 5e-324}], None),  # a subnormal
    ],
)
def test_a_shape_the_typed_route_cannot_carry_keeps_the_old_routes(
    db, bridge, rows, route
):
    rids = _load(db, "A", rows)
    assert len(rids) == len(rows)
    assert "createVerticesColumns" not in bridge["vertex"]
    if route is None:
        assert bridge["vertex"] == [] and bridge["matrix"] == [len(rows)]
    else:
        assert bridge["vertex"] == [route]


def test_a_chunk_that_is_not_carried_does_not_move_the_others(db, bridge):
    rows = [{"id": i, "v": i} for i in range(250)]
    rows[120] = {
        "id": 120,
        "v": 2.5,
    }  # chunk 2 holds an int column with a float: that chunk keeps the text
    rids = _load(db, "A", rows)
    assert len(rids) == 250
    assert bridge["vertex"] == [
        "createVerticesColumns",
        "createVerticesJson",
        "createVerticesColumns",
    ]
    stored = {r["id"]: r["v"] for r in db.query("sql", "SELECT id, v FROM A").to_list()}
    assert stored[119] == 119 and stored[120] == 2.5 and stored[121] == 121


# ---------------------------------------------------------------------------
# edges


def _edge_setup(db, n=250):
    vertices = _load(db, "VC", [{"id": i} for i in range(n)])
    return vertices


def _edge_props(n):
    return [
        {
            "id": i,
            "w": INTS[i % len(INTS)],
            "f": FLOATS[i % len(FLOATS)],
            "s": STRS[i % len(STRS)],
            "b": i % 3 == 0,
            "n": None if i % 4 == 0 else i,
            "z": None,
        }
        for i in range(n)
    ]


def _new_edges(db, edge_type, sources, destinations, properties=None):
    with db.graph_batch(parallel_flush=False) as gb:
        gb.new_edges(sources, edge_type, destinations, properties)


def test_edge_properties_of_scalars_cross_as_typed_columns(db, bridge):
    vertices = _edge_setup(db)
    del bridge["vertex"][:]
    sources, destinations = vertices, vertices[1:] + vertices[:1]
    _new_edges(db, "EA", sources, destinations, _edge_props(250))
    assert bridge["edge"] == ["newEdgesColumns"] * 3  # 100, 100, 50
    assert len(_stored(db, "EA")) == 250


def test_edges_without_a_property_value_in_any_dict_use_the_joined_call(db, bridge):
    vertices = _edge_setup(db, 12)
    _new_edges(db, "EA", vertices, vertices[1:] + vertices[:1], [{} for _ in vertices])
    _new_edges(
        db, "EB", vertices, vertices[1:] + vertices[:1], [None for _ in vertices]
    )
    assert bridge["edge"] == ["newEdgesJoined", "newEdgesJoined"]
    assert db.query("sql", "SELECT count(*) AS n FROM EA").to_list()[0]["n"] == 12
    assert db.query("sql", "SELECT count(*) AS n FROM EB").to_list()[0]["n"] == 12


def test_edge_properties_are_stored_as_the_old_route_stores_them(
    db, bridge, monkeypatch
):
    vertices = _edge_setup(db, 250)
    sources, destinations = vertices, vertices[1:] + vertices[:1]
    properties = _edge_props(250)
    _new_edges(db, "EA", sources, destinations, properties)
    monkeypatch.setattr(gbm, "_scalar_columns", lambda rows, reserved=(): None)
    _new_edges(db, "EB", sources, destinations, properties)
    assert bridge["edge"] == ["newEdgesColumns"] * 3 + ["newEdgesJson"] * 3
    typed, old = _stored(db, "EA"), _stored(db, "EB")
    assert len(typed) == len(old) == 250
    assert typed == old


def test_a_batch_of_one_edge(db, bridge, monkeypatch):
    vertices = _edge_setup(db, 2)
    _new_edges(db, "EA", vertices[:1], vertices[1:], [{"id": 1, "w": 5}])
    monkeypatch.setattr(gbm, "_scalar_columns", lambda rows, reserved=(): None)
    _new_edges(db, "EB", vertices[:1], vertices[1:], [{"id": 1, "w": 5}])
    assert bridge["edge"] == ["newEdgesColumns", "newEdgesJson"]
    assert _stored(db, "EA") == _stored(db, "EB") and len(_stored(db, "EA")) == 1


@pytest.mark.parametrize(
    "properties",
    [
        [{"id": 0, "a": 1}, {"id": 1}, {"id": 2, "a": 3}],  # dicts with other keys
        [{"id": 0}, None, {"id": 2}],  # a None among dicts
        [{"id": i, "l": ["a", i]} for i in range(3)],  # a list
        [
            {"id": 0, "x": 1},
            {"id": 1, "x": 2.5},
            {"id": 2, "x": 3},
        ],  # ints beside floats
        [
            {"id": i, "_src": "kept"} for i in range(3)
        ],  # a property named like a row's endpoint
    ],
)
def test_edge_shapes_the_typed_route_cannot_carry_keep_the_old_routes(
    db, bridge, properties
):
    vertices = _edge_setup(db, 3)
    _new_edges(db, "EA", vertices, vertices[1:] + vertices[:1], properties)
    assert "newEdgesColumns" not in bridge["edge"]
    assert len(_stored(db, "EA")) == 3


def test_an_edge_chunk_that_is_not_carried_sends_the_whole_call_the_old_way(db, bridge):
    vertices = _edge_setup(db, 250)
    properties = [{"id": i, "v": i} for i in range(250)]
    properties[120] = {"id": 120, "v": 2.5}
    _new_edges(db, "EA", vertices, vertices[1:] + vertices[:1], properties)
    assert "newEdgesColumns" not in bridge["edge"]
    assert len(_stored(db, "EA")) == 250
