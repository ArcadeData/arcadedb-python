"""The openCypher vector statements, as the section "openCypher: SEARCH and CREATE VECTOR INDEX" of the how-to page
``docs/modules/ROOT/pages/how-to/vectors.adoc`` says them (ArcadeData/arcadedb-python#267, row 7).

Every statement of that section is pinned here: what ``SEARCH`` returns and how its ``WHERE`` differs from a ``WHERE`` after it, the
forms of its query vector, the three mistakes it refuses, what ``CREATE VECTOR INDEX`` makes (the name, the type of the index and of
the property, the records it covers), the settings it takes and lists back, what a refused statement leaves behind, an index that
takes its dimensions from its first vector, and ``DROP INDEX``. The statements are the engine's (ArcadeDB 26.11.1: #10776, #10678,
#10893, #10915), and both builds run them in the same engine, so the tests use SQL, openCypher and the wrapper objects only, no Java
object, and run on the JVM build and the native build alike. The one difference between the builds is
``db.schema.get_vector_index`` for an index whose name is not ``Type[property]``: the native build returns ``None``
(ArcadeData/arcadedb-python#288), pinned by the last test as a strict ``xfail`` on the native build.
"""

import numpy as np
import pytest

import arcadedb
from arcadedb import ArcadeDBError

DOCS = [
    ("python", [1.0, 0.0, 0.0]),
    ("python-ish", [0.9, 0.1, 0.0]),
    ("graph", [0.0, 1.0, 0.0]),
    ("vector", [0.0, 0.0, 1.0]),
]
QUERY = [0.9, 0.1, 0.0]


@pytest.fixture
def db(temp_db):
    """The ``Doc`` type of the page: an index made by SQL, which the engine names ``Doc[embedding]``."""
    temp_db.command("sql", "CREATE VERTEX TYPE Doc")
    temp_db.command("sql", "CREATE PROPERTY Doc.name STRING")
    temp_db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    temp_db.command(
        "sql",
        "CREATE INDEX ON Doc (embedding) LSM_VECTOR "
        'METADATA {"dimensions": 3, "similarity": "COSINE"}',
    )
    with temp_db.transaction():
        for name, vector in DOCS:
            temp_db.command(
                "sql",
                "INSERT INTO Doc SET name = ?, embedding = ?",
                name,
                np.asarray(vector, dtype=np.float32),
            )
    return temp_db


def _search(db, vector, *, index="`Doc[embedding]`", inside="", k=2):
    statement = (
        "MATCH (d:Doc) "
        f"SEARCH d IN (VECTOR INDEX {index} FOR $vec {inside} LIMIT {k}) SCORE AS score "
        "RETURN d.name AS name, score ORDER BY score DESC"
    )
    return db.query("opencypher", statement, {"vec": vector}).to_list()


def _create_article_index(
    db, config="`vector.dimensions`: 3", name="article_embedding", label="Article"
):
    db.command(
        "opencypher",
        f"CREATE VECTOR INDEX {name} FOR (n:{label}) ON n.embedding "
        f"OPTIONS {{indexConfig: {{{config}}}}}",
    )


def _create_statement(db, name):
    row = db.query(
        "opencypher",
        "SHOW INDEXES YIELD name, createStatement WHERE name = $name",
        {"name": name},
    ).first()
    return row.get("createStatement")


def _listed(name, label, settings):
    return (
        f"CREATE VECTOR INDEX `{name}` FOR (n:`{label}`) ON n.`embedding` "
        f"OPTIONS {{indexConfig: {{{settings}}}}}"
    )


def _load_articles(db, label="Article"):
    with db.transaction():
        for title, vector in DOCS:
            db.command(
                "opencypher",
                f"CREATE (:{label} {{title: $title, embedding: $vec}})",
                {"title": title, "vec": vector},
            )


# --- SEARCH


def test_search_returns_the_k_nearest_with_the_scores_of_query_nodes(db):
    rows = _search(db, QUERY)
    assert [row["name"] for row in rows] == ["python-ish", "python"]
    scores = [row["score"] for row in rows]
    assert scores[0] > scores[1], "a higher score is closer"

    procedure = db.query(
        "opencypher",
        "CALL db.index.vector.queryNodes('Doc[embedding]', 2, $vec) "
        "YIELD node, score RETURN node.name AS name, score ORDER BY score DESC",
        {"vec": QUERY},
    ).to_list()
    assert [row["name"] for row in procedure] == ["python-ish", "python"]
    assert [row["score"] for row in procedure] == pytest.approx(scores)


@pytest.mark.parametrize(
    "vector",
    [
        QUERY,
        np.asarray(QUERY, dtype=np.float32),
        np.asarray(QUERY, dtype=np.float64),
    ],
    ids=["list", "float32-array", "float64-array"],
)
def test_the_query_vector_is_a_list_or_a_numpy_array(db, vector):
    assert [row["name"] for row in _search(db, vector)] == ["python-ish", "python"]


def test_limit_and_the_index_name_can_be_parameters(db):
    rows = db.query(
        "opencypher",
        "MATCH (d:Doc) SEARCH d IN (VECTOR INDEX $index FOR $vec LIMIT $k) "
        "SCORE AS score RETURN d.name AS name ORDER BY score DESC",
        {"index": "Doc[embedding]", "vec": QUERY, "k": 3},
    ).to_list()
    assert [row["name"] for row in rows] == ["python-ish", "python", "graph"]


def test_a_where_inside_the_search_runs_before_the_limit_and_one_after_it_runs_after(
    db,
):
    inside = _search(db, QUERY, inside="WHERE d.name <> 'python-ish'")
    assert [row["name"] for row in inside] == ["python", "graph"]

    after = db.query(
        "opencypher",
        "MATCH (d:Doc) "
        "SEARCH d IN (VECTOR INDEX `Doc[embedding]` FOR $vec LIMIT 2) "
        "WHERE d.name <> 'python-ish' RETURN d.name AS name",
        {"vec": QUERY},
    ).to_list()
    assert [row["name"] for row in after] == ["python"]


def test_search_mistakes_raise_arcadedb_error(db):
    bad_variable = (
        "MATCH (d:Doc) SEARCH x IN (VECTOR INDEX `Doc[embedding]` FOR $vec LIMIT 2) "
        "RETURN d.name AS name"
    )
    with pytest.raises(ArcadeDBError, match="'x' must be a node variable"):
        db.query("opencypher", bad_variable, {"vec": QUERY}).to_list()

    optional = (
        "OPTIONAL MATCH (d:Doc) "
        "SEARCH d IN (VECTOR INDEX `Doc[embedding]` FOR $vec LIMIT 2) "
        "RETURN d.name AS name"
    )
    with pytest.raises(ArcadeDBError, match="OPTIONAL MATCH"):
        db.query("opencypher", optional, {"vec": QUERY}).to_list()

    # no vector index covers Doc.name; the error surfaces while the rows are read
    unknown = (
        "MATCH (d:Doc) SEARCH d IN (VECTOR INDEX `Doc[name]` FOR $vec LIMIT 2) "
        "RETURN d.name AS name"
    )
    with pytest.raises(ArcadeDBError, match="No vector index found"):
        db.query("opencypher", unknown, {"vec": QUERY}).to_list()


# --- CREATE VECTOR INDEX


def test_create_vector_index_keeps_its_name_and_lists_its_statement(temp_db):
    _create_article_index(
        temp_db,
        "`vector.dimensions`: 3, `vector.similarity_function`: 'cosine'",
    )
    assert _create_statement(temp_db, "article_embedding") == _listed(
        "article_embedding",
        "Article",
        "`vector.dimensions`: 3, `vector.similarity_function`: 'cosine', "
        "`vector.quantization.type`: 'none'",
    )
    assert temp_db.schema.exists_index("article_embedding")


def test_the_index_is_found_by_its_name_by_search_and_by_query_nodes(temp_db):
    _create_article_index(temp_db)
    _load_articles(temp_db)

    rows = temp_db.query(
        "opencypher",
        "MATCH (n:Article) "
        "SEARCH n IN (VECTOR INDEX article_embedding FOR $vec LIMIT 2) SCORE AS score "
        "RETURN n.title AS title ORDER BY score DESC",
        {"vec": QUERY},
    ).to_list()
    assert [row["title"] for row in rows] == ["python-ish", "python"]

    nodes = temp_db.query(
        "opencypher",
        "CALL db.index.vector.queryNodes('article_embedding', 2, $vec) "
        "YIELD node, score RETURN node.title AS title ORDER BY score DESC",
        {"vec": QUERY},
    ).to_list()
    assert [row["title"] for row in nodes] == ["python-ish", "python"]


def test_the_label_becomes_a_vertex_type_and_the_property_is_declared(temp_db):
    assert not temp_db.schema.exists_type("Article")
    _create_article_index(temp_db)

    kind = temp_db.query("sql", "SELECT type FROM schema:types WHERE name = 'Article'")
    assert kind.first().get("type") == "vertex"
    properties = temp_db.query(
        "sql",
        "SELECT name, type FROM (SELECT expand(properties) FROM schema:types "
        "WHERE name = 'Article')",
    ).to_list()
    assert [(row["name"], str(row["type"])) for row in properties] == [
        ("embedding", "ARRAY_OF_FLOATS")
    ]


def test_the_index_covers_the_records_that_exist_and_those_written_later(temp_db):
    _load_articles(temp_db, "Note")  # the records first, with an undeclared property
    _create_article_index(temp_db, label="Note", name="note_embedding")

    def nearest(vector):
        return [
            row["title"]
            for row in temp_db.query(
                "opencypher",
                "MATCH (n:Note) SEARCH n IN (VECTOR INDEX note_embedding FOR $vec LIMIT 1) "
                "RETURN n.title AS title",
                {"vec": vector},
            ).to_list()
        ]

    assert nearest(QUERY) == ["python-ish"]
    with temp_db.transaction():
        temp_db.command(
            "opencypher",
            "CREATE (:Note {title: 'later', embedding: $vec})",
            {"vec": [0.0, 0.9, 0.1]},
        )
    assert nearest([0.0, 0.9, 0.1]) == ["later"]


def test_if_not_exists_makes_a_second_run_do_nothing_and_without_it_the_run_raises(
    temp_db,
):
    statement = (
        "CREATE VECTOR INDEX article_embedding IF NOT EXISTS FOR (n:Article) "
        "ON n.embedding OPTIONS {indexConfig: {`vector.dimensions`: 3}}"
    )
    temp_db.command("opencypher", statement)
    temp_db.command("opencypher", statement)
    listed = temp_db.query(
        "opencypher", "SHOW INDEXES YIELD name WHERE name = 'article_embedding'"
    ).to_list()
    assert len(listed) == 1

    with pytest.raises(ArcadeDBError, match="already exists"):
        _create_article_index(temp_db)


@pytest.mark.parametrize(
    "config, settings",
    [
        (
            "`vector.dimensions`: 3, `vector.similarity_function`: 'euclidean'",
            "`vector.dimensions`: 3, `vector.similarity_function`: 'euclidean', "
            "`vector.quantization.type`: 'none'",
        ),
        (
            "`vector.dimensions`: 3, `vector.quantization.type`: 'scalar'",
            "`vector.dimensions`: 3, `vector.similarity_function`: 'cosine', "
            "`vector.quantization.type`: 'scalar'",
        ),
        (
            "`vector.dimensions`: 3, `vector.quantization.type`: 'binary'",
            "`vector.dimensions`: 3, `vector.similarity_function`: 'cosine', "
            "`vector.quantization.type`: 'binary'",
        ),
        (
            "`vector.dimensions`: 3, efSearch: 64",
            "`vector.dimensions`: 3, `vector.similarity_function`: 'cosine', "
            "`vector.quantization.type`: 'none', `efSearch`: 64",
        ),
    ],
    ids=["euclidean", "scalar", "binary", "efSearch-by-its-own-name"],
)
def test_settings_are_listed_as_given_and_the_default_quantization_is_none(
    temp_db, config, settings
):
    _create_article_index(temp_db, config)
    assert _create_statement(temp_db, "article_embedding") == _listed(
        "article_embedding", "Article", settings
    )


def test_the_listed_statement_makes_the_same_index_again(temp_db):
    _create_article_index(
        temp_db,
        "`vector.dimensions`: 3, `vector.similarity_function`: 'euclidean', efSearch: 64",
    )
    listed = _create_statement(temp_db, "article_embedding")
    temp_db.command("opencypher", "DROP INDEX article_embedding")
    temp_db.command("opencypher", listed)
    assert _create_statement(temp_db, "article_embedding") == listed


@pytest.mark.parametrize(
    "config, message",
    [
        ("`vector.dimensions`: 3, efSerch: 64", "efSerch"),
        ("`vector.dimensions`: 3, dimensions: 3", "vector.dimensions"),
        (
            "`vector.dimensions`: 3, `vector.hnsw.m`: 8, maxConnections: 16",
            "maxConnections",
        ),
    ],
    ids=["unknown-name", "dimensions-twice", "degree-twice"],
)
def test_a_refused_statement_creates_no_index_and_no_type(temp_db, config, message):
    with pytest.raises(ArcadeDBError, match=message):
        _create_article_index(temp_db, config, name="refused", label="Refused")
    assert not temp_db.schema.exists_index("refused")
    assert not temp_db.schema.exists_type("Refused")


def test_dimensions_may_be_left_out_and_the_first_vector_decides(temp_db):
    temp_db.command(
        "opencypher",
        "CREATE VECTOR INDEX article_embedding FOR (n:Article) ON n.embedding",
    )
    assert "vector.dimensions" not in _create_statement(temp_db, "article_embedding")

    with temp_db.transaction():
        temp_db.command(
            "opencypher",
            "CREATE (:Article {title: 'a', embedding: $vec})",
            {"vec": [1.0, 0.0, 0.0]},
        )
    assert "`vector.dimensions`: 3" in _create_statement(temp_db, "article_embedding")

    rows = temp_db.query(
        "opencypher",
        "MATCH (n:Article) SEARCH n IN (VECTOR INDEX article_embedding FOR $vec LIMIT 1) "
        "RETURN n.title AS title",
        {"vec": [1.0, 0.1, 0.0]},
    ).to_list()
    assert [row["title"] for row in rows] == ["a"]

    with pytest.raises(ArcadeDBError, match="does not match index dimension 3"):
        temp_db.command(
            "opencypher",
            "CREATE (:Article {title: 'b', embedding: $vec})",
            {"vec": [1.0, 0.0]},
        )


def test_drop_index_drops_it_and_if_exists_does_not_raise_when_it_is_gone(temp_db):
    _create_article_index(temp_db)
    temp_db.command("opencypher", "DROP INDEX article_embedding")
    assert not temp_db.schema.exists_index("article_embedding")

    with pytest.raises(ArcadeDBError, match="does not exist"):
        temp_db.command("opencypher", "DROP INDEX article_embedding")
    temp_db.command("opencypher", "DROP INDEX article_embedding IF EXISTS")


# --- the Python VectorIndex object


@pytest.mark.xfail(
    arcadedb.__backend__ == "native",
    strict=True,
    reason="ArcadeData/arcadedb-python#288: the native get_vector_index finds only an index named Type[property]",
)
def test_get_vector_index_returns_an_index_made_by_create_vector_index(temp_db):
    _create_article_index(temp_db)
    _load_articles(temp_db)

    index = temp_db.schema.get_vector_index("Article", "embedding")
    assert index is not None
    meta = index.get_metadata()
    assert meta["index_name"] == "article_embedding"
    assert meta["dimensions"] == 3
    assert meta["quantization"] == "NONE"
    hits = index.find_nearest(np.asarray(QUERY, dtype=np.float32), k=2)
    assert [record.get("title") for record, _ in hits] == ["python-ish", "python"]
