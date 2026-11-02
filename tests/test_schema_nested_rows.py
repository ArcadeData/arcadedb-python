"""#195: a query row nested in a value (the entries of `schema:types.properties`) reaches Python as a map on both builds.

The native codec wrote a nested Result through String.valueOf, so each entry arrived as a str and code that reads the schema
(examples/04_csv_import_documents.py) stopped with AttributeError on the native package only.
"""

import arcadedb


def _as_map(entry):
    # the JVM build hands back the engine's Result object, the native build a dict
    return dict(entry.toMap()) if hasattr(entry, "toMap") else entry


def test_schema_type_properties_are_maps(temp_db_path):
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE Movie")
        db.command("sql", "CREATE PROPERTY Movie.title STRING")
        db.command("sql", "CREATE PROPERTY Movie.year INTEGER")

        rows = db.query(
            "sql", "SELECT properties FROM schema:types WHERE name = 'Movie'"
        ).to_list()
        assert len(rows) == 1
        entries = list(rows[0].get("properties"))
        assert len(entries) == 2

        maps = [_as_map(e) for e in entries]
        assert all(isinstance(m, dict) for m in maps), [
            type(e).__name__ for e in entries
        ]
        by_name = {str(m["name"]): m for m in maps}
        assert set(by_name) == {"title", "year"}
        assert str(by_name["title"]["type"]).upper() == "STRING"
        assert str(by_name["year"]["type"]).upper() == "INTEGER"

        if arcadedb.__backend__ == "native":
            assert all(isinstance(e, dict) for e in entries), [
                type(e).__name__ for e in entries
            ]
