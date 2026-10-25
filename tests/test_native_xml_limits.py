"""The native build lifts the JDK's XML entity limits for one import, and only when asked (the JVM build does the same with JVM
system properties, tests/test_xml_limits.py)."""

import pytest

import arcadedb
from arcadedb.exceptions import ArcadeDBError


@pytest.fixture
def db(tmp_path):
    database = arcadedb.create_database(str(tmp_path / "xml_limits_db"))
    database.command("sql", "CREATE DOCUMENT TYPE Item")
    yield database
    if database.is_transaction_active():
        database.rollback()
    database.drop()


@pytest.fixture
def big_xml(tmp_path):
    """One text node of 1,000,000 characters (200,000 entity references)."""
    path = tmp_path / "big.xml"
    path.write_text(
        "<root><item id='0'><t>" + "&amp;" * 200_000 + "</t></item>"
        "<item id='1'><t>small</t></item></root>"
    )
    return path


def test_import_hitting_a_limit_names_the_option(db, big_xml):
    with pytest.raises(ArcadeDBError) as excinfo:
        db.import_documents(big_xml, document_type="Item", file_type="xml")
    assert "JAXP0001000" in str(excinfo.value)
    assert "disable_xml_limits=True" in str(excinfo.value)


def test_per_call_option_lifts_the_limits_for_that_call_only(db, big_xml):
    result = db.import_documents(
        big_xml, document_type="Item", file_type="xml", disable_xml_limits=True
    )
    assert result.result == "OK"
    assert result.get("parsedRecords") == 2
    # the limits are back after the call: the same file is refused again
    with pytest.raises(ArcadeDBError, match="JAXP0001000"):
        db.import_documents(big_xml, document_type="Item", file_type="xml")


def test_limits_are_restored_when_the_import_fails(db, big_xml, tmp_path):
    missing = tmp_path / "missing.xml"
    with pytest.raises(ArcadeDBError) as excinfo:
        db.import_documents(
            missing, document_type="Item", file_type="xml", disable_xml_limits=True
        )
    assert "disable_xml_limits=True" not in str(excinfo.value)
    # the failed lifted import put the limits back: the big file is refused again
    with pytest.raises(ArcadeDBError, match="JAXP0001000"):
        db.import_documents(big_xml, document_type="Item", file_type="xml")
