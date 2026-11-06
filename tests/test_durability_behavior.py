"""The durability settings, as the how-to page ``docs/modules/ROOT/pages/how-to/durability.adoc`` says them (ArcadeData/arcadedb-python#267, row 10).

Each statement of that page is pinned here first: what ``set_wal_flush`` accepts and where it applies, what the database reports for
``arcadedb.txWalFlush``, ``arcadedb.txWalGroupCommit`` and ``arcadedb.fileSync``, how ``ALTER DATABASE`` treats them, and how the
async executor and ``GraphBatch`` keep durability settings of their own and put them back. The tests read the settings through
``SELECT settings FROM schema:database`` and the async executor's getters, which both builds have, so they run on the JVM build and
the native build alike. The few tests that read the engine's effective flush type of a thread need the Java ``Database`` and are
listed in ``conftest.NATIVE_NA_JVM_TESTS``.

What a crash loses at each level cannot be tested here, because a test process cannot lose power. The page states it only as what
the engine documents, with the source of each statement (``GlobalConfiguration.java`` and ``WALFile.java`` of engine 26.11.1).
"""

import threading

import pytest

from arcadedb import ArcadeDBError, create_database, open_database

FLUSH = "arcadedb.txWalFlush"
GROUP = "arcadedb.txWalGroupCommit"
FILE_SYNC = "arcadedb.fileSync"


def _settings(db):
    row = db.query("sql", "SELECT settings FROM schema:database").first()
    return {s["key"]: s for s in row.get("settings")}


def _alter(db, key, value):
    db.command("sql", f"ALTER DATABASE `{key}` {value}")


def _text(value):
    return str(value).lower()


# --- set_wal_flush


@pytest.mark.parametrize("mode", ["no", "yes_nometadata", "yes_full"])
def test_set_wal_flush_takes_these_three_modes(temp_db, mode):
    temp_db.set_wal_flush(mode)
    temp_db.set_wal_flush("no")


@pytest.mark.parametrize("mode", ["fsync", "YES_FULL", "", "1", 1])
def test_set_wal_flush_refuses_any_other_mode_with_the_list(temp_db, mode):
    with pytest.raises(ValueError) as info:
        temp_db.set_wal_flush(mode)
    assert (
        f"Invalid WAL flush mode: {mode}. "
        "Must be one of: ['no', 'yes_nometadata', 'yes_full']"
    ) == str(info.value)


def test_set_wal_flush_on_a_closed_database_raises_arcadedb_error(temp_db_path):
    db = create_database(temp_db_path)
    db.close()
    with pytest.raises(ArcadeDBError, match="closed"):
        db.set_wal_flush("no")


def test_set_wal_flush_is_not_a_stored_setting(temp_db):
    """It changes the open Database object (LocalDatabase.walFlushSetting, a field), not the saved configuration."""
    before = _settings(temp_db)[FLUSH]["value"]
    for mode in ("yes_nometadata", "yes_full", "no"):
        temp_db.set_wal_flush(mode)
        assert _settings(temp_db)[FLUSH]["value"] == before


def test_set_wal_flush_does_not_change_the_executors_own_flush(temp_db):
    """The async writers stamp their own flush on their transactions (engine default "no")."""
    executor = temp_db.async_executor()
    assert executor.get_transaction_sync() == "no"
    temp_db.set_wal_flush("yes_full")
    try:
        assert executor.get_transaction_sync() == "no"
    finally:
        temp_db.set_wal_flush("no")


def test_set_wal_flush_reaches_the_commits_of_this_database_only(temp_db, temp_db_path):
    """JVM only: the flush type a thread's next transaction will use, read from the Java database.

    The process default is not always "NO" (a production-mode server anywhere in the process sets it to 1), so the second database
    is compared with itself before and after, not with a constant.
    """
    other_path = temp_db_path + "_other"
    with create_database(other_path) as other:

        def flush_of(db):
            with db.transaction():
                return str(db._java_db.getTransaction().getWALFlush())

        other_before = flush_of(other)
        chosen = "YES_FULL" if flush_of(temp_db) != "YES_FULL" else "NO"
        temp_db.set_wal_flush({"YES_FULL": "yes_full", "NO": "no"}[chosen])
        try:
            assert flush_of(temp_db) == chosen
            assert flush_of(other) == other_before
        finally:
            temp_db.set_wal_flush("no")


def test_set_wal_flush_ends_with_the_open_database(temp_db_path):
    """JVM only: a reopened database follows the configuration again, not the earlier set_wal_flush."""

    def flush_of(db):
        with db.transaction():
            return str(db._java_db.getTransaction().getWALFlush())

    with create_database(temp_db_path) as db:
        configured = flush_of(db)
        chosen = "YES_FULL" if configured != "YES_FULL" else "NO"
        db.set_wal_flush({"YES_FULL": "yes_full", "NO": "no"}[chosen])
        assert flush_of(db) == chosen

    with open_database(temp_db_path) as db:
        assert flush_of(db) == configured


# --- the engine settings, read and written through SQL


def test_the_database_reports_the_defaults_of_the_settings_it_has(temp_db):
    settings = _settings(temp_db)
    assert str(settings[FLUSH]["default"]) == "0"
    assert _text(settings[GROUP]["default"]) == "true"


def test_file_sync_is_not_a_database_setting_and_alter_database_refuses_it(temp_db):
    """arcadedb.fileSync has JVM scope: it is not in the database's list and ALTER DATABASE does not take it."""
    assert FILE_SYNC not in _settings(temp_db)
    assert FLUSH in _settings(temp_db) and GROUP in _settings(temp_db)
    with pytest.raises(ArcadeDBError) as info:
        _alter(temp_db, FILE_SYNC, "false")
    assert "cannot be set per database" in str(info.value)
    assert "JVM scope" in str(info.value)


def test_alter_database_stores_the_flush_level_and_it_survives_a_reopen(temp_db_path):
    with create_database(temp_db_path) as db:
        _alter(db, FLUSH, 2)
        assert str(_settings(db)[FLUSH]["value"]) == "2"
        _alter(db, FLUSH, 1)
        assert str(_settings(db)[FLUSH]["value"]) == "1"
        db.command("sql", "CREATE DOCUMENT TYPE Note")
        with db.transaction():
            db.command("sql", "INSERT INTO Note SET text = 'kept'")

    with open_database(temp_db_path) as db:
        assert str(_settings(db)[FLUSH]["value"]) == "1"
        assert db.query("sql", "SELECT text FROM Note").first().get("text") == "kept"


def test_alter_database_stores_group_commit(temp_db):
    _alter(temp_db, GROUP, "false")
    assert _text(_settings(temp_db)[GROUP]["value"]) == "false"
    _alter(temp_db, GROUP, "true")
    assert _text(_settings(temp_db)[GROUP]["value"]) == "true"


def test_a_thread_that_starts_after_alter_database_commits_with_the_stored_level(
    temp_db_path,
):
    """JVM only: ALTER DATABASE writes the stored level, and a thread that has no transaction context yet takes it from there.

    This does not say what happens to the thread that ran the ALTER DATABASE itself: its context exists already and the engine
    reads the configured level when a context is created (TransactionContext.configuredWALFlush). The page does not claim it.
    """
    with create_database(temp_db_path) as db:
        _alter(db, FLUSH, 2)

        seen = {}

        def read():
            with db.transaction():
                seen["flush"] = str(db._java_db.getTransaction().getWALFlush())

        thread = threading.Thread(target=read)
        thread.start()
        thread.join()
        assert seen["flush"] == "YES_FULL"


# --- the async executor and GraphBatch keep settings of their own


def test_a_fresh_executor_writes_to_the_wal_and_does_not_flush(temp_db):
    executor = temp_db.async_executor()
    assert executor.is_transaction_use_wal() is True
    assert executor.get_transaction_sync() == "no"


def test_alter_database_does_not_change_the_executors_flush(temp_db):
    executor = temp_db.async_executor()
    _alter(temp_db, FLUSH, 2)
    assert executor.get_transaction_sync() == "no"


def test_a_graph_batch_relaxes_the_executors_durability_and_puts_it_back(temp_db):
    """A batch builds with use_wal=False and wal_flush="no"; parallel_flush (the default) applies them to the shared async executor."""
    temp_db.command("sql", "CREATE VERTEX TYPE City")
    executor = temp_db.async_executor()
    executor.set_transaction_use_wal(True)
    executor.set_transaction_sync("yes_full")

    with temp_db.graph_batch() as batch:
        assert executor.is_transaction_use_wal() is False
        assert executor.get_transaction_sync() == "no"
        batch.create_vertices("City", 2)

    assert executor.is_transaction_use_wal() is True
    assert executor.get_transaction_sync() == "yes_full"


def test_a_graph_batch_takes_the_durability_you_ask_for_and_puts_it_back(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE City")
    executor = temp_db.async_executor()
    executor.set_transaction_use_wal(True)
    executor.set_transaction_sync("no")

    with temp_db.graph_batch(use_wal=True, wal_flush="yes_full") as batch:
        assert executor.is_transaction_use_wal() is True
        assert executor.get_transaction_sync() == "yes_full"
        batch.create_vertices("City", 2)

    assert executor.is_transaction_use_wal() is True
    assert executor.get_transaction_sync() == "no"


def test_a_graph_batch_puts_the_settings_back_when_the_block_raises(temp_db):
    executor = temp_db.async_executor()
    executor.set_transaction_use_wal(True)
    executor.set_transaction_sync("yes_nometadata")

    with pytest.raises(RuntimeError, match="stop"):
        with temp_db.graph_batch():
            assert executor.get_transaction_sync() == "no"
            raise RuntimeError("stop")

    assert executor.is_transaction_use_wal() is True
    assert executor.get_transaction_sync() == "yes_nometadata"


def test_a_graph_batch_refuses_a_wal_flush_mode_it_does_not_know(temp_db):
    with pytest.raises(ValueError, match="Invalid wal_flush mode"):
        temp_db.graph_batch(wal_flush="fsync")


def test_a_graph_batch_leaves_the_flush_of_your_transactions_alone(temp_db):
    """JVM only: with the batch open, a transaction of yours commits with the flush you set, and after close() too."""
    temp_db.command("sql", "CREATE VERTEX TYPE City")

    def flush_of_a_transaction_of_mine():
        temp_db.begin()
        try:
            return str(temp_db._java_db.getTransaction().getWALFlush())
        finally:
            temp_db.rollback()

    temp_db.set_wal_flush("yes_full")
    try:
        assert flush_of_a_transaction_of_mine() == "YES_FULL"
        with temp_db.graph_batch(parallel_flush=False) as batch:
            batch.create_vertices("City", 2)
            assert flush_of_a_transaction_of_mine() == "YES_FULL"
        assert flush_of_a_transaction_of_mine() == "YES_FULL"
    finally:
        temp_db.set_wal_flush("no")
