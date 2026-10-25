"""A host that redirects fd 1 and 2 after the library started must not get engine log text written into database files.

On Windows the runtime caches the OS handle of stdout and stderr at start-up. pytest's fd capture, a service or an embedding
application that dup2s onto fd 1 or 2 afterwards closes that handle; a database file opened later can get the same handle value,
and the next console log line was then written into that file (a bucket file that started with
``2026-10-25 19:37:03.759 INFO [LSMVectorIndex] Discovered and loaded graph file ...``, vector index files with ``Graph build
validating ...``; in the tests that showed as ``Invalid vector id -N``, ``Invalid position N (size=65536)`` or no result after a
reopen). The fix is that the engine logs to its file only on Windows unless ``ARCADEDB_LOG_CONSOLE`` asks for the console.

The redirect test runs on every platform: on POSIX writes go to fd 1 and 2 by number, so it passes trivially there, and on Windows
it is the real check. It does the same work as ``test_lsm_persistence`` and ``test_lsm_vector_search_approximate_persistence`` in
``test_vector.py``: create a vector index, build the graph, close, reopen, query.
"""

import json
import os
import re
import subprocess  # nosec B404 - test-controlled child process
import sys
from pathlib import Path

import pytest

from arcadedb import _logdir

# One engine log line as the file and console formatters write it: date, time with milliseconds, level.
LOG_LINE = re.compile(
    rb"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} "
    rb"(?:SEVERE|WARNING|INFO|CONFIG|FINE|FINER|FINEST)\b"
)

CHILD = """
import json
import os
import sys
import time

import arcadedb

mode, data_dir, sink, result = sys.argv[1:5]

# start the library first: on Windows it caches the handles of fd 1 and 2 now
warm = arcadedb.create_database(os.path.join(data_dir, "warm"))
warm.command("sql", "CREATE DOCUMENT TYPE Warm")
warm.close()

if mode == "redirect":
    # what pytest's fd capture, a service or an embedding application does
    fd = os.open(sink, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)

db_path = os.path.join(data_dir, "db")
vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
for i in range(256):
    vectors.append([0.01, ((i % 16) + 1) / 10.0, (((i // 16) % 16) + 1) / 10.0])

with arcadedb.create_database(db_path) as db:
    db.command("sql", "CREATE VERTEX TYPE Doc")
    db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    index = db.create_vector_index(
        "Doc", "embedding", dimensions=3, quantization="PRODUCT"
    )
    with db.transaction():
        for vec in vectors:
            db.command(
                "sql",
                "INSERT INTO Doc SET embedding = ?",
                arcadedb.to_java_float_array(vec),
            )
    index.build_graph_now()

with arcadedb.open_database(db_path) as db:
    count = db.query("sql", "SELECT count(*) as count FROM Doc").first().get("count")
    neighbors = []
    for _ in range(100):  # the graph loads asynchronously after the reopen
        rows = db.query(
            "sql", "SELECT vectorNeighbors('Doc[embedding]', [1.0, 0.0, 0.0], 1) as res"
        ).to_list()
        neighbors = rows[0].get("res")
        if neighbors:
            break
        time.sleep(0.1)

with open(result, "w") as out:
    json.dump({"count": count, "neighbors": len(neighbors)}, out)
"""


def _run(tmp_path, mode, console=None):
    """Run CHILD in a fresh process; return (result dict, completed process, data dir, log dir, sink file)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    log_dir = tmp_path / "log"
    sink = tmp_path / "sink.txt"
    result = tmp_path / "result.json"
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("ARCADEDB_LOG_CONSOLE", "ARCADEDB_LOG_DIR")
    }
    env["ARCADEDB_LOG_DIR"] = str(log_dir)
    if console is not None:
        env["ARCADEDB_LOG_CONSOLE"] = console
    proc = subprocess.run(  # nosec B603 - fixed argument list, no shell
        [sys.executable, "-c", CHILD, mode, str(data_dir), str(sink), str(result)],
        env=env,
        capture_output=True,
        timeout=600,
    )
    sink_text = sink.read_bytes() if sink.exists() else b""
    assert proc.returncode == 0, (
        proc.stdout.decode(errors="replace")
        + proc.stderr.decode(errors="replace")
        + sink_text.decode(errors="replace")
    )
    return json.loads(result.read_text()), proc, data_dir, log_dir, sink_text


def _log_lines_in_files(root: Path):
    """Every file under ``root`` that holds an engine log line, with the first match."""
    found = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        match = LOG_LINE.search(path.read_bytes())
        if match:
            found.append((str(path.relative_to(root)), match.group(0)))
    return found


def _log_file_text(log_dir: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(log_dir.glob("arcadedb.log*")))


def test_redirecting_fd_1_and_2_leaves_the_database_files_clean(tmp_path):
    result, _, data_dir, log_dir, _ = _run(tmp_path, "redirect")
    assert result == {"count": 259, "neighbors": 1}
    assert _log_lines_in_files(data_dir) == []
    # the work did log, to the file: the scan above is not clean because nothing was logged
    assert LOG_LINE.search(_log_file_text(log_dir)), list(log_dir.iterdir())


def test_redirect_with_the_console_off_sends_nothing_to_the_redirect_target(tmp_path):
    # ARCADEDB_LOG_CONSOLE=0 is the Windows default on every platform: nothing reaches fd 1 or 2 even where that is safe
    result, _, data_dir, log_dir, sink = _run(tmp_path, "redirect", console="0")
    assert result == {"count": 259, "neighbors": 1}
    assert LOG_LINE.search(sink) is None, sink[:500]
    assert _log_lines_in_files(data_dir) == []
    assert LOG_LINE.search(_log_file_text(log_dir))


def test_console_log_can_be_turned_on(tmp_path):
    # no redirect here: on Windows the console is off by default because of the redirect hazard above, and a child whose fd 1 and 2
    # are the parent's pipes is the safe way to see that the opt-in works
    result, proc, data_dir, log_dir, _ = _run(tmp_path, "plain", console="1")
    assert result == {"count": 259, "neighbors": 1}
    console = proc.stdout + proc.stderr
    assert LOG_LINE.search(console), console[:500]
    assert _log_lines_in_files(data_dir) == []
    assert LOG_LINE.search(_log_file_text(log_dir))


def test_console_log_can_be_turned_off(tmp_path):
    result, proc, _, log_dir, _ = _run(tmp_path, "plain", console="0")
    assert result == {"count": 259, "neighbors": 1}
    console = proc.stdout + proc.stderr
    assert LOG_LINE.search(console) is None, console[:500]
    assert LOG_LINE.search(_log_file_text(log_dir))


@pytest.mark.parametrize(
    ("platform", "value", "expected"),
    [
        ("linux", None, True),
        ("darwin", None, True),
        ("win32", None, False),
        ("win32", "1", True),
        ("win32", "true", True),
        ("win32", " On ", True),
        ("linux", "0", False),
        ("linux", "false", False),
        ("darwin", "no", False),
        ("linux", "", True),
        ("win32", "", False),
        ("win32", "maybe", False),
        ("linux", "maybe", True),
    ],
)
def test_console_default_by_platform_and_env(monkeypatch, platform, value, expected):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("ARCADEDB_LOG_CONSOLE", raising=False)
    if value is not None:
        monkeypatch.setenv("ARCADEDB_LOG_CONSOLE", value)
    assert _logdir.console_enabled() is expected
    assert _logdir.logging_config_name() == (
        "logging.properties" if expected else "logging-file-only.properties"
    )


def test_render_without_the_console_keeps_only_the_file_handler():
    engine = (
        "handlers = java.util.logging.ConsoleHandler, java.util.logging.FileHandler\n"
        ".level = INFO\n"
        "java.util.logging.ConsoleHandler.level = INFO\n"
        "java.util.logging.ConsoleHandler.formatter = com.arcadedb.utility.AnsiLogFormatter\n"
        "java.util.logging.FileHandler.pattern=./log/arcadedb.log\n"
        "java.util.logging.FileHandler.level = INFO\n"
    )
    plain = _logdir.render_logging_config(engine, console=False)
    assert "ConsoleHandler" not in plain
    assert "handlers = java.util.logging.FileHandler\n" in plain
    assert "java.util.logging.FileHandler.level = INFO" in plain
    assert ".level = INFO" in plain
    assert "${arcadedb.server.logsDirectory}/arcadedb.log" in plain
    with_console = _logdir.render_logging_config(engine, console=True)
    assert (
        "handlers = java.util.logging.ConsoleHandler, java.util.logging.FileHandler"
        in with_console
    )
    assert "java.util.logging.ConsoleHandler.formatter" in with_console


def test_each_console_setting_has_its_own_config_file(tmp_path, monkeypatch):
    engine = _logdir.NATIVE_ENGINE_LOG_PROPERTIES
    monkeypatch.setenv("ARCADEDB_LOG_CONSOLE", "1")
    _logdir.write_logging_config(tmp_path, engine)
    monkeypatch.setenv("ARCADEDB_LOG_CONSOLE", "0")
    _logdir.write_logging_config(tmp_path, engine)
    full = (tmp_path / "logging.properties").read_text("iso-8859-1")
    file_only = (tmp_path / "logging-file-only.properties").read_text("iso-8859-1")
    assert "ConsoleHandler" in full
    assert "ConsoleHandler" not in file_only
    assert "FileHandler" in file_only


def test_both_builds_point_the_engine_at_the_config_for_the_setting(
    tmp_path, monkeypatch
):
    from arcadedb._native import lib as native_lib
    from arcadedb.jvm import _build_jvm_args

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("ARCADEDB_LOG_CONSOLE", raising=False)
    jvm_args = _build_jvm_args(
        heap_size="4g", disable_xml_limits=False, jvm_args=None, log_dir=tmp_path
    )
    native_args = native_lib._log_isolate_args([], tmp_path)
    expected = (
        f"-Djava.util.logging.config.file={tmp_path.as_posix()}"
        "/logging-file-only.properties"
    )
    assert expected in jvm_args
    assert expected in native_args
    monkeypatch.setenv("ARCADEDB_LOG_CONSOLE", "1")
    expected = (
        f"-Djava.util.logging.config.file={tmp_path.as_posix()}/logging.properties"
    )
    assert expected in _build_jvm_args(
        heap_size="4g", disable_xml_limits=False, jvm_args=None, log_dir=tmp_path
    )
    assert expected in native_lib._log_isolate_args([], tmp_path)
