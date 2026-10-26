"""A host that redirects fd 1 and 2 after the library started must not get engine log text, or Java's own output, written into database files.

On Windows the runtime caches the OS handle of stdout and stderr at start-up. pytest's fd capture, a service or an embedding
application that dup2s onto fd 1 or 2 afterwards closes that handle; a database file opened later can get the same handle value,
and the next write through the cached handle was then written into that file (a bucket file that started with
``2026-10-25 19:37:03.759 INFO [LSMVectorIndex] Discovered and loaded graph file ...``, vector index files with ``Graph build
validating ...``; in the tests that showed as ``Invalid vector id -N``, ``Invalid position N (size=65536)`` or no result after a
reopen). Two fixes close it, in two layers:

* The engine logs to its file only on Windows unless ``ARCADEDB_LOG_CONSOLE`` asks for the console (no JUL console handler).
* With the console off, the JVM build also replaces Java's ``System.out``, ``System.err`` and ``System.initialErr`` with a stream on
  ``arcadedb-stdio.log`` in the log directory (``jvm._redirect_java_stdio``). Text that the first fix left on the cached handle
  reached a page file with ``Invalid position 793803132 (size=65536)`` in the JVM test run 40250615309 on Windows, one run in
  four. The text was the runtime's warning ``A terminally deprecated method in sun.misc.Unsafe has been called`` (JEP 498), which
  jvector causes when it unmaps a graph file on the first reopen of a vector index. The runtime prints it to ``System.initialErr``,
  the copy of ``System.err`` that it makes before application code runs and that ``System.setErr`` does not change, so replacing
  ``System.out`` and ``System.err`` alone left it on the cached handle. (A canary file that took over the stale handle's value
  after the host's ``dup2`` held exactly these four lines.)

The redirect tests run on every platform and on both builds (the first fix is shared; the second is the JVM build's, see below):
on POSIX writes go to fd 1 and 2 by number, so a redirect is followed and the default passes trivially there, and on Windows it is
the real check. The JVM build's console-off test also fails on POSIX without the second fix, because the Java probes below then go
to the redirect target. The child does the same work as ``test_lsm_persistence`` and ``test_lsm_vector_search_approximate_persistence``
in ``test_vector.py`` (create a vector index, build the graph, close, reopen, query), ``ARCADEDB_REDIRECT_REPEATS`` times in one
process (default 10). On the JVM build it prints a probe line after every step through Java's ``System.out`` and ``System.err``: a
write through the cached handle that hits a file whenever a database file owns the handle value, which no log line has to be around
for. Every file under the data directory is then scanned for log text and probes.

The probes and the JEP 498 trigger call Java through JPype, so they run on the JVM build only (``JVM`` below). The native build has no
JVM and no Java streams: its child runs the same steps with zero probes, and the log text in the redirect target, in the database
files and on the console is checked the same way.

One text is not covered: the runtime prints ``WARNING: Using incubator modules: jdk.incubator.vector`` to stderr while the JVM is
created, before the library can install its streams. In the child the JVM starts before the redirect, so that line goes to the
parent's pipe (``proc.stderr``), never to the redirect target. Run 40250615309 showed it in the failure message only because the
old assertion message joined stdout, stderr and the redirect target into one text; the messages below name each part.
"""

import json
import os
import re
import subprocess  # nosec B404 - test-controlled child process
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import arcadedb
from arcadedb import _logdir

# The Java-level probes and the JEP 498 trigger call Java through JPype, so they need the JVM build (arcadedb-jvm), not the native one.
JVM = arcadedb.__backend__ == "jvm"

# One engine log line as the file and console formatters write it: date, time with milliseconds, level.
LOG_LINE = re.compile(
    rb"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} "
    rb"(?:SEVERE|WARNING|INFO|CONFIG|FINE|FINER|FINEST)\b"
)
# What the child prints through Java's System.out and System.err after every step.
PROBE = b"ARCADEDB-STDIO-PROBE"
# The runtime's warning for the first use of a sun.misc.Unsafe memory method (JEP 498). jvector prints it when it unmaps a graph
# file, on the first reopen of a vector index. The runtime writes it to System.initialErr, not to System.err.
UNSAFE_WARNING = b"A terminally deprecated method in sun.misc.Unsafe has been called"
UNSAFE_TRIGGER_FAILED = (
    "the child could not unmap a file with jvector's SimpleMappedReader.Supplier, the call that makes the runtime print "
    "UNSAFE_WARNING; if jvector no longer uses sun.misc.Unsafe, pick another call that does"
)
# Text of the runtime's own that never belongs in a database file.
STRAY_TEXT = re.compile(
    LOG_LINE.pattern
    + rb"|"
    + PROBE
    + rb"|WARNING: Using incubator modules|"
    + re.escape(UNSAFE_WARNING)
)

REPEATS = int(os.environ.get("ARCADEDB_REDIRECT_REPEATS", "10"))

CHILD = """
import json
import os
import sys
import time

import arcadedb

mode, data_dir, sink, result, repeats, marker = sys.argv[1:7]
# the probes and the JEP 498 trigger call Java through JPype: only the JVM build has a JVM to call
jvm = arcadedb.__backend__ == "jvm"

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

probes = 0
unsafe = False
if jvm:
    import jpype

    system = jpype.JClass("java.lang.System")

    # the runtime's own warning, now and not at some later reopen: jvector unmaps a file with sun.misc.Unsafe.invokeCleaner
    try:
        target = os.path.join(os.path.dirname(sink), "mapped.bin")
        with open(target, "wb") as handle:
            handle.write(b"x" * 64)
        supplier = jpype.JClass("io.github.jbellis.jvector.disk.SimpleMappedReader$Supplier")
        supplier(jpype.JClass("java.nio.file.Path").of(target, [])).close()
        unsafe = True
    except Exception:
        pass  # jvector changed: the reopen below still reaches the warning if the runtime prints it


def probe(step):
    # a Java-level write to stdout and to stderr, as a runtime warning or a library that prints does (none on the native build)
    global probes
    if not jvm:
        return
    probes += 1
    system.out.println(f"{marker} out {probes} {step}")
    system.err.println(f"{marker} err {probes} {step}")


vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
for i in range(256):
    vectors.append([0.01, ((i % 16) + 1) / 10.0, (((i // 16) % 16) + 1) / 10.0])

counts = set()
neighbor_counts = set()
for run in range(int(repeats)):
    db_path = os.path.join(data_dir, f"db{run}")
    probe("before create")
    with arcadedb.create_database(db_path) as db:
        db.command("sql", "CREATE VERTEX TYPE Doc")
        db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
        probe("schema")
        index = db.create_vector_index(
            "Doc", "embedding", dimensions=3, quantization="PRODUCT"
        )
        probe("index")
        with db.transaction():
            for vec in vectors:
                db.command(
                    "sql",
                    "INSERT INTO Doc SET embedding = ?",
                    arcadedb.to_java_float_array(vec),
                )
        probe("insert")
        index.build_graph_now()
        probe("graph")

    with arcadedb.open_database(db_path) as db:
        probe("reopen")
        counts.add(
            db.query("sql", "SELECT count(*) as count FROM Doc").first().get("count")
        )
        neighbors = []
        for _ in range(100):  # the graph loads asynchronously after the reopen
            rows = db.query(
                "sql",
                "SELECT vectorNeighbors('Doc[embedding]', [1.0, 0.0, 0.0], 1) as res",
            ).to_list()
            neighbors = rows[0].get("res")
            probe("query")
            if neighbors:
                break
            time.sleep(0.1)
        neighbor_counts.add(len(neighbors))

with open(result, "w") as out:
    json.dump(
        {
            "counts": sorted(counts),
            "neighbors": sorted(neighbor_counts),
            "runs": int(repeats),
            "probes": probes,
            "unsafe": unsafe,
        },
        out,
    )
"""


def _run(tmp_path, mode, console=None, repeats=REPEATS):
    """Run CHILD in a fresh process. Returns the result dict, the process, the data and log directories, the redirect target's
    text and the text of ``arcadedb-stdio.log`` (empty when there is none), in a namespace.
    """
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
        [
            sys.executable,
            "-c",
            CHILD,
            mode,
            str(data_dir),
            str(sink),
            str(result),
            str(repeats),
            PROBE.decode(),
        ],
        env=env,
        capture_output=True,
        timeout=1800,
    )
    sink_text = sink.read_bytes() if sink.exists() else b""
    # Each part is named: the runtime's start-up warning is in stderr (the pipe), never in the redirect target.
    assert proc.returncode == 0, (
        f"child exit {proc.returncode}\n"
        f"--- stdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"--- stderr:\n{proc.stderr.decode(errors='replace')}\n"
        f"--- redirect target:\n{sink_text.decode(errors='replace')}"
    )
    stdio = log_dir / _logdir.STDIO_LOG_NAME
    return SimpleNamespace(
        result=json.loads(result.read_text()),
        proc=proc,
        data_dir=data_dir,
        log_dir=log_dir,
        sink=sink_text,
        stdio=stdio.read_bytes() if stdio.exists() else b"",
        expected={
            "counts": [259],
            "neighbors": [1],
            "runs": repeats,
            "probes": json.loads(result.read_text())["probes"],
            "unsafe": json.loads(result.read_text())["unsafe"],
        },
    )


def _stray_text_in_files(root: Path):
    """Every file under ``root`` that holds engine log text, a probe or the runtime's warning, with the first match."""
    found = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        match = STRAY_TEXT.search(path.read_bytes())
        if match:
            found.append((str(path.relative_to(root)), match.group(0)))
    return found


def _log_file_text(log_dir: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(log_dir.glob("arcadedb.log*")))


def test_redirecting_fd_1_and_2_leaves_the_database_files_clean(tmp_path):
    run = _run(tmp_path, "redirect")
    assert run.result == run.expected
    if JVM:
        # the probes ran: a clean scan is not clean because nothing wrote
        assert run.result["probes"] >= 6 * REPEATS
    assert _stray_text_in_files(run.data_dir) == []
    # the work did log, to the file: the scan above is not clean because nothing was logged
    assert LOG_LINE.search(_log_file_text(run.log_dir)), list(run.log_dir.iterdir())


def test_redirect_with_the_console_off_sends_nothing_to_the_redirect_target(tmp_path):
    # ARCADEDB_LOG_CONSOLE=0 is the Windows default on every platform: nothing reaches fd 1 or 2 even where that is safe, and
    # Java's System.out and System.err append to arcadedb-stdio.log
    run = _run(tmp_path, "redirect", console="0")
    assert run.result == run.expected
    assert LOG_LINE.search(run.sink) is None, run.sink[:500]
    assert PROBE not in run.sink, run.sink[:500]
    assert _stray_text_in_files(run.data_dir) == []
    assert LOG_LINE.search(_log_file_text(run.log_dir))
    # every probe, from both streams, is in the file the library owns (and nowhere else); the native build has none
    assert run.stdio.count(PROBE) == 2 * run.result["probes"], run.stdio[:500]
    assert run.stdio.count(b" out ") == run.stdio.count(b" err ")
    if JVM:
        assert UNSAFE_WARNING not in run.sink, run.sink[:500]
        # the runtime's own warning is printed to System.initialErr, which setErr does not reach: it is there too
        assert run.result["unsafe"], UNSAFE_TRIGGER_FAILED
        assert UNSAFE_WARNING in run.stdio, run.stdio[:500]


def test_console_log_can_be_turned_on(tmp_path):
    # no redirect here: on Windows the console is off by default because of the redirect hazard above, and a child whose fd 1 and 2
    # are the parent's pipes is the safe way to see that the opt-in works
    run = _run(tmp_path, "plain", console="1", repeats=1)
    assert run.result == run.expected
    console = run.proc.stdout + run.proc.stderr
    assert LOG_LINE.search(console), console[:500]
    # the runtime keeps its own streams with the console on: the probes are on the console, there is no stdio file
    assert console.count(PROBE) == 2 * run.result["probes"], console[:500]
    assert run.stdio == b""
    assert _stray_text_in_files(run.data_dir) == []
    assert LOG_LINE.search(_log_file_text(run.log_dir))


def test_console_log_can_be_turned_off(tmp_path):
    run = _run(tmp_path, "plain", console="0", repeats=1)
    assert run.result == run.expected
    console = run.proc.stdout + run.proc.stderr
    assert LOG_LINE.search(console) is None, console[:500]
    assert PROBE not in console, console[:500]
    assert run.stdio.count(PROBE) == 2 * run.result["probes"], run.stdio[:500]
    if JVM:
        assert UNSAFE_WARNING not in console, console[:500]
        assert run.result["unsafe"], UNSAFE_TRIGGER_FAILED
        assert UNSAFE_WARNING in run.stdio, run.stdio[:500]
    assert LOG_LINE.search(_log_file_text(run.log_dir))


def test_the_stdio_log_is_in_the_log_dir(tmp_path):
    assert _logdir.stdio_log_path(tmp_path) == tmp_path / "arcadedb-stdio.log"
    # not a file the engine's file handler owns (arcadedb.log.N), so a rotation never renames it away
    assert not _logdir.STDIO_LOG_NAME.startswith("arcadedb.log")


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
