"""Native build: the isolate writes its log where the log-dir settings say, and never into the working directory.

The JVM build gets the same settings as JVM arguments (test_log_dir.py). The native library is a GraalVM isolate: the glue passes
the two system properties as isolate arguments, because the engine reads them at its first log call and an isolate argument is
the only setting that is in place before it. ARCADEDB_NATIVE_ISOLATE_ARGS takes the user's own arguments, which win.
"""

from arcadedb import _logdir

from .test_log_dir import _run_child


def test_isolate_argument_moves_the_engine_log(tmp_path):
    target = tmp_path / "from_isolate_args"
    cwd, _, _ = _run_child(
        tmp_path,
        {
            "ARCADEDB_NATIVE_ISOLATE_ARGS": f"-Darcadedb.server.logsDirectory={target.as_posix()}"
        },
    )
    assert list(cwd.iterdir()) == []
    assert list(target.glob("arcadedb.log*"))


def test_own_logging_config_wins(tmp_path):
    mine = tmp_path / "mine"
    mine.mkdir()
    config = tmp_path / "my.properties"
    rendered = _logdir.render_logging_config(_logdir.NATIVE_ENGINE_LOG_PROPERTIES)
    rendered = rendered.replace(
        "${arcadedb.server.logsDirectory}/arcadedb.log", f"{mine.as_posix()}/own.log"
    )
    config.write_text(rendered, encoding="iso-8859-1")
    cwd, log_dir, _ = _run_child(
        tmp_path,
        {
            "ARCADEDB_NATIVE_ISOLATE_ARGS": f"-Djava.util.logging.config.file={config.as_posix()}"
        },
    )
    assert list(cwd.iterdir()) == []
    assert list(mine.glob("own.log*"))
    assert not list(log_dir.glob("arcadedb.log*"))
