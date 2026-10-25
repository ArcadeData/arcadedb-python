"""Logs (and, on the JVM build, the JVM crash file) go to a per-user directory, never the working directory.

The subprocess tests run on both builds; the JVM-only ones are listed in conftest.NATIVE_NA_JVM_TESTS and
test_native_log_dir.py has their native counterparts."""

import os
import subprocess  # nosec B404 - test-controlled child process
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from arcadedb import _logdir, jvm
from arcadedb._native import lib as native_lib
from arcadedb.jvm import _build_jvm_args, _default_log_dir

CHILD = """
import sys
import arcadedb
from arcadedb import jvm
from arcadedb._native import enabled

jvm_args = sys.argv[2:]
# the JVM build takes its settings as jvm_kwargs; the native build has none (its log settings are tested in test_native_log_dir.py)
kwargs = {} if enabled() else {"jvm_kwargs": {"jvm_args": jvm_args}}
db = arcadedb.create_database(sys.argv[1], **kwargs)
db.command("sql", "CREATE DOCUMENT TYPE Doc")
db.close()
if enabled():
    from arcadedb._native import lib

    used = lib.LOG_DIR_USED
else:
    used = jvm._LOG_DIR_USED
print("LOGDIR=" + str(used))
"""


def _run_child(tmp_path, env_overrides, jvm_args=()):
    """Start a JVM in a child process whose working directory is empty and read-only."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    cwd.chmod(0o555)  # not enforced everywhere; the listing below holds regardless
    env = {k: v for k, v in os.environ.items() if k != "ARCADEDB_JVM_ARGS"}
    env.pop("ARCADEDB_LOG_DIR", None)
    env.pop("ARCADEDB_JVM_ERROR_FILE", None)
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_STATE_HOME=str(home / "state"),
        LOCALAPPDATA=str(home / "localappdata"),
    )
    env.update(env_overrides)
    try:
        result = subprocess.run(  # nosec B603 - fixed argument list, no shell
            [
                sys.executable,
                "-c",
                CHILD,
                str(tmp_path / "db"),
                *jvm_args,
            ],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        cwd.chmod(0o755)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(l for l in result.stdout.splitlines() if l.startswith("LOGDIR="))
    return cwd, Path(line[len("LOGDIR=") :]), home


def test_default_logs_land_in_the_state_dir_not_the_cwd(tmp_path):
    cwd, log_dir, home = _run_child(tmp_path, {})
    assert list(cwd.iterdir()) == []
    assert home in log_dir.parents
    assert log_dir.name == "log" and log_dir.parent.name == "arcadedb"
    assert list(log_dir.glob("arcadedb.log*")), list(log_dir.iterdir())
    config = (log_dir / "logging.properties").read_text("iso-8859-1")
    assert "${arcadedb.server.logsDirectory}/arcadedb.log" in config
    assert "./log" not in config


def test_env_var_sets_the_log_dir(tmp_path):
    target = tmp_path / "chosen logs"
    cwd, log_dir, _ = _run_child(tmp_path, {"ARCADEDB_LOG_DIR": str(target)})
    assert list(cwd.iterdir()) == []
    assert log_dir == target
    assert list(target.glob("arcadedb.log*"))


def test_jvm_args_can_move_the_engine_log(tmp_path):
    target = tmp_path / "from jvm args"
    cwd, _, home = _run_child(
        tmp_path, {}, jvm_args=[f"-Darcadedb.server.logsDirectory={target.as_posix()}"]
    )
    assert list(cwd.iterdir()) == []
    assert list(target.glob("arcadedb.log*"))


def test_unusable_log_dir_falls_back_to_a_temp_dir(tmp_path):
    blocker = tmp_path / "a file"
    blocker.write_text("not a directory")
    cwd, log_dir, _ = _run_child(tmp_path, {"ARCADEDB_LOG_DIR": str(blocker / "log")})
    assert list(cwd.iterdir()) == []
    assert log_dir != blocker / "log"
    assert list(log_dir.glob("arcadedb.log*"))


def test_args_never_name_the_working_directory():
    with patch.dict(os.environ, {}, clear=True):
        args = _build_jvm_args(heap_size="4g", disable_xml_limits=False, jvm_args=None)
        # the expected directory comes from the same (cleared) environment the args were built in: on Windows the real
        # environment has LOCALAPPDATA and the cleared one falls back, so computing it outside the block would differ
        log_dir = _default_log_dir().as_posix()
    assert not any("./log" in a for a in args)
    assert f"-Darcadedb.server.logsDirectory={log_dir}" in args
    assert f"-Djava.util.logging.config.file={log_dir}/logging.properties" in args
    assert f"-XX:ErrorFile={log_dir}/hs_err_pid%p.log" in args


def test_state_dir_by_platform(tmp_path):
    env = {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    with patch.dict(os.environ, {**env, "XDG_STATE_HOME": str(tmp_path / "xdg")}):
        with patch.object(jvm.sys, "platform", "linux"):
            assert _default_log_dir() == tmp_path / "xdg" / "arcadedb" / "log"
    with patch.dict(os.environ, {**env, "XDG_STATE_HOME": "relative"}):
        with patch.object(jvm.sys, "platform", "linux"):
            expected = tmp_path / ".local" / "state" / "arcadedb" / "log"
            assert _default_log_dir() == expected
    with patch.dict(os.environ, env):
        with patch.object(jvm.sys, "platform", "darwin"):
            expected = tmp_path / "Library" / "Logs" / "arcadedb" / "log"
            assert _default_log_dir() == expected
    with patch.dict(os.environ, {**env, "LOCALAPPDATA": str(tmp_path / "lad")}):
        with patch.object(jvm.sys, "platform", "win32"):
            assert _default_log_dir() == tmp_path / "lad" / "arcadedb" / "log"


def test_user_arguments_win_over_the_log_defaults(tmp_path):
    user = [
        "-Darcadedb.server.logsDirectory=/x/logs",
        "-Djava.util.logging.config.file=/x/my.properties",
        "-XX:ErrorFile=/x/crash.log",
    ]
    with patch.dict(os.environ, {"ARCADEDB_JVM_ERROR_FILE": "/y/env.log"}, clear=True):
        args = _build_jvm_args(
            heap_size="4g", disable_xml_limits=False, jvm_args=user, log_dir=tmp_path
        )
    for arg in user:
        assert args.count(arg) == 1
    assert sum(a.startswith("-Darcadedb.server.logsDirectory=") for a in args) == 1
    assert sum(a.startswith("-Djava.util.logging.config.file=") for a in args) == 1
    assert sum(a.startswith("-XX:ErrorFile=") for a in args) == 1


def test_env_error_file_still_overrides_the_default(tmp_path):
    with patch.dict(os.environ, {"ARCADEDB_JVM_ERROR_FILE": "/y/env.log"}, clear=True):
        args = _build_jvm_args(
            heap_size="4g", disable_xml_limits=False, jvm_args=None, log_dir=tmp_path
        )
    assert "-XX:ErrorFile=/y/env.log" in args
    assert sum(a.startswith("-XX:ErrorFile=") for a in args) == 1


def test_rendered_config_keeps_the_engine_settings_and_moves_the_file_log():
    engine = (
        "handlers = java.util.logging.ConsoleHandler, java.util.logging.FileHandler\n"
        "java.util.logging.FileHandler.pattern=./log/arcadedb.log\n"
        "java.util.logging.FileHandler.count=10\n"
    )
    rendered = jvm._render_logging_config(engine)
    assert "./log" not in rendered
    assert "FileHandler.count=10" in rendered
    assert rendered.count("FileHandler.pattern") == 1
    assert "${arcadedb.server.logsDirectory}/arcadedb.log" in rendered


def test_the_engine_jar_carries_the_log_config_the_render_starts_from():
    import glob

    jars = glob.glob(os.path.join(jvm.get_jar_path(), "*.jar"))
    text = jvm._read_engine_log_properties(jars)
    assert text is not None
    assert "java.util.logging.FileHandler.pattern" in text


def test_windows_branch_needs_no_home_directory(tmp_path):
    # The Windows runner clears the environment in the unit tests above; the base
    # directory must come from LOCALAPPDATA, or the temp dir, without a home directory.
    with patch.object(jvm.sys, "platform", "win32"):
        with patch.dict(os.environ, {"LOCALAPPDATA": str(tmp_path)}, clear=True):
            with patch.object(Path, "home", side_effect=RuntimeError("no home")):
                assert _default_log_dir() == tmp_path / "arcadedb" / "log"
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "home", side_effect=RuntimeError("no home")):
                expected = Path(tempfile.gettempdir()) / "arcadedb" / "log"
                assert _default_log_dir() == expected
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "home", return_value=tmp_path):
                expected = tmp_path / "AppData" / "Local" / "arcadedb" / "log"
                assert _default_log_dir() == expected


def test_arg_values_use_forward_slashes_on_every_platform(tmp_path):
    # as_posix() keeps a Windows drive path usable in -D and -XX values.
    with patch.dict(os.environ, {}, clear=True):
        args = _build_jvm_args(
            heap_size="4g", disable_xml_limits=False, jvm_args=None, log_dir=tmp_path
        )
    assert f"-Darcadedb.server.logsDirectory={tmp_path.as_posix()}" in args
    assert f"-XX:ErrorFile={tmp_path.as_posix()}/hs_err_pid%p.log" in args


def test_native_isolate_arguments_name_the_log_dir(tmp_path):
    args = native_lib._log_isolate_args([], tmp_path)
    assert f"-Darcadedb.server.logsDirectory={tmp_path.as_posix()}" in args
    config = f"-Djava.util.logging.config.file={tmp_path.as_posix()}/logging.properties"
    assert config in args
    assert not any("./log" in a for a in args)


def test_native_user_isolate_arguments_win_over_the_log_defaults(tmp_path):
    user = [
        "-Darcadedb.server.logsDirectory=/x/logs",
        "-Djava.util.logging.config.file=/x/my.properties",
    ]
    assert native_lib._log_isolate_args(user, tmp_path) == []
    only_dir = native_lib._log_isolate_args(user[:1], tmp_path)
    assert len(only_dir) == 1 and only_dir[0].startswith(
        "-Djava.util.logging.config.file="
    )


def test_native_engine_log_config_renders_like_the_jvm_one():
    rendered = _logdir.render_logging_config(_logdir.NATIVE_ENGINE_LOG_PROPERTIES)
    assert "./log" not in rendered
    assert rendered.count("FileHandler.pattern") == 1
    assert "${arcadedb.server.logsDirectory}/arcadedb.log" in rendered
    assert "java.util.logging.FileHandler.count=10" in rendered
    assert "handlers = java.util.logging.ConsoleHandler" in rendered


def _settings(text):
    """The key/value pairs of a .properties text, comments and spacing around = dropped."""
    pairs = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "!")):
            key, _, value = line.partition("=")
            pairs[key.strip()] = value.strip()
    return pairs


def test_native_engine_log_config_matches_the_engine_jar():
    # The native image has no jar to read, so its log settings are a copy of the engine's arcadedb-log.properties (without
    # the file pattern, which the render sets). This is the guard that fails when an engine upgrade changes that file.
    import glob

    jars = glob.glob(os.path.join(jvm.get_jar_path(), "*.jar"))
    engine = _settings(jvm._read_engine_log_properties(jars))
    engine.pop("java.util.logging.FileHandler.pattern")
    copy = _settings(_logdir.NATIVE_ENGINE_LOG_PROPERTIES)
    assert copy == engine
