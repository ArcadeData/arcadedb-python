"""Where the engine writes its log, for both builds (the JVM build and the native build).

The engine's own ``arcadedb-log.properties`` writes ``./log/arcadedb.log`` into the working directory. Both builds move the file
log to a per-user directory instead: ``ARCADEDB_LOG_DIR`` if set, else ``arcadedb/log`` under the user's state directory. They
write a ``logging.properties`` there whose file pattern is ``${arcadedb.server.logsDirectory}/arcadedb.log`` and point two
system properties at it: ``arcadedb.server.logsDirectory`` (the directory) and ``java.util.logging.config.file`` (that file).
The JVM build sets them as JVM arguments (``jvm.py``), the native build as isolate arguments (``_native/lib.py``). The engine
reads both when it first logs, so they have to be in place before the first engine call. This module has no JVM and no JPype.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

from ._logging import get_logger

_LOGGER = get_logger(__name__)

LOG_DIR_ENV = "ARCADEDB_LOG_DIR"
LOGGING_CONFIG_NAME = "logging.properties"

# The engine's arcadedb-log.properties without its file pattern, for the native build: the image holds the engine's copy as a
# resource, but an isolate argument has to name a real file, and there is no jar to read it from. tests/test_log_dir.py compares
# this text with the file in the engine jar, so an engine upgrade that changes the file fails there until this is updated.
NATIVE_ENGINE_LOG_PROPERTIES = """\
handlers = java.util.logging.ConsoleHandler, java.util.logging.FileHandler
.level = INFO
io.undertow.level=WARNING
org.xnio.level=WARNING
org.jboss.level=WARNING
org.apache.ratis.grpc.server.GrpcLogAppender.level = SEVERE
com.arcadedb.level = INFO
java.util.logging.ConsoleHandler.level = INFO
java.util.logging.ConsoleHandler.formatter = com.arcadedb.utility.AnsiLogFormatter
java.util.logging.FileHandler.level = INFO
java.util.logging.FileHandler.formatter = com.arcadedb.log.LogFormatter
java.util.logging.FileHandler.limit=100000000
java.util.logging.FileHandler.count=10
"""


def state_base_dir() -> Path:
    """Per-user base directory for state and logs, by platform (stdlib only).

    The home directory is read only where the platform needs it, and when it
    cannot be determined (no HOME or USERPROFILE, as in a stripped-down service
    environment) the system temp directory is used, so this never raises.
    """
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local)
    elif sys.platform != "darwin":
        xdg = os.environ.get("XDG_STATE_HOME")
        # The XDG spec ignores a relative value.
        if xdg and os.path.isabs(xdg):
            return Path(xdg)
    try:
        home = Path.home()
    except (RuntimeError, KeyError, OSError):
        return Path(tempfile.gettempdir())
    if sys.platform == "win32":
        return home / "AppData" / "Local"
    if sys.platform == "darwin":
        return home / "Library" / "Logs"
    return home / ".local" / "state"


def default_log_dir() -> Path:
    """Where the engine log, the crash log and logging.properties go.

    ``ARCADEDB_LOG_DIR`` if set, else ``arcadedb/log`` under the
    per-user state directory. Never the working directory.
    """
    override = os.environ.get(LOG_DIR_ENV)
    if override:
        return Path(os.path.abspath(os.path.expanduser(override)))
    return state_base_dir() / "arcadedb" / "log"


def render_logging_config(engine_text: str) -> str:
    """The engine's logging config with the file log in ``logsDirectory``.

    The jar's own file hard-codes ``./log/arcadedb.log``, which lands in the
    working directory. The engine resolves ``${arcadedb.server.logsDirectory}``
    in the pattern, and the argument that sets it is added by the caller.
    """
    lines = [
        line
        for line in engine_text.splitlines()
        if not re.match(r"\s*java\.util\.logging\.FileHandler\.pattern\s*[=:]", line)
    ]
    lines.append(
        "java.util.logging.FileHandler.pattern="
        "${arcadedb.server.logsDirectory}/arcadedb.log"
    )
    return "\n".join(lines) + "\n"


def usable_log_dir(log_dir: Path) -> Path:
    """Create the log directory; return it, or a fresh directory under the system temp directory when it cannot be created or
    written (a read-only home, for example), so that nothing is ever written to the working directory.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(log_dir, os.W_OK | os.X_OK):
            raise OSError(f"{log_dir} is not writable")
    except OSError as exc:
        fallback = Path(tempfile.mkdtemp(prefix="arcadedb-log-"))
        _LOGGER.warning(
            "Cannot use %s for ArcadeDB logs (%s); using %s. Set %s to choose one.",
            log_dir,
            exc,
            fallback,
            LOG_DIR_ENV,
        )
        return fallback
    return log_dir


def write_logging_config(log_dir: Path, engine_text: str) -> None:
    """Write ``logging.properties`` in ``log_dir`` (replaced atomically, and only when its text changed)."""
    config = log_dir / LOGGING_CONFIG_NAME
    rendered = render_logging_config(engine_text)
    try:
        if not config.exists() or config.read_text("iso-8859-1") != rendered:
            tmp = log_dir / f"{LOGGING_CONFIG_NAME}.{os.getpid()}.tmp"
            tmp.write_text(rendered, encoding="iso-8859-1", newline="\n")
            os.replace(tmp, config)
    except OSError as exc:
        _LOGGER.warning("Cannot write %s: %s", config, exc)
