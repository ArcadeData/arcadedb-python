"""Where the engine writes its log, for both builds (the JVM build and the native build).

The engine's own ``arcadedb-log.properties`` writes ``./log/arcadedb.log`` into the working directory. Both builds move the file
log to a per-user directory instead: ``ARCADEDB_LOG_DIR`` if set, else ``arcadedb/log`` under the user's state directory. They
write a ``logging.properties`` there whose file pattern is ``${arcadedb.server.logsDirectory}/arcadedb.log`` and point two
system properties at it: ``arcadedb.server.logsDirectory`` (the directory) and ``java.util.logging.config.file`` (that file).
The JVM build sets them as JVM arguments (``jvm.py``), the native build as isolate arguments (``_native/lib.py``). The engine
reads both when it first logs, so they have to be in place before the first engine call. This module has no JVM and no JPype.

The engine's console handler is off by default on Windows (``ARCADEDB_LOG_CONSOLE`` turns it on or off on any platform). On Windows
the runtime caches the OS handle of stdout and stderr at start-up. A host that redirects fd 1 or 2 afterwards (pytest's fd capture,
a service, an embedding application) closes that handle, a database file opened later can get the same handle value, and the next
console log line is written into that file. The log file does not have that problem: the engine opens it itself.

Console logging is not the only Java code that writes to the cached handle: a library that prints or ``e.printStackTrace()`` goes
through ``System.out`` and ``System.err``, and the runtime's own warnings (the JEP 498 warning for ``sun.misc.Unsafe``, which jvector
causes on the first reopen of a vector index) go through ``System.initialErr``, a copy of ``System.err`` that ``System.setErr`` does not
change. So with the console off, the JVM build points all three streams at ``arcadedb-stdio.log`` in the log directory, a file the
library opens itself, right after the JVM starts (``jvm.py``, :func:`stdio_log_path`). The runtime's start-up text (``WARNING: Using incubator modules: jdk.incubator.vector``) is printed while the
JVM is created, before any of this can run, and still goes to the real stderr of the process.
"""

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

from ._logging import get_logger

_LOGGER = get_logger(__name__)

LOG_DIR_ENV = "ARCADEDB_LOG_DIR"
LOG_CONSOLE_ENV = "ARCADEDB_LOG_CONSOLE"
LOGGING_CONFIG_NAME = "logging.properties"
# The config without the console handler has its own file name, so that two processes with different console settings that share a
# log directory never replace each other's file between the write and the engine's first log call.
FILE_ONLY_CONFIG_NAME = "logging-file-only.properties"
# Where Java's System.out and System.err go while the console is off. A file of its own, not arcadedb.log: the engine's file handler
# owns that one (it rotates it, and on Windows a rename of a file another handle holds open fails).
STDIO_LOG_NAME = "arcadedb-stdio.log"
CONSOLE_HANDLER = "java.util.logging.ConsoleHandler"

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


def console_enabled() -> bool:
    """Whether the engine logs to the console (stdout/stderr) as well as to its log file.

    ``ARCADEDB_LOG_CONSOLE`` decides on every platform: ``1``, ``true``, ``yes`` or ``on`` turns the console handler on; ``0``,
    ``false``, ``no`` or ``off`` turns it off. Unset (or any other value), it is on everywhere except Windows, where it is off
    (see the module docstring for why). A ``java.util.logging.config.file`` the caller passes replaces all of this.
    """
    value = os.environ.get(LOG_CONSOLE_ENV, "").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return sys.platform != "win32"


def stdio_log_path(log_dir: Path) -> Path:
    """The file that Java's ``System.out`` and ``System.err`` append to while the console is off (see the module docstring)."""
    return Path(log_dir) / STDIO_LOG_NAME


def logging_config_name(console: Optional[bool] = None) -> str:
    """The file name of the generated logging config: ``logging.properties`` with the console handler, ``logging-file-only.properties`` without."""
    if console is None:
        console = console_enabled()
    return LOGGING_CONFIG_NAME if console else FILE_ONLY_CONFIG_NAME


def render_logging_config(engine_text: str, console: Optional[bool] = None) -> str:
    """The engine's logging config with the file log in ``logsDirectory``, and without the console handler unless ``console``.

    The jar's own file hard-codes ``./log/arcadedb.log``, which lands in the
    working directory. The engine resolves ``${arcadedb.server.logsDirectory}``
    in the pattern, and the argument that sets it is added by the caller.
    ``console`` None means :func:`console_enabled`.
    """
    if console is None:
        console = console_enabled()
    lines = []
    for line in engine_text.splitlines():
        if re.match(r"\s*java\.util\.logging\.FileHandler\.pattern\s*[=:]", line):
            continue
        if not console:
            handlers = re.match(r"(\s*handlers\s*[=:]\s*)(.*)$", line)
            if handlers:
                kept = [
                    h
                    for h in re.split(r"[,\s]+", handlers.group(2).strip())
                    if h and h != CONSOLE_HANDLER
                ]
                line = f"{handlers.group(1)}{', '.join(kept)}"
            elif re.match(r"\s*java\.util\.logging\.ConsoleHandler\.", line):
                continue
        lines.append(line)
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
    """Write the logging config in ``log_dir`` under :func:`logging_config_name` (replaced atomically, and only when its text changed)."""
    console = console_enabled()
    name = logging_config_name(console)
    config = log_dir / name
    rendered = render_logging_config(engine_text, console)
    try:
        if not config.exists() or config.read_text("iso-8859-1") != rendered:
            tmp = log_dir / f"{name}.{os.getpid()}.tmp"
            tmp.write_text(rendered, encoding="iso-8859-1", newline="\n")
            os.replace(tmp, config)
    except OSError as exc:
        _LOGGER.warning("Cannot write %s: %s", config, exc)
