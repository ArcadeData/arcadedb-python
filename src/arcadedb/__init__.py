"""
ArcadeDB Python Embedded Bindings

A native Python bindings for ArcadeDB that embeds the Java database engine
directly in the Python process using JPype.
"""

# Which build this is: 'jvm' (the arcadedb-jvm distribution, the Java engine through JPype) or 'native' (the
# arcadedb distribution, the engine as a GraalVM native-image library). The import name is arcadedb for both.
# `_native.distribution_backend()` is fixed when the wheel is built (the native wheel build flips one line in
# `_native/__init__.py`). `__backend__` is the engine that runs now: ARCADEDB_BACKEND=native can switch it on in a
# source checkout, where a developer loads a library they built themselves.
from . import _native as _native_backend
from ._dist_conflict import DISTRIBUTIONS as _DISTRIBUTIONS
from ._dist_conflict import check as _check_distributions

_DISTRIBUTION_BACKEND = _native_backend.distribution_backend()
__backend__ = "native" if _native_backend.enabled() else "jvm"

# Both distributions install a package named arcadedb into the same directory; the installer overwrites one with the other
# silently. Refuse to import when the other one is installed too, before anything else runs.
_check_distributions(_DISTRIBUTION_BACKEND)

# The installed distribution's version is the single source of truth: the
# wheel is versioned from the release tag, while the generated _version.py
# is derived from the engine jar's version (scripts/write_version.py), so the
# two disagree on tagged releases
# (a 26.8.1.dev20 wheel carried __version__ == "26.8.1.dev0"). Fall back to
# the generated file, then to a placeholder, for source checkouts that were
# never installed.
try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version(_DISTRIBUTIONS[_DISTRIBUTION_BACKEND])
except Exception:  # not installed: fall back to the build-time file
    try:
        from ._version import __version__
    except ModuleNotFoundError:
        __version__ = "0.0.0"

# Import async execution
from .async_executor import AsyncExecutor  # noqa: E402

# Import record event handle
from .callbacks import EventHandle  # noqa: E402

# Import core database classes
from .core import (  # noqa: E402
    Database,
    DatabaseFactory,
    create_database,
    database_exists,
    open_database,
)

# Import exceptions
from .exceptions import ArcadeDBError  # noqa: E402

# Import exporter classes
from .exporter import export_database, export_to_csv  # noqa: E402

# Import graph classes
from .graph import Document, Edge, Vertex  # noqa: E402

# Import graph batch helper
from .graph_batch import GraphBatch  # noqa: E402

# Import importer helpers
from .importer import ImportResult  # noqa: E402

# Which engine this install actually carries. __version__ is the PACKAGE
# version and can disagree with the bundled JARs without any error.
from .jvm import jar_fingerprint  # noqa: E402

# Import result classes
from .results import Result, ResultSet  # noqa: E402

# Import schema classes
from .schema import IndexType, PropertyType, Schema  # noqa: E402

# Import server classes
from .server import ArcadeDBServer, create_server  # noqa: E402

# Import transaction management
from .transactions import TransactionContext  # noqa: E402

# Import type conversion utilities
from .type_conversion import (  # noqa: E402
    convert_java_to_python,
    convert_python_to_java,
)

# Import vector utilities and index
from .vector import (  # noqa: E402
    VectorIndex,
    to_java_byte_array,
    to_java_float_array,
    to_java_int_array,
    to_python_array,
)

# The native wheel records them in _build_info.py (and build_wheel.py imports them near the top of this file), the JVM wheel in
# _version.py. Look in _build_info first: a fallback that ran last and set None hid the native wheel's value (26.12.1.dev0).
try:
    from ._build_info import ENGINE_IMAGE_DIGEST, ENGINE_SNAPSHOT
except ImportError:
    try:
        from ._version import ENGINE_IMAGE_DIGEST, ENGINE_SNAPSHOT
    except (
        ImportError
    ):  # a source checkout without a generated _version.py, or an older build
        ENGINE_SNAPSHOT = None
        ENGINE_IMAGE_DIGEST = None

__all__ = [
    "__version__",
    "__backend__",
    "ENGINE_SNAPSHOT",
    "ENGINE_IMAGE_DIGEST",
    # Exceptions
    "ArcadeDBError",
    # Core classes
    "Database",
    "DatabaseFactory",
    "EventHandle",
    # Record wrappers
    "Document",
    "Vertex",
    "Edge",
    # Build provenance
    "jar_fingerprint",
    "create_database",
    "open_database",
    "database_exists",
    # Result classes
    "ResultSet",
    "Result",
    # Schema classes
    "Schema",
    "IndexType",
    "PropertyType",
    # Transaction management
    "TransactionContext",
    # Async execution
    "AsyncExecutor",
    "GraphBatch",
    "ImportResult",
    # Type conversion
    "convert_java_to_python",
    "convert_python_to_java",
    # Vector search
    "VectorIndex",
    "to_java_byte_array",
    "to_java_float_array",
    "to_java_int_array",
    "to_python_array",
    # Server classes
    "ArcadeDBServer",
    "create_server",
    # Data export
    "export_database",
    "export_to_csv",
]
