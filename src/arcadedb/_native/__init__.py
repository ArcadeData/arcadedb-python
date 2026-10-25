"""The native backend: the engine as a GraalVM native-image library, no JVM and no JPype.

It is the only backend of the `arcadedb` wheel. In a source checkout or in the `arcadedb-jvm` wheel, ``ARCADEDB_BACKEND=native``
selects it (with ``ARCADEDB_NATIVE_LIB`` pointing at a library you built); the default there is JPype. The engine runs as a GraalVM native-image shared
library (``libarcadedb``) that Python loads with ``ctypes``: no JVM, no JPype. The surface is the DSL-first core:
databases, ``query``/``command`` for SQL and openCypher, transactions, results and the in-process server.

What is not on native yet (each raises :class:`NotOnNativeYet`): Python UDFs and record callbacks, ``GraphBatch``, the
async executor, the Java-object APIs (``Document``/``Vertex``/``Edge`` wrappers, ``new_vertex``, ``new_document``,
``schema``, ``get_java_database``), the columnar/Arrow readers other than a simple row conversion, importers and
exporters that go through Java objects, and every JVM setting (``jvm_kwargs``, ``start_jvm``).
"""

import os


class NotOnNativeYet(NotImplementedError):
    """The feature needs the Java object API, which the native backend does not expose (yet)."""


# The backend used when ARCADEDB_BACKEND is unset. "jpype" here, which is the arcadedb-jvm wheel; the build of the native
# `arcadedb` wheel (native/wheel/build_wheel.py) rewrites this one line to "native".
_DEFAULT_BACKEND = "jpype"


def distribution_backend() -> str:
    """The build this code was packaged as: "native" (the `arcadedb` distribution) or "jvm" (the `arcadedb-jvm` distribution).

    Fixed when the wheel is built. ``ARCADEDB_BACKEND`` does not change it: the distribution check in ``arcadedb/__init__.py``
    asks which package this file belongs to, not which engine a developer switched on.
    """
    return "native" if _DEFAULT_BACKEND == "native" else "jvm"


def enabled() -> bool:
    """True when the native backend is selected: ``ARCADEDB_BACKEND=native``, or unset in the `arcadedb` (native) wheel (read at call time, so a test can switch it)."""
    return os.environ.get("ARCADEDB_BACKEND", _DEFAULT_BACKEND).lower() == "native"


def not_yet(what: str):
    raise NotOnNativeYet(f"{what} is not on the native backend yet")
