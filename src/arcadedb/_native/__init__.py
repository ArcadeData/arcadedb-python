"""The native backend: the engine as a GraalVM native-image shared library (``libarcadedb``) that Python loads with ``ctypes``. No JVM, no JPype.

It is the only backend of the ``arcadedb`` wheel. In a source checkout or in the ``arcadedb-jvm`` wheel, ``ARCADEDB_BACKEND=native`` selects it
(with ``ARCADEDB_NATIVE_LIB`` pointing at a library you built); the default there is JPype. The public API is the JVM build's, and the same
test suite (``tests/``) runs against both.

On native: databases and the factory, ``query``/``command`` (SQL and openCypher), transactions and ``run_in_transaction``, results as rows,
columns and Arrow, pandas, polars and numpy frames, the ``Document``/``Vertex``/``Edge`` wrappers, the schema (types, properties, indexes),
vector indexes (``find_nearest``, the PQ ``find_nearest_approximate``, ``get_metadata``, ``get_stats``), ``insert_many``, ``GraphBatch``,
the ``AsyncExecutor`` (``transaction`` included), Python functions in SQL and record callbacks, importers and exporters, and the in-process
server.

Not on native, by design (each raises :class:`NotOnNativeYet`): the Java objects of the JVM build. These are ``get_java_database``,
``get_java_document``, and the Java object a ``Schema``, type or index method returns where the native wrapper has no such method.
There is no JVM to start or tune: ``jvm_kwargs`` is ignored with a warning.
"""

import os


class NotOnNativeYet(NotImplementedError):
    """The call hands out a Java object of the JVM build (``get_java_database``, ``get_java_document``, a Schema method's Java object), which the native build does not have."""


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
