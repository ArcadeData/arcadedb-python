"""Refuse to import when both distributions that provide the `arcadedb` import are installed.

This repository builds two distributions. `arcadedb` is the native build (the engine as a GraalVM native-image
library, loaded with ctypes) and `arcadedb-jvm` is the JVM build (the Java engine in-process through JPype).
Both put a package named `arcadedb` into the same site-packages directory, so installing one after the other
overwrites files of the first with files of the second, and the installer says nothing. What would be imported is a mix
of two builds. Package metadata still lists both, so this module looks there and raises an ImportError that
names both packages and says how to fix it, instead of letting the mix run.

`check(backend)` is called first thing in `arcadedb/__init__.py`. `backend` is the build this code belongs to:
"jvm" or "native".
"""

from __future__ import annotations

from importlib import metadata
from typing import Callable

# backend -> the distribution that carries it
DISTRIBUTIONS = {"native": "arcadedb", "jvm": "arcadedb-jvm"}

# The page that explains the two packages (the "Which package?" page of the docs, linked from the README).
DOCS_URL = "https://github.com/ArcadeData/arcadedb-python/blob/main/docs/modules/ROOT/pages/which-package.adoc"


class DistributionConflictError(ImportError):
    """Both the native `arcadedb` and the JVM `arcadedb-jvm` distribution are installed."""


def other_backend(backend: str) -> str:
    if backend not in DISTRIBUTIONS:
        raise ValueError(f"backend must be 'jvm' or 'native', not {backend!r}")
    return "jvm" if backend == "native" else "native"


def installed(
    backends: tuple[str, ...] = ("native", "jvm"),
    lookup: Callable[[str], str] = metadata.version,
) -> list[str]:
    """The distribution names of `backends` that are installed, in the order given.

    `lookup` raises `importlib.metadata.PackageNotFoundError` for a name that is not installed.
    """
    found = []
    for backend in backends:
        name = DISTRIBUTIONS[backend]
        try:
            lookup(name)
        except metadata.PackageNotFoundError:
            continue
        found.append(name)
    return found


def message(backend: str) -> str:
    own, other = DISTRIBUTIONS[backend], DISTRIBUTIONS[other_backend(backend)]
    kind = {"native": "native", "jvm": "JVM"}
    return (
        f"The distribution '{own}' ({kind[backend]} build) cannot be imported while '{other}' "
        f"({kind[other_backend(backend)]} build) is installed. "
        f"Both provide the Python package 'arcadedb', so each install overwrote files of the other "
        f"without a message, and what you would import is a mix of the two builds. "
        f"Keep one: run 'uv remove {DISTRIBUTIONS['native']} {DISTRIBUTIONS['jvm']}', then install "
        f"'{DISTRIBUTIONS['native']}' (native, the default for most embedded use) or "
        f"'{DISTRIBUTIONS['jvm']}' (JVM, for heavy long-running loads). "
        f"Which one to pick: {DOCS_URL}"
    )


def check(backend: str, lookup: Callable[[str], str] = metadata.version) -> None:
    """Raise DistributionConflictError when the distribution of the other backend is installed too.

    Never raises for anything else: an import must not fail because the metadata could not be read.
    """
    other = other_backend(backend)
    try:
        found = installed((other,), lookup)
    except Exception:  # unreadable metadata, an odd sys.path entry: say nothing
        return
    if found:
        raise DistributionConflictError(message(backend))
