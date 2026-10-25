"""Warn when both distributions that provide the `arcadedb` import are installed.

The JVM wheel (`arcadedb`) and the native wheel (`arcadedb-native`) both import as `arcadedb`. Their
files land in the same directory, so installing one after the other overwrites files of the first with
files of the second, and pip says nothing. The result is a mix of two builds. Package metadata still
lists both, so this module looks there.
"""

from __future__ import annotations

import warnings
from importlib import metadata
from typing import Callable

# Distributions that put an `arcadedb` package into site-packages.
DISTRIBUTIONS = ("arcadedb", "arcadedb-native")


def installed_distributions(
    names: tuple[str, ...] = DISTRIBUTIONS,
    lookup: Callable[[str], str] = metadata.version,
) -> list[str]:
    """The names from `names` that are installed, in the order given.

    `lookup` raises `importlib.metadata.PackageNotFoundError` for a name that is not installed.
    """
    found = []
    for name in names:
        try:
            lookup(name)
        except metadata.PackageNotFoundError:
            continue
        found.append(name)
    return found


def warn_if_both_installed(
    lookup: Callable[[str], str] = metadata.version, stacklevel: int = 2
) -> bool:
    """Emit a RuntimeWarning when both distributions are installed. True when it did.

    Never raises: an import must not fail because the metadata could not be read.
    """
    try:
        found = installed_distributions(lookup=lookup)
    except Exception:  # unreadable metadata, an odd sys.path entry: say nothing
        return False
    if len(found) < len(DISTRIBUTIONS):
        return False
    warnings.warn(
        "Both the 'arcadedb' and the 'arcadedb-native' distributions are installed. "
        "Both provide the Python package 'arcadedb', so each install overwrote files of the "
        "other without a message, and what you import now is a mix of the two builds. "
        "Install only one: pip uninstall arcadedb arcadedb-native, then pip install arcadedb "
        "(or arcadedb-native).",
        RuntimeWarning,
        stacklevel=stacklevel,
    )
    return True
