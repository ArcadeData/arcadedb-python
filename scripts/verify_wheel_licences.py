#!/usr/bin/env python3
"""Fail a wheel that contains a file that is not open source.

This package is Apache-2.0 and ships nothing that is not open source. One known case today: ApexCharts, which has not
been open source since 5.1.0 (Community/Commercial licence). Studio, the web UI in `arcadedb-studio-*.jar`, bundled it
from upstream 26.1.1 until ArcadeData/arcadedb#10627 replaced it with Apache ECharts (first in engine 26.11.1). Studio
ships in the wheels again, so this check does not look for the Studio jar. It looks for the files that make a wheel
unfit to publish, whatever the cause: a wheel built from an engine older than 26.11.1, a shaded or renamed jar, a JAR
directory that brings its own copy. A wheel whose Studio carries ECharts passes.
See https://github.com/ArcadeData/arcadedb/issues/10579 and https://github.com/ArcadeData/arcadedb/pull/10627.

To add a case, add a row to NON_OPEN_SOURCE: a lower-case marker that appears in the path of every file of that product,
and a short description for the error message.

Usage:
    python verify_wheel_licences.py WHEEL [WHEEL ...]

Exit status: 0 all wheels clean, 1 a wheel contains a file that is not open source, 2 usage error or not a wheel.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

# (marker in the lower-cased path, description). A file matches when its path, or the path of an entry inside a jar,
# contains the marker.
NON_OPEN_SOURCE: tuple[tuple[str, str], ...] = (
    (
        "apexcharts",
        "ApexCharts, not open source since 5.1.0 (Studio in engines before 26.11.1)",
    ),
)

HELP = (
    "Do not publish this wheel. Studio uses Apache ECharts from engine 26.11.1 "
    "(https://github.com/ArcadeData/arcadedb/pull/10627): build from the 26.11.1 jars or newer "
    "(see https://github.com/ArcadeData/arcadedb/issues/10579)."
)


def _match(path: str) -> str | None:
    """The description of the non-open-source product the path belongs to, else None."""
    lowered = path.lower()
    for marker, description in NON_OPEN_SOURCE:
        if marker in lowered:
            return description
    return None


def find_offenders(wheel: Path) -> list[str]:
    """Names inside the wheel (and inside its jars) that belong to a product that is not open source."""
    offenders: list[str] = []
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            name = info.filename
            description = _match(name)
            if description is not None:
                offenders.append(f"{name} ({description})")
            elif name.lower().endswith(".jar"):
                try:
                    inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
                except zipfile.BadZipFile:
                    offenders.append(
                        f"{name} (jar that is not a zip file: cannot be checked)"
                    )
                    continue
                for entry in inner.namelist():
                    description = _match(entry)
                    if description is not None:
                        offenders.append(f"{name} contains {entry} ({description})")
                        break
    return offenders


def main(argv: list[str] | None = None) -> int:
    wheels = [Path(a) for a in (sys.argv[1:] if argv is None else argv)]
    if not wheels:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: verify_wheel_licences.py WHEEL [WHEEL ...]", file=sys.stderr)
        return 2
    status = 0
    for wheel in wheels:
        if not wheel.is_file() or not zipfile.is_zipfile(wheel):
            print(f"ERROR: {wheel} is not a wheel file", file=sys.stderr)
            return 2
        offenders = find_offenders(wheel)
        if offenders:
            status = 1
            print(
                f"ERROR: {wheel.name} contains files that are not open source:",
                file=sys.stderr,
            )
            for entry in offenders:
                print(f"   {entry}", file=sys.stderr)
            print(f"   {HELP}", file=sys.stderr)
        else:
            print(f"OK: {wheel.name} has no file that is not open source")
    return status


if __name__ == "__main__":
    sys.exit(main())
