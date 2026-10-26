#!/usr/bin/env python3
"""Fail a wheel that is not tagged py3-none-<platform>.

arcadedb-jvm ships one wheel for each platform, not one for each Python version: the JRE inside is platform-specific,
but nothing in the wheel is compiled against CPython (the Java side is reached through JPype, a dependency with its
own wheels). setup.py tags the wheel py3-none-<platform>; a wheel tagged cpXY-cpXY-<platform> would be one of sixteen
files for a release (about 65 MB each) where four do, and a wheel tagged -any would install on a platform whose JRE
it does not carry. Which Pythons the wheel installs on is `requires-python` in pyproject.toml.

A wheel passes when its file name says py3-none-<platform> with a real platform, and its WHEEL file agrees: every
`Tag:` line is that tag and `Root-Is-Purelib` is false (the files go to platlib, next to the JRE).

Usage:
    python verify_wheel_tag.py WHEEL [WHEEL ...]

Exit status: 0 every wheel passes, 1 a wheel does not, 2 usage error or a wheel that is missing or not a zip file.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def problems(wheel: Path) -> list[str]:
    """What is wrong with the tag of `wheel`; empty when it is py3-none-<platform> in both the name and the WHEEL file."""
    found: list[str] = []
    if not wheel.name.endswith(".whl"):
        return [f"{wheel.name}: not a .whl file"]
    # name-version(-build)-python-abi-platform.whl
    parts = wheel.name[: -len(".whl")].split("-")
    if len(parts) not in (5, 6):
        return [f"{wheel.name}: not a wheel file name (name-version-python-abi-platform.whl)"]
    python, abi, platform = parts[-3:]
    if (python, abi) != ("py3", "none"):
        found.append(f"{wheel.name}: tagged {python}-{abi}, expected py3-none")
    if platform == "any" or not platform:
        found.append(f"{wheel.name}: platform tag {platform!r}; the wheel bundles a platform-specific JRE")
    tag = f"{python}-{abi}-{platform}"

    with zipfile.ZipFile(wheel) as zf:
        names = [n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")]
        if len(names) != 1:
            return found + [f"{wheel.name}: expected one .dist-info/WHEEL, found {names}"]
        # strip: a wheel built on Windows has CRLF line endings
        lines = [line.strip() for line in zf.read(names[0]).decode("utf-8").splitlines()]
    tags = [line.split(":", 1)[1].strip() for line in lines if line.startswith("Tag:")]
    if tags != [tag]:
        found.append(f"{wheel.name}: the WHEEL file says Tag {tags}, the file name says {tag}")
    purelib = [line.split(":", 1)[1].strip().lower() for line in lines if line.startswith("Root-Is-Purelib:")]
    if purelib != ["false"]:
        found.append(f"{wheel.name}: Root-Is-Purelib is {purelib}, expected false (a platform wheel)")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args(argv)

    status = 0
    for wheel in args.wheels:
        try:
            found = problems(wheel)
        except (OSError, zipfile.BadZipFile) as error:
            print(f"ERROR: {wheel}: {error}", file=sys.stderr)
            return 2
        if found:
            status = 1
            for line in found:
                print(f"ERROR: {line}", file=sys.stderr)
        else:
            print(f"OK: {wheel.name} is tagged {'-'.join(wheel.name[: -len('.whl')].split('-')[-3:])}")
    if status:
        print(
            "   The wheel must be py3-none-<platform>: see setup.py (bdist_wheel.get_tag). Do not publish it.",
            file=sys.stderr,
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
