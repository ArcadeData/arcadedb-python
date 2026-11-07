#!/usr/bin/env python3
"""A wheel states which engine snapshot it embeds, and a release publishes no wheel that does not.

    verify_wheel_engine_pin.py VERSION WHEEL...

VERSION is the tag. For `A.B.C.devN` every wheel's recording (arcadedb/_build_info.py of the native build, arcadedb/_version.py of the
JVM build) must carry ENGINE_SNAPSHOT and ENGINE_IMAGE_DIGEST of scripts/dev_engine_pin.json. For a final or `.postN` version they must
be None, because such a wheel embeds its release engine. Exit 1 on any difference. Reads only. Tests: tests/test_verify_wheel_engine_pin.py.
"""

from __future__ import annotations

import ast
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_engine_pin  # noqa: E402

RECORD = re.compile(r"^[^/]+/(_build_info|_version)\.py$")


def recorded(wheel: Path) -> dict[str, object]:
    """ENGINE_SNAPSHOT and ENGINE_IMAGE_DIGEST as the wheel's recording file assigns them."""
    with zipfile.ZipFile(wheel) as zf:
        names = [n for n in zf.namelist() if RECORD.match(n)]
        found: dict[str, object] = {}
        for name in names:
            text = zf.read(name).decode("utf-8")
            for key in ("ENGINE_SNAPSHOT", "ENGINE_IMAGE_DIGEST"):
                m = re.search(rf"^{key} = (.+?)(?:\s+#.*)?\s*$", text, re.M)
                if m:
                    found[key] = ast.literal_eval(m.group(1))
    return found


def problems(version: str, wheels: list[Path]) -> list[str]:
    pin = dev_engine_pin.load_pin()
    if dev_engine_pin.DEV_TAG.fullmatch(version):
        bad = dev_engine_pin.tag_problem(version, pin)
        if bad:
            return [bad]
        want = {
            "ENGINE_SNAPSHOT": pin["snapshot"],
            "ENGINE_IMAGE_DIGEST": pin["image_digest"],
        }
    else:
        want = {"ENGINE_SNAPSHOT": None, "ENGINE_IMAGE_DIGEST": None}
    out = []
    for w in wheels:
        got = recorded(w)
        if got != want:
            out.append(
                f"{w.name}: the wheel records {got or 'nothing'}, a {version} wheel must record {want}"
            )
    if not wheels:
        out.append("no wheels given")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    bad = problems(argv[0], [Path(a) for a in argv[1:]])
    for line in bad:
        print(f"::error::{line}")
    if not bad:
        print(f"OK: {len(argv) - 1} wheels state the engine they embed ({argv[0]})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
