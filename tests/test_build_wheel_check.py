"""build.sh's local-jars check accepts the wheel each engine version builds.

After a build from a local jars directory, build.sh runs an inline Python check
(the ``python3 - << 'PY'`` heredoc) that finds this build's wheel in dist/ by
version and compares the integration jar inside it with the local one. It
turned the engine version into the wheel version with its own string rules,
which mapped ``26.11.1-RC1`` to ``26.11.1.RC1``: no RC wheel (``26.11.1rc1``)
ever matched, and an RC build failed with "no arcadedb-... wheel in
dist/" (#7). This runs that heredoc, unchanged, for each kind of version.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - test-controlled child process
import sys
import zipfile
from pathlib import Path

import pytest

BUILD_SH = Path(__file__).resolve().parents[1] / "scripts" / "build.sh"


def _wheel_check_source() -> str:
    text = BUILD_SH.read_text(encoding="utf-8")
    start = text.index("python3 - << 'PY'\n") + len("python3 - << 'PY'\n")
    end = text.index("\nPY\n", start)
    return text[start:end] + "\n"


@pytest.mark.parametrize(
    "engine, wheel_version",
    [
        ("26.11.1-SNAPSHOT", "26.11.1.dev0"),
        ("26.10.2", "26.10.2"),
        ("26.11.1-RC1", "26.11.1rc1"),
    ],
)
def test_wheel_check_finds_this_builds_wheel(tmp_path, engine, wheel_version):
    jar_name = f"arcadedb-integration-{engine}.jar"
    (tmp_path / "local-jars" / "lib").mkdir(parents=True)
    (tmp_path / "local-jars" / "lib" / jar_name).write_bytes(b"integration")
    (tmp_path / "dist").mkdir()
    wheel = tmp_path / "dist" / f"arcadedb-{wheel_version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(f"arcadedb/jars/{jar_name}", b"integration")

    env = dict(os.environ, ARCADEDB_VERSION=engine, BUILD_VERSION="", NEW_WHEELS="")
    out = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-c", _wheel_check_source()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "matches local-jars/lib" in out.stdout
