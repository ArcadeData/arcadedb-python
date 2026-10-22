"""A native build bundles the jars it was asked for, not the last build's (#9).

build-native.sh reused whatever was in src/arcadedb_embedded/jars: with no
jars directory given it never pulled ``arcadedata/arcadedb:${ARCADEDB_TAG}``,
so a wheel built after ``latest`` moved carried the old engine; with one given
(build.sh's third argument) it ignored it. scripts/stage_jars.sh now stages
the jars on every build: from the directory build.sh passes, or freshly from
the image. These tests run it under bash with a fake ``docker`` on PATH.
"""

from __future__ import annotations

import os
import stat
import subprocess  # nosec B404 - test-controlled child process
import tarfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
STAGE_JARS = SCRIPTS_DIR / "stage_jars.sh"


def _jars(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text(name)
    return directory


def _fake_docker(tmp_path: Path, *lib_jars: str, fail: bool = False) -> Path:
    """A docker that logs its argv and streams a tar of lib/ holding lib_jars."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    image_home = _jars(tmp_path / "image" / "lib", *lib_jars).parent
    tar_path = tmp_path / "lib.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(image_home / "lib", arcname="lib")
    docker = bin_dir / "docker"
    body = "exit 1" if fail else f'cat "{tar_path}"'
    docker.write_text(
        f'#!/usr/bin/env bash\necho "$*" >> "{tmp_path / "docker.log"}"\n{body}\n'
    )
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _stage(jars_dir: Path, source: str, tag: str, bin_dir=None):
    env = dict(os.environ)
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; stage_jars "$2" "$3" "$4"',
            "stage",
            str(STAGE_JARS),
            str(jars_dir),
            source,
            tag,
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def _names(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.jar")}


def test_no_source_pulls_the_tag_and_replaces_stale_jars(tmp_path):
    jars = _jars(
        tmp_path / "jars",
        "arcadedb-engine-26.10.2.jar",
        "arcadedb-python-bridge.jar",
    )
    bin_dir = _fake_docker(tmp_path, "arcadedb-engine-26.11.1-SNAPSHOT.jar", "x-1.jar")
    out = _stage(jars, "", "26.11.1-SNAPSHOT", bin_dir)
    assert out.returncode == 0, out.stderr
    assert _names(jars) == {"arcadedb-engine-26.11.1-SNAPSHOT.jar", "x-1.jar"}
    log = (tmp_path / "docker.log").read_text()
    assert "--pull always" in log
    assert "arcadedata/arcadedb:26.11.1-SNAPSHOT" in log


def test_source_dir_replaces_stale_jars(tmp_path):
    jars = _jars(tmp_path / "jars", "arcadedb-engine-26.10.2.jar", "stale-1.jar")
    source = _jars(tmp_path / "local", "arcadedb-engine-26.11.1-RC1.jar", "y-2.jar")
    out = _stage(jars, str(source), "latest")
    assert out.returncode == 0, out.stderr
    assert _names(jars) == {"arcadedb-engine-26.11.1-RC1.jar", "y-2.jar"}
    assert _names(source) == {"arcadedb-engine-26.11.1-RC1.jar", "y-2.jar"}


def test_source_dir_that_is_the_jars_dir_is_kept(tmp_path):
    # build-native.sh run by hand on its own jars dir: nothing to copy
    jars = _jars(tmp_path / "jars", "arcadedb-engine-26.11.1-SNAPSHOT.jar")
    out = _stage(jars, str(jars), "latest")
    assert out.returncode == 0, out.stderr
    assert _names(jars) == {"arcadedb-engine-26.11.1-SNAPSHOT.jar"}


def test_failed_pull_fails_the_build(tmp_path):
    jars = _jars(tmp_path / "jars", "arcadedb-engine-26.10.2.jar")
    bin_dir = _fake_docker(tmp_path, "unused.jar", fail=True)
    out = _stage(jars, "", "latest", bin_dir)
    assert out.returncode != 0


def test_build_sh_hands_its_jars_dir_to_the_native_build():
    # The staged local-jars directory, when build.sh was given one
    call = [
        line
        for line in (SCRIPTS_DIR / "build.sh").read_text(encoding="utf-8").splitlines()
        if '"$SCRIPT_DIR/build-native.sh"' in line
    ]
    assert len(call) == 1, call
    assert "NATIVE_JARS_SOURCE" in call[0]


def test_ci_path_restages_the_artifact_from_local_jars(tmp_path):
    # CI: build.sh copies its jars dir (the artifact, src/arcadedb_embedded/jars)
    # into local-jars/lib and hands that over; a previous build's leftovers in
    # the jars dir go, the artifact's jars come back.
    jars = _jars(
        tmp_path / "jars",
        "arcadedb-engine-26.11.1-SNAPSHOT.jar",
        "arcadedb-python-bridge.jar",
        "arcadedb-engine-26.10.2.jar",
    )
    local = _jars(
        tmp_path / "local-jars" / "lib", "arcadedb-engine-26.11.1-SNAPSHOT.jar"
    )
    out = _stage(jars, str(local), "26.11.1-SNAPSHOT")
    assert out.returncode == 0, out.stderr
    assert _names(jars) == {"arcadedb-engine-26.11.1-SNAPSHOT.jar"}


def test_pinned_tag_that_differs_from_the_given_jars_warns(tmp_path):
    jars = tmp_path / "jars"
    source = _jars(tmp_path / "local", "arcadedb-engine-26.10.2.jar")
    out = _stage(jars, str(source), "26.11.1-SNAPSHOT")
    assert out.returncode == 0, out.stderr
    assert "26.11.1-SNAPSHOT" in out.stderr and "26.10.2" in out.stderr


def test_matching_or_latest_tag_does_not_warn(tmp_path):
    source = _jars(tmp_path / "local", "arcadedb-engine-26.10.2.jar")
    for tag in ("26.10.2", "latest"):
        out = _stage(tmp_path / f"jars-{tag}", str(source), tag)
        assert out.returncode == 0, out.stderr
        assert "⚠️" not in out.stderr
