"""The package version comes from the ArcadeDB engine jar that ships in the wheel.

This repository has no ArcadeDB pom.xml to read, so the version is taken from the
``arcadedb-engine-<version>.jar`` that the build pulls out of the
``arcadedata/arcadedb`` image (``latest`` unless ARCADEDB_TAG pins one).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_version.py"


def _load():
    spec = importlib.util.spec_from_file_location("extract_version", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load()


def _jars(directory: Path, *names: str) -> Path:
    """A jars directory holding empty files with these names."""
    for name in names:
        (directory / name).touch()
    return directory


def test_snapshot_engine_jar_gives_raw_version(tmp_path):
    d = _jars(
        tmp_path,
        "arcadedb-engine-26.11.1-SNAPSHOT.jar",
        "arcadedb-server-26.11.1-SNAPSHOT.jar",
        "jline-3.30.0.jar",
    )
    assert mod.extract_raw_version_from_jars(d) == "26.11.1-SNAPSHOT"
    assert mod.extract_version(d, fmt="docker") == "26.11.1-SNAPSHOT"


def test_snapshot_converts_to_dev(tmp_path, monkeypatch):
    # The tracker lives in tmp_path so the repository's own tracker is untouched.
    monkeypatch.setattr(
        mod, "get_dev_version_tracker_path", lambda: tmp_path / "t.json"
    )
    d = _jars(tmp_path, "arcadedb-engine-26.11.1-SNAPSHOT.jar")
    assert mod.extract_version(d) == "26.11.1.dev0"


def test_release_and_rc_and_post():
    assert mod.maven_to_pep440("26.10.2") == "26.10.2"
    assert mod.maven_to_pep440("26.10.2", patch_version=1) == "26.10.2.post1"
    assert mod.maven_to_pep440("26.11.1-RC1") == "26.11.1rc1"


def test_no_engine_jar_raises(tmp_path):
    d = _jars(tmp_path, "arcadedb-server-26.11.1-SNAPSHOT.jar")
    with pytest.raises(ValueError, match="arcadedb-engine"):
        mod.extract_raw_version_from_jars(d)


def test_two_engine_versions_raise(tmp_path):
    d = _jars(
        tmp_path, "arcadedb-engine-26.11.1-SNAPSHOT.jar", "arcadedb-engine-26.10.2.jar"
    )
    with pytest.raises(
        ValueError, match="26.10.2.*26.11.1-SNAPSHOT|26.11.1-SNAPSHOT.*26.10.2"
    ):
        mod.extract_raw_version_from_jars(d)


def test_cli_prints_docker_format(tmp_path):
    d = _jars(tmp_path, "arcadedb-engine-26.11.1-SNAPSHOT.jar")
    out = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(d), "--format=docker"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "26.11.1-SNAPSHOT"
