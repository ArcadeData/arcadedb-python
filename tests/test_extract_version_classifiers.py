"""A classifier jar beside the engine jar is not a second engine version.

``arcadedb-engine-(.+)\\.jar`` also matched ``arcadedb-engine-26.11.1-SNAPSHOT-tests.jar``,
read ``26.11.1-SNAPSHOT-tests`` as a version, and failed the build with "engine
jars of more than one version" on a valid jar set (#8). The upstream image
already ships classifier jars for other modules (``arcadedb-bolt-<v>-shaded.jar``).
"""

from __future__ import annotations

import importlib.util
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


@pytest.mark.parametrize("classifier", ["tests", "sources", "javadoc", "shaded", "all"])
@pytest.mark.parametrize("version", ["26.11.1-SNAPSHOT", "26.10.2", "26.11.1-RC1"])
def test_classifier_jar_is_not_a_version(tmp_path, version, classifier):
    for name in (
        f"arcadedb-engine-{version}.jar",
        f"arcadedb-engine-{version}-{classifier}.jar",
    ):
        (tmp_path / name).touch()
    assert mod.extract_raw_version_from_jars(tmp_path) == version


def test_classifier_jar_alone_is_no_engine_jar(tmp_path):
    (tmp_path / "arcadedb-engine-26.11.1-SNAPSHOT-sources.jar").touch()
    with pytest.raises(ValueError, match="arcadedb-engine"):
        mod.extract_raw_version_from_jars(tmp_path)


def test_two_real_versions_still_raise(tmp_path):
    for name in (
        "arcadedb-engine-26.11.1-SNAPSHOT.jar",
        "arcadedb-engine-26.11.1-SNAPSHOT-tests.jar",
        "arcadedb-engine-26.10.2.jar",
    ):
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="more than one version"):
        mod.extract_raw_version_from_jars(tmp_path)


@pytest.mark.parametrize(
    "classifier", ["all", "shaded", "jar-with-dependencies", "native"]
)
def test_classifier_jar_of_another_version_still_raises(tmp_path, classifier):
    # A classifier jar carries the engine too, and ships on the classpath: one of
    # another version is a second engine, not something to skip.
    for name in (
        "arcadedb-engine-26.11.1-SNAPSHOT.jar",
        f"arcadedb-engine-26.10.2-{classifier}.jar",
    ):
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="26.10.2.*26.11.1-SNAPSHOT"):
        mod.extract_raw_version_from_jars(tmp_path)


@pytest.mark.parametrize(
    "classifier", ["jar-with-dependencies", "native", "test-sources"]
)
def test_any_classifier_is_read_under_its_version(tmp_path, classifier):
    for name in (
        "arcadedb-engine-26.11.1-RC1.jar",
        f"arcadedb-engine-26.11.1-RC1-{classifier}.jar",
    ):
        (tmp_path / name).touch()
    assert mod.extract_raw_version_from_jars(tmp_path) == "26.11.1-RC1"
