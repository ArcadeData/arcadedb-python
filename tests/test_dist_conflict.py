"""The `arcadedb` and `arcadedb-native` distributions both import as `arcadedb`.

pip overwrites one with the other without a message, so the package warns at import time when it
finds both distributions in the installed metadata. These tests need no engine and no JVM.
"""

import os
import subprocess  # nosec B404 - runs sys.executable on a one-line import
import sys
import warnings
from importlib import metadata

import pytest

from arcadedb import _dist_conflict


def lookup_with(*installed):
    """A stand-in for importlib.metadata.version that knows only the names given."""

    def lookup(name):
        if name in installed:
            return "26.11.1"
        raise metadata.PackageNotFoundError(name)

    return lookup


def test_distribution_names():
    assert _dist_conflict.DISTRIBUTIONS == ("arcadedb", "arcadedb-native")


def test_both_installed_warns():
    with pytest.warns(RuntimeWarning, match="arcadedb-native") as record:
        assert _dist_conflict.warn_if_both_installed(
            lookup_with("arcadedb", "arcadedb-native")
        )
    text = str(record[0].message)
    assert "'arcadedb'" in text
    assert "Install only one" in text


@pytest.mark.parametrize(
    "installed",
    [("arcadedb",), ("arcadedb-native",), ()],
    ids=["jvm", "native", "none"],
)
def test_one_or_none_installed_is_silent(installed):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _dist_conflict.warn_if_both_installed(lookup_with(*installed))


def test_unreadable_metadata_never_breaks_the_import():
    def broken(name):
        raise OSError("unreadable site-packages")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _dist_conflict.warn_if_both_installed(broken)


def fake_dist_info(site, name, version):
    """A minimal installed distribution: <name>-<version>.dist-info/METADATA."""
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )


def import_arcadedb(extra_site):
    """Import arcadedb in a fresh interpreter with `extra_site` ahead on sys.path; return stderr."""
    env = dict(os.environ)
    # sys.path of this run (installed wheel or source checkout) follows the fake site
    env["PYTHONPATH"] = os.pathsep.join([str(extra_site)] + [p for p in sys.path if p])
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-W", "default", "-c", "import arcadedb"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_import_warns_when_both_distributions_are_installed(tmp_path):
    site = tmp_path / "site"
    fake_dist_info(site, "arcadedb", "26.11.1")
    fake_dist_info(site, "arcadedb-native", "26.11.1")
    stderr = import_arcadedb(site)
    assert "RuntimeWarning" in stderr
    assert "arcadedb-native" in stderr
    assert "Install only one" in stderr


def test_import_is_silent_with_the_jvm_distribution_alone(tmp_path):
    site = tmp_path / "site"
    fake_dist_info(site, "arcadedb", "26.11.1")
    assert "Install only one" not in import_arcadedb(site)
