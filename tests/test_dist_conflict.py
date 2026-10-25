"""The `arcadedb` (native) and `arcadedb-jvm` (JVM) distributions both import as `arcadedb`.

pip overwrites one with the other without a message, so each build refuses to import when the metadata of
the other distribution is installed. `arcadedb.__backend__` says which build is running. These tests need no
engine and no JVM.
"""

import os
import subprocess  # nosec B404 - runs sys.executable on a one-line import
import sys
import warnings
from importlib import metadata

import pytest

import arcadedb
from arcadedb import _dist_conflict


def lookup_with(*installed):
    """A stand-in for importlib.metadata.version that knows only the names given."""

    def lookup(name):
        if name in installed:
            return "26.11.1"
        raise metadata.PackageNotFoundError(name)

    return lookup


def test_distribution_names():
    assert _dist_conflict.DISTRIBUTIONS == {"native": "arcadedb", "jvm": "arcadedb-jvm"}


def test_backend_is_one_of_the_two_builds():
    assert arcadedb.__backend__ in ("jvm", "native")
    assert "__backend__" in arcadedb.__all__


@pytest.mark.parametrize(
    "backend,own,other",
    [("jvm", "arcadedb-jvm", "arcadedb"), ("native", "arcadedb", "arcadedb-jvm")],
)
def test_the_other_distribution_installed_refuses_the_import(backend, own, other):
    with pytest.raises(ImportError) as info:
        _dist_conflict.check(backend, lookup_with(own, other))
    assert isinstance(info.value, _dist_conflict.DistributionConflictError)
    text = str(info.value)
    # names both packages, how to fix it, and where to read more
    assert f"'{own}'" in text and f"'{other}'" in text
    assert "pip uninstall arcadedb arcadedb-jvm" in text
    assert "which-package" in text


@pytest.mark.parametrize("backend", ["jvm", "native"])
def test_own_distribution_alone_or_none_is_silent(backend):
    own = _dist_conflict.DISTRIBUTIONS[backend]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _dist_conflict.check(backend, lookup_with(own))
        _dist_conflict.check(
            backend, lookup_with()
        )  # a source checkout: neither is installed


@pytest.mark.parametrize("backend", ["jvm", "native"])
def test_unreadable_metadata_never_breaks_the_import(backend):
    def broken(name):
        raise OSError("unreadable site-packages")

    _dist_conflict.check(backend, broken)


def test_an_unknown_backend_is_a_programming_error():
    with pytest.raises(ValueError, match="'jvm' or 'native'"):
        _dist_conflict.check("python", lookup_with())


def fake_dist_info(site, name, version):
    """A minimal installed distribution: <name>-<version>.dist-info/METADATA."""
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )


def import_arcadedb(extra_site):
    """Import arcadedb in a fresh interpreter with `extra_site` ahead on sys.path; return the CompletedProcess."""
    env = dict(os.environ)
    # sys.path of this run (installed wheel or source checkout) follows the fake site
    env["PYTHONPATH"] = os.pathsep.join([str(extra_site)] + [p for p in sys.path if p])
    return subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-W", "default", "-c", "import arcadedb"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def own_backend():
    """The build the installed package is: fixed at build time, not switched by ARCADEDB_BACKEND."""
    return arcadedb._native_backend.distribution_backend()


def sibling():
    """The distribution of the other build than the one under test."""
    return _dist_conflict.DISTRIBUTIONS[_dist_conflict.other_backend(own_backend())]


def test_import_fails_when_the_other_distribution_is_installed_too(tmp_path):
    site = tmp_path / "site"
    fake_dist_info(site, sibling(), "26.11.1")
    result = import_arcadedb(site)
    assert result.returncode != 0
    assert "DistributionConflictError" in result.stderr
    assert "pip uninstall arcadedb arcadedb-jvm" in result.stderr
    assert "'arcadedb'" in result.stderr and "'arcadedb-jvm'" in result.stderr


def test_import_is_silent_when_only_this_distribution_is_installed(tmp_path):
    site = tmp_path / "site"
    fake_dist_info(site, _dist_conflict.DISTRIBUTIONS[own_backend()], "26.11.1")
    result = import_arcadedb(site)
    assert result.returncode == 0, result.stderr
    assert "pip uninstall" not in result.stderr
