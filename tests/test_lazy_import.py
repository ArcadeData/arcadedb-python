"""`import arcadedb` does not import importlib.metadata or numpy itself (issue ArcadeData/arcadedb-python#477).

Both took a large share of the import: importlib.metadata for the check that only one of the two distributions is installed and for
`__version__`, numpy for the `_np` name of six modules. Both builds run this file. Each check is made in a fresh interpreter and
compares the modules loaded before `import arcadedb` with the modules loaded after it, so a dependency that imports numpy itself
(JPype does, when numpy is installed) does not count against arcadedb.
"""

import json
import subprocess  # nosec B404
import sys
import textwrap
from importlib import metadata
from importlib.util import find_spec

import pytest

import arcadedb
from arcadedb import _dist_conflict, _lazy_numpy


def _run(code, *args):
    proc = subprocess.run(  # nosec B603 - fixed argv, no shell, test-owned code string
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def test_import_loads_neither_importlib_metadata_nor_numpy_itself():
    out = _run("""
        import json, sys
        try:
            import jpype  # the JVM build imports it anyway; the native build has no JPype and it is skipped here
        except ImportError:
            pass
        watched = ("importlib.metadata", "numpy")
        before = {m: m in sys.modules for m in watched}
        import arcadedb
        after = {m: m in sys.modules for m in watched}
        print(json.dumps({"before": before, "after": after}))
        """)
    seen = json.loads(out.splitlines()[-1])
    assert seen["after"] == seen["before"], seen


def test_version_is_read_on_first_use_and_matches_the_distribution():
    out = _run("""
        import sys
        import arcadedb
        print("version" in vars(arcadedb), "__version__" in vars(arcadedb))
        value = arcadedb.__version__
        print("__version__" in vars(arcadedb), value)
        from arcadedb import __version__ as imported
        print(imported == value)
        """).splitlines()
    assert out[0] == "False False"
    assert out[1].startswith("True ")
    assert out[2] == "True"
    own = _dist_conflict.DISTRIBUTIONS[arcadedb.__backend__]
    assert out[1].split(" ", 1)[1] == arcadedb.__version__
    try:
        assert arcadedb.__version__ == metadata.version(own)
    except (
        metadata.PackageNotFoundError
    ):  # a source checkout: the build-time file or the placeholder
        assert arcadedb.__version__


def test_an_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        arcadedb.nope


def test_the_distribution_scan_agrees_with_importlib_metadata():
    for backend, name in _dist_conflict.DISTRIBUTIONS.items():
        try:
            metadata.version(name)
            expected = True
        except metadata.PackageNotFoundError:
            expected = False
        assert (name in _dist_conflict.installed((backend,))) is expected, name


def test_the_distribution_scan_finds_a_dist_info_on_sys_path(tmp_path, monkeypatch):
    (tmp_path / "arcadedb_jvm-26.11.1.dist-info").mkdir()
    (
        tmp_path / "arcadedb-jvm-9.dist-info"
    ).mkdir()  # not the normalized name of a wheel: not a match for "arcadedb"
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _dist_conflict._dist_info_present("arcadedb-jvm") is True
    only_wrong_name = tmp_path / "other"
    only_wrong_name.mkdir()
    (only_wrong_name / "arcadedb-jvm-9.dist-info").mkdir()
    monkeypatch.setattr(sys, "path", [str(only_wrong_name)])
    assert _dist_conflict._dist_info_present("arcadedb") is False


def test_a_zip_on_sys_path_falls_back_to_importlib_metadata(tmp_path, monkeypatch):
    archive = tmp_path / "libs.zip"
    archive.write_bytes(b"PK")
    monkeypatch.setattr(sys, "path", [str(archive)])
    assert _dist_conflict._dist_info_present("arcadedb") is None
    assert (
        _dist_conflict.installed(("native",)) == []
    )  # asked importlib.metadata: nothing there


def test_the_stand_in_is_none_exactly_when_numpy_is_not_installed(monkeypatch):
    monkeypatch.setattr(_lazy_numpy, "_HAS_NUMPY", False)
    assert _lazy_numpy.lazy_numpy() is None
    monkeypatch.setattr(_lazy_numpy, "_HAS_NUMPY", True)
    assert _lazy_numpy.lazy_numpy() is _lazy_numpy._LAZY


def test_value_tests_import_nothing_and_array_calls_import_numpy():
    out = _run("""
        import json, sys
        from arcadedb._lazy_numpy import lazy_numpy
        np = lazy_numpy()
        if np is None:
            print(json.dumps({"installed": False}))
            raise SystemExit
        # only builds that do not import numpy themselves are compared (JPype does when numpy is installed)
        had = "numpy" in sys.modules
        answers = [isinstance(1, np.ndarray), isinstance(1.5, np.floating), isinstance(1, np.generic), issubclass(int, np.generic)]
        after_tests = "numpy" in sys.modules
        array = np.asarray([1, 2, 3])
        import numpy
        print(json.dumps({"installed": True, "had": had, "answers": answers, "after_tests": after_tests,
                          "after_call": "numpy" in sys.modules, "is_ndarray": isinstance(array, np.ndarray),
                          "same_class": np.ndarray is numpy.ndarray, "real": bool(np.float32 is numpy.float32)}))
        """)
    seen = json.loads(out.splitlines()[-1])
    if not seen["installed"]:
        return
    assert seen["answers"] == [False, False, False, False]
    assert seen["after_tests"] == seen["had"]
    assert seen["after_call"] is True
    assert seen["is_ndarray"] and seen["same_class"] and seen["real"]


def test_a_numpy_returning_call_still_returns_numpy(tmp_path):
    if find_spec("numpy") is None:
        return  # without numpy to_columns() returns None: nothing numpy to check
    out = _run(
        """
        import sys
        import arcadedb
        with arcadedb.create_database(sys.argv[1]) as db:
            db.command("sql", "CREATE DOCUMENT TYPE T")
            with db.transaction():
                db.command("sql", "INSERT INTO T SET n = 1")
            columns = db.query("sql", "SELECT n FROM T").to_columns()
            print(type(columns["n"]).__module__)
        """,
        str(tmp_path / "db"),
    )
    assert out.splitlines()[-1] == "numpy"
