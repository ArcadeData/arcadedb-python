"""Both wheels are typed packages (PEP 561): they ship the py.typed marker, and every public callable carries type hints.

Issue ArcadeData/arcadedb-python#476. The marker is what tells an editor or a type checker to read the hints; without it
the package counts as untyped. The same file runs on both builds: against the installed wheel in CI it checks the file list of
what was built, and in a source checkout it also checks that both package definitions name the marker.
"""

import inspect
import tomllib
from pathlib import Path

import pytest

import arcadedb
import arcadedb.aio

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(arcadedb.__file__).resolve().parent

# Called by the runtime or by Python, not by a user, so there is nothing to hint at the call site: nothing is exempt today.
EXEMPT = frozenset()


def test_the_installed_package_ships_the_py_typed_marker():
    assert (PACKAGE / "py.typed").is_file(), f"no py.typed in {PACKAGE}"


def test_both_package_definitions_name_the_marker():
    if (
        not (ROOT / "pyproject.toml").exists()
        or not (ROOT / "native" / "wheel").exists()
    ):
        return  # the installed-wheel container has no checkout to read
    jvm = tomllib.loads((ROOT / "pyproject.toml").read_text())
    native = tomllib.loads((ROOT / "native" / "wheel" / "pyproject.toml").read_text())
    assert "py.typed" in jvm["tool"]["setuptools"]["package-data"]["arcadedb"]
    assert "py.typed" in native["tool"]["setuptools"]["package-data"]["arcadedb"]
    assert (ROOT / "src" / "arcadedb" / "py.typed").is_file()


def _public_callables():
    """(qualified name, function) for every public function, and every public method and property of every public class."""
    targets = [getattr(arcadedb, n) for n in arcadedb.__all__ if hasattr(arcadedb, n)]
    targets.append(arcadedb.aio.AsyncDatabase)
    for target in targets:
        module = getattr(target, "__module__", "") or ""
        if not module.startswith("arcadedb"):
            continue
        if inspect.isclass(target):
            if target.__name__.startswith("_") or issubclass(target, BaseException):
                continue
            for name, member in vars(target).items():
                if name.startswith("_") and name not in (
                    "__init__",
                    "__enter__",
                    "__exit__",
                ):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    member = member.fget
                if inspect.isfunction(member):
                    yield f"{target.__name__}.{name}", member
        elif inspect.isfunction(target):
            yield target.__name__, target


def test_there_are_public_callables_to_check():
    names = [name for name, _ in _public_callables()]
    assert (
        "Database.query" in names
        and "open_database" in names
        and "VectorIndex.find_nearest" in names
    ), names


def test_every_public_callable_has_type_hints():
    missing = []
    for name, fn in _public_callables():
        if name in EXEMPT:
            continue
        sig = inspect.signature(fn)
        bare = [
            p.name
            for p in sig.parameters.values()
            if p.name not in ("self", "cls") and p.annotation is inspect.Parameter.empty
        ]
        if bare:
            missing.append(f"{name}: parameters without a hint: {bare}")
        if sig.return_annotation is inspect.Signature.empty and not name.endswith(
            ".__init__"
        ):
            missing.append(f"{name}: no return hint")
    assert not missing, "\n".join(missing)


@pytest.mark.parametrize(
    "name",
    ["create_database", "open_database", "database_exists"],
)
def test_the_entry_points_name_the_database_type(name):
    ret = inspect.signature(getattr(arcadedb, name)).return_annotation
    assert ret in (arcadedb.Database, "Database", bool), (name, ret)
