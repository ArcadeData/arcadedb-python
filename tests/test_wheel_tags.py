"""The JVM wheel is tagged py3-none-<platform>: one wheel for each platform, not one for each Python version.

The wheel bundles a platform-specific JRE, but nothing in it is compiled against CPython, so setup.py keeps the platform tag and
drops the interpreter and ABI tags (a cpXY-cpXY wheel would be 16 files of about 65 MB for a release where 4 do). Which Pythons it
installs on is `requires-python` and the `jpype1` dependency, not the tag; tests/test_select_matrix.py pins `requires-python`.

In a source checkout the test builds the wheel with setuptools (no JARs, no JRE needed). In the wheel's test container there is no
``setup.py``, so it reads the WHEEL file of the installed distribution instead.
"""

import importlib.metadata
import importlib.util
import shutil
import subprocess  # nosec B404 - test-controlled child process
import sys
import zipfile
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parents[1]


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_wheel_tag", PYTHON_DIR / "scripts" / "verify_wheel_tag.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build(tmp_path: Path, *plat_args: str) -> Path:
    """The wheel `setup.py bdist_wheel` makes from a copy of the sources, with a stand-in JRE file so it is a platform wheel."""
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(PYTHON_DIR / "pyproject.toml", project)
    shutil.copy(PYTHON_DIR / "setup.py", project)
    shutil.copytree(
        PYTHON_DIR / "src" / "arcadedb",
        project / "src" / "arcadedb",
        ignore=shutil.ignore_patterns("__pycache__", "jars", "jre"),
    )
    shutil.copy(PYTHON_DIR / "README.md", project)
    dist = tmp_path / "dist"
    subprocess.run(  # nosec B603 - fixed argv (sys.executable setup.py bdist_wheel), no shell
        [sys.executable, "setup.py", "-q", "bdist_wheel", "--dist-dir", str(dist), *plat_args],
        cwd=project,
        check=True,
        capture_output=True,
    )
    (wheel,) = dist.glob("*.whl")
    return wheel


@pytest.mark.parametrize(
    "plat",
    ["manylinux_2_28_x86_64", "manylinux_2_28_aarch64", "win_amd64"],
)
def test_the_build_gives_a_py3_none_platform_wheel(tmp_path, plat):
    if not (PYTHON_DIR / "setup.py").exists():
        return  # the wheel's test container: test_installed_distribution_is_py3_none covers it
    wheel = _build(tmp_path, "--plat-name", plat)
    # not cpXY-cpXY-<platform>, whatever Python builds it
    assert wheel.name.endswith(f"-py3-none-{plat}.whl"), wheel.name
    assert _verifier().problems(wheel) == []
    with zipfile.ZipFile(wheel) as zf:
        (name,) = [n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")]
        text = zf.read(name).decode()
    assert f"Tag: py3-none-{plat}" in text
    assert "Root-Is-Purelib: false" in text


def test_the_build_without_a_platform_name_keeps_the_hosts(tmp_path):
    if not (PYTHON_DIR / "setup.py").exists():
        return
    wheel = _build(tmp_path)
    assert "-py3-none-" in wheel.name and not wheel.name.endswith("-any.whl"), wheel.name


def test_the_verifier_refuses_the_old_per_python_tag_and_the_any_tag(tmp_path):
    verifier = _verifier()
    for name, expected in [
        ("a-1-cp312-cp312-win_amd64.whl", "tagged cp312-cp312, expected py3-none"),
        ("a-1-py3-none-any.whl", "platform tag 'any'"),
    ]:
        wheel = tmp_path / name
        tag = "-".join(name[: -len(".whl")].split("-")[-3:])
        with zipfile.ZipFile(wheel, "w") as zf:
            zf.writestr("a-1.dist-info/WHEEL", f"Root-Is-Purelib: false\nTag: {tag}\n")
        assert any(expected in line for line in verifier.problems(wheel)), name


def test_installed_distribution_is_py3_none():
    try:
        dist = importlib.metadata.distribution("arcadedb-jvm")
    except importlib.metadata.PackageNotFoundError:
        return  # not installed as the arcadedb-jvm wheel (a source checkout run)
    text = dist.read_text("WHEEL")
    if text is None:
        return  # an editable install has no WHEEL tag of a built wheel
    lines = [line.strip() for line in text.splitlines()]
    tags = [line.split(":", 1)[1].strip() for line in lines if line.startswith("Tag:")]
    assert len(tags) == 1 and tags[0].startswith("py3-none-"), tags
    assert tags[0] != "py3-none-any", tags
    assert "Root-Is-Purelib: false" in lines
