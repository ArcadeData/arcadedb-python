"""The distribution-name guards of the release workflows: whitespace and spelling variants pass, another distribution is refused.

The guards are Python in a heredoc inside release-native.yml and release-jvm.yml. These tests pull that code out of the workflow
files and run it on small synthetic wheels, so the code that ships is the code that is tested. The METADATA of a wheel built on
Windows has CRLF line endings, which left a carriage return on the Name value and stopped 26.11.1.dev0 before anything was uploaded.
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec B404 - test-controlled child processes, no untrusted input
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"

GUARD_STEP = "- name: Refuse unless every wheel is the distribution"
VERIFY_STEP = "- name: Verify wheel count, versions and sizes"
TAG = "26.11.1.dev0"

# (workflow file, distribution it publishes)
PUBLISHERS = [
    ("release-native.yml", "arcadedb"),
    ("release-jvm.yml", "arcadedb-jvm"),
]


def _script(workflow: str, step: str) -> str:
    """The python3 heredoc of the first step called `step` in the workflow file, dedented."""
    lines = (WORKFLOWS / workflow).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(step))
    begin = next(i for i in range(start, len(lines)) if "python3 -I - <<'PY'" in lines[i])
    end = next(i for i in range(begin + 1, len(lines)) if lines[i].strip() == "PY")
    return textwrap.dedent("\n".join(lines[begin + 1 : end])) + "\n"


def _wheel(
    directory: Path,
    file_name: str,
    meta_name: str,
    newline: str = "\n",
    top_level: str | None = None,
) -> Path:
    """A wheel with a METADATA (and a top_level.txt when given) written with the line ending `newline`."""
    dist = file_name.split("-")[0]
    path = directory / f"{file_name}.whl"
    meta = newline.join(
        ["Metadata-Version: 2.4", f"Name: {meta_name}", f"Version: {TAG}", "Summary: x", ""]
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{dist}-{TAG}.dist-info/METADATA", meta)
        if top_level is not None:
            zf.writestr(f"{dist}-{TAG}.dist-info/top_level.txt", top_level + newline)
    return path


def _run(script: str, cwd: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # nosec B603 - the interpreter running the tests, the script comes from this repo
        [sys.executable, "-I", "-"],
        input=script,
        cwd=cwd,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def _guard(workflow: str, dist: str, tmp_path: Path, wheels) -> subprocess.CompletedProcess:
    (tmp_path / "dist").mkdir()
    for file_name, meta_name, newline in wheels:
        _wheel(tmp_path / "dist", file_name, meta_name, newline)
    return _run(_script(workflow, GUARD_STEP), tmp_path, DIST_NAME=dist)


@pytest.mark.parametrize("workflow, dist", PUBLISHERS)
@pytest.mark.parametrize(
    "newline, suffix",
    [("\n", ""), ("\r\n", ""), ("\n", "  "), ("\r\n", " \t")],
    ids=["lf", "crlf", "trailing-spaces", "crlf-and-tab"],
)
def test_the_right_distribution_passes_whatever_ends_the_line(workflow, dist, newline, suffix, tmp_path):
    file_name = f"{dist.replace('-', '_')}-{TAG}-py3-none-any"
    result = _guard(workflow, dist, tmp_path, [(file_name, dist + suffix, newline)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"all distribution '{dist}'" in result.stdout


@pytest.mark.parametrize("workflow, dist", PUBLISHERS)
def test_a_pep_503_spelling_of_the_name_passes(workflow, dist, tmp_path):
    spelled = dist.upper().replace("-", "_")
    file_name = f"{dist.replace('-', '_')}-{TAG}-py3-none-any"
    result = _guard(workflow, dist, tmp_path, [(file_name, spelled, "\r\n")])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("workflow, dist", PUBLISHERS)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_another_name_in_the_metadata_is_refused(workflow, dist, newline, tmp_path):
    file_name = f"{dist.replace('-', '_')}-{TAG}-py3-none-any"
    result = _guard(workflow, dist, tmp_path, [(file_name, "arcadedb-embedded", newline)])
    assert result.returncode != 0
    assert "Refusing to publish" in result.stdout + result.stderr


@pytest.mark.parametrize("workflow, dist", PUBLISHERS)
def test_another_name_in_the_file_name_is_refused(workflow, dist, tmp_path):
    result = _guard(workflow, dist, tmp_path, [(f"arcadedb_embedded-{TAG}-py3-none-any", dist, "\r\n")])
    assert result.returncode != 0
    assert "Refusing to publish" in result.stdout + result.stderr


@pytest.mark.parametrize("workflow, dist", PUBLISHERS)
def test_one_wrong_wheel_among_right_ones_is_refused(workflow, dist, tmp_path):
    right = f"{dist.replace('-', '_')}-{TAG}"
    wheels = [
        (f"{right}-py3-none-manylinux_2_28_x86_64", dist, "\n"),
        (f"{right}-py3-none-win_amd64", dist, "\r\n"),
        (f"{right}-py3-none-macosx_11_0_arm64", "arcadedb-embedded", "\n"),
    ]
    result = _guard(workflow, dist, tmp_path, wheels)
    assert result.returncode != 0
    assert "macosx_11_0_arm64" in result.stdout


def test_the_native_workflow_refuses_a_jvm_wheel_and_the_other_way_round(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    jvm = tmp_path / "jvm"
    jvm.mkdir()
    as_jvm = _guard("release-native.yml", "arcadedb", native, [(f"arcadedb_jvm-{TAG}-py3-none-any", "arcadedb-jvm", "\n")])
    as_native = _guard("release-jvm.yml", "arcadedb-jvm", jvm, [(f"arcadedb-{TAG}-py3-none-any", "arcadedb", "\n")])
    assert as_jvm.returncode != 0
    assert as_native.returncode != 0


def _jvm_release_tree(root: Path, windows_metadata_name: str | None = None) -> Path:
    """What the JVM "Verify wheel count, versions and sizes" step reads: pyproject.toml, a size script and 16 wheels."""
    (root / "scripts").mkdir()
    (root / "scripts" / "verify_wheel_size.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    (root / "pyproject.toml").write_bytes((REPO / "pyproject.toml").read_bytes())
    dist = root / "dist"
    dist.mkdir()
    pythons = re.findall(r"Programming Language :: Python :: (3\.\d+)", (REPO / "pyproject.toml").read_text(encoding="utf-8"))
    platforms = {
        "manylinux_2_28_x86_64": "\n",
        "manylinux_2_28_aarch64": "\n",
        "macosx_11_0_arm64": "\n",
        "win_amd64": "\r\n",  # built on Windows: CRLF, as in 26.11.1.dev0
    }
    for py in pythons:
        tag = "cp" + py.replace(".", "")
        for platform, newline in platforms.items():
            name = windows_metadata_name if (platform == "win_amd64" and windows_metadata_name) else "arcadedb-jvm"
            _wheel(dist, f"arcadedb_jvm-{TAG}-{tag}-{tag}-{platform}", name, newline, top_level="arcadedb")
    return root


def test_the_jvm_wheel_check_accepts_crlf_metadata_from_the_windows_wheels(tmp_path):
    root = _jvm_release_tree(tmp_path)
    output = tmp_path / "github_output"
    result = _run(_script("release-jvm.yml", VERIFY_STEP), root, TAG=TAG, GITHUB_OUTPUT=str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "distribution arcadedb-jvm, import package arcadedb" in result.stdout
    assert "import-name=arcadedb" in output.read_text(encoding="utf-8")


def test_the_jvm_wheel_check_still_refuses_wheels_of_two_distributions(tmp_path):
    root = _jvm_release_tree(tmp_path, windows_metadata_name="arcadedb-embedded")
    output = tmp_path / "github_output"
    result = _run(_script("release-jvm.yml", VERIFY_STEP), root, TAG=TAG, GITHUB_OUTPUT=str(output))
    assert result.returncode != 0
    assert "disagree on distribution name" in result.stdout
