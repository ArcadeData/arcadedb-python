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

# Another distribution, as test data: the guards must refuse it (allow-old-name: scripts/check_no_embedded_name.py lets a guard test name it)
OTHER_DIST = "arcadedb-embedded"  # allow-old-name
OTHER_FILE_STEM = "arcadedb_embedded"  # allow-old-name

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
        # the WHEEL file of a platform wheel, as setup.py builds it (scripts/verify_wheel_tag.py reads it)
        wheel_tag = "-".join(file_name.split("-")[-3:])
        zf.writestr(
            f"{dist}-{TAG}.dist-info/WHEEL",
            newline.join(["Wheel-Version: 1.0", "Root-Is-Purelib: false", f"Tag: {wheel_tag}", ""]),
        )
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
    result = _guard(workflow, dist, tmp_path, [(file_name, OTHER_DIST, newline)])
    assert result.returncode != 0
    assert "Refusing to publish" in result.stdout + result.stderr


@pytest.mark.parametrize("workflow, dist", PUBLISHERS)
def test_another_name_in_the_file_name_is_refused(workflow, dist, tmp_path):
    result = _guard(workflow, dist, tmp_path, [(f"{OTHER_FILE_STEM}-{TAG}-py3-none-any", dist, "\r\n")])
    assert result.returncode != 0
    assert "Refusing to publish" in result.stdout + result.stderr


@pytest.mark.parametrize("workflow, dist", PUBLISHERS)
def test_one_wrong_wheel_among_right_ones_is_refused(workflow, dist, tmp_path):
    right = f"{dist.replace('-', '_')}-{TAG}"
    wheels = [
        (f"{right}-py3-none-manylinux_2_28_x86_64", dist, "\n"),
        (f"{right}-py3-none-win_amd64", dist, "\r\n"),
        (f"{right}-py3-none-macosx_11_0_arm64", OTHER_DIST, "\n"),
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
    """What the JVM "Verify wheel count, versions and sizes" step reads: a size script, the real tag script and 4 wheels (one for each platform)."""
    (root / "scripts").mkdir()
    (root / "scripts" / "verify_wheel_size.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    (root / "scripts" / "verify_wheel_tag.py").write_bytes((REPO / "scripts" / "verify_wheel_tag.py").read_bytes())
    dist = root / "dist"
    dist.mkdir()
    platforms = {
        "manylinux_2_28_x86_64": "\n",
        "manylinux_2_28_aarch64": "\n",
        "macosx_11_0_arm64": "\n",
        "win_amd64": "\r\n",  # built on Windows: CRLF, as in 26.11.1.dev0
    }
    for platform, newline in platforms.items():
        name = windows_metadata_name if (platform == "win_amd64" and windows_metadata_name) else "arcadedb-jvm"
        _wheel(dist, f"arcadedb_jvm-{TAG}-py3-none-{platform}", name, newline, top_level="arcadedb")
    return root


def test_the_jvm_wheel_check_accepts_crlf_metadata_from_the_windows_wheels(tmp_path):
    root = _jvm_release_tree(tmp_path)
    output = tmp_path / "github_output"
    result = _run(_script("release-jvm.yml", VERIFY_STEP), root, TAG=TAG, GITHUB_OUTPUT=str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "distribution arcadedb-jvm, import package arcadedb" in result.stdout
    assert "import-name=arcadedb" in output.read_text(encoding="utf-8")


def test_the_jvm_wheel_check_still_refuses_wheels_of_two_distributions(tmp_path):
    root = _jvm_release_tree(tmp_path, windows_metadata_name=OTHER_DIST)
    output = tmp_path / "github_output"
    result = _run(_script("release-jvm.yml", VERIFY_STEP), root, TAG=TAG, GITHUB_OUTPUT=str(output))
    assert result.returncode != 0
    assert "disagree on distribution name" in result.stdout


def _jvm_release_result(tmp_path: Path, mutate) -> subprocess.CompletedProcess:
    root = _jvm_release_tree(tmp_path)
    mutate(root / "dist")
    return _run(_script("release-jvm.yml", VERIFY_STEP), root, TAG=TAG, GITHUB_OUTPUT=str(tmp_path / "github_output"))


def test_the_jvm_wheel_check_wants_one_wheel_for_each_platform_and_no_more(tmp_path):
    # the layout of 26.11.1.dev0 and before: one cpXY wheel for each Python and platform, 16 files
    def per_python(dist: Path) -> None:
        for py in ("cp311", "cp312"):
            for platform in ("manylinux_2_28_x86_64", "manylinux_2_28_aarch64", "macosx_11_0_arm64", "win_amd64"):
                _wheel(dist, f"arcadedb_jvm-{TAG}-{py}-{py}-{platform}", "arcadedb-jvm", top_level="arcadedb")

    result = _jvm_release_result(tmp_path, per_python)
    assert result.returncode != 0
    assert "expected 4 wheels" in result.stdout


def test_the_jvm_wheel_check_refuses_a_platform_wheel_tagged_for_one_python(tmp_path):
    def retag(dist: Path) -> None:
        (dist / f"arcadedb_jvm-{TAG}-py3-none-win_amd64.whl").unlink()
        _wheel(dist, f"arcadedb_jvm-{TAG}-cp312-cp312-win_amd64", "arcadedb-jvm", "\r\n", top_level="arcadedb")

    result = _jvm_release_result(tmp_path, retag)
    assert result.returncode != 0
    assert "tagged cp312-cp312, expected py3-none" in result.stderr


def test_the_jvm_wheel_check_refuses_a_wheel_for_no_platform(tmp_path):
    def drop_a_platform(dist: Path) -> None:
        (dist / f"arcadedb_jvm-{TAG}-py3-none-macosx_11_0_arm64.whl").unlink()
        _wheel(dist, f"arcadedb_jvm-{TAG}-py3-none-any", "arcadedb-jvm", top_level="arcadedb")

    result = _jvm_release_result(tmp_path, drop_a_platform)
    assert result.returncode != 0
    assert "darwin/arm64: expected 1 wheel, got 0" in result.stdout
