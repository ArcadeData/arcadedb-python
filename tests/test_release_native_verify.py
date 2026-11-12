"""The `verify` job of release-native.yml publishes nothing unless the `pyver` legs passed too (ArcadeData/arcadedb-python#349).

The job's check is bash inside the workflow file. These tests cut the shipped step out of release-native.yml and run it with bash on small
`passed/` and `wheels/` directories, as the job finds them after the artifact download, so the code that ships is the code that is tested.
The file is not collected on Windows (tests/conftest.py WINDOWS_NOT_APPLICABLE): the step runs in the bash of a Linux runner.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - test-controlled child processes, no untrusted input
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / ".github" / "workflows" / "release-native.yml"
STEP = "- name: Every required platform passed on every Python"

PLATFORM_TAGS = {
    "linux-x86_64": "manylinux_2_28_x86_64",
    "linux-aarch64": "manylinux_2_28_aarch64",
    "macos-arm64": "macosx_11_0_arm64",
    "windows-x64": "win_amd64",
}
PYTHONS = ["3.11", "3.12", "3.13", "3.14"]
PYVER_PYTHONS = ["3.15", "3.14t"]


def _script() -> str:
    """The `run: |` block of the verify step, dedented."""
    lines = RELEASE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(STEP))
    begin = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[begin]) - len(lines[begin].lstrip())
    body = []
    for line in lines[begin + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)) + "\n"


def _wheel(key: str, version: str) -> str:
    return f"arcadedb-{version}-py3-none-{PLATFORM_TAGS[key]}.whl"


def _stage(root: Path, version: str, drop=(), pyver_wheel: str | None = None) -> None:
    """What the job finds after the downloads: a marker per platform and Python, the wheels, and a marker per pyver leg."""
    passed, wheels = root / "passed", root / "wheels"
    passed.mkdir()
    wheels.mkdir()
    for key in PLATFORM_TAGS:
        name = _wheel(key, version)
        (wheels / name).write_bytes(b"wheel")
        for py in PYTHONS:
            (passed / f"{key}-py{py}.ok").write_text(name + "\n")
    for py in PYVER_PYTHONS:
        (passed / f"pyver-py{py}.ok").write_text(
            (pyver_wheel or _wheel("linux-x86_64", version)) + "\n"
        )
    for marker in drop:
        (passed / marker).unlink()


def _verify(root: Path, version: str, final: bool):
    output = root / "github_output"
    output.write_text("")
    env = {
        **os.environ,
        "VERSION": version,
        "FINAL": "true" if final else "false",
        "BAR": "true",
        "GITHUB_OUTPUT": str(output),
    }
    proc = subprocess.run(  # nosec B603 B607 - fixed argv, the script is this repository's own workflow step
        ["bash", "-c", _script()],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    dist = sorted(p.name for p in (root / "dist").glob("*.whl"))
    return proc, output.read_text(), dist


KINDS = [("26.99.1.dev0", False), ("26.99.1", True), ("26.99.1.post1", True)]


@pytest.mark.parametrize("version,final", KINDS)
def test_every_leg_passed_the_release_is_verified_and_all_four_wheels_are_collected(
    tmp_path, version, final
):
    _stage(tmp_path, version)
    proc, output, dist = _verify(tmp_path, version, final)
    assert proc.returncode == 0, proc.stderr
    assert "ok=true" in output
    assert dist == sorted(_wheel(key, version) for key in PLATFORM_TAGS)


@pytest.mark.parametrize("version,final", KINDS)
@pytest.mark.parametrize("py", PYVER_PYTHONS)
def test_a_pyver_leg_that_did_not_pass_stops_every_kind_of_release(
    tmp_path, version, final, py
):
    # a failed leg uploads no marker (test-native.yml reaches the marker step only after the gate), so the marker is missing
    _stage(tmp_path, version, drop=[f"pyver-py{py}.ok"])
    proc, output, dist = _verify(tmp_path, version, final)
    assert proc.returncode != 0, "the release went on although a pyver leg did not pass"
    assert f"pyver leg on Python {py} did not pass" in proc.stderr
    assert "ok=true" not in output
    assert dist == []


def test_a_pyver_leg_that_tested_another_wheel_than_the_published_one_stops_the_release(
    tmp_path,
):
    version = "26.99.1.dev0"
    _stage(
        tmp_path,
        version,
        pyver_wheel=f"arcadedb-26.99.0-py3-none-{PLATFORM_TAGS['linux-x86_64']}.whl",
    )
    proc, output, dist = _verify(tmp_path, version, final=False)
    assert proc.returncode != 0
    assert "tested another wheel" in proc.stderr
    assert "ok=true" not in output
    assert dist == []


def test_a_platform_leg_that_did_not_pass_still_stops_the_release(tmp_path):
    # the fixture is good enough to be verified, so the refusal above comes from the pyver check and not from a broken fixture
    version = "26.99.1.dev0"
    _stage(tmp_path, version, drop=["macos-arm64-py3.13.ok"])
    proc, output, dist = _verify(tmp_path, version, final=False)
    assert proc.returncode != 0
    assert "required leg macos-arm64 on Python 3.13" in proc.stderr
    assert "ok=true" not in output
