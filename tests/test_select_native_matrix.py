"""The legs of the native test workflow per event: a pull request must stay small (the org shares a runner cap with upstream)."""

import importlib.util
import json
import re
import subprocess  # nosec B404 - runs this repository's script with sys.executable
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_native_matrix.py"
spec = importlib.util.spec_from_file_location("select_native_matrix", SCRIPT)
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)


def test_pull_request_builds_one_platform_and_tests_one_python():
    chosen = sel.select("pull_request")
    assert [b["key"] for b in chosen["build"]] == ["linux-x86_64"]
    assert [(t["key"], t["python"]) for t in chosen["test"]] == [
        ("linux-x86_64", "3.12")
    ]


def test_main_nightly_and_dispatch_build_all_four_platforms():
    for event in ("push", "schedule", "workflow_dispatch"):
        chosen = sel.select(event)
        assert [b["key"] for b in chosen["build"]] == [
            "linux-x86_64",
            "linux-aarch64",
            "macos-arm64",
            "windows-x64",
        ]
        assert len(chosen["test"]) == 4 * len(sel.MAIN_PYTHONS)


def test_a_release_tests_every_supported_python_on_every_platform():
    chosen = sel.select("release")
    assert len(chosen["build"]) == 4
    assert len(chosen["test"]) == 4 * 4
    assert {t["python"] for t in chosen["test"]} == {"3.11", "3.12", "3.13", "3.14"}


def test_pythons_are_the_classifiers_of_the_native_wheel():
    text = tomllib.loads((ROOT / "native" / "wheel" / "pyproject.toml").read_text())
    prefix = "Programming Language :: Python :: 3."
    pythons = [
        "3." + c.rsplit(".", 1)[1]
        for c in text["project"]["classifiers"]
        if c.startswith(prefix)
    ]
    assert pythons == sel.ALL_PYTHONS


def test_graalvm_is_community_edition_with_a_sha256_per_platform():
    for key, leg in sel.PLATFORMS.items():
        assert "graalvm-community" in leg["graalvm_asset"], key
        assert re.fullmatch(r"[0-9a-f]{64}", leg["graalvm_sha256"]), key
        assert "oracle" not in leg["graalvm_asset"].lower(), key


def test_the_script_prints_github_output_lines():
    out = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, str(SCRIPT), "pull_request"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert [line.split("=", 1)[0] for line in out] == ["build", "test", "pyver"]
    assert json.loads(out[0].split("=", 1)[1])[0]["key"] == "linux-x86_64"
    assert [leg["python"] for leg in json.loads(out[2].split("=", 1)[1])] == [
        "3.11",
        "3.13",
        "3.14",
        "3.14t",
    ]


def _pyver(event):
    return [
        (leg["python"], leg["freethreaded"], leg["polars"])
        for leg in sel.select(event)["pyver"]
    ]


def test_the_python_version_job_covers_what_the_full_parity_run_leaves_out():
    # a pull request tests 3.12 only: the other three Pythons and the free-threaded one run the cheap subset
    assert _pyver("pull_request") == [
        ("3.11", False, True),
        ("3.13", False, True),
        ("3.14", False, True),
        ("3.14t", True, False),
    ]
    # main, nightly and dispatch test 3.12 and 3.14 in full
    for event in ("push", "schedule", "workflow_dispatch"):
        assert _pyver(event) == [
            ("3.11", False, True),
            ("3.13", False, True),
            ("3.14t", True, False),
        ]
    # a release tests 3.11 to 3.14 in full: only the free-threaded build is left
    assert _pyver("release") == [("3.14t", True, False)]


def test_the_python_version_job_never_repeats_a_full_parity_leg_and_has_no_polars_on_a_free_threaded_python():
    for event in ("pull_request", "push", "schedule", "workflow_dispatch", "release"):
        chosen = sel.select(event)
        full = {t["python"] for t in chosen["test"]}
        assert not full & {leg["python"] for leg in chosen["pyver"]}, event
        assert any(leg["freethreaded"] for leg in chosen["pyver"]), event
        for leg in chosen["pyver"]:
            assert leg["polars"] is (not leg["freethreaded"]), leg


def test_the_release_does_not_need_the_python_version_job():
    workflows = ROOT / ".github" / "workflows"
    test_native = (workflows / "test-native.yml").read_text()
    release = (workflows / "release-native.yml").read_text()
    assert re.search(r"^  pyver:\n", test_native, re.M)
    # no job of either workflow waits for it
    for text in (test_native, release):
        for needs in re.findall(r"^    needs: (.*)$", text, re.M):
            assert "pyver" not in needs, needs
    # release-native.yml reads the markers and the wheels only, never the artifacts of the pyver job
    patterns = re.findall(r"^\s+pattern: (.*)$", release, re.M)
    assert patterns and all(
        p in ("native-passed-*", "native-wheel-*") for p in patterns
    ), patterns
