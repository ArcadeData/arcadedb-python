"""The legs of the native workflow per event: a pull request must stay small (the org shares a runner cap with upstream)."""

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
    assert chosen["publish"] is False


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
        assert chosen["publish"] is False


def test_a_release_tag_tests_every_supported_python_on_every_platform():
    chosen = sel.select("tag")
    assert len(chosen["build"]) == 4
    assert len(chosen["test"]) == 4 * 4
    assert {t["python"] for t in chosen["test"]} == {"3.11", "3.12", "3.13", "3.14"}
    assert chosen["publish"] is True


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
    assert [line.split("=", 1)[0] for line in out] == ["build", "test", "publish"]
    assert json.loads(out[0].split("=", 1)[1])[0]["key"] == "linux-x86_64"
    assert out[2] == "publish=false"
