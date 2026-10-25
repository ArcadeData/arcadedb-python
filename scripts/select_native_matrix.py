#!/usr/bin/env python3
"""Choose the legs of the native workflow (.github/workflows/native.yml) for one event.

A native image build takes about 1 to 2.5 hours per platform and the ArcadeData organisation shares a runner cap with
upstream ArcadeDB, so a pull request builds one platform (linux-x86_64) and tests one Python (3.12), and only when
native paths changed (the workflow's `paths` filter). Push to main, the nightly run and a dispatch build all four
platforms and test Python 3.12 and 3.14 on each. A release tag tests every supported Python (3.11 to 3.14) on all four.

usage: select_native_matrix.py EVENT [--tag-kind final|dev|none]     EVENT: pull_request | push | schedule | workflow_dispatch | tag

Prints `build=<json>`, `test=<json>` and `publish=<true|false>` lines (GITHUB_OUTPUT format). `publish` is true for a
tag only. The GraalVM CE archives are pinned by sha256 per platform (GraalVM Community Edition only, never Oracle GraalVM).
"""

from __future__ import annotations

import argparse
import json
import sys

GRAALVM_VERSION = "25.4.4.1.1"

# key -> the build leg. `graalvm_sha256` is the release asset's digest (GitHub API), equal to the release's own <asset>.sha256.
PLATFORMS: dict[str, dict] = {
    "linux-x86_64": {
        "kind": "linux",
        "runner": "ubuntu-latest",
        "container": "quay.io/pypa/manylinux_2_34_x86_64:latest",
        "graalvm_asset": "graalvm-community-jdk-25i4-25.0.4.1.1_linux-x64_bin.tar.gz",
        "graalvm_sha256": "05ccbbe783210b6886ff7b08fcd0b061c5dce4852b05db87284fc0e24abb08e2",
        "plat_re": "manylinux_2_34_x86_64",
        "xmx": "12g",
        "ni_extra": "",
        "macos_target": "",
        "build_minutes": 150,
        "test_minutes": 120,
    },
    "linux-aarch64": {
        "kind": "linux",
        "runner": "ubuntu-24.04-arm",
        "container": "quay.io/pypa/manylinux_2_34_aarch64:latest",
        "graalvm_asset": "graalvm-community-jdk-25i4-25.0.4.1.1_linux-aarch64_bin.tar.gz",
        "graalvm_sha256": "e5f5e2f59643cf96765c741dc00b206f86c69c8c1bf843fe26050c871e0e2dbc",
        "plat_re": "manylinux_2_34_aarch64",
        "xmx": "12g",
        "ni_extra": "",
        "macos_target": "",
        "build_minutes": 150,
        "test_minutes": 120,
    },
    "macos-arm64": {
        "kind": "macos",
        "runner": "macos-15",
        "container": "",
        "graalvm_asset": "graalvm-community-jdk-25i4-25.0.4.1.1_macos-aarch64_bin.tar.gz",
        "graalvm_sha256": "4ec9932aa4aab1afca0ea31982d32d6f956615c61fe21a227b49756b25adda4f",
        "plat_re": "macosx_[0-9]+_[0-9]+_arm64",
        # 7 GB runner: a capped builder heap and two threads; -mmacosx-version-min because the link ignores
        # MACOSX_DEPLOYMENT_TARGET alone
        "xmx": "5g",
        "ni_extra": "-H:NumberOfThreads=2 -H:NativeLinkerOption=-mmacosx-version-min=11.0",
        "macos_target": "11.0",
        "build_minutes": 180,
        "test_minutes": 150,
    },
    "windows-x64": {
        "kind": "windows",
        "runner": "windows-latest",
        "container": "",
        "graalvm_asset": "graalvm-community-jdk-25i4-25.0.4.1.1_windows-x64_bin.zip",
        "graalvm_sha256": "2733ea1331f1a98b05dd55a768b07347051dad48f030dbbfa191b11a2ccf4144",
        "plat_re": "win_amd64",
        "xmx": "12g",
        "ni_extra": "",
        "macos_target": "",
        "build_minutes": 150,
        "test_minutes": 150,
    },
}

# Python versions the wheel supports (the classifiers of native/wheel/pyproject.toml).
ALL_PYTHONS = ["3.11", "3.12", "3.13", "3.14"]

# Platforms whose parity run must pass before a pre-release (A.B.C.devN) is published. Windows is the newest wheel and is
# informational for a pre-release; a FINAL release needs all four (see ALL_PLATFORMS and native.yml).
PRERELEASE_REQUIRED = ["linux-x86_64", "linux-aarch64", "macos-arm64"]
ALL_PLATFORMS = list(PLATFORMS)

PULL_REQUEST_PLATFORMS = ["linux-x86_64"]
PULL_REQUEST_PYTHONS = ["3.12"]
MAIN_PYTHONS = ["3.12", "3.14"]


def select(event: str) -> dict:
    if event == "pull_request":
        platforms, pythons = PULL_REQUEST_PLATFORMS, PULL_REQUEST_PYTHONS
    elif event in ("push", "schedule", "workflow_dispatch"):
        platforms, pythons = ALL_PLATFORMS, MAIN_PYTHONS
    elif event == "tag":
        platforms, pythons = ALL_PLATFORMS, ALL_PYTHONS
    else:
        raise ValueError(f"unknown event {event!r}")
    build = [{"key": k, **PLATFORMS[k]} for k in platforms]
    test = [
        {
            "key": k,
            "runner": PLATFORMS[k]["runner"],
            "python": py,
            "test_minutes": PLATFORMS[k]["test_minutes"],
        }
        for k in platforms
        for py in pythons
    ]
    return {"build": build, "test": test, "publish": event == "tag"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "event",
        choices=["pull_request", "push", "schedule", "workflow_dispatch", "tag"],
    )
    a = ap.parse_args(argv)
    chosen = select(a.event)
    print(f"build={json.dumps(chosen['build'], separators=(',', ':'))}")
    print(f"test={json.dumps(chosen['test'], separators=(',', ':'))}")
    print(f"publish={'true' if chosen['publish'] else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
