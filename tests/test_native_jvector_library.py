"""The native `arcadedb` wheel carries jvector's Highway library for the architecture it is built for.

The jvector 4.0.2 jar has `libjvector-x86_64.so` and `libjvector-aarch64.so` and nothing for macOS or Windows. The Linux x86-64
wheel shipped the first, the Linux aarch64 wheel shipped none (#144). These tests check the builder's ship list and the build
script's copy step, and ArcadeNative's comment on the provider default.

Repository tooling: no engine, no JVM and no native library needed.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_WHEEL = ROOT / "native" / "wheel" / "build_wheel.py"
BUILD_IMAGE = ROOT / "native" / "scripts" / "build-image.sh"
ARCADE_NATIVE = (
    ROOT
    / "native"
    / "src"
    / "main"
    / "java"
    / "org"
    / "humemai"
    / "nativelib"
    / "ArcadeNative.java"
)


def _build_wheel():
    spec = importlib.util.spec_from_file_location("native_build_wheel", BUILD_WHEEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "sys_platform, arch, libname, expected",
    [
        (
            "linux",
            "x86_64",
            "libarcadedb.so",
            ["libarcadedb.so", "libjvector-x86_64.so"],
        ),
        (
            "linux",
            "aarch64",
            "libarcadedb.so",
            ["libarcadedb.so", "libjvector-aarch64.so"],
        ),
        ("darwin", "aarch64", "libarcadedb.dylib", ["libarcadedb.dylib"]),
        ("win32", "x86_64", "libarcadedb.dll", ["libarcadedb.dll"]),
        ("linux", "riscv64", "libarcadedb.so", ["libarcadedb.so"]),
    ],
)
def test_the_wheel_ships_the_jvector_library_of_its_linux_architecture(
    sys_platform, arch, libname, expected
):
    assert _build_wheel().libraries_to_ship(sys_platform, arch, libname) == expected


def test_the_host_architecture_names_map_to_a_library_the_loader_looks_for():
    names = _build_wheel().JVECTOR_BY_ARCH
    assert names == {
        "x86_64": "libjvector-x86_64.so",
        "aarch64": "libjvector-aarch64.so",
    }
    loader = (
        ROOT
        / "native/src/main/java/io/github/jbellis/jvector/vector/cnative/LibraryLoader.java"
    ).read_text()
    for name in names.values():
        assert f'"/{name}"' in loader


def test_the_image_script_copies_the_library_of_the_linux_architecture():
    text = BUILD_IMAGE.read_text()
    assert "x86_64) JARCH=x86_64" in text
    assert re.search(r"aarch64 \| arm64\) JARCH=aarch64", text)
    assert 'unzip -p "$JAR" "$J" > "$OUT/libjvector-$JARCH.so"' in text
    assert 'LIBS+=("$OUT/libjvector-$JARCH.so")' in text
    # the copy no longer waits for uname -m to say x86_64
    assert '[ "$(uname -m)" = x86_64 ]' not in text


def test_the_provider_comment_states_the_real_default():
    text = ARCADE_NATIVE.read_text()
    assert "unset: jvector chooses" not in text
    assert "DefaultVectorizationProvider" in text
    assert "ARCADEDB_VECTOR_PROVIDER=native" in text
