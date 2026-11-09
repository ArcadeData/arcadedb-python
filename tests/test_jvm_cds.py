"""The bundled JRE ships a class-data-sharing (CDS) archive that the JVM uses at start (#313).

A JRE made by jlink has no archive, so every start loaded and verified the JDK's classes again. ``scripts/dump_cds_archive.sh`` dumps the
JVM's default archive as the last step of the JRE build, with ``-XX:+UseCompactObjectHeaders`` (which ``jvm.py`` sets: the JVM ignores an
archive dumped with another setting, and it does so without an error). These tests run on the installed wheel, on every platform and
Python of the JVM matrix: the archive is in the wheel, the JVM maps it with the flags ``start_jvm`` builds (``-Xshare:on`` turns an
unusable archive into a failed start; ``-Xlog:class+load`` shows a JDK class coming from the shared objects file), and an archive does not
serve a JVM with other settings. The archive is the JVM's own default one, so a missing or mismatched file only means a start without it;
the JDK AOT cache (``aot.py``, ``test_aot.py``) stays opt-in.
"""

import subprocess  # nosec B404 - runs the bundled JRE's java with fixed arguments
import sys

from arcadedb import jvm


def _java():
    name = "java.exe" if sys.platform == "win32" else "java"
    return str(jvm._extract_runtime_resource("jre") / "bin" / name)


def _flags(tmp_path):
    """The JVM arguments ``start_jvm`` passes, with a heap that keeps compressed oops on whatever the RAM of this host is."""
    return jvm._build_jvm_args(
        heap_size="1g", disable_xml_limits=False, jvm_args=None, log_dir=tmp_path
    )


def _run(*args):
    return subprocess.run(  # nosec B603 - the bundled java, arguments of this file
        [_java(), *args], capture_output=True, text=True, timeout=300
    )


def test_the_wheel_ships_the_archive():
    archives = list(jvm._extract_runtime_resource("jre").rglob("*.jsa"))
    assert archives, "no .jsa file in the bundled JRE"
    assert all(a.stat().st_size > 1_000_000 for a in archives)


def test_the_jvm_maps_the_archive_with_the_flags_the_wheel_starts_with(tmp_path):
    flags = _flags(tmp_path)
    assert "-XX:+UseCompactObjectHeaders" in flags
    # -Xshare:on: no usable archive is a failed start, not a quiet start without it
    result = _run(*flags, "-Xshare:on", "-version")
    assert result.returncode == 0, result.stderr
    result = _run(*flags, "-Xshare:auto", "-Xlog:class+load:stdout", "-version")
    assert result.returncode == 0, result.stderr
    assert "java.lang.Object source: shared objects file" in result.stdout


def test_the_default_start_does_not_turn_the_archive_off(tmp_path):
    for flag in _flags(tmp_path):
        assert not flag.startswith(("-Xshare:", "-XX:-UseSharedSpaces"))


def test_an_archive_does_not_serve_a_jvm_with_other_settings(tmp_path):
    flags = [f for f in _flags(tmp_path) if f != "-XX:+UseCompactObjectHeaders"]
    flags.append("-XX:-UseCompactObjectHeaders")
    # the archive was dumped with compact object headers: without them -Xshare:on cannot use it, and a plain start goes on without it
    assert _run(*flags, "-Xshare:on", "-version").returncode != 0
    result = _run(*flags, "-Xshare:auto", "-Xlog:class+load:stdout", "-version")
    assert result.returncode == 0, result.stderr
    assert "java.lang.Object source: shared objects file" not in result.stdout
