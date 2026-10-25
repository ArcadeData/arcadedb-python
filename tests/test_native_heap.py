"""Native build: the isolate gets a heap limit of 50% of the memory it can see, unless the user sets the heap.

GraalVM's own default is 80% of the memory (the container limit when there is one). Together with the engine's off-heap memory that
let the kernel kill the process in a memory-limited container, so the glue passes ``-XX:MaximumHeapSizePercent=50`` as an isolate
argument. An ``-Xmx``, ``-XX:MaximumHeapSizePercent`` or ``-XX:MaxRAM`` in ARCADEDB_NATIVE_ISOLATE_ARGS replaces it.
"""

import json
import os
import subprocess  # nosec B404 - test-controlled child process
import sys
from pathlib import Path

import pytest

from arcadedb._native import lib as native_lib

DEFAULT = "-XX:MaximumHeapSizePercent=50"

CHILD = """
import json
from arcadedb._native import lib
print(json.dumps(lib.diag()))
"""


def test_default_isolate_arguments_cap_the_heap_at_half():
    assert native_lib._heap_isolate_args([]) == [DEFAULT]


@pytest.mark.parametrize(
    "user",
    [
        ["-Xmx2g"],
        ["-XX:MaximumHeapSizePercent=25"],
        ["-XX:MaxRAM=4g"],
        ["-Dfoo=bar", "-Xmx512m", "-XX:InitialCollectionPolicy=Adaptive"],
    ],
)
def test_user_heap_arguments_win_over_the_default(user):
    assert native_lib._heap_isolate_args(user) == []


def test_other_user_arguments_keep_the_default():
    user = ["-XX:InitialCollectionPolicy=Adaptive", "-Dfoo=bar"]
    assert native_lib._heap_isolate_args(user) == [DEFAULT]


def _diag(tmp_path, isolate_args=None) -> dict:
    """``lib.diag()`` of an isolate that a fresh process starts with ``isolate_args`` in ARCADEDB_NATIVE_ISOLATE_ARGS."""
    env = {k: v for k, v in os.environ.items() if k != "ARCADEDB_NATIVE_ISOLATE_ARGS"}
    env["ARCADEDB_LOG_DIR"] = str(tmp_path / "log")
    if isolate_args is not None:
        env["ARCADEDB_NATIVE_ISOLATE_ARGS"] = isolate_args
    result = subprocess.run(  # nosec B603 - fixed argument list, no shell
        [sys.executable, "-c", CHILD],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _memory_limit() -> int:
    """Bytes the isolate sees on Linux: the smaller of the host memory and a cgroup limit."""
    limits = [os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")]
    for name in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            limits.append(int(Path(name).read_text()))
        except (OSError, ValueError):  # absent, or "max" (no limit)
            pass
    return min(limits)


def test_default_max_heap_is_half_of_the_memory(tmp_path):
    default = _diag(tmp_path)["max_heap"]
    full = _diag(tmp_path, "-XX:MaximumHeapSizePercent=100")["max_heap"]
    assert 0.45 <= default / full <= 0.55, (default, full)
    if sys.platform == "linux":  # follows the container limit, not the host RAM
        assert 0.40 <= default / _memory_limit() <= 0.60, default


def test_user_percent_replaces_the_default_in_the_isolate(tmp_path):
    default = _diag(tmp_path)["max_heap"]
    quarter = _diag(tmp_path, "-XX:MaximumHeapSizePercent=25")["max_heap"]
    assert 0.45 <= quarter / default <= 0.55, (quarter, default)


def test_user_xmx_replaces_the_default_in_the_isolate(tmp_path):
    max_heap = _diag(tmp_path, "-Xmx512m")["max_heap"]
    assert 0.8 * 512 * 2**20 <= max_heap <= 512 * 2**20, max_heap


def test_user_max_ram_replaces_the_default_in_the_isolate(tmp_path):
    # with MaxRAM the percentage is taken of that size; the default 50% is not added, so GraalVM's own percentage (80%) applies
    max_heap = _diag(tmp_path, "-XX:MaxRAM=4g")["max_heap"]
    assert 0.7 * 4 * 2**30 <= max_heap <= 0.85 * 4 * 2**30, max_heap
