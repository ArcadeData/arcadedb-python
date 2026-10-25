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


def _cgroup_memory_limit(proc_cgroup="/proc/self/cgroup", root="/sys/fs/cgroup"):
    """The cgroup v2 memory limit that applies to this process, or None when there is none.

    The process's own cgroup (the 0::<path> line of /proc/self/cgroup) is read first: <root>/<path>/memory.max. A value of
    max means no limit there, so the walk goes up one directory at a time to root and returns the first number it finds.
    Files that are absent or unreadable count as max.
    """
    try:
        lines = Path(proc_cgroup).read_text().splitlines()
    except OSError:
        return None
    path = next(
        (
            parts[2]
            for parts in (line.split(":", 2) for line in lines)
            if len(parts) == 3 and parts[0] == "0" and parts[1] == ""
        ),
        None,
    )
    if path is None:  # cgroup v1 only
        return None
    top = Path(root)
    here = top / path.lstrip("/")
    while True:
        try:
            return int((here / "memory.max").read_text())
        except (OSError, ValueError):  # absent, unreadable, or "max" (no limit here)
            pass
        if here == top or top not in here.parents:
            return None
        here = here.parent


def _memory_limit() -> int:
    """Bytes the isolate sees on Linux: the smaller of the host memory and the cgroup limit of this process."""
    limits = [os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")]
    limit = _cgroup_memory_limit()
    if limit is not None:
        limits.append(limit)
    try:  # cgroup v1
        limits.append(
            int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
        )
    except (OSError, ValueError):
        pass
    return min(limits)


def _fake_cgroup(tmp_path, own, limits):
    """(proc_cgroup, root) of a fake tree: this process is in cgroup own; limits maps a cgroup path to its memory.max text."""
    root = tmp_path / "cgroup"
    for name, text in limits.items():
        d = root / name.strip("/")
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.max").write_text(text + "\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text(f"0::{own}\n")
    return proc, root


def test_the_cgroup_limit_of_the_process_own_cgroup_wins(tmp_path):
    proc, root = _fake_cgroup(
        tmp_path,
        "/a/b/c",
        {"/": "max", "/a": "2097152", "/a/b": "1048576", "/a/b/c": "4096"},
    )
    assert _cgroup_memory_limit(proc, root) == 4096


def test_the_cgroup_limit_is_taken_from_the_first_parent_that_has_one(tmp_path):
    proc, root = _fake_cgroup(
        tmp_path,
        "/a/b/c",
        {"/": "max", "/a": "2097152", "/a/b": "1048576", "/a/b/c": "max"},
    )
    assert _cgroup_memory_limit(proc, root) == 1048576


def test_a_directory_without_memory_max_counts_as_unlimited(tmp_path):
    proc, root = _fake_cgroup(tmp_path, "/a/b/c", {"/a": "8192"})
    (root / "a" / "b" / "c").mkdir(parents=True)  # no memory.max file in c or b
    assert _cgroup_memory_limit(proc, root) == 8192


def test_the_root_cgroup_limit_counts_too(tmp_path):
    proc, root = _fake_cgroup(tmp_path, "/a", {"/": "65536", "/a": "max"})
    assert _cgroup_memory_limit(proc, root) == 65536


def test_no_cgroup_limit_at_all_is_none(tmp_path):
    proc, root = _fake_cgroup(
        tmp_path, "/a/b", {"/": "max", "/a": "max", "/a/b": "max"}
    )
    assert _cgroup_memory_limit(proc, root) is None


def test_the_cgroup_line_is_found_among_cgroup_v1_lines(tmp_path):
    proc, root = _fake_cgroup(tmp_path, "/x", {"/x": "512"})
    proc.write_text("12:memory:/legacy\n1:name=systemd:/legacy\n0::/x\n")
    assert _cgroup_memory_limit(proc, root) == 512


def test_a_process_without_a_v2_cgroup_or_without_proc_has_no_limit(tmp_path):
    proc, root = _fake_cgroup(tmp_path, "/x", {"/x": "512"})
    proc.write_text("12:memory:/legacy\n")
    assert _cgroup_memory_limit(proc, root) is None
    assert _cgroup_memory_limit(tmp_path / "missing", root) is None


def test_default_max_heap_is_half_of_the_memory(tmp_path):
    default = _diag(tmp_path)["max_heap"]
    full = _diag(tmp_path, "-XX:MaximumHeapSizePercent=100")["max_heap"]
    assert 0.45 <= default / full <= 0.55, (default, full)
    # on Linux the heap follows the cgroup limit (a container, a systemd scope), not the host RAM
    if sys.platform == "linux":
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
