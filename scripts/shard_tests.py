#!/usr/bin/env python3
"""Split the test files into shards, so a pull request runs one suite on several runners side by side.

A pull request tests one platform and one Python, and each suite there is cut into N shards (scripts/select_matrix.py and
scripts/select_native_matrix.py say how many). A shard is "I/N" with I from 1 to N. Every runner computes the same split from the same
list of files, so each file runs in exactly one shard. A push to main, the nightly run, a manual run of all legs and a release are not
sharded.

Two callers:

- `pytest --shard I/N` (tests/conftest.py), the JVM build: one process for the shard, the other files are not collected;
- `native/wheel/parity_wheel.py --shard I/N`, the native build: one process for each file of the shard.

The split is greedy: the files are taken from the heaviest to the lightest and each goes to the shard with the smallest total so far.
A weight is the seconds a file takes in CI. Only the files that matter for the balance are listed; a stale or missing number makes one
shard slower and never loses a file.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Seconds in the run of 2026-11-27 (linux-x86_64 / linux/amd64). Every file that is not listed counts DEFAULT_WEIGHT.
WEIGHTS: dict[str, dict[str, float]] = {
    "native": {
        "test_vector_product_quantization.py": 93,
        "test_log_redirect.py": 69,
        "test_native_vindex_bin.py": 15,
        "test_column_frames.py": 15,
        "test_vector_read_your_writes.py": 12,
        "test_ingest_integrity.py": 11,
        "test_native_column_frames_reference.py": 10,
        "test_vector.py": 8,
    },
    "jvm": {
        "test_ingest_integrity.py": 29,
        "test_column_frames.py": 20,
        "test_resultset.py": 13,
        "test_sigint.py": 12,
        "test_typed_batch_guard.py": 12,
        "test_vertex_vector_loads.py": 12,
        "test_nonfinite_floats.py": 11,
        "test_log_redirect.py": 11,
        "test_graph_batch_single_pass.py": 10,
        "test_dataframe_ingest.py": 10,
        "test_bulk_insert.py": 9,
        "test_columnar_readers.py": 8,
        "test_typed_rows.py": 8,
        "test_insert_many_typed_columns.py": 7,
        "test_graph_batch_typed_jvm.py": 7,
        "test_python_function_interrupt.py": 7,
        "test_jar_closure.py": 7,
    },
}
DEFAULT_WEIGHT = 1.5


def parse_shard(text: str) -> tuple[int, int]:
    """(I, N) of "I/N" with 1 <= I <= N; anything else raises ValueError."""
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", text)
    if not match or int(match.group(1)) > int(match.group(2)):
        raise ValueError(f"a shard is I/N with 1 <= I <= N, not {text!r}")
    return int(match.group(1)), int(match.group(2))


def split(files: Iterable[str], count: int, backend: str) -> list[list[str]]:
    """The `count` shards of `files` (file names), each sorted; `backend` ("native" or "jvm") chooses the weights."""
    weights = WEIGHTS["native" if backend == "native" else "jvm"]
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for name in sorted(set(files), key=lambda n: (-weights.get(n, DEFAULT_WEIGHT), n)):
        lightest = loads.index(min(loads))
        shards[lightest].append(name)
        loads[lightest] += weights.get(name, DEFAULT_WEIGHT)
    return [sorted(shard) for shard in shards]


def select(files: Iterable[str], shard: str, backend: str) -> list[str]:
    """The files of the shard "I/N"."""
    index, count = parse_shard(shard)
    return split(files, count, backend)[index - 1]
