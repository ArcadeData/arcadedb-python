#!/usr/bin/env python3
"""Release-time speed smoke of the native wheel: the single-record UPDATE against the previous release.

Two native wheels built from ONE commit can differ in speed by about a third on this statement, because a native-image
build is not reproducible and the speed of a hot path is a draw from a spread (ArcadeData/arcadedb-python#404). The release
step is therefore to measure the wheel that would be published, on a host of the class that measures the published numbers,
and to publish it only when it passes. This script is that measurement. The runbook is docs/modules/ROOT/pages/development/
release-native-smoke.adoc.

What it measures, on the CURRENT host, in one session: for each candidate wheel and the baseline wheel (the previous release
on PyPI), one fresh process per wheel per round runs the process state of the paper cell on synthetic rows (the async
executor, the parallel columnar load of LineItem, the Part load, the indexes, the OLTP and single-record warm-up), then the
timed single-record phases. The verdict figure is `UPDATE Crud SET qty = 2 WHERE ckey=:c` with a parameter through command()
inside begin and commit: the median latency per operation in microseconds. A round runs every wheel once, in an interleaved
order (ABBA over rounds: a round and the next one run the wheels in mirrored order, and the order rotates every two rounds).
The ratio of a candidate is the median over the rounds of the per-round ratio against the baseline (below 1 is faster), with
a 95% percentile bootstrap interval of that median. A control, the baseline wheel against itself in its own processes of the
same session, shows the noise: its interval must lie inside 0.97 to 1.03, else the session is not readable.

Each wheel is installed into its own throw-away environment with `uv run --no-project --with <wheel>` (the way the test
workflows install a wheel under test), from a private copy named by its sha256, so uv's cache cannot serve another file.
The library the process loads is hashed in the process and compared with the library member of the wheel.

Left out of the original cell, on purpose: real TPC-H data (the rows are synthetic, of the same columns and types, a tenth of
the LineItem rows of scale factor 1 and the scale-1 count of Part), the harness around the cell (its container, its memory
sampler, its other phases and queries), and the key streams of the cell (the keys are drawn from a seeded generator of this
script). The heap and the one engine setting of the cell are set through ARCADEDB_NATIVE_ISOLATE_ARGS (--isolate-args).

Usage:
    native_update_smoke.py --baseline PREVIOUS.whl CANDIDATE.whl [CANDIDATE.whl ...] [--out FILE] [--rounds 12]
    native_update_smoke.py --hash-only WHEEL [WHEEL ...]       # the sha256 of the library of each wheel, nothing else

Run it under the cores of one compare lane (the caller pins the process; the children inherit the set). The script reads
`Cpus_allowed_list` from /proc/self/status, records it, and refuses to run on more than 4 cores unless --quick. --quick is a
short run on the small scale for trying the tool: it is never release evidence.

Exit status: 0 at least one candidate passed (the upper end of its interval is below --max-ratio, default 1.10), 1 no
candidate passed, 2 the run was refused or a process failed (usage, cores, a wheel that does not load, a library that is not
the wheel's), 3 the control is not inside 0.97 to 1.03, so the session is not readable.

Tests of the pure parts: tests/test_native_update_smoke.py. Output is ASCII only.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random  # nosec B311 - the bootstrap resamples with a seeded generator, nothing secret
import re
import shutil
import signal
import statistics
import subprocess  # nosec B404 - runs uv with a fixed argument list, no shell
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

ISSUE = "ArcadeData/arcadedb-python#404"
MAX_CORES = 4  # more allowed cores than this is refused unless --quick
DEFAULT_MAX_RATIO = 1.10  # slow builds read 1.33 to 1.35 and fast ones 0.95 to 0.99 in the spread study of #404
CONTROL_LOW, CONTROL_HIGH = (
    0.97,
    1.03,
)  # the control (baseline against itself) must lie inside
DEFAULT_ROUNDS, MIN_ROUNDS = 12, 8
QUICK_ROUNDS, QUICK_MIN_ROUNDS = 4, 2
CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_SEED = 404
DEFAULT_PYTHON = "3.12"
# The heap of the paper cell at scale factor 1 and its relaxed WAL flush (the engine setting the cell runs with).
DEFAULT_ISOLATE_ARGS = "-Xmx8g -Darcadedb.txWalFlush=0"
VERDICT_FIGURE = "update"
FIGURES = ("new_order", "payment", "insert", "read", "update", "delete")
LIB_NAMES = ("libarcadedb.so", "libarcadedb.dylib", "libarcadedb.dll")
EXIT_PASS, EXIT_NONE_PASSED, EXIT_REFUSED, EXIT_CONTROL = 0, 1, 2, 3
RESULT_PREFIX = "SMOKE_RESULT "
SCALES = {
    # the counts of the paper cell: LineItem a tenth of scale factor 1, Part the scale-1 count
    "cell": {
        "li_rows": 300_000,
        "part_rows": 200_000,
        "warm": 2_000,
        "oltp_ops": 1_000,
        "ops": 1_000,
    },
    "small": {
        "li_rows": 20_000,
        "part_rows": 2_000,
        "warm": 100,
        "oltp_ops": 100,
        "ops": 200,
    },
}
LI_COLS = (
    "l_orderkey",
    "l_partkey",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
    "l_returnflag",
    "l_linestatus",
    "l_shipdate",
    "l_shipmode",
    "l_tax",
)
LI_TEXT_COLS = ("l_returnflag", "l_linestatus", "l_shipdate", "l_shipmode")
LI_FRAME_ROWS = 122_880  # rows per columnar frame of the load
PART_BATCH = 10_000


class Refused(Exception):
    """The run cannot give a reading: exit status 2."""


class ChildFailure(Refused):
    """A measurement process failed or did not report what it must."""


# ----- pure parts (tests/test_native_update_smoke.py) ------------------------------------------


def parse_cpu_list(text: str) -> list[int]:
    """Parse a kernel CPU list such as `0-3,8,10-11` into sorted core numbers."""
    text = text.strip()
    if not text:
        raise ValueError("empty CPU list")
    cores: set[int] = set()
    for part in text.split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part.strip())
        if m is None:
            raise ValueError(f"not a CPU list: {text!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo or hi - lo > 65535:
            raise ValueError(f"not a CPU list: {text!r}")
        cores.update(range(lo, hi + 1))
    return sorted(cores)


def read_cpus_allowed(status_text: str) -> tuple[str, list[int]]:
    """The `Cpus_allowed_list` line of a /proc/<pid>/status text: (the text as written, the core numbers)."""
    for line in status_text.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            text = line.split(":", 1)[1].strip()
            return text, parse_cpu_list(text)
    raise ValueError("no Cpus_allowed_list line in the status text")


def check_cores(cores: list[int], quick: bool) -> str | None:
    """None when the run may go on, else the reason it is refused."""
    if not cores:
        return "no allowed core was read"
    if len(cores) > MAX_CORES and not quick:
        return (
            f"{len(cores)} cores are allowed ({','.join(map(str, cores))}), more than {MAX_CORES}: "
            "pin the process to the cores of one lane, or pass --quick for a try that is not evidence"
        )
    return None


def round_order(arms: list[str], k: int) -> list[str]:
    """The order of the arms in round k: rotated by k // 2 and reversed on odd rounds, so rounds 2j and 2j+1 mirror each other."""
    n = len(arms)
    shift = (k // 2) % n
    rotated = arms[shift:] + arms[:shift]
    return rotated[::-1] if k % 2 else rotated


def build_schedule(arms: list[str], rounds: int, warmup_rounds: int) -> list[dict]:
    """The warm-up rounds first (their figures are recorded, never used), then the measured rounds."""
    schedule = [
        {"round": w, "warmup": True, "order": round_order(arms, w)}
        for w in range(warmup_rounds)
    ]
    schedule += [
        {"round": r, "warmup": False, "order": round_order(arms, r)}
        for r in range(rounds)
    ]
    return schedule


def round_ratios(
    base: dict[int, float], arm: dict[int, float]
) -> list[tuple[int, float]]:
    """(round, arm / baseline) for every round both have, in round order."""
    return [(r, arm[r] / base[r]) for r in sorted(arm) if r in base and base[r] > 0]


def bootstrap_median_ci(
    values: list[float], n_boot: int, seed: int, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Percentile bootstrap interval of the median: resample the rounds with replacement."""
    n = len(values)
    if n < 2:
        raise ValueError("a bootstrap needs at least two rounds")
    rng = random.Random(seed)  # nosec B311
    medians = sorted(
        statistics.median([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(n_boot)
    )
    lo_i = int(math.floor((1.0 - confidence) / 2.0 * n_boot))
    hi_i = min(n_boot - 1, int(math.floor((1.0 + confidence) / 2.0 * n_boot)))
    return medians[lo_i], medians[hi_i]


def summarise(values: list[float], n_boot: int, seed: int) -> dict:
    lo, hi = bootstrap_median_ci(values, n_boot, seed)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "lo": lo,
        "hi": hi,
        "ratios": list(values),
    }


def verdict(hi: float, max_ratio: float) -> str:
    """PASS when the upper end of the interval is below the threshold."""
    return "PASS" if hi < max_ratio else "FAIL"


def control_readable(lo: float, hi: float) -> bool:
    return lo >= CONTROL_LOW and hi <= CONTROL_HIGH


def exit_status(control_ok: bool, verdicts: list[str]) -> int:
    """3 when the session is not readable, else 0 when at least one candidate passed, else 1."""
    if not control_ok:
        return EXIT_CONTROL
    return EXIT_PASS if any(v == "PASS" for v in verdicts) else EXIT_NONE_PASSED


def library_member(names: list[str]) -> str:
    """The one member of a wheel that is the native library (`<package>/_native/libarcadedb.*`)."""
    found = [
        n
        for n in names
        if PurePosixPath(n).name in LIB_NAMES
        and PurePosixPath(n).parent.name == "_native"
    ]
    if len(found) != 1:
        raise ValueError(
            f"expected one native library member in the wheel, found {len(found)}: {found}"
        )
    return found[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_wheel_library(wheel: Path) -> dict:
    """The library member of a wheel: its name, size and sha256 (read from the zip, nothing is unpacked)."""
    try:
        zf = zipfile.ZipFile(wheel)
    except (zipfile.BadZipFile, OSError) as err:
        raise ValueError(f"{wheel} is not a readable wheel: {err}") from err
    with zf:
        member = library_member(zf.namelist())
        h = hashlib.sha256()
        size = 0
        with zf.open(member) as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                size += len(chunk)
    return {"member": member, "bytes": size, "sha256": h.hexdigest()}


def duplicate_libraries(libs: dict[str, str]) -> tuple[str, str] | None:
    """The first two labels that share a library sha256 (the same build twice measures nothing), else None."""
    seen: dict[str, str] = {}
    for label, sha in libs.items():
        if sha in seen:
            return seen[sha], label
        seen[sha] = label
    return None


def child_command(
    uv: str,
    wheel: Path,
    script: Path,
    python: str,
    numpy_spec: str,
    offline: bool,
    worker_args: list[str],
) -> list[str]:
    """One measurement process: a throw-away environment of the wheel (and numpy), the interpreter in isolated mode."""
    cmd = [uv, "run", "--no-project", "--python", python]
    if offline:
        cmd.append("--offline")
    cmd += ["--with", str(wheel), "--with", numpy_spec]
    cmd += ["python", "-I", str(script), "--worker", *worker_args]
    return cmd


def child_env(base: dict[str, str], isolate_args: str) -> dict[str, str]:
    """The environment of a measurement process: nothing that can swap the library or the backend, the cell's isolate arguments."""
    env = {
        k: v
        for k, v in base.items()
        if k
        not in (
            "ARCADEDB_NATIVE_LIB",
            "ARCADEDB_NATIVE_LIBDIR",
            "ARCADEDB_BACKEND",
            "ARCADEDB_JVM_ARGS",
            "PYTHONPATH",
        )
    }
    env["ARCADEDB_NATIVE_ISOLATE_ARGS"] = isolate_args
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def parse_child_output(stdout: str) -> dict:
    """The last `SMOKE_RESULT <json>` line of a process's stdout."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise ValueError("the process printed no result line")


def validate_child_result(
    result: dict, expected_lib: str, expected_cpus: str | None
) -> list[str]:
    """What is wrong with a process's report (empty: nothing). The arm must be what the file says it is."""
    problems = []
    if not result.get("ok"):
        problems.append(
            "the process reported a failure: "
            + str(result.get("error") or result.get("problems") or "no reason given")
        )
    if result.get("backend") != "native":
        problems.append(f"backend is {result.get('backend')!r}, wanted 'native'")
    if result.get("lib_sha256") != expected_lib:
        problems.append(
            f"the library it loaded has sha256 {result.get('lib_sha256')}, the wheel's is {expected_lib}"
        )
    if expected_cpus is not None and result.get("cpus_allowed") != expected_cpus:
        problems.append(
            f"it ran on cores {result.get('cpus_allowed')!r}, the session on {expected_cpus!r}"
        )
    figures = result.get("figures") or {}
    for name in FIGURES:
        v = figures.get(name)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            problems.append(f"figure {name} is {v!r}")
    return problems


def figures_by_arm(runs: list[dict]) -> dict[str, dict[str, dict[int, float]]]:
    """{arm: {figure: {round: microseconds}}} over the measured (not warm-up) rounds."""
    out: dict[str, dict[str, dict[int, float]]] = {}
    for run in runs:
        if run["warmup"]:
            continue
        figs = run["result"]["figures"]
        for name in FIGURES:
            out.setdefault(run["arm"], {}).setdefault(name, {})[run["round"]] = float(
                figs[name]
            )
    return out


def analyse(
    runs: list[dict],
    candidates: list[str],
    max_ratio: float,
    n_boot: int,
    seed: int,
    baseline: str = "baseline",
    control: str = "control",
) -> dict:
    """Ratios, intervals and verdicts of every candidate and of the control over the measured rounds."""
    by_arm = figures_by_arm(runs)

    def ratios_of(arm: str) -> dict:
        out = {}
        for name in FIGURES:
            pairs = round_ratios(by_arm[baseline][name], by_arm[arm][name])
            out[name] = summarise([ratio for _r, ratio in pairs], n_boot, seed)
            out[name]["rounds"] = [r for r, _ratio in pairs]
        return out

    ctl = ratios_of(control)
    ctl_summary = ctl[VERDICT_FIGURE]
    result: dict = {
        "figure": VERDICT_FIGURE,
        "max_ratio": max_ratio,
        "control": {
            "figures": ctl,
            "readable": control_readable(ctl_summary["lo"], ctl_summary["hi"]),
            "band": [CONTROL_LOW, CONTROL_HIGH],
        },
        "candidates": {},
    }
    for label in candidates:
        figs = ratios_of(label)
        result["candidates"][label] = {
            "figures": figs,
            "verdict": verdict(figs[VERDICT_FIGURE]["hi"], max_ratio),
        }
    result["exit_status"] = exit_status(
        result["control"]["readable"],
        [c["verdict"] for c in result["candidates"].values()],
    )
    return result


def make_streams(
    seed: int, warm: int, oltp_ops: int, ops: int, part_rows: int
) -> dict[str, list[int]]:
    """The keys of the run, a pure function of the seed: Part keys of the new-orders and of the single-record rows, and the order
    each payment picks (an index into the orders of the warm-up or of the timed phase).
    """
    rng = random.Random(seed)  # nosec B311
    return {
        "part_warm_orders": [rng.randrange(1, part_rows + 1) for _ in range(warm)],
        "pick_warm_payments": [rng.randrange(warm) for _ in range(warm)],
        "part_warm_crud": [rng.randrange(1, part_rows + 1) for _ in range(warm)],
        "part_orders": [rng.randrange(1, part_rows + 1) for _ in range(oltp_ops)],
        "pick_payments": [rng.randrange(oltp_ops) for _ in range(oltp_ops)],
        "part_crud": [rng.randrange(1, part_rows + 1) for _ in range(ops)],
    }


def kept_latencies(latencies: list[float], n: int) -> list[float]:
    """The first min(20, n // 10) operations of a phase are left out, as the cell leaves them out."""
    return latencies[min(20, n // 10) :]


def staged_name(sha256: str, name: str) -> str:
    """The private path (relative to the work folder) of a copy of a wheel, named by the content so uv's cache cannot mix files."""
    return f"wheels/{sha256[:16]}/{name}"


def format_report(report: dict) -> list[str]:
    """The text a person reads, ASCII only."""
    a = report["analysis"]
    cpus = report["cpus_allowed"]
    lines = []
    if report["quick"]:
        lines.append(
            "QUICK RUN on the small scale: a try of the tool, NOT release evidence."
        )
    lines.append(
        f"native update smoke: {report['rounds']} rounds (+{report['warmup_rounds']} warm-up), "
        f"cores {cpus['text']} ({cpus['count']}), bootstrap {report['bootstrap']} resamples, seed {report['seed']}"
    )
    lines.append(
        f"figure: single-record UPDATE through command(), median per operation; ratio against the baseline "
        f"{report['wheels']['baseline']['name']} (below 1 is faster); verdict PASS when the upper end is below {a['max_ratio']:.2f}"
    )
    ctl = a["control"]["figures"][VERDICT_FIGURE]
    lines.append(
        f"control (baseline against itself): {ctl['median']:.3f} [{ctl['lo']:.3f}, {ctl['hi']:.3f}] "
        + (
            "readable"
            if a["control"]["readable"]
            else f"NOT READABLE (must lie inside {CONTROL_LOW:g} to {CONTROL_HIGH:g}): "
            "the session says nothing, run it again on a quiet host"
        )
    )
    lines.append("")
    lines.append(
        f"{'candidate':<10} {'ratio':>7} {'95% interval':>17} {'verdict':>8}  {'engine':<14} library sha256"
    )
    for cand in report["wheels"]["candidates"]:
        label = cand["label"]
        s = a["candidates"][label]["figures"][VERDICT_FIGURE]
        v = a["candidates"][label]["verdict"]
        if not a["control"]["readable"]:
            v += "?"
        interval = "[{:.3f}, {:.3f}]".format(s["lo"], s["hi"])
        lines.append(
            f"{label:<10} {s['median']:>7.3f} {interval:>17} {v:>8}  "
            f"{str(cand.get('engine_version')):<14} {cand['library']['sha256']}"
        )
    lines.append("")
    for cand in report["wheels"]["candidates"]:
        lines.append(f"{cand['label']} = {cand['path']}")
    base = report["wheels"]["baseline"]
    lines.append(
        f"baseline = {base['path']} (engine {base.get('engine_version')}, library sha256 {base['library']['sha256']})"
    )
    lines.append("")
    lines.append("information only (ratio of the median, not part of the verdict):")
    for cand in report["wheels"]["candidates"]:
        parts = []
        for name in FIGURES:
            s = a["candidates"][cand["label"]]["figures"][name]
            parts.append(f"{name} {s['median']:.3f}")
        lines.append(f"  {cand['label']}: " + ", ".join(parts))
    passed = [
        c["label"]
        for c in report["wheels"]["candidates"]
        if a["candidates"][c["label"]]["verdict"] == "PASS"
    ]
    lines.append("")
    if not a["control"]["readable"]:
        lines.append("exit status 3: the control is not readable")
    elif passed:
        lines.append(f"PASS: {', '.join(passed)} (exit status 0)")
    else:
        lines.append("FAIL: no candidate passed (exit status 1)")
    if report["quick"]:
        lines.append(
            "QUICK RUN: do not record this result as the speed smoke of a release."
        )
    return lines


def run_session(
    schedule: list[dict],
    run_child: Callable[[str, dict], dict],
    log: Callable[[str], None] = lambda _line: None,
    runs: list[dict] | None = None,
) -> list[dict]:
    """Walk the schedule, one process per arm per round. A failure raises and stops the session (the rest would be wasted);
    the runs done so far stay in `runs`, which the caller can hand in to keep them."""
    if runs is None:
        runs = []
    total = sum(len(entry["order"]) for entry in schedule)
    for entry in schedule:
        for position, arm in enumerate(entry["order"]):
            started = time.monotonic()
            result = run_child(arm, entry)
            wall = time.monotonic() - started
            runs.append(
                {
                    "arm": arm,
                    "round": entry["round"],
                    "warmup": entry["warmup"],
                    "position": position,
                    "wall_s": round(wall, 3),
                    "result": result,
                }
            )
            kind = "warm-up" if entry["warmup"] else "round"
            log(
                f"[{len(runs)}/{total}] {kind} {entry['round']} {arm}: "
                f"update {result['figures'][VERDICT_FIGURE]:.1f} us ({wall:.1f} s)"
            )
    return runs


# ----- the host ------------------------------------------------------------------------------


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def host_facts() -> dict:
    cpuinfo = _read_text("/proc/cpuinfo") or ""
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        None,
    )
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": model,
        "parent_python": sys.version.split()[0],
    }


def loadavg() -> str | None:
    text = _read_text("/proc/loadavg")
    return text.strip() if text else None


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


# ----- one measurement process (runs inside the throw-away environment of one wheel) ------------


def _worker_synthetic_frames(np, n_rows: int, part_rows: int, seed: int):
    """LineItem columns of the cell's schema and types, as numpy arrays: int64 keys, float64 numbers, object arrays of str."""
    g = np.random.default_rng(seed)
    days = [
        (datetime.date(1992, 1, 1) + datetime.timedelta(days=d)).isoformat()
        for d in range(2526)
    ]
    dates = np.array(days, dtype=object)
    modes = np.array(
        ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"], dtype=object
    )
    flags = np.array(["A", "N", "R"], dtype=object)
    status = np.array(["F", "O"], dtype=object)
    done = 0
    while done < n_rows:
        m = min(LI_FRAME_ROWS, n_rows - done)
        yield {
            "l_orderkey": np.arange(done, done + m, dtype=np.int64) // 4 + 1,
            "l_partkey": g.integers(1, part_rows + 1, m, dtype=np.int64),
            "l_quantity": g.integers(1, 51, m).astype(np.float64),
            "l_extendedprice": np.round(g.uniform(900.0, 105000.0, m), 2),
            "l_discount": g.integers(0, 11, m) / 100.0,
            "l_returnflag": flags[g.integers(0, 3, m)],
            "l_linestatus": status[g.integers(0, 2, m)],
            "l_shipdate": dates[g.integers(0, len(dates), m)],
            "l_shipmode": modes[g.integers(0, 7, m)],
            "l_tax": g.integers(0, 9, m) / 100.0,
        }
        done += m


def _count(db, sql: str) -> int:
    return int(db.query("sql", sql).to_list()[0]["n"])


def _build_cell_state(db, sc: dict, seed: int, np) -> dict:
    """The process state the cell is in when it starts timing: the async executor, LineItem with one bucket per writer and its
    parallel columnar load, the Part load, the schema of the other types, the indexes.
    """
    ex = db.async_executor()
    writers = ex.get_parallel_level()
    ex.set_transaction_sync("no")
    sync = ex.get_transaction_sync()
    db.command("sql", f"CREATE DOCUMENT TYPE LineItem BUCKETS {writers}")
    for col in LI_COLS:
        if col in LI_TEXT_COLS:
            kind = "STRING"
        elif col.endswith("key"):
            kind = "LONG"
        else:
            kind = "DOUBLE"
        db.command("sql", f"CREATE PROPERTY LineItem.{col} {kind}")
    for ddl in (
        "CREATE DOCUMENT TYPE Part",
        "CREATE PROPERTY Part.p_partkey LONG",
        "CREATE INDEX ON Part (p_partkey) UNIQUE_HASH",
        "CREATE DOCUMENT TYPE OrderNew",
        "CREATE PROPERTY OrderNew.okey LONG",
        "CREATE INDEX ON OrderNew (okey) UNIQUE_HASH",
        "CREATE DOCUMENT TYPE Payment",
        "CREATE DOCUMENT TYPE Crud",
        "CREATE PROPERTY Crud.ckey LONG",
        "CREATE INDEX ON Crud (ckey) UNIQUE_HASH",
    ):
        db.command("sql", ddl)
    columnar = hasattr(db, "insert_columns")
    buf: list[dict] = []
    call_rows = writers * ex.get_commit_every()
    for cols in _worker_synthetic_frames(np, sc["li_rows"], sc["part_rows"], seed):
        if columnar:
            db.insert_columns("LineItem", cols, parallel=True)
            continue
        for row in zip(*[cols[c].tolist() for c in LI_COLS]):
            buf.append(dict(zip(LI_COLS, row)))
            if len(buf) >= call_rows:
                db.insert_many("LineItem", buf, parallel=True)
                buf = []
    if buf:
        db.insert_many("LineItem", buf, parallel=True)
    lineitem = _count(db, "SELECT count(*) AS n FROM LineItem")
    if lineitem != sc["li_rows"]:
        raise RuntimeError(
            f"LineItem holds {lineitem} of {sc['li_rows']} rows: the load lost records"
        )
    for start in range(0, sc["part_rows"], PART_BATCH):
        end = min(start + PART_BATCH, sc["part_rows"])
        db.insert_many(
            "Part",
            [
                {
                    "p_partkey": k,
                    "p_retailprice": 900.0 + (k % 1000) / 10.0,
                    "stock": 100,
                }
                for k in range(start + 1, end + 1)
            ],
            commit_every=PART_BATCH,
        )
    db.command("sql", "CREATE INDEX ON LineItem (l_shipdate) NOTUNIQUE")
    indexes = len(db.query("sql", "SELECT FROM schema:indexes").to_list())
    part = _count(db, "SELECT count(*) AS n FROM Part")
    if part != sc["part_rows"]:
        raise RuntimeError(f"Part holds {part} of {sc['part_rows']} rows")
    return {
        "lineitem": lineitem,
        "part": part,
        "writers": writers,
        "transaction_sync": sync,
        "columnar": "insert_columns" if columnar else "insert_many",
        "indexes": indexes,
    }


class _Operations:
    """The six statements of the cell, each in its own begin and commit pair (the read is one query)."""

    def __init__(self, db):
        self.db = db

    def new_order(self, okey: int, pkey: int) -> None:
        db = self.db
        db.begin()
        db.query(
            "sql",
            "SELECT p_retailprice, stock FROM Part WHERE p_partkey=:k",
            {"k": pkey},
        ).to_list()
        db.command(
            "sql",
            "INSERT INTO OrderNew SET okey=:o, pkey=:p, qty=1, paid=0",
            {"o": okey, "p": pkey},
        )
        db.command(
            "sql", "UPDATE Part SET stock = stock - 1 WHERE p_partkey=:k", {"k": pkey}
        )
        db.commit()

    def payment(self, okey: int) -> None:
        db = self.db
        db.begin()
        rows = db.query(
            "sql", "SELECT pkey, qty FROM OrderNew WHERE okey=:o", {"o": okey}
        ).to_list()
        db.command("sql", "UPDATE OrderNew SET paid = 1 WHERE okey=:o", {"o": okey})
        db.command(
            "sql",
            "INSERT INTO Payment SET okey=:o, pkey=:p, amount=1.0",
            {"o": okey, "p": (rows[0].get("pkey") if rows else 0)},
        )
        db.commit()

    def insert(self, ckey: int, pkey: int) -> None:
        db = self.db
        db.begin()
        db.command(
            "sql",
            "INSERT INTO Crud SET ckey=:c, pkey=:p, qty=1, price=9.99",
            {"c": ckey, "p": pkey},
        )
        db.commit()

    def read(self, ckey: int):
        return self.db.query(
            "sql", "SELECT ckey, pkey, qty FROM Crud WHERE ckey=:c", {"c": ckey}
        ).to_list()

    def update(self, ckey: int) -> None:
        db = self.db
        db.begin()
        db.command("sql", "UPDATE Crud SET qty = 2 WHERE ckey=:c", {"c": ckey})
        db.commit()

    def delete(self, ckey: int) -> None:
        db = self.db
        db.begin()
        db.command("sql", "DELETE FROM Crud WHERE ckey=:c", {"c": ckey})
        db.commit()


def _timed_phase(fn: Callable[[int], object], n: int) -> tuple[float, int, list]:
    """Time every operation on its own; the figure is the median of the kept ones, in microseconds."""
    import gc

    gc.collect()
    clock = time.perf_counter
    latencies = []
    returned = []
    for i in range(n):
        t = clock()
        r = fn(i)
        latencies.append((clock() - t) * 1e6)
        returned.append(r)
    kept = kept_latencies(latencies, n)
    return statistics.median(kept), len(kept), returned


def _worker_body(args, result: dict) -> object | None:
    import numpy as np

    import arcadedb as a

    result["backend"] = a.__backend__
    result["version"] = a.__version__
    result["engine_version"] = getattr(a, "ENGINE_VERSION", None)
    try:
        from arcadedb import _build_info

        result["source_commit"] = str(getattr(_build_info, "SOURCE_COMMIT", "unknown"))
    except ImportError:
        result["source_commit"] = "unknown"
    result["python"] = sys.version.split()[0]
    result["numpy"] = np.__version__
    result["isolate_args"] = os.environ.get("ARCADEDB_NATIVE_ISOLATE_ARGS", "")
    result["cpus_allowed"] = read_cpus_allowed(Path("/proc/self/status").read_text())[0]
    result["module_file"] = str(a.__file__)
    problems: list[str] = []
    result["problems"] = problems
    if a.__backend__ != "native":
        problems.append(f"backend {a.__backend__!r}, wanted 'native'")
        return None
    if args.expect_cpus and result["cpus_allowed"] != args.expect_cpus:
        problems.append(
            f"allowed cores {result['cpus_allowed']!r}, the session's {args.expect_cpus!r}"
        )
        return None
    try:
        from arcadedb._native import lib as native_lib

        lib_path = Path(native_lib.library_path())
    except Exception:
        # an older wheel: the library sits next to the module
        lib_path = Path(a.__file__).parent / "_native" / LIB_NAMES[0]
    result["lib_path"] = str(lib_path)
    result["lib_sha256"] = sha256_file(lib_path)
    if args.expect_lib_sha256 and result["lib_sha256"] != args.expect_lib_sha256:
        problems.append(
            f"the library at {lib_path} is not the wheel's ({result['lib_sha256']} against {args.expect_lib_sha256})"
        )
        return None
    sc = SCALES[args.scale]
    result["scale"] = {"name": args.scale, **sc}
    s = make_streams(args.seed, sc["warm"], sc["oltp_ops"], sc["ops"], sc["part_rows"])
    db = a.create_database(os.path.join(args.workdir, "db"))
    ops = _Operations(db)
    t0 = time.perf_counter()
    result["load"] = _build_cell_state(db, sc, args.seed, np)
    result["load_s"] = round(time.perf_counter() - t0, 3)
    warm, oltp, n = sc["warm"], sc["oltp_ops"], sc["ops"]
    # the untimed warm-up of the cell, in its order: new-orders, payments, then the four single-record operations
    for i in range(warm):
        ops.new_order(i, s["part_warm_orders"][i])
    for j in range(warm):
        ops.payment(s["pick_warm_payments"][j])
    for i in range(warm):
        ops.insert(i, s["part_warm_crud"][i])
    for i in range(warm):
        ops.read(i)
    for i in range(warm):
        ops.update(i)
    for i in range(warm):
        ops.delete(i)
    left = _count(db, "SELECT count(*) AS n FROM Crud")
    if left != 0:
        problems.append(f"Crud holds {left} rows after the warm-up, wanted 0")
    figures: dict[str, float] = {}
    kept: dict[str, int] = {}
    checks: dict[str, bool] = {}
    figures["new_order"], kept["new_order"], _ = _timed_phase(
        lambda i: ops.new_order(warm + i, s["part_orders"][i]), oltp
    )
    figures["payment"], kept["payment"], _ = _timed_phase(
        lambda j: ops.payment(warm + s["pick_payments"][j]), oltp
    )
    figures["insert"], kept["insert"], _ = _timed_phase(
        lambda i: ops.insert(warm + i, s["part_crud"][i]), n
    )
    checks["insert_rows"] = _count(db, "SELECT count(*) AS n FROM Crud") == n
    figures["read"], kept["read"], rows = _timed_phase(lambda i: ops.read(warm + i), n)
    checks["read_found_every_row"] = all(len(r) == 1 for r in rows)
    figures["update"], kept["update"], _ = _timed_phase(
        lambda i: ops.update(warm + i), n
    )
    checks["update_changed_every_row"] = (
        _count(db, "SELECT count(*) AS n FROM Crud WHERE qty = 2") == n
    )
    figures["delete"], kept["delete"], _ = _timed_phase(
        lambda i: ops.delete(warm + i), n
    )
    checks["delete_emptied_table"] = _count(db, "SELECT count(*) AS n FROM Crud") == 0
    result["figures"] = figures
    result["kept"] = kept
    result["checks"] = checks
    for name, ok in checks.items():
        if not ok:
            problems.append(f"check {name} failed")
    return db


def worker_main(args) -> int:
    """One measurement process: print one `SMOKE_RESULT` line, then leave without tearing the engine down."""
    sys.dont_write_bytecode = True
    result: dict = {"ok": False}
    db = None
    try:
        db = _worker_body(args, result)
        result["ok"] = not result.get("problems") and "figures" in result
    except BaseException as err:
        # reported in the result line, the parent decides
        result["ok"] = False
        result["error"] = f"{type(err).__name__}: {err}"[:500]
        traceback.print_exc()
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    if db is not None:
        try:
            db.close()
        except BaseException as err:  # the line is already written
            print(f"close failed: {type(err).__name__}: {err}", file=sys.stderr)
    os._exit(0 if result["ok"] else 3)


# ----- the session (the parent) --------------------------------------------------------------


def describe_wheel(path: Path) -> dict:
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "library": hash_wheel_library(path),
    }


def stage_wheel(info: dict, work: Path) -> Path:
    """A private copy of the wheel named by its sha256; the copy is hashed again so the measured bytes are the described ones."""
    dest = work / staged_name(info["sha256"], info["name"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(info["path"], dest)
    if sha256_file(dest) != info["sha256"]:
        raise Refused(
            f"the copy of {info['path']} differs from the file: disk trouble?"
        )
    return dest


def run_child_process(
    cmd: list[str], env: dict[str, str], timeout: int
) -> tuple[int, str, str]:
    """Run one process in its own group so a timeout ends uv and the interpreter below it."""
    proc = subprocess.Popen(  # nosec B603 - fixed argument list, no shell
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        raise ChildFailure(f"timed out after {timeout} s\n{_tail(err)}") from None
    except BaseException:
        # an interrupt of the parent: the group is in a session of its own and would run on
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return proc.returncode, out, err


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join(text.splitlines()[-lines:])


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="native_update_smoke.py",
        description="Speed smoke of native wheels: the single-record UPDATE against the previous release (#404).",
        epilog="Exit status: 0 a candidate passed, 1 none passed, 2 refused or a process failed, 3 the control is not readable.",
    )
    p.add_argument("candidates", nargs="*", metavar="CANDIDATE_WHEEL")
    p.add_argument(
        "--baseline", metavar="WHEEL", help="the previously published native wheel"
    )
    p.add_argument(
        "--out",
        metavar="FILE",
        help="the JSON report (default native-update-smoke-<UTC time>.json)",
    )
    p.add_argument(
        "--rounds",
        type=int,
        help=f"measured rounds, even (default {DEFAULT_ROUNDS}, quick {QUICK_ROUNDS})",
    )
    p.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="rounds run first and never used (default 1)",
    )
    p.add_argument(
        "--max-ratio",
        type=float,
        default=DEFAULT_MAX_RATIO,
        help="PASS when the upper end of the interval is below this (default 1.10)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="small scale, few rounds, any number of cores: a try of the tool, never release evidence",
    )
    p.add_argument(
        "--hash-only",
        action="store_true",
        help="print the library sha256 of each wheel and stop",
    )
    p.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help="interpreter of the environments (default 3.12)",
    )
    p.add_argument(
        "--numpy",
        dest="numpy_spec",
        default="numpy",
        help="requirement of numpy in the environments (default numpy)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="pass --offline to uv (everything must be in its cache)",
    )
    p.add_argument(
        "--work-dir",
        metavar="DIR",
        help="where the private copies and the databases go (a disk, not a tmpfs)",
    )
    p.add_argument("--keep-work", action="store_true", help="keep the work folder")
    p.add_argument(
        "--isolate-args",
        default=DEFAULT_ISOLATE_ARGS,
        help=f"ARCADEDB_NATIVE_ISOLATE_ARGS of every process (default {DEFAULT_ISOLATE_ARGS!r})",
    )
    p.add_argument(
        "--timeout", type=int, default=1800, help="seconds per process (default 1800)"
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help="bootstrap resamples (default 10000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed of the bootstrap and of the keys (default 404)",
    )
    # one measurement process, started by the parent
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--workdir", help=argparse.SUPPRESS)
    p.add_argument(
        "--scale", choices=sorted(SCALES), default="cell", help=argparse.SUPPRESS
    )
    p.add_argument("--expect-lib-sha256", help=argparse.SUPPRESS)
    p.add_argument("--expect-cpus", help=argparse.SUPPRESS)
    return p.parse_intermixed_args(argv)


def hash_only(paths: list[str]) -> int:
    if not paths:
        print("REFUSED: --hash-only needs at least one wheel", file=sys.stderr)
        return EXIT_REFUSED
    for name in paths:
        try:
            lib = hash_wheel_library(Path(name))
        except ValueError as err:
            print(f"REFUSED: {err}", file=sys.stderr)
            return EXIT_REFUSED
        print(f"{lib['sha256']}  {lib['member']}  {lib['bytes']} bytes  {name}")
    return EXIT_PASS


def check_args(args: argparse.Namespace) -> None:
    """Refuse what cannot give a reading, before anything is copied or started."""
    if not args.baseline:
        raise Refused("--baseline WHEEL is required")
    if not args.candidates:
        raise Refused("name at least one candidate wheel")
    if args.rounds is None:
        args.rounds = QUICK_ROUNDS if args.quick else DEFAULT_ROUNDS
    least = QUICK_MIN_ROUNDS if args.quick else MIN_ROUNDS
    if args.rounds < least or args.rounds % 2:
        raise Refused(f"--rounds must be even and at least {least}, got {args.rounds}")
    if args.warmup_rounds < 0:
        raise Refused("--warmup-rounds cannot be negative")
    if args.max_ratio <= 0 or args.bootstrap < 100 or args.timeout <= 0:
        raise Refused(
            "--max-ratio, --bootstrap (at least 100) and --timeout must be positive"
        )
    for name in [args.baseline, *args.candidates]:
        if not Path(name).is_file():
            raise Refused(f"no such wheel file: {name}")
    if shutil.which("uv") is None:
        raise Refused("uv is not on PATH")


def main_run(args: argparse.Namespace) -> int:
    check_args(args)
    status_text = _read_text("/proc/self/status")
    if status_text is None:
        raise Refused("cannot read /proc/self/status: this tool measures on Linux")
    try:
        cpus_text, cpus = read_cpus_allowed(status_text)
    except ValueError as err:
        raise Refused(f"cannot read the allowed cores: {err}") from err
    reason = check_cores(cpus, args.quick)
    if reason:
        raise Refused(reason)
    uv = shutil.which("uv")
    assert uv is not None  # nosec B101 - check_args found it
    scale = "small" if args.quick else "cell"
    out_path = Path(
        args.out or f"native-update-smoke-{now_utc().replace(':', '')}.json"
    )
    candidates = [f"cand{i + 1}" for i in range(len(args.candidates))]
    arms = ["baseline", "control", *candidates]

    base_dir = Path(args.work_dir) if args.work_dir else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="native-update-smoke-", dir=base_dir))
    report: dict = {
        "schema": 1,
        "tool": "scripts/native_update_smoke.py",
        "issue": ISSUE,
        "started_utc": now_utc(),
        "status": "aborted",
        "error": None,
        "quick": args.quick,
        "figure": VERDICT_FIGURE,
        "max_ratio": args.max_ratio,
        "control_band": [CONTROL_LOW, CONTROL_HIGH],
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "bootstrap": args.bootstrap,
        "confidence": CONFIDENCE,
        "seed": args.seed,
        "scale": {"name": scale, **SCALES[scale]},
        "isolate_args": args.isolate_args,
        "python": args.python,
        "numpy_spec": args.numpy_spec,
        "host": {**host_facts(), "loadavg_start": loadavg()},
        "cpus_allowed": {
            "text": cpus_text,
            "list": cpus,
            "count": len(cpus),
            "limit": MAX_CORES,
        },
        "work_dir": str(work),
    }
    try:
        uv_version = run_child_process([uv, "--version"], dict(os.environ), 60)
        report["host"]["uv"] = uv_version[1].strip()
        base_info = describe_wheel(Path(args.baseline))
        cand_infos = []
        for label, name in zip(candidates, args.candidates):
            info = describe_wheel(Path(name))
            info["label"] = label
            cand_infos.append(info)
        dup = duplicate_libraries(
            {"baseline": base_info["library"]["sha256"]}
            | {c["label"]: c["library"]["sha256"] for c in cand_infos}
        )
        if dup:
            raise Refused(
                f"{dup[0]} and {dup[1]} hold the same library (same sha256): the same build twice measures nothing"
            )
        report["wheels"] = {"baseline": base_info, "candidates": cand_infos}
        staged = {"baseline": stage_wheel(base_info, work)}
        staged["control"] = staged["baseline"]
        for info in cand_infos:
            staged[info["label"]] = stage_wheel(info, work)
        lib_of = {"baseline": base_info["library"]["sha256"]}
        lib_of["control"] = lib_of["baseline"]
        lib_of |= {c["label"]: c["library"]["sha256"] for c in cand_infos}
        schedule = build_schedule(arms, args.rounds, args.warmup_rounds)
        report["schedule"] = schedule
        script = Path(__file__).resolve()
        env = child_env(dict(os.environ), args.isolate_args)
        engine_of: dict[str, str | None] = {}

        def run_child(arm: str, entry: dict) -> dict:
            tag = ("w" if entry["warmup"] else "r") + str(entry["round"])
            run_dir = work / "runs" / f"{arm}-{tag}"
            run_dir.mkdir(parents=True)
            try:
                worker_args = [
                    "--workdir",
                    str(run_dir),
                    "--scale",
                    scale,
                    "--seed",
                    str(args.seed),
                    "--expect-lib-sha256",
                    lib_of[arm],
                    "--expect-cpus",
                    cpus_text,
                ]
                cmd = child_command(
                    uv,
                    staged[arm],
                    script,
                    args.python,
                    args.numpy_spec,
                    args.offline,
                    worker_args,
                )
                code, out, err = run_child_process(cmd, env, args.timeout)
                try:
                    result = parse_child_output(out)
                except ValueError as perr:
                    raise ChildFailure(
                        f"{arm} {tag}: exit {code}, {perr}\n{_tail(err)}"
                    ) from perr
                problems = validate_child_result(result, lib_of[arm], cpus_text)
                if code != 0 or problems:
                    raise ChildFailure(
                        f"{arm} {tag}: exit {code}; "
                        + "; ".join(problems)
                        + f"\n{_tail(err)}"
                    )
                first = engine_of.setdefault(arm, result.get("engine_version"))
                if result.get("engine_version") != first:
                    raise ChildFailure(
                        f"{arm} {tag}: engine version {result.get('engine_version')!r}, earlier {first!r}"
                    )
                return result
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)

        runs = report["runs"] = []
        run_session(schedule, run_child, lambda line: print(line, flush=True), runs)
        first_results = {}
        for run in runs:
            first_results.setdefault(run["arm"], run["result"])
        base_info["engine_version"] = first_results["baseline"].get("engine_version")
        base_info["version"] = first_results["baseline"].get("version")
        base_info["source_commit"] = first_results["baseline"].get("source_commit")
        for info in cand_infos:
            res = first_results[info["label"]]
            info["engine_version"] = res.get("engine_version")
            info["version"] = res.get("version")
            info["source_commit"] = res.get("source_commit")
        report["analysis"] = analyse(
            runs, candidates, args.max_ratio, args.bootstrap, args.seed
        )
        report["host"]["loadavg_end"] = loadavg()
        report["finished_utc"] = now_utc()
        report["status"] = "complete"
        report["exit_status"] = report["analysis"]["exit_status"]
        print()
        print("\n".join(format_report(report)))
        return report["exit_status"]
    except Refused as err:
        report["error"] = str(err)
        report["exit_status"] = EXIT_REFUSED
        raise
    finally:
        try:
            report.setdefault("finished_utc", now_utc())
            write_json(out_path, report)
            print(f"report: {out_path}")
        except OSError as werr:
            print(f"could not write {out_path}: {werr}", file=sys.stderr)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)
    if args.hash_only:
        return hash_only(([args.baseline] if args.baseline else []) + args.candidates)
    try:
        return main_run(args)
    except Refused as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
