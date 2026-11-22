"""The pure parts of scripts/native_update_smoke.py, the release-time speed smoke of the native wheel (#404).

No JVM, no native library and no real wheel are needed: the ratio and bootstrap arithmetic on synthetic numbers, the ABBA
schedule, the allowed cores read from a text, the verdict and the exit status, the library member hash of a tiny fake wheel,
the command and the environment of a measurement process, the check of what a process reports, and the session loop with a
fake process. Nothing here times anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "native_update_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("native_update_smoke", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["native_update_smoke"] = module
    spec.loader.exec_module(module)
    return module


smoke = _load()

STATUS_TEXT = (
    "Name:\tpython\n"
    "Cpus_allowed:\tc0\n"
    "Cpus_allowed_list:\t6-7\n"
    "Mems_allowed:\t1\n"
    "Mems_allowed_list:\t0\n"
)


# ----- the allowed cores -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, want",
    [
        ("0-3,8,10-11", [0, 1, 2, 3, 8, 10, 11]),
        ("6", [6]),
        (" 4-5 \n", [4, 5]),
        ("1,1,0-1", [0, 1]),
        ("0-15", list(range(16))),
    ],
)
def test_parse_cpu_list(text, want):
    assert smoke.parse_cpu_list(text) == want


@pytest.mark.parametrize("text", ["", "  ", "a", "3-1", "1,,2", "0-", "-3", "1;2"])
def test_parse_cpu_list_refuses(text):
    with pytest.raises(ValueError):
        smoke.parse_cpu_list(text)


def test_read_cpus_allowed_takes_the_list_line_not_the_mask():
    assert smoke.read_cpus_allowed(STATUS_TEXT) == ("6-7", [6, 7])


def test_read_cpus_allowed_without_the_line():
    with pytest.raises(ValueError):
        smoke.read_cpus_allowed("Name:\tpython\nCpus_allowed:\tc0\n")


def test_check_cores_limit_and_quick():
    four = [4, 5, 6, 7]
    five = [3, 4, 5, 6, 7]
    assert smoke.check_cores(four, quick=False) is None
    assert "5 cores" in smoke.check_cores(five, quick=False)
    assert smoke.check_cores(five, quick=True) is None
    assert smoke.check_cores([], quick=True) is not None


def test_main_refuses_more_than_four_allowed_cores(tmp_path, monkeypatch, capsys):
    base, cand = tmp_path / "base.whl", tmp_path / "cand.whl"
    base.write_bytes(b"x")
    cand.write_bytes(b"y")
    eight = "Cpus_allowed_list:\t0-7\n"
    monkeypatch.setattr(smoke, "_read_text", lambda _path: eight)
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "/usr/bin/uv")
    code = smoke.main(["--baseline", str(base), str(cand)])
    assert code == smoke.EXIT_REFUSED == 2
    assert "8 cores" in capsys.readouterr().err


# ----- the schedule ----------------------------------------------------------------------------


def test_round_order_two_arms_is_abba():
    arms = ["A", "B"]
    flat = [a for k in range(4) for a in smoke.round_order(arms, k)]
    assert flat == ["A", "B", "B", "A", "B", "A", "A", "B"]


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_round_order_mirrors_pairs_and_balances_positions(n):
    arms = [f"arm{i}" for i in range(n)]
    for k in range(0, 4 * n, 2):
        assert smoke.round_order(arms, k + 1) == smoke.round_order(arms, k)[::-1]
    positions = {a: [] for a in arms}
    for k in range(2 * n):
        order = smoke.round_order(arms, k)
        assert sorted(order) == sorted(arms)
        for pos, a in enumerate(order):
            positions[a].append(pos)
    for a in arms:
        # over one full period every arm stood at every position
        assert sorted(positions[a]) == sorted(list(range(n)) * 2)


def test_build_schedule_warmup_first_and_counts():
    arms = ["baseline", "control", "cand1"]
    sched = smoke.build_schedule(arms, rounds=6, warmup_rounds=2)
    assert [e["warmup"] for e in sched] == [True, True] + [False] * 6
    assert [e["round"] for e in sched if not e["warmup"]] == list(range(6))
    assert all(sorted(e["order"]) == sorted(arms) for e in sched)
    assert smoke.build_schedule(arms, rounds=4, warmup_rounds=0)[0]["warmup"] is False


# ----- ratios, bootstrap, verdict --------------------------------------------------------------


def test_round_ratios_pairs_rounds_present_in_both():
    base = {0: 100.0, 1: 200.0, 2: 0.0, 3: 50.0}
    arm = {0: 110.0, 1: 100.0, 2: 10.0, 4: 7.0}
    assert smoke.round_ratios(base, arm) == [(0, 1.1), (1, 0.5)]


def test_bootstrap_of_a_constant_is_the_constant():
    assert smoke.bootstrap_median_ci([1.25] * 10, 500, seed=1) == (1.25, 1.25)


def test_bootstrap_is_seeded_and_inside_the_data():
    values = [0.98, 1.0, 1.02, 1.01, 0.99, 1.03, 0.97, 1.0]
    first = smoke.bootstrap_median_ci(values, 2000, seed=404)
    assert first == smoke.bootstrap_median_ci(values, 2000, seed=404)
    lo, hi = first
    assert min(values) <= lo <= hi <= max(values)
    assert lo <= 1.0 <= hi


def test_bootstrap_needs_two_rounds():
    with pytest.raises(ValueError):
        smoke.bootstrap_median_ci([1.0], 100, seed=1)


def test_summarise_median_and_interval_of_a_slow_build():
    ratios = [1.31, 1.34, 1.33, 1.32, 1.35, 1.33, 1.34, 1.32, 1.33, 1.36]
    s = smoke.summarise(ratios, 2000, seed=404)
    assert s["n"] == 10 and s["median"] == pytest.approx(1.33)
    assert 1.31 <= s["lo"] <= s["median"] <= s["hi"] <= 1.36
    assert smoke.verdict(s["hi"], 1.10) == "FAIL"


@pytest.mark.parametrize(
    "hi, want",
    [(0.95, "PASS"), (1.099, "PASS"), (1.10, "FAIL"), (1.101, "FAIL"), (1.35, "FAIL")],
)
def test_verdict_is_strictly_below_the_threshold(hi, want):
    assert smoke.verdict(hi, 1.10) == want


@pytest.mark.parametrize(
    "lo, hi, want",
    [
        (0.99, 1.01, True),
        (0.97, 1.03, True),
        (0.969, 1.01, False),
        (0.99, 1.031, False),
        (0.90, 1.10, False),
    ],
)
def test_control_band(lo, hi, want):
    assert smoke.control_readable(lo, hi) is want


@pytest.mark.parametrize(
    "control_ok, verdicts, want",
    [
        (True, ["PASS"], 0),
        (True, ["FAIL", "PASS", "FAIL"], 0),
        (True, ["FAIL", "FAIL"], 1),
        (True, [], 1),
        (False, ["PASS"], 3),
        (False, ["FAIL"], 3),
    ],
)
def test_exit_status(control_ok, verdicts, want):
    assert smoke.exit_status(control_ok, verdicts) == want


# ----- the analysis of a whole session, on synthetic runs --------------------------------------


def _runs(factor, rounds=10, warmup=1):
    """Synthetic runs of the arms: factor(arm, round) times a level that drifts from round to round, shared by every arm."""
    runs = []
    arms = ["baseline", "control", "fast", "slow"]
    for w in range(warmup):
        for arm in arms:  # absurd warm-up figures: they must never enter a ratio
            figs = {n: 9999.0 for n in smoke.FIGURES}
            runs.append(_run(arm, w, True, figs))
    for r in range(rounds):
        level = 100.0 + (r % 4) * 2.0
        for arm in arms:
            value = level * factor(arm, r)
            runs.append(_run(arm, r, False, {n: value for n in smoke.FIGURES}))
    return runs


def _run(arm, rnd, warmup, figures):
    return {
        "arm": arm,
        "round": rnd,
        "warmup": warmup,
        "position": 0,
        "wall_s": 1.0,
        "result": {"figures": figures},
    }


def _quiet(arm, r):
    wiggle = 1.0 + 0.004 * ((r + len(arm)) % 3 - 1)
    return {"baseline": 1.0, "control": 1.0, "fast": 0.97, "slow": 1.33}[arm] * wiggle


def test_analyse_passes_the_fast_build_and_fails_the_slow_one():
    a = smoke.analyse(_runs(_quiet), ["fast", "slow"], 1.10, 2000, 404)
    assert a["control"]["readable"] is True
    assert a["candidates"]["fast"]["verdict"] == "PASS"
    assert a["candidates"]["slow"]["verdict"] == "FAIL"
    fast = a["candidates"]["fast"]["figures"]["update"]
    assert fast["n"] == 10 and fast["median"] == pytest.approx(0.97, abs=0.01)
    assert a["candidates"]["slow"]["figures"]["update"]["lo"] > 1.30
    assert a["exit_status"] == 0


def test_analyse_exit_one_when_no_candidate_passes():
    a = smoke.analyse(_runs(_quiet), ["slow"], 1.10, 2000, 404)
    assert a["exit_status"] == 1


def test_analyse_noisy_control_is_not_readable_whatever_the_verdicts():
    def noisy(arm, r):
        swing = 1.0 + (0.08 if r % 2 else -0.08) if arm == "control" else 1.0
        return _quiet(arm, r) * swing

    a = smoke.analyse(_runs(noisy), ["fast", "slow"], 1.10, 2000, 404)
    assert a["control"]["readable"] is False
    assert a["candidates"]["fast"]["verdict"] == "PASS"
    assert a["exit_status"] == 3


def test_analyse_ignores_the_warmup_rounds():
    a = smoke.analyse(_runs(_quiet, warmup=3), ["fast"], 1.10, 500, 404)
    assert a["candidates"]["fast"]["figures"]["update"]["n"] == 10
    assert a["exit_status"] == 0


def test_format_report_is_ascii_and_names_the_verdicts():
    runs = _runs(_quiet)
    analysis = smoke.analyse(runs, ["fast", "slow"], 1.10, 500, 404)
    report = {
        "quick": False,
        "rounds": 10,
        "warmup_rounds": 1,
        "bootstrap": 500,
        "seed": 404,
        "cpus_allowed": {"text": "6-7", "count": 2},
        "analysis": analysis,
        "wheels": {
            "baseline": {
                "name": "arcadedb-26.11.1-py3-none-manylinux_2_28_x86_64.whl",
                "path": "/w/base.whl",
                "engine_version": "26.11.1",
                "library": {"sha256": "b" * 64},
            },
            "candidates": [
                {
                    "label": "fast",
                    "path": "/w/fast.whl",
                    "engine_version": "26.12.1",
                    "library": {"sha256": "f" * 64},
                },
                {
                    "label": "slow",
                    "path": "/w/slow.whl",
                    "engine_version": "26.12.1",
                    "library": {"sha256": "5" * 64},
                },
            ],
        },
    }
    text = "\n".join(smoke.format_report(report))
    assert text.isascii()
    assert "f" * 64 in text and "5" * 64 in text and "b" * 64 in text
    assert "PASS: fast (exit status 0)" in text
    assert "QUICK" not in text
    report["quick"] = True
    assert "NOT release evidence" in "\n".join(smoke.format_report(report))


# ----- the session loop, with a fake process ---------------------------------------------------


def test_run_session_follows_the_schedule_and_keeps_partial_runs_on_failure():
    arms = ["baseline", "control", "cand1"]
    sched = smoke.build_schedule(arms, rounds=4, warmup_rounds=1)
    calls = []

    def child(arm, entry):
        calls.append((arm, entry["round"], entry["warmup"]))
        return {"figures": {"update": 100.0}}

    runs = smoke.run_session(sched, child)
    want = [(a, e["round"], e["warmup"]) for e in sched for a in e["order"]]
    assert calls == want and len(runs) == 15
    assert [r["position"] for r in runs[:3]] == [0, 1, 2]

    kept, seen = [], []

    def failing(arm, entry):
        seen.append(arm)
        if len(seen) == 4:
            raise smoke.ChildFailure("no good")
        return {"figures": {"update": 1.0}}

    with pytest.raises(smoke.ChildFailure):
        smoke.run_session(sched, failing, runs=kept)
    assert len(kept) == 3
    assert isinstance(smoke.ChildFailure("x"), smoke.Refused)


def test_make_streams_is_a_pure_function_of_the_seed():
    a = smoke.make_streams(404, warm=50, oltp_ops=40, ops=30, part_rows=1000)
    assert a == smoke.make_streams(404, 50, 40, 30, 1000)
    assert a != smoke.make_streams(405, 50, 40, 30, 1000)
    assert [len(a[k]) for k in a] == [50, 50, 50, 40, 40, 30]
    assert all(
        1 <= p <= 1000
        for k in ("part_warm_orders", "part_warm_crud", "part_orders", "part_crud")
        for p in a[k]
    )
    assert all(0 <= p < 50 for p in a["pick_warm_payments"])
    assert all(0 <= p < 40 for p in a["pick_payments"])


@pytest.mark.parametrize("n, dropped", [(1000, 20), (200, 20), (100, 10), (5, 0)])
def test_kept_latencies_leave_out_the_first_ones(n, dropped):
    latencies = list(range(n))
    assert smoke.kept_latencies(latencies, n) == latencies[dropped:]


# ----- the library member of a wheel -----------------------------------------------------------


def _fake_wheel(path, lib_bytes=b"library-bytes", extra=()):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("arcadedb/__init__.py", "x = 1\n")
        zf.writestr("arcadedb/_native/libarcadedb.so", lib_bytes)
        zf.writestr("arcadedb/_native/libjvector-x86_64.so", b"other library")
        zf.writestr("arcadedb-26.12.1.dist-info/METADATA", "Name: arcadedb\n")
        for name, data in extra:
            zf.writestr(name, data)
    return path


def test_hash_wheel_library_hashes_the_library_member_only(tmp_path):
    lib = b"\x7fELF" + b"a" * 3_000_000  # more than one read chunk
    wheel = _fake_wheel(tmp_path / "arcadedb-26.12.1-py3-none-any.whl", lib)
    info = smoke.hash_wheel_library(wheel)
    assert info == {
        "member": "arcadedb/_native/libarcadedb.so",
        "bytes": len(lib),
        "sha256": hashlib.sha256(lib).hexdigest(),
    }
    assert (
        smoke.describe_wheel(wheel)["sha256"]
        == hashlib.sha256(wheel.read_bytes()).hexdigest()
    )


def test_two_wheels_with_different_libraries_differ(tmp_path):
    a = smoke.hash_wheel_library(_fake_wheel(tmp_path / "a.whl", b"one"))
    b = smoke.hash_wheel_library(_fake_wheel(tmp_path / "b.whl", b"two"))
    assert a["sha256"] != b["sha256"]
    assert smoke.duplicate_libraries({"x": a["sha256"], "y": b["sha256"]}) is None
    assert smoke.duplicate_libraries(
        {"x": a["sha256"], "y": b["sha256"], "z": a["sha256"]}
    ) == ("x", "z")


@pytest.mark.parametrize(
    "names, ok",
    [
        (["arcadedb/_native/libarcadedb.so"], True),
        (["arcadedb/_native/libarcadedb.dylib", "arcadedb/__init__.py"], True),
        (["arcadedb/_native/libjvector-x86_64.so"], False),
        (["libarcadedb.so"], False),
        (["arcadedb/_native/libarcadedb.so", "other/_native/libarcadedb.so"], False),
        ([], False),
    ],
)
def test_library_member(names, ok):
    if ok:
        assert smoke.library_member(names) in names
    else:
        with pytest.raises(ValueError):
            smoke.library_member(names)


def test_hash_wheel_library_refuses_a_file_that_is_not_a_wheel(tmp_path):
    bad = tmp_path / "not-a-wheel.whl"
    bad.write_bytes(b"plain text")
    with pytest.raises(ValueError):
        smoke.hash_wheel_library(bad)
    with pytest.raises(ValueError):
        smoke.hash_wheel_library(tmp_path / "missing.whl")


def test_hash_only_prints_the_hash_and_the_exit_status(tmp_path, capsys):
    wheel = _fake_wheel(tmp_path / "w.whl", b"lib")
    assert smoke.main(["--hash-only", str(wheel)]) == 0
    out = capsys.readouterr().out
    assert hashlib.sha256(b"lib").hexdigest() in out and "libarcadedb.so" in out
    assert smoke.main(["--hash-only"]) == 2
    assert smoke.main(["--hash-only", str(tmp_path / "missing.whl")]) == 2


def test_staged_name_is_named_by_the_content():
    sha = "ab" * 32
    assert smoke.staged_name(sha, "w.whl") == f"wheels/{sha[:16]}/w.whl"


# ----- the measurement process: command, environment, report -----------------------------------


def test_child_command_uses_a_throwaway_environment_of_the_wheel():
    cmd = smoke.child_command(
        "uv",
        Path("/w/arcadedb.whl"),
        Path("/s/native_update_smoke.py"),
        "3.12",
        "numpy",
        False,
        ["--workdir", "/d"],
    )
    assert cmd[:5] == ["uv", "run", "--no-project", "--python", "3.12"]
    assert "--with" in cmd and cmd[cmd.index("--with") + 1] == str(
        Path("/w/arcadedb.whl")
    )
    assert "--offline" not in cmd
    assert cmd[cmd.index("python") + 1] == "-I"
    assert cmd[-3:] == ["--worker", "--workdir", "/d"]
    offline = smoke.child_command(
        "uv", Path("/w/a.whl"), Path("/s/x.py"), "3.13", "numpy==2.5.4", True, []
    )
    assert "--offline" in offline and "numpy==2.5.4" in offline


def test_child_env_removes_what_could_swap_the_library():
    base = {
        "PATH": "/bin",
        "ARCADEDB_NATIVE_LIB": "/elsewhere/libarcadedb.so",
        "ARCADEDB_NATIVE_LIBDIR": "/elsewhere",
        "ARCADEDB_BACKEND": "jvm",
        "ARCADEDB_JVM_ARGS": "-Xmx1g",
        "PYTHONPATH": "/x",
        "ARCADEDB_NATIVE_ISOLATE_ARGS": "-Xmx1g",
    }
    env = smoke.child_env(base, "-Xmx8g -Darcadedb.txWalFlush=0")
    assert env["PATH"] == "/bin"
    assert env["ARCADEDB_NATIVE_ISOLATE_ARGS"] == "-Xmx8g -Darcadedb.txWalFlush=0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    for name in (
        "ARCADEDB_NATIVE_LIB",
        "ARCADEDB_NATIVE_LIBDIR",
        "ARCADEDB_BACKEND",
        "ARCADEDB_JVM_ARGS",
        "PYTHONPATH",
    ):
        assert name not in env
    assert "ARCADEDB_NATIVE_LIB" in base  # the caller's environment is not touched


def _good_result(**over):
    result = {
        "ok": True,
        "backend": "native",
        "lib_sha256": "c" * 64,
        "cpus_allowed": "6-7",
        "figures": {n: 123.0 for n in smoke.FIGURES},
    }
    result.update(over)
    return result


def test_parse_child_output_takes_the_last_result_line():
    first = smoke.RESULT_PREFIX + json.dumps({"n": 1})
    last = smoke.RESULT_PREFIX + json.dumps({"n": 2})
    text = f"engine log line\n{first}\nmore noise\n{last}\ntrailing log\n"
    assert smoke.parse_child_output(text) == {"n": 2}
    with pytest.raises(ValueError):
        smoke.parse_child_output("only logs\n")


def test_validate_child_result_accepts_a_good_report():
    assert smoke.validate_child_result(_good_result(), "c" * 64, "6-7") == []
    assert smoke.validate_child_result(_good_result(), "c" * 64, None) == []


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"ok": False, "error": "boom"}, "boom"),
        ({"backend": "jvm"}, "backend"),
        ({"lib_sha256": "d" * 64}, "library it loaded"),
        ({"cpus_allowed": "0-15"}, "cores"),
        ({"figures": {"update": 5.0}}, "figure new_order"),
        ({"figures": {n: float("nan") for n in smoke.FIGURES}}, "figure"),
        ({"figures": {n: 0.0 for n in smoke.FIGURES}}, "figure"),
    ],
)
def test_validate_child_result_names_the_problem(over, needle):
    problems = smoke.validate_child_result(_good_result(**over), "c" * 64, "6-7")
    assert problems and any(needle in p for p in problems)


# ----- the arguments ---------------------------------------------------------------------------


def _args(*extra):
    return smoke.parse_args(["--baseline", "base.whl", "cand.whl", *extra])


def test_check_args_refuses_what_cannot_read(tmp_path, monkeypatch):
    with pytest.raises(smoke.Refused):
        smoke.check_args(smoke.parse_args(["cand.whl"]))  # no baseline
    with pytest.raises(smoke.Refused):
        smoke.check_args(smoke.parse_args(["--baseline", "b.whl"]))  # no candidate
    for bad in ("7", "6", "0", "4"):  # odd, or below the minimum of a release run
        with pytest.raises(smoke.Refused):
            smoke.check_args(_args("--rounds", bad))
    with pytest.raises(smoke.Refused):
        smoke.check_args(_args("--rounds", "12"))  # the files do not exist
    base, cand = tmp_path / "base.whl", tmp_path / "cand.whl"
    base.write_bytes(b"x")
    cand.write_bytes(b"y")
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "/usr/bin/uv")
    args = smoke.parse_args(["--baseline", str(base), str(cand)])
    smoke.check_args(args)
    assert args.rounds == smoke.DEFAULT_ROUNDS
    quick = smoke.parse_args(["--baseline", str(base), str(cand), "--quick"])
    smoke.check_args(quick)
    assert quick.rounds == smoke.QUICK_ROUNDS
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: None)
    with pytest.raises(smoke.Refused):
        smoke.check_args(args)


def test_main_without_arguments_is_a_refusal(capsys):
    assert smoke.main([]) == smoke.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().err
