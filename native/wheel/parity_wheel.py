#!/usr/bin/env python3
"""parity_wheel.py: run the test suite of this repository against the INSTALLED `arcadedb` (native) wheel, file by file, and count outcomes per file.

  python native/wheel/parity_wheel.py --repo <checkout> --python <venv python with the wheel installed> --out parity-native.tsv

A run per test file in its own process (a hang or crash costs one file), junit XML per file, then one table:
passed / failed / error / NOTYET (the test needs a Java-object API, reported by tests/conftest.py as a skip whose reason starts
with NOTYET, raised as NotOnNativeYet) / NA-JVM (a row of NATIVE_NA_JVM_TESTS: the test reaches into the JVM on purpose) / other skips / TIMEOUT-or-crash. Files in tests/conftest.py's NATIVE_NOT_APPLICABLE are not collected (JVM-only).

A file whose process dies without writing its result is a CRASH, recorded with its exit code: CRASH(rc=0xC0000005,3221225477). With
--retry-crash (test-native.yml passes it on Windows legs only; the default is strict) the file is run once more in a fresh process: a retry
that produces a result is FLAKY-CRASH(rc=...) in the status column (the gate lets it pass, the job summary and issue 295 record it), a second
crash stays CRASH and fails the leg. A failing test, a TIMEOUT and a BADXML are never retried. The decision is native/wheel/parity_crash.py.
On Windows each process also writes a faulthandler trace to <name>.attemptN.fault.txt next to its XML (tests/conftest.py,
ARCADEDB_TEST_FAULTHANDLER_FILE); an empty one is deleted. Beside the table, <out stem>.meta.json lists the wheel, the Python and every crash.
"""

import argparse
import ast
import glob
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import parity_crash as pc  # the directory of this script is sys.path[0]

COUNT_KEYS = ["passed", "failed", "error", "notyet", "na_jvm", "skipped"]

# On Windows the test process writes a faulthandler trace to this file (tests/conftest.py, native backend only).
FAULT_FILE_ENV = "ARCADEDB_TEST_FAULTHANDLER_FILE"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument(
        "--cores",
        default="0-5,8-11",
        help="taskset core list (Linux only; ignored where taskset does not exist)",
    )
    p.add_argument("--files", default="")
    p.add_argument(
        "--max-notyet",
        type=int,
        default=None,
        help="fail (exit 1) when more tests than this are NOTYET skips; a final release uses 0",
    )
    p.add_argument(
        "--retry-crash",
        action="store_true",
        help="run a test file whose process died without a result ONCE more in a fresh process (FLAKY-CRASH when that "
        "produces a result, CRASH when it dies again). Off by default: the workflow sets it on Windows legs only.",
    )
    p.add_argument(
        "--max-crash-retries",
        type=int,
        default=3,
        help="with --retry-crash, the most files of one run that may be retried; a crash after that stays a CRASH",
    )
    return p.parse_args(argv)


def write_text(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def count_xml(xml):
    """(counts, names of the failing tests) of a junit XML, or None when it does not parse."""
    c = dict.fromkeys(COUNT_KEYS, 0)
    fails = []
    try:
        for tc in ET.parse(xml).getroot().iter("testcase"):
            if tc.find("failure") is not None:
                c["failed"] += 1
                fails.append(tc.get("name"))
            elif tc.find("error") is not None:
                c["error"] += 1
                fails.append(tc.get("name"))
            elif tc.find("skipped") is not None:
                msg = tc.find("skipped").get("message") or ""
                c[
                    (
                        "notyet"
                        if msg.startswith("NOTYET")
                        else "na_jvm" if msg.startswith("NA-JVM") else "skipped"
                    )
                ] += 1
            else:
                c["passed"] += 1
    except ET.ParseError:
        return None
    return c, fails


def main(argv=None):
    a = parse_args(argv)
    pkg = os.path.abspath(a.repo)
    files = sorted(glob.glob(os.path.join(pkg, "tests", "test_*.py")))

    na_files, win_files = {}, {}
    for node in ast.parse(open(os.path.join(pkg, "tests", "conftest.py")).read()).body:
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", "")
            if name == "NATIVE_NOT_APPLICABLE":
                na_files = ast.literal_eval(node.value)
            elif name == "WINDOWS_NOT_APPLICABLE":
                win_files = ast.literal_eval(node.value)
    files = [f for f in files if os.path.basename(f) not in na_files]
    if (
        sys.platform == "win32"
    ):  # conftest.py's collect_ignore does not apply to a file named on the command line
        files = [f for f in files if os.path.basename(f) not in win_files]
    if not os.path.isfile(os.path.join(pkg, "docs", "index.md")):
        # tests/conftest.py collect-ignores test_docs_examples.py without the Markdown docs site; a file named on the command line is collected anyway
        files = [f for f in files if os.path.basename(f) != "test_docs_examples.py"]
    if a.files:
        want = set(a.files.split(","))
        files = [f for f in files if os.path.basename(f) in want]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)  # the wheel, not a source tree, must be what is imported
    env.pop("ARCADEDB_BACKEND", None)  # the backend comes from the installed distribution
    env.pop("ARCADEDB_NATIVE_LIB", None)  # the library comes from the installed wheel
    env.pop(FAULT_FILE_ENV, None)  # set per attempt below, on Windows only
    chk = subprocess.run(
        [
            a.python,
            "-I",
            "-c",
            "import json, sys, arcadedb as n; print(n.__file__, n.__version__, n.__backend__, n.ENGINE_VERSION, n.SOURCE_COMMIT); "
            'print("META", json.dumps({"python": sys.version.split()[0], "wheel_version": n.__version__, "engine_version": n.ENGINE_VERSION, "source_commit": n.SOURCE_COMMIT})); '
            'assert n.__backend__ == "native"; assert "site-packages" in n.__file__',
        ],
        env=env,
        cwd="/",
        capture_output=True,
        text=True,
    )
    print("WHEEL CHECK:", chk.stdout.strip(), chk.stderr.strip()[-300:], flush=True)
    assert chk.returncode == 0, "tests would not run against the installed wheel"
    ident = {}
    for line in chk.stdout.splitlines():
        if line.startswith("META "):
            ident = json.loads(line[len("META ") :])
    a.out = os.path.abspath(a.out)
    xmldir = os.path.splitext(a.out)[0] + "-xml"
    os.makedirs(xmldir, exist_ok=True)
    policy = pc.RetryPolicy(enabled=a.retry_crash, budget=a.max_crash_retries)
    if policy.enabled:
        print(
            f"CRASH RETRY: on, a file that dies without a result is run once more (at most {policy.budget} files)",
            flush=True,
        )
    rows, events = [], []
    tot = dict.fromkeys([*COUNT_KEYS, "timeout", "flaky_crash", "crash"], 0)

    for f in files:
        name = os.path.basename(f)
        xml = os.path.join(xmldir, name + ".xml")
        log = os.path.join(xmldir, name + ".log")
        parsed = {}  # attempt number -> (counts, failing tests) of its XML

        def attempt(n):
            if n > 1:
                # keep the log of the first attempt: the retry writes name.log
                os.replace(log, os.path.join(xmldir, name + ".attempt1.log"))
            if os.path.exists(xml):
                os.remove(xml)  # a stale XML of an earlier run must not pass for a result
            trace = os.path.join(xmldir, f"{name}.attempt{n}.fault.txt")
            env_n = dict(env)
            if sys.platform == "win32":
                env_n[FAULT_FILE_ENV] = trace
            cmd = [
                *(
                    ["taskset", "-c", a.cores]
                    if sys.platform == "linux" and shutil.which("taskset")
                    else []
                ),
                *(
                    ["nice", "-n", "10"]
                    if sys.platform != "win32" and shutil.which("nice")
                    else []
                ),
                a.python,
                "-m",
                "pytest",
                # No --capture=sys on Windows: pytest's default fd capture (dup2 onto fd 1 and 2) is the regression check for engine log
                # text landing in database files there (tests/test_log_redirect.py is the direct test; the whole-file runs of test_vector.py
                # were the ones that failed before the engine stopped logging to the console on Windows).
                f"tests/{name}",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={xml}",
                "-o",
                "junit_family=xunit1",
            ]
            timed_out, rc = False, None
            try:
                r = subprocess.run(
                    cmd,
                    cwd=pkg,
                    env=env_n,
                    capture_output=True,
                    text=True,
                    timeout=a.timeout,
                )
                rc = r.returncode
                write_text(log, r.stdout[-20000:] + "\n" + r.stderr[-5000:])
            except subprocess.TimeoutExpired as e:
                timed_out = True
                write_text(
                    log,
                    (
                        ((e.stdout or b"")[-8000:]).decode()
                        if isinstance(e.stdout, bytes)
                        else str(e.stdout)[-8000:]
                    ),
                )
            if os.path.exists(trace) and os.path.getsize(trace) == 0:
                os.remove(trace)
            result, bad_xml = False, False
            if os.path.exists(xml):
                got = count_xml(xml)
                if got is None:
                    bad_xml = True
                else:
                    result, parsed[n] = True, got
            return pc.Attempt(
                rc=rc, result=result, timed_out=timed_out, bad_xml=bad_xml
            )

        out = pc.run_file(attempt, policy)
        c, fails = parsed.get(len(out.attempts), (dict.fromkeys(COUNT_KEYS, 0), []))
        for k in c:
            tot[k] += c[k]
        tot["timeout"] += out.timed_out
        tot["flaky_crash"] += out.flaky
        tot["crash"] += out.hard_crash
        ev = pc.event_of(name, out, c["failed"] + c["error"])
        if ev:
            for n in range(1, len(out.attempts) + 1):
                trace = f"{name}.attempt{n}.fault.txt"
                if os.path.exists(os.path.join(xmldir, trace)):
                    ev[f"trace{n}"] = trace
            events.append(ev)
        rows.append((name, c, out.status, fails, len(out.attempts), out.final.rc))
        print(
            f"{name:45s} pass={c['passed']:4d} fail={c['failed']:3d} err={c['error']:3d} notyet={c['notyet']:3d} na_jvm={c['na_jvm']:3d} skip={c['skipped']:3d} {out.status}",
            flush=True,
        )
        if out.retried:
            print(
                f"  {name}: the first run died without a result ({pc.format_rc(out.attempts[0].rc)}); "
                f"the run once more: {'a result' if out.final.result else 'no result again'} ({pc.format_rc(out.final.rc)})",
                flush=True,
            )
        for n, at in enumerate(out.attempts, 1):
            if pc.dirty_exit(at):
                print(
                    f"::warning::{name} attempt {n} wrote its result and then exited with {pc.format_rc(at.rc)}",
                    flush=True,
                )
    with open(a.out, "w") as fh:
        fh.write(
            "file\tpassed\tfailed\terror\tnotyet\tna_jvm\tskipped\tstatus\tfailing_tests\tattempts\texit\n"
        )
        for name, c, status, fails, attempts, rc in rows:
            fh.write(
                f"{name}\t{c['passed']}\t{c['failed']}\t{c['error']}\t{c['notyet']}\t{c['na_jvm']}\t{c['skipped']}\t{status}\t{';'.join(fails[:40])}\t{attempts}\t{pc.format_rc(rc)}\n"
            )
        fh.write(
            f"TOTAL\t{tot['passed']}\t{tot['failed']}\t{tot['error']}\t{tot['notyet']}\t{tot['na_jvm']}\t{tot['skipped']}\ttimeouts={tot['timeout']} flaky_crash={tot['flaky_crash']} crash={tot['crash']}\t\t\t\n"
        )
    with open(os.path.splitext(a.out)[0] + ".meta.json", "w") as fh:
        json.dump(
            {
                "key": env.get("KEY", ""),
                "platform": sys.platform,
                "python": ident.get("python", ""),
                "wheel_version": ident.get("wheel_version", ""),
                "engine_version": ident.get("engine_version", ""),
                "source_commit": ident.get("source_commit", ""),
                "retry_crash": policy.enabled,
                "crash_events": events,
            },
            fh,
            indent=1,
        )
    print("TOTAL", tot)
    if a.max_notyet is not None and tot["notyet"] > a.max_notyet:
        sys.exit(
            f"{tot['notyet']} tests are NOTYET skips, above the limit {a.max_notyet}"
        )


if __name__ == "__main__":
    main()
