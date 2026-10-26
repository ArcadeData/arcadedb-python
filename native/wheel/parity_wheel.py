#!/usr/bin/env python3
"""parity_wheel.py: run the test suite of this repository against the INSTALLED `arcadedb` (native) wheel, file by file, and count outcomes per file.

  python native/wheel/parity_wheel.py --repo <checkout> --python <venv python with the wheel installed> --out parity-native.tsv

A run per test file in its own process (a hang or crash costs one file), junit XML per file, then one table:
passed / failed / error / NOTYET (the test needs a Java-object API, reported by tests/conftest.py as a skip whose reason starts
with NOTYET, raised as NotOnNativeYet) / NA-JVM (a row of NATIVE_NA_JVM_TESTS: the test reaches into the JVM on purpose) / other skips / TIMEOUT-or-crash. Files in tests/conftest.py's NATIVE_NOT_APPLICABLE are not collected (JVM-only).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

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
a = p.parse_args()

pkg = os.path.abspath(a.repo)
files = sorted(glob.glob(os.path.join(pkg, "tests", "test_*.py")))
import ast

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
chk = subprocess.run(
    [
        a.python,
        "-I",
        "-c",
        "import arcadedb as n; print(n.__file__, n.__version__, n.__backend__, n.ENGINE_VERSION, n.SOURCE_COMMIT); "
        'assert n.__backend__ == "native"; assert "site-packages" in n.__file__',
    ],
    env=env,
    cwd="/",
    capture_output=True,
    text=True,
)
print("WHEEL CHECK:", chk.stdout.strip(), chk.stderr.strip()[-300:], flush=True)
assert chk.returncode == 0, "tests would not run against the installed wheel"
a.out = os.path.abspath(a.out)
xmldir = os.path.splitext(a.out)[0] + "-xml"
os.makedirs(xmldir, exist_ok=True)
rows, tot = [], dict.fromkeys(
    ["passed", "failed", "error", "notyet", "na_jvm", "skipped", "timeout"], 0
)
for f in files:
    name = os.path.basename(f)
    xml = os.path.join(xmldir, name + ".xml")
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
    try:
        r = subprocess.run(
            cmd, cwd=pkg, env=env, capture_output=True, text=True, timeout=a.timeout
        )
        status = ""
        open(os.path.join(xmldir, name + ".log"), "w").write(
            r.stdout[-20000:] + "\n" + r.stderr[-5000:]
        )
    except subprocess.TimeoutExpired as e:
        status = "TIMEOUT"
        open(os.path.join(xmldir, name + ".log"), "w").write(
            ((e.stdout or b"")[-8000:]).decode()
            if isinstance(e.stdout, bytes)
            else str(e.stdout)[-8000:]
        )
    c = dict.fromkeys(["passed", "failed", "error", "notyet", "na_jvm", "skipped"], 0)
    fails = []
    if os.path.exists(xml):
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
            status = status or "BADXML"
    else:
        status = status or "CRASH"
    for k in c:
        tot[k] += c[k]
    if status == "TIMEOUT":
        tot["timeout"] += 1
    rows.append((name, c, status, fails))
    print(
        f"{name:45s} pass={c['passed']:4d} fail={c['failed']:3d} err={c['error']:3d} notyet={c['notyet']:3d} na_jvm={c['na_jvm']:3d} skip={c['skipped']:3d} {status}",
        flush=True,
    )
with open(a.out, "w") as fh:
    fh.write(
        "file\tpassed\tfailed\terror\tnotyet\tna_jvm\tskipped\tstatus\tfailing_tests\n"
    )
    for name, c, status, fails in rows:
        fh.write(
            f"{name}\t{c['passed']}\t{c['failed']}\t{c['error']}\t{c['notyet']}\t{c['na_jvm']}\t{c['skipped']}\t{status}\t{';'.join(fails[:40])}\n"
        )
    fh.write(
        f"TOTAL\t{tot['passed']}\t{tot['failed']}\t{tot['error']}\t{tot['notyet']}\t{tot['na_jvm']}\t{tot['skipped']}\ttimeouts={tot['timeout']}\t\n"
    )
print("TOTAL", tot)
if a.max_notyet is not None and tot["notyet"] > a.max_notyet:
    sys.exit(f"{tot['notyet']} tests are NOTYET skips, above the limit {a.max_notyet}")
