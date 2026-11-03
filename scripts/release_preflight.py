#!/usr/bin/env python3
"""Release-day preflight: is everything ready for the owner to push the tag VERSION?

Read-only. Every check prints PASS, FAIL or SKIP with its evidence; the exit code is the number
of FAIL lines (0 means the tag may be pushed). It fails closed: a network or tool error is a
FAIL, never a PASS. It never tags, uploads, edits or writes anything in the repository.

VERSION is the tag to push: A.B.C (final), A.B.C.postN (Python-only fix of a released engine) or
A.B.C.devN. The engine is the release the tag is named after (A.B.C), so for A.B.C.postN the
upstream checks (a) to (c) look at A.B.C. A devN tag is built from `latest` (JVM) and DEV_ENGINE
(native): upstream checks (a) to (c) are SKIPped for it.

Checks:
  (a) the GitHub release (not a draft or pre-release) of the engine exists on ArcadeData/arcadedb
  (b) Maven Central has com.arcadedb:arcadedb-engine and arcadedb-server at the engine version
      (the native build compiles them with GraalVM native-image)
  (c) Docker Hub has arcadedata/arcadedb:<engine> for linux/amd64 and linux/arm64
      (the JVM wheels take their engine jars from that image)
  (d) scripts/check_version.py accepts VERSION against the git tags and PyPI, for arcadedb-jvm
      and for arcadedb
  (e) the latest conclusive run of Test native, Test JVM and Test Python Examples on main is green
      on every job
  (f) the version and engine the release workflows derive from the tag are VERSION and the engine
      (the real `run:` code of the workflow steps is executed with TAG=VERSION, not copied here)
  (g) for a final or post tag: the repository variable ARCADEDB_NATIVE_READINESS_BAR_MET is true

The two upstream channels (Maven Central, Docker Hub) are published by separate steps of the
upstream release and can lag each other: both must pass before the tag is pushed.

usage:
    release_preflight.py VERSION [--remote URL] [--repo OWNER/NAME] [--checkout DIR]

Needs: network, git (for check_version.py), gh (logged in), bash (for check f).
Tests: tests/test_release_preflight.py. Output is ASCII only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess  # nosec B404 - runs git/gh/bash/python with fixed argument lists
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

UPSTREAM = "ArcadeData/arcadedb"
OWN_REPO = "ArcadeData/arcadedb-python"
OWN_REMOTE = "https://github.com/ArcadeData/arcadedb-python.git"
MAVEN = "https://repo1.maven.org/maven2/com/arcadedb"
MAVEN_ARTIFACTS = ("arcadedb-engine", "arcadedb-server")
DOCKER_TAG_URL = "https://hub.docker.com/v2/repositories/arcadedata/arcadedb/tags"
PACKAGES = ("arcadedb-jvm", "arcadedb")
TEST_WORKFLOWS = (
    "test-native.yml",
    "test-jvm.yml",
    "test-python-examples.yml",
)
BAR_VARIABLE = "ARCADEDB_NATIVE_READINESS_BAR_MET"
HTTP_TIMEOUT = 30
SCRIPTS = Path(__file__).resolve().parent


class Result(NamedTuple):
    status: str  # PASS, FAIL or SKIP
    check: str
    evidence: str


# ----- the two seams the tests replace: HTTP and subprocess -------------------------------


def http_get(url: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    """GET a fixed https URL. Returns (status, body); an HTTP error status is returned, not raised.

    Anything else (DNS, timeout, TLS) raises OSError, which the checks turn into a FAIL.
    """
    if not url.startswith("https://"):
        raise OSError(f"refusing URL {url}")
    request = urllib.request.Request(url, headers=headers or {})  # noqa: S310
    try:
        with urllib.request.urlopen(  # nosec B310 - https URL checked above
            request, timeout=HTTP_TIMEOUT
        ) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise OSError(f"cannot read {url}: {err}") from err


def run(
    cmd: list[str], env: dict[str, str] | None = None, cwd: str | None = None
) -> tuple[int, str, str]:
    """Run a command with a fixed argument list. Returns (returncode, stdout, stderr)."""
    try:
        done = subprocess.run(  # nosec B603 - fixed argument list, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=cwd,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as err:
        raise OSError(f"cannot run {cmd[0]}: {err}") from err
    return done.returncode, done.stdout, done.stderr


def _load_check_version():
    spec = importlib.util.spec_from_file_location(
        "check_version", SCRIPTS / "check_version.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # nosec B101 - spec of an existing file
    spec.loader.exec_module(module)
    return module


# ----- the version and the engine ---------------------------------------------------------


def engine_of(version: str) -> str | None:
    """The engine release a tag is named after: A.B.C for A.B.C and A.B.C.postN, None for a devN."""
    cv = _load_check_version()
    parsed = cv.parse_version(version)
    return None if parsed.kind == "dev" else parsed.base


def _http_error(check: str, err: OSError) -> Result:
    return Result("FAIL", check, f"cannot read the source (fail closed): {err}")


# ----- (a) upstream release ---------------------------------------------------------------


def check_upstream_release(engine: str) -> list[Result]:
    check = f"(a) GitHub release {engine} on {UPSTREAM}"
    base = f"https://api.github.com/repos/{UPSTREAM}"
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        status, body = http_get(f"{base}/releases/tags/{engine}", headers)
        if status == 200:
            data = json.loads(body)
            if data.get("tag_name") != engine:
                return [
                    Result("FAIL", check, f"release has tag {data.get('tag_name')!r}")
                ]
            if data.get("draft") or data.get("prerelease"):
                flags = [k for k in ("draft", "prerelease") if data.get(k)]
                return [Result("FAIL", check, f"release is marked {'/'.join(flags)}")]
            return [
                Result(
                    "PASS",
                    check,
                    f"published {data.get('published_at')} {data.get('html_url')}",
                )
            ]
        if status != 404:
            return [Result("FAIL", check, f"GitHub answered HTTP {status}")]
        status, _ = http_get(f"{base}/git/ref/tags/{engine}", headers)
        if status == 200:
            return [
                Result(
                    "FAIL", check, "the tag exists but there is no published release"
                )
            ]
        if status == 404:
            return [Result("FAIL", check, "neither a release nor a tag of that name")]
        return [Result("FAIL", check, f"GitHub answered HTTP {status} for the tag")]
    except (OSError, ValueError) as err:
        return [_http_error(check, OSError(str(err)))]


# ----- (b) Maven Central ------------------------------------------------------------------


def check_maven(engine: str) -> list[Result]:
    results = []
    for artifact in MAVEN_ARTIFACTS:
        url = f"{MAVEN}/{artifact}/{engine}/{artifact}-{engine}.pom"
        check = f"(b) Maven Central com.arcadedb:{artifact}:{engine}"
        try:
            status, _ = http_get(url)
        except OSError as err:
            results.append(_http_error(check, err))
            continue
        if status == 200:
            results.append(Result("PASS", check, f"HTTP 200 {url}"))
        elif status == 404:
            results.append(Result("FAIL", check, f"HTTP 404 {url} (not published yet)"))
        else:
            results.append(Result("FAIL", check, f"HTTP {status} {url}"))
    return results


# ----- (c) Docker Hub ---------------------------------------------------------------------


def check_docker(engine: str) -> list[Result]:
    check = f"(c) Docker Hub arcadedata/arcadedb:{engine}"
    url = f"{DOCKER_TAG_URL}/{engine}"
    try:
        status, body = http_get(url)
        if status == 404:
            return [Result("FAIL", check, f"HTTP 404 {url} (not published yet)")]
        if status != 200:
            return [Result("FAIL", check, f"HTTP {status} {url}")]
        data = json.loads(body)
    except (OSError, ValueError) as err:
        return [_http_error(check, OSError(str(err)))]
    if data.get("name") != engine:
        return [Result("FAIL", check, f"the tag answers as {data.get('name')!r}")]
    archs = sorted(
        {
            image.get("architecture")
            for image in data.get("images", [])
            if image.get("architecture") and image.get("architecture") != "unknown"
        }
    )
    missing = [a for a in ("amd64", "arm64") if a not in archs]
    if missing:
        return [
            Result(
                "FAIL",
                check,
                f"architectures {','.join(archs) or 'none'}; missing {','.join(missing)}",
            )
        ]
    return [
        Result(
            "PASS",
            check,
            f"architectures {','.join(archs)}, last updated {data.get('last_updated')}",
        )
    ]


# ----- (d) version rule -------------------------------------------------------------------


def check_version_rule(version: str, remote: str) -> list[Result]:
    results = []
    for package in PACKAGES:
        check = f"(d) check_version.py {version} for {package}"
        cmd = [
            sys.executable,
            "-I",
            str(SCRIPTS / "check_version.py"),
            version,
            "--git-remote",
            remote,
            "--pypi",
            package,
        ]
        try:
            code, out, err = run(cmd)
        except OSError as err_:
            results.append(_http_error(check, err_))
            continue
        line = (out.strip().splitlines() or [err.strip() or "no output"])[-1]
        if code == 0:
            results.append(Result("PASS", check, line))
        elif code == 1:
            results.append(Result("FAIL", check, line))
        else:
            results.append(
                Result(
                    "FAIL", check, f"exit {code}, a source could not be read: {line}"
                )
            )
    return results


# ----- (e) CI on main ---------------------------------------------------------------------


def _gh_json(args: list[str]):
    code, out, err = run(["gh", *args])
    if code != 0:
        raise OSError(f"gh {' '.join(args[:2])} failed: {err.strip() or out.strip()}")
    try:
        return json.loads(out)
    except ValueError as exc:
        raise OSError(f"gh {' '.join(args[:2])} printed no JSON") from exc


def check_ci(repo: str) -> list[Result]:
    results = []
    try:
        tip = _gh_json(["api", f"repos/{repo}/commits/main"]).get("sha", "")
    except OSError:
        tip = ""  # only used for a note
    for workflow in TEST_WORKFLOWS:
        check = f"(e) latest conclusive {workflow} run on main"
        try:
            runs = _gh_json(
                [
                    "run",
                    "list",
                    "-R",
                    repo,
                    "--workflow",
                    workflow,
                    "--branch",
                    "main",
                    "--limit",
                    "20",
                    "--json",
                    "databaseId,status,conclusion,event,headSha,createdAt",
                ]
            )
            # A cancelled run was replaced by a newer commit; the latest run that reached a verdict counts.
            verdicts = [
                r
                for r in runs
                if r.get("status") == "completed"
                and r.get("conclusion") not in ("cancelled", "skipped")
            ]
            if not verdicts:
                results.append(Result("FAIL", check, "no completed run on main found"))
                continue
            latest = verdicts[0]
            jobs = _gh_json(
                [
                    "run",
                    "view",
                    str(latest["databaseId"]),
                    "-R",
                    repo,
                    "--json",
                    "jobs",
                ]
            ).get("jobs", [])
        except (OSError, KeyError) as err:
            results.append(_http_error(check, OSError(str(err))))
            continue
        counts: dict[str, int] = {}
        for job in jobs:
            key = job.get("conclusion") or job.get("status") or "unknown"
            counts[key] = counts.get(key, 0) + 1
        bad = [
            f"{j.get('name')} ({j.get('conclusion') or j.get('status')})"
            for j in jobs
            if j.get("conclusion") not in ("success", "skipped")
        ]
        sha = str(latest.get("headSha", ""))
        note = ""
        if tip and sha and tip != sha:
            note = f"; main is now at {tip[:12]}, this run is for an earlier commit"
        newer = runs[: runs.index(latest)]
        if newer:
            note += "; newer runs: " + ",".join(
                f"{r.get('status')}/{r.get('conclusion') or '-'}" for r in newer[:3]
            )
        evidence = (
            f"run {latest['databaseId']} ({latest.get('event')}, {latest.get('createdAt')}, "
            f"sha {sha[:12]}) jobs "
            + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
            + note
        )
        if latest.get("conclusion") != "success" or bad or not jobs:
            detail = "; not green: " + ", ".join(bad[:6]) if bad else ""
            if not jobs:
                detail = "; the run lists no jobs"
            results.append(Result("FAIL", check, evidence + detail))
        elif counts.get("success", 0) == 0:
            results.append(Result("FAIL", check, evidence + "; no job succeeded"))
        else:
            results.append(Result("PASS", check, evidence))
    return results


# ----- (f) what the release workflows derive from the tag ---------------------------------


def extract_step_run(text: str, step_name: str) -> str | None:
    """The shell code of the `run: |` block of the step `- name: <step_name>`, dedented, or None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)- name: " + re.escape(step_name) + r"\s*$", line)
        if not m:
            continue
        step_indent = len(m.group(1))
        for j in range(i + 1, len(lines)):
            current = lines[j]
            if current.strip() and len(current) - len(current.lstrip()) <= step_indent:
                return None  # the next step started before a run block
            r = re.match(r"^(\s*)run: \|\s*$", current)
            if not r:
                continue
            run_indent = len(r.group(1))
            body = []
            for k in range(j + 1, len(lines)):
                nxt = lines[k]
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= run_indent:
                    break
                body.append(nxt)
            block_indent = min(
                (len(b) - len(b.lstrip()) for b in body if b.strip()), default=0
            )
            return "\n".join(b[block_indent:] for b in body) + "\n"
        return None
    return None


def _env_values(text: str, names: tuple[str, ...]) -> dict[str, str]:
    values = {}
    for name in names:
        m = re.search(r"^\s{2}" + name + r': "([^"]*)"\s*$', text, re.M)
        if m:
            values[name] = m.group(1)
    return values


def _run_step(script: str, env: dict[str, str]) -> dict[str, str]:
    """Run a workflow step's shell code under bash and return what it wrote to GITHUB_OUTPUT."""
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "github_output"
        out_file.write_text("", encoding="utf-8")
        full_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "GITHUB_OUTPUT": str(out_file),
            **env,
        }
        code, _, err = run(["bash", "-c", script], env=full_env, cwd=tmp)
        if code != 0:
            raise ValueError(f"the step exited {code}: {err.strip().splitlines()[-1:]}")
        pairs = {}
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                pairs[key] = value
        return pairs


def check_derivation(version: str, repo_root: Path) -> list[Result]:
    cv = _load_check_version()
    parsed = cv.parse_version(version)
    engine = None if parsed.kind == "dev" else parsed.base
    final = parsed.kind != "dev"
    if shutil.which("bash") is None:
        return [
            Result(
                "SKIP",
                "(f) workflow derivation of version and engine",
                "bash not found",
            )
        ]
    workflows = repo_root / ".github" / "workflows"
    results = []

    def step(file: str, name: str) -> tuple[str | None, str]:
        try:
            text = (workflows / file).read_text(encoding="utf-8")
        except OSError as err:
            return None, f"cannot read {file}: {err}"
        script = extract_step_run(text, name)
        if script is None:
            return (
                None,
                f"{file} has no step {name!r} with a run block (the workflow changed: update this check)",
            )
        return script, text

    # release-jvm.yml: image tag the JVM wheels take their jars from
    check = f"(f) release-jvm.yml derives, for tag {version}"
    script, text = step("release-jvm.yml", "Validate tag")
    if script is None:
        results.append(Result("FAIL", check, text))
    else:
        try:
            got = _run_step(script, {"TAG": version})
            want_image = engine if engine else "latest"
            problems = []
            if got.get("version") != version:
                problems.append(f"version {got.get('version')!r}")
            if got.get("arcadedb-tag") != want_image:
                problems.append(
                    f"image tag {got.get('arcadedb-tag')!r}, expected {want_image!r}"
                )
            if engine and got.get("base-version") != engine:
                problems.append(
                    f"engine {got.get('base-version')!r}, expected {engine!r}"
                )
            evidence = (
                f"version {got.get('version')}, engine {got.get('base-version')}, "
                f"image arcadedata/arcadedb:{got.get('arcadedb-tag')}"
            )
            results.append(
                Result("FAIL", check, evidence + "; wrong: " + ", ".join(problems))
                if problems
                else Result("PASS", check, evidence)
            )
        except (ValueError, OSError) as err:
            results.append(Result("FAIL", check, str(err)))

    # release-native.yml: the tag is accepted and its final flag
    check = f"(f) release-native.yml derives, for tag {version}"
    script, text = step("release-native.yml", "Validate tag")
    if script is None:
        results.append(Result("FAIL", check, text))
    else:
        try:
            got = _run_step(script, {"TAG": version})
            evidence = f"version {got.get('version')}, final {got.get('final')}"
            problems = []
            if got.get("version") != version:
                problems.append(f"version {got.get('version')!r}")
            if got.get("final") != ("true" if final else "false"):
                problems.append(f"final {got.get('final')!r}")
            results.append(
                Result("FAIL", check, evidence + "; wrong: " + ", ".join(problems))
                if problems
                else Result("PASS", check, evidence)
            )
        except (ValueError, OSError) as err:
            results.append(Result("FAIL", check, str(err)))

    # test-native.yml: the engine the native image links
    check = f"(f) test-native.yml links, for version {version}"
    script, text = step("test-native.yml", "Resolve the version and the engine")
    if script is None:
        results.append(Result("FAIL", check, text))
    else:
        env = {"INPUT_VERSION": version, "INPUT_ENGINE": ""}
        env.update(_env_values(text, ("DEFAULT_VERSION", "DEV_ENGINE")))
        try:
            got = _run_step(script, env)
            evidence = f"engine {got.get('engine')}, final {got.get('final')}"
            if not final:
                evidence += " (devN: DEV_ENGINE)"
            problems = []
            if got.get("version") != version:
                problems.append(f"version {got.get('version')!r}")
            if got.get("final") != ("true" if final else "false"):
                problems.append(f"final {got.get('final')!r}")
            if engine and got.get("engine") != engine:
                problems.append(f"engine {got.get('engine')!r}, expected {engine!r}")
            results.append(
                Result("FAIL", check, evidence + "; wrong: " + ", ".join(problems))
                if problems
                else Result("PASS", check, evidence)
            )
        except (ValueError, OSError) as err:
            results.append(Result("FAIL", check, str(err)))
    return results


# ----- (g) readiness bar ------------------------------------------------------------------


def check_readiness_bar(version: str, repo: str) -> list[Result]:
    check = f"(g) repository variable {BAR_VARIABLE}"
    if re.search(r"\.dev[0-9]+$", version):
        return [Result("SKIP", check, "a devN native release does not need the bar")]
    try:
        code, out, err = run(
            [
                "gh",
                "api",
                f"repos/{repo}/actions/variables/{BAR_VARIABLE}",
                "--jq",
                ".value",
            ]
        )
    except OSError as exc:
        return [_http_error(check, exc)]
    if code == 0:
        value = out.strip()
        if value == "true":
            return [Result("PASS", check, "true")]
        return [
            Result(
                "FAIL",
                check,
                f"value is {value!r}: release-native.yml publishes no final wheel until a maintainer sets it to true",
            )
        ]
    if "404" in err:
        return [
            Result(
                "FAIL",
                check,
                "the variable is not set: release-native.yml publishes no final wheel",
            )
        ]
    return [
        Result(
            "SKIP",
            check,
            f"cannot read repository variables with this gh login ({err.strip()[:120]}); the owner checks it",
        )
    ]


# ----- main -------------------------------------------------------------------------------


def run_checks(version: str, remote: str, repo: str, repo_root: Path) -> list[Result]:
    cv = _load_check_version()
    try:
        cv.parse_version(version)
    except cv.VersionError as err:
        return [Result("FAIL", "(0) version syntax", str(err))]
    engine = engine_of(version)
    results: list[Result] = []
    if engine is None:
        for letter, what in (
            ("a", "GitHub release"),
            ("b", "Maven Central"),
            ("c", "Docker Hub"),
        ):
            results.append(
                Result(
                    "SKIP",
                    f"({letter}) upstream {what}",
                    "a devN tag is built from latest (JVM) and DEV_ENGINE (native), not from a release",
                )
            )
    else:
        results += check_upstream_release(engine)
        results += check_maven(engine)
        results += check_docker(engine)
    results += check_version_rule(version, remote)
    results += check_ci(repo)
    results += check_derivation(version, repo_root)
    results += check_readiness_bar(version, repo)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "version", help="the tag to push (A.B.C, A.B.C.postN or A.B.C.devN)"
    )
    parser.add_argument(
        "--remote", default=OWN_REMOTE, help="git remote read for the tags"
    )
    parser.add_argument(
        "--repo", default=OWN_REPO, help="GitHub repository of the CI runs"
    )
    parser.add_argument("--checkout", dest="repo_root", default=str(SCRIPTS.parent))
    args = parser.parse_args(argv)

    results = run_checks(args.version, args.remote, args.repo, Path(args.repo_root))
    for r in results:
        print(f"{r.status:<5} {r.check}: {r.evidence}")
    fails = sum(1 for r in results if r.status == "FAIL")
    skips = sum(1 for r in results if r.status == "SKIP")
    print(f"\n{fails} FAIL(s), {skips} SKIP(s) for {args.version}")
    if fails == 0:
        print("Ready: the owner may push the tag (the only publishing step).")
    return fails


if __name__ == "__main__":
    sys.exit(main())
