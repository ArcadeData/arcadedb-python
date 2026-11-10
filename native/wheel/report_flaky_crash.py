#!/usr/bin/env python3
"""report_flaky_crash.py: make a parity crash loud, once per workflow run.

native/wheel/parity_wheel.py retries a test file whose process died without a result (Windows legs only) and writes
<out stem>.meta.json beside its table, with one crash event per file: FLAKY-CRASH (the retry produced a result, so the leg may pass)
or CRASH (it died again, or no retry was allowed, so the leg fails), or DIRTY-EXIT (the file wrote its result and the process then
died in shutdown: recorded, never retried, the leg is not failed). This script reads the meta files of every leg (the downloaded
`native-parity-*` artifacts), then

  1. appends a table to the job summary ($GITHUB_STEP_SUMMARY), and
  2. posts ONE comment on issue 295 listing every event of the run (run URL, run attempt, Python, wheel, exit codes), but only on a
     workflow_dispatch or a tag push (a release), never on a pull request, never from another repository than ArcadeData/arcadedb-python.

The comment is a log, not a gate: when the token cannot write issues, or gh fails for any other reason, the script prints a `::warning::`
line and still exits 0. It never fails the run. The text carries no em dash.

  python native/wheel/report_flaky_crash.py --artifacts <dir with the downloaded artifacts> [--summary <file>] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess  # nosec B404 - runs the gh CLI with a fixed argument list
import sys

REPO = "ArcadeData/arcadedb-python"
ISSUE = 295
META = "parity-wheel.meta.json"


def load_events(root: str) -> list[dict]:
    """Every crash event of every leg under `root` (one artifact directory per leg), each with its leg's identity."""
    events = []
    for path in sorted(glob.glob(os.path.join(root, "**", META), recursive=True)):
        try:
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as err:
            print(f"::warning::{path} could not be read: {err}", flush=True)
            continue
        artifact = os.path.basename(os.path.dirname(path))
        for ev in meta.get("crash_events", []):
            events.append(
                {
                    **ev,
                    "leg": meta.get("key") or artifact,
                    "python": meta.get("python", ""),
                    "wheel": meta.get("wheel_version", ""),
                    "artifact": artifact,
                    "retry_on": bool(meta.get("retry_crash")),
                }
            )
    return events


def _table(events: list[dict]) -> list[str]:
    lines = [
        "| leg | Python | wheel | file | first exit code | retry |",
        "|---|---|---|---|---|---|",
    ]
    for e in events:
        retry = e["retry"] if not e["retry_rc"] else f"{e['retry']} ({e['retry_rc']})"
        lines.append(
            f"| {e['leg']} | {e['python']} | {e['wheel']} | `{e['file']}` | {e['kind']} {e['first_rc']} | {retry} |"
        )
    return lines


def _where(events: list[dict]) -> list[str]:
    lines = []
    for e in events:
        files = [e[k] for k in ("trace1", "trace2") if e.get(k)]
        # name.attempt1.log exists only when a retry ran (the retry's own log is name.log); a crash that was not retried has name.log alone
        log = e.get("log") or (
            f"{e['file']}.attempt1.log" if e.get("retry_rc") else f"{e['file']}.log"
        )
        names = ", ".join(f"`{n}`" for n in [*files, log])
        lines.append(
            f"- `{e['file']}` ({e['leg']}, Python {e['python']}): artifact `{e['artifact']}`, folder `parity-wheel-xml/`: {names}"
        )
    return lines


def summary_md(events: list[dict]) -> str:
    """The job summary block; empty when no file crashed."""
    if not events:
        return ""
    flaky = [e for e in events if e["kind"] == "FLAKY-CRASH"]
    hard = [e for e in events if e["kind"] == "CRASH"]
    dirty = [e for e in events if e["kind"] == "DIRTY-EXIT"]
    out = ["### Parity crashes", ""]
    if flaky:
        out += [
            f"**FLAKY-CRASH: {len(flaky)} file(s) died without a result and passed when run once more in a fresh process.** "
            "A native crash was hidden by the retry on these legs. Each one is logged on issue "
            f"#{ISSUE} (when the run is a release or a dispatch).",
            "",
        ]
    if hard:
        out += [
            f"**CRASH: {len(hard)} file(s) died without a result and the leg failed.**",
            "",
        ]
    if dirty:
        out += [
            f"**DIRTY-EXIT: {len(dirty)} file(s) wrote their result and then the process died in shutdown.** "
            "The result stands: the file was not retried and the leg was not failed by this.",
            "",
        ]
    out += [*_table(events), "", "Crash traces and logs:", ""]
    out += _where(events)
    return "\n".join(out) + "\n"


def comment_md(
    events: list[dict], run_url: str, attempt: str, event: str, ref: str
) -> str:
    """The comment for issue 295: one per run, every event in it."""
    kinds = sorted({e["kind"] for e in events})
    out = [
        f"{' and '.join(kinds)} in a native parity run.",
        "",
        f"Run: {run_url} (attempt {attempt or '1'}, event {event}, ref {ref})",
        "",
        *_table(events),
        "",
        "A test file died without a result and the harness ran it once more in a fresh process (`parity_wheel.py --retry-crash`, Windows legs only). "
        "A retry that says `passed` made its leg green: without the retry that leg would have failed. "
        "`CRASH` means the file died again (or no retry was left) and the leg failed. "
        "`DIRTY-EXIT` means the file wrote its result and the process then died in shutdown: it was not retried and did not fail the leg.",
        "",
        "The faulthandler trace and the log are in the uploaded parity artifact of each leg:",
        "",
        *_where(events),
    ]
    return "\n".join(out) + "\n"


def may_post(event: str, ref_type: str, repo: str) -> str:
    """'' when the comment may be posted, else the reason it may not."""
    if repo != REPO:
        return f"repository {repo or '(unknown)'} is not {REPO}"
    if event == "workflow_dispatch":
        return ""
    if event == "push" and ref_type == "tag":
        return ""
    return f"event {event or '(unknown)'} on a {ref_type or 'ref'} is neither a release nor a dispatch"


def post_comment(body: str, run=subprocess.run) -> tuple[bool, str]:
    """POST the comment with the gh CLI and GH_TOKEN; (True, '') or (False, why). Never raises."""
    cmd = [
        "gh",
        "api",
        "--method",
        "POST",
        f"repos/{REPO}/issues/{ISSUE}/comments",
        "--input",
        "-",
    ]
    try:
        r = run(
            cmd,
            input=json.dumps({"body": body}),
            capture_output=True,
            text=True,
            timeout=120,
        )  # nosec B603
    except (OSError, subprocess.SubprocessError) as err:
        return False, f"{type(err).__name__}: {err}"
    if r.returncode != 0:
        return (
            False,
            (r.stderr or r.stdout or f"gh exited {r.returncode}").strip()[:300],
        )
    return True, ""


def report(events, env, summary_path="", dry_run=False, run=subprocess.run) -> int:
    """Write the summary, post the comment when allowed. Returns 0: a log is never a gate."""
    if not events:
        print("no crash event in this run", flush=True)
        return 0
    md = summary_md(events)
    print(md, flush=True)
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(md)
    server = env.get("GITHUB_SERVER_URL", "https://github.com")
    repo = env.get("GITHUB_REPOSITORY", "")
    run_url = f"{server}/{repo}/actions/runs/{env.get('GITHUB_RUN_ID', '')}"
    body = comment_md(
        events,
        run_url,
        env.get("GITHUB_RUN_ATTEMPT", ""),
        env.get("GITHUB_EVENT_NAME", ""),
        env.get("GITHUB_REF_NAME", ""),
    )
    why = may_post(
        env.get("GITHUB_EVENT_NAME", ""), env.get("GITHUB_REF_TYPE", ""), repo
    )
    if why:
        print(f"issue #{ISSUE}: no comment posted ({why})", flush=True)
        return 0
    if dry_run:
        print(f"issue #{ISSUE}: dry run, the comment would be:\n{body}", flush=True)
        return 0
    if not (env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")):
        print(
            f"::warning::the FLAKY-CRASH comment on #{ISSUE} was skipped: no GH_TOKEN",
            flush=True,
        )
        return 0
    ok, err = post_comment(body, run=run)
    if ok:
        print(f"issue #{ISSUE}: comment posted", flush=True)
    else:
        print(
            f"::warning::the FLAKY-CRASH comment on #{ISSUE} was skipped, the token cannot write issues or gh failed: {err}",
            flush=True,
        )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY", ""))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        return report(load_events(a.artifacts), os.environ, a.summary, a.dry_run)
    except Exception as err:  # noqa: BLE001 - a log is never a gate
        print(
            f"::warning::the crash report failed: {type(err).__name__}: {err}",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
