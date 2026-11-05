#!/usr/bin/env python3
"""Write the notes of the GitHub Release of one tag from the milestone of its version.

The tag is a bare version (docs page "Release"): `A.B.C.devN`, `A.B.C` or `A.B.C.postN`. The milestone is named after the base
`A.B.C`. The notes list the pull requests of that milestone that are merged, grouped by label, then the install lines, the engine
version and the PyPI pages of the two packages.

Which pull requests a tag lists:

- `A.B.C`           every merged pull request of the milestone that is in the tag (the first release of a base is the whole cycle)
- `A.B.C.devN`      those that are in the tag and not in the previous tag of the same base (`A.B.C.dev(N-1)`), so a preview lists only
                    what is new since the last preview
- `A.B.C.postN`     those that are in the tag and not in the previous tag of the same base (`A.B.C` or `A.B.C.post(N-1)`)

"In a tag" means the merge commit of the pull request is an ancestor of the tag (the GitHub compare API), so a pull request that merges
after the tag, while the release still runs, is not listed. A tag that does not exist yet (a dry run before the release) is read as the
tip of the default branch.

Only reads: every API call is a GET through `gh api` (the token is the GH_TOKEN of the caller). Nothing is created or changed here;
the release workflow does that with the file this script writes.

usage:
    release_notes.py TAG --dry-run               # print the notes
    release_notes.py TAG --output notes.md       # write them to a file
    release_notes.py TAG --repo OWNER/NAME       # default: $GITHUB_REPOSITORY, else ArcadeData/arcadedb-python

Exit codes: 0 notes written, 1 the tag is not a valid release tag, 2 an API call failed.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from urllib.parse import quote

DEFAULT_REPO = "ArcadeData/arcadedb-python"
UPSTREAM = "ArcadeData/arcadedb"
NATIVE_DIST = "arcadedb"
JVM_DIST = "arcadedb-jvm"

TAG_RE = re.compile(r"^(?P<base>\d+\.\d+\.\d+)(?:\.(?P<kind>dev|post)(?P<n>\d+))?$")

# First matching label wins; a pull request with none of them goes to "Other".
SECTIONS = [
    ("Bug fixes", "bug"),
    ("Performance", "performance"),
    ("Features", "enhancement"),
    ("Documentation", "documentation"),
]
OTHER = "Other"


class ApiError(Exception):
    pass


def gh_api(path):
    """GET one API path through `gh api` and return the decoded JSON (None for a 404)."""
    cmd = ["gh", "api", path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if "HTTP 404" in proc.stderr:
            return None
        raise ApiError(f"gh api {path}: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def gh_pages(path):
    """All pages of a list endpoint (100 per page)."""
    out = []
    sep = "&" if "?" in path else "?"
    page = 1
    while True:
        chunk = gh_api(f"{path}{sep}per_page=100&page={page}") or []
        out.extend(chunk)
        if len(chunk) < 100:
            return out
        page += 1


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(
            f"'{tag}' is not A.B.C, A.B.C.devN or A.B.C.postN (bare version, no v, no aN/bN/rcN)"
        )
    kind = m.group("kind") or "final"
    n = int(m.group("n") or 0)
    return m.group("base"), kind, n


def order_key(kind, n):
    """Order inside one base: dev0 < dev1 < ... < final < post1 < post2 ..."""
    return {"dev": (0, n), "final": (1, 0), "post": (2, n)}[kind]


def previous_tag(tag, existing):
    """The tag a release's list starts after, or None (a final lists the whole milestone)."""
    base, kind, n = parse_tag(tag)
    if kind == "final":
        return None
    me = order_key(kind, n)
    best = None
    for other in existing:
        m = TAG_RE.match(other)
        if not m or m.group("base") != base:
            continue
        okind = m.group("kind") or "final"
        key = order_key(okind, int(m.group("n") or 0))
        if key < me and (best is None or key > best[0]):
            best = (key, other)
    return best[1] if best else None


def section_of(labels):
    names = set(labels)
    for title, label in SECTIONS:
        if label in names:
            return title
    return OTHER


def clean_title(title, number):
    """The title without a trailing (#N) of the pull request itself, which the line adds."""
    return re.sub(rf"\s*\(#{number}\)\s*$", "", title.strip())


def safe_title(title):
    """No @mention in a release body: a zero width space after the @ stops the notification."""
    return re.sub(r"@(?=\w)", "@​", title.strip())


def is_ancestor(repo, sha, ref):
    """True when commit `sha` is `ref` itself or one of its ancestors."""
    data = gh_api(f"repos/{repo}/compare/{sha}...{quote(ref, safe='')}?per_page=1")
    return bool(data) and data.get("status") in ("ahead", "identical")


def milestone_prs(repo, base):
    """(milestone dict or None, merged pull requests of it as dicts with number, title, labels, merge sha)."""
    milestones = gh_pages(f"repos/{repo}/milestones?state=all")
    milestone = next((m for m in milestones if m["title"] == base), None)
    if milestone is None:
        return None, []
    items = gh_pages(f"repos/{repo}/issues?milestone={milestone['number']}&state=all")
    prs = []
    for item in items:
        pull = item.get("pull_request")
        if not pull or not pull.get("merged_at"):
            continue
        detail = gh_api(f"repos/{repo}/pulls/{item['number']}")
        if not detail or not detail.get("merge_commit_sha"):
            continue
        prs.append(
            {
                "number": item["number"],
                "title": item["title"],
                "labels": [lb["name"] for lb in item.get("labels", [])],
                "sha": detail["merge_commit_sha"],
                "merged_at": pull["merged_at"],
            }
        )
    return milestone, prs


def engine_of(repo, kind, base, ref):
    """(engine version, how it is known). A final or post wheel links the engine it is named after."""
    if kind != "dev":
        return base, "release"
    data = gh_api(
        f"repos/{repo}/contents/.github/workflows/test-native.yml?ref={quote(ref, safe='')}"
    )
    if data and data.get("content"):
        text = base64.b64decode(data["content"]).decode()
        m = re.search(r'^\s*DEV_ENGINE:\s*"?([0-9][^"\s]*)"?', text, re.M)
        if m:
            return m.group(1), "dev"
    return None, "dev"


def build_notes(repo, tag):
    base, kind, n = parse_tag(tag)
    default_branch = (gh_api(f"repos/{repo}") or {}).get("default_branch", "main")
    tag_ref = gh_api(f"repos/{repo}/git/ref/tags/{quote(tag, safe='')}")
    if tag_ref:
        ref = tag
    else:
        ref = default_branch
        print(
            f"note: tag {tag} does not exist yet, reading the tip of {default_branch}",
            file=sys.stderr,
        )

    tags = [t["name"] for t in gh_pages(f"repos/{repo}/tags")]
    prev = previous_tag(tag, [t for t in tags if t != tag])
    milestone, prs = milestone_prs(repo, base)

    listed = []
    for pr in prs:
        if not is_ancestor(repo, pr["sha"], ref):
            continue
        if prev and is_ancestor(repo, pr["sha"], prev):
            continue
        listed.append(pr)
    listed.sort(key=lambda p: p["number"])

    groups = {title: [] for title, _ in SECTIONS}
    groups[OTHER] = []
    for pr in listed:
        groups[section_of(pr["labels"])].append(pr)

    engine, engine_kind = engine_of(repo, kind, base, ref)
    lines = []
    if kind == "dev":
        lines.append(
            f"Pre-release of `{base}`: a preview built from the engine release named below, for trying the next release early."
        )
    elif kind == "post":
        lines.append(
            f"A fix in the Python code of `{base}`. The wheels still carry the `{base}` engine."
        )
    lines.append("")

    if milestone is None:
        lines.append(
            f"No milestone named `{base}` exists, so this release has no change list."
        )
        print(f"warning: no milestone named {base}", file=sys.stderr)
        lines.append("")
    else:
        scope = f"since `{prev}`" if prev else f"in the milestone `{base}`"
        lines.append(f"## What changed {scope}")
        lines.append("")
        if not listed:
            lines.append("No merged pull request.")
            lines.append("")
        for title in [t for t, _ in SECTIONS] + [OTHER]:
            if not groups[title]:
                continue
            lines.append(f"### {title}")
            lines.append("")
            for pr in groups[title]:
                shown = safe_title(clean_title(pr["title"], pr["number"]))
                lines.append(f"- {shown} (#{pr['number']})")
            lines.append("")

    lines.append("## Install")
    lines.append("")
    lines.append("```bash")
    if kind == "dev":
        lines.append(f'uv add "{NATIVE_DIST}=={tag}"      # native build')
        lines.append(f'uv add "{JVM_DIST}=={tag}"  # JVM build')
    else:
        lines.append(f"uv add {NATIVE_DIST}      # native build")
        lines.append(f"uv add {JVM_DIST}  # JVM build")
    lines.append("```")
    lines.append("")
    if kind == "dev":
        lines.append(
            "A pre-release is installed only when it is named exactly, as above, or with `--prerelease allow`."
        )
        lines.append("")

    lines.append("## Engine")
    lines.append("")
    if engine:
        lines.append(
            f"ArcadeDB engine [{engine}](https://github.com/{UPSTREAM}/releases/tag/{engine})."
        )
    else:
        lines.append("ArcadeDB engine: not found in `test-native.yml` for this tag.")
    if kind == "dev":
        lines.append(
            f"This is the engine the native `{NATIVE_DIST}` wheels link. The `{JVM_DIST}` preview wheels take the jars of the upstream image tag `latest`."
        )
    lines.append("")

    lines.append("## Packages")
    lines.append("")
    lines.append(
        f"- [{NATIVE_DIST} {tag}](https://pypi.org/project/{NATIVE_DIST}/{tag}/)"
    )
    lines.append(f"- [{JVM_DIST} {tag}](https://pypi.org/project/{JVM_DIST}/{tag}/)")
    lines.append("")
    if prev:
        lines.append(
            f"Full change list: https://github.com/{repo}/compare/{prev}...{tag}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tag", help="the bare version, for example 26.11.1")
    ap.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the notes (the default output)"
    )
    ap.add_argument("--output", help="write the notes to this file instead of stdout")
    args = ap.parse_args(argv)
    try:
        parse_tag(args.tag)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        notes = build_notes(args.repo, args.tag)
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output and not args.dry_run:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(notes)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
