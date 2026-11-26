"""scripts/release_notes.py: what an earlier release already shipped is not listed again, and internal pull requests are collapsed.

A final or `devN` release lists its milestone. Fixes that also went out in a `postN` release of the base before it (as an ancestor of
the tag, or as a cherry-pick that names the commit) are left out, and a pull request that changes only `.github/`, `tests/` or
`scripts/` goes into one collapsed block. No network: the API calls are faked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_notes.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "release_notes_shipped_and_internal", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


notes = _load()


@pytest.mark.parametrize(
    "tag,previous",
    [
        ("26.12.1", "26.11.1.post10"),
        ("26.12.1.dev0", "26.11.1.post10"),
        ("27.1.1", "26.12.1"),
        ("26.11.1", None),
        ("26.11.1.post3", None),
    ],
)
def test_the_previous_release_is_the_newest_final_or_post_of_an_earlier_base(
    tag, previous
):
    tags = [
        "26.11.1.dev0",
        "26.11.1",
        "26.11.1.post2",
        "26.11.1.post10",
        "26.12.1.dev2",
        "26.12.1",
        "not-a-release",
    ]
    assert notes.previous_release(tag, [t for t in tags if t != tag]) == previous


def _sha(n):
    return f"{n:040x}"


# number: (title, labels, the files it changes)
PULLS = {
    1: ("A fix a user sees", ["bug"], ["src/arcadedb/core.py", "tests/test_core.py"]),
    2: ("Merged before the post release", ["bug"], ["src/arcadedb/core.py"]),
    3: ("Ported to the post release", ["performance"], ["src/arcadedb/results.py"]),
    4: ("A flaky test", ["bug"], ["tests/test_a.py", ".github/workflows/x.yml"]),
    5: ("A release script", [], ["scripts/release_notes.py"]),
    6: ("A build script and a page", [], ["scripts/build.sh", "docs/page.adoc"]),
    7: ("A module that moved into tests", [], [("tests/helper.py", "src/helper.py")]),
}
TAGS = ["26.11.1", "26.11.1.post1", "26.12.1.dev0"]


def _fake(monkeypatch, tags=TAGS):
    """A repository whose milestone 26.12.1 holds PULLS: #2 is an ancestor of 26.11.1.post1, #3 is cherry-picked into it."""

    def gh_api(path):
        if path == "repos/o/r":
            return {"default_branch": "main"}
        if path.startswith("repos/o/r/pulls/"):
            number = int(path.rsplit("/", 1)[1])
            return {"merge_commit_sha": _sha(number)}
        if path == "repos/o/r/compare/26.11.1...26.11.1.post1?per_page=100&page=1":
            message = f"Port a fix\n\n(cherry picked from commit {_sha(3)})\n"
            return {"commits": [{"sha": _sha(99), "commit": {"message": message}}]}
        if path.startswith("repos/o/r/compare/"):
            pair = path[len("repos/o/r/compare/") :].split("?")[0]
            sha, ref = pair.split("...")
            number = int(sha, 16)
            inside = ref == "main" or (ref == "26.11.1.post1" and number == 2)
            return {"status": "ahead" if inside else "diverged"}
        return None

    def gh_pages(path):
        if path == "repos/o/r/tags":
            return [{"name": name} for name in tags]
        if path.startswith("repos/o/r/milestones"):
            return [{"title": "26.12.1", "number": 2}]
        if path.startswith("repos/o/r/issues?milestone=2"):
            return [
                {
                    "number": number,
                    "title": title,
                    "labels": [{"name": label} for label in labels],
                    "pull_request": {"merged_at": "2026-11-20T00:00:00Z"},
                }
                for number, (title, labels, _) in PULLS.items()
            ]
        if path.startswith("repos/o/r/pulls/") and path.endswith("/files"):
            files = PULLS[int(path.split("/")[4])][2]
            return [
                (
                    {"filename": f[0], "previous_filename": f[1]}
                    if isinstance(f, tuple)
                    else {"filename": f}
                )
                for f in files
            ]
        return []

    monkeypatch.setattr(notes, "gh_api", gh_api)
    monkeypatch.setattr(notes, "gh_pages", gh_pages)


def _split(text):
    """(the notes before the collapsed block, the block)."""
    head, _, rest = text.partition("<details>")
    return head, rest.partition("</details>")[0]


def test_a_final_release_leaves_out_what_shipped_and_collapses_internal_changes(
    monkeypatch,
):
    _fake(monkeypatch)
    text = notes.build_notes("o/r", "26.12.1")
    head, block = _split(text)

    assert "(#2)" not in text and "(#3)" not in text
    assert "What shipped in `26.11.1.post1` or an earlier release is not listed" in head
    assert "### Bug fixes\n\n- A fix a user sees (#1)\n" in head
    assert "- A build script and a page (#6)" in head
    assert "- A module that moved into tests (#7)" in head
    assert "(#4)" not in head and "(#5)" not in head
    assert "Internal: CI, tests, and release tooling only (2)" in block
    assert "- A flaky test (#4)\n- A release script (#5)\n" in block


def test_without_an_earlier_release_nothing_is_left_out(monkeypatch):
    _fake(monkeypatch, tags=["26.12.1.dev0"])
    text = notes.build_notes("o/r", "26.12.1")
    assert "(#2)" in text and "(#3)" in text
    assert "is not listed again" not in text


def test_a_release_with_only_internal_changes_still_lists_them(monkeypatch):
    _fake(monkeypatch)
    monkeypatch.setitem(PULLS, 1, ("A test only", ["bug"], ["tests/test_core.py"]))
    for number in (6, 7):
        monkeypatch.setitem(PULLS, number, ("Tooling", [], ["scripts/x.sh"]))
    text = notes.build_notes("o/r", "26.12.1")
    assert "###" not in text
    assert "No merged pull request." not in text
    assert (
        "<summary>Internal: CI, tests, and release tooling only (5)</summary>" in text
    )
