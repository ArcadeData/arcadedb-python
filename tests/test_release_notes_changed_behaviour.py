"""scripts/release_notes.py: the "Changed behaviour" line at the start of the notes.

It links the upgrade page of the docs as it is in the tag, and only when that page is about the release. No network: the API calls are faked.
"""

from __future__ import annotations

import base64
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "release_notes_changed_behaviour", ROOT / "scripts" / "release_notes.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


notes = _load()


def _notes(monkeypatch, tag, page):
    """The notes of `tag` in a repository with no tags and no milestone, whose upgrade page is `page` (None: no page)."""

    def gh_api(path):
        if path.startswith(f"repos/o/r/contents/{notes.UPGRADE_PAGE}?"):
            if page is None:
                return None
            return {"content": base64.b64encode(page.encode()).decode()}
        return {"default_branch": "main"} if path == "repos/o/r" else None

    monkeypatch.setattr(notes, "gh_api", gh_api)
    monkeypatch.setattr(notes, "gh_pages", lambda path: [])
    return notes.build_notes("o/r", tag)


@pytest.mark.parametrize("tag", ["26.12.1", "26.12.1.dev3"])
def test_the_notes_start_with_a_link_to_the_upgrade_page_in_the_tag(monkeypatch, tag):
    text = _notes(monkeypatch, tag, "= Upgrading to 26.12.1\n:description: x\n")
    assert text.splitlines()[0] == (
        "**Changed behaviour:** read [Upgrading to 26.12.1]"
        f"(https://github.com/o/r/blob/{tag}/docs/modules/ROOT/pages/upgrading.adoc) before you upgrade."
    )
    assert text.splitlines()[1] == ""


@pytest.mark.parametrize(
    "tag,page",
    [
        ("26.12.1", None),
        ("27.1.1", "= Upgrading to 26.12.1\n"),
        ("26.12.1", "= Upgrading to 26.12.10\n"),
        ("26.12.1", "= Upgrading\n\nUpgrading to 26.12.1 is described elsewhere.\n"),
        # a post release is a fix of a release that is out: its notes do not send the reader to the upgrade page
        ("26.12.1.post1", "= Upgrading to 26.12.1\n"),
    ],
)
def test_no_line_without_an_upgrade_page_for_the_release(monkeypatch, tag, page):
    assert "Changed behaviour" not in _notes(monkeypatch, tag, page)


def test_the_title_of_the_page_in_the_repository_names_a_release():
    page = (ROOT / notes.UPGRADE_PAGE).read_text(encoding="utf-8")
    assert re.match(r"= Upgrading to \d+\.\d+\.\d+\n", page)
