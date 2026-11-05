"""scripts/bump_engine.py: the four engine pins move together, drift is reported, a bad NEXT_DEV is refused.

The tests run on fixture copies of the real files, so an anchor that drifts in the repository fails here.
The versions are made-up ones (99.x), never a real release.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bump_engine.py"
FIXTURE_FILES = (
    ".github/workflows/test-native.yml",
    "native/pom.xml",
    "native/scripts/build-image.sh",
    "native/wheel/THIRD_PARTY_NOTICES.md",
    "native/wheel/build_wheel.py",
)


def _load():
    spec = importlib.util.spec_from_file_location("bump_engine", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["bump_engine"] = module
    spec.loader.exec_module(module)
    return module


be = _load()


@pytest.fixture
def tree(tmp_path):
    for rel in FIXTURE_FILES:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)
    published = tmp_path / "published.txt"
    published.write_text("98.1.1\n98.1.1.dev0\n98.12.1\n99.1.0\n", encoding="utf-8")
    return tmp_path


def _run(tree, *extra, engine="99.1.1", dev="99.2.1.dev0"):
    return be.main(
        [
            "--engine",
            engine,
            "--dev-version",
            dev,
            "--root",
            str(tree),
            "--published-file",
            str(tree / "published.txt"),
            *extra,
        ]
    )


def _snapshot(tree):
    return {rel: (tree / rel).read_bytes() for rel in FIXTURE_FILES}


def test_edit_moves_all_the_pins(tree, capsys):
    assert _run(tree) == 0
    workflow = (tree / ".github/workflows/test-native.yml").read_text()
    assert '  DEV_ENGINE: "99.1.1"\n' in workflow
    assert '  DEFAULT_VERSION: "99.2.1.dev0"\n' in workflow
    assert 'required: false\n        default: "99.2.1.dev0"' in workflow
    assert (
        '"--version",\n        default="99.2.1.dev0",'
        in (tree / "native/wheel/build_wheel.py").read_text()
    )
    assert (
        "<arcadedb.version>99.1.1</arcadedb.version>"
        in (tree / "native/pom.xml").read_text()
    )
    assert (
        "${ARCADEDB_ENGINE_VERSION:-99.1.1}"
        in (tree / "native/scripts/build-image.sh").read_text()
    )
    out = capsys.readouterr().out
    assert out.count("DRIFT") == 6
    assert "TODO" in out  # the notices follow the engine and are not edited here
    # nothing else changed: only those lines differ from the real files
    for rel in FIXTURE_FILES[:3]:
        old = (REPO / rel).read_text().splitlines()
        new = (tree / rel).read_text().splitlines()
        assert len(old) == len(new)
        assert sum(a != b for a, b in zip(old, new)) in (1, 3)
    assert (tree / "native/wheel/THIRD_PARTY_NOTICES.md").read_bytes() == (
        REPO / "native/wheel/THIRD_PARTY_NOTICES.md"
    ).read_bytes()


def test_second_run_is_a_no_op(tree, capsys):
    assert _run(tree) == 0
    after = _snapshot(tree)
    assert _run(tree) == 0
    assert _snapshot(tree) == after
    assert "nothing to edit" in capsys.readouterr().out


def test_check_reports_drift_and_writes_nothing(tree, capsys):
    before = _snapshot(tree)
    assert _run(tree, "--check") == 1
    assert _snapshot(tree) == before
    assert "DRIFT" in capsys.readouterr().out


def test_check_still_fails_while_the_notices_name_another_engine(tree, capsys):
    assert _run(tree) == 0
    capsys.readouterr()
    assert _run(tree, "--check") == 1  # pins match, notices do not
    out = capsys.readouterr().out
    assert "DRIFT" not in out
    assert "TODO" in out
    notices = tree / "native/wheel/THIRD_PARTY_NOTICES.md"
    notices.write_text(_with_engine(notices.read_text(), "99.1.1"), encoding="utf-8")
    assert _run(tree, "--check") == 0


def _with_engine(text: str, engine: str) -> str:
    return be.NOTICES_RE.sub(
        lambda m: m.group(0).replace(m.group(1), engine), text, count=1
    )


@pytest.mark.parametrize(
    "engine, dev, why",
    [
        ("99.1.1.dev0", "99.2.1.dev0", "must be a release"),
        ("99.1.1.post1", "99.2.1.dev0", "must be a release"),
        ("99.1.1", "99.2.1", "must be a devN"),
        ("99.1.1", "99.1.1.dev0", "higher than the engine"),
        ("99.1.1", "98.5.1.dev0", "higher than the engine"),
        ("99.1.1", "99.2.1.dev3", "next dev version of 99.2.1 is dev0"),
        ("99.1.1", "99.2.1.rc1", "not X.Y.Z"),
        ("99.1", "99.2.1.dev0", "not X.Y.Z"),
    ],
)
def test_refused_versions_write_nothing(tree, capsys, engine, dev, why):
    before = _snapshot(tree)
    assert _run(tree, engine=engine, dev=dev) == 1
    assert _snapshot(tree) == before
    assert why in capsys.readouterr().out


def test_dev_version_of_a_published_base_is_refused(tree, capsys):
    (tree / "published.txt").write_text("99.2.1\n", encoding="utf-8")
    before = _snapshot(tree)
    assert _run(tree) == 1
    assert "already released" in capsys.readouterr().out
    assert _snapshot(tree) == before


@pytest.mark.parametrize(
    "rel, old, new",
    [
        (FIXTURE_FILES[0], "DEV_ENGINE:", "DEV_ENGINE_RENAMED:"),
        (FIXTURE_FILES[0], "DEFAULT_VERSION:", "DEFAULT_RENAMED:"),
        (
            FIXTURE_FILES[0],
            "Wheel version to build (PEP 440: A.B.C.devN,",
            "Wheel version (PEP 440: A.B.C.devN,",
        ),
        (FIXTURE_FILES[1], "<arcadedb.version>", "<arcadedb.engine>"),
        (FIXTURE_FILES[2], "ARCADEDB_ENGINE_VERSION:-", "ARCADEDB_ENGINE:-"),
        (FIXTURE_FILES[4], '"--version",', '"--wheel-version",'),
    ],
)
def test_missing_anchor_fails_loudly_and_writes_nothing(tree, capsys, rel, old, new):
    path = tree / rel
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    before = _snapshot(tree)
    assert _run(tree) == 2
    assert _snapshot(tree) == before
    assert "exactly one anchor" in capsys.readouterr().out


def test_repeated_anchor_fails_loudly(tree, capsys):
    path = tree / "native/pom.xml"
    path.write_text(path.read_text() + "<arcadedb.version>1.0.0</arcadedb.version>\n")
    before = _snapshot(tree)
    assert _run(tree) == 2
    assert _snapshot(tree) == before
    assert "found 2" in capsys.readouterr().out


def test_missing_file_is_an_error(tree, capsys):
    (tree / "native/pom.xml").unlink()
    assert _run(tree) == 2
    assert "cannot read native/pom.xml" in capsys.readouterr().out


def test_published_file_read_error_is_exit_2(tree, capsys):
    (tree / "published.txt").unlink()
    assert _run(tree) == 2
    assert "ERROR" in capsys.readouterr().out
