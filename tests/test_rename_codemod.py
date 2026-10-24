"""The rename codemod (scripts/rename_to_arcadedb.py) and the old-name check.

Runs both on a small synthetic repository in a temporary directory, so no wheel is needed. What it
pins: the package directory moves, imports and metadata are rewritten, history on the allowlist is not
touched, issue references of the old repository keep their text, a second run changes nothing, and the
check passes on the result and fails on a planted old name.
"""

import subprocess  # nosec B404 - runs git and sys.executable on scripts this repository owns
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
CODEMOD = SCRIPTS / "rename_to_arcadedb.py"
CHECK = SCRIPTS / "check_no_embedded_name.py"

FILES = {
    "src/arcadedb_embedded/__init__.py": '"""arcadedb_embedded."""\n__version__ = "1"\n',
    "src/arcadedb_embedded/core.py": "from arcadedb_embedded import jvm\n",
    "pyproject.toml": (
        '[project]\nname = "arcadedb-embedded"\n'
        '[tool.setuptools]\npackages = ["arcadedb_embedded"]\n'
        '[tool.setuptools.package-data]\narcadedb_embedded = ["jars/*.jar"]\n'
        '[project.urls]\nHomepage = "https://github.com/humemai/arcadedb-embedded-python"\n'
    ),
    "tests/test_x.py": "import arcadedb_embedded as arcadedb\n\n\ndef test_x():\n    assert arcadedb\n",
    "scripts/build.sh": 'PACKAGE_NAME="arcadedb-embedded"\nls dist/arcadedb_embedded-*.whl\n',
    ".gitignore": "src/arcadedb_embedded/jars/\n",
    "README.md": (
        "# Title\n\n```bash\nuv add arcadedb-embedded   # or: pip install arcadedb-embedded\n```\n\n"
        "Fixed in humemai/arcadedb-embedded-python#12 and ARCADEDB_HOME stays.\n"
        "Engine: com.arcadedb.Constants and arcadedata/arcadedb stay.\n"
    ),
    "docs/superpowers/plans/history.md": "The old package arcadedb-embedded lives on in this plan.\n",
}


def git(root, *args):
    subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
    )  # nosec B603 B607


def run(script, root, *args):
    return subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-I", str(script), "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repo(tmp_path):
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))  # LF on every platform
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    return tmp_path


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git/" not in p.as_posix()
    }


def test_check_fails_on_the_old_name_and_passes_on_the_allowlist(repo):
    result = run(CHECK, repo)
    assert result.returncode == 1
    assert "pyproject.toml" in result.stdout
    assert "docs/superpowers" not in result.stdout  # allowlisted history
    assert (
        "humemai/arcadedb-embedded-python#12" not in result.stdout
    )  # allowlisted issue reference


def test_codemod_renames_and_is_idempotent(repo):
    first = run(CODEMOD, repo, "--no-format")
    assert first.returncode == 0, first.stdout + first.stderr

    assert (repo / "src/arcadedb/__init__.py").is_file()
    assert not (repo / "src/arcadedb_embedded").exists()
    assert (repo / "src/arcadedb/core.py").read_text() == "from arcadedb import jvm\n"
    assert (repo / "tests/test_x.py").read_text().startswith("import arcadedb\n")
    pyproject = (repo / "pyproject.toml").read_text()
    assert 'name = "arcadedb"' in pyproject
    assert 'packages = ["arcadedb"]' in pyproject
    assert '\narcadedb = ["jars/*.jar"]' in pyproject
    assert "github.com/ArcadeData/arcadedb-python" in pyproject
    assert (
        repo / "scripts/build.sh"
    ).read_text() == 'PACKAGE_NAME="arcadedb"\nls dist/arcadedb-*.whl\n'
    assert (repo / ".gitignore").read_text() == "src/arcadedb/jars/\n"

    readme = (repo / "README.md").read_text()
    assert "uv add arcadedb   # or: pip install arcadedb\n" in readme
    assert (
        "This package was published as arcadedb-embedded until 26.10.x; install arcadedb now."
        in readme
    )
    assert (
        "humemai/arcadedb-embedded-python#12" in readme
    )  # issue reference keeps its text
    assert (
        "ARCADEDB_HOME" in readme
        and "com.arcadedb.Constants" in readme
        and "arcadedata/arcadedb" in readme
    )

    history = (repo / "docs/superpowers/plans/history.md").read_text()
    assert history == FILES["docs/superpowers/plans/history.md"]  # not rewritten

    assert (repo / "tests/test_no_embedded_name.py").is_file()
    assert run(CHECK, repo).returncode == 0

    git(repo, "add", "-A")
    before = snapshot(repo)
    second = run(CODEMOD, repo, "--no-format")
    assert second.returncode == 0, second.stdout + second.stderr
    assert snapshot(repo) == before
    assert "total" in second.stdout and "   0 / 0" in second.stdout


def test_dry_run_changes_nothing(repo):
    before = snapshot(repo)
    result = run(CODEMOD, repo, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert snapshot(repo) == before
    assert "dry run" in result.stdout


def test_check_catches_a_planted_old_name_after_the_rename(repo):
    assert run(CODEMOD, repo, "--no-format").returncode == 0
    (repo / "src/arcadedb/core.py").write_text("import arcadedb_embedded\n")
    result = run(CHECK, repo)
    assert result.returncode == 1
    assert "src/arcadedb/core.py:1" in result.stdout
