"""scripts/dev_overlay.py: the Python sources of the checkout run over the installed wheel.

It is the shortcut of CONTRIBUTING.md ("Change Python code without building a wheel"). The test runs it against the wheel
under test, whichever of the two packages that is, so each CI leg shows that the overlay imports on its platform.
"""

import json
import os
import subprocess  # nosec B404 - runs this interpreter with a fixed argument list
import sys
from importlib import metadata
from pathlib import Path

import arcadedb
from arcadedb._dist_conflict import DISTRIBUTIONS

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "dev_overlay.py"
CODE = (
    "import arcadedb, json; "
    "print('RESULT ' + json.dumps([arcadedb.__file__, arcadedb.__backend__, arcadedb.__version__]))"
)


def test_overlay_imports_the_checkout_sources_over_the_installed_wheel(tmp_path):
    distribution = DISTRIBUTIONS[arcadedb._native.distribution_backend()]
    env = dict(os.environ, ARCADEDB_DEV_OVERLAY_DIR=str(tmp_path))
    run = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, str(SCRIPT), "-c", CODE],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    results = [ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(results) == 1, run.stdout + run.stderr
    file, backend, version = json.loads(results[0][len("RESULT ") :])

    overlay = tmp_path / distribution / "arcadedb"
    # the package that was imported is the overlay, not the wheel in site-packages
    assert Path(file).resolve() == (overlay / "__init__.py").resolve()
    # ... of the same build, with the version of the wheel it borrows the binaries from
    assert backend == ("native" if distribution == "arcadedb" else "jvm")
    assert version == metadata.version(distribution)
    # every source file of the checkout is there, as a file of its own (a link would be the wheel's)
    sources = sorted(
        p.relative_to(REPO / "src" / "arcadedb")
        for p in (REPO / "src" / "arcadedb").rglob("*.py")
        if "__pycache__" not in p.parts and p.name != "_version.py"
    )
    assert sources, "no sources found in the checkout"
    for rel in sources:
        assert (overlay / rel).is_file() and not (overlay / rel).is_symlink(), rel
    # the first line the script prints names the distribution and its version
    assert f"dev_overlay: {distribution} {version}:" in run.stdout
