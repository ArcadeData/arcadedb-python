"""The old package name must not come back.

Runs scripts/check_no_embedded_name.py over the repository. It needs no wheel. The allowlist (history,
issue references, the fork's name on lines marked allow-old-name) lives in that script.
"""

import subprocess  # nosec B404 - runs sys.executable on a script this repository owns
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECK = ROOT / "scripts" / "check_no_embedded_name.py"


def test_old_package_name_is_absent():
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-I", str(CHECK), "--root", str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wildcards_for_the_old_wheel_name_are_caught_but_markdown_bold_is_not():
    """`dist/*embed*.whl` slipped through the first rename (#42); `**embedding**` is ordinary prose."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_no_embedded_name", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for bad in ("uv pip install dist/*embed*.whl", 'glob("*embedded*.whl")'):
        assert mod.OLD_NAME.search(bad), bad
    for ok in (
        "**embedding** models",
        "embedded multi-model database",
        "dist/arcadedb-*.whl",
    ):
        assert not mod.OLD_NAME.search(ok), ok
