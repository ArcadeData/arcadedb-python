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
