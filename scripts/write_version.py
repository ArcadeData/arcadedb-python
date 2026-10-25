#!/usr/bin/env python3
"""Write version to _version.py file during build"""

import io
import sys
from pathlib import Path

from extract_version import default_jars_dir, extract_version

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def write_version_file(ver, out_path):
    """Write version to _version.py"""
    version_file_content = f'''"""Version information for arcadedb package.
Generated automatically during build from the bundled ArcadeDB engine jar.
"""

__version__ = "{ver}"
'''

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(version_file_content)

    print(f"✅ Version {ver} written to {out_path}")


if __name__ == "__main__":
    # Get the jars directory from the command line or use the package's own
    script_dir = Path(__file__).parent
    bindings_dir = script_dir.parent
    jars_dir = default_jars_dir()

    if len(sys.argv) > 1:
        jars_dir = Path(sys.argv[1])

    # Extract version from the engine jar
    try:
        version = extract_version(jars_dir)

        # Write to src/arcadedb/_version.py
        output_path = bindings_dir / "src/arcadedb/_version.py"
        write_version_file(version, output_path)

    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
