"""The template of the native `arcadedb` wheel (native/wheel/) names the right distribution and says which package is the sibling.

Repository tooling: no engine, no JVM and no native library needed.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "native" / "wheel"


def test_the_template_is_the_arcadedb_distribution():
    meta = tomllib.loads((WHEEL / "pyproject.toml").read_text())["project"]
    assert meta["name"] == "arcadedb"
    assert meta["version"] == "@VERSION@"  # build_wheel.py replaces it
    assert meta["dependencies"] == []  # no JVM and no JPype
    assert "arcadedb-jvm" in meta["description"]


def test_python_classifiers_match_the_jvm_wheel():
    def pythons(path):
        classifiers = tomllib.loads(path.read_text())["project"]["classifiers"]
        return [
            c
            for c in classifiers
            if c.startswith("Programming Language :: Python :: 3.")
        ]

    assert pythons(WHEEL / "pyproject.toml") == pythons(ROOT / "pyproject.toml")


def test_the_pypi_description_names_the_sibling_in_its_first_paragraph():
    text = (WHEEL / "README.md").read_text()
    paragraphs = [
        p for p in re.split(r"\n\s*\n", text) if p.strip() and not p.startswith("#")
    ]
    first = " ".join(p for p in paragraphs[:1])
    assert "`arcadedb-jvm`" in first, first
    jvm = (ROOT / "README.md").read_text()
    head = jvm.split("##", 1)[0]
    assert "`arcadedb`" in head and "arcadedb-jvm" in head


def test_the_builder_has_no_other_package_name():
    text = (WHEEL / "build_wheel.py").read_text()
    assert 'dst = stage / "src" / "arcadedb"' in text
    assert "arcadedb_native" not in text
