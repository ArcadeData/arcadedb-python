"""The package attributes say what the build recorded (the 26.12.1.dev0 native wheel recorded the snapshot but printed None).

native/wheel/build_wheel.py inserts its `from ._build_info import ...` near the top of arcadedb/__init__.py; a later fallback
in the same file once overwrote it. The attribute must be the recorded value on both builds.
"""

import importlib

import arcadedb


def test_the_package_attributes_are_what_the_build_recorded():
    module = (
        "arcadedb._build_info"
        if arcadedb.__backend__ == "native"
        else "arcadedb._version"
    )
    try:
        rec = importlib.import_module(module)
    except ImportError:
        assert arcadedb.ENGINE_SNAPSHOT is None and arcadedb.ENGINE_IMAGE_DIGEST is None
        return
    assert arcadedb.ENGINE_SNAPSHOT == getattr(rec, "ENGINE_SNAPSHOT", None)
    assert arcadedb.ENGINE_IMAGE_DIGEST == getattr(rec, "ENGINE_IMAGE_DIGEST", None)
