"""The native wheel starts quietly: no polyglot-engine warning (#477)."""

import re
from pathlib import Path

ARCADE_NATIVE = (
    Path(__file__).resolve().parents[1]
    / "native/src/main/java/org/humemai/nativelib/ArcadeNative.java"
)


def test_the_image_glue_switches_the_polyglot_engine_off_before_any_entry_point():
    """The image has no polyglot languages; leaving the engine on logs 'no languages found' at WARNING in every process.

    Static check (no native build needed): the glue class that holds the entry points must set the option in a static initializer.
    """
    assert (
        ARCADE_NATIVE.is_file()
    ), f"{ARCADE_NATIVE} is missing: run the tests from a checkout of the repository"
    source = ARCADE_NATIVE.read_text()
    assert re.search(
        r"static\s*\{[^{}]*GlobalConfiguration\.POLYGLOT_ENGINE_ENABLED\.setValue\(false\);[^{}]*\}",
        source,
    ), "ArcadeNative must run GlobalConfiguration.POLYGLOT_ENGINE_ENABLED.setValue(false) in a static block"


# The run-time half (the first open logs nothing at WARNING) needs the native library and lives in
# test_native_startup_runtime.py, which conftest.py collects on the native backend only.
