"""Native build: jvector's Highway kernels are the default vector provider where the wheel ships their library (#153).

The engine forces jvector's scalar provider in every native image unless ARCADEDB_VECTOR_PROVIDER says otherwise. The glue sets it to
``native`` (Highway) when libjvector-<arch>.so sits next to the engine library, which the wheel ships on Linux x86-64 and aarch64. A
value the user set wins, and ``default`` forces the scalar provider.
"""

import json
import os
import subprocess  # nosec B404 - test-controlled child process
import sys

import pytest

from arcadedb._native import lib as native_lib

CHILD = """
import json
from arcadedb._native import lib
print(json.dumps(lib.diag()))
"""


def test_highway_is_picked_when_its_library_is_next_to_the_engine(tmp_path):
    (tmp_path / "libjvector-x86_64.so").write_bytes(b"")
    env = {}
    native_lib._default_vector_provider(tmp_path, env)
    assert env == {"ARCADEDB_VECTOR_PROVIDER": "native"}


@pytest.mark.parametrize("value", ["default", "panama", "native"])
def test_a_value_the_user_set_wins(tmp_path, value):
    (tmp_path / "libjvector-x86_64.so").write_bytes(b"")
    env = {"ARCADEDB_VECTOR_PROVIDER": value}
    native_lib._default_vector_provider(tmp_path, env)
    assert env == {"ARCADEDB_VECTOR_PROVIDER": value}


def test_without_the_library_the_scalar_provider_stays(tmp_path):
    env = {}
    native_lib._default_vector_provider(tmp_path, env)
    assert env == {}


def _provider(tmp_path, value=None) -> str:
    """The VectorizationProvider class an isolate of a fresh process runs, with ARCADEDB_VECTOR_PROVIDER unset or set to ``value``.

    ``diag()`` is the child's first call, before any database: jvector picks its provider once per process, so this also checks that
    ``diag()`` leaves the choice to the engine (it picked Panama on macOS and Windows when it asked jvector first).
    """
    env = {k: v for k, v in os.environ.items() if k != "ARCADEDB_VECTOR_PROVIDER"}
    env["ARCADEDB_LOG_DIR"] = str(tmp_path / "log")
    if value is not None:
        env["ARCADEDB_VECTOR_PROVIDER"] = value
    result = subprocess.run(  # nosec B603 - fixed argument list, no shell
        [sys.executable, "-c", CHILD],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])["provider"]


def test_the_default_provider_follows_the_shipped_library(tmp_path):
    shipped = any(native_lib.library_path().resolve().parent.glob("libjvector-*.so"))
    expected = (
        "NativeVectorizationProvider" if shipped else "DefaultVectorizationProvider"
    )
    assert _provider(tmp_path).endswith(expected)


def test_default_forces_the_scalar_provider(tmp_path):
    assert _provider(tmp_path, "default").endswith("DefaultVectorizationProvider")
