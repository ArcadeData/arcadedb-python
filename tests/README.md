# ArcadeDB Python Bindings - Tests

Test suite for the ArcadeDB Python bindings. It runs against the built wheel, so it covers the embedded engine (SQL, OpenCypher, vectors, graphs) and the optional in-process HTTP server with Studio.

The [Testing Guide](https://docs.humem.ai/arcadedb/latest/development/testing/) lists every test file and what it checks, and describes the markers, the zero-skips gate, and how to write a test for this suite.

## Running Tests

```bash
# Once
uv venv

# After each build: the wheel and its test extra into .venv. --reinstall-package,
# as a rebuild keeps the version (26.11.1.dev0)
uv pip install --reinstall-package arcadedb "$(ls dist/*.whl | head -1)[test]"

# Run all tests
uv run pytest

# Run specific file
uv run pytest tests/test_core.py -v

# Run matching keyword
uv run pytest -k "transaction" -v

# Only the server tests, or everything except them (they start a real HTTP listener)
uv run pytest -m server -v
uv run pytest -m "not server"
```

## No skips

The server tests do not skip when the server stack is absent: the wheel always ships it, and a test that skips on a missing feature cannot notice the feature going missing (in 26.7.2 the server JARs were dropped from the wheel and the guarded tests skipped, so the suite stayed green while the feature was gone). `test_server_packaging.py` states the packaging requirement and fails without the JARs. CI also fails any skip that `scripts/check_test_skips.py` does not list, and that list is empty.

## The native build

The same suite runs against the native `arcadedb` wheel (`native/wheel/parity_wheel.py`). A test is reported as a skip on the native build in two explicit ways, never by what its failure says:

- `NotOnNativeYet`: the native code raised it because the feature needs a Java object API the library does not expose. The report says `NOTYET`, and a final release allows none.
- A row of `NATIVE_NA_JVM_TESTS` in `tests/conftest.py`, keyed `<file>::[<Class>::]<test>`: the test reaches into the JVM on purpose (`jpype.JClass`, a Java array, `db._java_db`, counted bridge crossings). The report says `NA-JVM`. Add a row only for such a test, and give the reason. When the behaviour is shared, write a native test beside it (`test_native_*.py`, collected on the native build only).

A test that fails on the native build for any other reason is a failure. Fix the native code, or mark a known engine bug with a strict `xfail`. Do not add a row to hide it.

## Need Help?

- **Found a bug?** [Open an issue](https://github.com/ArcadeData/arcadedb-python/issues)
- **Contributing?** Read the [Contributing Guide](https://docs.humem.ai/arcadedb/latest/development/contributing/)
