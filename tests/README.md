# ArcadeDB Python Bindings - Tests

Test suite for the ArcadeDB Python bindings. It runs against the built wheel, so it covers the embedded engine (SQL, OpenCypher, vectors, graphs) and the optional in-process HTTP server with Studio.

[Continuous integration](https://docs.arcadedb.com/arcadedb-python/development/ci) describes which test legs run for a pull request, for `main`, the nightly run and a release. How to run the suite, by marker or by file, and the no-skips rule are below.

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

## Tests that send SIGINT to their own process

A pytest started as a background job (a shell `cmd &` without job control, many launchers and agent shells) inherits SIGINT as ignored, and Python then installs no `KeyboardInterrupt` handler. A test that sends SIGINT to its own process (`test_python_function_interrupt.py`) installs `signal.default_int_handler` for its duration and restores the old handler; without it the signal is dropped, no query is interrupted and the test fails on such a runner while it passes in CI. Check a runner with `grep SigIgn /proc/self/status` (bit `0x2` is SIGINT) or `python -c "import signal; print(signal.getsignal(signal.SIGINT))"` (`<Handlers.SIG_IGN: 1>` means ignored). A test that starts a child process for SIGINT resets the signal in the child (`test_sigint.py`, `preexec_fn`).

## No skips

The server tests do not skip when the server stack is absent: the wheel always ships it, and a test that skips on a missing feature cannot notice the feature going missing (in 26.7.2 the server JARs were dropped from the wheel and the guarded tests skipped, so the suite stayed green while the feature was gone). `test_server_packaging.py` states the packaging requirement and fails without the JARs. CI also fails any skip that `scripts/check_test_skips.py` does not list, and that list is empty.

## The native build

The same suite runs against the native `arcadedb` wheel (`native/wheel/parity_wheel.py`). A test is reported as a skip on the native build in two explicit ways, never by what its failure says:

- `NotOnNativeYet`: the native code raised it because the feature needs a Java object API the library does not expose. The report says `NOTYET`, and a final release allows none. A call that is JVM-only by design (`get_java_database`, `get_java_document`, a Java object a `Schema` method returns) raises it with a message that names the `arcadedb-jvm` build; a shared test that reaches one is `NA-JVM`, or reads the value through a statement or the Python API instead.
- A row of `NATIVE_NA_JVM_TESTS` in `tests/conftest.py`, keyed `<file>::[<Class>::]<test>`: the test reaches into the JVM on purpose (`jpype.JClass`, a Java array, `db._java_db`, counted bridge crossings). The report says `NA-JVM`. Add a row only for such a test, and give the reason. When the behaviour is shared, write a native test beside it (`test_native_*.py`, collected on the native build only).

A test that fails on the native build for any other reason is a failure. Fix the native code, or mark a known engine bug with a strict `xfail`. Do not add a row to hide it.

## Need Help?

- **Found a bug?** [Open an issue](https://github.com/ArcadeData/arcadedb-python/issues)
- **Contributing?** Read the development pages: [Continuous integration](https://docs.arcadedb.com/arcadedb-python/development/ci) and [Release](https://docs.arcadedb.com/arcadedb-python/development/release)
