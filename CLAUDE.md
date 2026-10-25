# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`arcadedb`: Python bindings for ArcadeDB that run the Java engine in-process through JPype. Each wheel is platform-specific and bundles the ArcadeDB jars plus a jlink'd Java 25 JRE, so users need no Java install. Moved here from `bindings/python` in ArcadeData/arcadedb (history kept; issue numbers in older commit messages refer to that repo).

## Working rules

- Never use the em dash character in responses; use `-`, a comma, or rephrase.
- Before coding, state how the change will be verified; write the test first (a regression test for every fix), then run every test the change could affect.
- Don't add Claude as author of source code. Never write the `@claude` trigger phrase in a PR comment.
- New dependencies (Python extras or anything bundled in the wheel) only when strictly necessary, and only Apache-2.0 compatible: Apache 2.0, MIT, BSD, ISC, EPL, UPL, CC0; LGPL 2.1+/MPL-2.0/CDDL as unmodified libraries only. Never GPL (except GPL-2.0 with Classpath Exception), AGPL, SSPL, BUSL, Elastic, Commons Clause.
- Java bridge code (`src/java`) follows upstream ArcadeDB style: performance and low GC pressure first (primitive arrays over lists of objects), import classes instead of fully qualified names, `final` where possible, no braces around single-statement `if` bodies, `com.arcadedb.serializer.json.JSONObject`/`JSONArray` for JSON. Never submit work to `ForkJoinPool.commonPool()`.

## Build

```bash
./scripts/build.sh                                  # current platform, auto-detected
./scripts/build.sh linux/amd64 3.12                 # Linux builds run in Docker (manylinux)
./scripts/build.sh linux/amd64 3.12 /path/to/lib    # embed local ArcadeDB jars instead of the image's
ARCADEDB_TAG=26.11.1-SNAPSHOT ./scripts/build.sh    # pin the arcadedata/arcadedb image (default: latest)
```

- Linux builds use `scripts/Dockerfile.build`; macOS/Windows use `scripts/build-native.sh` with a native JDK 25+ (jlink).
- Jars come from the `arcadedata/arcadedb` Docker image, filtered by `scripts/jar_exclusions.txt`, staged by `scripts/stage_jars.sh` (always re-pulled; stale jars from a previous build are removed).
- `JAR_LIB_DIR` (third argument) may be a symlink, and so may any jar in it: `build.sh` follows both and stages real files into `local-jars/`.
- The package version is derived from the `arcadedb-engine-<version>.jar` (`scripts/extract_version.py`): `26.11.1-SNAPSHOT` → `26.11.1.devN`. The `version = "0.0.0"` in `pyproject.toml` is a placeholder overwritten at build time.
- `src/java/com/arcadedb/python/*.java` is compiled during the build into `arcadedb-python-bridge.jar` (both in `Dockerfile.build` and `build-native.sh`). Changing Java code requires a rebuild.
- Output: `dist/*.whl`. `setup.py` only exists to force a platform-specific wheel tag.

## Test

Tests run against the **built wheel**, not the source tree. `pyproject.toml` has `[tool.uv] managed = false`, so it is not a uv project.

```bash
uv venv                                                                          # once
uv pip install --reinstall-package arcadedb "$(ls dist/*.whl | head -1)[test]"   # after each build
uv run pytest                                    # all
uv run pytest tests/test_core.py -v              # one file
uv run pytest tests/test_core.py::TestX::test_y  # one test
uv run pytest -k transaction
uv run pytest -m server / -m "not server"        # markers: server, server_wire, integration
```

- `--reinstall-package` is needed because rebuilds keep the same version.
- **No skips policy**: CI fails on any skipped test not listed in `scripts/check_test_skips.py` (the list is empty). Don't add `pytest.skip`/`importorskip` for missing features or deps; tests that can't run on a platform go in `collect_ignore` in `tests/conftest.py`; known engine bugs use strict `xfail`.
- Never bind or address a hand-picked port (e.g. 2480) in a test: use `server.get_http_port()` or the `_free_ports()` pattern in `test_server_wire_protocols.py`. Parallel runs share the machine and collisions surface as unrelated failures.
- Never assert on raw wall-clock elapsed time; a JVM pause makes it flaky. Timeouts are hang detectors, not latency bounds.
- `test_docs_examples.py` is ignored unless the user docs site (`docs/index.md`) is present; it isn't in this repo.

## Lint / format

Pre-commit: black, isort (`--profile black`), shfmt (`-i 4 -ci -sr`), pretty-format-yaml (not on `.github/`), actionlint. CI also runs Bandit, which must be clean:

```bash
python -m bandit -c pyproject.toml -r src tests --severity-level low --confidence-level low
python -m bandit -c pyproject.toml -r examples --severity-level medium --confidence-level high
```

## Architecture

- `jvm.py`: locates jars/JRE and starts the single JVM (`start_jvm`). Installed wheels ship `jars/` and `jre/` inside the package; from a source checkout they are extracted from the newest wheel in `dist/` (by mtime) into `.runtime-cache/`, re-extracted when the wheel stamp changes. JVM args come from kwargs and `ARCADEDB_JVM_ARGS`; the JVM can only start once per process, so conflicting later configs raise.
- `core.py`: `Database` / `DatabaseFactory`, `create_database`/`open_database`, query/command, bulk inserts. Hot paths call the Java bridge classes (`DbCalls`, `DocumentBatcher`, `RowBatcher`, `ColumnBatcher`, ...) via `jpype.JClass("com.arcadedb.python.…")` to avoid per-call JPype overhead; Python falls back when the bridge is absent.
- `results.py`: `ResultSet`/`Result`, including columnar readers (`to_columns`, `to_arrow` → returns None without pyarrow, `to_dataframe` → raises ImportError without pandas; this difference is intentional).
- `type_conversion.py`: Python ↔ Java value conversion. `transactions.py`, `schema.py`, `graph.py`/`graph_batch.py`, `vector.py` (JVector HNSW), `importer.py`/`exporter.py`, `async_executor.py` wrap the corresponding engine APIs.
- `server.py`: optional in-process HTTP server + Studio (`create_server()`); the server jars always ship in the wheel and tests must fail, not skip, if they're missing (`test_server_packaging.py`).

## CI

`.github/workflows/test-python-bindings.yml` builds wheels for linux/amd64, linux/arm64, darwin/arm64, windows/amd64 and tests Python 3.10–3.14; also runs nightly against `arcadedata/arcadedb:latest`. Both workflows accept an `arcadedb-tag` input to pin the image. `test-python-examples.yml` runs `examples/`. A dependency-floor audit resolves extras with `--resolution lowest-direct` for each Python classifier and runs pip-audit, so raise lower bounds when a floor has an advisory.

`.github/workflows/release.yml` publishes on a bare-version tag (`26.11.1.dev0`, `26.11.1`, `26.11.1.post1`; no `v`, never `aN`/`bN`/`rcN`). Its first job runs `scripts/check_version.py` (the devN/postN counting rule, read-only, fails closed); `scripts/after_tag_verify.sh` checks the finished release. The rule is stated once, on the docs page `development/release.adoc`.
