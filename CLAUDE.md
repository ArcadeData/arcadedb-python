# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This repository builds two distributions that both import as `arcadedb`: `arcadedb` (the native build: the engine as a GraalVM native-image shared library loaded with ctypes) and `arcadedb-jvm` (the JVM build: the Java engine in-process through JPype). `arcadedb-jvm` is described first below. Each JVM wheel is platform-specific and bundles the ArcadeDB jars plus a jlink'd Java 25 JRE, so users need no Java install. Moved here from `bindings/python` in ArcadeData/arcadedb (history kept; issue numbers in older commit messages refer to that repo).

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
W="$(ls dist/*.whl | head -1)"
uv run --no-project --with "$W[test]" python -m pytest                                    # all
uv run --no-project --with "$W[test]" python -m pytest tests/test_core.py -v              # one file
uv run --no-project --with "$W[test]" python -m pytest tests/test_core.py::TestX::test_y  # one test
uv run --no-project --with "$W[test]" python -m pytest -k transaction
uv run --no-project --with "$W[test]" python -m pytest -m server                          # markers: server, server_wire, integration
```

- `--with` builds a throwaway environment from the wheel file, so there is no venv to keep and nothing to reinstall after a rebuild. If uv's cache still serves an older wheel of the same version (rebuilds keep the version), add `--reinstall-package arcadedb-jvm`.
- Use `python -m pytest`, not the `pytest` entry point: with `--with`, the entry point does not see the packages of the environment.
- **No skips policy**: CI fails on any skipped test not listed in `scripts/check_test_skips.py` (the list is empty). Don't add `pytest.skip`/`importorskip` for missing features or deps; tests that can't run on a platform go in `collect_ignore` in `tests/conftest.py`; known engine bugs use strict `xfail`.
- Never bind or address a hand-picked port (e.g. 2480) in a test: use `server.get_http_port()` or the `_free_ports()` pattern in `test_server_wire_protocols.py`. Parallel runs share the machine and collisions surface as unrelated failures.
- Never assert on raw wall-clock elapsed time; a JVM pause makes it flaky. Timeouts are hang detectors, not latency bounds.
- `test_docs_examples.py` is ignored unless the user docs site (`docs/index.md`) is present; it isn't in this repo.

## Lint / format

Pre-commit: black, isort (`--profile black`), shfmt (`-i 4 -ci -sr`), pretty-format-yaml (not on `.github/`), actionlint. CI also runs Bandit, which must be clean:

```bash
uvx bandit==1.9.4 -c pyproject.toml -r src tests --severity-level low --confidence-level low
uvx bandit==1.9.4 -c pyproject.toml -r examples --severity-level medium --confidence-level high
```

## Architecture

- `jvm.py`: locates jars/JRE and starts the single JVM (`start_jvm`). Installed wheels ship `jars/` and `jre/` inside the package; from a source checkout they are extracted from the newest wheel in `dist/` (by mtime) into `.runtime-cache/`, re-extracted when the wheel stamp changes. JVM args come from kwargs and `ARCADEDB_JVM_ARGS`; the JVM can only start once per process, so conflicting later configs raise.
- `core.py`: `Database` / `DatabaseFactory`, `create_database`/`open_database`, query/command, bulk inserts. Hot paths call the Java bridge classes (`DbCalls`, `DocumentBatcher`, `RowBatcher`, `ColumnBatcher`, ...) via `jpype.JClass("com.arcadedb.python.…")` to avoid per-call JPype overhead; Python falls back when the bridge is absent.
- `results.py`: `ResultSet`/`Result`, including columnar readers (`to_columns`, `to_arrow` → returns None without pyarrow, `to_dataframe` → raises ImportError without pandas; this difference is intentional).
- `type_conversion.py`: Python ↔ Java value conversion. `transactions.py`, `schema.py`, `graph.py`/`graph_batch.py`, `vector.py` (JVector HNSW), `importer.py`/`exporter.py`, `async_executor.py` wrap the corresponding engine APIs.
- `server.py`: optional in-process HTTP server + Studio (`create_server()`); the server jars always ship in the wheel and tests must fail, not skip, if they're missing (`test_server_packaging.py`).

## CI

Each package has one test and one release workflow: `test-jvm.yml` and `release-jvm.yml` for `arcadedb-jvm`, `test-native.yml` and `release-native.yml` for `arcadedb` (the release workflows call the test workflows with `workflow_call`; the PyPI trusted publishers are keyed on the release file names and the environments `pypi-jvm` and `pypi`, so never rename them without changing PyPI first). `test-python-examples.yml` runs `examples/`. `docs.yml` builds the Antora component in `docs/` as a site of its own (`docs/antora-playbook.yml`, Antora 3.2.1, a warning fails it) and attaches it as the `docs-site` artifact; it is the only check of a docs-only pull request. Link into the main manual with `{arcadedb-docs-url}/<page>`, never an `arcadedb::` xref (the component on its own cannot resolve it).

`.github/workflows/test-jvm.yml` builds ONE wheel per platform (linux/amd64, linux/arm64, darwin/arm64, windows/amd64), tagged `py3-none-<platform>` by `setup.py` (`bdist_wheel.get_tag`; `scripts/verify_wheel_tag.py` checks it), and tests that wheel on Python 3.11–3.14; also runs nightly against `arcadedata/arcadedb:latest`. `requires-python` is `>=3.11,<3.15` because the tag no longer keeps pip off 3.15 and JPype has no 3.15 wheel; drop the cap with it. The org shares a runner cap with upstream, so the unit tests run 10 legs on every event (linux/amd64 on all four Pythons, the other three platforms on 3.11 and 3.14, the shape of test-native.yml), and the examples run all four platforms on 3.14 on a pull request (4 legs, 28 checks in all) and 16 legs on push to main, the nightly run, dispatch and the release (40 checks). `scripts/select_matrix.py` holds both lists (`PULL_REQUEST_LEGS`, keyed `jvm` and `examples`), the first job of each workflow runs it, and `tests/test_select_matrix.py` pins the full matrix; docs page `development/ci.adoc`. Keep the job names `Test arcadedb (<platform>, Python <version>)` unchanged. Both workflows accept an `arcadedb-tag` input to pin the image. `test-python-examples.yml` runs `examples/`. A dependency-floor audit locks the dependencies and extras on a throwaway uv project with `--resolution lowest-direct` for each Python classifier and runs `uvx pip-audit` on the export, so raise lower bounds when a floor has an advisory. No tool is installed with pip anywhere in CI or the scripts: one-off tools run through `uvx tool==X`, the wheel is built with `uvx --from "build[uv]==1.6.1" pyproject-build --installer uv`, and the uv install URL is pinned (`0.12.0`).

`.github/workflows/release-jvm.yml` publishes `arcadedb-jvm` on a bare-version tag (`26.11.1.dev0`, `26.11.1`, `26.11.1.post1`; no `v`, never `aN`/`bN`/`rcN`). Its first job runs `scripts/check_version.py` (the devN/postN counting rule, read-only, fails closed); `scripts/after_tag_verify.sh` checks the finished release. The rule is stated once, on the docs page `development/release.adoc`.

`test-native.yml` builds the GraalVM native image and runs the parity suite against the installed wheel (selective: scripts/select_native_matrix.py, 1 to 2.5 hours per platform, do not add runs lightly). `.github/workflows/release-native.yml` publishes `arcadedb` on the same tag: it calls `test-native.yml` with `release: true` (4 platforms x Python 3.11 to 3.14), then the readiness gate (the repository variable `ARCADEDB_NATIVE_READINESS_BAR_MET`, for a final or post version), then one publish job through the environment `pypi`.
