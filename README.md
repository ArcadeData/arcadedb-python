# ArcadeDB Python

[![PyPI arcadedb](https://img.shields.io/pypi/v/arcadedb?label=arcadedb)](https://pypi.org/project/arcadedb/)
[![PyPI arcadedb-jvm](https://img.shields.io/pypi/v/arcadedb-jvm?label=arcadedb-jvm)](https://pypi.org/project/arcadedb-jvm/)
[![Python versions](https://img.shields.io/pypi/pyversions/arcadedb-jvm)](https://pypi.org/project/arcadedb-jvm/)
[![Tests JVM](https://github.com/ArcadeData/arcadedb-python/actions/workflows/test-jvm.yml/badge.svg)](https://github.com/ArcadeData/arcadedb-python/actions/workflows/test-jvm.yml)
[![Tests native](https://github.com/ArcadeData/arcadedb-python/actions/workflows/test-native.yml/badge.svg)](https://github.com/ArcadeData/arcadedb-python/actions/workflows/test-native.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/ArcadeData/arcadedb-python/blob/main/LICENSE)

Python bindings for [ArcadeDB](https://arcadedb.com), the multi-model database (graph, document, key/value, vector and time series, queried with SQL or openCypher). The engine runs inside your Python process, with no server and no network hop. This repository builds two packages, and both import as `arcadedb`: `arcadedb` (the native build, the default) for scripts, notebooks, tests, CLIs, serverless functions and agents, and `arcadedb-jvm` (the JVM build) for heavy, long-running loads. Status: beta.

| Package | Engine | Pick it for |
|---|---|---|
| [`arcadedb`](https://pypi.org/project/arcadedb/) (native, the default) | The engine compiled by GraalVM Community Edition into a shared library, loaded with `ctypes`. No JVM, no JPype | Scripts, notebooks, tests, CLIs, serverless functions, agents |
| [`arcadedb-jvm`](https://pypi.org/project/arcadedb-jvm/) (JVM) | The Java engine in your process through [JPype](https://jpype.readthedocs.io/), with a bundled Java 25 runtime | Heavy, long-running loads, where the JIT wins over time. Also the Java-object APIs and JVM options |

Both packages `import arcadedb`, and they have the same public API. Install one, not both. Not sure which? Read [Which package?](https://github.com/ArcadeData/arcadedb-python/blob/main/docs/modules/ROOT/pages/which-package.adoc).

## Install

```bash
pip install arcadedb       # native, the default (or: uv add arcadedb)
pip install arcadedb-jvm   # JVM, for heavy long-running loads (or: uv add arcadedb-jvm)
```

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
> Transition note: the first release of this layout is 26.11.1. Until `arcadedb` passes its readiness bar, it may appear only as pre-releases, `26.11.1.devN`. Until a final 26.11.1 exists, PyPI holds only these pre-releases, and a plain `pip install arcadedb` or `uv add arcadedb` installs the newest one. After a final release, a plain install takes the final, and a later pre-release needs `pip install --pre arcadedb` (or `uv add --prerelease allow arcadedb`). The same holds for `arcadedb-jvm`.

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
> Old names: `arcadedb-jvm` was formerly `arcadedb-embedded`, and `arcadedb` was formerly `arcadedb-embedded-native` (archived on PyPI). A last `arcadedb-embedded` release will be a deprecation shim that installs `arcadedb`, the native build.

- Python 3.11 to 3.14. CI builds and tests all four versions on every platform below for `main`, every night and each release. A pull request runs a smaller selection.
- Linux x86_64 and ARM64, macOS on Apple Silicon, Windows x86_64, for both packages. The Windows wheel of `arcadedb` is the newest.
- Import name: `arcadedb`. Check which package you have with `python -c "import arcadedb; print(arcadedb.__backend__)"`. It prints `native` or `jvm`.
- Optional extras: `arrow`, `pandas`, `vector` for `arcadedb-jvm`; `numpy`, `arrow`, `pandas`, `polars` for `arcadedb`.
- `arcadedb-jvm` bundles its own Java 25 runtime, so you install no Java. `arcadedb` needs no Java at all.

## Quick start

```python
import arcadedb

with arcadedb.create_database("./mydb") as db:
    db.command("sql", "CREATE DOCUMENT TYPE Person")
    db.command("sql", "CREATE PROPERTY Person.name STRING")
    db.command("sql", "CREATE PROPERTY Person.age INTEGER")

    # Writes need a transaction
    with db.transaction():
        db.command("sql", "INSERT INTO Person SET name = ?, age = ?", "Alice", 30)

    for record in db.query("sql", "SELECT FROM Person WHERE age > 25"):
        print(f"Name: {record.get('name')}")
```

This prints `Name: Alice` (the engine also logs to standard error). `create_database` raises if the path already holds a database, so run it in an empty directory. Everything else goes through the same two calls, `db.command(...)` and `db.query(...)`, in SQL or openCypher.

## What is included

- Embedded mode: the database is a directory, opened by one process at a time.
- Graph, document, key/value, vector (JVector HNSW) and time series models; ACID transactions.
- Import from CSV, XML and ArcadeDB JSONL; export to JSONL.
- Columnar results with `to_columns()`, `to_arrow()` and `to_dataframe()`.
- An optional in-process HTTP server (`create_server()`), off until you call it, for other processes that need the same data.

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
> New in 26.11.1: the `arcadedb-jvm` wheels carry the Studio web UI, which the server then serves. Studio used to bundle ApexCharts, which is no longer open source ([ArcadeData/arcadedb#10579](https://github.com/ArcadeData/arcadedb/issues/10579)), and [ArcadeData/arcadedb#10627](https://github.com/ArcadeData/arcadedb/pull/10627) replaced it with Apache ECharts. The earlier 26.10.x wheels on PyPI have no Studio.

For several processes, other languages, HA or replication, run the official Docker image `arcadedata/arcadedb` ([ArcadeDB server](https://docs.arcadedb.com)) and connect with [`arcadedb-driver`](https://pypi.org/project/arcadedb-driver/) (HTTP client, developed in [ArcadeData/arcadedb-drivers](https://github.com/ArcadeData/arcadedb-drivers)).

## Packages on PyPI

| Package | What it is | Status |
|---|---|---|
| [`arcadedb`](https://pypi.org/project/arcadedb/) | The native build: the engine compiled by GraalVM Community Edition, no JVM, no JPype. The default for embedded use | First release of this layout: 26.11.1. It may first appear as `26.11.1.devN` pre-releases |
| [`arcadedb-jvm`](https://pypi.org/project/arcadedb-jvm/) | The JVM build: the Java engine on a bundled JVM, through JPype. The package this README described before the two-package layout | First release from this repository: 26.11.1 |

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
Formerly `arcadedb-embedded` (now `arcadedb-jvm`, the same build) and formerly `arcadedb-embedded-native` (now `arcadedb`). They imported as `arcadedb_embedded` and `arcadedb_native`, and they are deprecated: `arcadedb-embedded-native` is archived, and `arcadedb-embedded` will get a last release that installs `arcadedb`.

Releases from this repository start with 26.11.1. The 26.10.x wheels were built and published from [humemai/arcadedb-embedded-python](https://github.com/humemai/arcadedb-embedded-python). <!-- allow-old-name -->

## Documentation

- Here: [`docs/`](https://github.com/ArcadeData/arcadedb-python/tree/main/docs/modules/ROOT/pages) is an Antora component (AsciiDoc) with Installation, [Which package?](https://github.com/ArcadeData/arcadedb-python/blob/main/docs/modules/ROOT/pages/which-package.adoc), Quickstart, Databases and queries, and Concurrency, processes and crashes. It covers only what is specific to the Python bindings.
- The component is not published on docs.arcadedb.com yet; adding it as a content source is open in [ArcadeData/arcadedb-docs#505](https://github.com/ArcadeData/arcadedb-docs/pull/505). Until then, the `Docs` workflow builds it as a site of its own on every change under `docs/` and attaches it as the `docs-site` artifact. To build it locally, run `npx -p @antora/cli@3.2.1 -p @antora/site-generator@3.2.1 antora docs/antora-playbook.yml` and open `docs/build/site/index.html`.
- The older guide, API reference and example walkthroughs are at [docs.humem.ai/arcadedb](https://docs.humem.ai/arcadedb/latest/). They describe the previous packages and use the import name `arcadedb_embedded`; read it as `arcadedb`. They stay until those pages are ported here ([#19](https://github.com/ArcadeData/arcadedb-python/issues/19)). <!-- allow-old-name -->
- ArcadeDB itself (SQL, openCypher, schema, server): [docs.arcadedb.com](https://docs.arcadedb.com). It is not repeated here.

## Examples

[`examples/`](https://github.com/ArcadeData/arcadedb-python/tree/main/examples) holds 27 runnable scripts, from a first document store to vector search, time series, graph algorithms and server mode. CI runs them ([`test-python-examples.yml`](https://github.com/ArcadeData/arcadedb-python/blob/main/.github/workflows/test-python-examples.yml)).

## Build and test from source

```bash
./scripts/build.sh   # an arcadedb-jvm wheel for your platform in dist/
```

The `arcadedb` (native) wheel is built from `native/` with GraalVM Community Edition; [`CLAUDE.md`](https://github.com/ArcadeData/arcadedb-python/blob/main/CLAUDE.md) has the commands. A native image build takes about 1 to 2.5 hours per platform.

The JVM build runs in Docker on Linux. macOS and Windows need a JDK 25 or later. The jars come from the `arcadedata/arcadedb` Docker image: `latest` by default, or the tag in `ARCADEDB_TAG` (for example `ARCADEDB_TAG=26.10.2 ./scripts/build.sh`). The package version is read from the engine jar in that image, so a `26.11.1-SNAPSHOT` image builds `26.11.1.devN`.

Tests run against the built wheel, not the source tree: install the wheel with its `test` extra into a virtual environment, then run `pytest`. [`CLAUDE.md`](https://github.com/ArcadeData/arcadedb-python/blob/main/CLAUDE.md) has the exact commands, the build options and the lint setup.

## Contributing

Open an [issue](https://github.com/ArcadeData/arcadedb-python/issues) or a pull request. Run `pre-commit run --all-files` first, and add a regression test with every fix. The rest of the bindings work from `humemai/arcadedb-embedded-python` is ported here one pull request at a time; [#19](https://github.com/ArcadeData/arcadedb-python/issues/19) lists what is done and what is next. <!-- allow-old-name -->

## Where this code comes from

It moved here from `bindings/python` in [ArcadeData/arcadedb](https://github.com/ArcadeData/arcadedb) at `ee04bd202abce0183c6263a3c4d8a61a4d526960`, with its git history. Issue numbers in older commit messages refer to that repository. [humemai/arcadedb-embedded-python](https://github.com/humemai/arcadedb-embedded-python) is the fork where the bindings kept developing (runtime fixes, JRE trim, the native build, benchmarks). Its benchmark tooling stays there. <!-- allow-old-name -->

## Citation

If you use these bindings in published work, cite them with the metadata in [`CITATION.cff`](https://github.com/ArcadeData/arcadedb-python/blob/main/CITATION.cff); GitHub shows it as "Cite this repository".
There is no DOI for this repository yet.

## License

Apache License 2.0, like ArcadeDB. See [`LICENSE`](https://github.com/ArcadeData/arcadedb-python/blob/main/LICENSE).
