# ArcadeDB Python

[![PyPI](https://img.shields.io/pypi/v/arcadedb-embedded)](https://pypi.org/project/arcadedb-embedded/)
[![Python versions](https://img.shields.io/pypi/pyversions/arcadedb-embedded)](https://pypi.org/project/arcadedb-embedded/)
[![Tests](https://github.com/ArcadeData/arcadedb-python/actions/workflows/test-python-bindings.yml/badge.svg)](https://github.com/ArcadeData/arcadedb-python/actions/workflows/test-python-bindings.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/ArcadeData/arcadedb-python/blob/main/LICENSE)

Python bindings for [ArcadeDB](https://arcadedb.com), the multi-model database (graph, document, key/value, vector and time series, queried with SQL or openCypher). The engine runs inside your Python process through [JPype](https://jpype.readthedocs.io/), with no server and no network hop. The wheel bundles its own Java 25 runtime, so you install no Java. Status: beta.

## Install

```bash
pip install arcadedb-embedded   # or: uv add arcadedb-embedded
```

- Python 3.10 to 3.14. CI builds and tests all five versions on every platform below.
- Linux x86_64 and ARM64, macOS on Apple Silicon, Windows x86_64. One wheel per platform and Python version.
- Import name: `arcadedb_embedded`. Optional extras: `arrow`, `pandas`, `vector`.

## Quick start

```python
import arcadedb_embedded as arcadedb

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
- An optional in-process HTTP server (`create_server()`), off until you call it, for other processes that need the same data. The Studio web UI is not in the wheels on PyPI because it bundles ApexCharts, which is no longer open source ([ArcadeData/arcadedb#10579](https://github.com/ArcadeData/arcadedb/issues/10579)); [#20](https://github.com/ArcadeData/arcadedb-python/pull/20) makes wheels built from this repository match.

For several processes, other languages, HA or replication, run the regular [ArcadeDB server](https://docs.arcadedb.com) and connect with [`arcadedb-driver`](https://pypi.org/project/arcadedb-driver/) (HTTP client, developed in [ArcadeData/arcadedb-drivers](https://github.com/ArcadeData/arcadedb-drivers)).

## Packages on PyPI

| Package | What it is | Status |
|---|---|---|
| [`arcadedb-embedded`](https://pypi.org/project/arcadedb-embedded/) | These bindings: the engine on a bundled JVM. The package this README describes | Release 26.10.2.post1 |
| [`arcadedb-embedded-native`](https://pypi.org/project/arcadedb-embedded-native/) | The same API with the engine compiled ahead of time by GraalVM Community Edition: no JVM, no JPype. Import `arcadedb_native`, Python 3.11 or later | Preview of alpha quality: `pip install --pre arcadedb-embedded-native`, currently 26.10.2.dev1 |
| [`arcadedb`](https://pypi.org/project/arcadedb/) and [`arcadedb-native`](https://pypi.org/project/arcadedb-native/) | Names reserved for the project. Today `arcadedb` 0.0.1 only installs `arcadedb-embedded`, and `arcadedb-native` 26.10.2.dev1 only installs `arcadedb-embedded-native` | Placeholders |

Two things are not true yet. The native build is not in this repository (porting is tracked in [#19](https://github.com/ArcadeData/arcadedb-python/issues/19)), and releases do not come from this repository: the wheels on PyPI are still built and published from [humemai/arcadedb-embedded-python](https://github.com/humemai/arcadedb-embedded-python). A release workflow for this repository is in review in [#22](https://github.com/ArcadeData/arcadedb-python/pull/22).

## Documentation

- Here: [`docs/`](https://github.com/ArcadeData/arcadedb-python/tree/main/docs/modules/ROOT/pages) is an Antora component (AsciiDoc) with Installation, Quickstart and Databases and queries. It covers only what is specific to the Python bindings.
- The component is not published on docs.arcadedb.com yet; adding it as a content source is open in [ArcadeData/arcadedb-docs#505](https://github.com/ArcadeData/arcadedb-docs/pull/505).
- The guide, API reference and example walkthroughs for the published package are still at [docs.humem.ai/arcadedb](https://docs.humem.ai/arcadedb/latest/) until those pages are ported here ([#19](https://github.com/ArcadeData/arcadedb-python/issues/19)).
- ArcadeDB itself (SQL, openCypher, schema, server): [docs.arcadedb.com](https://docs.arcadedb.com). It is not repeated here.

## Examples

[`examples/`](https://github.com/ArcadeData/arcadedb-python/tree/main/examples) holds 26 runnable scripts, from a first document store to vector search, time series, graph algorithms and server mode. CI runs them ([`test-python-examples.yml`](https://github.com/ArcadeData/arcadedb-python/blob/main/.github/workflows/test-python-examples.yml)).

## Build and test from source

```bash
./scripts/build.sh   # a wheel for your platform in dist/
```

Linux builds run in Docker. macOS and Windows need a JDK 25 or later. The jars come from the `arcadedata/arcadedb` Docker image: `latest` by default, or the tag in `ARCADEDB_TAG` (for example `ARCADEDB_TAG=26.10.2 ./scripts/build.sh`). The package version is read from the engine jar in that image, so a `26.11.1-SNAPSHOT` image builds `26.11.1.devN`.

Tests run against the built wheel, not the source tree: install the wheel with its `test` extra into a virtual environment, then run `pytest`. [`CLAUDE.md`](https://github.com/ArcadeData/arcadedb-python/blob/main/CLAUDE.md) has the exact commands, the build options and the lint setup.

## Contributing

Open an [issue](https://github.com/ArcadeData/arcadedb-python/issues) or a pull request. Run `pre-commit run --all-files` first, and add a regression test with every fix. The rest of the bindings work from `humemai/arcadedb-embedded-python` is ported here one pull request at a time; [#19](https://github.com/ArcadeData/arcadedb-python/issues/19) lists what is done and what is next.

## Where this code comes from

It moved here from `bindings/python` in [ArcadeData/arcadedb](https://github.com/ArcadeData/arcadedb) at `ee04bd202abce0183c6263a3c4d8a61a4d526960`, with its git history. Issue numbers in older commit messages refer to that repository. [humemai/arcadedb-embedded-python](https://github.com/humemai/arcadedb-embedded-python) is the fork where the bindings kept developing (runtime fixes, JRE trim, native build, benchmarks). Its benchmark tooling stays there.

## License

Apache License 2.0, like ArcadeDB. See [`LICENSE`](https://github.com/ArcadeData/arcadedb-python/blob/main/LICENSE).
