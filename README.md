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

## Embedded bindings or drivers?

This repository is the embedded bindings: the engine runs inside your Python process. The official [drivers](https://github.com/ArcadeData/arcadedb-drivers) are clients of a running ArcadeDB server. Do not mix them up.

| | Embedded bindings (this repository) | Drivers |
|---|---|---|
| What it is | The ArcadeDB engine, inside your Python process | Python clients of an ArcadeDB server, HTTP (sync and async) or gRPC. They contain no engine |
| Needs a server? | No. An in-process HTTP server and Studio are optional | Yes: the Docker image `arcadedata/arcadedb` or any ArcadeDB server. The gRPC driver also needs the server's gRPC plugin |
| Package and import | `arcadedb` or `arcadedb-jvm`, `import arcadedb` | [`arcadedb-driver`](https://pypi.org/project/arcadedb-driver/), `import arcadedb_driver` (HTTP); [`arcadedb-driver-grpc`](https://pypi.org/project/arcadedb-driver-grpc/), `import arcadedb_driver_grpc` (gRPC) |
| Use it when | One process, no server: tests, notebooks, scripts, an agent that owns its data | Several clients, an existing server, a service boundary, HA or replication, or clients that are not Python |

You can also use both, for example the bindings for a job and the HTTP driver for a shared server. Driver docs: [HTTP](https://docs.arcadedb.com/arcadedb/how-to/connectivity/drivers/python-http), [gRPC](https://docs.arcadedb.com/arcadedb/how-to/connectivity/drivers/python-grpc). More in [Which package?](https://github.com/ArcadeData/arcadedb-python/blob/main/docs/modules/ROOT/pages/which-package.adoc#embedded-or-driver).

## Install

```bash
uv add arcadedb       # native, the default
uv add arcadedb-jvm   # JVM, for heavy long-running loads
```

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
> The first release of this layout is 26.11.1, and both packages have a final release on PyPI. A plain `uv add arcadedb` installs the final, and a later pre-release (`A.B.C.devN`) needs `uv add --prerelease allow arcadedb`. The same holds for `arcadedb-jvm`.

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
> Old names: `arcadedb-jvm` was formerly `arcadedb-embedded`, and `arcadedb` was formerly `arcadedb-embedded-native` (archived on PyPI). The last `arcadedb-embedded` release, 26.11.1, is a deprecation shim: it installs `arcadedb`, the native build, and warns once at import. It does not install `arcadedb-jvm`.

- Python: `arcadedb` (native) is tested on 3.11 to 3.14 and also installs on 3.15 (`requires-python` is `>=3.11` with no upper bound; the CI runs a subset of the tests on 3.15). `arcadedb-jvm` needs 3.11 to 3.14 (`requires-python` is `>=3.11,<3.15`; its dependency JPype has no 3.15 wheel yet). Each release builds and tests both packages on Python 3.11 to 3.14 on all four platforms below. Between releases, `main` and the nightly run test `arcadedb` on 3.12 and 3.14 on every platform (with a short subset on 3.11, 3.13 and 3.15), and `arcadedb-jvm` on 3.11 to 3.14 on Linux x86-64 and on 3.11 and 3.14 on the other three. A pull request runs a smaller selection.
- Linux x86_64 and ARM64, macOS on Apple Silicon, Windows x86_64, for both packages. The Windows wheel of `arcadedb` is the newest.
- Import name: `arcadedb`. Check which package you have with `python -c "import arcadedb; print(arcadedb.__backend__)"`. It prints `native` or `jvm`.
- Optional extras: `arrow`, `pandas`, `polars`, `vector` for `arcadedb-jvm`; `numpy`, `arrow`, `pandas`, `polars` for `arcadedb`.
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

## Packages on PyPI

| Package | What it is | Status |
|---|---|---|
| [`arcadedb`](https://pypi.org/project/arcadedb/) | The native build: the engine compiled by GraalVM Community Edition, no JVM, no JPype. The default for embedded use | First release of this layout: 26.11.1 |
| [`arcadedb-jvm`](https://pypi.org/project/arcadedb-jvm/) | The JVM build: the Java engine on a bundled JVM, through JPype. The package this README described before the two-package layout | First release from this repository: 26.11.1 |

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
Formerly `arcadedb-embedded` (now `arcadedb-jvm`, the same build) and formerly `arcadedb-embedded-native` (now `arcadedb`). They imported as `arcadedb_embedded` and `arcadedb_native`, and they are deprecated: `arcadedb-embedded-native` is archived, and the last `arcadedb-embedded` release, 26.11.1, is a shim that installs `arcadedb` and warns once at import.

Releases from this repository start with 26.11.1. The 26.10.x wheels were built and published from [humemai/arcadedb-embedded-python](https://github.com/humemai/arcadedb-embedded-python). <!-- allow-old-name -->

## Documentation

- Here: [`docs/`](https://github.com/ArcadeData/arcadedb-python/tree/main/docs/modules/ROOT/pages) is an Antora component (AsciiDoc) with Installation, [Which package?](https://github.com/ArcadeData/arcadedb-python/blob/main/docs/modules/ROOT/pages/which-package.adoc), Quickstart, Databases and queries, Vectors, and Concurrency, processes and crashes. It covers only what is specific to the Python bindings.
- Published at [docs.arcadedb.com/arcadedb-python](https://docs.arcadedb.com/arcadedb-python/), the same site as the main ArcadeDB manual, rebuilt daily from `main` (ArcadeData/arcadedb-docs#505). The `Docs` workflow builds it as a site of its own on every change under `docs/` and attaches it as the `docs-site` artifact. To build it locally, run `npx -p @antora/cli@3.2.1 -p @antora/site-generator@3.2.1 antora docs/antora-playbook.yml` and open `docs/build/site/index.html`.
- ArcadeDB itself (SQL, openCypher, schema, server): [docs.arcadedb.com](https://docs.arcadedb.com). It is not repeated here.

## Examples

[`examples/`](https://github.com/ArcadeData/arcadedb-python/tree/main/examples) holds 29 runnable scripts, from a first document store to vector search, time series, graph algorithms and server mode. CI runs them ([`test-python-examples.yml`](https://github.com/ArcadeData/arcadedb-python/blob/main/.github/workflows/test-python-examples.yml)).

## Build and test from source

```bash
./scripts/build.sh   # an arcadedb-jvm wheel for your platform in dist/
```

The `arcadedb` (native) wheel is built from `native/` with GraalVM Community Edition; [`CLAUDE.md`](https://github.com/ArcadeData/arcadedb-python/blob/main/CLAUDE.md) has the commands. A native image build takes about 5 to 30 minutes per platform (macOS is the slowest).

The JVM build runs in Docker on Linux. macOS and Windows need a JDK 25 or later. The jars come from the `arcadedata/arcadedb` Docker image: `latest` by default, or the tag in `ARCADEDB_TAG` (for example `ARCADEDB_TAG=26.10.2 ./scripts/build.sh`). The package version is read from the engine jar in that image, so an `A.B.C-SNAPSHOT` image builds `A.B.C.devN`.

The native build does not use the image. It resolves `arcadedb-engine` and `arcadedb-server` from Maven Central at `arcadedb.version` in `native/pom.xml` (release versions only, because a snapshot is not on Maven Central) and compiles them with GraalVM Community Edition. Neither build compiles the engine from source: the engine code is upstream's, unmodified.

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
