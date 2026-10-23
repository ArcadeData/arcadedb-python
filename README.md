# ArcadeDB Python Bindings

Native Python bindings for ArcadeDB - the multi-model database that supports Graph, Document, Key/Value, Search Engine, Time Series, and Vector models.

**Status**: ✅ Production Ready | **Platforms**: 4 Supported

---

## 📚 Documentation

**[📖 Read the Full Documentation →](https://docs.humem.ai/arcadedb/latest)**

---

## 🚀 Quick Start

### Installation

```bash
uv add arcadedb-embedded   # or: pip install arcadedb-embedded
```

**Requirements:**

- **Python 3.10 to 3.14** (CI runs all five on every supported platform). No Java installation required!
- **Supported Platforms**: Prebuilt wheels for **4 platforms**
  - Linux: x86_64, ARM64
  - macOS: Apple Silicon (ARM64)
  - Windows: x86_64

### 5-Minute Example

```python
import arcadedb_embedded as arcadedb

# Create database (context manager for automatic open and close)
with arcadedb.create_database("./mydb") as db:
    # Create schema (DDL)
    db.command("sql", "CREATE DOCUMENT TYPE Person")
    db.command("sql", "CREATE PROPERTY Person.name STRING")
    db.command("sql", "CREATE PROPERTY Person.age INTEGER")

    # Insert data (requires transaction)
    with db.transaction():
        db.command("sql", "INSERT INTO Person SET name = ?, age = ?", "Alice", 30)

    # Query data
    result = db.query("sql", "SELECT FROM Person WHERE age > 25")
    for record in result:
        print(f"Name: {record.get('name')}")
```

**[👉 See full tutorial](https://docs.humem.ai/arcadedb/latest/getting-started/quickstart/)**

---

## ✨ Features

- ☕ **No Java Installation Required**: Bundled JRE (~63 MiB uncompressed)
- 🌍 **4 Platforms Supported**: Linux (x86_64, ARM64), macOS (ARM64), Windows (x86_64)
- 🚀 **Embedded Mode**: Direct database access in Python process (no network)
- 🌐 **Server Mode**: Optional in-process HTTP server with the Studio web UI: it costs ~8MB of wheel and nothing at runtime until you call `create_server()`
- 📦 **Self-contained**: All dependencies bundled (~69 MiB current Linux wheel)
- 🔄 **Multi-model**: Graph, Document, Key/Value, Vector, Time Series
- 🔍 **Multiple query languages**: SQL and OpenCypher
- ⚡ **High performance**: Direct JVM integration via JPype
- 🔒 **ACID transactions**: Full transaction support
- 🎯 **Vector storage**: Store and query vector embeddings with HNSW (JVector) indexing
- 📥 **Data import**: Built-in CSV and ArcadeDB JSONL import

---

## 📦 What's Inside

The `arcadedb-embedded` package is platform-specific and self-contained:

**Package Contents (the 26.10.1 Linux x86_64 wheel; varies by platform and version):**

- **Wheel size (compressed)**: ~69 MiB
- **ArcadeDB JARs (uncompressed)**: ~34 MiB across 64 JARs
- **Bundled JRE (uncompressed)**: ~63 MiB (platform-specific Java 25 runtime via jlink)
- **Installed package size**: ~97 MiB

The compressed wheel size is measured from `dist/*.whl`, and the installed package size
is measured from the extracted `site-packages/arcadedb_embedded/` directory.

Of that, the optional **server stack is 12 JARs, 8.58MB uncompressed**. [Server
Mode](https://docs.humem.ai/arcadedb/latest/guide/server/) breaks the cost down and
explains what you pay at runtime (nothing, until you start a server).

**Note**: Some JARs are excluded to optimize package size (e.g., gRPC wire protocol). See [`scripts/jar_exclusions.txt`](scripts/jar_exclusions.txt) for details.

Import: `import arcadedb_embedded as arcadedb`

---

## 🧪 Testing

Tests run against the built wheel, installed with its `test` extra into a
virtual environment. Once that is set up, `uv run` uses it without activation:

```bash
# Once
uv venv

# After each build: --reinstall-package, as a rebuild keeps the version (26.11.1.dev0)
uv pip install --reinstall-package arcadedb-embedded "$(ls dist/*.whl | head -1)[test]"

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_core.py -v
```

See [testing documentation](https://docs.humem.ai/arcadedb/latest/development/testing/) for detailed test documentation.

---

## 🔧 Building from Source (Advanced)

Linux uses Docker. macOS and Windows use a native Java 25+ JDK with jlink.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Build for your current platform (auto-detected)
./scripts/build.sh
```

Built wheels will be in `dist/`. To test one, install it together with the
`test` extra into a virtual environment and run `pytest tests`.

**[Build instructions](https://docs.humem.ai/arcadedb/latest/getting-started/installation/#building-from-source)**

> **Developer Note:** See [build architecture docs](https://docs.humem.ai/arcadedb/latest/development/build-architecture/) for comprehensive documentation of the multi-platform build architecture, including how we achieve platform-specific JRE bundling across the supported platforms on GitHub Actions.

## Development

The ArcadeDB jars come from the `arcadedata/arcadedb` Docker image: `latest` by
default, which follows ArcadeDB's current snapshot, or the tag in `ARCADEDB_TAG`
to pin one (`ARCADEDB_TAG=26.11.1-SNAPSHOT ./scripts/build.sh`). The package
version is taken from the `arcadedb-engine-<version>.jar` in that image, so
`26.11.1-SNAPSHOT` builds `26.11.1.devN`. CI does the same, and also runs every
night so each new snapshot is tested. Both workflows take an `arcadedb-tag` input
(manual run or `workflow_call`) to build from a pinned image: an exact version tag such as `26.10.2`. See [versioning strategy](https://docs.humem.ai/arcadedb/latest/development/release/#python-versioning-strategy) for details on development vs release mode handling.

## Origin

This code moved here from the `bindings/python` directory of
[ArcadeData/arcadedb](https://github.com/ArcadeData/arcadedb) at
`ee04bd202abce0183c6263a3c4d8a61a4d526960`, with its git history. Issue
references in imported commit messages point at `ArcadeData/arcadedb`.

---

## 📋 Package Structure

```bash
arcadedb_embedded/
├── async_executor.py    # Asynchronous command/query + record execution
├── core.py              # Database and DatabaseFactory
├── exceptions.py        # ArcadeDBError exception
├── exporter.py          # Data export (JSONL, CSV)
├── graph.py             # Graph wrappers
├── graph_batch.py       # GraphBatch high-throughput graph ingest wrapper
├── importer.py          # Data import (CSV, XML, ArcadeDB JSONL)
├── __init__.py          # Public API exports
├── jvm.py               # JVM lifecycle management
├── _logging.py          # Internal logging helpers
├── results.py           # ResultSet and Result wrappers
├── schema.py            # Schema management API
├── server.py            # ArcadeDBServer for optional HTTP/Studio mode
├── transactions.py      # TransactionContext manager
├── type_conversion.py   # Python-Java type conversion utilities
└── vector.py            # Vector search and HNSW (JVector) indexing
```

**[Architecture details](https://docs.humem.ai/arcadedb/latest/development/architecture/)**

---

## 🤝 Contributing

[See our contributing guidelines](https://docs.humem.ai/arcadedb/latest/development/contributing/)

## 📄 License

Both upstream ArcadeDB (Java) and this ArcadeDB Embedded Python project are licensed under Apache 2.0, fully open and free for everyone, including commercial use.

---

## 🔗 Links

- **Documentation**: <https://docs.humem.ai/arcadedb/latest/>
- **PyPI**: <https://pypi.org/project/arcadedb-embedded/>
- **GitHub**: <https://github.com/humemai/arcadedb-embedded-python>
- **ArcadeDB Main Docs**: <https://docs.arcadedb.com>
- **Issues**: <https://github.com/humemai/arcadedb-embedded-python/issues>
