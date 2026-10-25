# arcadedb (native build)

This is `arcadedb`, the native build of the ArcadeDB Python bindings, and the default choice for scripts, notebooks, tests, CLIs, serverless functions and agents. Its sibling `arcadedb-jvm` runs the Java engine on a bundled JVM: use `arcadedb-jvm` for heavy, long-running loads, where the JIT compiler makes the engine faster over time, and when you need the Java-object APIs. Both packages `import arcadedb` and have the same public API. Install one, not both. [Which package?](https://github.com/ArcadeData/arcadedb-python/blob/main/docs/modules/ROOT/pages/which-package.adoc)

ArcadeDB (SQL, openCypher, document, graph, vector, time series) runs in your Python process **without a JVM and without JPype**. The engine is compiled ahead of time with GraalVM Community Edition into a shared library (`libarcadedb.so`, `.dylib` or `.dll`), and Python loads it with `ctypes`. It starts in milliseconds and has a small memory footprint.

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
> Transition note: the first release of this layout is 26.11.1. Until `arcadedb` passes its readiness bar, it may appear only as pre-releases, `26.11.1.devN`. Install those with `pip install --pre arcadedb`. Report problems at <https://github.com/ArcadeData/arcadedb-python/issues> and say that you use the native build.

## Install

```bash
pip install arcadedb       # or: uv add arcadedb
```

Optional extras for the columnar helpers: `[numpy]`, `[arrow]`, `[pandas]`, `[polars]`.

<!-- REMOVE WHEN: the old arcadedb-embedded* shims are retired and two releases have shipped with the two-package layout allow-old-name -->
Old names: this package was formerly `arcadedb-embedded-native`, imported as `arcadedb_native`. The old package will get a deprecation shim that points here.

If `arcadedb-jvm` is installed in the same environment, `import arcadedb` raises an `ImportError` that names both packages. Run `pip uninstall arcadedb arcadedb-jvm`, then install the one you want.

## Use

```python
import arcadedb

print(arcadedb.__backend__)   # "native"

with arcadedb.create_database("/tmp/demo") as db:
    db.schema.create_vertex_type("Person")
    with db.transaction():
        db.command("sql", "CREATE VERTEX Person SET name = ?", "Ada")
    print(db.query("sql", "SELECT FROM Person").to_list())
```

`arcadedb.__backend__` is `native` for this package and `jvm` for `arcadedb-jvm`. `arcadedb.__version__` is the package version.

## What is in the wheel

- The shared library with the ArcadeDB engine: SQL and openCypher, the document, graph, key/value, vector and time series models, ACID transactions, and the in-process server with its HTTP API and the Postgres, Redis and Bolt wire protocols.
- The Python API on top of it, loaded with `ctypes`.
- Not in the image: gRPC, the MongoDB wire protocol, Raft/HA and Gremlin.
- No polyglot scripting languages (`js` or `python` as query languages, script triggers, polyglot functions). The polyglot engine is switched off at start-up (`arcadedb.polyglotEngineEnabled=false`), so the first database open writes nothing at WARNING.

## Platforms

CPython 3.11 to 3.14. One wheel per platform serves every Python version, because the library is loaded with `ctypes` and is not compiled against CPython. The images are built for the baseline CPU of the platform (`-march=compatibility`), so they do not need a special instruction set.

| Platform | Wheel tag | Note |
| --- | --- | --- |
| Linux x86-64, glibc 2.34+ | `manylinux_2_34_x86_64` | |
| Linux arm64, glibc 2.34+ | `manylinux_2_34_aarch64` | |
| macOS arm64 | `macosx_<minos>_arm64` | |
| Windows x64 | `win_amd64` | The newest wheel. It is built in CI, and its test run is the least proven of the four |

Intel macOS and Windows arm64 are not built.

## Not available or not yet

- **JVM-only APIs do not exist**: `get_java_database()` and the other `get_java_*` calls, `start_jvm()`, `jvm_kwargs` and JVM options. There is no JVM. Use `arcadedb-jvm` if you need them.
- **A few API calls raise `NotOnNativeYet`** (a `NotImplementedError`) instead of misbehaving: `AsyncExecutor.transaction`, `find_nearest_approximate` over product quantization, `get_metadata` of a foreign index, and `get_stats`.
- Free-threaded Python builds are not in the CI matrix.

The bindings' own tests run against the installed wheel (the parity run). What does not pass is the list above, or JVM-only by nature.

## Licence

Apache-2.0 for ArcadeDB and these bindings. The library also contains the GraalVM Community Edition runtime and JDK class library code, licensed GPLv2 **with the Classpath Exception**, plus the Apache/BSD/MIT/UPL/EDL third-party Java libraries listed in `THIRD_PARTY_NOTICES.md`. All texts and the source locations are in the wheel's `*.dist-info/licenses/`.
