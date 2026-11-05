## How to contribute to ArcadeDB Python

This repository builds two Python packages that both import as `arcadedb`: `arcadedb` (the native build, the engine compiled by GraalVM into a shared library) and `arcadedb-jvm` (the JVM build, the Java engine in your process through JPype). The engine itself is upstream's, from [ArcadeData/arcadedb](https://github.com/ArcadeData/arcadedb), unmodified. See [Which package?](docs/modules/ROOT/pages/which-package.adoc) for the difference.

### Contribution License

ArcadeDB Python is licensed under the [Apache License 2.0](LICENSE). By submitting a contribution (pull request, patch, or any other form), you agree that your contribution is licensed under the same terms, as stated in Section 5 of the Apache License 2.0. You keep your copyright. By submitting a contribution, you represent that you have the right to license it under the Apache License 2.0, that is, the work is your own, or you have permission from the copyright holder.

#### **Did you find a bug?**

- **Do not open a GitHub issue if the bug is a security vulnerability.** Report it privately, as described in [SECURITY.md](SECURITY.md).

- **Ensure the bug was not already reported** by searching under [Issues](https://github.com/ArcadeData/arcadedb-python/issues).

- **Decide where it belongs.** A bug in the Python API, the packaging, or the build goes here. A problem that also shows in the Java API or in Studio is an engine bug: open it in [ArcadeData/arcadedb](https://github.com/ArcadeData/arcadedb/issues/new) with a Java or SQL reproduction.

- If you cannot find an open issue, [open a new one](https://github.com/ArcadeData/arcadedb-python/issues/new). Include a **title and clear description**, a **code sample** or a failing test, the package (`arcadedb` or `arcadedb-jvm`) and its version, the output of `python -c "import arcadedb; print(arcadedb.__backend__)"`, your Python version, and your platform.

#### **Did you write a patch that fixes a bug?**

- Open a new GitHub pull request with the patch. The description states the problem and the solution, and says `Closes #N` for the issue it fixes.

- Add a regression test: it fails on the old code and passes on the new.

#### **Do you intend to add a new feature or change an existing one?**

- Open an issue first that states the problem the change solves, and wait for feedback before you write much code.

- A change to engine behavior (SQL, openCypher, storage, the server) goes to [ArcadeData/arcadedb](https://github.com/ArcadeData/arcadedb). This repository wraps the engine for Python.

#### **Did you fix whitespace, format code, or make a purely cosmetic patch?**

Changes that are cosmetic in nature and do not add anything substantial to the stability, functionality, or testability will generally not be accepted.

#### **Are you adding a new dependency?**

Add a dependency (a Python extra, or anything bundled in a wheel) only when it is strictly necessary, and only when its license is compatible with Apache License 2.0: Apache 2.0, MIT, BSD, ISC, EPL, UPL, CC0. LGPL 2.1 or later, MPL-2.0, and CDDL are acceptable as unmodified libraries only. GPL (except GPL-2.0 with the Classpath Exception), AGPL, SSPL, BUSL, Elastic, and Commons Clause are not. Say in the pull request why the dependency is needed.

### Prepare your environment

You need [uv](https://docs.astral.sh/uv/), Git, and Python 3.11 to 3.14. This repository is a package definition, not a uv project (`pyproject.toml` sets `managed = false`), so there is no environment to sync. You build a wheel, and `uv run --no-project --with <wheel>` makes a throwaway environment from it. Nothing is installed into your system Python.

```shell
git clone git@github.com:ArcadeData/arcadedb-python.git
cd arcadedb-python
```

### Build and test `arcadedb-jvm`

On Linux the build runs in Docker. On macOS and Windows it needs a JDK 25 or later. The jars come from the `arcadedata/arcadedb` Docker image.

```shell
./scripts/build.sh                                  # an arcadedb-jvm wheel for your platform in dist/
ARCADEDB_TAG=26.11.1-SNAPSHOT ./scripts/build.sh    # pin the arcadedata/arcadedb image (default: latest)
```

Tests run against the built wheel, not the source tree. Use `python -m pytest`, not the `pytest` entry point: with `--with`, the entry point does not see the packages of the environment.

```shell
W="$(ls dist/*.whl | head -1)"
uv run --no-project --with "$W[test]" python -m pytest                                    # all
uv run --no-project --with "$W[test]" python -m pytest tests/test_core.py -v              # one file
uv run --no-project --with "$W[test]" python -m pytest tests/test_core.py::TestX::test_y  # one test
```

If uv's cache still serves an older wheel of the same version after a rebuild, add `--reinstall-package arcadedb-jvm`. Changing Java code in `src/java` needs a rebuild. [CLAUDE.md](CLAUDE.md) has the rest of the build options.

### Build and test `arcadedb` (native)

The native image build takes about 5 to 30 minutes per platform and needs GraalVM Community Edition. Most changes to shared Python code are tested by CI, which builds the image and runs the tests against the installed wheel in `test-native.yml`. If you build it yourself, the steps are the ones in that workflow:

```shell
bash native/scripts/build-image.sh <outdir>
uv run --no-project --python 3.12 python native/wheel/build_wheel.py --lib-dir <outdir> --version <version> --engine <engine> --out <wheeldir>
```

`native/wheel/parity_wheel.py` runs the test suite against the installed wheel; its arguments are in the "Run the parity suite" step of `test-native.yml`. See [Continuous integration](docs/modules/ROOT/pages/development/ci.adoc) for when each workflow runs.

### Tests

- Every fix comes with a regression test. Write the test first and run every test the change could affect.
- No skips: CI fails on a skipped test that is not listed in `scripts/check_test_skips.py`. A test that cannot run on a platform goes in `collect_ignore` in `tests/conftest.py`, and a known engine bug uses a strict `xfail`.
- Never bind or address a hand-picked port in a test. Use `server.get_http_port()` or the `_free_ports()` pattern in `tests/test_server_wire_protocols.py`.
- Never assert on raw wall-clock time. A JVM pause makes it flaky.

### Pre-commit

This project uses [pre-commit](https://pre-commit.com/) (black, isort, shfmt, pretty-format-yaml, actionlint). Run it on the files you changed before you push, and again if it reformatted something:

```shell
uvx pre-commit run --files <changed files>
uvx pre-commit run --all-files
```

CI also runs Bandit, which must be clean:

```shell
uvx bandit==1.9.4 -c pyproject.toml -r src tests --severity-level low --confidence-level low
uvx bandit==1.9.4 -c pyproject.toml -r examples --severity-level medium --confidence-level high
```

### Workflow

1. An [issue](https://github.com/ArcadeData/arcadedb-python/issues) states the problem.
2. Make a branch from `main`. Do not commit to `main`.
3. Open a pull request that says `Closes #N`. Fill in the pull request template (`.github/pull_request_template.md`): what changed, the issue, the timed result for a performance change, the docs you updated, and the checklist.
4. The checks must pass before a pull request is merged. A draft pull request skips the platform jobs; mark it ready for review to start them.

A commit message starts with a short subject that says what changes and where, for example `Native find_nearest: send the query vector as one float32 array, not a Python list`. The body says why. Link the issue with `Closes #N` in the pull request.

### Performance changes

A change that claims a speed or memory difference comes with a timed A/B result: interleaved runs of the new build against `main`, a control run of `main` against itself to show the noise, and the median ratio with its interval. The pages [Bulk loads and commit size](docs/modules/ROOT/pages/how-to/bulk-loads.adoc) and [Concurrency, processes and crashes](docs/modules/ROOT/pages/how-to/concurrency.adoc) each state one such measurement.

### Documentation

The documentation lives in [`docs/`](docs/modules/ROOT/pages) as an Antora component (AsciiDoc) and covers only what is specific to the Python bindings. It is published at [docs.arcadedb.com/arcadedb-python](https://docs.arcadedb.com/arcadedb-python/) from `main`. A pull request that changes behavior updates the page that describes it, and `README.md` when it describes the same thing.

Link into the main ArcadeDB manual with `{arcadedb-docs-url}/<page>`, not with an `arcadedb::` xref. The `Docs` workflow builds the component as a site of its own and fails on any warning. To build it locally, run this from the repository root and open `docs/build/site/index.html`:

```shell
npx -p @antora/cli@3.2.1 -p @antora/site-generator@3.2.1 antora docs/antora-playbook.yml
```

The runnable scripts in [`examples/`](examples/README.md) are tested by CI too.

### Releases

Maintainers release both packages from a Git tag, one version a month after upstream releases the engine. The version rule and the steps are on the docs page [Release](docs/modules/ROOT/pages/development/release.adoc).

Thank you for contributing.
