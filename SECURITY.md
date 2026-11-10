# Security Policy

The ArcadeDB team takes the security of the database and its ecosystem seriously. This repository builds the Python packages `arcadedb` (the native build) and `arcadedb-jvm` (the JVM build). We appreciate responsible disclosure of vulnerabilities and aim to acknowledge, investigate, and remediate valid reports as quickly as possible.

## Reporting a vulnerability

**Please do not open public GitHub issues for security problems.**

Report it privately through GitHub's private vulnerability reporting for this repository: open the [Security tab](https://github.com/ArcadeData/arcadedb-python/security), choose **Report a vulnerability**, or go straight to https://github.com/ArcadeData/arcadedb-python/security/advisories/new.

This channel reaches the maintainers privately and lets us coordinate a fix and a CVE before the issue becomes public.

### Is it in the bindings or in the engine?

Both packages ship the ArcadeDB engine unmodified. The engine code is upstream's, from the [ArcadeData/arcadedb](https://github.com/ArcadeData/arcadedb) repository.

- A vulnerability in **this repository** (the Python code in `src/arcadedb`, the Java bridge in `src/java`, the native build in `native/`, the build and release scripts, or what goes into the wheels) belongs here.
- A vulnerability in **the engine itself** (the server, the HTTP API, SQL or openCypher execution, the Postgres, Redis, Bolt, and other wire protocols, storage, Studio) belongs in the private advisories of ArcadeData/arcadedb, as described in its [Security Policy](https://github.com/ArcadeData/arcadedb/blob/main/SECURITY.md): https://github.com/ArcadeData/arcadedb/security/advisories/new.

If you are not sure which one it is, report it here and say so. We will forward it.

### What to include in a report

To help us triage quickly, please include:

- The package (`arcadedb` or `arcadedb-jvm`) and its version. `python -c "import arcadedb; print(arcadedb.__backend__, arcadedb.__version__)"` prints both.
- Your platform (operating system and CPU architecture) and your Python version.
- A clear description of the issue and the security impact.
- Step-by-step reproduction instructions, or a minimal proof-of-concept.
- Any relevant logs, captures, or patches you already have.

## Supported versions

Security fixes are delivered in a new release of the package. Both packages follow the version of the ArcadeDB engine they ship:

| Version       | Meaning                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------- |
| `A.B.C`       | The release. The wheels ship the `A.B.C` engine.                                            |
| `A.B.C.postN` | A fix in the Python code of the released `A.B.C`. The wheels still ship the `A.B.C` engine. |
| `A.B.C.devN`  | A preview of the engine release `A.B.C`, built from upstream snapshot jars.                 |

The numbering rule is on the docs page [Release](docs/modules/ROOT/pages/development/release.adoc).

| Version                                         | Status                         |
| ----------------------------------------------- | ------------------------------ |
| Latest release of `arcadedb` and `arcadedb-jvm` | Supported                      |
| Older versions                                  | Not supported - please upgrade |

A `devN` pre-release is a preview and is not a supported line: a fix reaches it only through the next release. Check the [releases of `arcadedb`](https://pypi.org/project/arcadedb/#history) and the [releases of `arcadedb-jvm`](https://pypi.org/project/arcadedb-jvm/#history) on PyPI for the current version.

## Our response

When you report an issue we will:

1. Acknowledge receipt.
2. Assess the report (confirmed, needs more info, or not a vulnerability) and tell you.
3. Work on a fix and keep you updated while we do.
4. Coordinate a release and, if applicable, a CVE through the GitHub Security Advisory process.
5. Publish the advisory only after a fixed release is available, unless the vulnerability is already public.

## Disclosure and credit

We follow a **coordinated disclosure** model. By default we credit reporters in the published advisory and the release notes. If you prefer to remain anonymous or to use a specific handle, let us know in your report.

We ask that reporters:

- Give us reasonable time to fix the issue before public disclosure.
- Do not access, modify, or exfiltrate data that is not yours.
- Do not run denial-of-service tests against production systems.

## Scope

In scope:

- The Python packages `arcadedb` and `arcadedb-jvm` as published on PyPI, and the code, build scripts, and workflows in this repository.
- What the wheels bundle: the ArcadeDB jars, the Java runtime of `arcadedb-jvm`, and the native library of `arcadedb`. Engine vulnerabilities go to upstream as described above.

Out of scope:

- Third-party forks and derivative products not maintained by the ArcadeDB team.
- Vulnerabilities in upstream dependencies that are already tracked by their own advisories. (Please still let us know so we can upgrade.)
- Issues that require physical access to an already-compromised host.

## Dependencies

Every pull request runs Bandit over the source and the tests and a dependency floor audit with `pip-audit`. The workflows are described on the docs page [Continuous integration](docs/modules/ROOT/pages/development/ci.adoc). If you can show that a development-only dependency is reachable from a published wheel, or exploitable in a way we have not considered, please tell us through the private channel above.

Thank you for helping keep ArcadeDB and its users safe.
