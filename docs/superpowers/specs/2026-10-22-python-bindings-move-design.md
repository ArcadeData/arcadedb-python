# Move the Python bindings out of the ArcadeDB monorepo

Date: 2026-10-22
Source: `ArcadeData/arcadedb` @ `ee04bd202abce0183c6263a3c4d8a61a4d526960`, directory `bindings/python`
Target: `ArcadeData/arcadedb-python-bindings` (this repo; currently only `.gitignore`, `LICENSE`, `README.md`)

## Intent

The Python bindings (`arcadedb-embedded`, a JPype wrapper that bundles the ArcadeDB jars and a
JRE into a platform wheel) live in `bindings/python` of the main ArcadeDB repo. They get their own
repo so they build, test and evolve independently of the Java monorepo, and the monorepo stops
carrying Python CI.

The move is done in three stages, and each stage gates the next:

1. Move the code (with history) and CI into this repo.
2. Prove it works here: wheel builds and the test and example workflows go green on GitHub Actions.
3. Only then remove `bindings/python` and every reference to it from upstream.

## Goals

- All tracked files under `bindings/python` arrive here at the repo root, with their git history.
- The two Python workflows (`test-python-bindings.yml`, `test-python-examples.yml`) arrive and pass here.
- The build no longer depends on the monorepo: the ArcadeDB version comes from the
  `arcadedata/arcadedb:latest` Docker image, not from the parent `pom.xml`.
- The bindings follow upstream snapshots automatically. `latest` is republished together with each
  `X.Y.Z-SNAPSHOT` tag (verified 2026-10-22: `latest` and `26.11.1-SNAPSHOT` pushed at the same time),
  so when upstream moves to the next snapshot, the next build here picks it up with no edit.
- Upstream is cleaned up only after this repo is green.

## Non-goals

- The `humemai/arcadedb-embedded-python` repo. Its owner will deal with it later. Keep the
  `pyproject.toml` URLs and author fields as they are.
- PyPI publishing. Upstream has no publishing workflow today, and none is added here.
- Behaviour changes to the bindings, tests or examples.
- `e2e-python/` and the `python-e2e-tests` job in upstream `mvn-test.yml`. They test the server over
  the wire, not these bindings, so they stay upstream.

## Anti-goals

- Deleting anything upstream before this repo's CI is green on `main`.
- Losing history. `git blame` must keep working, because many comments in tests and config explain
  past failures by date and issue number.
- Hard-coding a version that someone has to bump by hand on every snapshot.

## Constraints and current coupling (what has to change)

- **Version**: `scripts/extract_version.py` and `scripts/write_version.py` read the parent `pom.xml`
  (`X.Y.Z-SNAPSHOT` → `X.Y.Z.devN`). `scripts/build.sh` uses it to choose the Docker tag.
  `scripts/Dockerfile.build` does `COPY pom.xml /arcadedb/pom.xml`.
- **Paths**: `Dockerfile.build` uses the monorepo root as the build context (`build.sh` passes `../..`)
  and copies `bindings/python/...`. Both workflows `cd bindings/python`, and both filter on
  `bindings/python/**` (26 occurrences in `test-python-bindings.yml` alone).
- **Engine from source**: the `engine-from-source` input in both workflows builds the jars from the
  same commit with `./mvnw -pl package`. This can't work outside the monorepo.
- **Triggers**: `test-python-bindings.yml` has a `workflow_run` on upstream's `Release` workflow,
  which doesn't exist here.
- No repository secrets are used by either workflow (only `upload-artifact`).
- The `[tool.uv]` comment in `pyproject.toml` mentions a "repo-root pyproject.toml" that doesn't exist upstream either.
- `git-filter-repo` isn't installed locally (`brew install git-filter-repo`).

## Approaches considered

1. **Move with history via `git filter-repo`.** **Chosen.** Filter a fresh clone of upstream down to
   `bindings/python/**` plus the two workflow files, renaming `bindings/python/` to the root, then merge
   into this repo's `main` with `--allow-unrelated-histories`. Keeps the 63 commits that touched the
   bindings and lets the code live at the repo root. Costs a one-off filtering step and requires
   rewriting the `bindings/python` paths.
2. **Plain copy.** One import commit referencing the upstream SHA. Simpler, but history and blame
   stay behind in upstream.
3. **Plain copy into `bindings/python/`.** Fewest path edits, but leaves a pointless nested folder in a
   single-purpose repo.

The user picked 1.

## Decisions

Decided by the user:

- Move what upstream has now. humemai integration comes later from its owner.
- Bind to the current snapshot and follow the next snapshot automatically, using the `latest` Docker image.
- Move with history (approach 1).

Calls made while writing this document (Claude's, open to correction):

- **Layout**: bindings content at the repo root (`src/`, `tests/`, `examples/`, `scripts/`,
  `pyproject.toml`, ...). The workflows go to `.github/workflows/`.
- **Version source**: `ARCADEDB_TAG` defaults to `latest` and can be overridden by environment or
  build arg (for example `26.11.1-SNAPSHOT` to pin). The Maven version is read from the engine jar
  name inside the image (`/home/arcadedb/lib/arcadedb-engine-<ver>.jar`), then put through the existing
  `X.Y.Z-SNAPSHOT → X.Y.Z.devN` / release conversion. The `pom.xml` reading code is replaced,
  not kept as a fallback.
- **Docker build context** becomes this repo's root. The `COPY pom.xml` line is removed.
- **`engine-from-source` input** is removed from both workflows. Its replacement is the
  `ARCADEDB_TAG` override or a local jars directory (`build.sh ... /path/to/jars` already supports one).
- **Triggers**: push and PR (no path filter needed in a single-purpose repo), `workflow_dispatch`, plus
  a nightly `schedule` so a new snapshot is exercised even when nothing here changes. The
  `workflow_run: Release` trigger is dropped. `workflow_call` is kept.
- **Repo hygiene that moves with the code**: the bindings' own `.gitignore` merges into the root
  `.gitignore`. The upstream config entries that apply to the bindings come here: the Dependabot
  `uv` entry → `.github/dependabot.yml` (directory `/`), the black/isort/yaml pre-commit hooks
  → `.pre-commit-config.yaml`, and the bindings line of `REUSE.toml`.
  Codacy and Meterian config are not carried over (no such integrations on this repo yet).
- **Test fixtures**: `tests/test_import_database.py` reads five files from upstream's
  `integration/src/test/resources/` via `parents[3]`. These are `importer-vertices.csv`,
  `importer-edges.csv`, `neo4j-export-mini.jsonl`, `importer-word2vec.txt` and
  `importer-rdf.xml`, about 10 KB in total. Copy them into `tests/fixtures/` and point the path
  there. This is a path change, not a behaviour change. Skipping the tests isn't an option:
  `scripts/check_test_skips.py` fails CI on any skip. Examples 07–12 get the same treatment for
  `parents[3]` and `bindings/python/dist`.
- **Issue references in imported history**: the `filter-repo` step rewrites `#NNNN` in commit
  messages to `ArcadeData/arcadedb#NNNN`, so the links keep pointing at upstream issues. This must
  be done before the first push.
- **`latest` vs. local-jar check**: `build.sh` compares the Docker tag to the jar version. With
  `latest`, it must compare against the version read from the jars instead.
- **Also carried over**: the fourth upstream pre-commit hook (shfmt on the bindings' `.sh`) and
  the `.github/actionlint.yaml` entries for both workflows. The root `.gitignore` here ignores
  `lib/`, which must not hide `local-jars/lib/` handling.
- **README**: the root README becomes the bindings README, plus a short note on where the code
  came from and how the version is chosen.
- **How it lands here**: the history merge is pushed to a branch and goes in via a PR to `main`, so CI
  runs before `main` changes. The PR is opened only after a local `scripts/build.sh` build plus its
  tests pass.

## Freeze

Decided by the user: `bindings/python` is frozen upstream from the moment the history is extracted
until Stage 3 deletes it. The extraction uses upstream `main` as of that moment, and the SHA in the
header of this document is updated to match. There's no re-sync step.

Stage 3 starts by running `git log <extracted SHA>..origin/main -- bindings/python`. It must be
empty. If anything landed despite the freeze, stop and port it here before deleting anything.
The Stage 3 PR states the extracted SHA.

## Stage 3: upstream cleanup (after green `main` here)

One PR on `ArcadeData/arcadedb` that:

- deletes `bindings/python/`, `.github/workflows/test-python-bindings.yml` and `test-python-examples.yml`;
- removes the `bindings/python` entries from `mvn-test.yml` and `mvn-deploy.yml` (`paths-ignore`), `meterian.yml`
  (`--exclude-folders`), `dependabot.yml`, `.codacy.yml`, `REUSE.toml` and `.pre-commit-config.yaml`
  (the three hooks filtered to `bindings/python` and the yaml exclude);
- checks `.github/scripts/check-gitignore-src-paths.sh`, `README.md` and `sbom.json` for references and
  points the README at the new repo;
- does not touch `e2e-python/` or historical docs under `docs/`.

## Success criteria

- `main` here has the bindings at the root, and `git log --follow src/arcadedb_embedded/<file>` shows
  upstream commits.
- `scripts/build.sh` builds a wheel locally against `latest` with no ArcadeDB checkout present, and the
  wheel's version matches the image's snapshot (today `26.11.1.devN`).
- Both workflows pass on `main` here, including a manual `workflow_dispatch` run.
- Upstream cleanup PR is merged with upstream CI green.

## Left to the builder

- Exact `git filter-repo` invocation and how the two workflow files are pulled in alongside the directory rename.
- How the `.devN` counter behaves without `pom.xml`. Keep the existing tracker logic if it works, otherwise
  pick something deterministic (for example a build number) and note it.
- Workflow job and matrix contents beyond the path, trigger and `engine-from-source` changes above.
- The order of commits within the PR.
