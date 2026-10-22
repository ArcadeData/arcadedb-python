# Python Bindings Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `bindings/python` from `ArcadeData/arcadedb`, with its history, into this repo. Make it build and pass CI here against `arcadedata/arcadedb:latest`, then delete it upstream.

**Architecture:**
- `git filter-repo` extracts `bindings/python/**` and the two Python workflows from a fresh upstream clone and moves the bindings to the repo root.
- The result is merged into a branch here, and the monorepo coupling is removed.
- The ArcadeDB version now comes from the `arcadedb-engine-<ver>.jar` name in the jars directory, not from `pom.xml`. The Docker tag defaults to `latest`.

**Tech Stack:** git-filter-repo, bash, Docker (buildx), Python 3.10+ (setuptools, pytest, JPype), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-22-python-bindings-move-design.md`

## Global Constraints

- **Source:** `ArcadeData/arcadedb` `main` at extraction time. After extraction, record its SHA in the spec header. `bindings/python` is frozen upstream from then on.
- **History:** keep all of it. Rewrite issue refs in commit messages from `#NNNN` to `ArcadeData/arcadedb#NNNN`.
- **Layout:** bindings at the repo root; workflows in `.github/workflows/`.
- **Docker tag:** default `latest`, overridable by the `ARCADEDB_TAG` environment variable. No `pom.xml` anywhere and no fallback to one.
- **Versions:** the Maven→PEP 440 mapping is unchanged: `X.Y.Z-SNAPSHOT` → `X.Y.Z.devN` (N from the existing tracker), `X.Y.Z-RCn` → `X.Y.Zrcn`, a release stays as it is, and `--python-patch=N` adds `.postN`.
- **Out of scope:**
  - Behaviour changes to bindings, tests or examples (path changes only).
  - Edits to the `pyproject.toml` humemai URLs and authors.
  - PyPI publishing.
  - Upstream's `e2e-python/`.
- **Merging:** the PR here must be merged with a **merge commit**, never squash or rebase, or the history is lost.
- **Upstream:** nothing is deleted upstream until both workflows are green on `main` here. Opening the upstream PR needs the user's go-ahead.
- **Commit trailer:** `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Concurrent snapshot publish:** `latest` is republished while a CI run is in flight. All jobs in one run must build against the same ArcadeDB version. The `download-jars` job resolves the version once and hands it to every later job as `ARCADEDB_TAG` (Task 5).
- **Bad jars directory:** a directory with no `arcadedb-engine-*.jar`, or with engine jars of two versions, must fail the build with a clear error, not produce a `0.0.0` or wrong-version wheel (Task 3 tests).
- **Pinned tag:** `ARCADEDB_TAG` set to a release such as `26.10.2` gives wheel version `26.10.2`. An RC tag gives `rc` (Task 3 tests).
- **Local jars:** `build.sh linux/amd64 3.12 /path/to/jars` with the default `latest` tag checks the integration jar against the version read from those jars, not against the string `latest` (Task 4 verification).
- **Native builds:** the macOS/Windows path in `build-native.sh` takes its version from the downloaded jars directory, because there is no `pom.xml` to read (Task 4, exercised by the CI matrix in Task 7).

---

## File Structure (after the move)

```
.github/workflows/test-python-bindings.yml   # moved; paths/triggers rewritten (Task 5)
.github/workflows/test-python-examples.yml   # moved; paths/triggers rewritten (Task 5)
.github/actionlint.yaml                       # new: the two upstream entries (Task 5)
.github/dependabot.yml                        # new: uv at "/" + github-actions (Task 5)
.pre-commit-config.yaml                       # new: black, isort, yaml, shfmt hooks (Task 5)
REUSE.toml                                    # new: bindings annotation, root paths (Task 5)
.gitignore                                    # merged root + bindings ignore rules (Task 1)
README.md                                     # bindings README + provenance/version note (Task 6)
pyproject.toml, setup.py                      # moved
scripts/extract_version.py                    # reads version from jars dir (Task 3)
scripts/write_version.py                      # takes jars dir (Task 3)
scripts/build.sh, build-native.sh, Dockerfile.build   # root build context, latest tag (Tasks 2, 4)
src/arcadedb_embedded/, src/java/             # moved
tests/                                        # moved; + test_extract_version.py (Task 3)
tests/fixtures/                               # 5 files copied from upstream integration/src/test/resources (Task 2)
examples/                                     # moved; 07–12 paths fixed (Task 2)
docs/superpowers/{specs,plans}/               # this spec and plan
```

---

### Task 1: Extract history and merge it into a branch

**Files:**
- Create: everything under the File Structure above that comes from upstream (via merge)
- Modify: `.gitignore`, `README.md` (merge conflict resolution), `docs/superpowers/specs/2026-10-22-python-bindings-move-design.md` (SHA)

**Interfaces:**
- Produces: branch `move-python-bindings` with upstream history at the root, and `EXTRACTED_SHA`, the upstream `main` commit it came from. Task 8 uses this SHA.

- [ ] **Step 1: Create the branch and commit the spec and plan**

```bash
git checkout -b move-python-bindings
git add docs/superpowers && git commit -m "docs: design and plan for moving the Python bindings here"
```

- [ ] **Step 2: Install filter-repo and make a fresh single-branch clone in the scratchpad**

Run: `brew install git-filter-repo && git clone --single-branch --branch main https://github.com/ArcadeData/arcadedb.git "$SCRATCH/arcadedb-extract"`. Then record `EXTRACTED_SHA=$(git -C "$SCRATCH/arcadedb-extract" rev-parse HEAD)`.

- [ ] **Step 3: Filter the clone**

```bash
cd "$SCRATCH/arcadedb-extract"
git filter-repo \
  --path bindings/python/ \
  --path .github/workflows/test-python-bindings.yml \
  --path .github/workflows/test-python-examples.yml \
  --path-rename bindings/python/: \
  --message-callback 'import re; return re.sub(rb"(?<![\w/])#(\d+)", rb"ArcadeData/arcadedb#\1", message)'
```

Expected:
- `git ls-files | wc -l` is 145 (143 bindings files plus 2 workflows).
- `git log --oneline | wc -l` is at least 63.
- `git log --grep='(#4037)' --oneline` is empty.
- `git log --grep='ArcadeData/arcadedb#4037' --oneline` is not empty.

- [ ] **Step 4: Merge into the branch**

```bash
cd <this repo>
git remote add extract "$SCRATCH/arcadedb-extract" && git fetch extract
git merge --allow-unrelated-histories extract/main -m "Import Python bindings history from ArcadeData/arcadedb@$EXTRACTED_SHA"
```

Resolve the add/add conflicts:
- **`README.md`:** take the bindings version. Task 6 adds the note.
- **`.gitignore`:** take the union, with the bindings block placed **last** so its `!src/**/` re-include wins. Remove the dead line `bindings/python/src/arcadedb_embedded/_version.py`.

Then `git remote remove extract`.

- [ ] **Step 5: Verify history and ignore rules**

Run each of these:
- `git log --oneline --follow src/arcadedb_embedded/core.py | wc -l`. Expected: more than 1, with upstream commits listed.
- `git status --ignored --short src/ scripts/ tests/`. Expected: no tracked-source path reported as ignored.
- `git check-ignore -v local-jars/lib/x.jar`. Expected: ignored, which is intended.

- [ ] **Step 6: Record the SHA in the spec and commit**

Replace the `Source:` SHA in the spec header with `$EXTRACTED_SHA`, then `git commit -am "docs: record extracted upstream SHA"`.

---

### Task 2: Rewrite monorepo paths to the root layout, and add test fixtures

**Files:**
- Modify: `scripts/Dockerfile.build`, `scripts/build.sh`, `scripts/build-native.sh`, `scripts/setup_jars.py`, `tests/test_import_database.py:66-75`, `examples/02,07,08,09,10,11,12_*.py`, comments in `src/arcadedb_embedded/results.py:210`, `src/arcadedb_embedded/type_conversion.py:77`, `tests/test_java_package_shadowing.py`, `tests/test_vector_params_verification.py`
- Create: `tests/fixtures/{importer-vertices.csv,importer-edges.csv,neo4j-export-mini.jsonl,importer-word2vec.txt,importer-rdf.xml}`, copied byte for byte from upstream `integration/src/test/resources/` at `EXTRACTED_SHA`

**Interfaces:**
- Consumes: Task 1 branch.
- Produces: the Docker build context is the repo root, and `Dockerfile.build` copies `src/`, `tests/`, `scripts/...`, `local-jars/lib/`, `pyproject.toml`, `setup.py` and `README.md` from there. Task 4 relies on this.

- [ ] **Step 1: Copy the five fixtures and point `_resource_path(name)` at `Path(__file__).resolve().parent / "fixtures" / name`**

- [ ] **Step 2: Rewrite paths**

Make these changes:
- **`Dockerfile.build`:** drop `bindings/python/` from every `COPY` source and comment.
- **`build.sh`:** change the two `../..` build contexts to `"$PY_BINDINGS_DIR"`. Change the header example to `./scripts/build.sh linux/amd64 3.12 /path/to/arcadedb/package/target/arcadedb-<version>.dir/arcadedb-<version>/lib`.
- **Examples 07–12:**
  - `parents[3]` → `parents[1]`
  - `repo_root / "bindings/python/dist"` → `repo_root / "dist"`
  - `"/workspace/bindings/python/..."` → `"/workspace/..."`
  - the `"...in bindings/python/dist..."` messages → `"...in dist..."`
- **Example 02's docstring:** `cd examples`.
- **`setup_jars.py:144`:** drop the `cd bindings/python` hint.
- **The four comments:** `bindings/python/src/java` → `src/java`, and `bindings/python/src` → `src`.

Leave the `"bindings/python/docs is not present on this branch"` test-data string in `tests/test_check_test_skips.py:53` as it is.

- [ ] **Step 3: Verify that no monorepo paths remain**

Run: `git grep -n -E 'bindings/python|parents\[3\]|\.\./\.\.' -- . ':!docs/superpowers' ':!.github/workflows'`
Expected: only `tests/test_check_test_skips.py:53`. The workflows are Task 5. `README.md` is Task 6 and may still match.

- [ ] **Step 4: Commit** — `refactor: build from the repo root instead of the monorepo; vendor importer fixtures`

---

### Task 3: Read the ArcadeDB version from the jars

**Files:**
- Modify: `scripts/extract_version.py`, `scripts/write_version.py`
- Test: `tests/test_extract_version.py` (new; loads the script with `importlib`, the same way `tests/test_check_test_skips.py` does)

**Interfaces:**
- Produces, in `scripts/extract_version.py`:
  - `extract_raw_version_from_jars(jars_dir: Path) -> str`. Returns the `<ver>` from `arcadedb-engine-<ver>.jar`. Raises `ValueError` when there is no engine jar, or when there are engine jars of more than one version.
  - `maven_to_pep440(raw: str, patch_version: int = 0, increment_dev: bool = False) -> str`. This is the existing conversion body of `extract_version_from_pom`, moved here unchanged.
  - `extract_version(jars_dir: Path, fmt: str = "pep440", patch_version: int = 0, increment_dev: bool = False) -> str`. With `fmt="docker"` it returns the raw version.
  - CLI: `python3 scripts/extract_version.py [JARS_DIR] [--format=pep440|docker] [--python-patch=N] [--increment-dev]`. The default `JARS_DIR` is `<repo>/src/arcadedb_embedded/jars`.
  - `extract_raw_version_from_pom` and `extract_version_from_pom` are deleted, and the module docstring is updated to say "engine jar name" instead of "pom.xml".
- Produces, in `scripts/write_version.py`: CLI `python3 scripts/write_version.py [JARS_DIR]` (same default). It writes `src/arcadedb_embedded/_version.py`, and the generated docstring says "from the bundled ArcadeDB engine jar".

- [ ] **Step 1: Write the failing tests**

```python
# helper: _jars(tmp_path, *names) creates empty files with those names and returns tmp_path
def test_snapshot_engine_jar_gives_raw_version(tmp_path):
    d = _jars(tmp_path, "arcadedb-engine-26.11.1-SNAPSHOT.jar", "arcadedb-server-26.11.1-SNAPSHOT.jar", "jline-3.30.0.jar")
    assert mod.extract_raw_version_from_jars(d) == "26.11.1-SNAPSHOT"
    assert mod.extract_version(d, fmt="docker") == "26.11.1-SNAPSHOT"

def test_snapshot_converts_to_dev(tmp_path, monkeypatch):
    # point the tracker at tmp_path so the repo's tracker is untouched
    monkeypatch.setattr(mod, "get_dev_version_tracker_path", lambda: tmp_path / "t.json")
    assert mod.extract_version(_jars(tmp_path, "arcadedb-engine-26.11.1-SNAPSHOT.jar")) == "26.11.1.dev0"

def test_release_and_rc_and_post(tmp_path):
    assert mod.maven_to_pep440("26.10.2") == "26.10.2"
    assert mod.maven_to_pep440("26.10.2", patch_version=1) == "26.10.2.post1"
    assert mod.maven_to_pep440("26.11.1-RC1") == "26.11.1rc1"

def test_no_engine_jar_raises(tmp_path):
    with pytest.raises(ValueError, match="arcadedb-engine"):
        mod.extract_raw_version_from_jars(_jars(tmp_path, "arcadedb-server-26.11.1-SNAPSHOT.jar"))

def test_two_engine_versions_raise(tmp_path):
    d = _jars(tmp_path, "arcadedb-engine-26.11.1-SNAPSHOT.jar", "arcadedb-engine-26.10.2.jar")
    with pytest.raises(ValueError, match="26.10.2.*26.11.1-SNAPSHOT|26.11.1-SNAPSHOT.*26.10.2"):
        mod.extract_raw_version_from_jars(d)

def test_cli_prints_docker_format(tmp_path):
    d = _jars(tmp_path, "arcadedb-engine-26.11.1-SNAPSHOT.jar")
    out = subprocess.run([sys.executable, str(SCRIPT_PATH), str(d), "--format=docker"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "26.11.1-SNAPSHOT"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python3 -m pytest tests/test_extract_version.py -v -p no:cacheprovider --noconftest`. Expected: FAIL with `AttributeError ... extract_raw_version_from_jars`. If `tests/conftest.py` needs the installed wheel, `--noconftest` keeps these pure-Python tests runnable without it.

- [ ] **Step 3: Implement the interfaces above.** Match engine jars with `^arcadedb-engine-(.+)\.jar$`. A CLI `ValueError` or `FileNotFoundError` prints `Error: ...` to stderr and exits with code 1, as it does today.

- [ ] **Step 4: Run the tests and confirm they pass** (same command). Expected: 6 passed.

- [ ] **Step 5: Commit** — `feat: derive the ArcadeDB version from the bundled engine jar instead of pom.xml`

---

### Task 4: Wire `latest` into the build scripts

**Files:**
- Modify: `scripts/build.sh:151-154, 404`, `scripts/build-native.sh:311, 330`, `scripts/Dockerfile.build:193-194, 210, 231`

**Interfaces:**
- Consumes: Task 3 CLIs. The Task 2 build context.
- Produces: `build.sh` honours `ARCADEDB_TAG` (default `latest`). Task 5's workflows set it.

- [ ] **Step 1: Make the changes**

| File | Change |
|---|---|
| `build.sh:151-154` | `DOCKER_TAG="${ARCADEDB_TAG:-latest}"`. Delete the "Detecting version from pom.xml" lines and print `Docker tag: $DOCKER_TAG`. |
| `build.sh:404` | Pass `ARCADEDB_VERSION="$(python3 "$SCRIPT_DIR/extract_version.py" "$LOCAL_JARS_DIR" --format=docker)"` instead of `$DOCKER_TAG`. |
| `build-native.sh:311` | `extract_version.py "$JARS_DIR" --format=pep440` |
| `build-native.sh:330` | `write_version.py "$JARS_DIR"` |
| `Dockerfile.build` | Delete `COPY pom.xml` and its comment. Change to `write_version.py /build/jars` and `extract_version.py --format=pep440 /build/jars`. The log text becomes "from engine jar". |

- [ ] **Step 2: Verify the build against `latest`, with no arcadedb checkout involved**

Run: `docker pull arcadedata/arcadedb:latest && ./scripts/build.sh linux/amd64 3.12`

Expected:
- The build succeeds and the Docker tester stage passes.
- `ls dist/` shows `arcadedb_embedded-<X.Y.Z>.dev0-*.whl`.
- `<X.Y.Z>-SNAPSHOT` equals the output of `docker run --rm --entrypoint ls arcadedata/arcadedb:latest /home/arcadedb/lib | grep arcadedb-engine`.

- [ ] **Step 3: Verify a pinned tag and the local-jars path**

Run each of these:
- `ARCADEDB_TAG=<the X.Y.Z-SNAPSHOT from step 2> ./scripts/build.sh linux/amd64 3.12`. Expected: the same wheel version.
- Copy the jars out of the image into `$SCRATCH/jars`, then run `./scripts/build.sh linux/amd64 3.12 "$SCRATCH/jars"`. Expected: the "Verifying embedded local integration JAR" step passes.

- [ ] **Step 4: Run the full test suite against the built wheel on the host**

Run: `python3 -m venv $SCRATCH/venv && $SCRATCH/venv/bin/pip install dist/*.whl ".[test]"` (if the extras resolve from the source tree, otherwise list them), then `$SCRATCH/venv/bin/pytest tests -q`. Expected: all pass. That includes `tests/test_import_database.py`, which reads from `tests/fixtures`, and `tests/test_extract_version.py`.

- [ ] **Step 5: Commit** — `build: default to arcadedata/arcadedb:latest, overridable via ARCADEDB_TAG`

---

### Task 5: Workflows and repository config

**Files:**
- Modify: `.github/workflows/test-python-bindings.yml`, `.github/workflows/test-python-examples.yml`
- Create: `.github/actionlint.yaml`, `.github/dependabot.yml`, `.pre-commit-config.yaml`, `REUSE.toml`

**Interfaces:**
- Consumes: `build.sh` reads `ARCADEDB_TAG` (Task 4). `extract_version.py JARS_DIR --format=docker` (Task 3).
- Produces: the `download-jars` job output `arcadedb-tag`, a concrete `X.Y.Z-SNAPSHOT`, which every later job passes to `build.sh` as `env: ARCADEDB_TAG`.

- [ ] **Step 1: Change both workflows**

**Triggers**
- `push` and `pull_request` with no `paths` filter.
- Keep `workflow_dispatch` and `workflow_call`. Keep the `build-version` input, but delete the `engine-from-source` inputs.
- Delete `workflow_run: Release`.
- Add `schedule: - cron: "0 3 * * *"` to `test-python-bindings.yml` and `"0 4 * * *"` to `test-python-examples.yml`.

**Paths**
- Delete every `working-directory: bindings/python` and every `cd bindings/python`.
- Rewrite `bindings/python/<x>` to `<x>`. This covers artifact paths and the examples glob description.

**`download-jars`**
- Delete the engine-from-source branch (`./mvnw ...`, `actions/setup-java`).
- `docker create arcadedata/arcadedb:latest`, copy the lib dir out, then `ARCADEDB_TAG=$(python3 scripts/extract_version.py src/arcadedb_embedded/jars --format=docker)`.
- Write `arcadedb-tag=$ARCADEDB_TAG` to `$GITHUB_OUTPUT` and declare it under `outputs:`.

**`test` and `test-examples`**
- Delete the `if engine-from-source` branch around `build.sh`.
- Set `env: ARCADEDB_TAG: ${{ needs.download-jars.outputs.arcadedb-tag }}`.

- [ ] **Step 2: Add the config files**

| File | Contents |
|---|---|
| `.github/actionlint.yaml` | The two workflow entries copied from upstream `.github/actionlint.yaml:20-25`. |
| `.github/dependabot.yml` | `version: 2`. Upstream's `uv` entry with `directory: "/"`, plus a weekly `github-actions` entry. Keep the upstream cooldown values for the uv entry. |
| `.pre-commit-config.yaml` | The upstream black, isort (`--profile black`), pretty-format-yaml (`--autofix --indent=2`) and shfmt hooks, with the same `rev`s. Each `files: ^bindings/python/...` regex becomes the same regex without the prefix. |
| `REUSE.toml` | `version = 1`, plus the upstream bindings annotation with paths rewritten to the root: `**/*.py`, `**/*.sh`, `scripts/Dockerfile.build`, `scripts/jar_exclusions.txt`, `pyproject.toml`, `setup.py`, `tests/fixtures/**`. |

- [ ] **Step 3: Verify**

Run each of these:
- `actionlint`. Expected: no errors. Install it with `brew install actionlint` if needed.
- `pre-commit run --all-files`. Expected: passes. If the yaml and shfmt autofix hooks rewrite moved files, commit those rewrites as a separate `style:` commit so the history stays readable.
- `git grep -n 'bindings/python\|engine-from-source\|mvnw\|pom.xml' .github`. Expected: no matches.

- [ ] **Step 4: Commit** — `ci: run the bindings workflows standalone against arcadedata/arcadedb:latest`

---

### Task 6: README and remaining references

**Files:**
- Modify: `README.md`, `pyproject.toml:6` (comment only)

- [ ] **Step 1: Update the docs**

Changes:
- **`README.md`:** drop the `cd bindings/python/` lines. Replace line 138's "extracted from the main ArcadeDB `pom.xml`" with: "taken from the `arcadedb-engine-<version>.jar` in the `arcadedata/arcadedb` image (`latest` by default, set `ARCADEDB_TAG` to pin)". Add a short "Origin" section: "Moved from ArcadeData/arcadedb `bindings/python` at `<EXTRACTED_SHA>`, with history."
- **`pyproject.toml:6`:** say "from the bundled engine jar". Leave the URLs and authors unchanged.

- [ ] **Step 2: Verify**

Run: `git grep -n -E 'bindings/python|pom\.xml' -- . ':!docs/superpowers'`. Expected: only `tests/test_check_test_skips.py:53`.

- [ ] **Step 3: Commit** — `docs: README for the standalone repository`

---

### Task 7: Push, CI, merge

- [ ] **Step 1: Push and open the PR**

Run `git push -u origin move-python-bindings`. Open a PR to `main` titled "Move Python bindings from ArcadeData/arcadedb (with history)". The body must state `EXTRACTED_SHA`, the freeze, and that the PR must be merged with a merge commit. End the body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

- [ ] **Step 2: Wait for both workflows to go green**

Run: `gh pr checks --watch`. Expected: every job passes, including the macOS and Windows native builds. When a job fails, use superpowers:systematic-debugging. Do not add skips, because `check_test_skips.py` forbids them.

- [ ] **Step 3: Merge**

Ask the user to confirm first. Then run `gh pr merge --merge`, never `--squash`. Afterwards, `git log --oneline origin/main | wc -l` must be at least 63.

- [ ] **Step 4: Prove that `main` and the manual trigger work**

Run: `gh workflow run test-python-bindings.yml --ref main && gh workflow run test-python-examples.yml --ref main`, then watch both with `gh run watch`. Expected: both green.

---

### Task 8: Upstream cleanup PR on ArcadeData/arcadedb

Start only after Task 7 step 4 is green. Opening this PR needs the user's go-ahead.

**Files (upstream):**
- Delete: `bindings/python/`, `.github/workflows/test-python-bindings.yml`, `.github/workflows/test-python-examples.yml`
- Modify:
  - `.github/workflows/mvn-test.yml`: the `bindings/python/**` paths-ignore entries.
  - `.github/workflows/mvn-deploy.yml`: its `paths-ignore` entry.
  - `.github/workflows/meterian.yml`: `--exclude-folders`, keeping `**/e2e-go`; trim the comment's bindings sentence.
  - `.github/dependabot.yml`: the "Bindings Python" uv entry.
  - `.github/actionlint.yaml`: the two entries.
  - `.codacy.yml`: `bindings/python/tests/**` and the comment reference.
  - `REUSE.toml`: the bindings annotation.
  - `.pre-commit-config.yaml`: the four bindings hooks and `bindings/python/mkdocs\.yml` in the yaml exclude.
  - `README.md`: point to `ArcadeData/arcadedb-python-bindings`.
- Check and edit if they match: `.github/scripts/check-gitignore-src-paths.sh`, `sbom.json`, root `.gitignore`.
- Leave alone: `e2e-python/`, `docs/**` (historical).

- [ ] **Step 1: Check the freeze**

Run, in a fresh worktree of upstream on a branch `chore/remove-python-bindings`: `git fetch origin && git log --oneline <EXTRACTED_SHA>..origin/main -- bindings/python`.

Expected: empty. If it is not empty, **stop**, port those commits here first, and tell the user.

- [ ] **Step 2: Make the deletions and edits listed above**

- [ ] **Step 3: Verify**

Run: `git grep -n -E 'bindings/python|arcadedb_embedded|arcadedb-embedded' -- . ':!docs' ':!e2e-python'`. Expected: no matches, except the README link to the new repo.

Also run `pre-commit run --all-files` and `actionlint`. Expected: pass.

- [ ] **Step 4: Commit, push and open the PR**

The PR body states `EXTRACTED_SHA`, links the merged PR here, and ends with the Claude Code line. Then wait for upstream CI to go green with `gh pr checks --watch`. Merging is left to the user.
