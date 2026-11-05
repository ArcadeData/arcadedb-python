---
name: Release
about: One monthly release of arcadedb-jvm and arcadedb (native), step by step
title: "Release <VERSION>"
labels: ""
---

Open this issue once per release, when upstream ArcadeDB has released an engine version. Replace the placeholders everywhere on this page before you start:

- `<VERSION>`: the tag to push, for example the engine release `A.B.C`. A Python-only fix of a released engine is `A.B.C.postN`.
- `<PREVIOUS>`: the previous release of the same package on PyPI (the JVM and the native package can differ), or `-` for the first release of a package.
- `<NEXT_DEV>`: the first dev version of the next base, `A.B.C.dev0` (see step 7).

Every step says who does it: script (run the command, read the result), CI (a workflow does it, you wait), or owner (a decision that is not automated). The procedure and the reasons are on the docs page `development/release.adoc`, section "Release day".

## Before the tag

- [ ] 1. script: upstream `<VERSION>` is out on both channels (GitHub release, Maven Central, Docker Hub image), the version rule holds for both packages, the CI of `main` is green, and the release workflows derive the right version and engine. Zero FAILs means ready. A FAIL on Maven Central or Docker Hub can mean the other channel is still catching up: run it again later.

  ```bash
  uv run --no-project python -I scripts/release_preflight.py <VERSION>
  ```

- [ ] 2. owner: for a final or post release, the native readiness bar is met and the repository variable `ARCADEDB_NATIVE_READINESS_BAR_MET` is `true` (the preflight line `(g)` shows it, or SKIPs when the login cannot read variables).
- [ ] 3. owner: the go. Nothing is published before this box is ticked. Before it, preview the GitHub Release notes and check that every pull request of this release has the milestone `<VERSION>` (a pull request without it is not listed in a `devN` or final release; a `postN` lists the commit range instead and needs no milestone):

  ```bash
  uv run --no-project python -I scripts/release_notes.py <VERSION> --dry-run
  ```

## The tag (the only publishing step)

- [ ] 4. owner: push the tag. It is the bare version, with no `v`, on the tip of `main`. One tag starts `release-jvm.yml` and `release-native.yml`.

  ```bash
  git fetch origin main --no-tags && git tag <VERSION> origin/main && git push origin <VERSION>
  ```

- [ ] 5. CI: both release workflows finish and publish, then `release-notes.yml` creates the GitHub Release and, for a final release, closes the milestone. Note the two run ids (the native build takes the longest). To retry a failed upload, re-run only the publish job, never the whole workflow.

  ```bash
  gh run list -R ArcadeData/arcadedb-python --branch <VERSION> --limit 6
  gh run watch <run-id> -R ArcadeData/arcadedb-python
  ```

## After the tag

- [ ] 6. script: verify both packages on PyPI. Run them under bash. The scripts read from GitHub and PyPI only, and download about 0.6 GB: point `WORK` at a disk, not a tmpfs. The exit code is the number of FAILs.

  ```bash
  scripts/after_tag_verify.sh <VERSION> <PREVIOUS> <jvm-run-id>
  PACKAGE=native scripts/after_tag_verify.sh <VERSION> <PREVIOUS> <native-run-id>
  ```

- [ ] 7. script, then CI: the dev-engine bump PR. Open it from a worktree on a branch off `origin/main`. Pick `<NEXT_DEV>` from the version of upstream `main`: the command prints the next base. If it prints `<VERSION>`, upstream has not moved on yet: wait, because a dev version of a released base is refused.

  ```bash
  curl -fsS https://raw.githubusercontent.com/ArcadeData/arcadedb/main/pom.xml | sed -n '/<artifactId>arcadedb-parent<\/artifactId>/,/<\/version>/ s/.*<version>\(.*\)-SNAPSHOT<\/version>.*/\1/p'
  # <NEXT_DEV> is that base followed by .dev0
  uv run --no-project python scripts/bump_engine.py --engine <VERSION> --dev-version <NEXT_DEV> --check
  uv run --no-project python scripts/bump_engine.py --engine <VERSION> --dev-version <NEXT_DEV>
  ```

- [ ] 8. script: refresh the notices the new engine needs. First refresh the kept engine NOTICE (`git -C <arcadedb clone> show <VERSION>:NOTICE > native/wheel/ARCADEDB_ENGINE_NOTICE`; the generated notices embed it). The first `Test native` run on the bump PR stops at the notices drift check on purpose and uploads the generated file. Download it, commit it, push, and wait for the run to go green.

  ```bash
  gh run list -R ArcadeData/arcadedb-python --workflow test-native.yml --branch <bump-branch> --limit 1
  gh run download <run-id> -R ArcadeData/arcadedb-python -n native-build-record-linux-x86_64 -D <scratch-dir>
  cp <scratch-dir>/THIRD_PARTY_NOTICES.generated.md native/wheel/THIRD_PARTY_NOTICES.md
  uv run --no-project python scripts/bump_engine.py --engine <VERSION> --dev-version <NEXT_DEV> --check
  ```


- [ ] 9. CI, then owner: merge the bump PR when every check is green, and delete the branch and the worktree. Then remove the tests and skip-list lines that waited for the new engine: `git grep -n -i "native build links\|native image links"`.
- [ ] 10. owner: check the GitHub Release (notes, pre-release flag) and, after a final release, that the milestone is closed and what was still open in it moved to the next milestone (the open one due next, or one the workflow created named after the next upstream engine version). If the run failed with a message about the next milestone, create it with a due date and re-run `GitHub Release` (Actions, `workflow_dispatch`, the tag). Close this issue with the two run ids and the two PyPI links.
