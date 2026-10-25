#!/bin/bash
# After the tag push: verify the release end to end. Read-only against GitHub and PyPI (no writes).
# Ported from humemai/arcadedb-embedded-python#479 and adapted to .github/workflows/release.yml.
#
# usage: after_tag_verify.sh <version> <previous-version|-> [<release-workflow-run-id>] [<expected-engine-jar-sha256> <expected-engine-buildNumber>]
# example: after_tag_verify.sh 26.11.1 26.10.1 <run-id>
#
# <previous-version> is the release whose PyPI file set this one must match in shape. Pass - when
# there is none (the first release of the distribution): the file set is then checked against
# pyproject.toml, one wheel per supported Python version and platform.
#
# Run it under bash (not zsh). Every step prints PASS or FAIL; the exit code is the number of FAILs.
# Set PIN=4-5 to pin the smoke runs to those cores. It downloads about 4 GB (the 20 checked wheels of the run and four from PyPI): TMPDIR or WORK should point at a disk, not a tmpfs. Needs: gh, curl, python3, uv.
set -uo pipefail
V="${1:?version}"
PREV="${2:?previous version}"
RUN="${3:-}"
ENG_SHA="${4:-}"
ENG_BUILD="${5:-}"
R=ArcadeData/arcadedb-python
DIST=arcadedb-jvm
W="${WORK:-$(mktemp -d "${TMPDIR:-$HOME}/after-tag-XXXXXX")}"
FAILS=0
pass() { printf 'PASS  %s\n' "$*"; }
fail() {
    printf 'FAIL  %s\n' "$*"
    FAILS=$((FAILS + 1))
}
mkdir -p "$W"
printf 'work dir: %s\n' "$W"

# (a) the release workflow, and the publish JOB's own conclusion (not only the run's: read what the upload step did)
if [ -n "$RUN" ]; then
    gh run view "$RUN" -R "$R" --json status,conclusion,headBranch,headSha,createdAt,updatedAt > "$W/run.json" 2>&1 || true
    python3 - "$W/run.json" "$V" << 'EOF' || FAILS=$((FAILS + 1))
import json,sys
d=json.load(open(sys.argv[1]))
ok = d.get("status")=="completed" and d.get("conclusion")=="success" and d.get("headBranch")==sys.argv[2]
print(("PASS" if ok else "FAIL"), " release run:", d.get("status"), d.get("conclusion"), "ref", d.get("headBranch"), "sha", (d.get("headSha") or "")[:12], d.get("createdAt"), "->", d.get("updatedAt"))
sys.exit(0 if ok else 1)
EOF
    gh api "repos/$R/actions/runs/$RUN/jobs" --paginate --jq '.jobs[] | select(.name|startswith("Publish")) | [.name,.status,.conclusion,(.steps|map(select(.name=="Publish to PyPI"))|.[0].conclusion)]|@tsv' > "$W/publish-job.tsv" 2>&1
    cat "$W/publish-job.tsv"
    if awk -F'\t' '$2=="completed" && $3=="success" && $4=="success"{ok=1} END{exit ok?0:1}' "$W/publish-job.tsv"; then pass "publish job and its 'Publish to PyPI' step both succeeded"; else fail "publish job or its upload step did not succeed"; fi
fi

# (b) PyPI file set against the previous release: the same files as the previous release with the new version, minus wheels for Python versions the release dropped, each under 100 MB (decimal). With "-" as the previous version: one wheel per Python version of the pyproject.toml classifiers and per platform
curl -fsS "https://pypi.org/pypi/$DIST/$V/json" -o "$W/pypi-new.json" || fail "PyPI has no $V yet"
if [ "$PREV" = "-" ]; then
    echo '{"urls": []}' > "$W/pypi-prev.json"
else
    curl -fsS "https://pypi.org/pypi/$DIST/$PREV/json" -o "$W/pypi-prev.json" || fail "cannot read PyPI $PREV"
fi
python3 - "$W/pypi-new.json" "$W/pypi-prev.json" "$V" "$PREV" "$(dirname "$0")/../pyproject.toml" << 'EOF' || FAILS=$((FAILS + 1))
import json,re,sys,tomllib
new,prev,v,pv=json.load(open(sys.argv[1])),json.load(open(sys.argv[2])),sys.argv[3],sys.argv[4]
project=tomllib.load(open(sys.argv[5],"rb"))["project"]
# A previous release may carry wheels for an interpreter this release dropped: they are not expected again.
floor=int(re.search(r">=\s*3\.(\d+)",project["requires-python"]).group(1))
def kept(f):
    m=re.search(r"-cp3(\d+)-",f)
    return m is None or int(m.group(1))>=floor
want={u["filename"].replace(pv,v) for u in prev["urls"] if kept(u["filename"])}
have={u["filename"]:u for u in new["urls"]}
gaps=0
if pv=="-":
    # No release to compare with: one wheel per supported Python version and platform, as release.yml counts them.
    pythons=[c.rsplit(".",1)[1] for c in project["classifiers"] if c.startswith("Programming Language :: Python :: 3.")]
    plats=[r"manylinux.*x86_64",r"manylinux.*aarch64",r"macosx.*arm64",r"win_amd64"]
    for py in pythons:
        for plat in plats:
            hits=[f for f in have if re.search(r"-cp3%s-"%py,f) and re.search(plat,f)]
            want.update(hits)
            if len(hits)!=1:
                print("EXPECTED one wheel for Python 3.%s on %s, got %d"%(py,plat,len(hits))); gaps+=1
missing=sorted(want-set(have)); extra=sorted(set(have)-want)
print("files on PyPI for %s: %d (previous release had %d)"%(v,len(have),len(prev["urls"])))
for m in missing: print("MISSING",m)
for e in extra: print("EXTRA  ",e)
big=[f for f,u in have.items() if u["size"]>=100_000_000]
yanked=[f for f,u in have.items() if u["yanked"]]
for f,u in sorted(have.items()): print("  %-62s %9.2f MB  %s"%(f,u["size"]/1e6,u["digests"]["sha256"][:16]))
ok=not missing and not extra and not gaps and not big and not yanked and new["info"]["version"]==v
print(("PASS" if ok else "FAIL"),"file set as expected (previous release, or pyproject.toml without one), all under 100 MB, none yanked")
sys.exit(0 if ok else 1)
EOF

# (c) sha256 of every file on PyPI against the checked wheels the workflow's verify job uploaded and the publish job sent
if [ -n "$RUN" ] && [ -s "$W/pypi-new.json" ]; then
    mkdir -p "$W/artifacts"
    gh run download "$RUN" -R "$R" -n release-dist -D "$W/artifacts" > /dev/null 2>&1 || fail "could not download the release-dist artifact of run $RUN (retention is 7 days)"
    python3 - "$W/pypi-new.json" "$W/artifacts" << 'EOF' || FAILS=$((FAILS + 1))
import json,sys,hashlib,pathlib
d=json.load(open(sys.argv[1])); art={}
for p in pathlib.Path(sys.argv[2]).rglob("*.whl"):
    art[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
bad=0
for u in d["urls"]:
    a=art.get(u["filename"])
    if a is None: print("NO ARTIFACT for",u["filename"]); bad+=1
    elif a!=u["digests"]["sha256"]: print("SHA MISMATCH",u["filename"]); bad+=1
print(("PASS" if not bad else "FAIL"),"%d PyPI files compared with %d workflow artifacts"%(len(d["urls"]),len(art)))
sys.exit(1 if bad else 0)
EOF
fi

# (d) every platform resolves, and a fresh environment installs and works (3.12 and 3.14)
for plat in manylinux_2_34_x86_64 manylinux_2_34_aarch64 macosx_11_0_arm64 win_amd64; do
    if uvx pip download --no-deps --only-binary=:all: --platform "$plat" --python-version 3.12 "$DIST==$V" -d "$W/dl-$plat" > "$W/dl-$plat.log" 2>&1; then pass "pip download $plat (cp312)"; else fail "pip download $plat: $(tail -1 "$W/dl-$plat.log")"; fi
done
for py in 3.12 3.14; do
    [ -n "${SKIP_SMOKE:-}" ] && {
        printf "SKIP  smoke on Python %s (SKIP_SMOKE set)\n" "$py"
        continue
    }
    # A throwaway `uv run` environment with the PyPI release; the fresh cache directory makes the
    # install download from PyPI (what `--no-cache` did), and the smoke run below reuses it.
    UVRUN=(uv run --no-project --cache-dir "$W/uv-cache-$py" --python "$py" --with "$DIST==$V")
    "${UVRUN[@]}" python -c "import arcadedb" > "$W/install-$py.log" 2>&1 || {
        fail "uv run --with $DIST==$V on $py: $(tail -1 "$W/install-$py.log")"
        continue
    }
    ENG_SHA="$ENG_SHA" ENG_BUILD="$ENG_BUILD" V="$V" ${PIN:+taskset -c "$PIN"} "${UVRUN[@]}" python - << 'EOF' > "$W/smoke-$py.log" 2>&1
import hashlib, os, pathlib, re, tempfile, zipfile
import arcadedb as adb
assert adb.__version__ == os.environ["V"], adb.__version__
jars = pathlib.Path(adb.__file__).parent / "jars"
eng = sorted(jars.glob("arcadedb-engine-*.jar"))
# The engine jar is named after the engine, not the wheel: a Python-only hotfix X.Y.Z.postN ships the
# X.Y.Z jar, and a snapshot build X.Y.Z.devN ships the X.Y.Z-SNAPSHOT jar (docs, Release page).
engine = re.sub(r"\.dev[0-9]+$", "-SNAPSHOT", re.sub(r"\.post[0-9]+$", "", os.environ["V"]))
assert len(eng) == 1 and eng[0].name == "arcadedb-engine-%s.jar" % engine, eng
sha = hashlib.sha256(eng[0].read_bytes()).hexdigest()
props = zipfile.ZipFile(eng[0]).read("com/arcadedb/arcadedb.properties").decode()
build = [l.split("=", 1)[1].strip() for l in props.splitlines() if l.startswith("buildNumber")][0]
print("engine jar", eng[0].name, sha, "buildNumber", build)
if os.environ["ENG_SHA"]: assert sha == os.environ["ENG_SHA"], "engine jar is not the official one"
if os.environ["ENG_BUILD"]: assert build == os.environ["ENG_BUILD"], "buildNumber differs"
with tempfile.TemporaryDirectory() as d:
    with adb.create_database(os.path.join(d, "smoke"), jvm_kwargs={"heap_size": "1g"}) as db:
        db.command("sql", "CREATE VERTEX TYPE P")
        with db.transaction():
            db.command("sql", "CREATE VERTEX P SET name = 'Ada'")
        assert db.query("sql", "SELECT name FROM P").to_list()[0]["name"] == "Ada"
        assert db.query("opencypher", "MATCH (n:P) RETURN n.name AS name").to_list()[0]["name"] == "Ada"
print("SMOKE OK", adb.__version__)
EOF
    if [ $? -eq 0 ] && grep -q 'SMOKE OK' "$W/smoke-$py.log"; then
        pass "fresh environment Python $py: install, import, SQL, openCypher, official engine jar"
        grep 'engine jar' "$W/smoke-$py.log"
    else
        fail "smoke on Python $py (see $W/smoke-$py.log)"
        tail -3 "$W/smoke-$py.log"
    fi
done

# There is no docs step: the Antora component is unversioned and builds in ArcadeData/arcadedb-docs, so a release changes no docs URL to check.

printf '\n%d FAIL(s)\n' "$FAILS"
exit "$FAILS"
