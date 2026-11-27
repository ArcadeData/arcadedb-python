"""The legs of the native test workflow per event: a pull request must stay small (the org shares a runner cap with upstream)."""

import fnmatch
import importlib.util
import json
import re
import subprocess  # nosec B404 - runs this repository's script with sys.executable
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_native_matrix.py"
spec = importlib.util.spec_from_file_location("select_native_matrix", SCRIPT)
sel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sel)


def test_pull_request_builds_one_platform_and_tests_one_python():
    chosen = sel.select("pull_request")
    assert [b["key"] for b in chosen["build"]] == ["linux-x86_64"]
    # one Python, the newest the full suite runs on (#517), in shards that run side by side, every one from 1 to N
    count = sel.PULL_REQUEST_SHARDS
    assert count > 1
    assert [
        (t["key"], t["python"], t["shard"], t["suffix"]) for t in chosen["test"]
    ] == [
        ("linux-x86_64", "3.14", f"{i}/{count}", f"-shard{i}")
        for i in range(1, count + 1)
    ]
    assert chosen["pyver"] == [] and chosen["examples"] == []


def test_only_the_fast_tier_is_sharded():
    for event in (
        "pull_request_full",
        "push",
        "schedule",
        "workflow_dispatch",
        "release",
    ):
        assert all("shard" not in leg for leg in sel.select(event)["test"]), event


def test_a_pull_request_that_asks_for_the_full_tier_runs_the_legs_of_a_push_to_main():
    # scripts/ci_tier.py passes pull_request_full for the label ci:full and for a Dependabot pull request
    assert sel.select("pull_request_full") == sel.select("push")


def test_the_workflow_asks_ci_tier_for_the_event_and_skips_an_empty_python_version_job():
    workflow = (ROOT / ".github" / "workflows" / "test-native.yml").read_text()
    plan = _job(workflow, "plan")
    tier = plan.index('event="$(python3 -I scripts/ci_tier.py "$EVENT")"')
    assert tier < plan.index('scripts/select_native_matrix.py "$event"')
    # a change to the script that picks the tier starts this workflow
    assert '      - "scripts/ci_tier.py"\n' in workflow.split("\njobs:\n", 1)[0]
    # the fast tier has no example leg, and an empty matrix is an error: the job is skipped then, and the summary with it
    assert sel.select("pull_request")["examples"] == []
    for name in ("examples", "examples-summary"):
        assert (
            "needs.plan.outputs.examples != '[]'"
            in _job(workflow, name).split("steps:", 1)[0]
        ), name
    assert '[ "$RESULT" = success ]' in _job(workflow, "examples-summary")
    # the same guard on the pyver job, which no other job needs
    assert (
        "needs.plan.outputs.pyver != '[]'"
        in _job(workflow, "pyver").split("steps:", 1)[0]
    )
    for needs in re.findall(r"^    needs: (.*)$", workflow, re.M):
        assert "pyver" not in needs, needs


def test_main_nightly_and_dispatch_build_all_four_platforms():
    for event in ("push", "schedule", "workflow_dispatch"):
        chosen = sel.select(event)
        assert [b["key"] for b in chosen["build"]] == [
            "linux-x86_64",
            "linux-aarch64",
            "macos-arm64",
            "windows-x64",
        ]
        assert len(chosen["test"]) == 4 * len(sel.MAIN_PYTHONS)


def test_a_release_tests_every_supported_python_on_every_platform():
    chosen = sel.select("release")
    assert len(chosen["build"]) == 4
    assert len(chosen["test"]) == 4 * 4
    assert {t["python"] for t in chosen["test"]} == {"3.11", "3.12", "3.13", "3.14"}


def test_pythons_are_the_classifiers_of_the_native_wheel():
    text = tomllib.loads((ROOT / "native" / "wheel" / "pyproject.toml").read_text())
    prefix = "Programming Language :: Python :: 3."
    pythons = [
        "3." + c.rsplit(".", 1)[1]
        for c in text["project"]["classifiers"]
        if c.startswith(prefix)
    ]
    assert pythons == sel.ALL_PYTHONS + sel.PYVER_ONLY_PYTHONS


def test_graalvm_is_community_edition_with_a_sha256_per_platform():
    for key, leg in sel.PLATFORMS.items():
        assert "graalvm-community" in leg["graalvm_asset"], key
        assert re.fullmatch(r"[0-9a-f]{64}", leg["graalvm_sha256"]), key
        assert "oracle" not in leg["graalvm_asset"].lower(), key


def test_the_script_prints_github_output_lines():
    out = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, str(SCRIPT), "pull_request"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert [line.split("=", 1)[0] for line in out] == [
        "build",
        "test",
        "pyver",
        "examples",
    ]
    assert json.loads(out[0].split("=", 1)[1])[0]["key"] == "linux-x86_64"
    # the fast tier: no pyver leg and no example leg (the text the workflow compares with '[]')
    assert out[2] == "pyver=[]"
    assert out[3] == "examples=[]"
    full = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, str(SCRIPT), "pull_request_full"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert [leg["python"] for leg in json.loads(full[2].split("=", 1)[1])] == [
        "3.11",
        "3.13",
        "3.15",
        "3.14t",
    ]


def _pyver(event):
    return [
        (leg["python"], leg["freethreaded"], leg["polars"], leg["jpype"])
        for leg in sel.select(event)["pyver"]
    ]


def test_the_python_version_job_covers_what_the_full_parity_run_leaves_out():
    # a pull request (the fast tier) runs one Python, the parity suite on 3.14: every other Python waits for the full tier
    assert _pyver("pull_request") == []
    # the selection itself still knows what one full Python leaves out: the other three Pythons, 3.15 (no jpype1 wheel yet) and the
    # free-threaded one
    assert [
        (leg["python"], leg["freethreaded"], leg["polars"], leg["jpype"])
        for leg in sel.pyver_legs(sel.PULL_REQUEST_PYTHONS)
    ] == [
        ("3.11", False, True, True),
        ("3.12", False, True, True),
        ("3.13", False, True, True),
        ("3.15", False, True, False),
        ("3.14t", True, False, True),
    ]
    # main, nightly, dispatch and a pull request that asks for the full tier test 3.12 and 3.14 in full
    for event in ("pull_request_full", "push", "schedule", "workflow_dispatch"):
        assert _pyver(event) == [
            ("3.11", False, True, True),
            ("3.13", False, True, True),
            ("3.15", False, True, False),
            ("3.14t", True, False, True),
        ]
    # a release tests 3.11 to 3.14 in full: 3.15 and the free-threaded build are left
    assert _pyver("release") == [
        ("3.15", False, True, False),
        ("3.14t", True, False, True),
    ]


def test_the_python_version_job_never_repeats_a_full_parity_leg_and_has_no_polars_on_a_free_threaded_python():
    for event in (
        "pull_request_full",
        "push",
        "schedule",
        "workflow_dispatch",
        "release",
    ):
        chosen = sel.select(event)
        full = {t["python"] for t in chosen["test"]}
        assert not full & {leg["python"] for leg in chosen["pyver"]}, event
        assert any(leg["freethreaded"] for leg in chosen["pyver"]), event
        for leg in chosen["pyver"]:
            assert leg["polars"] is (not leg["freethreaded"]), leg


def _job(text, name):
    """The text of one job of a workflow file (a job starts at two spaces of indentation)."""
    parts = re.split(
        r"^  ([A-Za-z0-9_-]+):\n", text.split("\njobs:\n", 1)[1], flags=re.M
    )
    return dict(zip(parts[1::2], parts[2::2]))[name]


def test_a_release_needs_a_marker_from_every_leg_of_the_python_version_job():
    # ArcadeData/arcadedb-python#349: the pyver job was informational (a red leg could not stop a release); now a leg that passes uploads a
    # marker and verify in release-native.yml requires the markers of the legs a release has. The bash of verify is run in
    # tests/test_release_native_verify.py; here the two workflow files and the selector are checked against one another.
    workflows = ROOT / ".github" / "workflows"
    test_native = (workflows / "test-native.yml").read_text()
    release = (workflows / "release-native.yml").read_text()
    pyver = _job(test_native, "pyver")
    # the marker is written and uploaded after the gate step and by steps that run on success only, and a missing file fails the upload
    gate = pyver.find("Gate: no failure, error, crash, timeout or uncounted skip")
    mark = pyver.find("- name: Mark this leg as passed")
    assert 0 <= gate < mark, "the pyver job has no marker step after its gate"
    upload = re.search(
        r"      - uses: actions/upload-artifact@\S+ # v[0-9.]+\n"
        r"        with:\n"
        r"          name: native-passed-pyver-py\$\{\{ matrix\.python \}\}\n"
        r"          path: \$\{\{ runner\.temp \}\}/passed/pyver-py\$\{\{ matrix\.python \}\}\.ok\n"
        r"          if-no-files-found: error\n",
        pyver,
    )
    assert upload, "the pyver job uploads no marker native-passed-pyver-py<version>"
    assert gate < mark < upload.start()
    marker_step = pyver[mark : upload.start()]
    assert "if:" not in marker_step and 'pyver-py$PY.ok"' in marker_step
    # verify requires the marker of every leg the selector gives a release, under the file name the upload has
    chosen = re.search(r'^\s+pyver_pythons="([^"]*)"$', release, re.M)
    assert chosen, "verify names no pyver legs"
    assert chosen.group(1).split() == [
        leg["python"] for leg in sel.select("release")["pyver"]
    ]
    assert '"passed/pyver-py$py.ok"' in release
    # the marker artifact is downloaded by a pattern verify already has; the artifacts of the parity runs of this job are not
    patterns = re.findall(r"^\s+pattern: (.*)$", release, re.M)
    assert patterns and all(
        p in ("native-passed-*", "native-wheel-*") for p in patterns
    ), patterns
    assert fnmatch.fnmatch("native-passed-pyver-py3.14t", "native-passed-*")
    # nothing calls the job informational any more
    assert "informational" not in sel.PYVER_NOTE.lower()
    assert "It is informational" not in test_native
    assert "INFORMATIONAL for now" not in test_native


def test_the_weekly_engine_snapshot_run_calls_the_native_workflow_once_a_week_and_is_reported():
    workflows = ROOT / ".github" / "workflows"
    weekly = (workflows / "test-native-snapshot.yml").read_text(encoding="utf-8")
    crons = re.findall(r'^\s+- cron: "([^"]+)"', weekly, re.M)
    # one schedule, on one day of the week: the runners are shared with upstream
    assert len(crons) == 1 and crons[0].split()[4].isdigit(), crons
    assert "uses: ./.github/workflows/test-native.yml" in weekly
    assert re.search(r"^      engine-snapshot: true$", weekly, re.M)
    # a called workflow starts no workflow_run of its own, so the report listens for the name of the caller
    name = re.search(r"^name: (.+)$", weekly, re.M).group(1)
    report = (workflows / "report-red-main.yml").read_text(encoding="utf-8")
    assert f'\n      - "{name}"\n' in report


def test_the_linux_legs_build_in_the_manylinux_container_of_the_tag_and_the_workflow_gates_that_floor():
    linux = {k: leg for k, leg in sel.PLATFORMS.items() if leg["kind"] == "linux"}
    assert set(linux) == {"linux-x86_64", "linux-aarch64"}
    for key, leg in linux.items():
        arch = key.split("-", 1)[1]
        assert leg["container"] == f"quay.io/pypa/manylinux_2_28_{arch}:latest", key
        assert leg["plat_re"] == f"manylinux_2_28_{arch}", key
    workflow = (ROOT / ".github" / "workflows" / "test-native.yml").read_text()
    assert "GLIBC_2.28" in workflow
    assert "GLIBC_2.34" not in workflow


def test_the_examples_job_runs_on_the_parity_pythons_3_12_and_3_14_of_every_platform_it_builds():
    # a pull request of the fast tier has no example leg
    assert sel.select("pull_request")["examples"] == []
    for event in (
        "pull_request_full",
        "push",
        "schedule",
        "workflow_dispatch",
        "release",
    ):
        legs = sel.select(event)["examples"]
        assert {leg["key"] for leg in legs} == set(sel.ALL_PLATFORMS), event
        assert {leg["python"] for leg in legs} == {"3.12", "3.14"}, event
        assert len(legs) == 2 * len(sel.ALL_PLATFORMS), event


def _step(job, name):
    """The text of one named step of a job."""
    return job.split(f"      - name: {name}\n", 1)[1].split("\n      - ", 1)[0]


def test_only_a_pull_request_of_the_fast_tier_restores_the_library_and_a_release_never_touches_the_cache():
    workflow = (ROOT / ".github" / "workflows" / "test-native.yml").read_text()
    plan = _job(workflow, "plan")
    assert (
        'if [ "$EVENT" = pull_request ] && [ "$event" = pull_request ] && [ "${ENGINE_SNAPSHOT:-}" != true ]; then fast_pr=true;'
        in plan
    )
    build = _job(workflow, "build")
    restore = _step(build, "Restore the native library")
    assert "needs.plan.outputs.fast-pr == 'true'" in restore
    assert "!endsWith(needs.plan.outputs.maven-engine, '-SNAPSHOT')" in restore
    assert "restore-keys" not in restore.split("uses:", 1)[1]
    save = _step(build, "Save the native library")
    assert "!inputs.release" in save
    assert "!endsWith(needs.plan.outputs.maven-engine, '-SNAPSHOT')" in save
    # restore and save use the one key and the one pair of directories
    for step in (restore, save):
        assert "key: ${{ steps.library-key.outputs.key }}" in step
        assert "${{ runner.temp }}/lib\n" in step
        assert "${{ runner.temp }}/m2\n" in step
    # a cache hit skips the image build and nothing else of the gates; the library is saved after every gate
    assert "steps.library.outputs.cache-hit != 'true'" in _step(
        build, "Build the native image in manylinux_2_28"
    )
    assert build.index("Gate: twine check") < build.index("Save the native library")
    assert workflow.count("actions/cache/restore@") == 1
    assert workflow.count("actions/cache/save@") == 1


def test_the_key_of_the_cached_library_covers_every_file_the_image_is_built_from():
    # native/wheel/ only packs the wheel, which every run builds from its own checkout; everything else under native/ decides
    # what the image is
    workflow = (ROOT / ".github" / "workflows" / "test-native.yml").read_text()
    key = _step(_job(workflow, "build"), "Key of the native library")
    hashed = key.split("hashFiles(", 1)[1].split(")", 1)[0]
    patterns = re.findall(r"'([^']+)'", hashed)
    for part in (
        "matrix.key",
        "needs.plan.outputs.maven-engine",
        "matrix.graalvm_sha256",
        "matrix.xmx",
        "matrix.ni_extra",
    ):
        assert part in key, part
    assert ".github/workflows/test-native.yml" in patterns
    native = ROOT / "native"
    sources = [native / "pom.xml", native / "mvnw"] + [
        p
        for d in (".mvn", "src", "scripts")
        for p in (native / d).rglob("*")
        if p.is_file()
    ]
    for path in sources:
        name = path.relative_to(ROOT).as_posix()
        assert any(
            name == p or (p.endswith("/**") and name.startswith(p[:-2]))
            for p in patterns
        ), name
    # a new directory or file directly under native/ must be added to the key (or to this list, when the build does not read it)
    known = {".gitignore", ".mvn", "mvnw", "mvnw.cmd", "pom.xml", "scripts", "src"}
    built = {"target", "reports"}
    assert {p.name for p in native.iterdir()} - built <= known | {"wheel"}


def test_the_examples_job_installs_the_wheel_of_the_build_job_and_shares_the_runner_with_the_jvm_job():
    workflows = ROOT / ".github" / "workflows"
    test_native = (workflows / "test-native.yml").read_text()
    jvm = (workflows / "test-python-examples.yml").read_text()
    assert re.search(r"^  examples:\n", test_native, re.M)
    # one runner for both packages: neither workflow carries the per-example arguments any more
    assert "scripts/run_examples.sh run" in test_native
    assert "scripts/run_examples.sh run" in jvm
    assert "21_graph_analytical_view_sql.py" not in test_native + jvm
    runner = (ROOT / "scripts" / "run_examples.sh").read_text()
    assert "21_graph_analytical_view_sql.py" in runner
    # the wheel is the one the build job uploaded (no rebuild), installed with uv
    assert "native-wheel-${{ matrix.key }}" in test_native
    assert "uv pip" not in test_native


def test_the_skip_list_of_the_examples_job_names_a_reason_and_an_example_that_exists():
    skip = ROOT / "scripts" / "native_examples_skip.txt"
    for line in skip.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, sep, reason = (part.strip() for part in line.partition("|"))
        assert sep and reason, line
        assert (ROOT / "examples" / name).is_file(), line
