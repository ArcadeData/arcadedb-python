"""The engine snapshot of a .devN release (scripts/dev_engine_pin.py, scripts/dev_engine_pin.json).

The rule: only A.B.C.devN of the pinned base may link a snapshot engine, and then only the pinned one. A final or .postN
release refuses a snapshot. These tests run the real `run:` code of the plan step of test-native.yml (bash) for every
case that needs no network, and call the pure functions of the script for the rest.
This file is not collected on Windows (tests/conftest.py): the step code is bash.
"""

import copy
import hashlib
import json
import subprocess  # nosec B404 - runs bash on a workflow step with a fixed environment
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import dev_engine_pin as pin_mod  # noqa: E402

PIN = pin_mod.load_pin()


def _plan_step() -> str:
    lines = (
        (ROOT / ".github/workflows/test-native.yml")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    start = next(
        i
        for i, ln in enumerate(lines)
        if ln.strip() == "- name: Resolve the version and the engine"
    )
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[run + 1]) - len(lines[run + 1].lstrip())
    body = []
    for ln in lines[run + 1 :]:
        if ln.strip() and len(ln) - len(ln.lstrip()) < indent:
            break
        body.append(ln[indent:])
    return "\n".join(body) + "\n"


def _run_plan(tmp_path, version, engine="", release="true", maven_engine=""):
    out = tmp_path / "out"
    out.write_text("")
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "INPUT_VERSION": version,
        "INPUT_ENGINE": engine,
        "INPUT_RELEASE": release,
        "INPUT_MAVEN_ENGINE": maven_engine,
        "DEV_ENGINE": "26.11.1",
        "DEFAULT_VERSION": "26.12.1.dev0",
        "GITHUB_OUTPUT": str(out),
    }
    r = subprocess.run(  # nosec B603 B607
        ["bash", "-c", _plan_step()],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r, out.read_text()


def test_pin_file_is_well_formed():
    assert PIN["engine"] == "26.12.1-SNAPSHOT"
    assert PIN["snapshot"].startswith("26.12.1-") and PIN["image_digest"].startswith(
        "sha256:"
    )
    assert "arcadedb-engine" in PIN["jars"] and "arcadedb-server" in PIN["jars"]


@pytest.mark.parametrize(
    "tag", ["26.12.1", "26.12.1.post1", "26.11.1.dev0", "26.12.1rc1", "v26.12.1.dev0"]
)
def test_only_a_dev_tag_of_the_pinned_base_may_link_the_snapshot(tag):
    assert pin_mod.tag_problem(tag, PIN) is not None


@pytest.mark.parametrize("tag", ["26.12.1.dev0", "26.12.1.dev7"])
def test_a_dev_tag_of_the_pinned_base_may(tag):
    assert pin_mod.tag_problem(tag, PIN) is None


@pytest.mark.parametrize("tag", ["26.12.1", "26.12.1.post1"])
def test_check_refuses_a_final_or_post_tag_before_any_network(tag, capsys):
    assert pin_mod.main(["check", "--tag", tag]) == 1
    assert "never a snapshot" in capsys.readouterr().out


@pytest.mark.parametrize("version", ["26.12.1", "26.12.1.post1"])
def test_release_plan_refuses_a_snapshot_for_a_final_or_post_version(tmp_path, version):
    r, _ = _run_plan(tmp_path, version, engine="26.12.1-SNAPSHOT")
    assert r.returncode != 0
    assert "links engine 26.12.1, not 26.12.1-SNAPSHOT" in r.stderr


def test_release_plan_of_a_final_version_links_its_release_engine(tmp_path):
    r, out = _run_plan(tmp_path, "26.12.1")
    assert r.returncode == 0, r.stderr
    assert "engine=26.12.1\n" in out and "SNAPSHOT" not in out


def test_release_plan_refuses_another_snapshot_for_a_dev_version(tmp_path):
    r, _ = _run_plan(tmp_path, "26.12.1.dev0", engine="26.12.2-SNAPSHOT")
    assert r.returncode != 0
    assert "links the pinned engine 26.12.1-SNAPSHOT, not 26.12.2-SNAPSHOT" in r.stderr


def test_release_plan_refuses_a_dev_version_of_another_base(tmp_path):
    r, _ = _run_plan(tmp_path, "26.13.1.dev0")
    assert r.returncode != 0
    assert "another base than the pinned engine" in r.stderr


def test_a_manual_dev_run_may_still_link_any_snapshot(tmp_path):
    r, out = _run_plan(
        tmp_path, "26.12.1.dev0", engine="26.12.2-SNAPSHOT", release="false"
    )
    assert r.returncode == 0, r.stderr
    assert "engine=26.12.2-SNAPSHOT\n" in out


def test_a_manual_final_run_still_refuses_a_snapshot(tmp_path):
    r, _ = _run_plan(tmp_path, "26.12.1", engine="26.12.1-SNAPSHOT", release="false")
    assert r.returncode != 0


def _hashes(**changes):
    h = copy.deepcopy(PIN["jars"])
    h.update(changes)
    return h


def _maven(**changes):
    m = {k: PIN["jars"][k] for k in ("arcadedb-engine", "arcadedb-server")}
    m.update(changes)
    return m


def test_compare_passes_when_both_channels_hold_the_pin():
    assert (
        pin_mod.compare(PIN, {"amd64": _hashes(), "arm64": _hashes()}, _maven()) == []
    )


def test_compare_names_an_image_that_moved():
    bad = pin_mod.compare(
        PIN,
        {"amd64": _hashes(), "arm64": _hashes(**{"arcadedb-engine": "0" * 64})},
        _maven(),
    )
    assert any("image arm64: arcadedb-engine" in b for b in bad)


def test_compare_names_a_sonatype_snapshot_that_moved():
    bad = pin_mod.compare(
        PIN,
        {"amd64": _hashes(), "arm64": _hashes()},
        _maven(**{"arcadedb-engine": "1" * 64}),
    )
    assert any(b.startswith("Sonatype: arcadedb-engine") for b in bad)


def test_compare_needs_engine_and_server_on_sonatype():
    maven = {"arcadedb-engine": PIN["jars"]["arcadedb-engine"]}
    bad = pin_mod.compare(PIN, {"amd64": _hashes(), "arm64": _hashes()}, maven)
    assert any("no arcadedb-server" in b for b in bad)


def _pin_for(directory, jars):
    path = directory / "pin.json"
    path.write_text(
        json.dumps(
            dict(PIN, jars={m: hashlib.sha256(c).hexdigest() for m, c in jars.items()})
        )
    )
    return path


def _write_jars(directory, jars):
    directory.mkdir()
    for mod, content in jars.items():
        (directory / f"{mod}-26.12.1-SNAPSHOT.jar").write_bytes(content)


def test_pinned_snapshot_is_recorded_only_for_the_pinned_jars(tmp_path):
    jars = {"arcadedb-engine": b"engine", "arcadedb-server": b"server"}
    d = tmp_path / "jars"
    _write_jars(d, jars)
    pin_path = _pin_for(tmp_path, jars)
    assert pin_mod.pinned_snapshot(d, pin_path)["snapshot"] == PIN["snapshot"]
    (d / "arcadedb-engine-26.12.1-SNAPSHOT.jar").write_bytes(b"another engine build")
    assert pin_mod.pinned_snapshot(d, pin_path) is None


def test_strict_needs_the_whole_pinned_set(tmp_path):
    d = tmp_path / "jars"
    _write_jars(d, {"arcadedb-engine": b"engine"})
    pin = json.loads(
        _pin_for(
            tmp_path, {"arcadedb-engine": b"engine", "arcadedb-server": b"server"}
        ).read_text()
    )
    assert pin_mod.jars_are_pinned(pin_mod.jar_hashes(d), pin, strict=False)
    assert not pin_mod.jars_are_pinned(pin_mod.jar_hashes(d), pin, strict=True)


STAMP = "26.12.1-20261106.183100-26"


def test_a_manual_dev_run_may_resolve_a_timestamped_snapshot(tmp_path):
    r, out = _run_plan(
        tmp_path,
        "26.12.1.dev0",
        engine="26.12.1-SNAPSHOT",
        release="false",
        maven_engine=STAMP,
    )
    assert r.returncode == 0, r.stderr
    assert "engine=26.12.1-SNAPSHOT\n" in out and f"maven-engine={STAMP}\n" in out


def test_without_a_timestamp_maven_resolves_the_engine_itself(tmp_path):
    r, out = _run_plan(
        tmp_path, "26.12.1.dev0", engine="26.12.1-SNAPSHOT", release="false"
    )
    assert r.returncode == 0, r.stderr
    assert "maven-engine=26.12.1-SNAPSHOT\n" in out


def test_a_timestamp_needs_a_snapshot_engine_of_its_own_base(tmp_path):
    r, _ = _run_plan(
        tmp_path, "26.12.1.dev0", engine="26.11.1", release="false", maven_engine=STAMP
    )
    assert r.returncode != 0 and "needs a SNAPSHOT engine" in r.stderr
    r, _ = _run_plan(
        tmp_path,
        "26.12.1.dev0",
        engine="26.12.1-SNAPSHOT",
        release="false",
        maven_engine="26.12.2-20261106.183100-26",
    )
    assert (
        r.returncode != 0
        and "not a timestamped snapshot of 26.12.1-SNAPSHOT" in r.stderr
    )
    r, _ = _run_plan(
        tmp_path,
        "26.12.1.dev0",
        engine="26.12.1-SNAPSHOT",
        release="false",
        maven_engine="26.12.1-SNAPSHOT",
    )
    assert r.returncode != 0


def test_a_final_release_refuses_a_timestamped_snapshot(tmp_path):
    r, _ = _run_plan(tmp_path, "26.12.1", maven_engine=STAMP)
    assert r.returncode != 0 and "needs a SNAPSHOT engine" in r.stderr
