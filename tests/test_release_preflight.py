"""scripts/release_preflight.py: every check passes on good data and fails on each way it can go wrong.

HTTP and subprocess are replaced by fakes, so nothing here touches the network, gh or git. The one
real process is bash, which runs the shell code of the workflow steps (check f) on the real workflow
files. The versions are made-up ones (99.x), never a real release.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "release_preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["release_preflight"] = module
    spec.loader.exec_module(module)
    return module


rp = _load()
REAL_RUN = rp.run
V = "99.1.1"
JOBS_OK = {
    "jobs": [
        {"name": "build", "conclusion": "success"},
        {"name": "x", "conclusion": "skipped"},
    ]
}


class Net:
    """A fake world: the answers of GitHub, Maven Central, Docker Hub, gh and check_version.py."""

    def __init__(self, version=V):
        self.version = version
        self.http = {}  # url -> (status, body) or an OSError
        self.runs = [
            {
                "databaseId": 7,
                "status": "completed",
                "conclusion": "success",
                "event": "schedule",
                "headSha": "a" * 40,
                "createdAt": "t",
            }
        ]
        self.jobs = {7: JOBS_OK}
        self.check_version = {
            "arcadedb-jvm": (0, "OK: fine\n", ""),
            "arcadedb": (0, "OK: fine\n", ""),
        }
        self.gh_variable = (0, "true\n", "")
        self.gh_error = None
        engine = version.split(".post")[0]
        self.http[
            f"https://api.github.com/repos/ArcadeData/arcadedb/releases/tags/{engine}"
        ] = (
            200,
            json.dumps(
                {
                    "tag_name": engine,
                    "draft": False,
                    "prerelease": False,
                    "published_at": "2099-01-01",
                    "html_url": "https://example/r",
                }
            ),
        )
        for artifact in ("arcadedb-engine", "arcadedb-server"):
            self.http[
                f"https://repo1.maven.org/maven2/com/arcadedb/{artifact}/{engine}/{artifact}-{engine}.pom"
            ] = (200, "<pom/>")
        self.http[
            f"https://hub.docker.com/v2/repositories/arcadedata/arcadedb/tags/{engine}"
        ] = (
            200,
            json.dumps(
                {
                    "name": engine,
                    "last_updated": "2099-01-01",
                    "images": [
                        {"architecture": "amd64"},
                        {"architecture": "arm64"},
                        {"architecture": "unknown"},
                    ],
                }
            ),
        )

    def http_get(self, url, headers=None):
        answer = self.http.get(url, (404, ""))
        if isinstance(answer, OSError):
            raise answer
        return answer

    def run(self, cmd, env=None, cwd=None):
        if cmd[0] == "bash":
            return REAL_RUN(cmd, env=env, cwd=cwd)
        if self.gh_error:
            raise OSError(self.gh_error)
        if cmd[0] == "gh" and cmd[1] == "api" and "commits/main" in cmd[2]:
            return 0, json.dumps({"sha": "a" * 40}), ""
        if cmd[0] == "gh" and cmd[1] == "api":
            return self.gh_variable
        if cmd[:3] == ["gh", "run", "list"]:
            return 0, json.dumps(self.runs), ""
        if cmd[:3] == ["gh", "run", "view"]:
            return 0, json.dumps(self.jobs[int(cmd[3])]), ""
        if "check_version.py" in " ".join(cmd):
            return self.check_version[cmd[-1]]
        raise AssertionError(f"unexpected command {cmd}")


@pytest.fixture
def net(monkeypatch):
    n = Net()
    monkeypatch.setattr(rp, "http_get", n.http_get)
    monkeypatch.setattr(rp, "run", n.run)
    return n


def _checks(version=V, root=REPO):
    return rp.run_checks(version, "remote", "owner/repo", root)


def _fails(results):
    return [r for r in results if r.status == "FAIL"]


def _only_fail(results, prefix):
    fails = _fails(results)
    assert [r.check[: len(prefix)] for r in fails] == [prefix], fails
    return fails[0]


@pytest.mark.parametrize("version", [V, V + ".post2", "99.1.1"])
def test_everything_ready_is_zero_fails(net, version):
    net.__init__(version)
    results = _checks(version)
    assert not _fails(results), _fails(results)
    assert {r.check[:3] for r in results} >= {
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
        "(g)",
    }


def test_dev_version_skips_the_upstream_checks(net):
    results = _checks("99.1.1.dev0")
    skipped = [r.check[:3] for r in results if r.status == "SKIP"]
    assert skipped[:3] == ["(a)", "(b)", "(c)"] and "(g)" in skipped
    assert not _fails(results)


def test_post_version_checks_the_engine_it_is_named_after(net):
    net.__init__("99.1.1.post3")
    results = _checks("99.1.1.post3")
    assert any("release 99.1.1 on" in r.check for r in results)
    assert not _fails(results)


@pytest.mark.parametrize("version", ["v99.1.1", "99.1.1rc1", "99.1.1.post0", "99.1"])
def test_bad_version_syntax_fails_alone(net, version):
    results = _checks(version)
    assert [r.check for r in results] == ["(0) version syntax"] and results[
        0
    ].status == "FAIL"


# (a)
def test_a_missing_release_and_tag(net):
    net.http.clear()
    r = _fails(_checks())[0]
    assert r.check.startswith("(a)") and "neither a release nor a tag" in r.evidence


def test_a_tag_without_release(net):
    del net.http[f"https://api.github.com/repos/ArcadeData/arcadedb/releases/tags/{V}"]
    net.http[f"https://api.github.com/repos/ArcadeData/arcadedb/git/ref/tags/{V}"] = (
        200,
        "{}",
    )
    assert "no published release" in _only_fail(_checks(), "(a)").evidence


@pytest.mark.parametrize("flag", ["draft", "prerelease"])
def test_a_draft_or_prerelease(net, flag):
    url = f"https://api.github.com/repos/ArcadeData/arcadedb/releases/tags/{V}"
    body = json.loads(net.http[url][1])
    body[flag] = True
    net.http[url] = (200, json.dumps(body))
    assert flag in _only_fail(_checks(), "(a)").evidence


def test_a_http_error_and_network_error_fail_closed(net):
    url = f"https://api.github.com/repos/ArcadeData/arcadedb/releases/tags/{V}"
    net.http[url] = (500, "")
    assert "HTTP 500" in _only_fail(_checks(), "(a)").evidence
    net.http[url] = OSError("timed out")
    assert "fail closed" in _only_fail(_checks(), "(a)").evidence


# (b)
@pytest.mark.parametrize("artifact", ["arcadedb-engine", "arcadedb-server"])
def test_b_missing_artifact(net, artifact):
    del net.http[
        f"https://repo1.maven.org/maven2/com/arcadedb/{artifact}/{V}/{artifact}-{V}.pom"
    ]
    r = _only_fail(_checks(), "(b)")
    assert artifact in r.check and "not published yet" in r.evidence


def test_b_errors_fail_closed(net):
    url = f"https://repo1.maven.org/maven2/com/arcadedb/arcadedb-server/{V}/arcadedb-server-{V}.pom"
    net.http[url] = (503, "")
    assert "HTTP 503" in _only_fail(_checks(), "(b)").evidence
    net.http[url] = OSError("dns")
    assert "fail closed" in _only_fail(_checks(), "(b)").evidence


# (c)
def test_c_missing_tag(net):
    del net.http[f"https://hub.docker.com/v2/repositories/arcadedata/arcadedb/tags/{V}"]
    assert "not published yet" in _only_fail(_checks(), "(c)").evidence


def test_c_missing_architecture(net):
    url = f"https://hub.docker.com/v2/repositories/arcadedata/arcadedb/tags/{V}"
    net.http[url] = (
        200,
        json.dumps({"name": V, "images": [{"architecture": "amd64"}]}),
    )
    assert "missing arm64" in _only_fail(_checks(), "(c)").evidence


def test_c_wrong_name_bad_json_and_network_error(net):
    url = f"https://hub.docker.com/v2/repositories/arcadedata/arcadedb/tags/{V}"
    net.http[url] = (200, json.dumps({"name": "latest", "images": []}))
    assert "answers as" in _only_fail(_checks(), "(c)").evidence
    net.http[url] = (200, "not json")
    assert _only_fail(_checks(), "(c)")
    net.http[url] = OSError("tls")
    assert "fail closed" in _only_fail(_checks(), "(c)").evidence


# (d)
@pytest.mark.parametrize("code", [1, 2])
def test_d_check_version_refuses(net, code):
    net.check_version["arcadedb"] = (code, "FAIL: nope\n", "")
    r = _only_fail(_checks(), "(d)")
    assert "for arcadedb:" in r.check or r.check.endswith("for arcadedb")
    assert "nope" in r.evidence


def test_d_both_packages_are_checked(net):
    results = [r for r in _checks() if r.check.startswith("(d)")]
    assert [r.check.rsplit(" ", 1)[1] for r in results] == ["arcadedb-jvm", "arcadedb"]


# (e)
def test_e_failed_job(net):
    net.jobs[7] = {
        "jobs": [
            {"name": "build mac", "conclusion": "failure"},
            {"name": "ok", "conclusion": "success"},
        ]
    }
    net.runs[0]["conclusion"] = "failure"
    fails = _fails(_checks())
    assert len(fails) == 3 and all(f.check.startswith("(e)") for f in fails)
    assert "build mac (failure)" in fails[0].evidence


def test_e_cancelled_newest_run_is_skipped_but_shown(net):
    net.runs.insert(
        0,
        {
            "databaseId": 9,
            "status": "completed",
            "conclusion": "cancelled",
            "event": "push",
            "headSha": "b" * 40,
            "createdAt": "t",
        },
    )
    results = [r for r in _checks() if r.check.startswith("(e)")]
    assert all(
        r.status == "PASS"
        and "run 7" in r.evidence
        and "completed/cancelled" in r.evidence
        for r in results
    )


def test_e_in_progress_run_does_not_count_as_a_verdict(net):
    net.runs = [
        {
            "databaseId": 9,
            "status": "in_progress",
            "conclusion": None,
            "event": "push",
            "headSha": "b" * 40,
            "createdAt": "t",
        }
    ]
    assert all("no completed run" in r.evidence for r in _fails(_checks()))


def test_e_no_jobs_or_no_success(net):
    net.jobs[7] = {"jobs": []}
    assert "no jobs" in _fails(_checks())[0].evidence
    net.jobs[7] = {"jobs": [{"name": "a", "conclusion": "skipped"}]}
    assert "no job succeeded" in _fails(_checks())[0].evidence


def test_e_gh_error_fails_closed(net):
    net.gh_error = "gh: command not found"
    fails = _fails(_checks())
    assert fails and all(
        "fail closed" in f.evidence for f in fails if f.check.startswith("(e)")
    )


# (f)
@pytest.mark.parametrize("version", [V, V + ".post1", V + ".dev0"])
def test_f_real_workflows_derive_the_tag(net, version):
    results = [r for r in _checks(version) if r.check.startswith("(f)")]
    assert len(results) == 3 and all(r.status == "PASS" for r in results), results
    assert any(
        "image arcadedata/arcadedb:" + ("latest" if "dev" in version else V)
        in r.evidence
        for r in results
    )


def test_f_extract_step_run_finds_the_real_steps():
    text = (REPO / ".github/workflows/release-jvm.yml").read_text()
    script = rp.extract_step_run(text, "Validate tag")
    assert script and 'IMAGE_TAG="$BASE"' in script and not script.startswith(" ")
    assert rp.extract_step_run(text, "No such step") is None


def test_f_missing_step_fails(net, tmp_path):
    for rel in (
        ".github/workflows/release-jvm.yml",
        ".github/workflows/release-native.yml",
        ".github/workflows/test-native.yml",
    ):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = (
            (REPO / rel)
            .read_text()
            .replace("- name: Validate tag", "- name: Check the tag")
        )
        dst.write_text(text)
    results = [r for r in _checks(root=tmp_path) if r.check.startswith("(f)")]
    assert [r.status for r in results] == ["FAIL", "FAIL", "PASS"]
    assert "has no step" in results[0].evidence


def test_f_wrong_derivation_fails(net, monkeypatch):
    def lying_run(cmd, env=None, cwd=None):
        if cmd[0] == "bash":
            with open(env["GITHUB_OUTPUT"], "w") as f:
                f.write(
                    "version=99.1.1\nbase-version=99.1.0\narcadedb-tag=latest\nfinal=false\nengine=99.0.0\n"
                )
            return 0, "", ""
        return net.run(cmd, env, cwd)

    monkeypatch.setattr(rp, "run", lying_run)
    results = [r for r in _checks() if r.check.startswith("(f)")]
    assert [r.status for r in results] == ["FAIL"] * 3
    assert "image tag 'latest', expected '99.1.1'" in results[0].evidence
    assert "final 'false'" in results[1].evidence
    assert "engine '99.0.0', expected '99.1.1'" in results[2].evidence


def test_f_step_that_exits_nonzero_fails(net, monkeypatch):
    monkeypatch.setattr(
        rp,
        "run",
        lambda cmd, env=None, cwd=None: (
            (1, "", "boom") if cmd[0] == "bash" else net.run(cmd, env, cwd)
        ),
    )
    assert all(
        r.status == "FAIL" and "exited 1" in r.evidence
        for r in _checks()
        if r.check.startswith("(f)")
    )


def test_f_without_bash_is_a_skip(net, monkeypatch):
    monkeypatch.setattr(rp.shutil, "which", lambda name: None)
    results = [r for r in _checks() if r.check.startswith("(f)")]
    assert [r.status for r in results] == ["SKIP"] and "bash not found" in results[
        0
    ].evidence


# (g)
def test_g_variable_false_or_unset_fails(net):
    net.gh_variable = (0, "false\n", "")
    assert "publishes no final wheel" in _only_fail(_checks(), "(g)").evidence
    net.gh_variable = (1, "", "gh: Not Found (HTTP 404)")
    assert "not set" in _only_fail(_checks(), "(g)").evidence


def test_g_unreadable_variable_is_a_skip(net):
    net.gh_variable = (1, "", "gh: Must have admin rights (HTTP 403)")
    results = [r for r in _checks() if r.check.startswith("(g)")]
    assert results[0].status == "SKIP" and "403" in results[0].evidence
    assert not _fails(_checks())


# main
def test_exit_code_is_the_number_of_fails(net, monkeypatch, capsys):
    net.http.clear()
    monkeypatch.setattr(rp, "http_get", net.http_get)
    code = rp.main([V, "--checkout", str(REPO)])
    out = capsys.readouterr().out
    assert code == sum(1 for line in out.splitlines() if line.startswith("FAIL "))
    assert code >= 4 and f"{code} FAIL(s)" in out
    assert "Ready" not in out


def test_main_ready_prints_the_go(net, capsys):
    assert rp.main([V, "--checkout", str(REPO)]) == 0
    assert "Ready" in capsys.readouterr().out


def test_output_is_ascii(net, capsys):
    rp.main([V, "--checkout", str(REPO)])
    capsys.readouterr().out.encode("ascii")
