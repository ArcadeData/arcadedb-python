"""scripts/verify_wheel_engine_pin.py: a dev wheel must state the pinned snapshot, a final wheel must not state one."""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import dev_engine_pin  # noqa: E402
import verify_wheel_engine_pin as v  # noqa: E402

PIN = dev_engine_pin.load_pin()


def _wheel(tmp_path, record, name="_build_info.py", text=None):
    path = tmp_path / "arcadedb-26.12.1.dev0-py3-none-any.whl"
    body = text
    if body is None:
        body = f'ENGINE_VERSION = "x"\nENGINE_SNAPSHOT = {record[0]!r}  # note\nENGINE_IMAGE_DIGEST = {record[1]!r}  # note\n'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"arcadedb/{name}", body)
    return path


def test_a_dev_wheel_with_the_pinned_snapshot_passes(tmp_path):
    w = _wheel(tmp_path, (PIN["snapshot"], PIN["image_digest"]))
    assert v.problems("26.12.1.dev0", [w]) == []


def test_a_dev_wheel_of_the_jvm_build_records_in_version_py(tmp_path):
    w = _wheel(tmp_path, (PIN["snapshot"], PIN["image_digest"]), name="_version.py")
    assert v.problems("26.12.1.dev1", [w]) == []


def test_a_dev_wheel_that_records_none_fails(tmp_path):
    w = _wheel(tmp_path, (None, None))
    assert len(v.problems("26.12.1.dev0", [w])) == 1


def test_a_dev_wheel_with_another_snapshot_fails(tmp_path):
    w = _wheel(tmp_path, ("26.12.1-20261201.000000-1", PIN["image_digest"]))
    assert len(v.problems("26.12.1.dev0", [w])) == 1


def test_a_dev_wheel_without_a_recording_fails(tmp_path):
    w = _wheel(tmp_path, None, text="ENGINE_VERSION = 'x'\n")
    assert "records nothing" in v.problems("26.12.1.dev0", [w])[0]


def test_a_final_wheel_must_not_state_a_snapshot(tmp_path):
    assert v.problems("26.12.1", [_wheel(tmp_path, (None, None))]) == []
    assert (
        len(
            v.problems(
                "26.12.1", [_wheel(tmp_path, (PIN["snapshot"], PIN["image_digest"]))]
            )
        )
        == 1
    )


def test_no_wheels_fail():
    assert v.problems("26.12.1.dev0", []) == ["no wheels given"]
