"""The shards of a pull request (scripts/shard_tests.py): every test file runs in exactly one of them."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "shard_tests", ROOT / "scripts" / "shard_tests.py"
)
shards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shards)

FILES = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))


@pytest.mark.parametrize("backend", ["jvm", "native"])
@pytest.mark.parametrize("count", [1, 2, 3, 4, 7])
def test_every_test_file_is_in_exactly_one_shard(backend, count):
    parts = [shards.select(FILES, f"{i}/{count}", backend) for i in range(1, count + 1)]
    assert sorted(name for part in parts for name in part) == FILES
    assert all(parts)


@pytest.mark.parametrize("backend", ["jvm", "native"])
def test_the_split_does_not_depend_on_the_order_of_the_files(backend):
    assert shards.split(reversed(FILES), 3, backend) == shards.split(FILES, 3, backend)


def test_the_two_longest_native_files_run_in_different_shards():
    heavy = ("test_vector_product_quantization.py", "test_log_redirect.py")
    parts = shards.split(FILES, 3, "native")
    assert [sum(name in part for name in heavy) for part in parts].count(1) == 2


@pytest.mark.parametrize("backend", ["jvm", "native"])
def test_every_weighted_file_exists(backend):
    # a renamed file would silently lose its weight and unbalance the shards
    assert set(shards.WEIGHTS[backend]) <= set(FILES)


@pytest.mark.parametrize("text", ["", "1", "0/2", "3/2", "1/0", "a/b", "1/2/3", " 1/2"])
def test_a_shard_that_is_not_i_of_n_is_refused(text):
    with pytest.raises(ValueError):
        shards.parse_shard(text)


def test_both_suites_get_the_shard_of_the_leg():
    workflows = ROOT / ".github" / "workflows"
    assert '--shard="$SHARD"' in (workflows / "test-jvm.yml").read_text(
        encoding="utf-8"
    )
    native = (workflows / "test-native.yml").read_text(encoding="utf-8")
    assert '--shard "$SHARD"' in native
    # a shard ran part of the suite: it uploads no marker of a passed platform (a release is never sharded)
    marker = native.split("          name: native-passed-${{ matrix.key }}-py", 1)[0]
    assert marker.rstrip().endswith("with:")
    assert "        if: ${{ !matrix.shard }}\n        with:" in marker[-200:]
    # a change to the split starts the native workflow
    assert '      - "scripts/shard_tests.py"\n' in native
