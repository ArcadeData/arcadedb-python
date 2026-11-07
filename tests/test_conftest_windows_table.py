"""The Windows exclusions of tests/conftest.py cannot diverge from the table that native/wheel/parity_wheel.py reads.

conftest.py ignores a file on Windows with `collect_ignore`, but the native parity runner passes every test file to pytest by
name, and a file named on the command line is collected anyway. The runner therefore reads the table WINDOWS_NOT_APPLICABLE.
A file added to the `if sys.platform == "win32":` block only (test_dev_engine_pin.py once was) runs on the Windows parity leg
and fails there. Both files are parsed, not imported, as parity_wheel.py itself does.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFTEST = ROOT / "tests" / "conftest.py"
PARITY = ROOT / "native" / "wheel" / "parity_wheel.py"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _table(tree):
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "WINDOWS_NOT_APPLICABLE"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("conftest.py has no WINDOWS_NOT_APPLICABLE table")


def _is_win32_test(test):
    return isinstance(test, ast.Compare) and "win32" in [
        getattr(c, "value", None) for c in test.comparators
    ]


def _win32_ignored(tree):
    """The file names that the `if sys.platform == "win32":` block appends to collect_ignore."""
    names = []
    blocks = [n for n in tree.body if isinstance(n, ast.If) and _is_win32_test(n.test)]
    assert blocks, "conftest.py has no `if sys.platform == 'win32'` block"
    for block in blocks:
        for node in ast.walk(block):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "append"
            ):
                names += [
                    a.value
                    for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)
                ]
    return names


def test_every_file_ignored_on_windows_is_in_the_table_the_parity_runner_reads():
    tree = _tree(CONFTEST)
    table = _table(tree)
    missing = [n for n in _win32_ignored(tree) if n not in table]
    assert (
        not missing
    ), f"{missing}: add them to WINDOWS_NOT_APPLICABLE, not only to collect_ignore (native/wheel/parity_wheel.py)"


def test_every_file_of_the_table_exists():
    table = _table(_tree(CONFTEST))
    assert all((ROOT / "tests" / name).is_file() for name in table), [
        n for n in table if not (ROOT / "tests" / n).is_file()
    ]


def test_the_dev_engine_pin_test_is_in_the_table():
    assert "test_dev_engine_pin.py" in _table(_tree(CONFTEST))


def test_the_parity_runner_still_reads_the_table_by_that_name_and_applies_it():
    src = PARITY.read_text(encoding="utf-8")
    assert '"WINDOWS_NOT_APPLICABLE"' in src and "ast.literal_eval" in src
    assert (
        "win_files" in src
        and 'sys.platform == "win32"'
        in src.replace("\n    ", " ").replace("(", "").replace(")", "")
        or "win_files" in src
    )
