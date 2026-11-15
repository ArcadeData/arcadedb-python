"""The signature table of ``lib._load()`` uses only the one-letter ctypes aliases that the function defines (ArcadeData/arcadedb-python #388, #398).

``_load()`` writes ``sigs = {"arcadedb_n_...": [P, L, S, ...]}`` with one letter per ctypes type, defined a few lines above it
(``P, L, N, S = ctypes.c_void_p, ...``). A letter that is not defined is a NameError on the first load of the native library, so no
native test can run and no wheel loads. Which letter the integer is called differs between bases (``main`` has ``N``, the 26.11.1.post3 release
line has ``I``), so a branch that adds a table line on one base and is merged on the other is broken in a way that no single branch shows
and no test of a function finds. This test reads the source: no library, no native backend, any base.
"""

import ast
import inspect

from arcadedb._native import lib


def undefined_signature_names(source: str) -> list:
    """The names the ``sigs`` table of ``_load`` uses that ``_load`` does not assign before it (sorted, without repeats)."""
    tree = ast.parse(source)
    load = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load"
    )
    defined, used = set(), set()
    for node in ast.walk(load):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        defined.add(name.id)
            if any(isinstance(t, ast.Name) and t.id == "sigs" for t in node.targets):
                assert isinstance(node.value, ast.Dict), "sigs is a dict literal"
                for value in node.value.values:
                    for name in ast.walk(value):
                        if isinstance(name, ast.Name):
                            used.add(name.id)
    assert used, "the sigs table was not found in _load"
    return sorted(used - defined)


def test_every_letter_of_the_signature_table_is_defined():
    assert undefined_signature_names(inspect.getsource(lib)) == []


def test_the_check_finds_a_letter_that_is_not_defined():
    """The integration branch of post4 had ``N`` in the table of a base that calls the integer ``I``: the same mistake, planted."""
    broken = """
def _load():
    P, L, I, S = 1, 2, 3, 4
    sigs = {
        "arcadedb_n_open": [P, S, I],
        "arcadedb_n_vindex_bin": [P, L, S, S, N, P],
    }
"""
    assert undefined_signature_names(broken) == ["N"]
    assert undefined_signature_names(broken.replace(", N, ", ", I, ")) == []
