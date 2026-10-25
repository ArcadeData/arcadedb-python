"""The ingest defaults, read from the source so the JVM is not started and the native library is not loaded.

The native build (the ``arcadedb`` package) commits every 1000 rows in ``insert_many`` and ``graph_batch`` (a laptop A/B: smaller
transactions copy less in the young GC and ran faster). The JVM build (``arcadedb-jvm``) keeps its old defaults until they are measured:
``insert_many`` 10,000, ``graph_batch`` unset (one commit per flush), ``create_vertices`` 100,000 rows per crossing. These tests pin both
sides, and run on both backends because they only parse the files.
"""

import ast
from pathlib import Path

import arcadedb

SRC = Path(arcadedb.__file__).parent


def _tree(rel):
    return ast.parse((SRC / rel).read_text(encoding="utf-8"))


def _param_default(rel, func, arg):
    """The AST node of the default of ``arg`` in the only ``def func`` of the file."""
    funcs = [
        n
        for n in ast.walk(_tree(rel))
        if isinstance(n, ast.FunctionDef) and n.name == func
    ]
    assert len(funcs) == 1, (rel, func, len(funcs))
    a = funcs[0].args
    positional = a.posonlyargs + a.args
    pairs = list(zip(positional[len(positional) - len(a.defaults) :], a.defaults))
    pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
    for node, default in pairs:
        if node.arg == arg:
            return default
    raise AssertionError(f"{rel}: {func}() has no default for {arg}")


def _module_constant(rel, name):
    for node in _tree(rel).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{rel}: no module constant {name}")


def _class_constant(rel, cls, name):
    for node in ast.walk(_tree(rel)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in item.targets
                ):
                    return ast.literal_eval(item.value)
    raise AssertionError(f"{rel}: no {cls}.{name}")


def test_jvm_insert_many_default_is_10000():
    node = _param_default("core.py", "insert_many", "commit_every")
    assert ast.literal_eval(node) == 10_000


def test_jvm_graph_batch_commit_every_stays_unset():
    node = _param_default("core.py", "graph_batch", "commit_every")
    assert ast.literal_eval(node) is None
    # the JVM wrapper passes the builder option only when it is given, so unset stays one commit per flush
    node = _param_default("graph_batch.py", "create", "commit_every")
    assert ast.literal_eval(node) is None


def test_jvm_create_vertices_chunk_stays_100000():
    assert _class_constant("graph_batch.py", "GraphBatch", "_BULK_CHUNK") == 100_000


def test_native_insert_many_default_is_1000():
    assert (
        _module_constant("_native/database.py", "DEFAULT_INSERT_MANY_COMMIT_EVERY")
        == 1000
    )
    node = _param_default("_native/database.py", "insert_many", "commit_every")
    assert isinstance(node, ast.Name)
    assert node.id == "DEFAULT_INSERT_MANY_COMMIT_EVERY"


def test_native_graph_batch_default_is_1000_and_the_edge_chunk_is_unchanged():
    assert (
        _class_constant("_native/extras.py", "NativeGraphBatch", "DEFAULT_COMMIT_EVERY")
        == 1000
    )
    assert (
        _class_constant("_native/extras.py", "NativeGraphBatch", "_BULK_CHUNK")
        == 10_000
    )
