"""scripts/engine_watch.py: what changed in the upstream engine that may need a change in the bindings.

No network, no git, and no engine: the engine is a few small files held in memory, one set for each commit, and the GitHub API is a
list of the calls that were made.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "engine_watch.py"


def _load():
    spec = importlib.util.spec_from_file_location("engine_watch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["engine_watch"] = module
    spec.loader.exec_module(module)
    return module


ew = _load()


# -- the Java reader ------------------------------------------------------------------------------------------------------------------

JAVA = r'''
package com.arcadedb.demo;

import com.arcadedb.database.RID;
import java.util.*;

/** A comment with a brace { and public void notAMethod(). */
public class Demo<T extends Comparable<T>> extends Base implements Reader, AutoCloseable {
  private static final String[] NAMES = { "}", "public void nor()" };
  public static final String TEXT = """
      public void inATextBlock() {
      """;
  char brace = '{';

  public Demo(final String path) { this.run(); }

  @Deprecated
  public <K> Map<K, List<RID>> group(@Nullable final com.arcadedb.database.RID rid, String... names) throws IOException {
    new Runnable() { public void run() { } };
    return null;
  }

  @Override
  public void close() { }

  static void hidden() { }

  protected void alsoHidden() { }

  public static abstract class Builder {
    public abstract Builder with(int[] values);
  }

  public enum Mode {
    FAST("f") { int x() { return 1; } }, @Deprecated SLOW("s");

    Mode(String s) { }

    public String letter() { return null; }
  }

  public record Pair(String left, int right) {
    public Pair { }
  }
}

interface Hidden {
  void implicit();

  default int twice(int value) { return value * 2; }

  private void secret() { }
}
'''


def test_the_reader_lists_the_public_methods_constructors_and_nested_types_only():
    file = ew.parse_java(JAVA)
    assert file.package == "com.arcadedb.demo"
    assert file.imports == ["com.arcadedb.database.RID", "java.util.*"]
    demo = file.types["Demo"]
    assert demo.supers == ["Base", "Reader", "AutoCloseable"]
    assert sorted(m.show("Demo") for m in demo.methods) == [
        "Map<K, List<RID>> group(RID, String...) throws IOException",
        "new Demo(String)",
        "void close()",
    ]
    flags = {m.name: f for m, f in demo.methods.items()}
    assert flags["group"].deprecated and flags["close"].override
    assert [m.show("") for m in file.types["Demo.Builder"].methods] == [
        "Builder with(int[])"
    ]
    mode = file.types["Demo.Mode"]
    assert list(mode.constants) == ["FAST", "SLOW"]
    assert mode.constants["SLOW"].deprecated
    assert [m.name for m in mode.methods] == ["letter"]
    assert sorted(m.name for m in file.types["Demo.Pair"].methods) == ["left", "right"]


def test_every_method_of_an_interface_is_public_unless_it_says_private():
    hidden = ew.parse_java(JAVA).types["Hidden"]
    assert not hidden.public
    assert sorted(m.name for m in hidden.methods) == ["implicit", "twice"]


def test_a_parameter_name_or_an_import_form_is_not_part_of_a_signature():
    old = ew.parse_java("interface A { void f(java.util.List<String> names); }")
    new = ew.parse_java("interface A { void f(final List<String> other); }")
    assert list(old.types["A"].methods) == list(new.types["A"].methods)


# -- settings and grammar -------------------------------------------------------------------------------------------------------------


def settings_java(*constants: str) -> str:
    return (
        "package com.arcadedb;\npublic enum GlobalConfiguration {\n"
        + ",\n".join(constants)
        + ";\n  public String getKey() { return null; }\n}\n"
    )


PAGE = 'PAGE("arcadedb.page", SCOPE.DATABASE, "Page " + "size", Integer.class, 64)'


def test_a_setting_is_read_with_its_default_and_its_limits_but_not_its_callback():
    text = settings_java(
        PAGE,
        'WAL("arcadedb.wal", SCOPE.JVM, "x", Integer.class, 0, range(0, 2))',
        'LOG("arcadedb.log", SCOPE.JVM, "y", String.class, "a", value -> { return value; })',
    )
    found = ew.settings_of(ew.parse_java(text))
    assert found["PAGE"]["default"] == "64"
    assert found["PAGE"]["description"] == "Page size"
    assert found["PAGE"]["type"] == "Integer"
    assert found["WAL"]["limits"] == "range(0, 2)"
    assert found["LOG"]["limits"] == ""


def test_statements_and_keywords_come_from_the_grammar():
    parser = """
statement
    : selectStatement                # selectStmt
    // | oldStatement                # oldStmt
    | CREATE INDEX createIndexBody   # createIndexStmt
    ;

scriptStatement
    : statement                      # scriptRegularStmt
    ;
"""
    assert ew.statements_of(parser) == {
        "selectStmt": "selectStatement",
        "createIndexStmt": "CREATE INDEX createIndexBody",
        "scriptRegularStmt": "statement",
    }
    lexer = """
SELECT: S E L E C T;
ORDER_BY: O R D E R GAP+ B Y;
FULL_TEXT: F U L L UNDERSCORE T E X T;
SEMICOLON: ';';
IDENTIFIER: [a-z]+;
fragment S: [sS];
"""
    assert ew.keywords_of(lexer) == {"SELECT", "ORDER_BY", "FULL_TEXT"}


def test_the_pull_request_of_a_commit_is_the_last_number_of_its_subject():
    def number(subject):
        return ew.Commit("a", "d", subject).pull_request

    assert number("fix(#1): a thing (#2)") == 2
    assert number("fix(#1): a thing (#2). [skip ci]") == 2
    assert number("perf(#3): faster (#4)[skip ci]") == 4
    assert number("Merge pull request #5 from x/y [skip ci]") == 5
    assert number("fix(#6): an issue number is no pull request") is None


# -- the bindings' own sources ----------------------------------------------------------------------------------------------------------


def write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_the_wrapped_classes_are_derived_from_the_sources(tmp_path):
    write(
        tmp_path,
        "src/arcadedb/a.py",
        """
X = jpype.JClass("com.arcadedb.database.RID")
@jpype.JImplements("com.arcadedb.database.async.OkCallback")
class C: pass
T = jpype.JPackage("com").arcadedb.schema.Schema.INDEX_TYPE
PLUGIN = "Redis:com.arcadedb.redis.RedisProtocolPlugin"
Y = jpype.JClass("com.arcadedb.python.DbCalls")
db.getSchema()
""",
    )
    write(
        tmp_path,
        "src/java/com/arcadedb/python/B.java",
        """
package com.arcadedb.python;
import com.arcadedb.database.Database;
import com.arcadedb.graph.*;
// com.arcadedb.commented.Away
class B { Edge e; com.arcadedb.index.TypeIndex i; String s = "com.arcadedb.in.Text"; }
""",
    )
    found = ew.read_bindings(tmp_path)
    assert found.wrapped >= {
        "com.arcadedb.database.RID",
        "com.arcadedb.database.async.OkCallback",
        "com.arcadedb.schema.Schema.INDEX_TYPE",
        "com.arcadedb.database.Database",
        "com.arcadedb.graph.Edge",
        "com.arcadedb.index.TypeIndex",
    }
    assert not any("python" in n or "commented" in n for n in found.wrapped)
    assert "com.arcadedb.redis.RedisProtocolPlugin" in found.named - found.wrapped
    assert "getSchema" in found.calls


def test_the_sources_of_this_repository_name_the_classes_the_package_is_built_on():
    found = ew.read_bindings(ROOT)
    for name in (
        "com.arcadedb.database.Database",
        "com.arcadedb.database.DatabaseFactory",
        "com.arcadedb.graph.GraphBatch",
        "com.arcadedb.query.sql.executor.ResultSet",
        "com.arcadedb.server.ArcadeDBServer",
        "com.arcadedb.integration.importer.Importer",
        "com.arcadedb.GlobalConfiguration",
    ):
        assert name in found.wrapped
    assert any(n.startswith("com.arcadedb.schema.Schema") for n in found.wrapped)
    assert not [n for n in found.wrapped if n.startswith("com.arcadedb.python.")]


# -- the scan -------------------------------------------------------------------------------------------------------------------------

E = ew.ENGINE


class FakeRepo:
    """Commits held in memory: each one is the files it changes (None removes a file)."""

    def __init__(self, first: dict, *commits: tuple):
        self.trees = {"0" * 40: dict(first)}
        self.log: list = []
        tree = dict(first)
        for n, (subject, changes) in enumerate(commits, start=1):
            sha = f"{n:x}" * 40
            for path, text in changes.items():
                if text is None:
                    tree.pop(path)
                else:
                    tree[path] = text
            self.trees[sha] = dict(tree)
            self.log.append(
                ew.Commit(sha, "2026-01-02T00:00:00+00:00", subject, tuple(changes))
            )
        self.base = "0" * 40
        self.head = self.log[-1].sha if self.log else self.base

    def files(self, rev):
        return list(self.trees[rev])

    def read(self, rev, path):
        return self.trees[rev].get(path)

    def date(self, rev):
        return "2026-01-01T00:00:00+00:00"

    def commits(self, base, head):
        order = list(self.trees)
        return self.log[order.index(base) : order.index(head)]


def database(*methods: str) -> str:
    return (
        "package com.arcadedb.database;\npublic interface Database extends Basic {\n"
        + "\n".join(f"  {m};" for m in methods)
        + "\n}\n"
    )


BASIC = "package com.arcadedb.database;\npublic interface Basic { %s }\n"
LOCAL = "package com.arcadedb.database;\npublic class LocalDatabase implements Database { %s }\n"
SCHEMA = (
    "package com.arcadedb.schema;\npublic interface Schema { enum INDEX_TYPE { %s } }\n"
)
FACTORY = """package com.arcadedb.function.sql;
import com.arcadedb.function.sql.math.Sum;
public class DefaultSQLFunctionFactory {
  private DefaultSQLFunctionFactory() {
%s
  }
}
"""
METHODS = """package com.arcadedb.query.sql.method;
public class DefaultSQLMethodFactory {
  private DefaultSQLMethodFactory() {
    register("asString", null);
  }
}
"""
GRAMMAR = (
    "statement\n    : selectStatement   # selectStmt\n%s    ;\n"
    "scriptStatement\n    : statement   # scriptRegularStmt\n    ;\n"
)
LEXER = "SELECT: S E L E C T;\n%s"

FIRST = {
    f"{E}/database/Database.java": database(
        "void begin()",
        "void drop(String name)",
        "int count(int limit)",
        "void moveUp()",
    ),
    f"{E}/database/Basic.java": BASIC % "",
    f"{E}/database/LocalDatabase.java": LOCAL % "public void begin() { }",
    ew.SETTINGS_FILE: settings_java(PAGE),
    ew.SCHEMA_FILE: SCHEMA % "LSM_TREE",
    ew.FUNCTIONS_FILE: FACTORY % "    register(Sum.NAME, null);",
    f"{E}/function/sql/math/Sum.java": 'package com.arcadedb.function.sql.math;\npublic class Sum { public static final String NAME = "sum"; }\n',
    ew.METHODS_FILE: METHODS,
    ew.PARSER_FILE: GRAMMAR % "",
    ew.LEXER_FILE: LEXER % "",
}


def bindings(calls=("begin", "drop")):
    return ew.Bindings(
        {"com.arcadedb.database.Database", "com.arcadedb.database.LocalDatabase"}
        | {"com.arcadedb.GlobalConfiguration", "com.arcadedb.schema.Schema"},
        set(),
        set(calls),
        "arcadedb.page",
    )


def changed_engine():
    avg = 'package com.arcadedb.function.sql.math;\npublic class Avg { public static final String NAME = "avg"; }\n'
    factory = (
        FACTORY % "    register(Sum.NAME, null);\n    register(Avg.NAME, null);"
    ).replace(
        "import com.arcadedb.function.sql.math.Sum;",
        "import com.arcadedb.function.sql.math.Sum;\nimport com.arcadedb.function.sql.math.Avg;",
    )
    return FakeRepo(
        FIRST,
        (
            "feat: a signature (#101)",
            {
                f"{E}/database/Database.java": database(
                    "void begin()",
                    "long count(int limit)",
                    "void added(String a)",
                )
            },
        ),
        (
            "refactor: move a method up (#102) [skip ci]",
            {f"{E}/database/Basic.java": BASIC % "void moveUp();"},
        ),
        (
            "feat: internals and an override (#103)",
            {
                f"{E}/database/LocalDatabase.java": LOCAL
                % "public void begin() { } @Override public void added(String a) { } public void internalOnly() { }"
            },
        ),
        (
            "feat: settings, index type, function (#104)",
            {
                ew.SETTINGS_FILE: settings_java(
                    PAGE.replace("64", "128"),
                    'NEW("arcadedb.new", SCOPE.JVM, "New one", Boolean.class, true)',
                ),
                ew.SCHEMA_FILE: SCHEMA % "LSM_TREE, COLUMNAR",
                ew.FUNCTIONS_FILE: factory,
                f"{E}/function/sql/math/Avg.java": avg,
            },
        ),
        (
            "Merge pull request #105 from a/b",
            {
                ew.PARSER_FILE: GRAMMAR % "    | FACETS body   # facetsStmt\n",
                ew.LEXER_FILE: LEXER % "FACETS: F A C E T S;\n",
            },
        ),
        (
            "fix: @someone only a body, closes #9 (#106)",
            {
                f"{E}/database/LocalDatabase.java": LOCAL
                % "public void begin() { run(); } @Override public void added(String a) { } public void internalOnly() { }"
            },
        ),
    )


def scan_of(repo, base=None, head=None):
    return ew.Scanner(repo, bindings(), base or repo.base, head or repo.head).run()


def found(scan):
    return {
        (f.section, f.change, f.subject, f.detail): f.commit.pull_request
        for f in scan.findings
    }


def test_the_scan_reports_what_changed_with_the_pull_request_that_changed_it():
    result = found(scan_of(changed_engine()))
    assert result[("broke", "removed", "Database", "void drop(String)")] == 101
    assert (
        result[
            (
                "broke",
                "changed",
                "Database",
                "int count(int)  ->  long count(int)",
            )
        ]
        == 101
    )
    assert result[("added", "new", "Database", "void added(String)")] == 101
    assert result[("added", "new", "LocalDatabase", "void internalOnly()")] == 103
    new_setting = "Boolean, default true, scope SCOPE.JVM: New one"
    assert result[("settings", "new", "arcadedb.new", new_setting)] == 104
    assert result[("settings", "changed", "arcadedb.page", "default 64 -> 128")] == 104
    assert result[("index", "new", "COLUMNAR", "Schema.INDEX_TYPE")] == 104
    assert result[("sql", "new", "avg", "SQL function, class Avg")] == 104
    assert result[("sql", "new", "facetsStmt", "SQL statement: FACETS body")] == 105
    assert result[("sql", "new", "FACETS", "SQL keyword")] == 105


def test_a_method_moved_to_a_supertype_or_overridden_is_no_finding():
    result = found(scan_of(changed_engine()))
    # moveUp left Database for its supertype Basic: a caller still has it, so it is neither removed nor new
    assert not [k for k in result if "moveUp" in k[3]]
    # LocalDatabase.added only implements the method of the interface
    assert not [k for k in result if k[2] == "LocalDatabase" and "added" in k[3]]


def test_a_new_method_has_a_row_only_on_a_class_the_bindings_name():
    repo = FakeRepo(
        {**FIRST, f"{E}/database/Basic.java": BASIC % ""},
        (
            "feat: methods (#1)",
            {
                # Basic is watched as a supertype of Database, which the bindings name; they do not name Basic
                f"{E}/database/Basic.java": BASIC % "void onTheSupertype();",
                f"{E}/database/LocalDatabase.java": LOCAL
                % "public void begin() { } public void onTheClass() { }",
            },
        ),
    )
    scan = scan_of(repo)
    assert [(f.subject, f.detail) for f in scan.findings] == [
        ("LocalDatabase", "void onTheClass()")
    ]
    assert scan.folded == 1
    body = ew.render(scan)
    assert "1 more new method names in watched classes" in body
    assert "onTheSupertype" not in body


def test_an_implemented_interface_lists_its_new_abstract_methods_only():
    hook = "package com.arcadedb.database;\npublic interface Hook { %s }\n"
    path = f"{E}/database/Hook.java"
    repo = FakeRepo(
        {**FIRST, path: hook % "void call();"},
        (
            "feat: hook (#1)",
            {
                path: hook
                % "void call(); void required(int n); default void optional() { } static void helper() { }"
            },
        ),
    )
    used = bindings()
    used.wrapped.add("com.arcadedb.database.Hook")
    used.implemented.add("com.arcadedb.database.Hook")
    scan = ew.Scanner(repo, used, repo.base, repo.head).run()
    assert [(f.subject, f.detail) for f in scan.findings] == [
        ("Hook", "void required(int)")
    ]
    assert scan.folded == 2


def test_used_needs_the_class_and_the_method():
    basic = f"{E}/database/Basic.java"
    other = f"{E}/database/Other.java"
    text = "package com.arcadedb.database;\npublic class Other { %s }\n"
    returned = database("void begin()", "Other other()")
    repo = FakeRepo(
        {
            **FIRST,
            f"{E}/database/Database.java": returned,
            basic: BASIC % "void drop(int n);",
            other: text % "public void drop(String s) { }",
        },
        ("refactor: drop (#1)", {basic: BASIC % "", other: text % ""}),
    )
    scan = ew.Scanner(
        repo, bindings(calls=("other", "drop")), repo.base, repo.head
    ).run()
    used = {f.subject: f.used for f in scan.findings}
    # Basic is above Database, which the bindings name; Other is only returned, so the name alone does not make it used
    assert used == {"Basic": True, "Other": False}


def test_new_and_removed_exception_classes_below_a_named_one_are_reported():
    package = "package com.arcadedb.exception;\n"
    need = package + "public class NeedRetryException extends RuntimeException { }\n"
    listed = package + "public class SchemaException extends RuntimeException { }\n"

    def below(name, parent):
        return package + f"public class {name} extends {parent} {{ }}\n"

    x = f"{E}/exception"
    repo = FakeRepo(
        {
            **FIRST,
            f"{x}/NeedRetryException.java": need,
            f"{x}/SchemaException.java": listed,
            f"{x}/OldSchemaException.java": below(
                "OldSchemaException", "SchemaException"
            ),
        },
        (
            "feat: refusals (#8)",
            {
                f"{x}/BusyException.java": below("BusyException", "NeedRetryException"),
                f"{x}/StillBusyException.java": below(
                    "StillBusyException", "BusyException"
                ),
                f"{x}/UnrelatedException.java": below(
                    "UnrelatedException", "RuntimeException"
                ),
                "server/src/main/java/com/arcadedb/server/Holder.java": "package com.arcadedb.server;\n"
                "public class Holder { static class Bad extends IllegalArgumentException { } }\n",
                f"{x}/OldSchemaException.java": None,
            },
        ),
    )
    used = bindings()
    used.errors.update(
        {"com.arcadedb.exception.SchemaException", "java.lang.IllegalArgumentException"}
    )
    scan = ew.Scanner(repo, used, repo.base, repo.head).run()
    got = {
        (f.change, f.subject, f.detail.rsplit(" ", 1)[-1])
        for f in scan.findings
        if f.section == "exceptions"
    }
    assert got == {
        ("new exception", "BusyException", "NeedRetryException"),
        ("new exception", "StillBusyException", "NeedRetryException"),
        ("new exception", "Holder.Bad", "IllegalArgumentException"),
        ("exception removed", "OldSchemaException", "SchemaException"),
    }
    assert {f.commit.pull_request for f in scan.findings} == {8}


def test_a_removed_class_that_only_the_native_image_configuration_names(tmp_path):
    config = "native/src/main/resources/META-INF/native-image/org.humemai/x"
    write(
        tmp_path,
        f"{config}/reachability-metadata.json",
        '{"reflection": [{"type": "com.arcadedb.query.OldStatement"}]}',
    )
    write(
        tmp_path,
        f"{config}/native-image.properties",
        "Args = --initialize-at-run-time=com.arcadedb.query.OldHolder\n",
    )
    read = ew.read_bindings(tmp_path)
    names = {"com.arcadedb.query.OldStatement", "com.arcadedb.query.OldHolder"}
    assert read.configured == names and names <= read.named
    gone = {
        f"{E}/query/{n}.java": f"package com.arcadedb.query;\npublic class {n} {{ }}\n"
        for n in ("OldStatement", "OldHolder")
    }
    repo = FakeRepo({**FIRST, **gone}, ("drop (#2)", dict.fromkeys(gone)))
    used = bindings()
    used.named.update(read.named)
    used.configured.update(read.configured)
    scan = ew.Scanner(repo, used, repo.base, repo.head).run()
    assert sorted(f.detail for f in scan.findings) == [
        f"the native image configuration names {n}" for n in sorted(names)
    ]


def test_the_interfaces_the_bindings_implement_are_read_from_the_sources(tmp_path):
    write(
        tmp_path,
        "src/arcadedb/a.py",
        """
@jpype.JImplements("com.arcadedb.database.async.OkCallback")
class A: pass
@jpype.JImplements(jpype.JClass("com.arcadedb.database.async.ErrorCallback"))
class B: pass
""",
    )
    write(
        tmp_path,
        "src/java/com/arcadedb/python/C.java",
        """
package com.arcadedb.python;
import com.arcadedb.query.sql.executor.ResultSet;
import com.arcadedb.database.Database;
import com.arcadedb.database.DocumentCallback;
class C implements ResultSet {
  Database db;
  Object o = new DocumentCallback() { public boolean onRecord(Object d) { return true; } };
}
""",
    )
    assert ew.read_bindings(tmp_path).implemented == {
        "com.arcadedb.database.async.OkCallback",
        "com.arcadedb.database.async.ErrorCallback",
        "com.arcadedb.query.sql.executor.ResultSet",
        "com.arcadedb.database.DocumentCallback",
    }


def test_the_classes_of_the_table_of_codes_are_read_from_this_repository():
    errors = ew.read_bindings(ROOT).errors
    assert "com.arcadedb.exception.NeedRetryException" in errors
    assert "com.arcadedb.exception.DuplicatedKeyException" in errors


def test_a_change_inside_a_method_body_is_no_finding_but_its_pull_request_is_listed():
    scan = scan_of(changed_engine())
    assert 106 in [commit.pull_request for commit, _ in scan.others]
    assert 106 not in found(scan).values()


def test_nothing_changed_gives_no_finding():
    repo = FakeRepo(
        FIRST,
        (
            "fix: body only (#1)",
            {
                f"{E}/database/LocalDatabase.java": LOCAL
                % "public void begin() { x(); }"
            },
        ),
    )
    assert scan_of(repo).findings == []


def test_a_removed_class_is_one_finding():
    repo = FakeRepo(
        FIRST,
        ("drop it (#7)", {f"{E}/database/LocalDatabase.java": None}),
    )
    scan = scan_of(repo)
    assert [(f.change, f.subject, f.commit.pull_request) for f in scan.findings] == [
        ("class removed or moved", "LocalDatabase", 7)
    ]


def test_a_file_the_scan_reads_that_moved_is_an_error_not_silence():
    repo = FakeRepo(FIRST, ("move (#1)", {ew.LEXER_FILE: None}))
    with pytest.raises(ew.WatchError, match="SQLLexer"):
        scan_of(repo)


# -- the text -------------------------------------------------------------------------------------------------------------------------


def test_the_text_never_links_into_upstream_or_names_a_person_or_a_local_issue():
    scan = scan_of(changed_engine())
    body = ew.render(scan, "https://example.invalid/run")
    assert "https://github.com/ArcadeData/arcadedb/" not in body
    assert "https://redirect.github.com/ArcadeData/arcadedb/pull/104" in body
    outside_code = re.sub(r"\]\([^)]*\)", "]", re.sub(r"`[^`\n]*`", "", body))
    assert not re.search(r"(?<![\w&])#\d", outside_code)
    assert "@" not in outside_code
    assert chr(0x2014) not in body
    assert ew.read_marker(body) == {
        "base": scan.base,
        "head": scan.head,
        "digest": scan.digest,
    }


def test_a_long_result_is_cut_until_github_takes_it_and_says_so():
    scan = scan_of(changed_engine())
    scan.findings = [
        ew.Finding("added", "new", "Database", f"void m{n}()", f"{E}/X.java")
        for n in range(900)
    ]
    body = ew.fit(scan, "https://example.invalid/run")
    assert len(body) <= ew.BODY_LIMIT
    assert "more rows are in the [run](https://example.invalid/run)" in body
    assert ew.read_marker(body)["head"] == scan.head


# -- the issue ------------------------------------------------------------------------------------------------------------------------


def issue(number, state, base, head, login=ew.BOT, title=ew.TITLE, closed_at=None):
    marker = f'<!-- engine-watch-state {{"base": "{base}", "digest": "d", "head": "{head}"}} -->'
    return {
        "number": number,
        "state": state,
        "title": title,
        "user": {"login": login},
        "body": f"text\n\n{marker}\n",
        "closed_at": closed_at,
    }


A, B, C = "a" * 40, "b" * 40, "c" * 40


def test_the_mark_is_the_open_issue_then_the_last_closed_one_then_the_first_mark():
    assert ew.choose_mark([])[:2] == (ew.FIRST_MARK, None)
    closed = [
        issue(5, "closed", A, B, closed_at="2026-01-02T00:00:00Z"),
        issue(3, "closed", A, C, closed_at="2026-02-02T00:00:00Z"),
    ]
    assert ew.choose_mark(closed)[:2] == (C, None)
    opened = issue(9, "open", B, C)
    base, current, _ = ew.choose_mark(closed + [opened])
    assert (base, current["number"]) == (B, 9)


def test_an_open_issue_with_a_removed_marker_is_updated_from_the_last_reviewed_commit():
    opened = issue(9, "open", B, C)
    opened["body"] = "edited by hand"
    closed = issue(5, "closed", A, B, closed_at="2026-01-02T00:00:00Z")
    base, current, _ = ew.choose_mark([opened, closed])
    assert (base, current["number"]) == (B, 9)


def test_only_issues_written_by_the_workflow_move_the_mark():
    listed = [
        issue(1, "open", A, B, login="someone-else"),
        issue(2, "open", A, B, title="Another title"),
        issue(3, "closed", A, B),
        dict(issue(4, "open", A, B), pull_request={}),
    ]
    got = ew.issues_of_the_job(lambda method, path: listed, "o/r")
    assert [i["number"] for i in got] == [3]


class FakeApi:
    def __init__(self):
        self.calls = []

    def __call__(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        return {"html_url": "https://example.invalid/1"}


def test_the_first_result_opens_one_issue_with_labels_and_an_assignee():
    api, scan = FakeApi(), scan_of(changed_engine())
    ew.publish(api, "o/r", scan, None, "")
    ((method, path, payload),) = api.calls
    assert (method, path) == ("POST", "repos/o/r/issues")
    assert payload["title"] == "Engine changes to review"
    assert payload["labels"] == ew.LABELS and payload["assignees"] == ["tae898"]
    assert ew.read_marker(payload["body"])["head"] == scan.head


def test_an_unchanged_result_writes_nothing_and_an_empty_one_opens_nothing():
    api, scan = FakeApi(), scan_of(changed_engine())
    current = {"number": 4, "body": ew.render(scan)}
    ew.publish(api, "o/r", scan, current, "")
    scan.findings = []
    ew.publish(api, "o/r", scan, None, "")
    ew.publish(api, "o/r", scan, current, "")
    assert api.calls == []


def test_a_newer_finding_rewrites_the_text_and_comments_on_it_only():
    repo = changed_engine()
    earlier = scan_of(repo, head=repo.log[2].sha)
    later = scan_of(repo)
    api = FakeApi()
    ew.publish(api, "o/r", later, {"number": 4, "body": ew.render(earlier)}, "")
    assert [(m, p) for m, p, _ in api.calls] == [
        ("PATCH", "repos/o/r/issues/4"),
        ("POST", "repos/o/r/issues/4/comments"),
    ]
    comment = api.calls[1][2]["body"]
    assert "COLUMNAR" in comment and "void drop(String)" not in comment


# -- when and how it runs -------------------------------------------------------------------------------------------------------------


def test_the_daily_run_looks_only_for_a_release_of_the_last_hours():
    now = datetime.datetime(2026, 1, 10, tzinfo=datetime.timezone.utc)
    published = [
        {"tag_name": "26.2.1", "published_at": "2026-01-09T10:00:00Z"},
        {"tag_name": "26.1.1", "published_at": "2026-01-01T10:00:00Z"},
    ]
    assert ew.released_within(published, 48, now) == ["26.2.1"]
    assert ew.released_within(published, 2, now) == []


@pytest.mark.parametrize("flag", ["--since", "--until"])
def test_a_run_that_writes_cannot_pick_its_own_range(flag, capsys):
    with pytest.raises(SystemExit) as stop:
        ew.main(["--clone", "x", "--publish", flag, "abcdef0"])
    assert stop.value.code == 2
    assert "dry runs" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["--upload-pack=x", "HEAD~1", "main;rm"])
def test_a_range_is_a_commit_a_release_tag_or_main(value):
    with pytest.raises(SystemExit):
        ew.main(["--clone", "x", f"--since={value}"])


def test_the_workflow_runs_weekly_and_daily_and_writes_only_issues():
    text = (ROOT / ".github" / "workflows" / "engine-watch.yml").read_text(
        encoding="utf-8"
    )
    crons = re.findall(r'cron: "([^"]+)"', text)
    assert crons == ["23 5 * * 1", "23 5 * * 0,2-6"]
    assert f'"$SCHEDULE" = "{crons[1]}"' in text
    assert "--if-released-within 48" in text
    assert "pull_request" not in text.split("\non:\n", 1)[1].split("\njobs:\n")[0]
    granted = re.findall(r"(?m)^\s+([a-z-]+): (read|write)\b", text)
    assert granted == [("contents", "read"), ("issues", "write")]
    assert not re.search(r"uses: \S+@(?![0-9a-f]{40} # v)", text)
