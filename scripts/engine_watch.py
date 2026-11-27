#!/usr/bin/env python3
"""Report what changed in the upstream engine that may need a change in the bindings.

The scan compares two commits of ArcadeData/arcadedb, the last one that was reviewed (`base`) and the newest one (`head`), and
reports only differences of what the engine offers, never of how it is written:

- the public methods, constructors, enum constants and nested types of the Java classes the bindings wrap. The list of classes is
  derived on every run from the sources of this repository: each engine class that `src/arcadedb` names in `JClass`, `JImplements`
  or `JPackage`, each one that `src/java` and `native/src` import or spell out, the supertypes of those classes, and the engine
  types returned by a method whose name the bindings call. A new method has a row of its own only when it is abstract and on
  an interface the bindings implement, or on a class they name; the others are one count;
- the exception classes that descend from a class named in `src/arcadedb/exceptions.py`, or from `NeedRetryException`: new
  ones and removed ones, whatever their module. Such a class changes the code an error gets, or whether it is retried;
- the engine classes that the native image configuration (`reachability-metadata.json`, `.properties`) names and that are gone;
- the settings of `GlobalConfiguration` (new, removed, renamed key, scope, type, default value, allowed values, deprecated);
- the index types (`Schema.INDEX_TYPE`);
- the SQL statements and keywords (the ANTLR grammar) and the SQL functions and methods (the two default factories).

Every finding names the upstream commit (and its pull request) that introduced it and the file. The other merged pull requests that
touched the same files are listed for context; they do not count as findings and never open an issue on their own.

The result goes into one issue, "Engine changes to review". Its state is a hidden comment at the end of the issue text, so no file
in the repository changes: while the issue is open, a run rewrites it with everything since its `base` and comments when there is
something new; when a person closes it, its `head` is reviewed, and the next run starts there. Only issues written by the workflow
token (`github-actions[bot]`) are read, so nobody else can move the mark. Without such an issue the scan starts at FIRST_MARK.
When nothing relevant changed, nothing is written.

Without --publish nothing is written to GitHub (a dry run). --since and --until pick a range to look at and are for dry runs only:
a run that writes always starts at the mark, so that only a closed issue can move it. No model and no API key is used: git,
`gh api`, and this file.

usage:
    engine_watch.py --clone DIR                          # dry run from the mark to the newest upstream commit
    engine_watch.py --clone DIR --since SHA --until SHA  # dry run over a range
    engine_watch.py --clone DIR --publish                # what the workflow runs

Exit codes: 0 done (with or without findings), 1 a wrong argument, 2 git or the API failed, or the engine moved a file the scan
reads (the scan must then be updated, so it fails instead of reporting nothing).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import os
import re
import subprocess  # nosec B404 - runs git and gh with argument lists, never a shell
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "ArcadeData/arcadedb-python"
UPSTREAM = "ArcadeData/arcadedb"
UPSTREAM_GIT = f"https://github.com/{UPSTREAM}.git"
UPSTREAM_BRANCH = "main"
# redirect.github.com: a link that does not put a "mentioned this" line into the upstream pull request or commit.
UPSTREAM_WEB = f"https://redirect.github.com/{UPSTREAM}"

TITLE = "Engine changes to review"
LABELS = ["enhancement", "jvm", "native"]
ASSIGNEE = "tae898"
BOT = "github-actions[bot]"
# Where the first scan starts, used only while no issue of the workflow exists: the newest upstream commit before the engine
# snapshot of 2026-11-06 18:31 UTC that scripts/dev_engine_pin.json named when this scan was written.
FIRST_MARK = "4a91847a1f8c65239a33e860ee8801bcff546f9b"
# GitHub refuses an issue body of more than 65536 characters.
BODY_LIMIT = 60000
SUMMARY_LIMIT = 900000

ENGINE = "engine/src/main/java/com/arcadedb"
GRAMMAR = "engine/src/main/antlr4/com/arcadedb/query/sql/grammar"
SETTINGS_FILE = f"{ENGINE}/GlobalConfiguration.java"
SCHEMA_FILE = f"{ENGINE}/schema/Schema.java"
FUNCTIONS_FILE = f"{ENGINE}/function/sql/DefaultSQLFunctionFactory.java"
METHODS_FILE = f"{ENGINE}/query/sql/method/DefaultSQLMethodFactory.java"
PARSER_FILE = f"{GRAMMAR}/SQLParser.g4"
LEXER_FILE = f"{GRAMMAR}/SQLLexer.g4"
FIXED_FILES = [
    SETTINGS_FILE,
    SCHEMA_FILE,
    FUNCTIONS_FILE,
    METHODS_FILE,
    PARSER_FILE,
    LEXER_FILE,
]

BRIDGE_PACKAGE = "com.arcadedb.python"
PYTHON_SOURCES = ["src/arcadedb"]
JAVA_SOURCES = ["src/java", "native/src/main/java"]
# The files of the native image that name engine classes (reflection metadata and builder arguments).
NATIVE_CONFIG = "native/src/main/resources/META-INF/native-image"
EXCEPTIONS_SOURCE = "src/arcadedb/exceptions.py"
NEED_RETRY = "com.arcadedb.exception.NeedRetryException"
CLASS_NAME = re.compile(r"com\.arcadedb(?:\.[a-z_]\w*)*\.[A-Z]\w*(?:\$\w+)*")


class WatchError(Exception):
    """The scan cannot give a trustworthy answer."""


# ---------------------------------------------------------------------------------------------------------------------------------
# Java: just enough of a parser to list what a file offers (types, public methods, enum constants), with the standard library.

JAVA_TOKEN = re.compile(
    r"""
     (?P<skip>\s+|//[^\n]*|/\*.*?\*/)
    |(?P<textblock>\"\"\"(?:\\.|[^\\])*?\"\"\")
    |(?P<string>"(?:\\.|[^"\\\n])*")
    |(?P<char>'(?:\\.|[^'\\\n])+')
    |(?P<word>[^\W\d][\w$]*|\$[\w$]*)
    |(?P<number>\d[\w.]*)
    |(?P<punct>\.\.\.|->|::|.)
    """,
    re.S | re.X,
)
MODIFIERS = {
    "public",
    "protected",
    "private",
    "static",
    "final",
    "abstract",
    "synchronized",
    "native",
    "strictfp",
    "default",
    "transient",
    "volatile",
    "sealed",
}
OPENERS = {"(": ")", "{": "}", "[": "]"}


def java_tokens(text: str) -> list[str]:
    return [m.group() for m in JAVA_TOKEN.finditer(text) if m.lastgroup != "skip"]


def is_word(token: str) -> bool:
    return bool(token) and (token[0].isalpha() or token[0] in "_$")


def render_tokens(tokens: list[str]) -> str:
    """Tokens back to compact source text: a space only between two words, after a comma, and around an operator in a value."""
    out = ""
    previous = ""
    for token in tokens:
        if previous and (
            (
                (is_word(previous) or previous[0].isdigit())
                and (is_word(token) or token[0].isdigit())
            )
            or previous in (",", "?")
            and token not in (",", ">", ")")
        ):
            out += " "
        out += token
        previous = token
    return out


@dataclasses.dataclass(frozen=True)
class Method:
    name: str  # "<init>" for a constructor
    params: str
    returns: str
    static: bool
    throws: str = ""

    def show(self, type_name: str) -> str:
        tail = f" throws {self.throws}" if self.throws else ""
        if self.name == "<init>":
            return f"new {type_name}({self.params}){tail}"
        prefix = "static " if self.static else ""
        return f"{prefix}{self.returns} {self.name}({self.params}){tail}"


@dataclasses.dataclass
class Flags:
    override: bool = False
    deprecated: bool = False
    abstract: bool = False  # an implementation has to supply it


@dataclasses.dataclass
class Constant:
    args: list[list[str]]  # the top-level arguments, as tokens
    deprecated: bool = False


@dataclasses.dataclass
class JavaType:
    name: str  # "Schema", or "Schema.INDEX_TYPE" for a nested type
    kind: str
    public: bool
    supers: list[str] = dataclasses.field(default_factory=list)
    methods: dict[Method, Flags] = dataclasses.field(default_factory=dict)
    constants: dict[str, Constant] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class JavaFile:
    package: str = ""
    imports: list[str] = dataclasses.field(default_factory=list)
    types: dict[str, JavaType] = dataclasses.field(default_factory=dict)


class _JavaParser:
    def __init__(self, text: str):
        self.t = java_tokens(text)
        self.match: dict[int, int] = {}
        stack: list[int] = []
        for i, token in enumerate(self.t):
            if token in OPENERS:
                stack.append(i)
            elif token in (")", "}", "]") and stack:
                self.match[stack.pop()] = i
        if stack:
            raise WatchError("unbalanced brackets in a Java file")
        self.file = JavaFile()

    def at(self, i: int) -> str:
        return self.t[i] if i < len(self.t) else ""

    def parse(self) -> JavaFile:
        i = 0
        while i < len(self.t):
            token = self.t[i]
            if token in ("package", "import"):
                end = self.t.index(";", i)
                words = [w for w in self.t[i + 1 : end] if w != "static"]
                if token == "package":
                    self.file.package = "".join(words)
                else:
                    self.file.imports.append("".join(words))
                i = end + 1
                continue
            i, annotations, modifiers = self.prefix(i)
            if self.starts_type(i):
                i = self.type_declaration(i, "", modifiers, False)
            else:
                i += 1
        return self.file

    def prefix(self, i: int) -> tuple[int, set[str], set[str]]:
        """Skip the annotations and modifiers at `i`."""
        annotations: set[str] = set()
        modifiers: set[str] = set()
        while i < len(self.t):
            token = self.t[i]
            if token == "@" and self.at(i + 1) != "interface":
                i += 1
                while self.at(i + 1) == ".":
                    i += 2
                annotations.add(self.at(i))
                i += 1
                if self.at(i) == "(":
                    i = self.match[i] + 1
            elif token == "non" and self.at(i + 1) == "-":
                i += 3
            elif token in MODIFIERS:
                modifiers.add(token)
                i += 1
            else:
                break
        return i, annotations, modifiers

    def starts_type(self, i: int) -> bool:
        token = self.at(i)
        if token in ("class", "interface", "enum"):
            return is_word(self.at(i + 1))
        if token == "record":
            return is_word(self.at(i + 1)) and self.at(i + 2) in ("(", "<")
        return token == "@" and self.at(i + 1) == "interface"

    def type_declaration(
        self, i: int, outer: str, modifiers: set[str], in_interface: bool
    ) -> int:
        kind = self.t[i]
        if kind == "@":
            kind, i = "annotation", i + 1
        simple = self.t[i + 1]
        name = f"{outer}.{simple}" if outer else simple
        declared = JavaType(name, kind, "public" in modifiers or in_interface)
        j, angle, clause, current = i + 2, 0, "", []
        components: list[str] = []
        while self.at(j) != "{":
            token = self.at(j)
            if not token:
                raise WatchError(f"no body for the Java type {name}")
            if token == "<":
                angle += 1
            elif token == ">":
                angle -= 1
            elif token == "(":
                if kind == "record" and angle == 0 and not clause:
                    components = self.t[j + 1 : self.match[j]]
                j = self.match[j]
            elif angle == 0 and token in ("extends", "implements", "permits"):
                if current:
                    declared.supers.append("".join(current))
                clause, current = token, []
            elif angle == 0 and clause in ("extends", "implements"):
                if token == ",":
                    declared.supers.append("".join(current))
                    current = []
                elif is_word(token) or token == ".":
                    current.append(token)
            j += 1
        if current:
            declared.supers.append("".join(current))
        self.file.types[name] = declared
        for param in split_arguments(components):
            if len(param) >= 2:
                accessor = Method(param[-1], "", type_text(param[:-1]), False)
                declared.methods[accessor] = Flags()
        end = self.match[j]
        k = j + 1
        if kind == "enum":
            k = self.constants(k, end, declared)
        self.members(k, end, declared, simple)
        return end + 1

    def constants(self, i: int, end: int, declared: JavaType) -> int:
        while i < end:
            i, annotations, _ = self.prefix(i)
            token = self.at(i)
            if token == ";":
                return i + 1
            if not is_word(token):
                return i
            i += 1
            args: list[list[str]] = []
            if self.at(i) == "(":
                args = split_arguments(self.t[i + 1 : self.match[i]])
                i = self.match[i] + 1
            if self.at(i) == "{":
                i = self.match[i] + 1
            declared.constants[token] = Constant(args, "Deprecated" in annotations)
            if self.at(i) == ",":
                i += 1
        return i

    def members(self, i: int, end: int, declared: JavaType, simple: str) -> None:
        in_interface = declared.kind in ("interface", "annotation")
        while i < end:
            i, annotations, modifiers = self.prefix(i)
            if i >= end:
                return
            token = self.t[i]
            if token == ";":
                i += 1
                continue
            if token == "{":  # an initializer block
                i = self.match[i] + 1
                continue
            if self.starts_type(i):
                i = self.type_declaration(i, declared.name, modifiers, in_interface)
                continue
            j = i
            while j < end and self.t[j] not in ("(", "=", ";", "{"):
                if self.t[j] == "[":
                    j = self.match[j]
                elif self.t[j] == "@":  # an annotation inside a type
                    j += 1
                    if self.at(j + 1) == "(":
                        j = self.match[j + 1]
                j += 1
            if j >= end:
                return
            if self.t[j] == "{":  # a compact record constructor
                i = self.match[j] + 1
                continue
            if self.t[j] != "(":  # a field
                while j < end and self.t[j] != ";":
                    j = self.match[j] + 1 if self.t[j] in OPENERS else j + 1
                i = j + 1
                continue
            head = self.t[i:j]
            close = self.match[j]
            k = close + 1
            while k < end and self.t[k] not in ("{", ";"):
                k = self.match[k] + 1 if self.t[k] in OPENERS else k + 1
            has_body = self.at(k) == "{"
            i = (self.match[k] + 1) if has_body else k + 1
            clause = self.t[close + 1 : k]
            if "throws" in clause:
                clause = split_arguments(clause[clause.index("throws") + 1 :])
            else:
                clause = []
            throws = ", ".join(sorted(type_text(name) for name in clause))
            public = "public" in modifiers or (
                in_interface and "private" not in modifiers
            )
            if not public or not head:
                continue
            name, returns = head[-1], head[:-1]
            if returns[:1] == ["<"]:  # the method's own type parameters
                depth = 0
                for n, word in enumerate(returns):
                    depth += {"<": 1, ">": -1}.get(word, 0)
                    if depth == 0:
                        returns = returns[n + 1 :]
                        break
            if not returns:
                if name != simple:
                    continue
                name = "<init>"
            params = ", ".join(
                parameter_type(p) for p in split_arguments(self.t[j + 1 : close]) if p
            )
            static = "static" in modifiers
            method = Method(name, params, type_text(returns), static, throws)
            abstract = "abstract" in modifiers or (
                in_interface
                and not has_body
                and not modifiers & {"default", "static", "private"}
            )
            declared.methods[method] = Flags(
                "Override" in annotations, "Deprecated" in annotations, abstract
            )


def split_arguments(tokens: list[str]) -> list[list[str]]:
    """Split at the commas that are not inside brackets or type arguments."""
    parts: list[list[str]] = []
    current: list[str] = []
    depth = angle = 0
    for n, token in enumerate(tokens):
        if token in OPENERS:
            depth += 1
        elif token in (")", "}", "]"):
            depth -= 1
        elif token == "<" and n and (tokens[n - 1][:1].isupper() or angle):
            angle += 1
        elif token == ">" and angle:
            angle -= 1
        if token == "," and depth == 0 and angle == 0:
            parts.append(current)
            current = []
        else:
            current.append(token)
    if current:
        parts.append(current)
    return parts


def type_text(tokens: list[str]) -> str:
    """A type as short text: no annotations, no `final`, no package names (an import change is not a signature change)."""
    out: list[str] = []
    n = 0
    while n < len(tokens):
        token = tokens[n]
        if token == "@":
            n += 2
            if n < len(tokens) and tokens[n] == "(":
                depth = 0
                while n < len(tokens):
                    depth += {"(": 1, ")": -1}.get(tokens[n], 0)
                    n += 1
                    if depth == 0:
                        break
            continue
        if (
            is_word(token)
            and token[0].islower()
            and n + 1 < len(tokens)
            and tokens[n + 1] == "."
        ):
            n += 2
            continue
        if token != "final":
            out.append(token)
        n += 1
    return render_tokens(out)


def parameter_type(tokens: list[str]) -> str:
    """The type of one parameter, without its name (`String args[]` is `String[]`)."""
    brackets = []
    while tokens[-2:] == ["[", "]"] and len(tokens) > 2 and tokens[-3] != "]":
        if not is_word(tokens[-3]):
            break
        brackets += ["[", "]"]
        tokens = tokens[:-2]
    return type_text(tokens[:-1] + brackets)


def parse_java(text: str) -> JavaFile:
    return _JavaParser(text).parse()


# ---------------------------------------------------------------------------------------------------------------------------------
# The other things the engine offers.


def settings_of(file: JavaFile) -> dict[str, dict[str, str]]:
    """Constant name -> key, scope, type, default value, limits, deprecated, description of each setting."""
    declared = file.types.get("GlobalConfiguration")
    found = {}
    for name, constant in (declared.constants if declared else {}).items():
        args = constant.args
        if len(args) < 4 or not args[0] or not args[0][0].startswith('"'):
            continue
        # What follows the default value limits it (allowed values, a range, a minimum); a callback with a body is code, not a limit.
        limits = [render_tokens(a) for a in args[5:] if a != ["null"] and "{" not in a]
        found[name] = {
            "key": args[0][0].strip('"'),
            "scope": type_text(args[1]),
            "type": render_tokens(args[3]).removesuffix(".class"),
            "default": render_tokens(args[4]) if len(args) > 4 else "",
            "limits": ", ".join(limits),
            "deprecated": "yes" if constant.deprecated else "no",
            "description": "".join(w.strip('"') for w in args[2] if w.startswith('"')),
        }
    return found


def grammar_without_comments(text: str) -> str:
    pattern = re.compile(r"('(?:\\.|[^'\\\n])*')|//[^\n]*|/\*.*?\*/", re.S)
    return pattern.sub(lambda m: m.group(1) or " ", text)


def statements_of(parser_grammar: str) -> dict[str, str]:
    """Label -> the start of each alternative of the rules `statement` and `scriptStatement`."""
    text = grammar_without_comments(parser_grammar)
    found = {}
    for rule in ("statement", "scriptStatement"):
        body = re.search(rf"(?m)^{rule}\s*\n?\s*:(.*?)^\s*;", text, re.S)
        if not body:
            continue
        for alternative, label in re.findall(
            r"([^\n#]+?)\s*#\s*([A-Za-z]\w*)", body.group(1)
        ):
            found[label] = " ".join(alternative.lstrip(" \t:|").split())
    return found


# Single letters (the case-insensitive fragments), and what joins two words of one keyword (`ORDER GAP+ BY`, `FULL _ TEXT`).
KEYWORD_BODY = (
    r"(?:\s*(?:[A-Z](?![A-Za-z0-9_])|UNDERSCORE\b|GAP\+?|WS\+?|'_'|'-'|' '))+\s*"
)


def keywords_of(lexer_grammar: str) -> set[str]:
    """The names of the tokens spelled letter by letter (`SELECT: S E L E C T;`): the words of the language."""
    text = grammar_without_comments(lexer_grammar)
    return {
        name
        for name, body in re.findall(r"(?m)^([A-Z][A-Z0-9_]*)\s*:([^;\n]*);", text)
        if re.fullmatch(KEYWORD_BODY, body) and len(re.findall(r"\b[A-Z]\b", body)) > 1
    }


# ---------------------------------------------------------------------------------------------------------------------------------
# The engine at one commit.


@dataclasses.dataclass(frozen=True)
class Commit:
    sha: str
    date: str
    subject: str
    files: tuple[str, ...] = ()

    @property
    def pull_request(self) -> int | None:
        """The number at the end of a squash subject (`fix(#1): ... (#2) [skip ci]` -> 2), or of a merge subject."""
        found = re.search(
            r"^Merge pull request #(\d+)\b|\(#(\d+)\)(?:[\s.]|\[[^\]]*\])*$",
            self.subject,
        )
        return int(found.group(1) or found.group(2)) if found else None


class GitRepo:
    """Read-only access to a bare clone."""

    def __init__(self, path: str):
        self.path = path
        self._batch: subprocess.Popen | None = None

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        done = subprocess.run(  # nosec B603 B607 - fixed program, argument list
            ["git", "-c", "core.quotepath=off", "-C", self.path, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if check and done.returncode != 0:
            raise WatchError(f"git {' '.join(args[:3])}: {done.stderr.strip()}")
        return done

    @classmethod
    def clone_or_fetch(cls, path: str, url: str = UPSTREAM_GIT) -> GitRepo:
        repo = cls(path)
        if Path(path).exists():
            repo._git(
                "fetch",
                "--quiet",
                "--tags",
                url,
                f"+refs/heads/{UPSTREAM_BRANCH}:refs/heads/{UPSTREAM_BRANCH}",
            )
        else:
            done = subprocess.run(  # nosec B603 B607 - fixed program, argument list
                ["git", "clone", "--quiet", "--bare", url, path],
                capture_output=True,
                text=True,
            )
            if done.returncode != 0:
                raise WatchError(f"git clone: {done.stderr.strip()}")
        return repo

    def resolve(self, rev: str) -> str:
        done = self._git(
            "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False
        )
        if done.returncode != 0:
            raise WatchError(f"the commit {rev} is not in the upstream clone")
        return done.stdout.strip()

    def is_ancestor(self, older: str, newer: str) -> bool:
        return (
            self._git(
                "merge-base", "--is-ancestor", older, newer, check=False
            ).returncode
            == 0
        )

    def date(self, rev: str) -> str:
        return self._git("log", "-1", "--format=%cI", rev).stdout.strip()

    def files(self, rev: str) -> list[str]:
        return self._git("ls-tree", "-r", "--name-only", rev).stdout.splitlines()

    def read(self, rev: str, path: str) -> str | None:
        if self._batch is None:
            self._batch = subprocess.Popen(  # nosec B603 B607 - fixed program
                ["git", "-C", self.path, "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        self._batch.stdin.write(f"{rev}:{path}\n".encode())
        self._batch.stdin.flush()
        header = self._batch.stdout.readline().split()
        if not header or header[-1] == b"missing":
            return None
        data = self._batch.stdout.read(int(header[-1]) + 1)[:-1]
        return data.decode("utf-8", errors="replace")

    def commits(self, base: str, head: str) -> list[Commit]:
        """The commits after `base` up to `head`, oldest first, each with the files it changed."""
        out = self._git(
            "log",
            "--first-parent",
            "--diff-merges=first-parent",
            "--no-renames",
            "--reverse",
            "--name-only",
            "--format=%x01%H%x00%cI%x00%s",
            f"{base}..{head}",
        ).stdout
        found = []
        for block in out.split("\x01")[1:]:
            first, *rest = block.split("\n")
            sha, date, subject = first.split("\x00", 2)
            found.append(Commit(sha, date, subject, tuple(f for f in rest if f)))
        return found


class Engine:
    """Parsed files and class names of one commit."""

    def __init__(self, repo, rev: str, cache: dict):
        self.repo, self.rev, self.cache = repo, rev, cache
        self.classes: dict[str, str] = {}
        marker = "/src/main/java/"
        # The engine module first: it wins when two modules hold the same class name.
        for path in sorted(
            repo.files(rev), key=lambda p: (not p.startswith("engine/"), p)
        ):
            if path.endswith(".java") and marker in path:
                name = path.split(marker, 1)[1][: -len(".java")].replace("/", ".")
                self.classes.setdefault(name, path)

    def parsed(self, path: str) -> JavaFile | None:
        text = self.repo.read(self.rev, path)
        if text is None:
            return None
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in self.cache:
            try:
                self.cache[digest] = parse_java(text)
            except (WatchError, ValueError, KeyError) as error:
                raise WatchError(
                    f"cannot parse {path} at {self.rev}: {error}"
                ) from error
        return self.cache[digest]

    def path_of(self, dotted: str) -> str | None:
        """The file of a class name, which may go on into a nested type or a constant (`...Schema.INDEX_TYPE`)."""
        parts = dotted.replace("$", ".").split(".")
        for end in range(len(parts), 1, -1):
            found = self.classes.get(".".join(parts[:end]))
            if found:
                return found
        return None

    def resolve(self, name: str, context: JavaFile) -> str | None:
        """The file of a type name as written inside the file `context`."""
        if name.startswith("com.arcadedb."):
            return self.path_of(name)
        first = name.split(".")[0]
        for imported in context.imports:
            if imported.endswith("." + first):
                return self.path_of(imported)
        found = self.classes.get(f"{context.package}.{first}")
        if found:
            return found
        for imported in context.imports:
            if imported.endswith(".*"):
                found = self.classes.get(f"{imported[:-2]}.{first}")
                if found:
                    return found
        return None

    def inherited(
        self, path: str, type_name: str, name: str, params: str
    ) -> tuple[str, str] | None:
        """The (file, type) above the type that declares the method, if any supertype does."""
        if name == "<init>":  # constructors are not inherited
            return None
        seen: set[tuple[str, str]] = set()
        queue = [(path, type_name)]
        while queue:
            at, current = queue.pop()
            file = self.parsed(at)
            declared = file.types.get(current) if file else None
            if not declared:
                continue
            for above in declared.supers:
                target = at if above in file.types else self.resolve(above, file)
                if not target:
                    continue
                above_file = self.parsed(target)
                for candidate in above_file.types.values() if above_file else []:
                    if candidate.name.split(".")[-1] != above.split(".")[-1]:
                        continue
                    if (target, candidate.name) in seen:
                        continue
                    seen.add((target, candidate.name))
                    if any(
                        m.name == name and m.params == params for m in candidate.methods
                    ):
                        return target, candidate.name
                    queue.append((target, candidate.name))
        return None

    def moved_up(
        self, older: Engine, paths: set[str]
    ) -> set[tuple[str, str, str, str]]:
        """(file, type, name, parameters) of each method that left a type since `older` and is now declared above it."""
        found = set()
        for path in paths:
            was, now = older.parsed(path), self.parsed(path)
            for name, declared in was.types.items() if was and now else []:
                if name not in now.types:
                    continue
                for method in declared.methods:
                    if method in now.types[name].methods:
                        continue
                    above = self.inherited(path, name, method.name, method.params)
                    if above:
                        found.add((*above, method.name, method.params))
        return found


# ---------------------------------------------------------------------------------------------------------------------------------
# What the bindings use of the engine, from their own sources.


@dataclasses.dataclass
class Bindings:
    wrapped: set[str]  # classes the code works with
    named: set[str]  # every engine class name written anywhere, also in a string
    calls: set[str]  # method names called with `.name(`
    text: str  # all sources, to say whether a setting or a function is named
    # engine types the bindings implement (`JImplements`, `implements`, an anonymous class)
    implemented: set[str] = dataclasses.field(default_factory=set)
    configured: set[str] = dataclasses.field(
        default_factory=set
    )  # named in NATIVE_CONFIG
    errors: set[str] = dataclasses.field(
        default_factory=set
    )  # classes that exceptions.py decides by


def implemented_by(file: JavaFile, tokens: list[str]) -> set[str]:
    """The engine types that a Java file of the bindings implements or extends, as full names."""
    written = [name for declared in file.types.values() for name in declared.supers]
    for n, token in enumerate(tokens):
        if token != "new":
            continue
        end, depth, parts = n + 1, 0, []
        while end < len(tokens) and tokens[end] not in (";", "["):
            if tokens[end] == "(" and depth == 0:
                break
            depth += {"<": 1, ">": -1}.get(tokens[end], 0)
            if depth == 0 and tokens[end] != ">":
                parts.append(tokens[end])
            end += 1
        while end < len(tokens) and tokens[end] in ("(", ")") or depth:
            depth += {"(": 1, ")": -1}.get(tokens[end], 0)
            end += 1
            if not depth:
                break
        if end < len(tokens) and tokens[end] == "{":  # an anonymous class
            written.append("".join(parts))
    found: set[str] = set()
    for name in written:
        if name.startswith("com.arcadedb."):
            found.add(name)
            continue
        first, dot, rest = name.partition(".")
        exact = [i for i in file.imports if i.endswith("." + first)]
        packages = [i[:-2] for i in file.imports if i.endswith(".*")]
        for full in exact or [f"{package}.{first}" for package in packages]:
            if full.startswith("com.arcadedb."):
                found.add(full + dot + rest)
    return found


def error_classes(root: Path) -> set[str]:
    """The Java classes that the table of codes in exceptions.py names; that file imports the standard library only."""
    path = root / EXCEPTIONS_SOURCE
    if not path.is_file():
        return set()
    spec = importlib.util.spec_from_file_location("_engine_watch_exceptions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name for rule in module._RULES for name in rule.java}


def read_bindings(root: Path) -> Bindings:
    wrapped: set[str] = set()
    named: set[str] = set()
    calls: set[str] = set()
    implemented: set[str] = set()
    configured: set[str] = set()
    texts = []
    python = [p for d in PYTHON_SOURCES for p in sorted((root / d).rglob("*.py"))]
    java = [p for d in JAVA_SOURCES for p in sorted((root / d).rglob("*.java"))]
    for path in python + java:
        text = path.read_text(encoding="utf-8", errors="replace")
        texts.append(text)
        calls.update(re.findall(r"\.([A-Za-z_]\w*)\s*\(", text))
        named.update(CLASS_NAME.findall(text))
        if path.suffix == ".py":
            implemented.update(
                re.findall(
                    r"""(?:JImplements|JProxy)\(\s*(?:(?:jpype\.)?JClass\(\s*)?["'](com\.arcadedb\.[\w.$]+)["']""",
                    text,
                )
            )
            wrapped.update(
                re.findall(
                    r"""J(?:Class|Implements)\(\s*["'](com\.arcadedb\.[\w.$]+)["']""",
                    text,
                )
            )
            wrapped.update(
                "com.arcadedb" + tail
                for tail in re.findall(
                    r"""JPackage\(\s*["']com["']\s*\)\s*\.arcadedb((?:\s*\.\s*\w+)+)""",
                    text,
                )
            )
            continue
        tokens = java_tokens(text)
        implemented.update(implemented_by(parse_java(text), tokens))
        wildcard = []
        n = 0
        while n < len(tokens):
            if tokens[n : n + 3] != ["com", ".", "arcadedb"]:
                n += 1
                continue
            end = n + 3
            while (
                end + 1 < len(tokens)
                and tokens[end] == "."
                and (is_word(tokens[end + 1]) or tokens[end + 1] == "*")
            ):
                end += 2
            name = "".join(tokens[n:end])
            declaration = n and tokens[n - 1] == "package"
            if name.endswith(".*"):
                wildcard.append(name[:-2])
            elif not declaration and re.search(r"\.[A-Z]", name):
                wrapped.add(name)
            n = end
        words = {t for t in tokens if t[:1].isupper()}
        wrapped.update(f"{package}.{word}" for package in wildcard for word in words)

    def engine_only(names: set[str]) -> set[str]:
        return {n for n in names if not n.startswith(BRIDGE_PACKAGE + ".")}

    for path in sorted((root / NATIVE_CONFIG).rglob("*")):
        if path.suffix in (".json", ".properties"):
            configured.update(
                CLASS_NAME.findall(path.read_text(encoding="utf-8", errors="replace"))
            )

    return Bindings(
        engine_only(wrapped),
        engine_only(named | configured),
        calls,
        "\n".join(texts),
        engine_only(implemented),
        engine_only(configured),
        error_classes(root),
    )


def watched_files(engine: Engine, bindings: Bindings) -> dict[str, str]:
    """File -> why it is watched, at one commit."""
    why: dict[str, str] = {}
    for name in sorted(bindings.wrapped):
        path = engine.path_of(name)
        if path:
            why.setdefault(path, "named by the bindings")
    direct = list(why)

    def supertypes(queue: list[str]) -> None:
        while queue:
            path = queue.pop()
            file = engine.parsed(path)
            for declared in file.types.values():
                for above in declared.supers:
                    target = engine.resolve(above, file)
                    if target and target not in why:
                        why[target] = f"supertype of {declared.name}"
                        queue.append(target)

    supertypes(list(direct))
    returned = []
    for path in list(why):
        file = engine.parsed(path)
        for declared in file.types.values():
            for method in sorted(declared.methods, key=repr):
                if method.name not in bindings.calls:
                    continue
                for word in re.findall(r"\b[A-Z]\w*(?:\.[A-Z]\w*)*", method.returns):
                    target = engine.resolve(word, file)
                    if target and target not in why:
                        why[target] = f"returned by {declared.name}.{method.name}()"
                        returned.append(target)
    supertypes(returned)
    return why


def with_supertypes(engine: Engine, paths: set[str]) -> set[str]:
    """The files of `paths`, and of every type above a type that one of them holds."""
    found, queue = set(paths), list(paths)
    while queue:
        file = engine.parsed(queue.pop())
        for declared in file.types.values() if file else []:
            for above in declared.supers:
                target = engine.resolve(above, file)
                if target and target not in found:
                    found.add(target)
                    queue.append(target)
    return found


# ---------------------------------------------------------------------------------------------------------------------------------
# The comparison.

SECTIONS = [
    ("broke", "Removed or changed in classes the bindings use"),
    ("added", "New in classes the bindings use"),
    ("exceptions", "Exception classes under the ones the bindings name"),
    ("settings", "Settings (GlobalConfiguration)"),
    ("index", "Index types"),
    ("sql", "SQL statements, keywords, functions, and methods"),
]


@dataclasses.dataclass
class Finding:
    section: str
    change: str  # "removed", "changed", "new", "deprecated"
    subject: str  # the class, setting, index type, or SQL name
    detail: str
    path: str
    used: bool = False
    commit: Commit | None = None

    @property
    def key(self) -> str:
        return "|".join(
            (self.section, self.change, self.subject, self.detail, self.path)
        )


@dataclasses.dataclass
class Scan:
    base: str
    head: str
    base_date: str
    head_date: str
    findings: list[Finding]
    others: list[tuple[Commit, list[str]]]  # commits on watched files without a finding
    watched: dict[str, str]
    commits: list[Commit]
    releases: list[dict] = dataclasses.field(default_factory=list)
    folded: int = 0  # new methods that have no row of their own

    @property
    def digest(self) -> str:
        joined = "\n".join(sorted(f.key for f in self.findings))
        return hashlib.sha256(joined.encode()).hexdigest()[:16]


class Scanner:
    def __init__(self, repo, bindings: Bindings, base: str, head: str):
        self.repo, self.bindings = repo, bindings
        self.base, self.head = base, head
        self.cache: dict = {}
        self.engines: dict[str, Engine] = {}
        self.old = self.engine(base)
        self.new = self.engine(head)
        self.commits = repo.commits(base, head)
        self.by_file: dict[str, list[Commit]] = {}
        for commit in self.commits:
            for path in commit.files:
                self.by_file.setdefault(path, []).append(commit)
        self.findings: list[Finding] = []
        self.folded = 0
        self.direct: set[str] = set()
        self.reached: set[str] = set()
        self.implemented: set[str] = set()

    def engine(self, rev: str) -> Engine:
        if rev not in self.engines:
            self.engines[rev] = Engine(self.repo, rev, self.cache)
        return self.engines[rev]

    def introduced(self, path: str, value) -> Commit | None:
        """The commit after which `value(rev)` has been what it is at the head. `value` reads the file `path`."""
        touching = self.by_file.get(path, [])
        final = value(self.head)
        for n in range(len(touching) - 1, -1, -1):
            before = touching[n - 1].sha if n else self.base
            if value(before) != final:
                return touching[n]
        return None

    def text_at(self, rev: str, path: str) -> str:
        return self.repo.read(rev, path) or ""

    def java_at(self, rev: str, path: str) -> JavaFile:
        text = self.repo.read(rev, path)
        if text is None:
            return JavaFile()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in self.cache:
            self.cache[digest] = parse_java(text)
        return self.cache[digest]

    def add(self, section, change, subject, detail, path, value, used=False) -> None:
        commit = self.introduced(path, value)
        self.findings.append(
            Finding(section, change, subject, detail, path, used, commit)
        )

    def named(self, *words: str) -> bool:
        return any(
            re.search(rf"(?<![\w.]){re.escape(w)}(?![\w])", self.bindings.text)
            for w in words
            if w
        )

    # -- classes ------------------------------------------------------------------------------------------------------------------

    def classes(self, watched: set[str]) -> None:
        self.moved = self.new.moved_up(self.old, watched)
        for path in sorted(watched):
            old, new = self.old.parsed(path), self.new.parsed(path)
            simple = Path(path).stem
            if old is None:
                continue  # a new file: nothing of it was in use at the mark
            if new is None:
                moved = [
                    p
                    for p in self.new.classes.values()
                    if p.endswith(f"/{simple}.java")
                ]
                detail = f"now at {moved[0]}" if moved else "no class of this name left"
                self.add(
                    "broke",
                    "class removed or moved",
                    simple,
                    detail,
                    path,
                    lambda rev, p=path: self.repo.read(rev, p) is None,
                    used=True,
                )
                continue
            for name, was in old.types.items():
                if not was.public:
                    continue
                now = new.types.get(name)
                if now is None or not now.public:
                    self.add(
                        "broke",
                        "type removed",
                        name,
                        f"{was.kind} {name}",
                        path,
                        lambda rev, p=path, t=name: self._public(rev, p, t),
                        used=self.named(name.split(".")[-1]),
                    )
                    continue
                self.methods(path, name, was, now)
                if path != SETTINGS_FILE and not (
                    path == SCHEMA_FILE and name == "Schema.INDEX_TYPE"
                ):
                    self.enum_constants(path, name, was, now)
            for name, now in new.types.items():
                outer = name.rpartition(".")[0]
                if (
                    now.public
                    and name not in old.types
                    and (not outer or outer in old.types)
                ):
                    self.add(
                        "added",
                        "new type",
                        name,
                        f"{now.kind} {name}",
                        path,
                        lambda rev, p=path, t=name: self._public(rev, p, t),
                    )

    def _public(self, rev: str, path: str, name: str) -> bool:
        declared = self.java_at(rev, path).types.get(name)
        return bool(declared and declared.public)

    def _overloads(self, rev: str, path: str, type_name: str, name: str) -> frozenset:
        declared = self.java_at(rev, path).types.get(type_name)
        return frozenset(
            (m, f.deprecated)
            for m, f in (declared.methods if declared else {}).items()
            if m.name == name
        )

    def methods(self, path: str, name: str, was: JavaType, now: JavaType) -> None:
        gone = [
            m
            for m in was.methods
            if m not in now.methods
            # still there for the caller when a supertype declares it
            and not self.new.inherited(path, name, m.name, m.params)
        ]
        came = [
            m
            for m, flags in now.methods.items()
            if m not in was.methods
            and not flags.override
            and not self.old.inherited(path, name, m.name, m.params)
            # a method that a subtype had and that moved up to this type is not new for a caller
            and (path, name, m.name, m.params) not in self.moved
        ]
        # An @Override method whose own old signature went away is the new form of it, not a new method.
        came += [
            m
            for m, flags in now.methods.items()
            if m not in was.methods
            and flags.override
            and any(g.name == m.name for g in gone)
        ]
        names = sorted({m.name for m in gone + came})
        for method_name in names:
            old_forms = sorted(m.show(name) for m in gone if m.name == method_name)
            new_forms = sorted(m.show(name) for m in came if m.name == method_name)
            value = lambda rev, p=path, t=name, m=method_name: self._overloads(  # noqa: E731
                rev, p, t, m
            )
            # used only when both match: the bindings name this class (or one below it) and the method
            used = path in self.reached and (
                self.named(name.split(".")[-1])
                if method_name == "<init>"
                else method_name in self.bindings.calls
            )
            if old_forms and new_forms:
                detail = f"{'; '.join(old_forms)}  ->  {'; '.join(new_forms)}"
                self.add("broke", "changed", name, detail, path, value, used)
            elif old_forms:
                detail = "; ".join(old_forms)
                self.add("broke", "removed", name, detail, path, value, used)
            else:
                detail = "; ".join(new_forms)
                if path in self.implemented and now.kind == "interface":
                    listed = any(
                        now.methods[m].abstract for m in came if m.name == method_name
                    )
                else:
                    listed = path in self.direct
                if listed:
                    self.add("added", "new", name, detail, path, value, used)
                else:
                    self.folded += 1
        for method, flags in now.methods.items():
            before = was.methods.get(method)
            if before and flags.deprecated and not before.deprecated:
                self.add(
                    "broke",
                    "deprecated",
                    name,
                    method.show(name),
                    path,
                    lambda rev, p=path, t=name, m=method.name: self._overloads(
                        rev, p, t, m
                    ),
                    path in self.reached and method.name in self.bindings.calls,
                )

    def enum_constants(self, path, name, was: JavaType, now: JavaType) -> None:
        for section, change, constants in (
            ("broke", "constant removed", set(was.constants) - set(now.constants)),
            ("added", "new constant", set(now.constants) - set(was.constants)),
        ):
            for constant in sorted(constants):
                self.add(
                    section,
                    change,
                    name,
                    constant,
                    path,
                    lambda rev, p=path, t=name, c=constant: c
                    in getattr(self.java_at(rev, p).types.get(t), "constants", {}),
                    used=path in self.reached and self.named(constant),
                )

    def references(self) -> None:
        """A class the bindings name only in a string (a server plugin, say) that is gone."""
        for name in sorted(self.bindings.named - self.bindings.wrapped):
            path = self.old.path_of(name)
            if path and not self.new.path_of(name):
                self.add(
                    "broke",
                    "class removed or moved",
                    name.rsplit(".", 1)[-1],
                    (
                        f"the native image configuration names {name}"
                        if name in self.bindings.configured
                        else f"the bindings name {name}"
                    ),
                    path,
                    lambda rev, p=path: self.repo.read(rev, p) is None,
                    used=True,
                )

    # -- exception classes ----------------------------------------------------------------------------------------------------------

    def exception_classes(self, engine: Engine, paths: list[str]) -> dict:
        """Full name -> (file, type, the named class above it) for each type of `paths` below a class of the table of codes."""
        targets = self.bindings.errors | {NEED_RETRY}
        outside = {t.rsplit(".", 1)[-1]: t for t in targets if t.startswith("java.")}

        def above(path: str, type_name: str, seen: set) -> str | None:
            file = engine.parsed(path)
            declared = file.types.get(type_name) if file else None
            for written in declared.supers if declared else []:
                target = (
                    path if written in file.types else engine.resolve(written, file)
                )
                if target is None:
                    if written.rsplit(".", 1)[-1] in outside:
                        return outside[written.rsplit(".", 1)[-1]]
                    continue
                parent = engine.parsed(target)
                for candidate in parent.types.values():
                    if candidate.name.split(".")[-1] != written.split(".")[-1]:
                        continue
                    full = f"{parent.package}.{candidate.name}"
                    if full in targets:
                        return full
                    if full not in seen:
                        seen.add(full)
                        found = above(target, candidate.name, seen)
                        if found:
                            return found
            return None

        classes = {}
        for path in paths:
            file = engine.parsed(path)
            for type_name, declared in file.types.items() if file else []:
                if declared.kind == "class":
                    root = above(path, type_name, set())
                    if root:
                        classes[f"{file.package}.{type_name}"] = (path, type_name, root)
        return classes

    def exceptions(self, watched: set[str]) -> None:
        """New and removed exception classes below a named one. Only a file that a commit touched can hold one."""
        touched = sorted(
            p for p in self.by_file if p.endswith(".java") and "/src/main/java/" in p
        )
        before = self.exception_classes(self.old, touched)
        after = self.exception_classes(self.new, touched)
        for change, full in [("new exception", n) for n in set(after) - set(before)] + [
            ("exception removed", n) for n in set(before) - set(after)
        ]:
            path, type_name, root = (after if full in after else before)[full]
            # a watched class that is gone has its finding already
            if full not in after and path in watched and self.new.parsed(path) is None:
                continue
            self.add(
                "exceptions",
                change,
                type_name,
                f"{full}, below {root.rsplit('.', 1)[-1]}",
                path,
                lambda rev, p=path, t=type_name: getattr(
                    self.java_at(rev, p).types.get(t), "supers", None
                ),
                used=full not in after and self.named(type_name.split(".")[-1]),
            )

    # -- settings, index types, SQL -------------------------------------------------------------------------------------------------

    def settings(self) -> None:
        path = SETTINGS_FILE
        old = settings_of(self.java_at(self.base, path))
        new = settings_of(self.java_at(self.head, path))
        if not old or not new:
            raise WatchError(f"no setting could be read from {path}")

        parsed: dict[str, dict] = {self.base: old, self.head: new}

        def value(rev: str, constant: str):
            if rev not in parsed:
                parsed[rev] = settings_of(self.java_at(rev, path))
            found = parsed[rev].get(constant)
            return found and tuple(
                v for k, v in sorted(found.items()) if k != "description"
            )

        for constant in sorted(set(old) | set(new)):
            was, now = old.get(constant), new.get(constant)
            at = lambda rev, c=constant: value(rev, c)  # noqa: E731
            if was is None:
                detail = f"{now['type']}, default {now['default'] or 'none'}, scope {now['scope']}: {shorten(now['description'], 140)}"
                self.add(
                    "settings",
                    "new",
                    now["key"],
                    detail,
                    path,
                    at,
                    self.named(constant, now["key"]),
                )
            elif now is None:
                self.add(
                    "settings",
                    "removed",
                    was["key"],
                    constant,
                    path,
                    at,
                    self.named(constant, was["key"]),
                )
            else:
                changes = [
                    f"{k} {was[k] or 'none'} -> {now[k] or 'none'}"
                    for k in ("key", "scope", "type", "default", "limits", "deprecated")
                    if was[k] != now[k]
                ]
                if changes:
                    self.add(
                        "settings",
                        "changed",
                        now["key"],
                        "; ".join(changes),
                        path,
                        at,
                        self.named(constant, was["key"], now["key"]),
                    )

    def index_types(self) -> None:
        path = SCHEMA_FILE

        def types(rev: str) -> set[str]:
            declared = self.java_at(rev, path).types.get("Schema.INDEX_TYPE")
            return set(declared.constants) if declared else set()

        old, new = types(self.base), types(self.head)
        if not old or not new:
            raise WatchError(f"no index type could be read from {path}")
        for change, names in (("new", new - old), ("removed", old - new)):
            for name in sorted(names):
                self.add(
                    "index",
                    change,
                    name,
                    "Schema.INDEX_TYPE",
                    path,
                    lambda rev, n=name: n in types(rev),
                    self.named(name),
                )

    def registered(self, rev: str, path: str) -> dict[str, str]:
        """SQL name -> the class a factory registers it with."""
        factory = self.text_at(rev, path)
        file = self.java_at(rev, path)
        engine = self.engine(rev)
        found = {}
        for argument in re.findall(
            r"(?m)^\s*register\(\s*([^,()]+?)\s*,", grammar_without_comments(factory)
        ):
            if argument.startswith('"'):
                found[argument.strip('"')] = Path(path).stem
                continue
            owner, _, constant = argument.rpartition(".")
            target = engine.resolve(owner, file) if owner else path
            name = None
            for _ in range(5):  # the constant may be declared by a superclass
                source = self.text_at(rev, target) if target else ""
                declared = re.search(
                    rf'\bString\s+{re.escape(constant)}\s*=\s*"([^"]*)"', source
                )
                if declared:
                    name = declared.group(1)
                    break
                parsed = self.java_at(rev, target) if target else JavaFile()
                top = parsed.types.get(Path(target).stem) if target else None
                above = top.supers[0] if top and top.supers else None
                target = engine.resolve(above, parsed) if above else None
                if not target:
                    break
            found[name or argument] = owner or Path(path).stem
        return found

    def sql(self) -> None:
        for kind, path in (("function", FUNCTIONS_FILE), ("method", METHODS_FILE)):
            memo: dict[str, dict[str, str]] = {}

            def names(rev: str, p: str = path, m: dict = memo) -> dict[str, str]:
                if rev not in m:
                    m[rev] = self.registered(rev, p)
                return m[rev]

            old, new = names(self.base), names(self.head)
            if not old or not new:
                raise WatchError(f"no SQL {kind} could be read from {path}")
            for change, found, source in (
                ("new", set(new) - set(old), new),
                ("removed", set(old) - set(new), old),
            ):
                for name in sorted(found):
                    self.add(
                        "sql",
                        change,
                        name,
                        f"SQL {kind}, class {source[name]}",
                        path,
                        lambda rev, n=name, f=names: n in f(rev),
                        self.named(name) if "." in name else False,
                    )
        statements = (
            statements_of(self.text_at(self.base, PARSER_FILE)),
            statements_of(self.text_at(self.head, PARSER_FILE)),
        )
        words = (
            keywords_of(self.text_at(self.base, LEXER_FILE)),
            keywords_of(self.text_at(self.head, LEXER_FILE)),
        )
        if not all(statements) or not all(words):
            raise WatchError(
                "no SQL statement or keyword could be read from the grammar"
            )
        old, new = statements
        for change, labels, source in (
            ("new", set(new) - set(old), new),
            ("removed", set(old) - set(new), old),
        ):
            for label in sorted(labels):
                self.add(
                    "sql",
                    change,
                    label,
                    f"SQL statement: {source[label]}",
                    PARSER_FILE,
                    lambda rev, n=label: n
                    in statements_of(self.text_at(rev, PARSER_FILE)),
                )
        for change, found in (
            ("new", words[1] - words[0]),
            ("removed", words[0] - words[1]),
        ):
            for word in sorted(found):
                self.add(
                    "sql",
                    change,
                    word,
                    "SQL keyword",
                    LEXER_FILE,
                    lambda rev, n=word: n in keywords_of(self.text_at(rev, LEXER_FILE)),
                )

    def run(self) -> Scan:
        for path in FIXED_FILES:
            if self.repo.read(self.head, path) is None:
                raise WatchError(
                    f"{path} is not in the engine at {self.head}: update scripts/engine_watch.py"
                )
        old = watched_files(self.old, self.bindings)
        new = watched_files(self.new, self.bindings)
        watched = {**old, **new}
        if not any(p.endswith("/Database.java") for p in watched):
            raise WatchError("the list of wrapped classes does not even hold Database")
        # the classes the bindings name, and with them (for "used") the types above those
        named = {p for p, why in watched.items() if why == "named by the bindings"}
        self.direct = named
        self.reached = with_supertypes(self.old, named) | with_supertypes(
            self.new, named
        )
        interfaces = {
            path
            for engine in (self.old, self.new)
            for name in self.bindings.implemented
            if (path := engine.path_of(name))
        }
        self.implemented = with_supertypes(self.old, interfaces) | with_supertypes(
            self.new, interfaces
        )
        self.classes(set(watched))
        self.references()
        self.exceptions(set(watched))
        self.settings()
        self.index_types()
        self.sql()
        order = {name: n for n, (name, _) in enumerate(SECTIONS)}
        self.findings.sort(
            key=lambda f: (
                order[f.section],
                not f.used,
                f.subject,
                f.change,
                f.detail,
            )
        )
        cited = {f.commit.sha for f in self.findings if f.commit}
        paths = set(watched) | set(FIXED_FILES)
        others = [
            (commit, touched)
            for commit in reversed(self.commits)
            if commit.sha not in cited
            and commit.pull_request
            and (touched := [p for p in commit.files if p in paths])
        ]
        return Scan(
            self.base,
            self.head,
            self.repo.date(self.base),
            self.repo.date(self.head),
            self.findings,
            others,
            watched,
            self.commits,
            folded=self.folded,
        )


# ---------------------------------------------------------------------------------------------------------------------------------
# The text.


def shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def code(text: str, limit: int = 300) -> str:
    """Upstream text as a code span: nothing in it becomes a link, a mention, or a table border."""
    return "`" + shorten(text, limit).replace("`", "'").replace("|", "\\|") + "`"


def upstream_link(
    commit: Commit | None, base: str, head: str, short: bool = False
) -> str:
    """The pull request and the commit of a change; with `short`, only the pull request when there is one."""
    if commit is None:
        return f"[range]({UPSTREAM_WEB}/compare/{base}...{head})"
    sha = f"[{commit.sha[:9]}]({UPSTREAM_WEB}/commit/{commit.sha})"
    if commit.pull_request:
        number = commit.pull_request
        pull = f"[PR {number}]({UPSTREAM_WEB}/pull/{number})"
        return pull if short else f"{pull}, commit {sha}"
    return f"commit {sha}"


def file_link(path: str, scan: Scan, removed: bool) -> str:
    rev = scan.base if removed else scan.head
    return f"[{Path(path).name}]({UPSTREAM_WEB}/blob/{rev}/{path})"


def finding_row(finding: Finding, scan: Scan) -> str:
    gone = finding.change in ("class removed or moved",)
    return (
        "| "
        + " | ".join(
            [
                code(finding.subject),
                finding.change,
                code(finding.detail),
                "yes" if finding.used else "",
                upstream_link(finding.commit, scan.base, scan.head),
                file_link(finding.path, scan, gone),
            ]
        )
        + " |"
    )


def state_marker(scan: Scan) -> str:
    state = {"base": scan.base, "head": scan.head, "digest": scan.digest}
    return f"<!-- engine-watch-state {json.dumps(state, sort_keys=True)} -->"


def render(
    scan: Scan,
    run_url: str = "",
    rows: int | None = None,
    other_rows: int | None = None,
) -> str:
    base, head = scan.base[:9], scan.head[:9]
    count = len(scan.findings)
    full = f"the [run]({run_url})" if run_url else "a dry run of the script"
    lines = [
        f"The upstream engine changed in ways that may need a change in the Python bindings: {count} finding{'' if count == 1 else 's'} between `{base}` ({scan.base_date[:10]}) and `{head}` ({scan.head_date[:10]}), {len(scan.commits)} upstream commits ([compare]({UPSTREAM_WEB}/compare/{scan.base}...{scan.head})).",
        "",
        f"To review: decide for each line whether the bindings need a change, open an issue for each one that does, then close this issue. Closing it marks `{head}` as reviewed, and the next run starts there. While it stays open, a run rewrites this text with everything since `{base}` and adds a comment when there is something new.",
        "",
    ]
    for release in scan.releases:
        lines += [
            f"Upstream published the release {code(release['tag_name'])} on {release['published_at'][:10]}.",
            "",
        ]
    lines += ["| Section | Findings | Name used in the bindings |", "|---|---|---|"]
    for section, title in SECTIONS:
        found = [f for f in scan.findings if f.section == section]
        lines.append(f"| {title} | {len(found)} | {sum(f.used for f in found)} |")
    for section, title in SECTIONS:
        listed = [f for f in scan.findings if f.section == section]
        fold = section == "added" and scan.folded
        if not listed and not fold:
            continue
        shown = listed if rows is None else listed[:rows]
        lines += ["", f"## {title}"]
        if shown:
            lines += [
                "",
                "| Name | Change | What | Used | Upstream | File |",
                "|---|---|---|---|---|---|",
            ]
            lines += [finding_row(f, scan) for f in shown]
        if len(listed) > len(shown):
            lines += ["", f"{len(listed) - len(shown)} more rows are in {full}."]
        if fold:
            lines += [
                "",
                f"Not listed: {scan.folded} more new method names in watched classes (neither abstract on an interface the bindings implement, nor on a class they name).",
            ]
    if scan.others:
        shown_others = (
            scan.others if other_rows is None else scan.others[: other_rows or 0]
        )
        lines += [
            "",
            f"<details><summary>{len(scan.others)} other merged pull requests touched the watched files without such a change (newest first)</summary>",
            "",
        ]
        for commit, touched in shown_others:
            names = ", ".join(sorted({Path(p).name for p in touched})[:4])
            more = len({Path(p).name for p in touched}) - 4
            names += f", and {more} more" if more > 0 else ""
            lines.append(
                f"- [PR {commit.pull_request}]({UPSTREAM_WEB}/pull/{commit.pull_request}) {code(commit.subject, 140)} ({names})"
            )
        if len(scan.others) > len(shown_others):
            lines += [
                "",
                f"{len(scan.others) - len(shown_others)} more are in {full}.",
            ]
        lines += ["", "</details>"]
    lines += [
        "",
        "<details><summary>How this was made</summary>",
        "",
        f'`scripts/engine_watch.py` (workflow `engine-watch.yml`) compared the two commits. It watches {len(scan.watched)} engine files that it derived from the sources of the bindings, plus the settings, index types, SQL grammar, and SQL function factories. It reports differences of signatures and registrations, not changes of behavior inside a method. "Used" means only that the sources of the bindings hold that name, so check it before relying on it. The comment below this block is the state of the job: do not edit it.',
        "",
        "</details>",
        "",
        state_marker(scan),
    ]
    return "\n".join(lines) + "\n"


def fit(scan: Scan, run_url: str = "", limit: int = BODY_LIMIT) -> str:
    """The text of the issue, cut down until GitHub takes it: first the other pull requests, then the rows of each section."""
    for rows, others in (
        (None, None),
        (None, 100),
        (None, 30),
        (None, 0),
        (150, 0),
        (80, 0),
        (50, 0),
        (30, 0),
        (15, 0),
        (5, 0),
    ):
        body = render(scan, run_url, rows, others)
        if len(body) <= limit:
            return body
    raise WatchError(f"the issue text is longer than {limit} characters")


def new_since(scan: Scan, previous_head: str) -> list[Finding]:
    """The findings whose commit came after the head the issue showed before."""
    shas = [c.sha for c in scan.commits]
    if previous_head not in shas:
        return list(scan.findings) if previous_head == scan.base else []
    later = set(shas[shas.index(previous_head) + 1 :])
    return [f for f in scan.findings if f.commit and f.commit.sha in later]


def comment_text(scan: Scan, fresh: list[Finding], limit: int = 25) -> str:
    lines = [
        f"{len(fresh)} new finding{'' if len(fresh) == 1 else 's'}, the engine is now at `{scan.head[:9]}`. The text above holds all of them.",
        "",
    ]
    for finding in fresh[:limit]:
        what = code(f"{finding.subject}: {finding.detail}", 200).replace("\\|", "|")
        lines.append(
            f"- {finding.change} {what} ({upstream_link(finding.commit, scan.base, scan.head)})"
        )
    if len(fresh) > limit:
        lines.append(f"- and {len(fresh) - limit} more")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------------------
# GitHub.


def gh_api(method: str, path: str, payload: dict | None = None):
    command = ["gh", "api", "--method", method, path]
    if payload is not None:
        command += ["--input", "-"]
    done = subprocess.run(  # nosec B603 B607 - fixed program, argument list
        command,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if done.returncode != 0:
        raise WatchError(f"gh api {method} {path}: {done.stderr.strip()}")
    return json.loads(done.stdout or "null")


def read_marker(body: str | None) -> dict | None:
    found = re.findall(r"<!-- engine-watch-state (\{[^\n]*\}) -->", body or "")
    if not found:
        return None
    try:
        state = json.loads(found[-1])
    except ValueError:
        return None
    sha = re.compile(r"[0-9a-f]{40}")
    if not all(sha.fullmatch(str(state.get(k, ""))) for k in ("base", "head")):
        return None
    return state


def issues_of_the_job(api, repo: str) -> list[dict]:
    found, page = [], 1
    while True:
        batch = api(
            "GET",
            f"repos/{repo}/issues?state=all&creator=github-actions%5Bbot%5D&per_page=100&page={page}",
        )
        found += [
            i
            for i in batch
            if i.get("title") == TITLE
            and (i.get("user") or {}).get("login") == BOT
            and "pull_request" not in i
        ]
        if len(batch) < 100:
            return found
        page += 1


def choose_mark(issues: list[dict]) -> tuple[str, dict | None, str]:
    """(the commit to start from, the open issue to update, where the mark comes from)."""
    opened = sorted(
        (i for i in issues if i["state"] == "open"), key=lambda i: i["number"]
    )
    current = opened[-1] if opened else None
    if current:
        state = read_marker(current.get("body"))
        if state:
            return state["base"], current, f"the open issue {current['number']}"
    closed = sorted(
        (i for i in issues if i["state"] != "open" and read_marker(i.get("body"))),
        key=lambda i: (i.get("closed_at") or "", i["number"]),
    )
    if closed:
        last = closed[-1]
        return (
            read_marker(last["body"])["head"],
            current,
            f"the closed issue {last['number']}",
        )
    return FIRST_MARK, current, "FIRST_MARK (no issue of this job yet)"


def publish(api, repo: str, scan: Scan, current: dict | None, run_url: str) -> str:
    """Write the result. Returns what was done."""
    if not scan.findings:
        return "nothing relevant changed: nothing written"
    if current is None:
        made = api(
            "POST",
            f"repos/{repo}/issues",
            {
                "title": TITLE,
                "body": fit(scan, run_url),
                "labels": LABELS,
                "assignees": [ASSIGNEE],
            },
        )
        return f"opened {made['html_url']}"
    before = read_marker(current.get("body")) or {}
    if before.get("digest") == scan.digest and before.get("base") == scan.base:
        return f"nothing new: issue {current['number']} left as it is"
    number = current["number"]
    api("PATCH", f"repos/{repo}/issues/{number}", {"body": fit(scan, run_url)})
    fresh = new_since(scan, before.get("head", scan.base))
    if not fresh:
        return f"updated issue {number}, no new finding"
    api(
        "POST",
        f"repos/{repo}/issues/{number}/comments",
        {"body": comment_text(scan, fresh)},
    )
    return f"updated issue {number} and commented on {len(fresh)} new findings"


def releases(api) -> list[dict]:
    found = api("GET", f"repos/{UPSTREAM}/releases?per_page=30")
    return [r for r in found if not r.get("draft") and r.get("published_at")]


def moment(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))


def released_within(
    published: list[dict], hours: float, now: datetime.datetime
) -> list[str]:
    limit = now - datetime.timedelta(hours=hours)
    return [r["tag_name"] for r in published if moment(r["published_at"]) >= limit]


# ---------------------------------------------------------------------------------------------------------------------------------


def revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{7,40}|\d+\.\d+\.\d+|main", value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a commit SHA, a release tag, or main"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO
    )
    parser.add_argument(
        "--clone", required=True, help="a bare clone of the engine; made when missing"
    )
    parser.add_argument(
        "--no-fetch", action="store_true", help="use the clone as it is"
    )
    parser.add_argument("--bindings", default=str(ROOT), help="the bindings checkout")
    parser.add_argument(
        "--since", type=revision, help="dry run: start here, not at the mark"
    )
    parser.add_argument(
        "--until", type=revision, help="dry run: stop here, not at the newest commit"
    )
    parser.add_argument("--publish", action="store_true", help="write the issue")
    parser.add_argument("--report", help="write the full text to this file")
    parser.add_argument(
        "--if-released-within",
        type=float,
        metavar="HOURS",
        help="do nothing unless upstream published a release in the last HOURS",
    )
    args = parser.parse_args(argv)
    if args.publish and (args.since or args.until):
        parser.error(
            "--since and --until are for dry runs: --publish starts at the mark"
        )
    run_url = ""
    if os.environ.get("GITHUB_RUN_ID"):
        run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{args.repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    try:
        published = releases(gh_api)
        if args.if_released_within is not None:
            now = datetime.datetime.now(datetime.timezone.utc)
            recent = released_within(published, args.if_released_within, now)
            if not recent:
                print(
                    f"no upstream release in the last {args.if_released_within:g} hours: nothing to do"
                )
                return 0
            print(f"upstream released {', '.join(recent)}: scanning")
        if args.since:
            base, current, source = args.since, None, "--since"
        else:
            base, current, source = choose_mark(issues_of_the_job(gh_api, args.repo))
        repo = (
            GitRepo(args.clone) if args.no_fetch else GitRepo.clone_or_fetch(args.clone)
        )
        base, head = repo.resolve(base), repo.resolve(args.until or UPSTREAM_BRANCH)
        if not repo.is_ancestor(base, head):
            raise WatchError(f"{base} is not an ancestor of {head}")
        print(f"scanning {base[:9]}..{head[:9]}, the start from {source}")
        scan = Scanner(repo, read_bindings(Path(args.bindings)), base, head).run()
        scan.releases = sorted(
            (
                r
                for r in published
                if moment(scan.base_date)
                < moment(r["published_at"])
                <= moment(scan.head_date) + datetime.timedelta(days=2)
            ),
            key=lambda r: r["published_at"],
        )
        full = render(scan, run_url)
        if args.report and scan.findings:
            Path(args.report).write_text(full[:SUMMARY_LIMIT], encoding="utf-8")
        counts = ", ".join(
            f"{name} {sum(f.section == name for f in scan.findings)}"
            for name, _ in SECTIONS
        )
        print(
            f"{len(scan.commits)} commits, {len(scan.watched)} watched files, {len(scan.findings)} findings ({counts}), {len(scan.others)} other pull requests"
        )
        if args.publish:
            print(publish(gh_api, args.repo, scan, current, run_url))
        else:
            print("dry run: nothing is written to GitHub")
            if scan.findings and not args.report:
                print(fit(scan, run_url))
    except WatchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
