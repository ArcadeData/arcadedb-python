"""More positional values than a statement has placeholders is a wrong call, refused before it runs.

``query("sql", "... WHERE @rid IN ?", [rid_a, rid_b])`` reads the lone list as the positional-parameter list (the documented form,
one element per ``?``). The statement holds one ``?``. Engines before 26.11.1 bound the first element, ignored the second without a
word, and returned fewer rows than the caller meant (and bound null for a missing value). Since 26.11.1 the engine refuses a
mismatched count itself (ArcadeData/arcadedb#10721); this check still runs because it fails earlier, without an engine round trip,
with a TypeError that names the fix, and because on a build linked to an older engine nothing else stops it. The bindings cannot ask
the engine for the count: the parsed statement's placeholder count is no public engine call.
So the check is the one that never refuses a call that binds: an UPPER BOUND on the placeholders, the characters ``?``, ``$`` and
``:`` anywhere in the SQL text (a ``?`` in a string literal counts, so such a statement is checked more loosely, never refused
wrongly). A call with more values than that bound has a value no placeholder can take.
"""

from collections.abc import Mapping
from functools import lru_cache

_COUNTED_LANGUAGES = frozenset(("sql", "sqlscript"))
_NAME_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")


@lru_cache(maxsize=1024)
def _placeholder_kinds(command: str) -> tuple:
    """``(has_positional, has_named)``: a ``?`` and a ``:name`` / ``$name`` in the SQL text outside quotes.

    Skips single- and double-quoted string literals (a backslash escapes the next character, a doubled quote closes and reopens) and
    backtick identifiers, so a ``?`` or ``:x`` inside them counts for nothing. It is a scanner, not a SQL parser: comments are not
    understood, and ``$1`` (a numbered positional) is neither kind.
    """
    positional = named = False
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if ch in "'\"`":
            i += 1
            while i < n and command[i] != ch:
                i += 2 if command[i] == "\\" and ch != "`" else 1
            i += 1
            continue
        if ch == "?":
            positional = True
        elif ch in ":$" and i + 1 < n and command[i + 1] in _NAME_START:
            named = True
        i += 1
    return positional, named


def refuse_dict_for_positionals(
    name: str, language: str, command: str, args: tuple
) -> None:
    """Raise ``TypeError`` when a lone dict is passed to a statement that only has positional ``?`` placeholders.

    A lone dict (also ``[d]`` and ``(d,)``) is the NAMED-parameter map. The engine binds a positional ``?`` that has no key in that
    map to null, without a word, so ``INSERT INTO T SET v = ?`` with ``{"a": 1}`` stores no value. The call is refused only when the
    language is SQL or SQLScript, the text has a ``?`` and no ``:name`` or ``$name`` outside string literals, and no key is a digit
    string (``{"0": v}`` is the engine's way to bind positional slot 0 by key, and it keeps working).
    """
    if len(args) != 1 or language.lower() not in _COUNTED_LANGUAGES:
        return
    values = args[0]
    if isinstance(values, (list, tuple)) and len(values) == 1:
        values = values[0]
    if not isinstance(values, Mapping):
        return
    positional, named = _placeholder_kinds(command)
    if not positional or named:
        return
    if any(isinstance(k, str) and k.isascii() and k.isdigit() for k in values):
        return
    raise TypeError(
        f"{name}() got a dict, which is the map of NAMED parameters, but the statement has only positional ? placeholders. "
        "The engine would bind each ? to null and nothing would fail. "
        "A positional ? takes a list with one element per ?, as in [value]. "
        'A dict binds by name: use :name in the statement and pass {"name": value}. '
        "A dict meant as ONE value cannot be the only argument: put it in a list together with another positional value, "
        "or bind it by name."
    )


@lru_cache(maxsize=1024)
def _placeholder_bound(command: str) -> int:
    """Upper bound on the positional placeholders of ``command``: ``?``, ``$1`` and ``:name`` characters, wherever they stand."""
    return command.count("?") + command.count("$") + command.count(":")


def refuse_surplus_positionals(
    name: str, language: str, command: str, args: tuple
) -> None:
    """Raise ``TypeError`` when ``args`` (the extra arguments of query()/command()) hold more positional values than ``command`` can bind.

    A lone list or tuple is the positional list itself, so its elements are the values; anything else counts one value per argument.
    Fewer than two values never mismatch an upper bound of one or more, and a mapping alone is the named form: both return at once.
    A lone dict against a statement with only ``?`` placeholders is refused first (``refuse_dict_for_positionals``).
    """
    refuse_dict_for_positionals(name, language, command, args)
    if len(args) == 1:
        values = args[0]
        if not isinstance(values, (list, tuple)):
            return
        count = len(values)
    else:
        count = len(args)
    if count < 2 or language.lower() not in _COUNTED_LANGUAGES:
        return
    bound = _placeholder_bound(command)
    if count <= bound:
        return
    raise TypeError(
        f"{name}() got {count} positional values, but the statement has at most {bound} placeholder{'' if bound == 1 else 's'}. "
        "A list or tuple on its own is the positional-parameter list, one element per ?, so it does not bind one list to one ?. "
        "To bind a list as ONE parameter, wrap it in another list, [[a, b]], or pass it by name: "
        '"... WHERE x IN :ids" with {"ids": [a, b]}.'
    )
