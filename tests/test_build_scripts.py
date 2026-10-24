"""The build scripts stop on the first failing command.

They run under ``set -euo pipefail``, but bash ignores the exit status of a
command substitution used as a prefix assignment (``VAR="$(cmd)" other_cmd``):
only ``other_cmd``'s status counts. scripts/build.sh read the engine version
that way, so a failing extract_version.py (no engine jar, two engine versions)
went on with an empty version and failed later with an unrelated message
(#4). ``local``/``export``/``readonly``/``declare V="$(cmd)"`` hide the status
the same way (the builtin's own status counts), so they are flagged too.

The scan is per line and covers the forms these scripts use; a substitution
split across lines or built by eval is not seen.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

ASSIGNMENT = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*=")
MASKING_BUILTIN = re.compile(
    r"^\s*(?:local|export|readonly|declare)(?:\s+-\w+)*\s+[A-Za-z_][A-Za-z0-9_]*="
)


def _word_end(line: str, start: int) -> int:
    """Index just past the shell word starting at ``start``.

    Tracks quotes, ``$(...)`` nesting and backticks, which is all the build
    scripts use. An unquoted ``;``, ``&`` or ``|`` ends the word as whitespace does.
    """
    depth = 0
    quote = None
    i = start
    while i < len(line):
        c = line[i]
        if c == "\\":
            i += 2
            continue
        if quote == "'":
            quote = None if c == "'" else quote
        elif line.startswith("$(", i):
            depth += 1
            i += 1
        elif c == "(" and depth:
            depth += 1  # <(...) and subshells inside the substitution
        elif c == ")" and depth:
            depth -= 1
        elif c in "\"'" and (quote is None or quote == c):
            quote = None if quote == c else c
        elif c == "`":
            quote = None if quote == "`" else ("`" if quote is None else quote)
        elif (c.isspace() or c in ";&|") and quote is None and depth == 0:
            return i
        i += 1
    return i


def _has_substitution(word: str) -> bool:
    return "$(" in word or "`" in word


def is_prefix_substitution(line: str) -> bool:
    """A command substitution whose exit status set -e never sees.

    Either ``NAME="$(cmd)" other ...`` (an assignment prefixing a further
    command) or ``local NAME="$(cmd)"`` and its export/readonly/declare kin.
    """
    builtin = MASKING_BUILTIN.match(line)
    if builtin is not None:
        return _has_substitution(line[builtin.end() : _word_end(line, builtin.end())])
    match = ASSIGNMENT.match(line)
    if match is None:
        return False
    end = _word_end(line, match.end())
    if not _has_substitution(line[match.end() : end]):
        return False
    rest = line[end:].strip()
    return bool(rest) and not rest.startswith(("#", ";", "&", "|"))


def _shell_scripts():
    return sorted(SCRIPTS_DIR.glob("*.sh"))


def test_scanner_tells_a_prefix_from_a_plain_assignment():
    assert is_prefix_substitution("V=\"$(extract)\" python3 - << 'PY'")
    assert is_prefix_substitution("V=$(extract) cmd")
    assert not is_prefix_substitution('V="$(extract)"')
    assert not is_prefix_substitution('D="$(cd "$(dirname "$0")" && pwd)"')
    assert not is_prefix_substitution('V="$(extract)"  # a comment')
    assert not is_prefix_substitution('N=$(comm -13 <(echo "$A") <(echo "$B") || true)')
    assert not is_prefix_substitution('V="$V" B="${B:-}" python3 - << \'PY\'')
    assert not is_prefix_substitution('V="$(a)"; cmd')
    assert not is_prefix_substitution('V="$(a)" && cmd')
    assert is_prefix_substitution("V=`extract` cmd")
    assert is_prefix_substitution('    local V="$(extract)"')
    assert is_prefix_substitution("export V=$(extract)")
    assert is_prefix_substitution('readonly -a V="$(extract)"')
    assert not is_prefix_substitution('local V="$1"')


def test_no_command_substitution_in_a_prefix_assignment():
    offenders = [
        f"{script.name}:{number}: {line.strip()}"
        for script in _shell_scripts()
        for number, line in enumerate(
            script.read_text(encoding="utf-8").splitlines(), start=1
        )
        if is_prefix_substitution(line)
    ]
    assert offenders == []
