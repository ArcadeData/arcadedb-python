"""A path of the caller, made absolute before the engine sees it, for both builds.

The engine resolves a part of the files of a relative path against the directory that was current when it started (Java reads
``user.dir`` once, and ``File.getAbsolutePath()`` uses it) and the rest against the current directory of the process. After
``os.chdir`` one relative path then named two directories. An absolute path has one meaning for both.
"""

from __future__ import annotations

import os
from pathlib import Path


def absolute_path(path) -> str:
    """``path`` (a ``str`` or a path object) as an absolute path, a relative one taken from ``os.getcwd()``.

    An absolute path and an empty one (which the engine refuses) come back as they are.
    """
    text = os.fspath(path)
    # on Windows a path such as "\\data" names no drive, so it still depends on the current one
    complete = os.path.isabs(text) and (os.name != "nt" or os.path.splitdrive(text)[0])
    if not text.strip() or complete:
        return text
    if os.name == "nt":
        # Windows resolves ".." by the text itself, and abspath knows a drive-relative path such as "D:db"
        return os.path.abspath(text)
    # pathlib keeps "..", which the operating system resolves after a symbolic link in front of it
    return str(Path(text).absolute())
