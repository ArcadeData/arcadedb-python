"""numpy without paying for it at ``import arcadedb``.

NumPy is optional, and importing it costs about 60 ms. The modules used to do ``import numpy as _np`` (or ``_np = None`` without it)
at the top, so every ``import arcadedb`` paid for it, also in a process that never touches an array. ``lazy_numpy()`` gives them the
same ``_np`` name with the same two properties, and imports the real module on first use:

* ``_np is None`` exactly when NumPy is not installed (a ``find_spec`` lookup, no import), so every ``if _np is None`` fallback
  (the standard-library float32 tail of the native reply reader, the column builders) behaves as before.
* ``_np.ndarray``, ``_np.generic``, ``_np.floating`` and ``_np.dtype`` are only ever used to ask whether a value the caller passed in
  is a NumPy object. While NumPy is not imported in the process, no such value can exist, so the answer is a class nothing is an
  instance of, and nothing is imported. Once the process has numpy, these are the real classes.
* Every other attribute (``_np.asarray``, ``_np.float32``, ``_np.fromiter`` ...) is a call that builds or reads an array: it imports
  NumPy on the first use and is then a plain instance attribute.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from typing import Any, Optional

# Used only in isinstance() and issubclass() on a caller's value (see the module docstring). Every other name is always the real one.
_VALUE_TESTS = frozenset({"ndarray", "generic", "floating", "dtype"})


def _stand_in(name: str) -> type:
    """Stands for the NumPy class ``name`` while NumPy is not imported: nothing is an instance and nothing is a subclass.

    Called like the class (``_np.dtype("datetime64[ns]")``), it imports NumPy and builds the real object.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import numpy

        return getattr(numpy, name)(*args, **kwargs)

    return type(f"_NoValueIs_{name}", (), {"__new__": __new__})


_STAND_INS = {name: _stand_in(name) for name in _VALUE_TESTS}


class _LazyNumpy:
    """The ``_np`` name of a module, until a call needs the real NumPy."""

    def __getattr__(self, name: str) -> Any:
        # only reached for a name this object does not hold yet
        module = sys.modules.get("numpy")
        if name in _VALUE_TESTS:
            if module is None:
                return _STAND_INS[
                    name
                ]  # not cached: the process may import numpy later
            value = getattr(module, name)
        else:
            if module is None:
                import numpy as module
            value = getattr(module, name)
        object.__setattr__(
            self, name, value
        )  # the next access is an ordinary attribute read
        return value


_LAZY = _LazyNumpy()
_HAS_NUMPY: Optional[bool] = None


def lazy_numpy() -> Optional[_LazyNumpy]:
    """``None`` when NumPy is not installed, else the lazy stand-in for the ``numpy`` module."""
    global _HAS_NUMPY
    if _HAS_NUMPY is None:
        try:
            _HAS_NUMPY = find_spec("numpy") is not None
        except (ImportError, ValueError):
            _HAS_NUMPY = False
    return _LAZY if _HAS_NUMPY else None
