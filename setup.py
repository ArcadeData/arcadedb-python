"""
Setup configuration for arcadedb-jvm.

The wheel is platform-specific because it bundles a platform-specific JRE, but it is not tied to a
Python version: nothing in it is compiled against CPython (the Java side is reached through JPype,
a dependency with its own wheels). So the tag is py3-none-<platform>, one wheel per platform and the
same shape as the native `arcadedb` wheel, not cpXY-cpXY-<platform>, one wheel per Python.

The Pythons the wheel installs on are decided by `requires-python` and the `jpype1` dependency in
pyproject.toml, not by the wheel tag.
"""

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Mark package as having platform-specific content."""

    def has_ext_modules(self):
        # Return True to get a platform tag (and a platlib wheel) even though we have no C
        # extensions. It would also stamp the interpreter and ABI tags of the building Python
        # (cp312-cp312); bdist_wheel.get_tag below replaces those.
        return True


class bdist_wheel(_bdist_wheel):
    """Keep the platform tag (--plat-name, _PYTHON_HOST_PLATFORM or the host's), drop the Python and ABI tags."""

    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


# All other configuration is in pyproject.toml
setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
