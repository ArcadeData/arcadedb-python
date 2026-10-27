"""The Java objects a long float list or a float64 array becomes on the JVM build (see test_query_vector_params.py).

Before: ``list`` and float64 ``ndarray`` parameters became a ``java.util.ArrayList`` filled one boxed object at a time (about 0.33 us
each through JPype, 505 us for 1536 numbers). Now they cross as one ``double[]`` that Java boxes (20 to 45 us). The object
the engine receives is the same ``ArrayList<Double>``; the tests count the crossings and compare the objects.
"""

import jpype
import numpy as np
import pytest

from arcadedb import type_conversion
from arcadedb.core import Database
from arcadedb.type_conversion import convert_python_to_java

N = 100


@pytest.fixture
def crossings(monkeypatch):
    """The calls of ``_boxed_doubles``, the one place a vector crosses as a ``double[]``."""
    calls = []
    original = type_conversion._boxed_doubles

    def spy(values, java_python_types):
        calls.append(len(values))
        return original(values, java_python_types)

    monkeypatch.setattr(type_conversion, "_boxed_doubles", spy)
    return calls


def _elementwise(values):
    """The list as it was built before: one boxed object per number."""
    java_list = jpype.JClass("java.util.ArrayList")()
    for value in values:
        java_list.add(value)
    return java_list


def _classes(java_list):
    return {str(java_list.get(i).getClass().getName()) for i in range(java_list.size())}


def _floats(n=N, seed=1):
    return np.random.default_rng(seed).random(n).tolist()


class TestFloatLists:
    def test_a_float_list_crosses_as_one_buffer(self, crossings):
        values = _floats()
        converted = convert_python_to_java(values)
        assert crossings == [N]
        assert str(converted.getClass().getName()) == "java.util.ArrayList"
        assert _classes(converted) == {"java.lang.Double"}
        assert converted.equals(_elementwise(values))

    def test_the_awkward_numbers_are_the_same_objects(self, crossings):
        values = _floats()[:-5] + [
            float("nan"),
            float("inf"),
            -float("inf"),
            -0.0,
            5e-324,
        ]
        converted = convert_python_to_java(values)
        assert crossings == [N]
        # Double.equals compares the bits: NaN equals NaN, -0.0 differs from 0.0
        assert converted.equals(_elementwise(values))

    def test_a_tuple_of_floats_too(self, crossings):
        values = tuple(_floats())
        assert convert_python_to_java(values).equals(_elementwise(values))
        assert crossings == [N]

    def test_a_nested_list_crosses_its_long_float_lists(self, crossings):
        values = {"q": _floats(), "k": 10, "o": [_floats(40), [1, 2]]}
        converted = convert_python_to_java(values)
        assert crossings == [N, 40]
        assert converted.get("k") == 10
        assert converted.get("q").equals(_elementwise(values["q"]))

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(_floats(31), id="short"),
            pytest.param(_floats()[:-1] + [1], id="an-int"),
            pytest.param(_floats()[:-1] + [True], id="a-bool"),
            pytest.param(_floats()[:-1] + [None], id="a-null"),
            pytest.param(_floats()[:-1] + ["x"], id="a-str"),
            pytest.param(list(range(N)), id="ints"),
            pytest.param(list(np.random.default_rng(2).random(N)), id="numpy-scalars"),
        ],
    )
    def test_anything_else_keeps_the_boxed_path(self, crossings, values):
        converted = convert_python_to_java(values)
        assert crossings == []
        assert converted.size() == len(values)

    def test_an_int_stays_a_long(self, crossings):
        converted = convert_python_to_java(_floats()[:-1] + [7])
        assert crossings == []
        assert str(converted.get(N - 1).getClass().getName()) == "java.lang.Long"
        assert str(converted.get(0).getClass().getName()) == "java.lang.Double"

    def test_a_float_subclass_stays_on_the_boxed_path(self, crossings):
        class Fl(float):
            pass

        convert_python_to_java([Fl(0.5)] * N)
        assert crossings == []


class TestFloat64Arrays:
    def test_a_float64_array_crosses_as_one_buffer(self, crossings):
        array = np.random.default_rng(3).random(N)
        converted = convert_python_to_java(array)
        assert crossings == [N]
        assert str(converted.getClass().getName()) == "java.util.ArrayList"
        assert converted.equals(
            _elementwise(array.tolist())
        )  # the list tolist() made before

    def test_a_strided_view_keeps_its_values(self, crossings):
        array = np.random.default_rng(4).random(2 * N)[::2]
        assert not array.flags.c_contiguous
        assert convert_python_to_java(array).equals(_elementwise(array.tolist()))
        assert crossings == [N]

    def test_a_big_endian_array_goes_through_its_list(self, crossings):
        array = np.random.default_rng(5).random(N).astype(">f8")
        assert convert_python_to_java(array).equals(_elementwise(array.tolist()))
        assert crossings == [N]  # tolist() gives Python floats, which cross as the list

    def test_a_short_or_two_dimensional_array_keeps_the_old_path(self, crossings):
        convert_python_to_java(np.random.default_rng(6).random(31))
        two = convert_python_to_java(np.random.default_rng(6).random((4, N)))
        assert (
            crossings == [N] * 4
        )  # each row is the float list it was; the matrix itself never crosses whole
        assert two.size() == 4

    def test_a_float32_array_is_still_a_float_array(self, crossings):
        converted = convert_python_to_java(
            np.random.default_rng(7).random(N).astype(np.float32)
        )
        assert crossings == []
        assert str(converted.getClass().getName()) == "[F"

    def test_an_int_array_is_still_a_list_of_longs(self, crossings):
        converted = convert_python_to_java(np.arange(N, dtype=np.int64))
        assert crossings == []
        assert _classes(converted) == {"java.lang.Long"}


class TestQueryParameters:
    """The statement entry points reach the same conversion."""

    @pytest.mark.parametrize(
        "args, crossed",
        [
            pytest.param(lambda v: ({"q": v},), 1, id="named-list"),
            pytest.param(lambda v: ([v],), 1, id="positional-list"),
            pytest.param(
                lambda v: ({"q": np.asarray(v)},), 1, id="named-float64-array"
            ),
            pytest.param(lambda v: (np.asarray(v),), 0, id="positional-float64-array"),
            pytest.param(
                lambda v: ({"q": np.asarray(v, dtype=np.float32)},),
                0,
                id="named-float32-array",
            ),
        ],
    )
    def test_the_parameter_object_of_query(self, crossings, args, crossed):
        # a positional float64 array was already one float[] (to_java_float_array); the others were boxed lists
        Database._java_parameters(args(_floats()))
        assert len(crossings) == crossed
