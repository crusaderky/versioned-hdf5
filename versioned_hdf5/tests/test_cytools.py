import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ..cytools import (
    cartesian_product,
    ceil_a_over_b,
    count2stop,
    empty_view,
    np_hsize_t,
    smallest_step_after,
    stop2count,
    view_from_tuple,
)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_empty_view(n):
    l = list(range(n))
    v = empty_view(n)
    for i in range(n):
        v[i] = l[i]
    npt.assert_array_equal(v, np.asarray(l, dtype=np_hsize_t), strict=True)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_view_from_tuple(n):
    t = tuple(range(n))
    v = view_from_tuple(t)
    npt.assert_array_equal(v, np.asarray(t, dtype=np_hsize_t), strict=True)


def free_slices_st(size: int):
    """Hypothesis draw of a slice object to slice an array of up to size elements"""
    start_st = st.integers(0, size)
    stop_st = st.integers(0, size)
    # only non-negative steps are allowed
    step_st = st.integers(1, size)
    return st.builds(slice, start_st, stop_st, step_st)


@given(free_slices_st(5))
def test_stop2count_count2stop(s):
    count = stop2count(s.start, s.stop, s.step)
    assert count == len(range(s.start, s.stop, s.step))

    stop = count2stop(s.start, count, s.step)
    # When count == 0 or when step>1, multiple stops yield the same count,
    # so stop won't necessarily be equal to s.stop
    assert count == len(range(s.start, stop, s.step))


def test_cartesian_product():
    a = np.array([1, 2, 3], dtype=np_hsize_t)
    b = np.array([4, 5], dtype=np_hsize_t)
    c = np.array([6, 7], dtype=np_hsize_t)

    expect = np.array(
        [
            [1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7],
        ],
        dtype=np_hsize_t,
    )

    actual = np.asarray(cartesian_product([a, b, c]))
    npt.assert_array_equal(actual, expect, strict=True)


def test_cartesian_product_one_dim():
    a = np.array([1, 2, 3])

    expect = np.array([[1], [2], [3]], dtype=np_hsize_t)
    actual = np.asarray(cartesian_product([a]))
    npt.assert_array_equal(actual, expect, strict=True)


@pytest.mark.parametrize(
    "args",
    [
        ([1, 2, 3], [], [5, 6]),
        ([1, 2, 3], []),
        ([], [1, 2, 3]),
        ([], [], []),
        ([], []),
        ([],),
        (),
    ],
)
def test_cartesian_product_size_zero(args):
    expect = np.empty((0, len(args)), dtype=np_hsize_t)
    args = [np.array(a, dtype=np_hsize_t) for a in args]
    actual = np.asarray(cartesian_product(args))
    npt.assert_array_equal(actual, expect, strict=True)
