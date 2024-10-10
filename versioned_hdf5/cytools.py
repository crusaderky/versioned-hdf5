# Note: this entire module is compiled by cython with wraparound=False
# See meson.build for details

from __future__ import annotations

import array
from typing import TYPE_CHECKING

import cython
import numpy as np

# This is for mypy and for pure-Python modules.
# When compiled, we use the definition from cytools.pxd.
from cython import ulonglong as hsize_t

# numpy equivalent dtype
np_hsize_t = np.ulonglong


_empty_tpl = array.array("Q", [])
_tpl_nbytes = _empty_tpl.itemsize
assert np_hsize_t().nbytes == _tpl_nbytes

if cython.compiled:  # pragma: nocover
    from cython.cimports.cpython import array  # type: ignore

    @cython.ccall
    def empty_view(n: hsize_t) -> hsize_t[:]:
        """Functionally the same, but faster, as

        cdef size_t[:] v = np.empty(n, dtype=np.intp)

        Note that this is limited to one dimension.
        """
        # array.clone exists only in compiled Cython
        return array.clone(_empty_tpl, n, zero=False)  # type: ignore[attr-defined]

    @cython.ccall
    def view_from_tuple(t: tuple[int, ...]) -> hsize_t[:]:
        """Functionally the same, but faster, as

        cdef hsize_t[:] v = array.array("Q", t)
        """
        n = len(t)
        v = empty_view(n)
        for i in range(n):
            v[i] = t[i]
        return v

else:

    def empty_view(n: hsize_t) -> hsize_t[:]:
        return array.array("L", b" " * _tpl_nbytes * n)

    def view_from_tuple(t: tuple[int, ...]) -> hsize_t[:]:
        return array.array("L", t)


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def stop2count(start: hsize_t, stop: hsize_t, step: hsize_t) -> hsize_t:
    """Given a start:stop:step slice or range, return the number of elements yielded.

    This is functionally identical to::

        len(range(start, stop, step))

    Doesn't assume that stop >= start. Assumes that step >= 1.
    """
    # Note that hsize_t is unsigned so stop - start could underflow.
    if stop <= start:
        return 0
    return (stop - start - 1) // step + 1


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def count2stop(start: hsize_t, count: hsize_t, step: hsize_t) -> hsize_t:
    """Inverse of stop2count.

    When count == 0 or when step>1, multiple stops can yield the same count.
    This function returns the smallest stop >= start.
    """
    if count == 0:
        return start
    return start + (count - 1) * step + 1


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def ceil_a_over_b(a: hsize_t, b: hsize_t) -> hsize_t:
    """Returns ceil(a/b). Assumes a >= 0 and b > 0.

    Note
    ----
    This module is compiled with the cython.cdivision flag. This causes behaviour to
    change if a and b have opposite signs and you try debugging the module in pure
    python, without compiling it. This function blindly assumes that a and b are always
    the same sign.
    """
    return a // b + (a % b > 0)


@cython.ccall
@cython.nogil
@cython.exceptval(check=False)
def smallest_step_after(x: hsize_t, a: hsize_t, m: hsize_t) -> hsize_t:
    """Find the smallest integer y >= x where y = a + k*m for whole k's
    Assumes 0 <= a <= x and m >= 1.

    a                  x    y
    | <-- m --> | <-- m --> |
    """
    return a + ceil_a_over_b(x - a, m) * m


@cython.ccall
def cartesian_product(views: list[hsize_t[:]]) -> hsize_t[:, :]:
    """Cartesian product of 1D views of indices

    Same as np.array(list(itertools.product(*arrays)))

    Adapted from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points

    >>> np.asarray(cartesian_product([np.array([1, 2]), np.array([3, 4])]))
    array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]])
    """
    arrays = [np.asarray(v) for v in views]
    la = len(arrays)
    if not la:
        return np.empty((0, 0), dtype=np_hsize_t)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=np_hsize_t)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
