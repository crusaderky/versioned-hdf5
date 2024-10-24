from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Literal

import hypothesis
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from ..staged_changes import StagedChangesArray
from .test_subchunk_map import idx_st, shape_chunks_st

max_examples = 10_000


def action_st(shape: tuple[int, ...], max_size: int = 20):
    resize_st = st.tuples(*(st.integers(0, max_size) for _ in shape))
    return st.one_of(
        st.tuples(st.just("getitem"), idx_st(shape)),
        st.tuples(st.just("setitem"), idx_st(shape)),
        st.tuples(st.just("resize"), resize_st),
    )


@st.composite
def staged_array_st(
    draw, max_ndim: int = 4, max_size: int = 20, max_actions: int = 6
) -> tuple[
    Literal["full", "from_array"],  # how to create the base array
    tuple[int, ...],  # initial shape
    tuple[int, ...],  # chunk size
    list[tuple[Literal["getitem", "setitem", "resize"], Any]],  # 0 or more actions
]:
    base, (shape, chunks), n_actions = draw(
        st.tuples(
            st.one_of(st.just("full"), st.just("from_array")),
            shape_chunks_st(max_ndim=max_ndim, min_size=0, max_size=max_size),
            st.integers(0, max_actions),
        )
    )

    orig_shape = shape
    actions = []
    for _ in range(n_actions):
        label, arg = draw(action_st(shape, max_size=max_size))
        actions.append((label, arg))
        if label == "resize":
            shape = arg

    return base, orig_shape, chunks, actions


@pytest.mark.slow
@given(staged_array_st())
@hypothesis.settings(max_examples=max_examples, deadline=None)
def test_staged_array(args):
    base, shape, chunks, actions = args

    rng = np.random.default_rng(0)
    fill_value = 42

    if base == "full":
        base = np.full(shape, fill_value, dtype="u4")
        arr = StagedChangesArray.full(shape, chunks, fill_value, dtype=base.dtype)
        assert arr.n_base_slabs == 0
    elif base == "from_array":
        base = rng.integers(2**32, size=shape, dtype="u4")
        # copy base to detect bugs where the StagedChangesArray
        # accidentally writes back to the base slabs
        arr = StagedChangesArray.from_array(base.copy(), chunks, fill_value)
        if base.size == 0:
            assert arr.n_base_slabs == 0
        else:
            assert arr.n_base_slabs > 0
    else:
        raise AssertionError("unreachable")

    assert arr.dtype == base.dtype
    assert arr.fill_value.shape == ()
    assert arr.fill_value == fill_value
    assert arr.fill_value.dtype == base.dtype
    assert arr.slabs[0].dtype == base.dtype
    assert arr.slabs[0].shape == chunks
    assert arr.itemsize == 4
    assert arr.size == np.prod(shape)
    assert arr.nbytes == np.prod(shape) * 4
    assert arr.shape == shape
    assert len(arr) == shape[0]
    assert arr.ndim == len(shape)
    assert arr.chunk_size == chunks
    assert len(arr.n_chunks) == len(shape)
    for n, s, c in zip(arr.n_chunks, shape, chunks):
        assert n == s // c + (s % c > 0)
    assert arr.n_staged_slabs == 0
    assert arr.n_base_slabs + 1 == arr.n_slabs == len(arr.slabs)
    assert not arr.has_changes

    expect = base.copy()

    for label, arg in actions:
        if label == "getitem":
            assert_array_equal(arr[arg], expect[arg], strict=True)

        elif label == "setitem":
            value = rng.integers(2**32, size=expect[arg].shape, dtype="u4")
            expect[arg] = value
            arr[arg] = value
            assert_array_equal(arr, expect, strict=True)

        elif label == "resize":
            # Can't use np.resize(), which works differently as it reflows the data
            new = np.full(arg, fill_value, dtype="u4")
            common_idx = tuple(slice(min(o, n)) for o, n in zip(arr.shape, arg))
            new[common_idx] = expect[common_idx]
            expect = new
            arr.resize(arg)
            assert arr.shape == arg
            assert_array_equal(arr, expect, strict=True)

        else:
            raise AssertionError("unreachable")

    # Test has_changes property.
    # Edge cases would be complicated to test:
    # - a __setitem__ with empty index or a resize with the same shape won't
    #   flip has_changes
    # - a round-trip where an array is created empty, resized, filled, wiped by
    #   resize(0), and then # resized to the original shape will have has_changes=True
    #   even if identical to the original.
    if expect.shape != base.shape or (expect != base).any():
        assert arr.has_changes
    elif all(label == "getitem" for label, _ in actions):
        assert not arr.has_changes

    # One final __getitem__ of everything
    assert_array_equal(arr, expect, strict=True)

    # Reconstruct the array starting from the base + changes
    final = np.full(expect.shape, fill_value, dtype=base.dtype)
    shapes = [base.shape] + [arg for label, arg in actions if label == "resize"]
    common_idx = tuple(slice(min(sizes)) for sizes in zip(*shapes))
    final[common_idx] = base[common_idx]
    for value_idx, _, chunk in arr.changes():
        if not isinstance(chunk, tuple):
            assert chunk.dtype == base.dtype
            final[value_idx] = chunk
    assert_array_equal(final, expect, strict=True)

    # Test __iter__
    assert_array_equal(list(arr), list(expect), strict=True)


class MyArray(Mapping):
    """A minimal numpy-like read-only array"""

    def __init__(self, arr):
        self.arr = np.asarray(arr)

    @property
    def shape(self):
        return self.arr.shape

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def itemsize(self):
        return self.arr.itemsize

    @property
    def dtype(self):
        return self.arr.dtype

    def astype(self, dtype):
        return MyArray(self.arr.astype(dtype))

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.arr, dtype=dtype, copy=copy)

    def __getitem__(self, idx):
        return MyArray(self.arr[idx])

    def __iter__(self):
        return (MyArray(row) for row in self.arr)

    def __len__(self):
        return len(self.arr)


def test_array_like_setitem():
    arr = StagedChangesArray.full((3, 3), (3, 1), dtype="f4")
    arr[:2, :2] = MyArray([[1, 2], [3, 4]])
    assert all(isinstance(slab, np.ndarray) for slab in arr.slabs)
    assert_array_equal(arr, np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]], dtype="f4"))


def test_array_like_from_array():
    orig = MyArray(np.arange(9).reshape(3, 3))
    arr = StagedChangesArray.from_array(orig, (2, 2))
    assert isinstance(arr.full_slab, np.ndarray)
    assert arr.n_base_slabs == 2

    # Because MyArray supports views (see MyArray.__getitem__), then the base slabs
    # must be views of the original array.
    for slab in arr.base_slabs:
        assert isinstance(slab, MyArray)
        assert slab.arr.base is orig.arr.base

    assert_array_equal(arr, orig.arr, strict=True)


def test_array_like_from_slabs():
    base_slab = MyArray(np.arange(9).reshape(3, 3))
    arr = StagedChangesArray(
        shape=(3, 3),
        chunk_size=(2, 3),
        base_slabs=[base_slab],
        slab_indices=[[1], [1]],
        slab_offsets=[[0], [2]],
    )
    assert arr.slabs[1] is base_slab
    assert_array_equal(arr, base_slab.arr, strict=True)


def test_asarray():
    arr = StagedChangesArray.full((2, 2), (2, 2))
    assert isinstance(np.asarray(arr), np.ndarray)
    assert_array_equal(np.asarray(arr), arr)

    assert isinstance(np.asarray(arr, copy=True), np.ndarray)
    with pytest.raises(TypeError):
        np.asarray(arr, copy=False)

    assert isinstance(np.asarray(arr, dtype="i1"), np.ndarray)
    assert np.asarray(arr, dtype="i1").dtype == "i1"


def test_load():
    arr = StagedChangesArray.from_array(np.arange(4).reshape(2, 2), (2, 1), 42)
    arr.resize((3, 2))
    assert_array_equal(arr.slab_indices, [[1, 2], [0, 0]])
    assert_array_equal(arr, np.array([[0, 1], [2, 3], [42, 42]]))

    arr.load()
    assert_array_equal(arr.slab_indices, [[3, 3], [0, 0]])
    assert_array_equal(arr, np.array([[0, 1], [2, 3], [42, 42]]))

    # Base slabs were dereferenced; full slab was not
    assert_array_equal(arr.slabs[0], [[42], [42]])
    assert arr.slabs[1] is None
    assert arr.slabs[2] is None

    arr.load()  # No-op
    assert_array_equal(arr.slab_indices, [[3, 3], [0, 0]])
    assert_array_equal(arr, np.array([[0, 1], [2, 3], [42, 42]]))


def test_shrinking_dereferences_slabs():
    arr = StagedChangesArray.from_array([[1, 2, 3, 4, 5, 6, 7]], (1, 2), fill_value=42)
    arr[0, -1] = 8
    arr.resize((1, 10))
    assert_array_equal(arr.slab_indices, [[1, 2, 3, 5, 0]])
    assert_array_equal(arr, [[1, 2, 3, 4, 5, 6, 8, 42, 42, 42]])
    arr.resize((1, 3))
    assert_array_equal(arr.slab_indices, [[1, 2]])
    assert arr.slabs[3:] == [None, None, None]
    assert_array_equal(arr, [[1, 2, 3]])

    arr.resize((0, 0))
    assert_array_equal(arr.slab_indices, np.empty((0, 0)))
    # The base slab is never dereferenced
    assert_array_equal(arr.slabs[0], [[42, 42]])
    assert arr.slabs[1:] == [None, None, None, None, None]


def test_copy():
    a = StagedChangesArray.from_array([[1, 2]], chunk_size=(1, 1), fill_value=42)
    a.resize((1, 3))
    a[0, 1] = 4
    assert_array_equal(a, [[1, 4, 42]])
    assert_array_equal(a.slab_indices, [[1, 3, 0]])

    b = a.copy(deep=True)
    assert b.slabs[0] is a.slabs[0]  # full slab is shared
    assert b.slabs[1] is a.slabs[1]  # base slabs are shared
    # staged slabs are deep-copied
    b[0, 1] = 5
    assert_array_equal(b.slab_indices, [[1, 3, 0]])
    assert_array_equal(a, [[1, 4, 42]])
    assert_array_equal(b, [[1, 5, 42]])

    c = a.copy(deep=False)
    assert b.slabs[0] is a.slabs[0]  # full slab is shared
    assert c.slabs[1] is a.slabs[1]  # base slabs are shared
    # staged slabs are turned into read-only views
    assert c.slabs[2].base is a.slabs[2].base
    assert not c.slabs[2].flags.writeable
    with pytest.raises(ValueError):
        c[0, 1] = 5


def test_astype():
    a = StagedChangesArray.full((2, 2), (2, 2), dtype="i1")
    a[0, 0] = 1

    # astype() with no type change deep-copies
    b = a.astype("i1")
    b[0, 0] = 2
    b.refill(123)
    assert a[0, 0] == 1
    assert a[0, 1] == 0

    # Create array with base slabs, full chunks, and staged slabs
    a = StagedChangesArray.from_array(
        np.asarray([[1, 2, 3]], dtype="f4"), chunk_size=(1, 1)
    )
    a.resize((1, 2))  # dereference a slab
    a.resize((1, 3))
    a[0, 1] = 4
    assert a.n_base_slabs == 3
    assert_array_equal(a.slab_indices, [[1, 4, 0]])

    b = a.astype("i2")

    assert_array_equal(a.slab_indices, [[1, 4, 0]])
    # all slabs have been loaded
    assert_array_equal(b.slab_indices, [[5, 4, 0]])

    assert a.dtype == "f4"
    for slab in a.slabs:
        assert slab is None or slab.dtype == "f4"

    assert b.dtype == "i2"
    for slab in b.slabs:
        assert slab is None or slab.dtype == "i2"

    assert_array_equal(a, np.array([[1, 4, 0]], dtype="f4"), strict=True)
    assert_array_equal(b, np.array([[1, 4, 0]], dtype="i2"), strict=True)


def test_refill():
    # Actual base slabs can entirely or partially contain the fill_value
    a = StagedChangesArray.from_array([[1], [42]], chunk_size=(2, 1), fill_value=42)
    a.resize((2, 3))  # Create full chunks full of 42
    a[0, 2] = 2  # Create staged slab with a non-42 and a 42
    assert_array_equal(a, [[1, 42, 2], [42, 42, 42]])
    assert_array_equal(a.slab_indices, [[1, 0, 2]])

    b = a.refill(99)
    # a is unchanged
    assert_array_equal(a, [[1, 42, 2], [42, 42, 42]])
    assert_array_equal(a.slab_indices, [[1, 0, 2]])

    # full slabs -> still full slabs, but slabs[0] has changed value
    # base slabs -> staged slabs with 42 points replaced with 99
    # staged slabs -> same slab index, but 42 points have been replaced with 99
    assert_array_equal(b, [[1, 99, 2], [99, 99, 99]])
    assert_array_equal(b.slab_indices, [[3, 0, 2]])


def test_big_O_performance():
    """Test that __getitem__ and __setitem__ performance is
    O(selected number of chunks) and not O(total number of chunks).
    """

    def benchmark(shape):
        arr = StagedChangesArray.full(shape, (1, 1))
        # Don't measure page faults on first access to slab_indices and slab_offsets.
        # In the 10 chunks use case, it's almost certainly reused memory.
        _ = arr[0, -3:]

        t0 = time.thread_time()
        # Let's access the end, just in case there's something that
        # performs a full scan which stops when the selection ends.
        arr[0, -1] = 42
        assert arr[0, -1] == 42
        assert arr[0, -2] == 0
        t1 = time.thread_time()
        return t1 - t0

    # trivially sized baseline: 10 chunks
    a = benchmark((1, 10))

    # 10 million chunks, small rulers
    # Test will trip if __getitem__ or __setitem__ perform a
    # full scan of slab_indices and/or slab_offsets anywhere
    b = benchmark((2_500, 4_000))
    np.testing.assert_allclose(b, a, rtol=0.2)

    # 10 million chunks, long rulers
    # Test will trip if __getitem__ or __setitem__ construct or iterate upon a ruler as
    # long as the number of chunks along one axis, e.g. np.arange(mapper.n_chunks)
    c = benchmark((1, 10_000_000))
    np.testing.assert_allclose(c, a, rtol=0.2)


# TODO weird dtypes
# TODO __repr__
# TODO invalid parameters
