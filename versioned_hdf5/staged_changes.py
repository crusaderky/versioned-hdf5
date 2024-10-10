# Note: this entire module is compiled by cython with wraparound=False
# See meson.build for details

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, cast

import cython
import numpy as np
from cython import bint, ssize_t
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .cytools import ceil_a_over_b, count2stop, np_hsize_t
from .slicetools import read_many_slices
from .subchunk_map import IndexChunkMapper, IntegerMapper, index_chunk_mappers

if cython.compiled:
    from cython.cimports.versioned_hdf5.cytools import (  # type: ignore
        ceil_a_over_b,
        count2stop,
        hsize_t,
    )
    from cython.cimports.versioned_hdf5.subchunk_map import (  # type: ignore
        IndexChunkMapper,
    )

T = TypeVar("T", bound=np.generic)


class StagedChangesArray(Generic[T]):
    """A numpy array-like that wraps around an underlying read-only, potentially sparse
    array-like of chunks, known as the base slab, and presents it reordered and reshaped
    to the final user.

    The base slab is a concatenation of chunks along axis 0, with shape
    (n*chunk_size[0], *chunk_size[1:]) with n <= total number of chunks.

    There must be a mapping from each chunk of the presented array to an offset
    along axis 0 of the base slab; alternatively the chunk can be tagged as being
    completely full of the fill_value.

    e.g.

    chunks in the         chunks in
    presented array       the base slab
    (- = fill_value)      (* = not referenced)

    3 2 -                 0*
    - 5 1                 1
    - - 6                 2
                    ==>   3
                          4*
                          5
                          6

    All changes to the data or the shape are stored in memory, on top of additional
    slabs backed by plain numpy arrays. Nothing writes back to the base slab.

    High level documentation on how the class works internally: :doc:`staged_changes`.

    Parameters
    ----------
    shape:
        The shape of the presented array
    chunk_size:
        The shape of each chunk
    base_slab:
        A read-only numpy-like object containint the baseline data.
        It must have shape (n*chunk_size[0], *chunk_size[1:]) for n sufficiently large
        to accomodate all chunks that aren't covered in fill_value.
    slab_indices:
        Numpy array of integers with shape equal to the number of chunks along each
        dimension, set to 0 for chunks that are covered by the base slab and 1 for
        chunks that are full of fill_value.
        It will be modified in place.
    slab_offsets:
        Numpy array of integers with shape matching slab_indices, mapping each chunk
        that isn't full of fill_value to the offset along axis 0 of the base slab.
        Chunks full of fill_value must be set to zero.
        It will be modified in place.
    fill_value: optional
        Value to fill chunks with where slab_indices=1. Defaults to 0.
    """

    #: current shape of the StagedChangesArray, e.g. downstream of resize()
    shape: tuple[int, ...]

    #: True if the user called resize() to alter the shape of the array; False otherwise
    _resized: bool

    #: size of the tiles that will be modified at once. A write to
    #: less than a whole chunk will cause the remainder of the chunk
    #: to be read from the underlying array.
    chunk_size: tuple[int, ...]

    #: Map from each chunk to the index of the corresponding slab in the slabs list
    slab_indices: NDArray[np_hsize_t]

    #: Offset of each chunk within the corresponding slab along axis 0
    slab_offsets: NDArray[np_hsize_t]

    #: Slabs of data, each containing one or more chunk stacked on top of each other
    #: along axis 0.
    #:
    #: slabs[0] is the slab containing the unmodified data and can be any read-only
    #: numpy-like object.
    #: slabs[1] contains exactly one chunk full of fill_value. It's broadcasted to the
    #: chunk_size and read only.
    #: slabs[2:] contain the modified chunks.
    #:
    #: Edge slabs that don't fully cover the chunk_size are padded with uninitialized
    #: cells; the shape of each slab is always (n*chunk_size[0], *chunk_size[1:]).
    #: Deleted slabs are replaced with None.
    slabs: list[NDArray[T] | None]

    __slots__ = tuple(__annotations__)

    def __init__(
        self,
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        base_slab: NDArray[T],
        slab_indices: ArrayLike,
        slab_offsets: ArrayLike,
        fill_value: Any | None = None,
    ):
        ndim = len(shape)
        if any(s < 0 for s in shape):
            raise ValueError("shape must be non-negative")
        if any(c <= 0 for c in chunk_size):
            raise ValueError("chunk_size must be strictly positive")
        if len(chunk_size) != ndim:
            raise ValueError("shape and chunk_size must have the same length")
        if base_slab.shape[1:] != shape[1:]:
            raise ValueError(f"{base_slab.shape[1:]=}; expected {shape[1:]})")

        self.shape = shape
        self.chunk_size = chunk_size
        self._resized = False

        if fill_value is None:
            # Unlike 0.0, this works for weird dtypes such as np.void
            fill_value = np.zeros((), dtype=base_slab.dtype)
        else:
            fill_value = np.array(fill_value, dtype=base_slab.dtype, copy=True)
            if fill_value.ndim != 0:
                raise ValueError("fill_value must be a scalar")
        assert fill_value.base is None

        self.slabs = [base_slab, np.broadcast_to(fill_value, chunk_size)]

        nchunks = self.nchunks
        self.slab_indices = slab_indices = np.asarray(slab_indices, dtype=np_hsize_t)
        self.slab_offsets = slab_offsets = np.asarray(slab_offsets, dtype=np_hsize_t)

        if slab_indices.shape != nchunks:
            raise ValueError(f"{slab_indices.shape=}; expected {nchunks}")
        if slab_offsets.shape != nchunks:
            raise ValueError(f"{slab_offsets.shape=}; expected {nchunks}")

    @property
    def dtype(self) -> np.dtype[T]:
        return self.fill_value.dtype

    @property
    def fill_value(self) -> NDArray[T]:
        return self.slabs[1].base  # type: ignore

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Size in bytes of this array if it were completely loaded. Actual used memory
        can be less, as only modified chunks are stored in memory and even then they may
        be CoW copies of another array.
        """
        return self.size * self.fill_value.nbytes

    @property
    def nchunks(self) -> tuple[int, ...]:
        """Number of chunks on each axis"""
        return tuple(ceil_a_over_b(s, c) for s, c in zip(self.shape, self.chunk_size))

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self) -> NDArray[T]:
        return self[()]

    @property
    def has_changes(self) -> bool:
        """Return True if any chunks have been modified or if a lazy transformation
        took place; False otherwise.
        """
        return len(self.slabs) > 2 or self._resized

    def __repr__(self) -> str:
        return (
            f"StagedChangesArray<shape={self.shape}, chunk_size={self.chunk_size}, "
            f"dtype={self.dtype}, fill_value={self.fill_value.item()}, "
            f"{len(self.slabs) - 2} slabs of modified chunks>\n"
            f"slab_indices:\n{self.slab_indices}"
            f"slab_offsets:\n{self.slab_offsets}"
        )

    def _changes_plan(self) -> ChangesPlan:
        return ChangesPlan(
            chunk_size=self.chunk_size,
            shape=self.shape,
            slab_indices=self.slab_indices,
            slab_offsets=self.slab_offsets,
        )

    def _getitem_plan(self, idx: Any) -> GetItemPlan:
        return GetItemPlan(
            idx,
            chunk_size=self.chunk_size,
            shape=self.shape,
            slab_indices=self.slab_indices,
            slab_offsets=self.slab_offsets,
        )

    def _setitem_plan(self, idx: Any) -> SetItemPlan:
        return SetItemPlan(
            idx,
            chunk_size=self.chunk_size,
            shape=self.shape,
            slab_indices=self.slab_indices,
            slab_offsets=self.slab_offsets,
            nslabs=len(self.slabs),
        )

    def _resize_plan(self, shape: tuple[int, ...]) -> ResizePlan:
        return ResizePlan(
            chunk_size=self.chunk_size,
            old_shape=self.shape,
            new_shape=shape,
            slab_indices=self.slab_indices,
            slab_offsets=self.slab_offsets,
            nslabs=len(self.slabs),
        )

    def _load_plan(self) -> LoadPlan:
        return LoadPlan(
            chunk_size=self.chunk_size,
            shape=self.shape,
            slab_indices=self.slab_indices,
            slab_offsets=self.slab_offsets,
            nslabs=len(self.slabs),
        )

    def _get_slab(
        self, idx: int | None, default: NDArray[T] | None = None
    ) -> NDArray[T]:
        slab = self.slabs[idx] if idx is not None else default
        assert slab is not None
        return slab

    def changes(
        self, *, load_base: bool = False
    ) -> Iterator[tuple[tuple[slice, ...], NDArray[T] | tuple[slice, ...]]]:
        """Yield all the changed chunks so far, as tuples of

        - slice index in the base array
        - chunk value, if modified, or slice of the base slab along axis 0 otherwise
          (the slices on axes 1+ are identical to the matching ones in the base array)

        This lets you update the base array:

        >> for idx, value in staged_array.changes():
        ..     if not isinstance(value, slice):
        ..         base[idx] = value

        Note
        ----
        If a chunk is full of the fill_value it will not be yielded by this method.

        """
        if not self.has_changes and not load_base:
            return

        plan = self._changes_plan()

        for base_slice, slab_idx, slab_slice in plan.chunks:
            if slab_idx == 0:
                yield base_slice, slab_slice
            else:
                slab = self._get_slab(slab_idx)
                chunk = slab[slab_slice]
                yield base_slice, chunk

    def __getitem__(self, idx: Any) -> NDArray[T]:
        """Get a slice of data from the array. This reads from the staged slabs
        in memory when available and from either the base slab or the fill_value
        otherwise.
        """
        plan = self._getitem_plan(idx)

        out = np.empty(plan.output_shape, dtype=self.dtype)
        out_view = out[plan.output_view]

        for tplan in plan.transfers:
            src_slab = self._get_slab(tplan.src_slab_idx)
            assert tplan.dst_slab_idx is None
            tplan.transfer(src_slab, out_view)

        return out

    def _apply_mutating_plan(
        self, plan: MutatingPlan, default_slab: NDArray[T] | None = None
    ) -> None:
        """Implement common workflow of __setitem__, resize, and load."""
        for shape in plan.append_slabs:
            self.slabs.append(np.empty(shape, dtype=self.dtype))

        for tplan in plan.transfers:
            src_slab = self._get_slab(tplan.src_slab_idx, default_slab)
            dst_slab = self._get_slab(tplan.dst_slab_idx)
            tplan.transfer(src_slab, dst_slab)

        for slab_idx in plan.drop_slabs:
            self.slabs[slab_idx] = None

        self.slab_indices = plan.slab_indices
        self.slab_offsets = plan.slab_offsets

    def __setitem__(self, idx: Any, value: ArrayLike) -> None:
        """Update the slabs containing the chunks selected by the index.

        Slab 0 (the base slab) and slab 1 (the fill_value slab) are read-only. If the
        selected chunks lay on either of them, append a new empty slab with the
        necessary space to hold all such chunks, then copy the chunks from slab 0 or 1
        to the new slab, and finally update the new slab from the value parameter.
        """
        plan = self._setitem_plan(idx)
        if not plan.mutates:
            return

        # Preprocess value parameter
        # Avoid double deep-copy of array-like objects that support the __array_*
        # interface (e.g. sparse arrays).
        if not hasattr(value, "dtype") or not hasattr(value, "shape"):
            value = np.asarray(value, self.dtype)
        else:
            value = cast(np.ndarray, value).astype(self.dtype, copy=False)
        value = cast(NDArray[T], value)

        if plan.value_shape != value.shape:
            value = np.broadcast_to(value, plan.value_shape)
        value_view = value[plan.value_view]

        self._apply_mutating_plan(plan, value_view)

    def resize(self, shape: tuple[int, ...]) -> None:
        """Change the array shape in place and fill new elements with self.fill_value.

        When enlarging, edge chunks which are not exactly divisible by chunk size are
        partially filled with fill_value. This is a transfer from slab 1 to slab >=2.

        If such slabs are not already in memory, they are first loaded.
        Just like in __setitem__, this appends a new empty slab to the slabs list,
        then transfers from slab 0 to the new slab, and finally transfers from slab 1
        to the new slab.

        Slabs that are no longer needed are dereferenced; their location in the slabs
        list is replaced with None.
        This may cause slab 0 to be dereferenced, but never slab 1.
        """
        if self.shape == shape:
            return

        plan = self._resize_plan(shape)
        assert plan.mutates
        self._apply_mutating_plan(plan)
        self.shape = shape
        self._resized = True

    def load(self) -> None:
        """Load all chunks that are not yet in memory from the base array."""
        if self.slabs[0] is None:
            return
        plan = self._load_plan()
        assert plan.mutates
        self._apply_mutating_plan(plan)

    def copy(self, deep: bool = True) -> StagedChangesArray[T]:
        """Return a copy of self. If deep=True, slabs are deep-copied.
        If deep=False, the copy gets read-only views of the slabs and attempts to
        change them will fail.

        Read-only slabs 0 and 1 are not copied.
        """
        out = copy.copy(self)
        out.slab_indices = self.slab_indices.copy()
        out.slab_offsets = self.slab_offsets.copy()
        out.slabs = self.slabs[:2]
        for slab in self.slabs[2:]:
            if slab is not None:
                if deep:
                    slab = slab.copy()
                else:
                    slab = slab[()]
                    slab.flags.writeable = False
            out.slabs.append(slab)
        return out

    def astype(self, dtype: DTypeLike, casting: Any = "unsafe") -> StagedChangesArray:
        """Return a new StagedChangesArray with a different dtype.

        Chunks that are not yet in memory are loaded.
        """
        if self.dtype == dtype:
            return self.copy(deep=True)

        out = self.copy(deep=False)
        out.load()  # Create new slabs and set slab 0 to None
        out.slabs[1] = np.broadcast_to(
            out.fill_value.astype(dtype, casting=casting),
            out.chunk_size,
        )
        for i, slab in enumerate(out.slabs[2:], 2):
            if slab is not None:
                out.slabs[i] = slab.astype(dtype, casting=casting)
        return out

    def refill(self, fill_value: Any) -> StagedChangesArray[T]:
        """Create a copy of self with changed fill_value."""
        fill_value = np.asarray(fill_value, self.dtype)
        if fill_value.ndim != 0:
            raise ValueError("fill_value must be a scalar")

        out = self.copy(deep=True)
        if fill_value != self.fill_value:
            out.load()
            out.slabs[1] = np.broadcast_to(fill_value, out.chunk_size)
            for slab in out.slabs[2:]:
                if slab is not None:
                    slab[slab == self.fill_value] = fill_value

        return out

    @staticmethod
    def full(
        shape: tuple[int, ...],
        chunk_size: tuple[int, ...],
        fill_value: Any | None = None,
        dtype: DTypeLike | None = None,
    ) -> StagedChangesArray:
        """Create a new StagedChangesArray with all chunks already in memory and
        full of fill_value.
        It won't consume any significant amounts of memory until it's modified.
        """
        nchunks = tuple(ceil_a_over_b(s, c) for s, c in zip(shape, chunk_size))
        if fill_value is not None:
            dtype = np.array(fill_value, dtype=dtype).dtype

        out = StagedChangesArray(
            shape=shape,
            chunk_size=chunk_size,
            base_slab=np.empty((0, *shape[1:]), dtype=dtype),  # dummy
            slab_indices=np.ones(nchunks, dtype=np_hsize_t),
            slab_offsets=np.zeros(nchunks, dtype=np_hsize_t),
            fill_value=fill_value,
        )
        out.slabs[0] = None
        return out

    @staticmethod
    def from_array(
        arr: NDArray[T],
        chunk_size: tuple[int, ...],
        fill_value: Any | None = None,
        as_base_slab: bool = False,
    ) -> StagedChangesArray:
        """Create a new StagedChangesArray from an existing array-like, which will be
        immediately deep-copied.
        """
        arr = np.asarray(arr)
        out = StagedChangesArray.full(arr.shape, chunk_size, fill_value, arr.dtype)
        out[()] = arr

        if as_base_slab:
            assert (out.slab_indices == 2).all()
            out.slab_indices[()] = 0
            out.slabs = [out.slabs[2], out.slabs[1]]

        return out


@cython.cclass
@dataclass(init=False)
class TransferPlan:
    """Instructions to transfer data:

    - from a slab to the return value of __getitem__, or
    - from the value parameter of __setitem__ to a slab, or
    - between two slabs.
    """

    #: Index of the source slab in StagedChangesArray.slabs.
    #: During __setitem__, it can be None to indicate the value parameter.
    src_slab_idx: int | None

    #: Index of the destination slab in StagedChangesArray.slabs.
    #: During __getitem__, it's always None to indicate the output array.
    dst_slab_idx: int | None

    #: Parameters for read_many_slices().
    src_start: hsize_t[:, :]
    dst_start: hsize_t[:, :]
    count: hsize_t[:, :]
    src_stride: hsize_t[:, :]
    dst_stride: hsize_t[:, :]

    def __init__(
        self,
        src_slab_idx: int | None,
        dst_slab_idx: int | None,
        nslices: int,
        ndim: int,
        src_stride: bint,
        dst_stride: bint,
    ):
        self.src_slab_idx = src_slab_idx
        self.dst_slab_idx = dst_slab_idx

        nbufs = 3 + src_stride + dst_stride
        buf: hsize_t[:, :, :] = np.empty((nbufs, nslices, ndim), dtype=np_hsize_t)
        self.src_start = buf[0]
        self.dst_start = buf[1]
        self.count = buf[2]

        if src_stride and dst_stride:
            self.src_stride = buf[3]
            self.dst_stride = buf[4]
        else:
            stride_buf: hsize_t[:, :] = np.broadcast_to(
                # read_many_slices indices must be contiguous along the columns,
                # or they will be deep-copied
                np.ones(ndim, dtype=np_hsize_t),
                (nslices, ndim),
            )
            self.src_stride = buf[3] if src_stride else stride_buf
            self.dst_stride = buf[3] if dst_stride else stride_buf

    @cython.ccall
    def transfer(self, src: NDArray[T], dst: NDArray[T]):
        read_many_slices(
            src,
            dst,
            self.src_start,
            self.dst_start,
            self.count,
            self.src_stride,
            self.dst_stride,
        )

    def __len__(self) -> int:
        return self.src_start.shape[0]

    @cython.cfunc
    def _repr_idx(self, i: ssize_t, start: hsize_t[:, :], stride: hsize_t[:, :]) -> str:
        """Return a string representation of the i-th row"""
        ndim = start.shape[1]
        idx = []
        for j in range(ndim):
            start_ij = start[i, j]
            count_ij = self.count[i, j]
            stride_ij = stride[i, j]
            stop_ij = count2stop(start_ij, count_ij, stride_ij)
            idx.append(slice(start_ij, stop_ij, stride_ij))
        return _fmt_fancy_index(tuple(idx))

    def __repr__(self) -> str:
        """This is meant to be incorporated by the __repr__ method of the
        other *Plan classes
        """
        src = f"slabs[{i}]" if (i := self.src_slab_idx) is not None else "value"
        dst = f"slabs[{i}]" if (i := self.dst_slab_idx) is not None else "out"
        nslices = self.src_start.shape[0]
        s = ""
        for i in range(nslices):
            src_idx = self._repr_idx(i, self.src_start, self.src_stride)
            dst_idx = self._repr_idx(i, self.dst_start, self.dst_stride)
            s += f"\n  {dst}[{dst_idx}] = {src}[{src_idx}]"

        return s

    def __eq__(self, other):
        # Hack around Cython bug
        raise NotImplementedError()


@cython.cclass
@dataclass(init=False)
class GetItemPlan:
    """Instructions to execute StagedChangesArray.__getitem__"""

    #: Shape of the array returned by __getitem__
    output_shape: tuple[int, ...]

    #: Index to slice the output array to add extra dimensions to ensure it's got
    #: the same dimensionality as the base array
    output_view: tuple[slice | None, ...]

    transfers: list[TransferPlan]

    def __init__(
        self,
        idx: Any,
        chunk_size: tuple[int, ...],
        shape: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
    ):
        """Generate instructions to execute StagedChangesArray.__getitem__

        Parameters
        ----------
        idx:
            Arbitrary numpy fancy index passed as a parameter to __getitem__

        All other parameters are the matching attributes of StagedChangesArray
        """
        idx, mappers = index_chunk_mappers(idx, chunk_size, shape)
        self.output_shape = idx.newshape(shape)
        self.output_view = same_shape_view(mappers)
        self.transfers = []

        if not mappers:
            # Empty selection
            return

        chunks = _chunks_in_selection(slab_indices, slab_offsets, mappers, sort=True)
        self.transfers.extend(
            _build_transfers(
                slab_indices,
                slab_offsets,
                mappers,
                chunks,
                src_slab_idx="chunks",
                dst_slab_idx=None,
            )
        )

    @property
    def head(self) -> str:
        ntransfers = sum(len(tplan) for tplan in self.transfers)
        return (
            f"GetItemPlan<output_shape={self.output_shape}, "
            f"output_view={self.output_view}, {ntransfers} transfers>"
        )

    def __repr__(self) -> str:
        return self.head + "".join(str(tplan) for tplan in self.transfers)


@cython.cclass
@dataclass
class MutatingPlan:
    """Common ancestor of all plans that mutate StagedChangesArray"""

    #: Updated metadata arrays of StagedChangesArray
    slab_indices: NDArray[np_hsize_t]
    slab_offsets: NDArray[np_hsize_t]

    #: Create new uninitialized slabs with the given shapes
    #: and append them StagedChangesArray.slabs
    append_slabs: list[tuple[int, ...]] = field(init=False, default_factory=list)

    #: data transfers between slabs or from the __setitem__ value to a slab.
    #: dst_slab_idx can include the slabs just created by append_slabs.
    transfers: list[TransferPlan] = field(init=False, default_factory=list)

    #: indices of StagedChangesArray.slabs to replace with None,
    #: thus dereferencing the slab. This must happen *after* the transfers.
    drop_slabs: list[int] = field(init=False, default_factory=list)

    @property
    def mutates(self) -> bool:
        """Return True if this plan alters the state of the
        StagedChangesArray in any way; False otherwise
        """
        return bool(self.transfers or self.drop_slabs)

    @property
    def head(self) -> str:
        """This is meant to be incorporated by the head() property of the subclasses"""
        ntransfers = sum(len(tplan) for tplan in self.transfers)
        return (
            f"append {len(self.append_slabs)} empty slabs, "
            f"{ntransfers} transfers, drop ({len(self.drop_slabs)} slabs>"
        )

    def __repr__(self) -> str:
        """This is meant to be incorporated by the __repr__ method of the subclasses"""
        s = self.head

        if self.append_slabs:
            max_slab_idx = self.slab_indices.max()
            assert max_slab_idx > 1
            slab_start_idx = max_slab_idx - len(self.append_slabs) - 1
            for slab_idx, shape in enumerate(self.append_slabs, slab_start_idx):
                s += f"\n  slabs.append(np.empty({shape}))  # slabs[{slab_idx}]"

        s += "".join(str(tplan) for tplan in self.transfers)
        for slab_idx in self.drop_slabs:
            s += f"\n  slabs[{slab_idx}] = None"
        s += f"\nslab_indices:\n{self.slab_indices}"
        s += f"\nslab_offsets:\n{self.slab_offsets}"
        return s


@cython.cclass
@dataclass(init=False)
class SetItemPlan(MutatingPlan):
    """Instructions to execute StagedChangesArray.__setitem__"""

    #: Shape the value parameter must be broadcasted to
    value_shape: tuple[int, ...]

    #: Index to slice the value parameter array to add extra dimensions to ensure it's
    #: got the same dimensionality as the base array
    value_view: tuple[slice | None, ...]

    def __init__(
        self,
        idx: Any,
        chunk_size: tuple[int, ...],
        shape: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
        nslabs: int,
    ):
        """Generate instructions to execute StagedChangesArray.__setitem__.

        Parameters
        ----------
        idx:
            Arbitrary numpy fancy index passed as a parameter to __setitem__
        nslabs:
            len(StagedChangesArray.slabs)

        All other parameters are the matching attributes of StagedChangesArray.
        """
        super().__init__(slab_indices, slab_offsets)

        idx, mappers = index_chunk_mappers(idx, chunk_size, shape)
        self.value_shape = idx.newshape(shape)
        self.value_view = same_shape_view(mappers)

        # We'll deep-copy later, only if needed
        if not mappers:
            # Empty selection
            return

        # Two pass query to _chunks_in_selection:
        # 1. Get all chunks in slabs 0 (base slab) and 1 (fill_value) that are only
        #    partially covered by the selection. If there are any, append a new slab
        #    and transfers those chunks to it.
        # 2. Update all chunks, which now lay on slabs[2:]

        # Pass 1
        chunks = _chunks_in_selection(
            slab_indices,
            slab_offsets,
            mappers,
            filter=lambda slab_idx: slab_idx < 2,
            only_partial=True,
            sort=True,
        )
        nchunks = chunks.shape[0]
        if nchunks:
            self.slab_indices = slab_indices.copy()
            self.slab_offsets = slab_offsets.copy()
            self.append_slabs = [(nchunks * chunk_size[0],) + chunk_size[1:]]

            _, whole_mappers = index_chunk_mappers((), chunk_size, shape)
            self.transfers.extend(
                _build_transfers(
                    self.slab_indices,  # Modified in place
                    self.slab_offsets,  # Modified in place
                    whole_mappers,
                    chunks,
                    src_slab_idx="chunks",
                    dst_slab_idx=nslabs,
                )
            )
            if not (self.slab_indices == 0).any():
                self.drop_slabs = [0]

        # Pass 2
        chunks = _chunks_in_selection(
            self.slab_indices,  # Updated deep copy
            self.slab_offsets,  # Updated deep copy
            mappers,
            sort=True,
        )
        self.transfers.extend(
            _build_transfers(
                slab_indices,
                slab_offsets,
                mappers,
                chunks,
                src_slab_idx=None,
                dst_slab_idx="chunks",
            )
        )

    @property
    def head(self) -> str:
        return (
            f"SetItemPlan<value_shape={self.value_shape}, "
            f"value_view={self.value_view}, " + super().head
        )


@cython.cclass
@dataclass(init=False)
class LoadPlan(MutatingPlan):
    """Load all chunks that have not been loaded yet from slab 0."""

    def __init__(
        self,
        chunk_size: tuple[int, ...],
        shape: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
        nslabs: int,
    ):
        super().__init__(slab_indices, slab_offsets)
        _, mappers = index_chunk_mappers((), chunk_size, shape)
        if not mappers:
            return  # size 0

        chunks = _chunks_in_selection(
            slab_indices, slab_offsets, mappers, lambda slab_idx: slab_idx == 0
        )
        nchunks = chunks.shape[0]

        if nchunks == 0:
            return

        self.slab_indices = slab_indices.copy()
        self.slab_offsets = slab_offsets.copy()
        self.append_slabs = [(nchunks * chunk_size[0],) + chunk_size[1:]]
        self.drop_slabs = [0]

        self.transfers.extend(
            _build_transfers(
                self.slab_indices,  # Modified in place
                self.slab_offsets,  # Modified in place
                mappers,
                chunks,
                src_slab_idx=0,
                dst_slab_idx=nslabs,
            )
        )

    @property
    def head(self) -> str:
        return "LoadPlan<" + super().head


@cython.cclass
@dataclass(init=False)
class ChangesPlan:
    """Instructions to execute StagedChangesArray.changes()."""

    #: List of all chunks that have either changed (slab_idx >= 2) or have not been
    #: altered (slab_idx = 0).
    #: This never includes full chunks.
    #:
    #: List of tuples of
    #: - index to slice the base array with
    #: - index of StagedChangesArray.slabs
    #: - index to slice the slab to retrieve the chunk value
    chunks: list[tuple[tuple[slice, ...], int, tuple[slice, ...]]]

    def __init__(
        self,
        chunk_size: tuple[int, ...],
        shape: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
    ):
        """Generate instructions to execute StagedChangesArray.changes().

        All parameters are the matching attributes of StagedChangesArray.
        """
        self.chunks = []

        _, mappers = index_chunk_mappers((), chunk_size, shape)
        if not mappers:
            return  # size 0

        # Build rulers of slices for each axis
        dst_slices: list[list[slice]] = []  # Slices in the represented array
        slab_slices: list[list[slice]] = [[]]  # Slices in the slab (except axis 0)

        mapper: IndexChunkMapper
        for mapper in mappers:
            dst_slices_ix = []
            a: ssize_t = 0
            assert mapper.n_chunks > 0  # not size 0
            for _ in mapper.n_chunks - 1:
                b = a + mapper.chunk_size
                dst_slices_ix.append(slice(a, b, 1))
                a = b
            b = a + mapper.last_chunk_size
            dst_slices_ix.append(slice(a, b, 1))
            dst_slices.append(dst_slices_ix)

        # slab slices on axis 0 must be built on the fly for each chunk,
        # as each chunk has a different slab offset
        mapper = mappers[0]
        axis0_chunk_sizes: hsize_t[:] = np.full(mapper.n_chunks, mapper.chunk_size)
        axis0_chunk_sizes[mapper.n_chunks - 1] = mapper.last_chunk_size

        # slab slices on the other axes can be built with a ruler
        # (and they'll be all the same except for the last chunk)
        for mapper in mappers[1:]:
            slab_slices.append(
                [slice(0, mapper.chunk_size, 1)] * (mapper.n_chunks - 1)
                + [slice(0, mapper.last_chunk_size, 1)]
            )

        # Find all non-full chunks
        chunks = _chunks_in_selection(
            slab_indices, slab_offsets, mappers, lambda slab_idx: slab_idx != 1
        )
        nchunks = chunks.shape[0]
        ndim = chunks.shape[1] - 2

        for i in range(nchunks):
            dst_ndslice = []
            for j in range(ndim):
                chunk_idx = chunks[i, j]
                dst_ndslice.append(dst_slices[j][chunk_idx])

            chunk_idx = chunks[i, 0]
            slab_idx = chunks[i, ndim]  # slab_indices[chunk_idx]
            start = chunks[i, ndim + 1]  # slab_offsets[chunk_idx]
            stop = start + axis0_chunk_sizes[chunk_idx]
            slab_ndslice = [slice(start, stop, 1)]
            for j in range(1, ndim):
                chunk_idx = chunks[i, j]
                slab_ndslice.append(slab_slices[j][chunk_idx])

            self.chunks.append((tuple(dst_ndslice), slab_idx, tuple(slab_ndslice)))

    @property
    def head(self) -> str:
        nmodified = sum(slab_idx > 0 for _, slab_idx, _ in self.chunks)
        return (
            f"ChangesPlan<{nmodified} modified chunks in memory, "
            f"{len(self.chunks) - nmodified} unmodified chunks>"
        )

    def __repr__(self) -> str:
        s = self.head
        fmt = _fmt_fancy_index
        for base_slice, slab_idx, slab_slice in self.chunks:
            s += f"\n  base[{fmt(base_slice)}] = slabs[{slab_idx}][{fmt(slab_slice)}]"
        return s


@cython.cclass
@dataclass(init=False)
class ResizePlan(MutatingPlan):
    """Instructions to execute StagedChangesArray.resize()"""

    def __init__(
        self,
        chunk_size: tuple[int, ...],
        old_shape: tuple[int, ...],
        new_shape: tuple[int, ...],
        slab_indices: NDArray[np_hsize_t],
        slab_offsets: NDArray[np_hsize_t],
        nslabs: int,
    ):
        """Generate instructions to execute StagedChangesArray.resize().

        Parameters
        ----------
        old_shape:
            StagedChangesArray.shape before the resize operation
        new_shape:
            StagedChangesArray.shape after the resize operation
        nslabs:
            len(StagedChangesArray.slabs)

        All other parameters are the matching attributes of StagedChangesArray.
        """
        if len(new_shape) != len(old_shape):
            raise ValueError(
                "Number of dimensions in resize from {old_shape} to {new_shape}"
            )

        super().__init__(slab_indices.copy(), slab_offsets.copy())

        # Shrink first, then enlarge
        shrinks = []
        enlarges = []
        for axis, (old_size, new_size) in enumerate(zip(old_shape, new_shape)):
            if new_size < old_size:
                shrinks.append(axis)
            elif new_size > old_size:
                enlarges.append(axis)

        prev_shape = old_shape
        for axis in shrinks + enlarges:
            next_shape = tuple(
                n if i == axis else p
                for i, (p, n) in enumerate(zip(prev_shape, new_shape))
            )
            nslabs = self._resize_along_axis(
                chunk_size, prev_shape, next_shape, axis, nslabs
            )
            prev_shape = next_shape
        assert next_shape == new_shape

        if any(p > n for p, n in zip(slab_indices.shape, self.slab_indices.shape)):
            # When shrinking, we could drop any slab.
            drop_slabs = np.setdiff1d(slab_indices, self.slab_indices)
            drop_slabs = drop_slabs[drop_slabs != 1]  # Never drop the full chunk
            self.drop_slabs = drop_slabs.tolist()
        elif (
            any(tplan.src_slab_idx == 0 for tplan in self.transfers)
            and self.slab_indices.all()
        ):
            # When enlarging, we could drop the base slab
            self.drop_slabs = [0]

    @cython.cfunc
    def _resize_along_axis(
        self,
        chunk_size: tuple[int, ...],
        old_shape: tuple[int, ...],
        new_shape: tuple[int, ...],
        axis: ssize_t,
        nslabs: ssize_t,
    ) -> ssize_t:
        """Resize along a single axis.

        Return the updated nslabs.
        """
        ndim = len(chunk_size)
        old_size: hsize_t = old_shape[axis]
        new_size: hsize_t = new_shape[axis]
        new_nchunks = ceil_a_over_b(new_size, chunk_size[axis])
        is_enlarge = new_size > old_size
        max_shape = new_shape if is_enlarge else old_shape

        # Fill new chunks with fill_value. This deep-copies the two arrays.
        self.slab_indices = _resize_array_along_axis(
            self.slab_indices, axis, new_nchunks, fill_value=1
        )
        self.slab_offsets = _resize_array_along_axis(
            self.slab_offsets, axis, new_nchunks, fill_value=0
        )

        # Two passes:
        # 1. Find edge chunks on slab 0 that were partial and became full, or
        #    vice versa, or remain partial but with a different size.
        #    Load them into a new slab at slab_idx=nslabs.
        # 2. Only when enlarging: find edge chunks on slab_idx >=2 that need filling
        #    with fill_value (this includes slabs we just copied from slab 0 in pass 1)
        #    and transfer from slab 1 (the full chunk).

        # We can do both passes in one go here
        idx = (
            (slice(None),) * axis + slice(old_size, new_size)
            if is_enlarge
            else slice(new_size, old_size)
        )
        _, mappers = index_chunk_mappers(idx, chunk_size, max_shape)
        chunks = _chunks_in_selection(
            self.slab_indices,
            self.slab_offsets,
            mappers,
            lambda slab_idx: slab_idx != 1 if is_enlarge else slab_idx == 0,
            only_partial=True,
            sort=is_enlarge,
        )

        if is_enlarge:
            cut: ssize_t = np.searchsorted(chunks[:, ndim], 2)
            chunks0 = chunks[:cut]
            chunks2plus = chunks[cut:]
        else:
            chunks0 = chunks

        # Pass 1
        nchunks = chunks0.shape[0]
        if nchunks > 0:
            self.append_slabs.append((nchunks * chunk_size[0],) + chunk_size[1:])
            _, whole_mappers = index_chunk_mappers((), chunk_size, max_shape)
            self.transfers.extend(
                _build_transfers(
                    self.slab_indices,  # Modified in place
                    self.slab_offsets,  # Modified in place
                    whole_mappers,
                    chunks0,
                    src_slab_idx=0,
                    dst_slab_idx=nslabs,
                )
            )
            nslabs += 1

        # Pass 2
        if is_enlarge and chunks2plus.shape[0] > 0:
            self.transfers.extend(
                _build_transfers(
                    self.slab_indices,  # Modified in place
                    self.slab_offsets,  # Modified in place
                    whole_mappers,
                    chunks2plus,
                    src_slab_idx=1,
                    dst_slab_idx="chunks",
                )
            )

        return nslabs

    @property
    def head(self) -> str:
        return "ResizePlan<" + super().head


@cython.ccall
def _resize_array_along_axis(
    arr: NDArray[T],
    axis: ssize_t,
    new_size: hsize_t,
    fill_value: T,
):  # -> NDArray[T]:
    """Either shrink or enlarge an array along the right edge of a single axis.

    Parameters
    ----------
    arr:
        The array to be transformed
    axis:
        The axis along which to resize
    new_size:
        The new size of the array along the axis
    fill_value:
        The value to fill the new elements with
    """
    i: ssize_t  # noqa: F841
    old_size: hsize_t = arr.shape[axis]

    if new_size < old_size:
        idx = (slice(None),) * axis + (slice(new_size),)
        return arr[idx]
    else:
        return np.pad(
            arr,
            [(0, new_size - old_size if i == axis else 0) for i in range(arr.ndim)],
            mode="constant",
            constant_values=fill_value,
        )


@cython.cfunc
def same_shape_view(mappers: list[IndexChunkMapper]) -> tuple[slice | None, ...]:
    """Return an index to apply to a __getitem__ return value or to a
    __setitem__ value parameter to obtain an array with the same dimensionality as the
    StagedChangesArray.
    """
    out: list[slice | None] = []
    last_scalar = -1
    for i, mapper in enumerate(mappers):
        if isinstance(mapper, IntegerMapper):
            out.append(None)
            last_scalar = i
        else:
            out.append(slice(None))

    if last_scalar == -1:
        return ()
    return tuple(out[: last_scalar + 1])


@cython.ccall
def _chunks_in_selection(
    slab_indices: NDArray[np_hsize_t],
    slab_offsets: NDArray[np_hsize_t],
    mappers: list[IndexChunkMapper],
    filter: Callable[[NDArray[np_hsize_t]], NDArray[np.bool]] | None = None,
    only_partial: bint = False,
    sort: bint = False,
) -> hsize_t[:, :]:
    """Find all chunks within a selection

    Parameters
    ----------
    slab_indices:
        StagedChangesArray.slab_indices
    slab_offsets:
        StagedChangesArray.slab_offsets
    mappers:
        Output of index_chunk_mappers()
    filter: optional
        Function to apply to a (selection of) slab_indices that returns a boolean mask.
        Chunks where the mask is False won't be returned.
    only_partial: optional
        If True, only return the chunks which are only partially covered by the
        selection along one or more axes.
    sort: optional
        If set to True, sort the result by slab_idx before returning it

    Returns
    -------
    2D view of indices where each row corresponds to a chunk and ndim+2 columns:

    - columns 0:ndim are the chunk indices
    - column ndim is the slab_idx (point of slab_indices)
    - column ndim+1 is the slab offset (point of slab_offsets)

    **Example**
    TODO

    >>> index = (slice(0, 20), slice(15, 45))
    >>> chunk_size = (10, 10)
    >>> shape = (30, 60)
    >>> chunk_states = np.array(
    ...   [[0, 0, 1, 4, 0, 0],
    ...    [0, 0, 0, 0, 5, 0],
    ...    [0, 0, 2, 0, 3, 0]]
    ... )
    >>> _, mappers = index_chunk_mappers(index, chunk_size, shape)
    >>> tuple(np.asarray(m.chunk_indices) for m in mappers)
    (array([0, 1]), array([1, 2, 3, 4]))
    >>> tuple(m.chunks_indexer() for m in mappers)
    (slice(0, 2, 1), slice(1, 5, 1))
    >>> tuple(m.whole_chunks_indexer() for m in mappers)
    (slice(0, 2, 1), slice(2, 4, 1))
    >>> _modified_chunks_in_selection(chunk_states, mappers)
    (array([[0, 2, 1],
           [0, 3, 4],
           [1, 4, 5]]), True)
    >>> _modified_chunks_in_selection(chunk_states, mappers, plus_whole=True)
    (array([[0, 2, 1],
           [0, 3, 4],
           [1, 2, 0],
           [1, 3, 0],
           [1, 4, 5]]), True)

    chunk_indices        = ([0, 1], [1, 2, 3, 4])
    whole chunks_indices = ([0, 1], [   2, 3   ])

    chunk_states    selection    plus_whole=False    plus_whole=True
    001400          .pppp.       ..14..              ..14..
    000050          .pwwp.       ....5.              ..005.
    002030          ......       ......              ......

                    (p=partial,  (0, 2) | 1          (0, 2) | 1
                     w=whole)    (0, 3) | 4          (0, 3) | 4
                                 (1, 4) | 5          (1, 2) | 0
                                                     (1, 3) | 0
                                                     (1, 4) | 5
    """
    axis: ssize_t
    mapper: IndexChunkMapper
    ndim = len(mappers)

    indexers = tuple([mapper.chunks_indexer() for mapper in mappers])

    if only_partial:
        has_partial = False
        whole_indexers = []
        for mapper, idx in zip(mappers, indexers):
            widx = mapper.whole_chunks_indexer()
            whole_indexers.append(widx)
            if not has_partial:
                if isinstance(idx, slice):
                    has_partial = not isinstance(widx, slice) or widx != idx
                else:
                    assert isinstance(idx, np.ndarray)
                    has_partial = not isinstance(widx, np.ndarray) or len(widx) < len(
                        idx
                    )

        if not has_partial == 0:
            # All chunks are wholly selected
            return np.empty((0, ndim + 2), dtype=np_hsize_t)

        # Add chunks that are wholly selected along all axes to the mask
        wholes = np.zeros_like(slab_indices)
        for axis, widx in enumerate(whole_indexers):
            widx_nd = (slice(None),) * axis + (widx,)
            wholes[widx_nd] += 1

    # Slice chunk_states
    indexers = _independent_ndindex(indexers)

    if only_partial:
        mask = partial_mask = wholes[indexers] < ndim
    if filter:
        mask = filter(slab_indices[indexers])
    if only_partial and filter:
        mask &= partial_mask
    elif not only_partial and not filter:
        mask = np.broadcast_to(True, slab_indices[indexers].shape)

    idxidx = np.nonzero(mask)
    chunk_indices = tuple(
        [
            np.asarray(mapper.chunk_indices)[idxidx_i]
            for mapper, idxidx_i in zip(mappers, idxidx)
        ]
    )

    flt_slab_indices = slab_indices[chunk_indices]
    flt_slab_offsets = slab_offsets[chunk_indices]
    stacked = np.stack(chunk_indices + (flt_slab_indices, flt_slab_offsets), axis=1)
    if sort:
        stacked = stacked[np.argsort(flt_slab_indices)]
    return stacked


def _build_transfers(
    slab_indices: NDArray[np_hsize_t],
    slab_offsets: NDArray[np_hsize_t],
    mappers: list[IndexChunkMapper],
    chunks: hsize_t[:, :],
    src_slab_idx: ssize_t | Literal["chunks"] | None = None,
    dst_slab_idx: ssize_t | Literal["chunks"] | None = None,
) -> Iterator[TransferPlan]:
    # TODO
    yield from []


@cython.ccall
def _independent_ndindex(idx: tuple) -> tuple:
    """Given an n-dimensional array and a tuple index where each element could be a flat
    array-like, convert it to an index that numpy understands as 'for each axis, take
    these indices' - which is what most users would intuitively expect.

    Example
    >>> a = np.array([[ 0, 10, 20],
    ...               [30, 40, 50],
    ...               [60, 70, 80]])
    >>> a[[1, 2], [0, 1]]
    array([30, 70])
    >>> a[_independent_ndindex(([1, 2], [0, 1]))]
    array([[30, 40],
           [60, 70]])
    """
    arr_indices = []
    for i, idx_i in enumerate(idx):
        if isinstance(idx_i, slice):
            continue
        idx_i = np.asarray(idx_i)
        if idx_i.ndim == 0:
            continue
        arr_indices.append((i, idx_i))

    ndim = len(arr_indices)
    if ndim < 2:
        return idx

    out = list(idx)
    for j, (i, idx_i) in zip(range(ndim - 1, 0, -1), arr_indices[: ndim - 1]):
        out[i] = idx_i[(...,) + (None,) * j]
    return tuple(out)


def _fmt_slice(s: slice) -> str:
    start = "" if s.start is None else s.start
    stop = "" if s.stop is None else s.stop
    step = "" if s.step in (1, None) else f":{s.step}"
    return f"{start}:{stop}{step}"


def _fmt_fancy_index(idx: Any) -> str:
    if isinstance(idx, tuple):
        if idx == ():
            return "()"
    else:
        idx = (idx,)

    return ", ".join(_fmt_slice(i) if isinstance(i, slice) else str(i) for i in idx)
