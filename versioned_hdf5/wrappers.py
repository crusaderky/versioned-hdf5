"""
Wrappers of h5py objects that work in memory

Much of this code is modified from code in h5py. See the LICENSE file for the
h5py license.
"""

from __future__ import annotations

import posixpath
import textwrap
import warnings
from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property
from typing import Any
from weakref import WeakValueDictionary

import numpy as np
from h5py import Dataset, Datatype, Empty, Group
from h5py import __version__ as h5py_version
from h5py import h5a, h5d, h5g, h5i, h5p, h5r, h5s, h5t
from h5py._hl import filters
from h5py._hl.base import guess_dtype, phil, with_phil
from h5py._hl.dataset import _LEGACY_GZIP_COMPRESSION_VALS
from h5py._hl.selections import guess_shape
from ndindex import ChunkSize, Slice, Tuple, ndindex

from .backend import DEFAULT_CHUNK_SIZE
from .slicetools import build_slab_indices_and_offsets
from .staged_changes import StagedChangesArray

_groups = WeakValueDictionary({})


class InMemoryGroup(Group):
    def __new__(cls, bind: h5g.GroupID, _committed: bool = False):
        # Make sure each group only corresponds to one InMemoryGroup instance.
        # Otherwise a new instance would lose track of any datasets or
        # subgroups created in the old one.
        if bind in _groups:
            return _groups[bind]
        obj = super().__new__(cls)
        obj._initialized = False
        _groups[bind] = obj
        return obj

    def __init__(self, bind: h5g.GroupID, _committed: bool = False):
        """Create a new InMemoryGroup object by binding to a low-level GroupID.

        Parameters
        ----------
        bind : h5g.GroupID
            Low-level GroupID to bind to
        _committed : bool
            True if the group has already been committed, False otherwise.
        """
        if self._initialized:
            return
        self._data = {}
        self._subgroups = {}
        self._chunks = defaultdict(type(None))
        self._compression = defaultdict(type(None))
        self._compression_opts = defaultdict(type(None))
        self._parent = None
        self._initialized = True
        self._committed = _committed
        super().__init__(bind)

    def close(self):
        """Mark self and any subgroups as committed."""
        self._committed = True
        for name in self:
            obj = self[name]
            if isinstance(obj, InMemoryGroup):
                obj.close()

    # Based on Group.__repr__
    def __repr__(self):
        namestr = f'"{self.name}"' if self.name is not None else "(anonymous)"
        if not self:
            return "<Closed InMemoryGroup>"
        if self._committed:
            return f"<Committed InMemoryGroup {namestr}>"

        text = [f"<InMemoryGroup {namestr} ({len(list(self))} members)>"]
        for item in self.values():
            text.append(textwrap.indent(repr(item), prefix="  "))
        return "\n".join(text)

    def _check_committed(self):
        if self._committed:
            namestr = ('"%s"' % self.name) if self.name is not None else "(anonymous)"
            raise ValueError("InMemoryGroup %s has already been committed" % namestr)

    def __getitem__(self, name):
        dirname, basename = posixpath.split(name)
        if dirname:
            return self.__getitem__(dirname)[basename]

        if name in self._data:
            return self._data[name]
        if name in self._subgroups:
            return self._subgroups[name]

        # If the name doesn't exist in self._data (the "in-memory"
        # part of InMemoryGroup), retrieve it from the actual underlying
        # h5py.Group, (i.e. the file itself).
        res = super().__getitem__(name)
        if isinstance(res, Group):
            self._subgroups[name] = self.__class__(res.id)
            return self._subgroups[name]
        elif isinstance(res, Dataset):
            self._add_to_data(name, res)
            return self._data[name]
        else:
            raise NotImplementedError(f"Cannot handle {type(res)!r}")

    def __setitem__(self, name, obj):
        self._check_committed()
        self._add_to_data(name, obj)

    def _add_to_data(self, name, obj):
        dirname, basename = posixpath.split(name)
        if dirname:
            if dirname not in self:
                self.create_group(dirname)
            self[dirname][basename] = obj
            return

        if isinstance(obj, Dataset):
            wrapped_dataset = self._data[name] = DatasetWrapper(
                InMemoryDataset(obj.id, parent=self)
            )
            self.set_compression(name, wrapped_dataset.dataset.id.raw_data.compression)
            self.set_compression_opts(
                name, wrapped_dataset.dataset.id.raw_data.compression_opts
            )
        elif isinstance(obj, Group):
            self._subgroups[name] = InMemoryGroup(obj.id)
        elif isinstance(obj, InMemoryGroup):
            self._subgroups[name] = obj
        elif isinstance(obj, DatasetLike):
            self._data[name] = obj
            self.set_compression(name, obj.compression)
            self.set_compression_opts(name, obj.compression_opts)
        else:
            self._data[name] = InMemoryArrayDataset(name, np.asarray(obj), parent=self)

    def __delitem__(self, name):
        self._check_committed()
        dirname, basename = posixpath.split(name)
        if dirname:
            if not basename:
                del self[dirname]
            else:
                del self[dirname][basename]
            return

        if name in self._data:
            del self._data[name]
        elif name in self._subgroups:
            for i in self[name]:
                del self[name][i]
            del self._subgroups[name]
            super().__delitem__(name)
        else:
            raise KeyError(f"{name!r} is not in {self}")

    @property
    def parent(self):
        if self._parent is None:
            return super().parent
        return self._parent

    @parent.setter
    def parent(self, p):
        self._parent = p

    def create_group(self, name, track_order=None):
        self._check_committed()
        if name.startswith("/"):
            raise ValueError(
                "Root level groups cannot be created inside of versioned groups"
            )
        group = type(self)(super().create_group(name, track_order=track_order).id)
        g = group
        n = name
        while n:
            dirname, basename = posixpath.split(n)
            if not dirname:
                parent = self
            else:
                parent = type(self)(g.parent.id)
            parent._subgroups[basename] = g
            g.parent = parent
            g = parent
            n = dirname
        return group

    def create_dataset(
        self,
        name,
        shape=None,
        dtype: np.dtype | str | None = None,
        data: np.ndarray | None = None,
        fillvalue=None,
        **kwds,
    ):
        self._check_committed()
        dirname, _ = posixpath.split(name)
        if dirname and dirname not in self:
            self.create_group(dirname)
        if "maxshape" in kwds and any(i != None for i in kwds["maxshape"]):
            warnings.warn(
                "The maxshape parameter is currently ignored for versioned datasets."
            )
        if dtype is not None and not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        data = _make_new_dset(
            data=data, shape=shape, dtype=dtype, fillvalue=fillvalue, **kwds
        )
        if shape is None:
            shape = data.shape
        if fillvalue is not None and isinstance(data, np.ndarray):
            data = InMemoryArrayDataset(name, data, parent=self, fillvalue=fillvalue)
        chunks = kwds.get("chunks")
        if data is None:
            data = InMemorySparseDataset(
                name,
                shape=shape,
                dtype=dtype,
                parent=self,
                fillvalue=fillvalue,
                chunks=chunks,
            )
        if chunks in [True, None]:
            if len(shape) == 1:
                chunks = (DEFAULT_CHUNK_SIZE,)
            else:
                raise NotImplementedError(
                    "chunks must be specified for multi-dimensional datasets"
                )
        if isinstance(chunks, int) and not isinstance(chunks, bool):
            chunks = (chunks,)
        if len(shape) != len(chunks):
            raise ValueError(
                f"Dimensions of chunks ({chunks}) must equal the dimensions of the shape ({shape})"
            )
        if len(shape) == 0:
            raise NotImplementedError("Scalar datasets are not implemented.")

        self.set_chunks(name, chunks)
        self.set_compression(name, kwds.get("compression"))
        self.set_compression_opts(name, kwds.get("compression_opts"))
        self[name] = data
        if dtype is not None:
            self[name]._dtype = dtype
        return self[name]

    def __iter__(self):
        names = list(self._data) + list(self._subgroups)
        for i in super().__iter__():
            if i in names:
                names.remove(i)
            yield i
        for i in names:
            yield i

    def __contains__(self, item):
        item = item + "/"
        root = self.versioned_root.name + "/"
        if item.startswith(root):
            item = item[len(root) :]
            if not item.rstrip("/"):
                return self == self.versioned_root
        item = item.rstrip("/")
        dirname, data_name = posixpath.split(item)
        if dirname not in ["", "/"]:
            return dirname in self and data_name in self[dirname]
        for i in self:
            if i == item:
                return True
        return False

    def datasets(self):
        res = self._data.copy()

        def _get(name, item):
            if name in res:
                return
            if isinstance(item, (Dataset, DatasetLike, np.ndarray)):
                res[name] = item

        self.visititems(_get)

        return res

    @property
    def versioned_root(self):
        p = self
        while p._parent is not None:
            p = p._parent
        return p

    @property
    def chunks(self):
        return self._chunks

    # TODO: Can we generalize this, set_compression, and set_compression_opts
    # into a single method? Descriptors?
    def set_chunks(self, item, value):
        full_name = item
        p = self
        while p._parent:
            p._chunks[full_name] = value
            _, basename = posixpath.split(p.name)
            full_name = basename + "/" + full_name
            p = p._parent
        self.versioned_root._chunks[full_name] = value

        dirname, basename = posixpath.split(item)
        while dirname:
            self[dirname]._chunks[basename] = value
            dirname, b = posixpath.split(dirname)
            basename = posixpath.join(b, basename)

    @property
    def compression(self):
        return self._compression

    def set_compression(self, item, value):
        full_name = item
        p = self
        while p._parent:
            p._compression[full_name] = value
            _, basename = posixpath.split(p.name)
            full_name = basename + "/" + full_name
            p = p._parent
        self.versioned_root._compression[full_name] = value

        dirname, basename = posixpath.split(item)
        while dirname:
            self[dirname]._compression[basename] = value
            dirname, b = posixpath.split(dirname)
            basename = posixpath.join(b, basename)

    @property
    def compression_opts(self):
        return self._compression_opts

    def set_compression_opts(self, item, value):
        full_name = item
        p = self
        while p._parent:
            p._compression_opts[full_name] = value
            _, basename = posixpath.split(p.name)
            full_name = basename + "/" + full_name
            p = p._parent
        self.versioned_root._compression_opts[full_name] = value

        dirname, basename = posixpath.split(item)
        while dirname:
            self[dirname]._compression_opts[basename] = value
            dirname, b = posixpath.split(dirname)
            basename = posixpath.join(b, basename)

    def visititems(self, func):
        self._visit("", func)

    def _visit(self, prefix, func):
        for name in self:
            func(posixpath.join(prefix, name), self[name])
            if isinstance(self[name], InMemoryGroup):
                self[name]._visit(posixpath.join(prefix, name), func)

    # TODO: override other relevant methods here


def _make_new_dset(
    shape: int | tuple[int, ...] | None = None,
    dtype: np.dtype | None = None,
    data: np.ndarray | None = None,
    chunks: tuple[int, ...] | None = None,
    compression: int | str | bool | None = None,
    shuffle: bool | None = None,
    fletcher32: bool | None = None,
    maxshape: tuple[int, ...] | None = None,
    compression_opts: int | tuple[int, ...] | None = None,
    fillvalue: int | str | float | None = None,
    scaleoffset: bool | int | None = None,
    track_times: bool | None = None,
    external: Iterable[tuple[str, int, int]] | None = None,
    track_order: bool | None = None,
    dcpl: h5p.PropDCID | None = None,
) -> np.ndarray:
    """Create a new low-level dataset identifier.

    Based on h5py._hl.dataset.make_new_dset(), except it doesn't actually create
    the dataset, it just canonicalizes the arguments. Additionally, this function
    allows datasets which are smaller than the data in any dimension to be
    instantiated, whereas the upstream h5py version does not.

    See the LICENSE file for the h5py license.

    Parameters
    ----------
    shuffle : bool | None
        Whether to call dcpl.set_shuffle() on the underlying PropDCID
    fletcher32 : bool | None
        Whether to call dcpl.set_fletcher32(). Note that scale/offset following
        fletcher32 in the filter chain will (almost?) always triggers a read
        error, as most scale/offset settings are lossy. Since fletcher32 must
        come first (see comment in h5py._hl.filters.fill_dcpl) combination of
        fletcher32 and scale/offset is prohibited.
    maxshape : tuple[int, ...] | None
        Max shape of the dataste
    compression_opts : int | tuple[int, ...] | None
        Compression options passed to h5py._hl.filters.fill_dcpl
    fillvalue : int | str | float | None
        Value used to fill the parts of chunks that extend beyond the dataset
    scaleoffset : bool | int | None
        This must be an integer when it is not None or False, except for integral
        data, for which scaleoffset == True is permissible (will use
        SO_INT_MINBITS_DEFAULT)
    track_times : bool |  None
        Argument to pass to dcpl.set_obj_track_times(track_times)
    shape : int | tuple[int, ...] | None
        Shape of the dataset to create
    dtype : np.dtype | None
        Dtype of the dataset to create
    data : np.ndarray | None
        Dataset to store
    chunks : tuple[int, ...] | None
        Chunk size in each dimension
    compression : int | str | bool | None
        Compression to use for the dataset
    external : Iterable[tuple[str, int, int]] | None
        List of tuples to be passed to h5p.set_external
    track_order : bool | None
        If True, set tracking and indexing of creation order for object attributes;
        otherwise, dcpl.set_attr_creation_order(0) is called
    dcpl : h5p.PropDCID | None
        Dataset Creation Property List to use; if unspecified, an new dcpl is created

    Returns
    -------
    np.ndarray | None
        Data used to create the dataset
    """
    # Convert data to a C-contiguous ndarray
    if data is not None and not isinstance(data, Empty):
        # normalize strings -> np.dtype objects
        if dtype is not None:
            _dtype = np.dtype(dtype)
        else:
            _dtype = None

        # Do not convert in memory input numpy data with the wrong dtype
        # However, if we are going to a f2 datatype, pre-convert hee
        # to workaround a possible h5py bug in the conversion.
        if _dtype is not None and _dtype.kind == "f" and _dtype.itemsize == 2:
            preconvert_dtype = _dtype
        else:
            # Special cases for vlen strings
            preconvert_dtype = guess_dtype(data)
        if preconvert_dtype is None and not isinstance(data, np.ndarray):
            preconvert_dtype = _dtype

        data = np.asarray(data, order="C", dtype=preconvert_dtype)

    # Validate shape
    if shape is None:
        if data is None:
            if dtype is None:
                raise TypeError("One of data, shape or dtype must be specified.")
            data = Empty(dtype)
        shape = data.shape
    else:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if data is not None and (
            np.prod(shape, dtype=np.ulonglong)
            != np.prod(data.shape, dtype=np.ulonglong)
        ):
            raise ValueError(
                f"Shape tuple {shape} is incompatible with data shape {data.shape}"
            )

    if isinstance(maxshape, int):
        maxshape = (maxshape,)

    # Validate chunk shape
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        chunks = (chunks,)

    # The original make_new_dset errors here if the shape is less than the
    # chunk size, but we avoid doing that as we cannot change the chunk size
    # for a dataset for any version once it is created. See #34.

    if isinstance(dtype, Datatype):
        # Named types are used as-is
        tid = dtype.id
        dtype = tid.dtype  # Following code needs this
    else:
        # Validate dtype
        if dtype is None and data is None:
            dtype = np.dtype("=f4")
        elif dtype is None and data is not None:
            dtype = data.dtype
        else:
            dtype = np.dtype(dtype)
        tid = h5t.py_create(dtype, logical=1)

    # Legacy
    if (
        any((compression, shuffle, fletcher32, maxshape, scaleoffset))
        and chunks is False
    ):
        raise ValueError(
            "Chunked format required for given storage options:\n"
            + textwrap.indent(
                "\n".join(
                    [
                        f"compression: {compression}",
                        f"shuffle: {shuffle}",
                        f"fletcher32: {fletcher32}",
                        f"maxshape: {maxshape}",
                        f"scaleoffset: {scaleoffset}",
                    ]
                ),
                "  ",
            )
        )

    # Legacy
    if compression is True:
        if compression_opts is None:
            compression_opts = 4
        compression = "gzip"

    # Legacy
    if compression in _LEGACY_GZIP_COMPRESSION_VALS:
        if compression_opts is not None:
            raise TypeError(
                "Conflict in compression options; "
                f"if compression is one of {_LEGACY_GZIP_COMPRESSION_VALS},"
                f"compression_opts must be None.\n"
                + textwrap.indent(
                    "\n".join(
                        f"compression: {compression}",
                        f"compression_opts: {compression_opts}",
                    )
                )
            )
        compression_opts = compression
        compression = "gzip"
    dcpl = filters.fill_dcpl(
        dcpl or h5p.create(h5p.DATASET_CREATE),
        shape,
        dtype,
        chunks,
        compression,
        compression_opts,
        shuffle,
        fletcher32,
        maxshape,
        scaleoffset,
        external,
    )

    if fillvalue is not None:
        fillvalue = np.array(fillvalue)
        dcpl.set_fill_value(fillvalue)

    if track_times in (True, False):
        dcpl.set_obj_track_times(track_times)
    elif track_times is not None:
        raise TypeError("track_times must be either True or False")
    if track_order == True:
        dcpl.set_attr_creation_order(h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
    elif track_order == False:
        dcpl.set_attr_creation_order(0)
    elif track_order is not None:
        raise TypeError("track_order must be either True or False")

    if maxshape is not None:
        maxshape = tuple(m if m is not None else h5s.UNLIMITED for m in maxshape)

    if isinstance(data, Empty):
        raise NotImplementedError("Empty datasets are not supported.")
    return data


class InMemoryDataset(Dataset):
    """
    Class that looks like a h5py.Dataset but is backed by a versioned dataset

    The versioned dataset can be modified, which performs modifications
    in-memory only.
    """

    def __init__(self, bind, parent, **kwargs):
        # Hold a reference to the original bind so h5py doesn't invalidate the id
        # XXX: We need to handle deallocation here properly when our object
        # gets deleted or closed.
        self.orig_bind = bind
        super().__init__(InMemoryDatasetID(bind.id), **kwargs)
        self._init_kwargs = kwargs
        self._parent = parent
        self._attrs = dict(super().attrs)

    @cached_property
    def staged_changes(self) -> StagedChangesArray:
        dcpl = self.id.get_create_plist()
        slab_indices, slab_offsets = build_slab_indices_and_offsets(
            dcpl, self.id.shape, self.id.chunks
        )
        return StagedChangesArray(
            shape=self.id.shape,
            chunk_size=self.id.chunks,
            base_slabs=[self.id.raw_data],
            slab_indices=slab_indices,
            slab_offsets=slab_offsets,
            fill_value=self.fillvalue,
        )

    def __repr__(self) -> str:
        name = posixpath.basename(posixpath.normpath(self.name))
        namestr = '"%s"' % (name if name != "" else "/")
        return '<%s %s: shape %s, type "%s">' % (
            self.__class__.__name__,
            namestr,
            self.shape,
            self.dtype.str,
        )

    @property
    def data_dict(self) -> dict[Tuple, Slice | np.ndarray]:
        return _staged_changes_to_data_dict(self.staged_changes)

    @property
    def compression(self):
        name = self.name
        if self.parent.name in name:
            name = name[len(self.parent.name) + 1 :]
        return self.parent.compression[name]

    @compression.setter
    def compression(self, value):
        self.parent.set_compression(self.item, value)

    @property
    def compression_opts(self):
        name = self.name
        if self.parent.name in name:
            name = name[len(self.parent.name) + 1 :]
        return self.parent.compression_opts[self.name]

    @compression_opts.setter
    def compression_opts(self, value):
        self.parent.set_compression_opts(self.name, value)

    def as_dtype(self, name, dtype, parent, casting="unsafe"):
        """
        Return a copy of `self` as a new dataset with the given `name` and `dtype`
        in the group `parent`.

        `casting` should be as in the numpy astype() method.

        """
        fillvalue = super().fillvalue
        init_kwargs = self._init_kwargs.copy()
        if fillvalue is not None:
            fillvalue = fillvalue.astype(dtype, casting=casting)
            init_kwargs.setdefault("fillvalue", fillvalue)

        new_dataset = InMemoryDataset(
            bind=self.orig_bind,
            parent=parent,
            **init_kwargs,
        )

        new_dataset.staged_changes = self.staged_changes.astype(dtype, casting=casting)
        parent[name] = new_dataset
        return new_dataset

    @property
    def fillvalue(self) -> Any:
        if super().fillvalue is not None:
            return super().fillvalue
        if self.dtype.metadata:
            # Custom h5py string dtype. Make sure to use a fillvalue of ''
            if "vlen" in self.dtype.metadata:
                # h5py 3 reads str variable length datasets as bytes. See
                # https://docs.h5py.org/en/stable/whatsnew/3.0.html#breaking-changes-deprecations.
                if h5py_version.startswith("3") and self.dtype.metadata["vlen"] == str:
                    return bytes()
                return self.dtype.metadata["vlen"]()
            elif "h5py_encoding" in self.dtype.metadata:
                return self.dtype.type()
        return np.zeros((), dtype=self.dtype)[()]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.id.shape

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def chunks(self):
        return ChunkSize(self.id.chunks)

    @property
    def attrs(self):
        return self._attrs

    @property
    def parent(self) -> InMemoryGroup:
        return self._parent

    def __array__(self, dtype=None):
        return self.__getitem__(())

    def resize(self, size, axis=None):
        """Resize the dataset, or the specified axis.

        The rank of the dataset cannot be changed.

        "Size" should be a shape tuple, or if an axis is specified, an integer.

        BEWARE: This functions differently than the NumPy resize() method!
        The data is not "reshuffled" to fit in the new shape; each axis is
        grown or shrunk independently.  The coordinates of existing data are
        fixed.
        """
        self.parent._check_committed()
        # This boilerplate code is based on h5py.Dataset.resize
        if axis is not None:
            if not (axis >= 0 and axis < self.id.rank):
                raise ValueError("Invalid axis (0 to %s allowed)" % (self.id.rank - 1))
            try:
                newlen = int(size)
            except TypeError:
                raise TypeError("Argument must be a single int if axis is specified")
            size = list(self.shape)
            size[axis] = newlen

        size = tuple(size)
        # === END CODE FROM h5py.Dataset.resize ===

        self.staged_changes.resize(size)
        self.id.shape = size

    @with_phil
    def __getitem__(
        self,
        args: (
            str
            | slice
            | tuple
            | list[int]
            | list[bool]
            | np.ndarray
            | h5r.RegionReference
        ),
    ) -> np.ndarray:
        """Read a slice from the HDF5 dataset given by the index.

        Takes slices and recarray-style field names (more than one is
        allowed!) in any order.  Obeys basic NumPy rules, including
        broadcasting.

        Parameters
        ----------
        args : any numpy one-dimensional or n-dimensional index | h5r.RegionReference
            Index to read from the Dataset.

            **Note:** more than one list/ndarray index will behave differently as numpy,
            as it will be interpreted to pick the given indices independently on each
            axis. Non-flat list/ndarray indices are not supported.

        Returns
        -------
        np.ndarray
            Array containing data from this dataset from the requested index
        """
        # This boilerplate code is based on h5py.Dataset.__getitem__
        args = args if isinstance(args, tuple) else (args,)

        # Sort field names from the rest of the args.
        names = tuple(x for x in args if isinstance(x, str))

        if names:
            # Read a subset of the fields in this structured dtype
            if len(names) == 1:
                names = names[0]  # Read with simpler dtype of this field
            args = tuple(x for x in args if not isinstance(x, str))
            return self.fields(names)[args]

        # === Special-case region references ====

        if len(args) == 1 and isinstance(args[0], h5r.RegionReference):
            mtype = h5t.py_create(self.dtype)
            obj = h5r.dereference(args[0], self.id)
            if obj != self.id:
                raise ValueError("Region reference must point to this dataset")

            sid = h5r.get_region(args[0], self.id)
            mshape = guess_shape(sid)
            if mshape is None:
                # 0D with no data (NULL or deselected SCALAR)
                return Empty(self.dtype)
            out = np.empty(mshape, self.dtype)
            if out.size == 0:
                return out

            sid_out = h5s.create_simple(mshape)
            sid_out.select_all()
            self.id.read(sid_out, sid, out, mtype)
            return out

        # === END CODE FROM h5py.Dataset.__getitem__ ===

        return self.staged_changes[args]

    @with_phil
    def __setitem__(self, args, val):
        """Write to the HDF5 dataset from a NumPy array.

        NumPy's broadcasting rules are honored, for "simple" indexing
        (slices and integers).  For advanced indexing, the shapes must
        match.
        """
        self.parent._check_committed()
        # This boilerplate code is based on h5py.Dataset.__setitem__
        args = args if isinstance(args, tuple) else (args,)

        # Sort field indices from the slicing
        names = tuple(x for x in args if isinstance(x, str))
        args = tuple(x for x in args if not isinstance(x, str))

        # Generally we try to avoid converting the arrays on the Python
        # side.  However, for compound literals this is unavoidable.
        vlen = h5t.check_vlen_dtype(self.dtype)
        if vlen is not None and vlen not in (bytes, str):
            try:
                val = np.asarray(val, dtype=vlen)
            except ValueError:
                try:
                    val = np.array(
                        [np.array(x, dtype=vlen) for x in val], dtype=self.dtype
                    )
                except ValueError:
                    pass
            if vlen == val.dtype:
                if val.ndim > 1:
                    tmp = np.empty(shape=val.shape[:-1], dtype=object)
                    tmp.ravel()[:] = [
                        i
                        for i in val.reshape(
                            (
                                np.prod(val.shape[:-1], dtype=np.ulonglong),
                                val.shape[-1],
                            )
                        )
                    ]
                else:
                    tmp = np.array([None], dtype=object)
                    tmp[0] = val
                val = tmp
        elif self.dtype.kind == "O" or (
            self.dtype.kind == "V"
            and (not isinstance(val, np.ndarray) or val.dtype.kind != "V")
            and (self.dtype.subdtype == None)
        ):
            if len(names) == 1 and self.dtype.fields is not None:
                # Single field selected for write, from a non-array source
                if not names[0] in self.dtype.fields:
                    raise ValueError("No such field for indexing: %s" % names[0])
                dtype = self.dtype.fields[names[0]][0]
                cast_compound = True
            else:
                dtype = self.dtype
                cast_compound = False

            val = np.asarray(val, dtype=dtype.base, order="C")
            if cast_compound:
                val = val.view(np.dtype([(names[0], dtype)]))
                val = val.reshape(val.shape[: len(val.shape) - len(dtype.shape)])
        else:
            val = np.asarray(val, order="C")

        # Check for array dtype compatibility and convert
        if self.dtype.subdtype is not None:
            shp = self.dtype.subdtype[1]
            valshp = val.shape[-len(shp) :]
            if valshp != shp:  # Last dimension has to match
                raise TypeError(
                    "When writing to array types, last N dimensions have to match (got %s, but should be %s)"
                    % (
                        valshp,
                        shp,
                    )
                )
            mtype = h5t.py_create(np.dtype((val.dtype, shp)))
            # mshape = val.shape[0:len(val.shape)-len(shp)]

        # Make a compound memory type if field-name slicing is required
        elif len(names) != 0:
            # mshape = val.shape

            # Catch common errors
            if self.dtype.fields is None:
                raise TypeError("Illegal slicing argument (not a compound dataset)")
            mismatch = [x for x in names if x not in self.dtype.fields]
            if len(mismatch) != 0:
                mismatch = ", ".join('"%s"' % x for x in mismatch)
                raise ValueError(
                    "Illegal slicing argument (fields %s not in dataset type)"
                    % mismatch
                )

            # Write non-compound source into a single dataset field
            if len(names) == 1 and val.dtype.fields is None:
                subtype = h5t.py_create(val.dtype)
                mtype = h5t.create(h5t.COMPOUND, subtype.get_size())
                mtype.insert(self._e(names[0]), 0, subtype)

            # Make a new source type keeping only the requested fields
            else:
                fieldnames = [
                    x for x in val.dtype.names if x in names
                ]  # Keep source order
                mtype = h5t.create(h5t.COMPOUND, val.dtype.itemsize)
                for fieldname in fieldnames:
                    subtype = h5t.py_create(val.dtype.fields[fieldname][0])
                    offset = val.dtype.fields[fieldname][1]
                    mtype.insert(self._e(fieldname), offset, subtype)

        # Use mtype derived from array (let DatasetID.write figure it out)
        else:
            mtype = None

        # === END CODE FROM h5py.Dataset.__setitem__ ===
        self.staged_changes[args] = val


class DatasetLike:
    """
    Superclass for classes that look like h5py.Dataset

    Subclasses should have the following properties defined (properties
    starting with an underscore will be computed if they are None)

    name
    shape
    dtype
    _fillvalue
    parent (the parent group)
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    _fillvalue: object | None
    parent: InMemoryGroup

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def fillvalue(self) -> Any:
        if self._fillvalue is not None:
            return np.array([self._fillvalue], dtype=self.dtype)[0]
        if self.dtype.metadata:
            # Custom h5py string dtype. Make sure to use a fillvalue of ''
            if "vlen" in self.dtype.metadata:
                # h5py 3 reads str variable length datasets as bytes. See
                # https://docs.h5py.org/en/stable/whatsnew/3.0.html#breaking-changes-deprecations.
                if h5py_version.startswith("3") and self.dtype.metadata["vlen"] == str:
                    return bytes()
                return self.dtype.metadata["vlen"]()
            elif "h5py_encoding" in self.dtype.metadata:
                return b""
        return np.zeros((), dtype=self.dtype)[()]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __bool__(self) -> bool:
        return bool(self.size)

    def __len__(self) -> int:
        return self.len()

    def len(self) -> int:
        """Length of the first axis."""
        if len(self.shape) == 0:
            raise TypeError("Attempt to take len() of scalar dataset")
        return self.shape[0]

    def __repr__(self) -> str:
        name = posixpath.basename(posixpath.normpath(self.name))
        namestr = '"%s"' % (name if name != "" else "/")
        return '<%s %s: shape %s, type "%s">' % (
            self.__class__.__name__,
            namestr,
            self.shape,
            self.dtype.str,
        )

    def __iter__(self) -> Iterable[np.ndarray | np.generic]:
        """Iterate over the first axis. TypeError if scalar.

        BEWARE: Modifications to the yielded data are *NOT* written to file.
        """
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Can't iterate over a scalar dataset")
        for i in range(shape[0]):
            yield self[i]

    @property
    def compression(self):
        name = self.name
        if self.parent.name in name:
            name = name[len(self.parent.name) + 1 :]
        return self.parent.compression[name]

    @compression.setter
    def compression(self, value):
        name = self.name
        if self.parent.name in name:
            name = name[len(self.parent.name) + 1 :]
        self.parent.set_compression(name, value)

    @property
    def compression_opts(self):
        name = self.name
        if self.parent.name in name:
            name = name[len(self.parent.name) + 1 :]
        return self.parent.compression_opts[name]

    @compression_opts.setter
    def compression_opts(self, value):
        name = self.name
        if self.parent.name in name:
            name = name[len(self.parent.name) + 1 :]
        self.parent.set_compression_opts(name, value)


class InMemoryArrayDataset(DatasetLike):
    """
    Class that looks like a h5py.Dataset but is backed by an array
    """

    def __init__(self, name, array, parent, fillvalue=None, chunks=None):
        self.name = name
        self._array = array  # May have a different dtype!
        self._dtype = None  # Maybe overwritten by create_dataset
        self.attrs = {}
        self.parent = parent
        self._fillvalue = fillvalue
        if chunks is None:
            chunks = parent.chunks[name]
        self._chunks = chunks

    def as_dtype(self, name, dtype, parent, casting="unsafe"):
        """
        Return a copy of `self` as a new dataset with the given `name` and `dtype`
        in the group `parent`.

        `casting` should be as in the numpy astype() method.

        """
        return self.__class__(
            name, self._array.astype(dtype, casting=casting), parent=parent
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype or self._array.dtype

    def __getitem__(self, item):
        return np.asarray(self._array[item], dtype=self._dtype)[()]

    def __setitem__(self, item, value):
        self.parent._check_committed()
        self._array[item] = value

    def __array__(self, dtype=None):
        return np.asarray(self._array, dtype=dtype or self._dtype)

    @property
    def chunks(self):
        return self._chunks

    def resize(self, size, axis=None):
        self.parent._check_committed()
        if axis is not None:
            if not (axis >= 0 and axis < self.ndim):
                raise ValueError("Invalid axis (0 to %s allowed)" % (self.ndim - 1))
            try:
                newlen = int(size)
            except TypeError:
                raise TypeError("Argument must be a single int if axis is specified")
            size = list(self.shape)
            size[axis] = newlen

        old_shape = self.shape
        size = tuple(size)
        if all(new <= old for new, old in zip(size, old_shape)):
            # Don't create a new array if the old one can just be sliced in
            # memory.
            idx = tuple(slice(0, i) for i in size)
            self._array = self._array[idx]
        else:
            old_shape_idx = Tuple(*[Slice(0, i) for i in old_shape])
            new_shape_idx = Tuple(*[Slice(0, i) for i in size])
            new_array = np.full(size, self.fillvalue, dtype=self.dtype)
            new_array[old_shape_idx.as_subindex(new_shape_idx).raw] = self._array[
                new_shape_idx.as_subindex(old_shape_idx).raw
            ]
            self._array = new_array


class InMemorySparseDataset(DatasetLike):
    """
    Class that looks like a Dataset that has no data (only the fillvalue)
    """

    def __init__(self, name, *, shape, dtype, parent, chunks=None, fillvalue=None):
        if shape is None:
            raise TypeError("shape must be specified for sparse datasets")
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.attrs = {}
        self._fillvalue = fillvalue
        if chunks in [True, None]:
            if len(shape) == 1:
                chunks = (DEFAULT_CHUNK_SIZE,)
            else:
                raise NotImplementedError(
                    "chunks must be specified for multi-dimensional datasets"
                )
        self.chunks = ChunkSize(chunks)
        self.parent = parent

        self.staged_changes = StagedChangesArray.full(
            shape=shape,
            chunk_size=tuple(chunks),
            dtype=dtype,
            fill_value=self.fillvalue,
        )

    @property
    def data_dict(self) -> dict[Tuple, Slice | np.ndarray]:
        return _staged_changes_to_data_dict(self.staged_changes)

    def as_dtype(self, name, dtype, parent, casting="unsafe"):
        """
        Return a copy of `self` as a new dataset with the given `name` and `dtype`
        in the group `parent`.

        `casting` should be as in the numpy astype() method.

        """
        if self.fillvalue is not None:
            new_fillvalue = self.fillvalue.astype(dtype, casting=casting)
        else:
            new_fillvalue = None

        out = type(self)(
            name=name,
            shape=(),
            dtype=dtype,
            parent=parent,
            chunks=self.chunks,
            fillvalue=new_fillvalue,
        )
        out.staged_changes = self.staged_changes.astype(dtype, casting=casting)
        out.shape = self.shape
        return out

    @classmethod
    def from_dataset(cls, dataset, parent=None):
        # np.testing.assert_equal(dataset[()], dataset.fillvalue)
        return cls(
            dataset.name,
            shape=dataset.shape,
            dtype=dataset.dtype,
            parent=parent or dataset.parent,
            chunks=dataset.chunks,
            fillvalue=dataset.fillvalue,
        )

    def resize(self, size, axis=None):
        if axis is not None:
            if not (axis >= 0 and axis < self.ndim):
                raise ValueError("Invalid axis (0 to %s allowed)" % (self.ndim - 1))
            try:
                newlen = int(size)
            except TypeError:
                raise TypeError("Argument must be a single int if axis is specified")
            size = list(self.shape)
            size[axis] = newlen

        size = tuple(size)
        self.staged_changes.resize(size)
        self.shape = size

    def __getitem__(self, index):
        return self.staged_changes[index]

    def __setitem__(self, index, value):
        self.parent._check_committed()
        self.staged_changes[index] = value


def _staged_changes_to_data_dict(
    staged_changes: StagedChangesArray,
) -> dict[Tuple, Slice | np.ndarray]:
    """Transitional hack that converts a StagedChangsArray to a legacy data_dict.

    This was introduced when replacing the legacy system, which was wholly designed
    around the data_dict, with StagedChangesArray and it allowed not to modify from the
    get go all the code that is triggered upon commit.

    We intend to clean this up eventually.
    """
    # InMemoryDataset has exactly one raw_data buffer underlying
    # InMemorySparseDataset has none
    assert staged_changes.n_base_slabs < 2
    return {
        Tuple(*k): Slice(v[0]) if isinstance(v, tuple) else v
        for k, _, v in staged_changes.changes()
    }


class DatasetWrapper(DatasetLike):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __setitem__(self, index, value):
        if isinstance(self.dataset, InMemoryDataset) and ndindex(index).expand(
            self.shape
        ) == Tuple().expand(self.shape):
            new_dataset = InMemoryArrayDataset(
                self.name,
                np.broadcast_to(value, self.shape).astype(self.dtype),
                self.parent,
                fillvalue=self.fillvalue,
                chunks=self.chunks,
            )
            new_dataset.attrs = self.dataset.attrs
            self.dataset = new_dataset
            return
        self.dataset.__setitem__(index, value)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)


class InMemoryDatasetID(h5d.DatasetID):
    def __init__(self, _id):
        # super __init__ is handled by DatasetID.__cinit__ automatically
        with phil:
            sid = self.get_space()
            self._shape = sid.get_simple_extent_dims()

        attr = h5a.open(self, b"raw_data")
        htype = h5t.py_create(attr.dtype)
        _arr = np.ndarray(attr.shape, dtype=attr.dtype, order="C")
        attr.read(_arr, mtype=htype)
        raw_data_name = _arr[()]
        if isinstance(raw_data_name, bytes):
            raw_data_name = raw_data_name.decode("utf-8")

        fid = h5i.get_file_id(self)
        g = Group(fid)
        self.raw_data = g[raw_data_name]
        self.chunks = tuple(self.raw_data.attrs["chunks"])

        fillvalue_a = np.empty((1,), dtype=self.dtype)
        dcpl = self.get_create_plist()
        dcpl.get_fill_value(fillvalue_a)
        self.fillvalue = fillvalue_a[0]

    def set_extent(self, shape):
        raise NotImplementedError("Resizing an InMemoryDataset other than via resize()")

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, size):
        self._shape = size

    def write(self, mspace, fspace, arr_obj, mtype=None, dxpl=None):
        raise NotImplementedError(
            "Writing to an InMemoryDataset other than via __setitem__"
        )
