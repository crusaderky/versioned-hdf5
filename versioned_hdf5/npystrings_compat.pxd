"""Backport NpyStrings from 2.3 to 2.0
The C API was available since 2.0, but the Cython API was added in 2.3.
This is a copy-paste from numpy/__init__.pxd of NumPy 2.3.
Replace this whole module with just `from numpy cimport ...` when we drop
build support for NumPy < 2.3.
"""
# Requires compiling against NumPy 2.3 or later
# from numpy cimport (
#     npy_string_allocator,
#     npy_static_string,
#     npy_packed_static_string,
#     NpyString_pack,
#     NpyString_load,
#     NpyString_acquire_allocator,
#     NpyString_release_allocator,
#     PyArray_StringDTypeObject,
# )

from cpython.ref cimport PyObject
from numpy cimport PyArray_Descr

cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct npy_string_allocator:
        pass

    ctypedef struct npy_packed_static_string:
        pass

    ctypedef struct npy_static_string:
        size_t size
        const char *buf

    ctypedef struct PyArray_StringDTypeObject:
        PyArray_Descr base
        PyObject *na_object
        char coerce
        char has_nan_na
        char has_string_na
        char array_owned
        npy_static_string default_string
        npy_static_string na_name
        npy_string_allocator *allocator

cdef extern from "numpy/arrayobject.h":
    npy_string_allocator *NpyString_acquire_allocator(const PyArray_StringDTypeObject *descr)
    void NpyString_acquire_allocators(size_t n_descriptors, PyArray_Descr *const descrs[], npy_string_allocator *allocators[])
    void NpyString_release_allocator(npy_string_allocator *allocator)
    void NpyString_release_allocators(size_t length, npy_string_allocator *allocators[])
    int NpyString_load(npy_string_allocator *allocator, const npy_packed_static_string *packed_string, npy_static_string *unpacked_string)
    int NpyString_pack_null(npy_string_allocator *allocator, npy_packed_static_string *packed_string)
    int NpyString_pack(npy_string_allocator *allocator, npy_packed_static_string *packed_string, const char *buf, size_t size)
