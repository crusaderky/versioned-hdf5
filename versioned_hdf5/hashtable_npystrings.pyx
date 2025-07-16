import numpy as np
cimport numpy as np

cdef extern from "openssl/evp.h":
    ctypedef struct EVP_MD:
        pass
    ctypedef struct EVP_MD_CTX:
        pass

    const EVP_MD *EVP_sha256() nogil
    EVP_MD_CTX *EVP_MD_CTX_new() nogil
    int EVP_DigestInit_ex(EVP_MD_CTX *ctx, const EVP_MD *type, void *impl) nogil
    int EVP_DigestUpdate(EVP_MD_CTX *ctx, const void *data, size_t len) nogil
    int EVP_DigestFinal_ex(EVP_MD_CTX *ctx, unsigned char *md, unsigned int *s) nogil
    void EVP_MD_CTX_free(EVP_MD_CTX *ctx) nogil

from versioned_hdf5.npystrings_compat cimport (
    npy_string_allocator,
    npy_static_string,
    npy_packed_static_string,
    NpyString_pack,
    NpyString_load,
    NpyString_acquire_allocator,
    NpyString_release_allocator,
    PyArray_StringDTypeObject,
)

cpdef hash_npystrings_chunk(np.ndarray arr):
    cdef EVP_MD_CTX *ctx = NULL
    cdef np.NpyIter *iter = NULL
    cdef np.NpyIter_IterNextFunc *iternext
    cdef char *data
    cdef char **dataptr
    cdef np.npy_intp *strideptr
    cdef np.npy_intp *innersizeptr
    cdef np.npy_intp stride
    cdef np.npy_intp count
    cdef unsigned int digest_len = 32  # SHA256
    cdef unsigned char[32] digest

    try:
        ctx = EVP_MD_CTX_new()
        if ctx is NULL:
            raise RuntimeError("EVP_MD_CTX_new failed")
        if EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1:
            raise RuntimeError("EVP_DigestInit_ex failed")

        iter = np.NpyIter_New(
            arr,
            np.NPY_ITER_READONLY | np.NPY_ITER_EXTERNAL_LOOP | np.NPY_ITER_REFS_OK,
            np.NPY_KEEPORDER, np.NPY_NO_CASTING,
            None,
        )
        if iter is NULL:
            raise RuntimeError("NpyIter_New failed")

        iternext = np.NpyIter_GetIterNext(iter, NULL)
        if iternext is NULL:
            raise RuntimeError("NpyIter_GetIterNext failed")

        # The location of the data pointer which the iterator may update
        dataptr = np.NpyIter_GetDataPtrArray(iter)
        # The location of the stride which the iterator may update
        strideptr = np.NpyIter_GetInnerStrideArray(iter)
        # The location of the inner loop size which the iterator may update
        innersizeptr = np.NpyIter_GetInnerLoopSizePtr(iter)

        while True:
            # Get the inner loop data/stride/count values
            data = dataptr[0]
            stride = strideptr[0]
            count = innersizeptr[0]

            # This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP
            while count > 0:
                count -= 1

                # TODO: data points to the current point of the array
                # if EVP_DigestUpdate(ctx, data, chunk_len) != 1:
                #     raise RuntimeError("EVP_DigestUpdate failed")

                data += stride
            
            if not iternext[0](iter):
                break

        # Finally return digest as bytes
        if EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1:
            raise RuntimeError("EVP_DigestFinal_ex failed")
        return bytes(digest[:digest_len])

    finally:
        np.NpyIter_Deallocate(iter)
        EVP_MD_CTX_free(ctx)
