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


cpdef hash_slab_fast(
    np.ndarray slab,
    np.ndarray[np.uint8_t, ndim=2] digests,
    int chunk_size,
):
    nchunks = slab.shape[0] // chunk_size
    cdef unsigned int digest_len = 32
    cdef size_t chunk_len = slab.strides[0] * chunk_size
    slab_ptr = slab.data
    digest_ptr = <unsigned char*>digests.data

    assert slab.flags.c_contiguous
    assert digests.shape[0] == nchunks
    assert digests.shape[1] == digest_len
    assert not slab.shape[0] % chunk_size

    with nogil:
        ctx = EVP_MD_CTX_new()
        if ctx is NULL:
            raise MemoryError("EVP_MD_CTX_new failed")

        try:
            for _ in range(nchunks):
                if EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) != 1:
                    raise RuntimeError("EVP_DigestInit_ex failed")
                if EVP_DigestUpdate(ctx, slab_ptr, chunk_len) != 1:
                    raise RuntimeError("EVP_DigestUpdate failed")
                if EVP_DigestFinal_ex(ctx, digest_ptr, &digest_len) != 1:
                    raise RuntimeError("EVP_DigestFinal_ex failed")
                slab_ptr += chunk_len
                digest_ptr += digest_len
        finally:
            EVP_MD_CTX_free(ctx)


import hashlib


cpdef hash_slab_slow(
    np.ndarray slab,
    np.ndarray[np.uint8_t, ndim=2] digests,
    int chunk_size,
):
    nchunks = slab.shape[0] // chunk_size
    cdef unsigned int digest_len = 32

    assert slab.flags.c_contiguous
    assert digests.shape[0] == nchunks
    assert digests.shape[1] == digest_len
    assert not slab.shape[0] % chunk_size

    for i in range(nchunks):
        chunk = slab[(start := i * chunk_size) : start + chunk_size]
        digests[i] = np.frombuffer(hashlib.sha256(chunk).digest(), dtype=np.uint8)
