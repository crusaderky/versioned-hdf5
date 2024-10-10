# This file allows cimport'ing the functions declared below from other Cython modules

from cython cimport ssize_t

from .cytools cimport hsize_t


cdef class IndexChunkMapper:
    cdef readonly hsize_t[:] chunk_indices
    cdef readonly hsize_t chunk_size
    cdef readonly hsize_t dset_size
    cdef readonly hsize_t n_chunks
    cdef readonly hsize_t last_chunk_size
    cdef readonly hsize_t max_read_many_slices_rows_count

    cpdef tuple[object, object, object] chunk_submap(self, hsize_t chunk_idx)
    cpdef tuple[object, object | None] read_many_slices_params(self)

    cpdef object chunks_indexer(self)
    cpdef object whole_chunks_indexer(self)

    # Private methods. Cython complains if we don't export _everything_.
    cdef tuple[hsize_t, hsize_t] _chunk_start_stop(
        self, hsize_t chunk_idx
    ) noexcept nogil
