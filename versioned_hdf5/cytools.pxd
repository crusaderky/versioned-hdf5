# This file allows cimport'ing the functions declared below from other Cython modules

# versioned-hdf5 uses the same datatype for indexing as libhdf5. Notably, this differs
# from numpy's ssize_t and allows indexing datasets with >=2**31 points per axis on
# disk, as long as you don't load them in memory all at once.
ctypedef unsigned long long hsize_t  # as per C99, uint64 or wider

cpdef hsize_t[:] empty_view(hsize_t n)
cpdef hsize_t[:] view_from_tuple(tuple[int, ...] t)
cpdef hsize_t stop2count(hsize_t start, hsize_t stop, hsize_t step) noexcept nogil
cpdef hsize_t count2stop(hsize_t start, hsize_t count, hsize_t step) noexcept nogil
cpdef hsize_t ceil_a_over_b(hsize_t a, hsize_t b) noexcept nogil
cpdef hsize_t smallest_step_after(hsize_t x, hsize_t a, hsize_t m) noexcept nogil
cpdef hsize_t[:, :] cartesian_product(list[hsize_t[:]] views)
