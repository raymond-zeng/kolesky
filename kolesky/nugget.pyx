import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

cdef inline double innerprod(int iter1, int u1, int iter2, int u2, int[::1] indices, double[::1] data) noexcept nogil:
    cdef: 
        double prod = 0
    while iter1 <= u1 and iter2 <= u2:
        while iter1 <= u1 and iter2 <= u2 and indices[iter1] == indices[iter2]:
            prod += data[iter1] * data[iter2]
            iter1 += 1
            iter2 += 1
        if indices[iter1] < indices[iter2]:
            iter1 += 1
        else:
            iter2 += 1
    return prod

cpdef void ichol(int[::1] indptr, int[::1] indices, double[::1] data):
    # indptr = A.indptr #ind
    # indices = A.indices #jnd
    # data = A.data
    cdef:
        int i, j, iter_i, iter_j
    for i in range(indptr.shape[0] - 1):
        for j in range(indptr[i], indptr[i + 1]):
            iter_i = indptr[i]
            iter_j = indptr[indices[j]]
            data[j] -= innerprod(iter_i, indptr[i + 1] - 2, iter_j, indptr[indices[j] + 1] - 2, indices, data)
            if data[indptr[indices[j] + 1] - 1] > 0:
                if indices[j] < i:
                    data[j] /= data[indptr[indices[j] + 1] - 1]
                    if np.isnan(data[j]) or np.isinf(data[j]):
                        print("nan or inf")
                else:
                    if j != indptr[i + 1] - 1:
                        print("not diagonal")
                    data[j] = np.sqrt(data[j])
            else:
                data[j] = 0

cpdef void parallel_ichol(int[::1] indptr, int[::1] indices, double[::1] a_data, double[::1] u_data, int sweeps=10):
    cdef:
        int i, j, iter_i, iter_j
    for _ in range(sweeps):
        for i in prange(indptr.shape[0] - 1, nogil=True):
            for j in range(indptr[i], indptr[i + 1]):
                iter_i = indptr[i]
                iter_j = indptr[indices[j]]
                u_data[j] = a_data[j] - innerprod(iter_i, indptr[i + 1] - 2, iter_j, indptr[indices[j] + 1] - 2, indices, u_data)
                if a_data[indptr[indices[j] + 1] - 1] > 0:
                    if indices[j] < i:
                        u_data[j] /= u_data[indptr[indices[j] + 1] - 1]
                    else:
                        u_data[j] = sqrt(u_data[j])
                else:
                    u_data[j] = 0