# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport dtrsv
from scipy.linalg.cython_lapack cimport dpotrf

np.import_array()


cdef inline void cols_cholesky_inplace(double[::1, :] K, int n) noexcept nogil:
    cdef int i, j, ii, jj, info
    cdef double temp
    cdef char uplo = b'L'

    for i in range(n):
        for j in range(n):
            ii = n - 1 - i
            jj = n - 1 - j

            if i < ii or (i == ii and j < jj):
                temp = K[i, j]
                K[i, j] = K[ii, jj]
                K[ii, jj] = temp
    
    dpotrf(&uplo, &n, &K[0, 0], &n, &info)
    
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = 0.0
    
    for i in range(n):
        for j in range(n):
            ii = n - 1 - i
            jj = n - 1 - j
            if i < ii or (i == ii and j < jj):
                temp = K[i, j]
                K[i, j] = K[ii, jj]
                K[ii, jj] = temp
    
    for i in range(n):
        for j in range(i + 1, n):
            temp = K[i, j]
            K[i, j] = K[j, i]
            K[j, i] = temp


cpdef tuple aggregate_chol(
    double[:, ::1] points,
    object kernel,
    int[:] agg_sparsity_indices,
    int[:] agg_sparsity_indptr,
    int[:] groups_data,
    int[:] groups_indptr
):
    cdef int n = points.shape[0]
    cdef int num_groups = groups_indptr.shape[0] - 1
    
    cdef double[::1] data = np.zeros(agg_sparsity_indptr[n], dtype=np.float64)
    cdef int[::1] indices = np.zeros(agg_sparsity_indptr[n], dtype=np.int32)
    
    cdef int[::1] indptr = np.array(agg_sparsity_indptr, dtype=np.int32)
    
    cdef int g, m, i, j, k, len_s, len_i
    cdef int s_start, s_end, i_start, i_end
    cdef int group_start, group_end, leader
    cdef double[::1, :] L_group
    cdef double[::1] col
    cdef int[:] s_indices

    cdef double[:, ::1] points_subset
    cdef double[:, ::1] K_group_c

    # BLAS parameters for dtrsv
    cdef char uplo = b'L'
    cdef char trans = b'N'
    cdef char diag = b'N'
    cdef int incx = 1
    
    for g in range(num_groups):
        group_start = groups_indptr[g]
        group_end = groups_indptr[g + 1]
        leader = groups_data[group_start]
        
        s_start = agg_sparsity_indptr[leader]
        s_end = agg_sparsity_indptr[leader + 1]
        len_s = s_end - s_start
        
        if len_s == 0:
            continue
    
        s_indices = agg_sparsity_indices[s_start:s_end]
        
        points_subset = np.empty((len_s, points.shape[1]), dtype=np.float64)
        for m in range(len_s):
            for j in range(points.shape[1]):
                points_subset[m, j] = points[s_indices[m], j]
        
        K_group_c = kernel(points_subset)
        L_group = np.asfortranarray(K_group_c)
        cols_cholesky_inplace(L_group, len_s)
        for m in range(group_start, group_end):
            i = groups_data[m]
            
            i_start = agg_sparsity_indptr[i]
            i_end = agg_sparsity_indptr[i + 1]
            len_i = i_end - i_start

            k = len_s - len_i
            col = np.zeros(len_s, dtype=np.float64)
            col[k] = 1.0
            dtrsv(&uplo, &trans, &diag, &len_s, &L_group[0, 0], &len_s, &col[0], &incx)
            for j in range(len_i):
                data[i_start + j] = col[k + j]
                indices[i_start + j] = s_indices[k + j]

    return data, indices, indptr
