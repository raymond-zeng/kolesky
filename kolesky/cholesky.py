from kolesky.ordering import p_reverse_maximin
from kolesky.ordering import sparsity_pattern
from kolesky.nugget import ichol
from kolesky.nugget import parallel_ichol

import numpy as np
from scipy.spatial import KDTree
import scipy.linalg
import scipy.sparse as sparse
import sklearn.gaussian_process.kernels as kernels
from copy import deepcopy

from multiprocessing import Pool, RawArray

def __supernodes(sparsity, lengths, lamb):
    groups = []
    candidates = set(range(len(lengths)))
    agg_sparsity = {}
    i = 0
    while len(candidates) > 0:
        while i not in candidates:
            i += 1
        group = sorted(j for j in sparsity[i] if lengths[j] <= lamb * lengths[i] and j in candidates)
        groups.append(group)
        candidates -= set(group)
        s = sorted({k for j in group for k in sparsity[j]})
        agg_sparsity[group[0]] = s
        positions = {k: j for j, k in enumerate(s)}
        for j in group[1:]:
            agg_sparsity[j] = np.empty(len(s) - positions[j], dtype=int)
    return groups, agg_sparsity

def __cols(theta):
    return np.flip(np.linalg.cholesky(np.flip(theta))).T

def __aggregate_chol(points, kernel, sparsity, groups):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for group in groups:
        s = sorted(sparsity[group[0]])
        positions = {i: k for k, i in enumerate(s)}
        L_group = __cols(kernel(points[s]))
        for i in group:
            k = positions[i]
            e_k = np.zeros(len(s))
            e_k[k] = 1
            col = scipy.linalg.solve_triangular(L_group, e_k, lower=True, check_finite=False)
            data[ptr[i] : ptr[i + 1]] = col[k:]
            indices[ptr[i] : ptr[i + 1]] = s[k:]
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

def init_shared(data, indices):
    global shared_data, shared_indices
    shared_data = data
    shared_indices = indices

def group_chol(points, kernel, sparsity, group, ptr):
    s = sorted(sparsity[group[0]])
    positions = {i: k for k, i in enumerate(s)}
    L_group = __cols(kernel(points[s]))
    for i in group:
        k = positions[i]
        e_k = np.zeros(len(s))
        e_k[k] = 1
        col = scipy.linalg.solve_triangular(L_group, e_k, lower=True, check_finite=False)
        shared_data[ptr[i] : ptr[i + 1]] = col[k:]
        shared_indices[ptr[i] : ptr[i + 1]] = s[k:]

def parallel_aggregate_chol(points, kernel, sparsity, groups):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data = RawArray('d', int(ptr[-1]))
    indices = RawArray('i', int(ptr[-1]))
    with Pool(initializer=init_shared, initargs=(data, indices)) as pool:
        pool.starmap(group_chol, [(points, kernel, sparsity, group, ptr) for group in groups])
    data_np = np.frombuffer(data, dtype=np.float64)
    indices_np = np.frombuffer(indices, dtype=np.int32)
    return sparse.csc_matrix((data_np, indices_np, ptr), shape=(n, n))

def kl_cholesky(points, kernel, rho, lamb, initial = None, p = 1):
    indices, lengths = p_reverse_maximin(points, initial, p)
    ordered_points = points[indices]
    sparsity = sparsity_pattern(ordered_points, lengths, rho)
    groups, agg_sparsity = __supernodes(sparsity, lengths, lamb)
    return __aggregate_chol(ordered_points, kernel, agg_sparsity, groups), indices
    # return parallel_aggregate_chol(ordered_points, kernel, agg_sparsity, groups), indices

def innerprod(iter1, u1, iter2, u2, indices, data):
    prod = 0
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

def py_ichol(A):
    indptr = A.indptr #ind
    indices = A.indices #jnd
    data = A.data
    for i in range(len(indptr) - 1):
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

def noise_cholesky(points, kernel, rho, lamb, noise, initial = None, p = 1):
    n = len(points)
    indices, lengths = p_reverse_maximin(points, initial, p)
    ordered_points = points[indices]
    sparsity = sparsity_pattern(ordered_points, lengths, rho)
    groups, agg_sparsity = __supernodes(sparsity, lengths, lamb)
    L = __aggregate_chol(ordered_points, kernel, agg_sparsity, groups)
    A = sparse.triu(L @ L.T, format='csc')
    A += sparse.csc_matrix(np.linalg.inv(noise))
    U = deepcopy(A)
    parallel_ichol(A.indptr, A.indices, A.data, U.data, sweeps=5)
    # ichol(U.indptr, U.indices, U.data)
    # py_ichol(A)
    return L, U, indices
    