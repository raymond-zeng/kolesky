from kolesky.ordering import p_reverse_maximin
from kolesky.ordering import sparsity_pattern

import numpy as np
from scipy.spatial import KDTree
import scipy.linalg
import scipy.sparse as sparse
import sklearn.gaussian_process.kernels as kernels


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

def kl_cholesky(points, kernel, rho, lamb, initial = None, p = 1):
    indices, lengths = p_reverse_maximin(points, initial, p)
    ordered_points = points[indices]
    sparsity = sparsity_pattern(ordered_points, lengths, rho)
    groups, agg_sparsity = __supernodes(sparsity, lengths, lamb)
    return __aggregate_chol(ordered_points, kernel, agg_sparsity, groups), indices
