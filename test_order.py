from kolesky.ordering import p_reverse_maximin, sparsity_pattern, optimized_sparsity_pattern, supernodes
from kolesky.chol import aggregate_chol
from kolesky.kernels import MaternKernel
import scipy.linalg
import scipy.sparse as sparse
import sklearn.gaussian_process.kernels as kernels

def __chol(theta, sigma = 1e-6):
    try:
        return np.linalg.cholesky(theta)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(theta + sigma * np.eye(len(theta)))

def __cols(theta):
    return np.flip(__chol(np.flip(theta))).T

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

def __aggregate_chol(points, kernel, sparsity, groups):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for group in groups:
        # s = sorted(sparsity[group[0]])
        s = sparsity[group[0]]
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

def __aggregate_chol_optimized(points, kernel, agg_sparsity_indices, agg_sparsity_indptr, groups_data, groups_indptr):
    n = len(points)
    
    # The 'starts' array from Cython is exactly the 'ptr' (indptr) 
    # required for a CSC matrix.
    ptr = agg_sparsity_indptr
    
    data = np.zeros(ptr[-1])
    indices = np.zeros(ptr[-1], dtype=int)
    
    num_groups = len(groups_indptr) - 1
    for g in range(num_groups):
        # Extract the group from CSR format
        group = groups_data[groups_indptr[g]:groups_indptr[g + 1]]
        leader = group[0]
        
        # Get the aggregate sparsity list 's' for the group leader
        # This is the slice from agg_sparsity_indices
        s_start, s_end = ptr[leader], ptr[leader + 1]
        s = agg_sparsity_indices[s_start : s_end]
        
        # Compute the Cholesky factor for the entire aggregate group
        L_group = __cols(kernel(points[s]))
        len_s = len(s)
        
        for i in group:
            # The length of node i's sparsity list
            i_start, i_end = ptr[i], ptr[i + 1]
            len_i = i_end - i_start
            
            # k is the relative offset. In __supernodes, we defined:
            # len(agg_sparsity[j]) = len(s) - positions[j]
            # Therefore: k = len(s) - len(agg_sparsity[j])
            k = len_s - len_i
            
            # Solve L_group * col = e_k
            e_k = np.zeros(len_s)
            e_k[k] = 1
            col = scipy.linalg.solve_triangular(L_group, e_k, lower=True, check_finite=False)
            
            # Fill the sparse matrix arrays
            # We slice the result from k to the end
            data[i_start:i_end] = col[k:]
            indices[i_start:i_end] = s[k:]
            
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

import numpy as np
import time

np.random.seed(0)
n = 100
points = np.zeros((n * n, 2))
for i in range(n):
    for j in range(n):
        perturbation = np.random.uniform(-0.2, 0.2, 2)
        points[i * n + j] = np.array([i - n/2, j - n/2]) + perturbation
noise = np.eye(len(points)) * 0.3

order, lengths = p_reverse_maximin(points)
ordered_points = points[order]
sparsity = sparsity_pattern(ordered_points, lengths, 10.0)
kernel = kernels.Matern(nu=0.5, length_scale=1)
test_kernel = MaternKernel(nu=0.5, length_scale=1)
start = time.time()
groups, agg_sparsity = __supernodes(sparsity, lengths, 1.5)
print("Supernodes time:", time.time() - start)
start = time.time()
L = __aggregate_chol(ordered_points, kernel, agg_sparsity, groups)
print("Aggregate time:", time.time() - start)

start = time.time()
test_groups_data, test_groups_indptr, test_agg_sparsity, test_agg_sparsity_indptr = supernodes(sparsity, lengths, 1.5)
print("Supernodes time:", time.time() - start)
start = time.time()
data, indices, indptr = aggregate_chol(ordered_points, test_kernel, test_agg_sparsity, test_agg_sparsity_indptr, test_groups_data, test_groups_indptr)
print("Optimized aggregate time:", time.time() - start)
test_L = sparse.csc_matrix((data, indices, indptr), shape=(len(points), len(points)))

L = L.toarray()
test_L = test_L.toarray()

print(np.linalg.norm(L - test_L))
# assert np.allclose(L, test_L)
