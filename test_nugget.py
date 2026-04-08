import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from kolesky import ordering

def logdet_chol(A):
    return 2 * np.sum(np.log(A.diagonal()))

def kl_div(A, B):
    n = A.shape[0]
    return 0.5 * (np.trace(np.linalg.solve(B, A)) - n + np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1])

def sparse_kl_div(A, L):
    n = A.shape[0]
    return 0.5 * (-logdet_chol(L) - np.linalg.slogdet(A)[1])

pressure = np.load("./100x4x100channelpressure4000.npy")
# with open('13x13x13iso1024.pickle', 'rb') as f:
    # pressure = pickle.load(f)[:,:,0]
pressure = np.ascontiguousarray(pressure)
print(pressure.shape, flush=True)

nx = 100
ny = 4
nz = 100
# x_points = np.linspace(0.0, 8 * np.pi / 2048 * 99, nx, dtype = np.float64)
# x_points = np.linspace(0.0, 18 * np.pi / 512, nx, dtype = np.float64)
x_points = np.linspace(0.0, 98 * np.pi / 512, nx, dtype = np.float64)
# y_points = np.pi / 2
# x_points = np.linspace(0.0, 8  * np.pi / 2048 * 99, nx, dtype = np.float64)
y_points = [0, 2 / 512, 4 / 512, 6 / 512]
# z_points = np.linspace(0.0, 3 * np.pi / 1536 * 99, nz, dtype = np.float64)
# y_points = np.linspace(0.0, 18 * np.pi / 512, ny, dtype = np.float64)
# z_points = np.linspace(0.0, 18 * np.pi / 1024, nz, dtype = np.float64)
# z_points = np.linspace(0.0, 18 * np.pi / 512, nz, dtype = np.float64)
z_points = np.linspace(0.0, 98 * np.pi / 512, nz, dtype = np.float64)
#x_points = np.linspace(0.0, 24 * np.pi / 512, nx, dtype = np.float64)
points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, z_points, indexing = 'ij')], dtype = np.float64).T
points = np.ascontiguousarray(points)


from scipy.spatial import KDTree
import scipy.linalg
import scipy.sparse as sparse

def col(theta):
    m = np.linalg.inv(theta)
    return m[:, 0] / np.sqrt(m[0, 0])

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

def chol(points, covariance, sparsity):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for i in range(n):
        s = sorted(sparsity[i])
        theta = np.zeros((len(s), len(s)))
        for j in range(len(s)):
            for k in range(len(s)):
                theta[j, k] = covariance[s[j], s[k]]
        c = col(theta)
        data[ptr[i] : ptr[i + 1]] = c
        indices[ptr[i] : ptr[i + 1]] = s
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

def __chol(theta, sigma = 1e-6):
    try:
        return np.linalg.cholesky(theta)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(theta + sigma * np.eye(len(theta)))

def __cols(theta):
    return np.flip(__chol(np.flip(theta))).T

def __aggregate_chol(points, velocity, sparsity, groups, sigma=1e-6):
    n = len(points)
    ptr = np.cumsum([0] + [len(sparsity[i]) for i in range(n)])
    data, indices = np.zeros(ptr[-1]), np.zeros(ptr[-1])
    for group in groups:
        s = sorted(sparsity[group[0]])
        positions = {i: k for k, i in enumerate(s)}
        # data_group = np.cov(velocity[:, s], rowvar=False, bias=True)
        # if len(s) == 1:
            # data_group = np.array([data_group], ndmin=2)
        # data_group += sigma * np.eye(len(data_group))
        data_group = velocity[s, s]
        if len(s) == 1:
            data_group = np.array([data_group], ndmin=2)
        L_group = __cols(data_group)
        for i in group:
            k = positions[i]
            e_k = np.zeros(len(s))
            e_k[k] = 1
            col = scipy.linalg.solve_triangular(L_group, e_k, lower=True, check_finite=False)
            data[ptr[i] : ptr[i + 1]] = col[k:]
            indices[ptr[i] : ptr[i + 1]] = s[k:]
    return sparse.csc_matrix((data, indices, ptr), shape=(n, n))

points = np.ascontiguousarray(points)

order, lengths = ordering.p_reverse_maximin(points)
ordered_points = points[order]
ordered_pressure = pressure[:, order]
cov = np.cov(ordered_pressure, rowvar=False, bias=True)
cov += np.eye(len(cov))

def kl_plot(points, ordered_data, cov, ordered_points, lengths):
    delta = 0.5
    kls = []
    for i in range(10):
        rho = 2.0 + i * delta
        sparsity = ordering.sparsity_pattern(ordered_points, lengths, rho)
        groups, agg_sparsity = __supernodes(sparsity, lengths, 1.5)
        L = __aggregate_chol(points, cov, agg_sparsity, groups)
        kl = sparse_kl_div(cov, L)
        kls.append(kl)
        print(f"rho: {rho}, nnzs: {L.nnz}, kl: {kl}", flush=True)
        plt.scatter(rho, np.log(kl), color='blue')
        plt.xlabel("rho")
        plt.ylabel("log10(kl)")
    plt.savefig("1024x4x10_periodic_kl_1.png")
    return kls

kls = kl_plot(points, ordered_pressure, cov, ordered_points, lengths)
print(f"Final kls: {kls}", flush=True)

from sklearn.utils.extmath import randomized_svd

def randomized_svd_plot(cov):
    rank = 10
    kls = []
    for i in range(10):
        u, s, v = randomized_svd(cov, rank, random_state=0)
        approx = u @ np.diag(s) @ v
        kl = kl_div(cov, approx)
        kls.append(kl)
        nnzs = np.count_nonzero(u) + np.count_nonzero(s)
        print(f"rank: {rank}, nnzs: {nnzs}, kl: {kl}")
        plt.scatter(rank, np.log(kl), color='blue')
        plt.xlabel("rank")
        plt.ylabel("log10(kl)")
        rank += 10
    plt.savefig("1024x4x10_periodic_randomizedsvd_kl_1.png")
    return kls

plt.clf()
kls = randomized_svd_plot(cov)
print(f"Final kls: {kls}", flush=True)

# print(np.linalg.cond(cov))
