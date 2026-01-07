import numpy as np
import scipy
import scipy.sparse as sparse
from kolesky.cholesky import joint_cholesky, fast_joint_cholesky, train_test_order, __supernodes, __cols
from kolesky.nugget import ichol, parallel_ichol
from kolesky.ordering import p_reverse_maximin, sparsity_pattern
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from copy import deepcopy

class IChol:

    def __init__(self, L, U, inv_sigma=None):
        self.L = L
        self.U = U
        if inv_sigma is None:
            self.inv_sigma = np.zeros(L.shape[0])
        else:
            self.inv_sigma = inv_sigma
    
    def solve(self, x):
        temp = sparse.linalg.solve_triangular(self.L, x, lower=True)
        return sparse.linalg.solve_triangular(self.U, temp, lower=False)

    def __matmul__(self, x):
        return self.L @ (self.U @ x) + self.inv_sigma * x

class NoiseCov:

    def __init__(self, L, U, noise):
        self.L = L
        self.U = U
        if noise is not None:
            self.inv_noise = 1 / noise
            self.LICholNoise = IChol(L, U, inv_sigma=1 / noise)
        else:
            self.inv_noise = None
            self.LICholNoise = IChol(L, U)
        self.LIChol = IChol(L, L.T)
        self.UIChol = IChol(U.T, U)

    def solve(self, x):
        LICholNoiseOperator = LinearOperator(shape=self.LICholNoise.L.shape, matvec=self.LICholNoise.__matmul__)
        UICholOperator = LinearOperator(shape=self.UIChol.L.shape, matvec=self.UIChol.__matmul__)
        return self.inv_noise * sparse.linalg.cg(LICholNoiseOperator, self.LIChol @ x, M=UICholOperator, maxiter=5, atol=1e-6)[0]
    
    def __matmul__(self, x):
        temp = x / self.inv_noise
        temp = self.LICholNoise @ temp
        return self.LIChol.solve(temp)

def inv_order(order):
    n = len(order)
    inv_order = np.arange(n)
    inv_order[order] = np.arange(n)
    return inv_order 

def fast_estimate(x_train, y_train, x_test, kernel, rho, lamb, p=1):
    #Prediction points first without noise
    n = len(x_test)
    L, order = fast_joint_cholesky(x_train, x_test, kernel, rho, lamb, p=p)
    inv_test_order, train_order = inv_order(order[:n]), order[n:] - n
    L11 = L[:n, :n]
    L21 = L[n:, :n]
    mu = -sparse.linalg.spsolve_triangular(L11.T, L21.T @ y_train[train_order], lower=False)
    e_i = sparse.linalg.spsolve_triangular(L11, np.eye(n), lower=True)
    var = np.sum(e_i * e_i, axis=0)
    return mu[inv_test_order], var[inv_test_order], L

def fast_estimate_with_noise(x_train, y_train, x_test, kernel, rho, lamb, noise, p=1):
    #Prediction points first with noise
    n = len(x_test)
    m = len(x_train)
    L, order = fast_joint_cholesky(x_train, x_test, kernel, rho, lamb, p=p)
    print(L)
    inv_test_order, train_order = inv_order(order[:n]), order[n:] - n
    ordered_y_train = y_train[train_order]
    if noise is not None:
        ordered_noise = noise[train_order]
    L11 = L[:n, :n]
    L21 = L[n:, :n]
    L22 = L[n:, n:]
    # points = np.vstack((x_test[order[:n]], x_train[train_order]))
    # cov = kernel(points)
    # noiseCov = cov[n:, n:] + np.diag(ordered_noise * ordered_noise)
    # Theta22_approx_inv= np.linalg.inv(noiseCov)
    # Theta12_approx = cov[:n, n:]
    A = sparse.triu(L22 @ L22.T, format='csc')
    if noise is not None:
        A += sparse.csc_matrix(np.diag(1 / (ordered_noise * ordered_noise)))
    ichol(A.indptr, A.indices, A.data)
    U22_tilde_inv = sparse.linalg.spsolve_triangular(A, np.eye(m), lower=False)
    Theta22_approx_inv = np.diag(1 / (ordered_noise * ordered_noise)) @ U22_tilde_inv @ U22_tilde_inv.T @ L22 @ L22.T
    L22_inv = sparse.linalg.spsolve_triangular(L22, np.eye(m), lower=True)
    U12 = -sparse.linalg.inv(L11.T) @ L21.T @ L22_inv.T
    Theta12_approx =  U12 @ L22_inv
    mu = Theta12_approx @ Theta22_approx_inv @ ordered_y_train
    mu = mu[inv_test_order]
    L11_inv = sparse.linalg.spsolve_triangular(L11, np.eye(n), lower=True)
    Theta11_approx = L11_inv.T @ L11_inv + U12 @ U12.T
    cov = Theta11_approx - Theta12_approx @ Theta22_approx_inv @ Theta12_approx.T
    var = np.diag(cov)
    return mu, var

def estimate(x_train, y_train, x_test, kernel, rho, lamb, p=1):
    #Prediction points last
    n = len(x_train)
    L, order = joint_cholesky(x_train, x_test, kernel, rho, lamb, p=p)
    train_order = order[:n]
    ordered_y_train = y_train[train_order]
    inv_test_order = inv_order(order[n:] - n)
    L11 = L[:n, :n]
    L21 = L[n:, :n]
    L22 = L[n:, n:]
    A21 = L21 @ L21.T
    mu = -sparse.linalg.spsolve(A21 + L22 @ L22.T, L21 @ L11.T @ ordered_y_train)
    cov = sparse.linalg.spsolve(L21 @ L21.T + L22 @ L22.T, np.eye(L22.shape[0]))
    return mu[inv_test_order], np.diag(cov)[inv_test_order]

def estimate_optimized(x_train, y_train, x_test, kernel, rho, lamb, p=1):
    """
    Calculates GP posterior mean and variance using an optimized iterative approach.
    """
    n = len(x_train)
    m = len(x_test)
    
    L, order, agg_sparsity, groups = joint_cholesky(x_train, x_test, kernel, rho, lamb, p=p)
    
    train_order = order[:n]
    test_order = order[n:]
    ordered_y_train = y_train[train_order]
    inv_test_order = inv_order(test_order - n)
    
    L11 = L[:n, :n]
    L21 = L[n:, :n]
    L22 = L[n:, n:]
    
    A_b_b = (L22 @ L22.T).toarray()
    
    rhs = np.zeros(m)

    for j in range(n):
        l_b_j = L21[:, j].toarray().flatten()
        l_Tr_j = L11[:, j].toarray().flatten()
        
        A_b_b += np.outer(l_b_j, l_b_j)
        
        c_j = l_Tr_j.T @ ordered_y_train
        rhs += l_b_j * c_j
        
    mu_ordered = -np.linalg.solve(A_b_b, rhs)
    cov_matrix_ordered = np.linalg.inv(A_b_b)
    var_ordered = np.diag(cov_matrix_ordered)

    return mu_ordered[inv_test_order], var_ordered[inv_test_order]

def estimate_optimized_supernodal(x_train, y_train, x_test, kernel, rho, lamb, p=1):
    n = len(x_train)
    m = len(x_test)

    train_order, lengths = p_reverse_maximin(x_train, p=p)
    sparsity = sparsity_pattern(x_train[train_order], lengths, rho)
    groups, agg_sparsity = __supernodes(sparsity, lengths, lamb)
    x = np.vstack((x_train[train_order], x_test))

    ordered_y_train = y_train[train_order]

    mu = np.zeros(m)
    delta = kernel.diag(x[n:])
    sigma = 1 / delta

    for group in groups:
        s = np.array(agg_sparsity[group[0]])
        if len(s) == 0:
            continue
        
        U = __cols(kernel(x[s])).T
        B = kernel(x[s], x[n:])
        B = scipy.linalg.solve_triangular(U, B, lower=False)
        y = scipy.linalg.solve_triangular(U, ordered_y_train[s], lower=False)

        alpha = y.T @ B
        beta = np.sum(B * B, axis=0)

        for k in range(len(group)):
            B_k = B[k, :]
            y_k = y[k]

            gamma = np.sqrt(1 + (B_k * B_k) / (delta - beta))
            l = - 1 / (delta * gamma) * B_k * (1 + beta / (delta - beta))
            mu += (l / gamma) * (y_k + (B_k * alpha) / (delta - beta))
            sigma += l * l

            alpha -= y_k * B_k
            beta -= B_k * B_k
        
    sigma = 1 / sigma
    mu = -sigma * mu
    return mu, sigma
    
def estimate_with_noise(x_train, y_train, x_test, kernel, rho, lamb, noise, p=1):
    # n = len(x_train)
    # L, order = joint_cholesky(x_train, x_test, kernel, rho, lamb, p=p)
    # train_order = order[:n]
    # ordered_y_train = y_train[train_order]
    # if noise is not None:
    #     ordered_noise = noise[train_order]
    # # inv_train_order, test_order = inv_order(order[:n]), order[n:] - n
    # inv_test_order = inv_order(order[n:] - n)
    # print(L.toarray())
    # L11 = L[:n, :n]
    # L22 = L[n:, n:]
    # U11 =  sparse.triu(L11 @ L11.T, format='csc') + sparse.csc_matrix(np.diag(1 / ordered_noise))
    # ichol(U11.indptr, U11.indices, U11.data)
    # noiseCov = NoiseCov(L11, U11, ordered_noise)
    # mu_temp = noiseCov.solve(ordered_y_train)
    # mu_temp = np.hstack((np.zeros((len(x_test))), mu_temp))
    # mu_temp = sparse.linalg.spsolve_triangular(L, mu_temp, lower=True)
    # mu_pred = sparse.linalg.spsolve_triangular(L.T, mu_temp, lower=False)
    # sigma_pred = np.linalg.inv((L22 @ L22.T).toarray())
    # # print(sigma_pred)
    # var = np.diag(sigma_pred)
    # # LIChol = L11 @ L11.T
    # # LICholNoise = LIChol + sparse.csc_matrix(np.diag(1 / ordered_noise))
    # # UIChol = U11.T @ U11
    # # noisecov = (1 / noise) * cg(LICholNoise, LIChol @ ordered_y_train, M=UIChol, atol=1e-6, maxiter=5)[0]
    # # mu_temp = np.hstack((np.zeros((len(x_test))), noisecov))
    # # mu_temp = sparse.linalg.spsolve_triangular(L, mu_temp, lower=True)
    # # mu_pred = sparse.linalg.spsolve_triangular(L.T, mu_temp, lower=False)
    # # sigma_pred = np.linalg.inv((L22 @ L22.T).toarray())
    # return mu_pred[inv_test_order], var[inv_test_order]
    n = len(x_test)
    m = len(x_train)
    L, order = fast_joint_cholesky(x_train, x_test, kernel, rho, lamb, p=p)
    inv_test_order, train_order = inv_order(order[:n]), order[n:] - n
    ordered_y_train = y_train[train_order]
    if noise is not None:
        ordered_noise = noise[train_order]
    L11 = L[:n, :n]
    L22 = L[n:, n:]
    U22 = sparse.triu(L22 @ L22.T, format='csc')
    if noise is not None:
        U22 += sparse.csc_matrix(np.diag(1 / ordered_noise / ordered_noise))
    ichol(U22.indptr, U22.indices, U22.data)
    noiseCov = NoiseCov(L22, U22, ordered_noise * ordered_noise)
    mu_temp = noiseCov.solve(ordered_y_train)
    mu_temp = np.hstack((np.zeros((len(x_test))), mu_temp))
    mu_temp = sparse.linalg.spsolve_triangular(L, mu_temp, lower=True)
    mu = sparse.linalg.spsolve_triangular(L.T, mu_temp, lower=False)
    mu = mu[inv_test_order]
    sigma = np.diag(np.linalg.inv((L11 @ L11.T).toarray()))
    return mu, sigma