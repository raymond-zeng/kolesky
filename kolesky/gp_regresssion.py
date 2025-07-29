import numpy as np
import scipy
import scipy.sparse as sparse
from kolesky.cholesky import joint_cholesky, fast_joint_cholesky
from kolesky.nugget import ichol
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

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
        self.inv_noise = 1 / noise
        self.LICholNoise = IChol(L, U, inv_sigma=1 / noise)
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
    n = len(x_test)
    L, order = fast_joint_cholesky(x_train, x_test, kernel, rho, lamb, p=p)
    inv_test_order, train_order = inv_order(order[:n]), order[n:] - n
    L11 = L[:n, :n]
    L21 = L[n:, :n]
    mu = -sparse.linalg.solve_triangular(L11.T, L21.T @ y_train(train_order), lower=False)
    e_i = sparse.linalg.solve_triangular(L11.T, np.eye(n), lower=False)
    var = np.sum(e_i * e_i, axis=0)
    return mu[inv_test_order], var[inv_test_order], L

def estimate(x_train, y_train, x_test, kernel, rho, lamb, noise, p=1):
    n = len(x_train)
    L, order = joint_cholesky(x_train, x_test, kernel, rho, lamb, p=p)
    train_order = order[:n]
    ordered_y_train = y_train[train_order]
    ordered_noise = noise[train_order]
    # inv_train_order, test_order = inv_order(order[:n]), order[n:] - n
    inv_test_order = inv_order(order[n:] - n)
    L11 = L[:n, :n]
    L22 = L[n:, n:]
    U11 =  sparse.triu(L11 @ L11.T, format='csc') + sparse.csc_matrix(np.diag(1 / ordered_noise))
    ichol(U11.indptr, U11.indices, U11.data)
    noiseCov = NoiseCov(L11, U11, ordered_noise)
    mu_temp = noiseCov.solve(ordered_y_train)
    mu_temp = np.hstack((np.zeros((len(x_test))), mu_temp))
    mu_temp = sparse.linalg.spsolve_triangular(L, mu_temp, lower=True)
    mu_pred = sparse.linalg.spsolve_triangular(L.T, mu_temp, lower=False)
    sigma_pred = np.linalg.inv((L22 @ L22.T).toarray())
    # LIChol = L11 @ L11.T
    # LICholNoise = LIChol + sparse.csc_matrix(np.diag(1 / ordered_noise))
    # UIChol = U11.T @ U11
    # noisecov = (1 / noise) * cg(LICholNoise, LIChol @ ordered_y_train, M=UIChol, atol=1e-6, maxiter=5)[0]
    # mu_temp = np.hstack((np.zeros((len(x_test))), noisecov))
    # mu_temp = sparse.linalg.spsolve_triangular(L, mu_temp, lower=True)
    # mu_pred = sparse.linalg.spsolve_triangular(L.T, mu_temp, lower=False)
    # sigma_pred = np.linalg.inv((L22 @ L22.T).toarray())
    return mu_pred, sigma_pred, L, order
    