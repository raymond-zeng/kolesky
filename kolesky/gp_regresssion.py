import numpy as np
import scipy
import scipy.sparse as sparse
from kolesky.cholesky import fast_joint_cholesky

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