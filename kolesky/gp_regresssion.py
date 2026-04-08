import numpy as np
import scipy
import scipy.sparse as sparse
from kolesky.cholesky import joint_cholesky, fast_joint_cholesky, train_test_order, __cols
from kolesky.nugget import ichol, parallel_ichol
from kolesky.ordering import p_reverse_maximin, sparsity_pattern, supernodes

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
    groups_data, groups_indptr, agg_data, agg_indptr = supernodes(sparsity, lengths, lamb)
    
    # Convert CSR format back to lists for compatibility
    groups = [list(groups_data[groups_indptr[i]:groups_indptr[i+1]]) 
              for i in range(len(groups_indptr) - 1)]
    agg_sparsity = {}
    for g in groups:
        leader = g[0]
        agg_sparsity[leader] = list(agg_data[agg_indptr[leader]:agg_indptr[leader+1]])
    
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