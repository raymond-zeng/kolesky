import numpy as np
import sklearn.gaussian_process.kernels as kernels
import kolesky

def logdet_chol(A):
    return 2 * np.sum(np.log(A.diagonal()))

def kl_div(A, B):
    n = A.shape[0]
    return 0.5 * (np.trace(np.linalg.solve(B, A)) - n + np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1])

def sparse_kl_div(A, L):
    n = A.shape[0]
    return 0.5 * (-logdet_chol(L) - np.linalg.slogdet(A)[1])

np.random.seed(0)
n = 4
points = np.zeros((n * n, 2))
for i in range(n):
    for j in range(n):
        perturbation = np.random.uniform(-0.2, 0.2, 2)
        points[i * n + j] = np.array([i - n/2, j - n/2]) + perturbation
noise = np.eye(len(points))
# # L, ordering = kolesky.kl_cholesky(points, kernels.Matern(nu=0.5, length_scale=1), 3.0, 1.5)
# # print(L)
kernel = kernels.Matern(nu=0.5, length_scale=1)
L, U_tilde, ordering = kolesky.noise_cholesky(points, kernel, 10, 1.5, noise)
L = L.toarray()
ordered_points = points[ordering]
true = kernel(ordered_points) + noise
approx = np.linalg.inv(L.T @ L) @ U_tilde.T @ U_tilde @ noise
print(approx)
print("True covariance matrix:\n", true)
kl = kl_div(true, approx)
print("KL divergence:", kl)

# L, order = kolesky.test_ichol(points, kernel, 10.0)
# ordered_points = points[order]
# theta = kernel(ordered_points)
# # print(L)
# approx = L @ L.T
# print(kl_div(approx, theta))