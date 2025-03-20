import numpy as np
import sklearn.gaussian_process.kernels as kernels
import kolesky

np.random.seed(0)
n = 50
points = np.zeros((n * n, 2))
for i in range(n):
    for j in range(n):
        perturbation = np.random.uniform(-0.2, 0.2, 2)
        points[i * n + j] = np.array([i - n/2, j - n/2]) + perturbation

L, ordering = kolesky.kl_cholesky(points, kernels.Matern(nu=0.5, length_scale=1), 3.0, 1.5)
print(L)