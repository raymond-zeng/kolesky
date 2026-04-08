import numpy as np
import sklearn.gaussian_process.kernels as kernels
import kolesky
import kolesky.gp_regresssion as gp
import matplotlib.pyplot as plt

def logdet_chol(A):
    return 2 * np.sum(np.log(A.diagonal()))

def kl_div(A, B):
    n = A.shape[0]
    return 0.5 * (np.trace(np.linalg.solve(B, A)) - n + np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1])

def sparse_kl_div(A, L):
    n = A.shape[0]
    return 0.5 * (-logdet_chol(L) - np.linalg.slogdet(A)[1])

# np.random.seed(0)
# n = 50
# points = np.zeros((n * n, 2))
# for i in range(n):
#     for j in range(n):
#         perturbation = np.random.uniform(-0.2, 0.2, 2)
#         points[i * n + j] = np.array([i - n/2, j - n/2]) + perturbation
# noise = np.eye(len(points)) * 0.3
# # # L, ordering = kolesky.kl_cholesky(points, kernels.Matern(nu=0.5, length_scale=1), 3.0, 1.5)
# # # print(L)
# kernel = kernels.Matern(nu=0.5, length_scale=1)
# L, U_tilde, ordering = kolesky.noise_cholesky(points, kernel, 3, 1.5, noise)
# L = L.toarray()
# ordered_points = points[ordering]
# true = kernel(ordered_points) + noise
# approx = np.linalg.inv(L @ L.T) @ U_tilde.T @ U_tilde @ noise
# kl = kl_div(true, approx)
# print("KL divergence:", kl)
seed = 42
np.random.seed(seed)
# kernel = kernels.Matern(nu=2.5, length_scale=1)
kernel = kernels.RBF(length_scale=1.0)
x_train = np.random.uniform(-5, 5, 15).reshape(-1, 1)
y_train = x_train * np.sin(x_train)
# extra_row = np.ones(15).reshape(-1, 1)
# x_train = np.hstack((x_train, extra_row))
noise_std_dev = 0.4
# noise = np.random.normal(0, noise_std_dev, size=y_train.shape)
# noise = np.ones(y_train.shape) * 1e-50
# y_train += noise
x_test = np.linspace(-5, 5, 50).reshape(-1, 1)
# extra_row = np.ones(50).reshape(-1, 1)
# x_test = np.hstack((x_test, extra_row))
# mu, var = gp.fast_estimate_with_noise(x_train, y_train, x_test, kernel, 3.0, 1.5, noise.flatten(), p=1)
# mu, var = gp.estimate_with_noise(x_train, y_train.flatten(), x_test, kernel, 3.0, 1.5, noise.flatten(), p=1)
# mu, var = gp.estimate(x_train, y_train.flatten(), x_test, kernel, 10.0, 1.5, p=1)
# mu_reference, var_reference = gp.estimate(x_train, y_train.flatten(), x_test, kernel, 10.0, 1.5, p=1)
mu, var = gp.estimate_optimized_supernodal(x_train, y_train.flatten(), x_test, kernel, 10.0, 1.5, p=1)
# print(np.linalg.norm(mu_reference - mu))
# print(np.linalg.norm(var_reference - var))
# x_train = x_train.flatten()
y_train = y_train.flatten()
# x_test = x_test.flatten()
mu = mu.flatten()
# var = var.flatten()
plt.figure(figsize=(10, 5))
plt.plot(x_train[:, 0], y_train, 'ro', label='Training Data')
plt.plot(x_test[:, 0], mu, 'b-', label='Predicted Mean')
plt.plot(x_test[:, 0], x_test[:, 0] * np.sin(x_test[:, 0]), 'g--', label='Unnoisy Function')
plt.fill_between(x_test[:, 0], mu - 1.96 * np.sqrt(var), mu + 1.96 * np.sqrt(var), color='lightblue', alpha=0.5, label='95% Confidence Interval')
plt.title('Gaussian Process Regression with Sparse Cholesky')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
plt.savefig("gp_regression.png")