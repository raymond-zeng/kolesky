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
seed = 43
np.random.seed(seed)
nTrain = 30
nTest = 10
noise = np.ones(nTrain + nTest) * 0.1
h = 1 / nTrain
xTrain = np.linspace(start=nTrain, stop = 1, num = nTrain)
row1 = np.sin(2 * np.pi * xTrain)
row2 = np.cos(2 * np.pi * xTrain)
xTrain = np.vstack((row1, row2)).T
# print(xTrain.shape)
xTest = 0.1 * np.random.rand(nTest, 2)
x = np.vstack((xTrain, xTest))
xTrain = np.ascontiguousarray(xTrain, dtype=np.float64)
for i in range(nTrain):
    for j in range(i + 1, nTrain):
        if np.isclose(xTrain[i, 0], xTrain[j, 0] and np.isclose(xTrain[i, 1], xTrain[j, 1])):
            print(i, j)
xTest = np.ascontiguousarray(xTest, dtype=np.float64)
# for i in range(nTrain + nTest):
#     for j in range(i + 1, nTrain + nTest):
#         if np.isclose(x[i], x[j]):
#             print(i, j)
kernel = kernels.Matern(nu=0.5, length_scale=1)
theta = kernel(x) + np.diag(noise)
L = np.linalg.cholesky(theta)
y = L @ np.random.randn(nTrain + nTest)

mu, sigma, L, order = gp.estimate(xTrain, y[:nTrain], xTest, kernel, rho=3.0, lamb=1.5, noise=noise[:nTrain], p=1)
#plot xTrain and yTrain in 3D
ordered_xTrain = xTrain[order[:nTrain]]
ordered_yTrain = y[:nTrain][order[:nTrain]]
ordered_xTest = xTest[order[nTrain:] - nTrain]
print(mu)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ordered_xTrain[:, 0], ordered_xTrain[:, 1], ordered_yTrain, c='b', label='Train Data')
ax.scatter(ordered_xTrain[:, 0], ordered_xTrain[:, 1], mu[:nTrain], c='r', label='Predicted Mean')
ax.scatter(ordered_xTest[:, 0], ordered_xTest[:, 1], mu[nTrain:], c='g', label='Predicted Test Mean')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()


# noise = 0.1 * np.ones()
# L, order = kolesky.test_ichol(points, kernel, 10.0)
# ordered_points = points[order]
# theta = kernel(ordered_points)
# # print(L)
# approx = L @ L.T
# print(kl_div(approx, theta))