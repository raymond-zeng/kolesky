# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.math cimport sqrt, exp
import numpy as np
cimport numpy as np

np.import_array()


cdef inline double euclidean_distance(double* x, double* y, int dim):
    cdef double dist = 0.0
    cdef double diff
    cdef int i
    for i in range(dim):
        diff = x[i] - y[i]
        dist += diff * diff
    return sqrt(dist)


cdef void matern_kernel_matrix(
    double[:, ::1] points,
    double[:, ::1] result,
    double length_scale,
    double nu
):
    """
    Compute Matern kernel matrix in-place (nogil compatible).
    
    Supports nu = 0.5, 1.5, 2.5 (exponential, Matern 3/2, Matern 5/2).
    
    Parameters:
    - points: (n, dim) array of points
    - result: (n, n) pre-allocated output array
    - length_scale: kernel length scale
    - nu: Matern smoothness parameter (0.5, 1.5, or 2.5)
    """
    cdef int n = points.shape[0]
    cdef int dim = points.shape[1]
    cdef int i, j
    cdef double d, scaled_d, k_val
    cdef double sqrt3 = 1.7320508075688772  # sqrt(3)
    cdef double sqrt5 = 2.23606797749979    # sqrt(5)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                result[i, j] = 1.0
            else:
                d = euclidean_distance(&points[i, 0], &points[j, 0], dim)
                scaled_d = d / length_scale
                
                if nu == 0.5:
                    # Exponential kernel: exp(-d / l)
                    k_val = exp(-scaled_d)
                elif nu == 1.5:
                    # Matern 3/2: (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)
                    k_val = (1.0 + sqrt3 * scaled_d) * exp(-sqrt3 * scaled_d)
                elif nu == 2.5:
                    # Matern 5/2: (1 + sqrt(5)*d/l + 5*d^2/(3*l^2)) * exp(-sqrt(5)*d/l)
                    k_val = (1.0 + sqrt5 * scaled_d + 5.0 * scaled_d * scaled_d / 3.0) * exp(-sqrt5 * scaled_d)
                else:
                    # Default to exponential
                    k_val = exp(-scaled_d)
                
                result[i, j] = k_val
                result[j, i] = k_val


cpdef np.ndarray[np.float64_t, ndim=2] matern_kernel(
    double[:, ::1] points,
    double length_scale = 1.0,
    double nu = 0.5
):
    """
    Compute Matern kernel matrix.
    
    This function can be used as a drop-in replacement for sklearn kernels:
        kernel = lambda pts: matern_kernel(pts, length_scale=1.0, nu=0.5)
        L_group = __cols(kernel(points[s]))
    
    Parameters:
    - points: (n, dim) array of points
    - length_scale: kernel length scale (default 1.0)
    - nu: Matern smoothness parameter (0.5, 1.5, or 2.5, default 0.5)
    
    Returns:
    - (n, n) kernel matrix
    """
    cdef int n = points.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty((n, n), dtype=np.float64)
    cdef double[:, ::1] result_view = result
    
    matern_kernel_matrix(points, result_view, length_scale, nu)
    
    return result


cdef class MaternKernel:
    """
    Matern kernel class with sklearn-like callable interface.
    
    Usage:
        kernel = MaternKernel(length_scale=1.0, nu=0.5)
        K = kernel(points)  # Compute kernel matrix
    """
    cdef public double length_scale
    cdef public double nu
    
    def __init__(self, double length_scale=1.0, double nu=0.5):
        self.length_scale = length_scale
        self.nu = nu
    
    def __call__(self, double[:, ::1] points):
        """Compute the kernel matrix for the given points."""
        return matern_kernel(points, self.length_scale, self.nu)
    
    cpdef void compute_inplace(
        self,
        double[:, ::1] points,
        double[:, ::1] result
    ):
        """
        Compute kernel matrix in-place (nogil-compatible inner loop).
        
        Parameters:
        - points: (n, dim) input points
        - result: (n, n) pre-allocated output array
        """
        matern_kernel_matrix(points, result, self.length_scale, self.nu)
