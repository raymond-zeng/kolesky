from .ordering import p_reverse_maximin
from .ordering import sparsity_pattern

import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.scipy.sparse as sparse
import gpjax as gpx
from jax import jit, vmap
from jax.experimental import checkify
from scipy.spatial import KDTree
import scipy

def create_points(n, d):
    """Create a grid of points in d dimensions with uniform perturbation using numpy."""
    space = np.linspace(-n/2, n/2, n)
    cube = (space, ) * d
    return np.stack(np.meshgrid(*cube, indexing='ij'), axis=-1).reshape(-1, d) + np.random.uniform(-0.2, 0.2, (n**d, d))

def py_sparsity_pattern(points, lengths, rho):
    tree, offset, length_scale = KDTree(points), 0, lengths[0]
    sparsity = {}
    for i in range(len(points)):
        if lengths[i] > 2 * length_scale:
            tree, offset, length_scale = KDTree(points[i:]), i, lengths[i]
        sparsity[i] = jnp.array([
            offset + j
            for j in tree.query_ball_point(points[i], rho * lengths[i])
            if offset + j >= i
        ])
    return sparsity

# @jit
def col(theta):
    m = jsl.inv(theta)
    return m[:, 0] / jnp.sqrt(m[0, 0])

def chol(points, kernel, sparsity, ptr, data, indices):
    n = points.shape[0]
    flattened, _ = jax.tree_util.tree_flatten(sparsity)
    for i in range(n):
        s = flattened[i]
        pts = points[s]
        c = col(kernel.cross_covariance(pts, pts))
        data = lax.dynamic_update_slice(data, c, (ptr[i],))
        indices = lax.dynamic_update_slice(indices, s, (ptr[i],))
    return data, indices

def kl_cholesky(points, kernel, rho, initial=None, p=1):
    n = len(points)
    indices, lengths = p_reverse_maximin(points, initial, p)
    ordered_points = points[indices]

    # sparsity = sparsity_pattern(ordered_points, lengths, rho)
    sparsity = py_sparsity_pattern(ordered_points, lengths, rho)
    ptr = jnp.cumsum(jnp.array([0] + [len(sparsity[i]) for i in range(n)]))

    total_nnz = ptr[-1]
    data = jnp.zeros(total_nnz)
    indices_np = np.zeros(total_nnz, dtype=np.int32)

    for i in range(n):
        s = sorted(sparsity[i])
        indices_np[ptr[i]:ptr[i + 1]] = s

    indices = jnp.array(indices_np)
    
    # JIT-chol safely
    # chol_jitted = jax.jit(checkify.checkify(chol), static_argnums=(1))
    data_checked = chol(jnp.array(ordered_points), kernel, sparsity, ptr, data, indices)
    
    return scipy.sparse.csc_matrix((data_checked, indices, ptr), shape=(n, n)), indices