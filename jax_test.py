import kolesky.jax_cholesky as koleskyjax
import jax
import jax.numpy as jnp
import gpjax as gpx
import jax.lax as lax
from jax import jit, vmap

# @jit
def func():
    a = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return jnp.linalg.cholesky(a @ a.T)

# @jit
def main():
    a = jnp.array([[1, 0], [0, 1]])
    print(jnp.linalg.inv(a))

if __name__ == "__main__":
    main()