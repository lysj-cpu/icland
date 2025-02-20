import jax
import jax.numpy as jnp
from jax import lax, vmap
import time

# Define condition and body functions for jax.lax.while_loop

n = 100_000
def cond_fun(x):
    return x < n

def body_fun(x):
    return x + 1

# JAX while loop for a single value
def jax_while_loop(x):
    return lax.while_loop(cond_fun, body_fun, x)

# Standard while loop for a single value
def python_while_loop(x):
    while x < 100_000:
        x += 1
    return x

# Test with batch sizes of powers of 2
batch_sizes = [2**i for i in range(4, 21)]  # 16 to 1024

print(f"{'Batch Size':<10}{'Python While (s)':<20}{'JAX While (s)':<20}{'JAX Compiled (s)':<20}")

for batch_size in batch_sizes:
    inputs = jnp.zeros(batch_size, dtype=jnp.int32)

    # Python while loop with batch
    # start = time.time()
    # list(map(python_while_loop, inputs))
    # python_time = time.time() - start

    # JAX while loop with vmap (first run with compilation)
    start = time.time()
    jax.vmap(jax_while_loop)(inputs)
    jax_time = time.time() - start

    # JAX while loop with vmap (compiled run)
    start = time.time()
    jax.vmap(jax_while_loop)(inputs)
    compiled_jax_time = time.time() - start

    print(f"{batch_size:<10}{jax_time:<20.6f}{compiled_jax_time:<20.6f}")
