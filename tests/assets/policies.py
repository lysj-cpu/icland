"""This module defines several policies as JAX numpy arrays for use in testing.

Policies:
    NOOP_POLICY: A policy that represents no operation, with all zeros.
    FORWARD_POLICY: A policy that represents moving forward, with a value of 1 in the first position.
    BACK_POLICY: A policy that represents moving backward, with a value of -1 in the first position.
    LEFT_POLICY: A policy that represents moving left, with a value of -1 in the second position.
    RIGHT_POLICY: A policy that represents moving right, with a value of 1 in the second position.
"""

import jax.numpy as jnp

NOOP_POLICY = jnp.array([0, 0, 0])
FORWARD_POLICY = jnp.array([1, 0, 0])
BACK_POLICY = jnp.array([-1, 0, 0])
LEFT_POLICY = jnp.array([0, -1, 0])
RIGHT_POLICY = jnp.array([0, 1, 0])
