"""This module defines several policies as JAX numpy arrays for use in testing.

Policies:
    NOOP_POLICY: A policy that represents no operation, with all zeros.
    FORWARD_POLICY: A policy that represents moving forward, with a value of 1 in the first position.
    BACKWARD_POLICY: A policy that represents moving backward, with a value of -1 in the first position.
    LEFT_POLICY: A policy that represents moving left, with a value of -1 in the second position.
    RIGHT_POLICY: A policy that represents moving right, with a value of 1 in the second position.
    ANTI_CLOCKWISE_POLICY: A policy that represents turning anti-clockwise, with a value of -1 in the third position.
    CLOCKWISE_POLICY: A policy that represents turning clockwise, with a value of 1 in the third position.
"""

import jax.numpy as jnp

NOOP_POLICY = jnp.array([0, 0, 0])
FORWARD_POLICY = jnp.array([1, 0, 0])
BACKWARD_POLICY = jnp.array([-1, 0, 0])
LEFT_POLICY = jnp.array([0, 1, 0])
RIGHT_POLICY = jnp.array([0, -1, 0])
ANTI_CLOCKWISE_POLICY = jnp.array([0, 0, -1])
CLOCKWISE_POLICY = jnp.array([0, 0, 1])
