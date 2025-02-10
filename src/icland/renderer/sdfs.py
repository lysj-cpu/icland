"""Signed distance functions."""

import jax
import jax.numpy as jnp


# Smooth approximations to minimum and maximum that avoid exact kinks.
def __safe_max(a: jnp.float32, b: jnp.float32, eps: jnp.float32 = 1e-6) -> jnp.float32:
    # Smooth maximum: approx max(a, b) that is differentiable everywhere.
    return 0.5 * (a + b + jnp.sqrt((a - b) ** 2 + eps))


def __safe_min(a: jnp.float32, b: jnp.float32, eps: jnp.float32 = 1e-6) -> jnp.float32:
    # Smooth minimum: approx min(a, b) that is differentiable everywhere.
    return 0.5 * (a + b - jnp.sqrt((a - b) ** 2 + eps))


@jax.jit
def ramp_sdf(p: jax.Array, w: jnp.float32, h: jnp.float32) -> jnp.float32:
    """Signed distance for ramp."""

    @jax.jit
    def sd_trapezoid_2d(p: jax.Array, w: jnp.float32, h: jnp.float32) -> jnp.float32:
        # --- Rectangle part ---
        rect_center = jnp.array([(w - h) * 0.5, h * 0.5])
        rect_half = jnp.array([(w - h) * 0.5, h * 0.5])
        p_rect = p - rect_center
        # Compute the signed distance for the rectangle.
        d_rect = jnp.abs(p_rect) - rect_half
        # Replace non-differentiable max with __safe_max applied elementwise.
        d_rect_pos0 = __safe_max(d_rect[0], 0.0)
        d_rect_pos1 = __safe_max(d_rect[1], 0.0)
        norm_rect = jnp.sqrt(d_rect_pos0 * 2 + d_rect_pos1 * 2)
        # For the interior, use a smooth min.
        interior_rect = __safe_min(__safe_max(d_rect[0], d_rect[1]), 0.0)
        sd_rect = norm_rect + interior_rect

        # --- Triangle part ---
        # The triangle is a right isosceles triangle with vertices:
        # (w-h,0), (w,0), (w-h,h)
        q = p - jnp.array([w - h, 0.0])
        sqrt2 = jnp.sqrt(2.0)
        # Use __safe_min for the inner minimum and __safe_max for the outer maximum.
        sd_tri = __safe_max((q[0] + q[1] - h) / sqrt2, -__safe_min(q[0], q[1]))

        # The trapezoid is the union of the rectangle and triangle.
        result = __safe_min(sd_rect, sd_tri)
        return result

    d2d = sd_trapezoid_2d(p[:2], w, h)
    d_z = jnp.abs(p[2]) - h * 0.5
    # Combine the 2D distance and the distance along z.
    d0 = __safe_max(d2d, 0.0)
    d1 = __safe_max(d_z, 0.0)
    outside = jnp.sqrt(d0 * 2 + d1 * 2)
    inside = __safe_min(__safe_max(d2d, d_z), 0.0)
    result = inside + outside
    return result


@jax.jit
def box_sdf(p: jax.Array, w: jnp.float32, h: jnp.float32) -> jnp.float32:
    """Signed distance function for box."""
    q = jnp.abs(p[:3]) - jnp.array([w / 2, h, w / 2])
    return jnp.linalg.norm(__safe_max(q, jnp.zeros_like(q))) + __safe_min(
        __safe_max(q[0], __safe_max(q[1], q[2])), 0.0
    )


@jax.jit
def capsule_sdf(p: jax.Array, h: jnp.float32, r: jnp.float32) -> jnp.float32:
    """Signed distance function for capsule (agent)."""
    py = p[1] - __safe_min(__safe_max(p[1], 0.0), h)
    return jnp.linalg.norm(p[:3]) - r


@jax.jit
def sphere_sdf(p: jax.Array, r: jnp.float32) -> jnp.float32:
    """Signed distance function for a sphere."""
    return jnp.linalg.norm(p) - r


@jax.jit
def cube_sdf(p: jax.Array, size: jnp.float32) -> jnp.float32:
    """Signed distance function for a cube."""
    return box_sdf(p, size, size / 2)