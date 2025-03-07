"""Signed distance functions."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Real


# Smooth approximations to minimum and maximum that avoid exact kinks.
@jax.jit
def __safe_max(
    a: Float[Array, "..."], b: Real[Array, "..."] | float, eps: float = 1e-6
) -> Float[Array, "..."]:
    # Smooth maximum: approx max(a, b) that is differentiable everywhere.
    return 0.5 * (a + b + jnp.sqrt((a - b) ** 2 + eps))


@jax.jit
def __safe_min(
    a: Float[Array, "..."], b: Real[Array, "..."] | float, eps: float = 1e-6
) -> Float[Array, "..."]:
    # Smooth minimum: approx min(a, b) that is differentiable everywhere.
    return 0.5 * (a + b - jnp.sqrt((a - b) ** 2 + eps))


@jax.jit
def ramp_sdf(p: jax.Array, w: Int[Array, ""], h: Real[Array, ""]) -> Float[Array, ""]:
    """Signed distance for ramp."""

    @jax.jit
    def sd_trapezoid_2d(
        p: jax.Array, w: Int[Array, ""], h: Int[Array, ""]
    ) -> Float[Array, ""]:
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
def box_sdf(p: jax.Array, w: Real[Array, ""], h: Real[Array, ""]) -> Float[Array, ""]:
    """Signed distance function for box."""
    q = jnp.abs(p[:3]) - jnp.array([w / 2, h, w / 2])
    return jnp.linalg.norm(__safe_max(q, jnp.zeros_like(q))) + __safe_min(
        __safe_max(q[0], __safe_max(q[1], q[2])), 0.0
    )


@jax.jit
def capsule_sdf(p: jax.Array, h: float, r: float) -> Float[Array, ""]:
    """Signed distance function for capsule (agent)."""
    pn = p.at[1].subtract(__safe_min(__safe_max(p[1], 0.0), h))
    return jnp.linalg.norm(pn[:3]) - r


@jax.jit
def sphere_sdf(p: jax.Array, r: float) -> Float[Array, ""]:
    """Signed distance function for a sphere."""
    return jnp.linalg.norm(p) - r


@jax.jit
def cube_sdf(p: jax.Array, size: float) -> Float[Array, ""]:
    """Signed distance function for a cube."""
    return box_sdf(p, jnp.array(size), jnp.array(size / 2))


@jax.jit
def beam_sdf(
    p: jax.Array, view_dir: jax.Array, beam_range: jax.Array
) -> Float[Array, ""]:
    """Special signed distance function for a tagging beam."""
    # SDF of a cylinder: from agent_pos to agent_pos + view_dir * beam_range, radius 0.01
    # Pre: the view_dir is normalized
    BEAM_RADIUS = 0.01
    b = view_dir * beam_range

    bb = jnp.dot(b, b)
    pb = jnp.dot(p, b)

    x = jnp.linalg.norm(p * bb - b * pb) - BEAM_RADIUS * bb
    y = jnp.abs(pb - bb * 0.5) - bb * 0.5
    x2 = x * x
    y2 = y * y * bb

    selector = __safe_max(x, y) < 0
    dist = selector * -__safe_min(x2, y2) + (1 - selector) * (
        (x > 0) * x2 + (y > 0) * y2
    )

    return jnp.sign(dist) * jnp.sqrt(jnp.abs(dist)) / bb
