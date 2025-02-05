"""Signed distance functions."""

import jax
import jax.numpy as jnp
from jax._src import pjit
from typing import NamedTuple
import numpy as np
from jax.experimental import host_callback as hcb


@jax.jit
def __smooth_min(a: jnp.float32, b: jnp.float32, k: jnp.float32):
    h = jnp.minimum(jnp.maximum(0.5 + 0.5 * (b - a) / k, 0.0), 1.0)
    return (b * (1 - h) + a * h) - k * h * (1.0 - h )

@jax.jit
def ramp_sdf(p, w, h):
    """Signed distance for ramp."""
    
    @jax.jit
    def sd_trapezoid_2d(p, w, h):
        # --- Rectangle part ---
        rect_center = jnp.array([(w - h) * 0.5, h * 0.5])
        rect_half = jnp.array([(w - h) * 0.5, h * 0.5])
        p_rect = p - rect_center
        # Compute the signed distance for the rectangle.
        d_rect = jnp.abs(p_rect) - rect_half
        # Replace non-differentiable max with safe_max applied elementwise.
        d_rect_pos0 = safe_max(d_rect[0], 0.0)
        d_rect_pos1 = safe_max(d_rect[1], 0.0)
        norm_rect = jnp.sqrt(d_rect_pos0*2 + d_rect_pos1*2)
        # For the interior, use a smooth min.
        interior_rect = safe_min(safe_max(d_rect[0], d_rect[1]), 0.0)
        sd_rect = norm_rect + interior_rect

        # --- Triangle part ---
        # The triangle is a right isosceles triangle with vertices:
        # (w-h,0), (w,0), (w-h,h)
        q = p - jnp.array([w - h, 0.0])
        sqrt2 = 1.41421356
        # Use safe_min for the inner minimum and safe_max for the outer maximum.
        sd_tri = safe_max((q[0] + q[1] - h) / sqrt2, -safe_min(q[0], q[1]))

        # The trapezoid is the union of the rectangle and triangle.
        result = safe_min(sd_rect, sd_tri)
        return result

    # Smooth approximations to minimum and maximum that avoid exact kinks.
    def safe_max(a: jnp.float32, b: jnp.float32, eps=1e-6):
        # Smooth maximum: approx max(a, b) that is differentiable everywhere.
        return 0.5 * (a + b + jnp.sqrt((a - b) ** 2 + eps))


    def safe_min(a: jnp.float32, b: jnp.float32, eps=1e-6):
        # Smooth minimum: approx min(a, b) that is differentiable everywhere.
        return 0.5 * (a + b - jnp.sqrt((a - b) ** 2 + eps))
    
    d2d = sd_trapezoid_2d(p[:2], w, h)
    d_z = jnp.abs(p[2]) - h * 0.5
    # Combine the 2D distance and the distance along z.
    d0 = safe_max(d2d, 0.0)
    d1 = safe_max(d_z, 0.0)
    outside = jnp.sqrt(d0*2 + d1*2)
    inside = safe_min(safe_max(d2d, d_z), 0.0)
    result = inside + outside
    return result

@jax.jit
def box_sdf(p: jax.Array, w: jnp.float32, h: jnp.float32) -> jnp.float32:
    """Signed distance function for box."""
    q = jnp.abs(p[:3]) - jnp.array([w, h, h])
    return jnp.linalg.norm(jnp.maximum(q, jnp.zeros_like(q))) + jnp.minimum(
        jnp.maximum(q[0], jnp.maximum(q[1], q[2])), 0.0
    )


@jax.jit
def capsule_sdf(p: jax.Array, h: jnp.float32, r: jnp.float32) -> jnp.float32:
    """Signed distance function for capsule (agent)."""
    py = p[1] - jnp.minimum(jnp.maximum(p[1], 0.0), h)
    return jnp.linalg.norm(p[:3]) - r


def _process_column(
    tt: jnp.int32,
    i: jnp.int32,
    p: jax.Array,
    x: jnp.float32,
    y: jnp.float32,
    rot: jnp.int32,
    w: jnp.float32,
    h: jnp.float32,
) -> pjit.JitWrapped:
    angle = -jnp.pi * rot / 2
    cos_t = jnp.cos(angle)
    sin_t = jnp.sin(angle)
    transformed = jnp.matmul(
        jnp.linalg.inv(
            jnp.array(
                [
                    [1, 0, 0, (x + 0.5) * w],
                    [0, 1, 0, (h * w) / 2],
                    [0, 0, 1, (y + 0.5) * w],
                    [0, 0, 0, 1],
                ]
            )
        ),
        jnp.append(p, 1),
    )
    return box_sdf(transformed[:3], w, (h * w) / 2)


def _process_ramp(
    tt: jnp.int32,
    i: jnp.int32,
    p: jax.Array,
    x: jnp.float32,
    y: jnp.float32,
    rot: jnp.int32,
    w: jnp.float32,
    h: jnp.float32,
) -> pjit.JitWrapped:
    angle = -jnp.pi * rot / 2
    cos_t = jnp.cos(angle)
    sin_t = jnp.sin(angle)
    upright = jnp.array([[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    rotation = jnp.matmul(
        jnp.array(
            [
                [cos_t[0], 0, sin_t[0], x * h],
                [0, 1, 0, 0],
                [-sin_t[0], 0, cos_t[0], y * h],
                [0, 0, 0, 1],
            ]
        ),
        jnp.array([[1, 0, 0, -0.5 * h], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )
    rotation = jnp.matmul(
        jnp.array([[1, 0, 0, 0.5 * h], [0, 1, 0, 0], [0, 0, 1, 0.5 * h], [0, 0, 0, 1]]),
        rotation,
    )
    transformed = jnp.matmul(
        jnp.linalg.inv(
            jnp.matmul(
                rotation,
                upright,
            )
            # upright
        ),
        jnp.append(p, 1),
    )
    return ramp_sdf(transformed[:3], w, h)

def _scene_sdf_from_tilemap(
    objects: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], p: jax.Array, floor_level=-3.0, has_floor=False
):

    # 10, 10
    w, h = objects.pos.shape[0], objects.pos.shape[1]
    dists = jnp.arange(w * h, dtype=jnp.int32)
    tile_width = 1
    process_tile = lambda i, p, x, y, t_size, t_color, t_rot, t_type: jax.lax.switch(
        t_type[0], # type: cuboid, ramp
        [
            lambda tt, i, p, x, y, rot, w, h: _process_column(tt, i, p, x, y, rot, w, h),
            lambda tt, i, p, x, y, rot, w, h: _process_ramp(tt, i, p, x, y, rot, h, w),
            lambda tt, i, p, x, y, rot, w, h: _process_column(tt, i, p, x, y, rot, w, h),
        ],
        t_type[0],
        i,
        p,
        x,
        y,
        t_rot,  # rotation
        tile_width,
        t_size[1],  # highest height
    )
    jax.debug.print("tile_type")
    tile_dists = jax.vmap(
        lambda i: process_tile(i, p, objects.pos[i // w, i % w][0], objects.pos[i // w, i % w][1], objects.size[i // w, i % w], objects.color[i // w, i % w], objects.rotation[i // w, i % w], objects.tile_type[i // w, i % w])
    )(dists)

    floor = p[1] - floor_level

    return jax.lax.cond(
        has_floor,
        lambda _: jnp.minimum(floor, tile_dists.min()),
        lambda _: tile_dists.min(),
        None,
    )


if __name__ == "__main__":
    jitted = jax.jit(capsule_sdf)
    for i in range(10):
        d = i / 2
        print(jitted(jnp.full((3,), d), 2, 1))
        # print(_scene_sdf_from_tilemap(one_hot, jnp.full((3,), d)))
