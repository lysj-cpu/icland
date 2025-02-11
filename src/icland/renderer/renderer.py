from functools import partial  # noqa: D100
from typing import Any, Callable, List, Tuple

import imageio
import jax
import jax.numpy as jnp
import numpy as np

from icland.renderer.sdfs import box_sdf, ramp_sdf
from icland.types import ICLandState

# Enable JAX debug flags
# jax.config.update("jax_debug_nans", True)  # Check for NaNs
jax.config.update("jax_log_compiles", True)  # Log compilations
# jax.config.update("jax_debug_infs", True)  # Check for infinities

# Constants
DEFAULT_VIEWSIZE: Tuple[jnp.int32, jnp.int32] = (92, 76)
DEFAULT_COLOR: jax.Array = jnp.array([0.2588, 0.5294, 0.9607])
WORLD_UP: jax.Array = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
NUM_CHANNELS: jnp.int32 = 3


@partial(jax.jit, static_argnames=["axis", "keepdims"])
def __norm(
    v: jax.Array,
    axis: jnp.int32 = -1,
    keepdims: jnp.bool = False,
    eps: jnp.float32 = 0.0,
) -> jax.Array:
    return jnp.sqrt((v * v).sum(axis, keepdims=keepdims).clip(min=eps))


@jax.jit
def __normalize(
    v: jax.Array, axis: jnp.int32 = -1, eps: jnp.float32 = 1e-20
) -> jax.Array:
    return v / __norm(v, axis, keepdims=True, eps=eps)


@jax.jit
def __process_column(
    p: jax.Array,
    x: jnp.float32,
    y: jnp.float32,
    rot: jnp.int32,
    w: jnp.float32,
    h: jnp.float32,
) -> jnp.float32:
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


@jax.jit
def __process_ramp(
    p: jax.Array,
    x: jnp.float32,
    y: jnp.float32,
    rot: jnp.int32,
    h: jnp.float32,
    w: jnp.float32,
) -> jnp.float32:
    angle = -jnp.pi * rot / 2
    cos_t = jnp.cos(angle)
    sin_t = jnp.sin(angle)
    upright = jnp.array([[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    rotation = jnp.matmul(
        jnp.array(
            [
                [cos_t, 0, sin_t, x * h],
                [0, 1, 0, 0],
                [-sin_t, 0, cos_t, y * h],
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


@jax.jit
def __scene_sdf_from_tilemap(
    tilemap: jax.Array, p: jax.Array, floor_height: jnp.float32 = 0.0
) -> jax.Array:
    w, h = tilemap.shape[0], tilemap.shape[1]
    dists = jnp.arange(w * h, dtype=jnp.int32)
    tile_width = 1

    def process_tile(
        p: jax.Array, x: jnp.int32, y: jnp.int32, tile: jax.Array
    ) -> jnp.float32:
        return jax.lax.switch(
            tile[0],
            [
                __process_column,
                __process_ramp,
                __process_column,
            ],
            p,
            x,
            y,
            tile[1],
            tile_width,
            tile[3],
        )

    tile_dists = jax.vmap(
        lambda i: process_tile(p, i // w, i % w, tilemap[i // w, i % w])
    )(dists)

    floor_dist = p[1] - floor_height

    return jnp.minimum(floor_dist, tile_dists.min())


@partial(jax.jit, static_argnames=["sdf"])
def __raycast(
    sdf: Callable[[jax.Array], jax.Array],
    p0: jax.Array,
    dir: jax.Array,
    step_n: jnp.int32 = 50,
) -> Any:
    def f(_: Any, p: jax.Array) -> Any:
        return p + sdf(p) * dir

    return jax.lax.fori_loop(0, step_n, f, p0)


@partial(jax.jit, static_argnames=["w", "h"])
def __camera_rays(
    cam_pos: jax.Array,
    forward: jax.Array,
    # view_size: tuple[jnp.int32, jnp.int32],
    w: jnp.int32,
    h: jnp.int32,
    fx: jnp.float32 = 0.6,  # Changed type hint to float
) -> jax.Array:
    """Finds camera rays."""

    # Define a helper normalization function.
    def normalize(v: jax.Array) -> jax.Array:
        return v / jnp.linalg.norm(v, axis=-1, keepdims=True)  # type: ignore

    # Ensure the forward direction is normalized.
    forward = normalize(forward)

    # Compute the camera's right and "down" directions.
    # (The original code computed "down" via cross(right, forward).)
    right = normalize(jnp.cross(forward, WORLD_UP))
    down = normalize(jnp.cross(right, forward))

    # Build a rotation matrix from camera space to world space.
    # Rows correspond to the right, down, and forward directions.
    R = jnp.vstack([right, down, forward])  # shape (3,3)

    # Compute a corresponding vertical field-of-view parameter.
    fy = fx / w * h

    # Create a grid of pixel coordinates.
    # We let y vary from fy to -fy so that positive y in the image moves "down" in world space.

    # Use jnp.linspace instead of jp.mgrid for JIT compatibility
    x = jnp.linspace(-fx, fx, w)
    y = jnp.linspace(fy, -fy, h)
    xv, yv = jnp.meshgrid(x, y)

    x = xv.reshape(-1)
    y = yv.reshape(-1)

    # In camera space, assume the image plane is at z=1.
    # For each pixel, the unnormalized direction is (x, y, 1).
    pixel_dirs = jnp.stack([x, y, jnp.ones_like(x)], axis=-1)
    pixel_dirs = normalize(pixel_dirs)

    # Rotate the pixel directions from camera space into world space.
    ray_dir = (
        pixel_dirs @ R
    )  # shape (num_pixels, 3)  Transpose R for correct multiplication

    # (Optionally, you could also return the ray origins, which would be
    #  a copy of cam_pos for every pixel.)
    return ray_dir


@partial(jax.jit, static_argnames=["sdf"])
def __cast_shadow(
    sdf: Callable[[jax.Array], jax.Array],
    light_dir: jax.Array,
    p0: jax.Array,
    step_n: jnp.int32 = 50,
    hardness: jnp.float32 = 8.0,
) -> Any:
    def f(_: Any, carry: jnp.float32) -> Any:
        t, shadow = carry
        h = sdf(p0 + light_dir * t)
        return t + h, jnp.clip(hardness * h / t, 0.0, shadow)

    return jax.lax.fori_loop(0, step_n, f, (1e-2, 1.0))[1]


@jax.jit
def __scene_sdf_from_tilemap_color(
    tilemap: jax.Array,
    p: jax.Array,
    terrain_color: jax.Array = DEFAULT_COLOR,
    with_color: bool = False,
    floor_height: jnp.float32 = 0.0,
) -> Tuple[jnp.float32, jax.Array]:
    """SDF for the world terrain."""
    tile_dist = __scene_sdf_from_tilemap(tilemap, p, floor_height - 1)
    floor_dist = p[1] - floor_height
    min_dist = jnp.minimum(tile_dist, floor_dist)

    def process_without_color(_: Any) -> Tuple[jnp.float32, jax.Array]:
        return min_dist, jnp.zeros((3,))

    def process_with_color(_: Any) -> Tuple[jnp.float32, jax.Array]:
        x, _, z = jnp.tanh(jnp.sin(p * jnp.pi) * 20.0)
        floor_color = (0.5 + (x * z) * 0.1) * jnp.ones(3)
        color = jnp.choose(
            jnp.int32(floor_dist < tile_dist), [terrain_color, floor_color], mode="clip"
        )
        return min_dist, color

    return jax.lax.cond(with_color, process_with_color, process_without_color, None)  # type: ignore


@jax.jit
def __shade_f(
    surf_color: jax.Array,
    shadow: jax.Array,
    raw_normal: jax.Array,
    ray_dir: jax.Array,
    light_dir: jax.Array,
) -> jax.Array:
    ambient = __norm(raw_normal)
    normal = raw_normal / ambient
    diffuse = normal.dot(light_dir).clip(0.0) * shadow
    half = __normalize(light_dir - ray_dir)
    spec = 0.3 * shadow * half.dot(normal).clip(0.0) ** 200.0
    light = 0.7 * diffuse + 0.2 * ambient
    return surf_color * light + spec


@partial(jax.jit, static_argnames=["view_width", "view_height"])
def render_frame(
    cam_pos: jax.Array,
    cam_dir: jax.Array,
    tilemap: jax.Array,
    terrain_color: jax.Array = DEFAULT_COLOR,
    light_dir: jax.Array = __normalize(jnp.array([5.0, 10.0, 5.0])),
    view_width: jnp.int32 = DEFAULT_VIEWSIZE[0],
    view_height: jnp.int32 = DEFAULT_VIEWSIZE[1],
    # view_size: Tuple[jnp.int32, jnp.int32] = DEFAULT_VIEWSIZE,
) -> jax.Array:
    """Renders one frame given camera position, direction, and world terrain."""
    # Ray casting
    ray_dir = __camera_rays(cam_pos, cam_dir, view_width, view_height, fx=0.6)
    sdf = partial(__scene_sdf_from_tilemap, tilemap)
    hit_pos = jax.vmap(partial(__raycast, sdf, cam_pos))(ray_dir)

    # Shading
    raw_normal = jax.vmap(jax.grad(sdf))(hit_pos)
    shadow = jax.vmap(partial(__cast_shadow, sdf, light_dir))(hit_pos)
    color_sdf = partial(
        __scene_sdf_from_tilemap_color,
        tilemap,
        terrain_color=terrain_color,
        with_color=True,
    )
    _, surf_color = jax.vmap(color_sdf)(hit_pos)

    # Frame export
    f = partial(__shade_f, light_dir=light_dir)
    frame = jax.vmap(f)(surf_color, shadow, raw_normal, ray_dir)
    frame = frame ** (1.0 / 2.2)  # gamma correction

    return frame.reshape((view_height, view_width, NUM_CHANNELS))


transform_axes = jnp.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])


@jax.jit
def get_agent_camera_from_mjx(
    icland_state: ICLandState,
    world_width: jnp.int32,
    body_id: jnp.int32,
    camera_height: jnp.float32 = 0.2,
    camera_offset: jnp.float32 = 0.06,
) -> Tuple[jax.Array, jax.Array]:
    """Get the camera position and direction from the MuJoCo data."""
    data = icland_state.pipeline_state.mjx_data
    agent_id = icland_state.pipeline_state.component_ids[body_id, 0]

    agent_pos = jnp.array(
        [
            -data.xpos[agent_id][0] + world_width,
            data.xpos[agent_id][2],
            data.xpos[agent_id][1],
        ]
    )

    # Direct matrix multiplication using precomputed transform_axes
    yaw = data.qpos[3]
    forward_dir = jnp.array([-jnp.cos(yaw), 0.0, jnp.sin(yaw)])
    height_offset = jnp.array([0, camera_height, 0])
    camera_pos = agent_pos + height_offset + forward_dir * camera_offset

    return camera_pos, forward_dir


if __name__ == "__main__":  # pragma: no cover
    tilemap = jnp.array(
        [
            [
                [0, 0, 0, 2],
                [0, 2, 0, 2],
                [0, 2, 0, 2],
                [0, 1, 0, 2],
                [0, 0, 0, 5],
                [0, 2, 0, 5],
                [0, 1, 0, 5],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 1, 0, 3],
            ],
            [
                [0, 3, 0, 3],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 3, 0, 4],
                [0, 2, 0, 4],
                [0, 3, 0, 5],
                [0, 0, 0, 5],
                [0, 2, 0, 5],
                [0, 3, 0, 6],
                [0, 2, 0, 6],
            ],
            [
                [0, 1, 0, 3],
                [0, 0, 0, 3],
                [1, 1, 3, 4],
                [0, 0, 0, 4],
                [0, 1, 0, 4],
                [0, 0, 0, 5],
                [0, 2, 0, 5],
                [0, 1, 0, 5],
                [0, 0, 0, 6],
                [0, 1, 0, 6],
            ],
            [
                [1, 3, 3, 4],
                [0, 0, 0, 3],
                [0, 3, 0, 3],
                [0, 3, 0, 6],
                [0, 2, 0, 6],
                [0, 3, 0, 4],
                [0, 0, 0, 4],
                [1, 2, 4, 5],
                [0, 0, 0, 4],
                [0, 2, 0, 4],
            ],
            [
                [0, 1, 0, 3],
                [0, 0, 0, 3],
                [0, 3, 0, 3],
                [0, 0, 0, 6],
                [0, 1, 0, 6],
                [0, 1, 0, 4],
                [0, 0, 0, 4],
                [0, 0, 0, 4],
                [0, 0, 0, 4],
                [0, 3, 0, 4],
            ],
            [
                [1, 3, 3, 4],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 3, 0, 4],
                [0, 1, 0, 4],
                [0, 0, 0, 4],
                [0, 3, 0, 4],
                [0, 2, 0, 4],
                [0, 1, 0, 4],
            ],
            [
                [0, 1, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 3, 0, 3],
                [0, 1, 0, 4],
                [0, 0, 0, 4],
                [0, 3, 0, 4],
                [2, 1, 4, 6],
                [0, 0, 0, 6],
                [0, 2, 0, 6],
            ],
            [
                [0, 1, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [1, 1, 3, 4],
                [0, 1, 0, 4],
                [0, 0, 0, 4],
                [0, 3, 0, 4],
                [0, 1, 0, 6],
                [0, 0, 0, 6],
                [0, 3, 0, 6],
            ],
            [
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 2, 0, 3],
                [0, 1, 0, 3],
                [0, 0, 0, 4],
                [0, 2, 0, 4],
                [0, 1, 0, 4],
                [0, 0, 0, 6],
                [0, 2, 0, 6],
                [0, 1, 0, 6],
            ],
            [
                [0, 3, 0, 2],
                [0, 0, 0, 2],
                [0, 0, 0, 2],
                [0, 2, 0, 2],
                [0, 3, 0, 5],
                [0, 0, 0, 5],
                [0, 2, 0, 5],
                [0, 3, 0, 3],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
            ],
        ]
    )
    frames: List[Any] = []
    for i in range(24):
        f = render_frame(
            cam_pos=jnp.array([5.0, 10.0, -10.0 + (i * 10 / 72)]),
            cam_dir=jnp.array([0.0, -0.5, 1.0]),
            tilemap=tilemap,
            terrain_color=jnp.array([1.0, 0.0, 0.0]),
            view_width=256,
            view_height=144,
        )
        frames.append(np.array(f))
        print(f"Rendered frame {i}")

    imageio.mimsave(
        f"tests/video_output/sdf_world_scene.mp4", frames, fps=24, quality=8
    )
