"""Renderer for the ICLand environment."""

from collections.abc import Callable
from functools import partial
from typing import Any

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, Bool, Float, Int
from mujoco.mjx._src.dataclasses import PyTreeNode

from icland.renderer.sdfs import box_sdf, capsule_sdf, cube_sdf, ramp_sdf, sphere_sdf
from icland.types import ICLandState

# Constants
DEFAULT_VIEWSIZE: tuple[int, int] = (92, 76)
DEFAULT_COLOR: jax.Array = jnp.array([0.2588, 0.5294, 0.9607])
WORLD_UP: jax.Array = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
NUM_CHANNELS: int = 3


@partial(jax.jit, static_argnames=["axis", "keepdims"])
def __norm(
    v: jax.Array,
    axis: int = -1,
    keepdims: bool = False,
    eps: float = 0.0,
) -> jax.Array:
    return jnp.sqrt((v * v).sum(axis, keepdims=keepdims).clip(min=eps))


@jax.jit
def __normalize(v: jax.Array, axis: int = -1, eps: float = 1e-20) -> jax.Array:
    return v / __norm(v, axis, keepdims=True, eps=eps)  # type: ignore


@jax.jit
def __process_column(
    p: jax.Array,
    x: Int[Array, ""],
    y: Int[Array, ""],
    rot: Int[Array, ""],
    w: Int[Array, ""],
    h: Int[Array, ""],
) -> Float[Array, ""]:
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
    x: Int[Array, ""],
    y: Int[Array, ""],
    rot: Int[Array, ""],
    h: Int[Array, ""],
    w: Int[Array, ""],
) -> Float[Array, ""]:
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
def scene_sdf_from_tilemap(  # pragma: no cover
    tilemap: jax.Array, p: jax.Array, floor_height: float = 0.0
) -> tuple[jax.Array, Int[Array, ""], Int[Array, ""]]:
    """Generates the signed distance function from the terrain."""
    w, h = tilemap.shape[0], tilemap.shape[1]
    dists = jnp.arange(w * h, dtype=jnp.int32)
    tile_width = 1

    def process_tile(
        p: jax.Array, x: Int[Array, ""], y: Int[Array, ""], tile: jax.Array
    ) -> Float[Array, ""]:
        return jax.lax.switch(
            tile[0],
            [
                __process_column,
                __process_ramp,
                __process_column,
            ],
            p,
            w - x - 1,
            y,
            tile[1],
            tile_width,
            tile[3],
        )

    tile_dists = jax.vmap(
        lambda i: process_tile(p, i // w, i % w, tilemap[i // w, i % w])
    )(dists)
    min_dist_index = jnp.argmin(tile_dists)

    floor_dist = p[1] - floor_height

    return (
        jnp.minimum(floor_dist, tile_dists.min()),
        min_dist_index // w,
        min_dist_index % w,
    )


@partial(jax.jit, static_argnames=["sdf"])
def __raycast(
    sdf: Callable[[jax.Array], jax.Array],
    p0: jax.Array,
    rdir: jax.Array,
    step_n: int = 50,
) -> Any:  # typing: ignore
    def f(_: Int[Array, ""], p: jax.Array) -> Any:  # typing: ignore
        res = p + sdf(p) * rdir
        return res

    return jax.lax.fori_loop(0, step_n, f, p0)


@partial(jax.jit, static_argnames=["w", "h"])
def __camera_rays(
    cam_pos: jax.Array,
    forward: jax.Array,
    # view_size: tuple[jnp.int32, jnp.int32],
    w: int,
    h: int,
    fx: float = 0.6,  # Changed type hint to float
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
    step_n: int = 50,
    hardness: float = 8.0,
) -> Any:
    def f(_: Any, carry: tuple[jax.Array, jax.Array]) -> Any:
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
    floor_height: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """SDF for the world terrain."""
    tile_dist, _, _ = scene_sdf_from_tilemap(tilemap, p, floor_height - 1)
    floor_dist = p[1] - floor_height
    min_dist = jnp.minimum(tile_dist, floor_dist)

    def process_without_color(_: Any) -> tuple[Float[Array, ""], jax.Array]:
        return min_dist, jnp.zeros((3,))

    def process_with_color(_: Any) -> tuple[Float[Array, ""], jax.Array]:
        x, _, z = jnp.tanh(jnp.sin(p * jnp.pi) * 20.0)
        floor_color = (0.5 + (x * z) * 0.1) * jnp.ones(3)
        color = jnp.choose(
            jnp.int32(floor_dist < tile_dist), [terrain_color, floor_color], mode="clip"
        )
        return min_dist, color

    return jax.lax.cond(with_color, process_with_color, process_without_color, None)  # type: ignore


def __scene_sdf_with_objs(
    # Scene
    tilemap: jax.Array,
    # Props (list of ints to represent which prop it is)
    props: jax.Array,  # shape: (n_props, )
    # Players positions and rotation
    # TODO: Change to adapt with mjx data
    agent_pos: jax.Array,  # shape: (agent_count, 3)
    agent_col: jax.Array,  # shape: (agent_count, 3)
    # Prop positions and rotation
    prop_pos: jax.Array,  # shape: (n_props, 3)
    prop_rot: jax.Array,  # shape: (n_props, 4)
    prop_col: jax.Array,  # shape: (n_props, 3)
    terrain_cmap: jax.Array,
    # Ray point
    p: jax.Array,
    # Extra kwargs
    floor_height: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """SDF for the agents and props."""
    # Pre: the lengths of agent_pos and agent_col are the same.
    # Pre: the lengths of prop_pos, prop_rot and prop_col are the same.

    # Add distances computed by SDFs here
    tile_dist, cx, cy = scene_sdf_from_tilemap(tilemap, p, floor_height - 1)
    floor_dist = p[1] - floor_height

    def process_agent_sdf(i: Int[Array, ""]) -> tuple[Float[Array, ""], jax.Array]:
        curr_pos = agent_pos[i]
        curr_col = agent_col[i]

        transform = jnp.array(
            [
                [1, 0, 0, -curr_pos[0]],
                [0, 1, 0, -curr_pos[1] + 0.6],
                [0, 0, 1, -curr_pos[2]],
                [0, 0, 0, 1],
            ]
        )

        return capsule_sdf(
            jnp.matmul(transform, jnp.append(p, 1))[:3], 0.4, 0.06
        ), curr_col

    def process_prop_sdf(i: Int[Array, ""]) -> tuple[jax.Array, jax.Array]:
        curr_pos = prop_pos[i]
        curr_rot = prop_rot[i]
        curr_col = prop_col[i]

        curr_type = props[i]

        def get_transformation_matrix(
            qpos: jax.Array, curr_pos: jax.Array
        ) -> jax.Array:
            # Extract rotation matrix from quaternion
            # TODO: Transform from MJ coordinates to world coordinates
            R = Rotation.from_quat(qpos[:4]).as_matrix()  # 3x3 rotation matrix

            # Create the 4x4 transformation matrix
            transform = jnp.eye(4)  # Start with an identity matrix
            transform = transform.at[:3, :3].set(R)  # Set the rotation part
            transform = transform.at[:3, 3].set(
                jnp.array(curr_pos) + jnp.array([0, 0.25, 0])
            )  # Set the translation part

            return transform

        # We currently support 2 prop types: the cube and the sphere
        # 0: ignore (in which case we set dist to infinity), 1: cube, 2: sphere
        # Apply the sdf based on prop type
        return jax.lax.switch(
            curr_type,
            [
                lambda _: jnp.inf,
                partial(cube_sdf, size=0.5),
                partial(sphere_sdf, r=0.25),
            ],
            jnp.matmul(
                jnp.linalg.inv(get_transformation_matrix(curr_rot, curr_pos)),
                jnp.append(p, 1),
            ),
        ), curr_col

    # Prop distances: Tuple[Array of floats, Array of colors]
    prop_dists = jax.vmap(process_prop_sdf)(jnp.arange(props.shape[0]))

    # Player distances
    agent_dists = jax.vmap(process_agent_sdf)(jnp.arange(agent_pos.shape[0]))

    # Get minimum distance and color
    min_prop_dist, min_prop_col = (
        jnp.min(prop_dists[0]),
        prop_col[jnp.argmin(prop_dists[0])],
    )
    min_agent_dist, min_agent_col = (
        jnp.min(agent_dists[0]),
        agent_col[jnp.argmin(agent_dists[0])],
    )

    # Get the absolute minimum distance and color
    candidates = jnp.array([tile_dist, floor_dist, min_prop_dist, min_agent_dist])
    min_dist = jnp.min(candidates)

    x, _, z = jnp.tanh(jnp.sin(p * jnp.pi) * 20.0)
    floor_color = (0.5 + (x * z) * 0.1) * jnp.ones(3)
    terrain_color = terrain_cmap[cx, cy]

    min_dist_col = jnp.array([terrain_color, floor_color, min_prop_col, min_agent_col])[
        jnp.argmin(candidates)
    ]

    return min_dist, min_dist_col


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
    return surf_color * light + spec  # type: ignore


def can_see_object(
    agent_pos: jax.Array,
    agent_dir: jax.Array,
    obj_pos: jax.Array,
    obj_sdf: Callable[[Any], Any],
    terrain_sdf: Callable[[Any], Any],
    eps: float = 1e-03,
    step_n: int = 100,
) -> Bool[Array, ""]:
    """Determines whether the specified player can see the object."""
    # All the positions and directions are in world coords.

    # Find ray from player direction towards object.
    ray_length = jnp.linalg.norm(obj_pos - agent_pos)
    direction = agent_dir / jnp.linalg.norm(agent_dir)

    state_init = (jnp.array(0.0), jnp.array(0), jnp.array(0))

    def cond_fn(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        t, flag, step = state
        return jnp.logical_and(
            t < ray_length + eps, jnp.logical_and(flag == 0, step < step_n)
        )

    def body_fn(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        t, flag, step = state
        pos = agent_pos + t * direction

        d_obj = obj_sdf(pos - obj_pos)  # Relative to obj pos
        d_ter = terrain_sdf(pos)

        flag = jax.lax.select(d_obj < eps, 1, flag)
        flag = jax.lax.select(d_ter < eps, -1, flag)

        # Determine the next step: advance by the smallest safe distance.
        step_size = jnp.minimum(d_obj, d_ter)
        t_new = t + step_size
        return t_new, flag, step + 1

    t_f, flag_f, step_f = jax.lax.while_loop(cond_fn, body_fn, state_init)

    visible = jnp.where(flag_f == 1, True, False)
    visible = jnp.where((flag_f == 0) & (t_f >= ray_length), True, visible)
    return visible


def generate_colormap(key: jax.Array, width: int, height: int) -> jax.Array:
    """Generates a colormap array with random colors from a set."""
    # rgb = 243/255, 180/255, 139/255
    colors = jnp.array(
        [
            [0.5, 0.5, 0.5],  # Dark gray
            [0.6, 0.6, 0.6],  # Lighter gray
            [0.7, 0.7, 0.7],  # Even lighter gray
            [0.8, 0.8, 0.8],  # Light gray
        ]
    )  # Shape (4, 3) - 4 colors, 3 channels (RGB)

    num_colors = colors.shape[0]
    total_elements = width * height  # For 2D part of the array

    # Generate random indices (0, 1, 2, 3) for the colors
    color_indices = jax.random.randint(key, (total_elements,), 0, num_colors)

    # Use advanced indexing to select the colors
    selected_colors = colors[color_indices]  # Shape (total_elements, 3)

    # Reshape to the desired colormap shape
    colormap = selected_colors.reshape((width, height, 3))

    return colormap


class PlayerInfo(PyTreeNode):  # type: ignore[misc]
    """Player info."""

    pos: jax.Array
    col: jax.Array


class PropInfo(PyTreeNode):  # type: ignore[misc]
    """Prop info."""

    prop_type: jax.Array
    pos: jax.Array
    rot: jax.Array
    col: jax.Array


@partial(jax.jit, static_argnames=["view_width", "view_height"])
def render_frame_with_objects(
    cam_pos: jax.Array,
    cam_dir: jax.Array,
    tilemap: jax.Array,
    terrain_cmap: jax.Array,
    players: PlayerInfo,
    props: PropInfo = PropInfo(
        prop_type=jnp.array([0]),
        pos=jnp.empty((1, 3)),
        rot=jnp.array([[1, 0, 0, 0]]),
        col=jnp.empty((1, 3)),
    ),
    light_dir: jax.Array = __normalize(jnp.array([5.0, 10.0, 5.0])),
    view_width: int = DEFAULT_VIEWSIZE[0],
    view_height: int = DEFAULT_VIEWSIZE[1],
    camera_height: float = 0.4,
    camera_offset: float = 0.2,
) -> jax.Array:
    """Renders one frame given camera position, direction, and world terrain."""
    agent_pos = players.pos
    agent_col = players.col
    prop_pos = props.pos
    prop_rot = props.rot
    prop_col = props.col
    prop_types = props.prop_type

    cam_pos = cam_pos.at[1].subtract(0.4 - camera_height)
    cam_pos = cam_pos + camera_offset * cam_dir

    # Ray casting
    ray_dir = __camera_rays(cam_pos, cam_dir, view_width, view_height, fx=0.6)
    sdf = partial(
        __scene_sdf_with_objs,
        tilemap,
        prop_types,
        agent_pos,
        agent_col,
        prop_pos,
        prop_rot,
        prop_col,
        terrain_cmap,
    )
    sdf_dists_only = lambda p: sdf(p)[0]
    hit_pos = jax.vmap(partial(__raycast, sdf_dists_only, cam_pos))(ray_dir)

    # Shading
    raw_normal = jax.vmap(jax.grad(sdf_dists_only))(hit_pos)
    shadow = jax.vmap(partial(__cast_shadow, sdf_dists_only, light_dir))(hit_pos)
    _, surf_color = jax.vmap(sdf)(hit_pos)

    # Frame export
    f = partial(__shade_f, light_dir=light_dir)
    frame = jax.vmap(f)(surf_color, shadow, raw_normal, ray_dir)
    frame = frame ** (1.0 / 2.2)  # gamma correction

    return frame.reshape((view_height, view_width, NUM_CHANNELS))


@partial(jax.jit, static_argnames=["view_width", "view_height"])
def render_frame(
    cam_pos: jax.Array,
    cam_dir: jax.Array,
    tilemap: jax.Array,
    terrain_color: jax.Array = DEFAULT_COLOR,
    light_dir: jax.Array = __normalize(jnp.array([5.0, 10.0, 5.0])),
    view_width: int = DEFAULT_VIEWSIZE[0],
    view_height: int = DEFAULT_VIEWSIZE[1],
    # view_size: tuple[jnp.int32, jnp.int32] = DEFAULT_VIEWSIZE,
) -> jax.Array:
    """Renders one frame given camera position, direction, and world terrain."""
    # Ray casting
    ray_dir = __camera_rays(cam_pos, cam_dir, view_width, view_height, fx=0.6)
    sdf = partial(scene_sdf_from_tilemap, tilemap)
    sdf_dist_only = lambda p: sdf(p)[0]
    hit_pos = jax.vmap(partial(__raycast, sdf_dist_only, cam_pos))(ray_dir)

    # Shading
    raw_normal = jax.vmap(jax.grad(sdf_dist_only))(hit_pos)
    shadow = jax.vmap(partial(__cast_shadow, sdf_dist_only, light_dir))(hit_pos)
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


@jax.jit
def get_agent_camera_from_mjx(
    icland_state: ICLandState,
    world_width: int,
    body_id: int,
) -> tuple[jax.Array, jax.Array]:
    """Get the camera position and direction from the MuJoCo data."""
    data = icland_state.pipeline_state.mjx_data
    agent_id = icland_state.pipeline_state.component_ids[body_id, 0].astype(int)
    pitch = icland_state.pipeline_state.component_ids[body_id, 3] * 0.5
    dof_address = icland_state.pipeline_state.component_ids[body_id, 2].astype(int)

    agent_pos = jnp.array(
        [
            -data.xpos[agent_id][0] + world_width,
            data.xpos[agent_id][2],
            data.xpos[agent_id][1],
        ]
    )

    # Get yaw from the MuJoCo data
    yaw = data.qpos[dof_address + body_id * 4 + 3]  # Get angle from dof address

    # Compute forward direction using both yaw and pitch.
    # When pitch=0, this reduces to [-cos(yaw), 0, sin(yaw)] as before.
    forward_dir = jnp.array(
        [-jnp.cos(pitch) * jnp.cos(yaw), jnp.sin(pitch), jnp.cos(pitch) * jnp.sin(yaw)]
    )

    camera_pos = agent_pos

    return camera_pos, forward_dir


if __name__ == "__main__":  # pragma: no cover
    tilemap2 = jnp.array(
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
    tilemap = jnp.array(
        [
            [
                [0, 0, 0, 6],
                [0, 2, 0, 6],
                [0, 1, 0, 6],
                [0, 1, 0, 4],
                [0, 3, 0, 4],
                [0, 3, 0, 6],
                [0, 2, 0, 6],
                [0, 0, 0, 2],
                [0, 2, 0, 2],
                [0, 1, 0, 2],
            ],
            [
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 0, 0, 4],
                [0, 1, 0, 4],
                [0, 0, 0, 6],
                [0, 1, 0, 6],
                [0, 3, 0, 5],
                [0, 2, 0, 5],
                [0, 3, 0, 3],
            ],
            [
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 3, 0, 5],
                [0, 0, 0, 5],
                [0, 2, 0, 5],
                [0, 1, 0, 5],
                [0, 3, 0, 5],
                [0, 1, 0, 3],
            ],
            [
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 3, 0, 3],
                [0, 0, 0, 5],
                [0, 2, 0, 5],
                [0, 1, 0, 5],
                [0, 1, 0, 5],
                [0, 3, 0, 5],
                [0, 1, 0, 3],
            ],
            [
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 3, 0, 3],
                [0, 3, 0, 6],
                [0, 0, 0, 6],
                [0, 2, 0, 6],
                [0, 0, 0, 5],
                [0, 1, 0, 5],
                [0, 1, 0, 3],
            ],
            [
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 3, 0, 3],
                [0, 0, 0, 6],
                [0, 2, 0, 6],
                [0, 1, 0, 6],
                [0, 3, 0, 4],
                [0, 2, 0, 4],
                [1, 3, 3, 4],
            ],
            [
                [0, 2, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 0, 0, 4],
                [0, 1, 0, 4],
                [0, 0, 0, 3],
            ],
            [
                [0, 1, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 3, 0, 6],
                [0, 2, 0, 6],
            ],
            [
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [0, 2, 0, 3],
                [0, 2, 0, 3],
                [0, 2, 0, 3],
                [0, 2, 0, 3],
                [0, 0, 0, 3],
                [0, 3, 0, 3],
                [0, 0, 0, 6],
                [0, 1, 0, 6],
            ],
            [
                [0, 3, 0, 6],
                [0, 0, 0, 6],
                [0, 2, 0, 6],
                [0, 3, 0, 4],
                [0, 2, 0, 4],
                [0, 0, 0, 3],
                [0, 2, 0, 3],
                [2, 3, 2, 3],
                [0, 0, 0, 2],
                [0, 2, 0, 2],
            ],
        ]
    )
    frames: list[Any] = []
    cmap = generate_colormap(jax.random.PRNGKey(0), 10, 10)
    print(cmap.shape)

    players = PlayerInfo(
        pos=jnp.array([[8.5, 3, 1]]),
        col=jnp.array([[1, 0, 0]]),
    )
    props = PropInfo(
        prop_type=jnp.array([1]),
        pos=jnp.array([[4, 3, 1]]),
        rot=jnp.array([[1, 0, 0, 0]]),
        col=jnp.array([[1.0, 1.0, 0.0]]),
    )

    for i in range(72):
        # f = render_frame(
        # cam_pos=jnp.array([5.0, 10.0, -10.0 + (i * 10 / 72)]),
        # cam_dir=jnp.array([0.0, -0.5, 1.0]),
        # tilemap=tilemap,
        # terrain_color=jnp.array([1.0, 0.0, 0.0]),
        # view_width=256,
        # view_height=144,
        # )
        f = render_frame_with_objects(
            cam_pos=jnp.array([5.0, 10.0, -10.0 + (i * 10 / 72)]),
            cam_dir=jnp.array([0.0, -0.5, 1.0]),
            tilemap=tilemap,
            terrain_cmap=cmap,
            players=players,
            props=props,
            view_width=96,
            view_height=72,
        )
        frames.append(np.array(f))
        print(f"Rendered frame {i}")

    imageio.mimsave(
        f"tests/video_output/sdf_world_scene.mp4",
        frames,
        fps=24,
        quality=8,
    )
