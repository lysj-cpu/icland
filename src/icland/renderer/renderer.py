"""Renderer for the ICLand environment."""

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, Bool, Float, Int
from mujoco.mjx._src.dataclasses import PyTreeNode

from icland.constants import *
from icland.presets import DEFAULT_VIEWSIZE
from icland.renderer.sdfs import (
    beam_sdf,
    box_sdf,
    capsule_sdf,
    cube_sdf,
    ramp_sdf,
    sphere_sdf,
)
from icland.types import *


class RenderAgentInfo(PyTreeNode):  # type: ignore[misc]
    """Player info."""

    pos: jax.Array
    rot: jax.Array
    col: jax.Array


class RenderPropInfo(PyTreeNode):  # type: ignore[misc]
    """Prop info."""

    prop_type: jax.Array
    pos: jax.Array
    rot: jax.Array
    col: jax.Array


# Constants
DEFAULT_COLOR: jax.Array = jnp.array([0.2588, 0.5294, 0.9607])
DEFAULT_COLORS = jnp.array(
    [
        [0.4764706, 0.3529412, 0.27254903],
        [0.5717647, 0.42352945, 0.32705885],
        [0.6670588, 0.49411765, 0.38156864],
        [0.76235294, 0.5647059, 0.43607846],
    ]
)
# NO_PROPS = (  # Empty prop placeholder
#     RenderPropInfo(
#         prop_type=jnp.array([0]),
#         pos=jnp.empty((1, 3)),
#         rot=jnp.array([[1, 0, 0, 0]]),
#         col=jnp.empty((1, 3)),
#     ),
# )
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
    angle = -jnp.pi * ((3 * rot + 2) % 4) / 2
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
    # Agent positions and rotation
    agent_pos: jax.Array,  # shape: (agent_count, 3)
    agent_rot: jax.Array,  # shape: (agent_count, 3)
    agent_col: jax.Array,  # shape: (agent_count, 3)
    actions: jax.Array,  # shape: (agent_count, 3)
    # Prop positions and rotation
    prop_pos: jax.Array,  # shape: (n_props, 3)
    prop_rot: jax.Array,  # shape: (n_props, 4)
    prop_col: jax.Array,  # shape: (n_props, 3)
    cmap: jax.Array,
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
            jnp.matmul(transform, jnp.append(p, 1))[:3], AGENT_HEIGHT, AGENT_RADIUS
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
            # Transform from MJ coordinates to world coordinates
            R = Rotation.from_quat(qpos[:4]).as_matrix()  # 3x3 rotation matrix

            # Create the 4x4 transformation matrix
            transform = jnp.eye(4)  # Start with an identity matrix
            transform = transform.at[:3, :3].set(R)  # Set the rotation part
            transform = transform.at[:3, 3].set(
                jnp.array(curr_pos)
            )  # Set the translation part

            return transform

        # We currently support 2 prop types: the cube and the sphere
        # This follows the enums defined in prop.py
        # 0: ignore (in which case we set dist to infinity), 1: cube, 2: sphere
        # Apply the sdf based on prop type
        return jax.lax.switch(
            curr_type,
            [
                lambda _: jnp.inf,
                partial(cube_sdf, size=0.2),
                partial(sphere_sdf, r=0.1),
            ],
            jnp.matmul(
                jnp.linalg.inv(get_transformation_matrix(curr_rot, curr_pos)),
                jnp.append(p, 1),
            ),
        ), curr_col

    def process_beam_sdf(i: Int[Array, ""]) -> jax.Array:
        curr_pos = agent_pos[i]
        curr_rot = agent_rot[i]

        curr_action = actions[i]

        # Check if it's grabbing or tagging. If not, return infinity.
        is_tagging = curr_action[4] > 0.5
        is_grabbing = curr_action[5] > 0.5
        is_tagging_or_grabbing = jnp.logical_or(is_tagging, is_grabbing)
        beam_length = jnp.maximum(
            is_tagging * AGENT_MAX_TAG_DISTANCE, is_grabbing * AGENT_GRAB_RANGE
        )
        dist = jax.lax.cond(
            is_tagging_or_grabbing,
            lambda _: beam_sdf(
                p - curr_pos - jnp.array([0, -0.1, 0]), curr_rot, beam_length
            ),
            lambda _: jnp.inf,
            None,
        )

        return cast(jax.Array, dist)

    # Prop distances: Tuple[Array of floats, Array of colors]
    prop_dists = jax.vmap(process_prop_sdf)(jnp.arange(props.shape[0]))

    # Player distances
    agent_dists = jax.vmap(process_agent_sdf)(jnp.arange(agent_pos.shape[0]))

    # Agent beam distances
    beam_dists = jax.vmap(process_beam_sdf)(jnp.arange(agent_pos.shape[0]))

    # Get minimum distance and color
    min_prop_dist, min_prop_col = (
        jnp.min(prop_dists[0]),
        prop_col[jnp.argmin(prop_dists[0])],
    )
    min_agent_dist, min_agent_col = (
        jnp.min(agent_dists[0]),
        agent_col[jnp.argmin(agent_dists[0])],
    )
    min_beam_dist = jnp.min(beam_dists)
    BEAM_COLOR = jnp.array([1, 0, 0])

    # Get the absolute minimum distance and color
    candidates = jnp.array(
        [tile_dist, floor_dist, min_prop_dist, min_agent_dist, min_beam_dist]
    )
    min_dist = jnp.min(candidates)
    x, _, z = jnp.tanh(jnp.sin(p * jnp.pi) * 20.0)
    floor_color = (0.5 + (x * z) * 0.1) * jnp.ones(3)
    terrain_color = cmap[cx, cy]

    min_dist_col = jnp.array(
        [terrain_color, floor_color, min_prop_col, min_agent_col, BEAM_COLOR]
    )[jnp.argmin(candidates)]

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


def generate_colormap(
    key: jax.Array, width: int, height: int, colors: jax.Array = DEFAULT_COLORS
) -> jax.Array:
    """Generates a colormap array with random colors from a set."""  # Shape (n_colors, 3) - 3 channels (RGB)
    num_colors = colors.shape[0]
    total_elements = width * height  # For 2D part of the array

    # Generate random indices (0, 1, 2, 3) for the colors
    color_indices = jax.random.randint(key, (total_elements,), 0, num_colors)

    # Use advanced indexing to select the colors
    selected_colors = colors[color_indices]  # Shape (total_elements, 3)

    # Reshape to the desired colormap shape
    colormap = selected_colors.reshape((width, height, 3))

    return colormap


def select_random_color(
    key: jax.Array, colors: jax.Array, num_colors: int = 1
) -> jax.Array:
    """Sample the world and generate the initial parameters for the ICLand environment.

    Args:
        key: The random key for sampling.
        colors: An array of shape `(num_colors, 3)` storing all possible colors.
        num_colors: The number of randomized colors to be returned.

    Returns:
        The initial parameters for the ICLand environment.
    """
    return colors[jax.random.randint(key, (num_colors,), 0, colors.shape[0])]


@partial(jax.jit, static_argnames=["view_width", "view_height"])
def render_frame_with_objects(
    cam_pos: jax.Array,
    cam_dir: jax.Array,
    tilemap: jax.Array,
    cmap: jax.Array,
    agents: RenderAgentInfo,
    props: RenderPropInfo,
    actions: jax.Array,
    light_dir: jax.Array = __normalize(jnp.array([5.0, 10.0, 5.0])),
    view_width: int = DEFAULT_VIEWSIZE[0],
    view_height: int = DEFAULT_VIEWSIZE[1],
    camera_height: float = 0.4,
    camera_offset: float = 0.2,
) -> jax.Array:
    """Renders one frame given camera position, direction, and world terrain.

    This function is used in the top-level `render()` call by `icland.step`.

    Args:
        cam_pos: The camera's position as a JAX array of shape (3,).
        cam_dir: The camera's direction as a JAX array of shape (3,).
        tilemap: A JAX array representing the tilemap of the world.
        cmap: A JAX array representing the color map of the world.
        agents: Information about the agents to render, as a RenderAgentInfo namedtuple.
        props: Information about the props to render, as a RenderPropInfo namedtuple.
        actions: Information about the agents' actions to render (e.g. tagging), as a JAX array of shape (agent_count, ACTION_SPACE_DIM).
        light_dir: The direction of the light source, as a JAX array of shape (3,). Defaults to a normalized vector.
        view_width: The width of the rendered frame in pixels. Defaults to the first element of DEFAULT_VIEWSIZE.
        view_height: The height of the rendered frame in pixels. Defaults to the second element of DEFAULT_VIEWSIZE.
        camera_height: The height of the camera above the agent, as a float. Defaults to 0.4.
        camera_offset: The offset of the camera from the agent, as a float. Defaults to 0.2.

    Returns:
        A JAX array representing the rendered frame, with shape (view_height, view_width, NUM_CHANNELS).
    """
    agent_pos = agents.pos
    agent_rot = agents.rot
    agent_col = agents.col
    prop_pos = props.pos
    prop_rot = props.rot
    prop_col = props.col
    prop_types = props.prop_type

    cam_pos = cam_pos.at[1].subtract(AGENT_HEIGHT - camera_height)
    cam_pos = cam_pos + camera_offset * cam_dir

    # Ray casting
    ray_dir = __camera_rays(cam_pos, cam_dir, view_width, view_height, fx=0.6)
    sdf = partial(
        __scene_sdf_with_objs,
        tilemap,
        prop_types,
        agent_pos,
        agent_rot,
        agent_col,
        actions,
        prop_pos,
        prop_rot,
        prop_col,
        cmap,
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


def _get_props_info(
    mjx_data: MjxStateType,
    agent_info: ICLandAgentInfo,
    prop_info: ICLandPropInfo,
    prop_vars: ICLandPropVariables,
    max_world_width: int,
) -> RenderPropInfo:
    max_prop_count = prop_info.spawn_points.shape[0]
    max_agent_count = agent_info.spawn_points.shape[0]
    prop_count = prop_info.prop_count
    prop_indices = jnp.arange(max_prop_count)

    prop_mask = jax.vmap(lambda i: i < prop_count)(prop_indices)

    def __get_prop_data(prop_id: jax.Array) -> RenderPropInfo:
        body_id = prop_info.body_ids[prop_id].astype(int)
        prop_dof = PROP_DOF_MULTIPLIER + 1
        prop_pos_index = AGENT_DOF_OFFSET * max_agent_count + prop_dof * prop_id
        prop_pos = jnp.array(
            [
                -mjx_data.qpos[prop_pos_index] + max_world_width,
                mjx_data.qpos[prop_pos_index + 2],
                mjx_data.qpos[prop_pos_index + 1],
            ]
        )

        # Get rotation from the MJX data
        prop_quat = jax.lax.dynamic_slice_in_dim(
            mjx_data.qpos,
            AGENT_DOF_OFFSET * max_agent_count
            + prop_id * prop_dof
            + PROP_DOF_OFFSET
            - 1,
            PROP_DOF_OFFSET,
        )
        # Transform quat to fit renderer coordinate system
        # (w, x, y, z) --> (w, -x, z, y)
        prop_quat = jnp.array([prop_quat[0], -prop_quat[1], prop_quat[3], prop_quat[2]])
        prop_type = prop_info.prop_types[prop_id]
        colour = prop_info.colour[prop_id] * prop_mask[prop_id]

        return RenderPropInfo(
            prop_type=prop_type, pos=prop_pos, rot=prop_quat, col=colour
        )

    return jax.vmap(__get_prop_data)(prop_indices)


def _get_agents_info(
    mjx_data: MjxStateType,
    agent_info: ICLandAgentInfo,
    agent_vars: ICLandAgentVariables,
    max_world_width: int,
) -> RenderAgentInfo:
    max_agent_count = agent_info.spawn_points.shape[0]
    agent_count = agent_info.agent_count
    agent_indices = jnp.arange(max_agent_count)

    agent_mask = jax.vmap(lambda i: i < agent_count)(agent_indices)

    def __get_agent_data(
        agent_id: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        body_id = agent_info.body_ids[agent_id].astype(int)
        pitch = agent_vars.pitch[agent_id]  # TODO: Change multiplier
        dof_address = agent_info.dof_addresses[agent_id].astype(int)

        agent_pos = jnp.array(
            [
                -mjx_data.xpos[body_id][0] + max_world_width,
                mjx_data.xpos[body_id][2] + AGENT_HEIGHT / 2,
                mjx_data.xpos[body_id][1],
            ]
        )

        # Get yaw from the MJX data
        # yaw = mjx_data.qpos[dof_address + body_id * 4 + 3]
        yaw = mjx_data.qpos[dof_address + 3]  # Get angle from dof address

        # Compute forward direction using both yaw and pitch.
        # When pitch=0, this reduces to [-cos(yaw), 0, sin(yaw)] as before.
        forward_dir = (
            jnp.array(
                [
                    -jnp.cos(pitch) * jnp.cos(yaw),
                    jnp.sin(pitch),
                    jnp.cos(pitch) * jnp.sin(yaw),
                ]
            )
            * agent_mask[agent_id]
        )

        agent_pos = agent_pos * agent_mask[agent_id]
        colour = agent_info.colour[agent_id] * agent_mask[agent_id]

        return RenderAgentInfo(pos=agent_pos, rot=forward_dir, col=colour)

    return cast(RenderAgentInfo, jax.vmap(__get_agent_data)(agent_indices))


@partial(jax.jit, static_argnames=["view_width", "view_height"])
def render(
    agent_info: ICLandAgentInfo,
    agent_vars: ICLandAgentVariables,
    prop_info: ICLandPropInfo,
    prop_vars: ICLandPropVariables,
    agent_actions: jax.Array,
    world: ICLandWorld,
    mjx_data: MjxStateType,
    view_width: int = DEFAULT_VIEWSIZE[0],
    view_height: int = DEFAULT_VIEWSIZE[1],
) -> jax.Array:  # pragma: no cover
    """Top-level render function.

    Called by `icland.step` once every `FPS / Physics steps per second` physical steps.

    Args:
        agent_info: Information about the agents to render, as an `ICLandAgentInfo` namedtuple.
        agent_vars: Variables associated with the agents, as an `ICLandAgentVariables` namedtuple.
        prop_info: Information about the props to render, as an `ICLandPropInfo` namedtuple.
        prop_vars: Variables associated with the props, as an `ICLandPropVariables` namedtuple.
        agent_actions: Information about the agents' actions to render, such as tagging and grabbing.
        world: The ICLand world information, as an `ICLandWorld` namedtuple.
        mjx_data: The current state of the MuJoCo simulation, as an `MjxStateType`.
        view_width: The width of the rendered frame in pixels. Defaults to the first element of `DEFAULT_VIEWSIZE`.
        view_height: The height of the rendered frame in pixels. Defaults to the second element of `DEFAULT_VIEWSIZE`.
        top_down:

    Returns:
        A JAX array representing the rendered frames, with shape (num_agents, view_height, view_width, NUM_CHANNELS).
    """
    render_agent_info = _get_agents_info(
        mjx_data, agent_info, agent_vars, world.max_world_width
    )
    render_prop_info = _get_props_info(
        mjx_data, agent_info, prop_info, prop_vars, world.max_world_width
    )
    frames = jax.vmap(
        partial(
            render_frame_with_objects,
            tilemap=world.tilemap,
            cmap=world.cmap,
            agents=render_agent_info,
            props=render_prop_info,
            actions=agent_actions,
            view_width=view_width,
            view_height=view_height,
        ),
        in_axes=(0, 0),
    )(render_agent_info.pos, render_agent_info.rot)

    return frames


@partial(jax.jit, static_argnames=["view_width", "view_height"])
def render_top_down(
    agent_info: ICLandAgentInfo,
    agent_vars: ICLandAgentVariables,
    prop_info: ICLandPropInfo,
    prop_vars: ICLandPropVariables,
    agent_actions: jax.Array,
    world: ICLandWorld,
    mjx_data: MjxStateType,
    view_width: int = DEFAULT_VIEWSIZE[0],
    view_height: int = DEFAULT_VIEWSIZE[1],
) -> jax.Array:  # pragma: no cover
    """Render one "top-down" frame for monitoring.

    Args:
        agent_info: Information about the agents to render, as an `ICLandAgentInfo` namedtuple.
        agent_vars: Variables associated with the agents, as an `ICLandAgentVariables` namedtuple.
        prop_info: Information about the props to render, as an `ICLandPropInfo` namedtuple.
        prop_vars: Variables associated with the props, as an `ICLandPropVariables` namedtuple.
        agent_actions: Information about the agents' actions to render, such as tagging and grabbing.
        world: The ICLand world information, as an `ICLandWorld` namedtuple.
        mjx_data: The current state of the MuJoCo simulation, as an `MjxStateType`.
        view_width: The width of the rendered frame in pixels. Defaults to the first element of `DEFAULT_VIEWSIZE`.
        view_height: The height of the rendered frame in pixels. Defaults to the second element of `DEFAULT_VIEWSIZE`.

    Returns:
        A JAX array representing the rendered frame, with shape (view_height, view_width, NUM_CHANNELS).
    """
    render_agent_info = _get_agents_info(
        mjx_data, agent_info, agent_vars, world.max_world_width
    )
    render_prop_info = _get_props_info(
        mjx_data, agent_info, prop_info, prop_vars, world.max_world_width
    )
    cam_pos = jnp.array(
        [
            world.tilemap.shape[0] / 2,
            2 * (jnp.maximum(world.tilemap.shape[0], world.tilemap.shape[1])),
            world.tilemap.shape[1] / 2,
        ]
    )
    cam_dir = jnp.array([0.0, -1.0, 0.01])
    frame = render_frame_with_objects(
        cam_pos,
        cam_dir,
        tilemap=world.tilemap,
        cmap=world.cmap,
        agents=render_agent_info,
        props=render_prop_info,
        actions=agent_actions,
        view_width=view_width,
        view_height=view_height,
    )

    return frame
