# from functools import partial
# import mujoco
# from flax import struct
# import jax
# import jax.numpy as jnp
# # from types import MjxModelType, MjxStateType, MjxGeomType, MjxObjType
# from typing import Tuple, Union, Optional, NamedTuple
# from dataclasses import dataclass
# from enum import Enum
# from mujoco.mjx._src.types import Model, Data, GeomType
# from icland.renderer.jax_raycast import camera_rays, raycast, imshow
# from icland.renderer.sdfs import _scene_sdf_from_tilemap, box_sdf, ramp_sdf
# import mujoco.mjx as mjx
# import imageio

# import numpy as np

# class Tiles(struct.PyTreeNode):
#     pos: jnp.ndarray
#     size: jnp.ndarray  # e.g., shape (3,) for a cuboid
#     color: jnp.ndarray
#     rotation: jnp.ndarray  # make sure this is an array, not a Python int
#     tile_type: jnp.ndarray         # renamed from 'type' to avoid clashing with the built-in

# # @dataclass
# # class Object(struct.PyTreeNode):
# #   objType: MjxObjType
# #   id: jnp.int32          # Unique idenifier for the object
# #   position: jnp.ndarray  # Shape (3, ) for position of object in 3D space (x, y, z)
# #   rotation: jnp.ndarray  # Shape (3, 3) for rotation matrix
# #   rgba: jnp.ndarray      # Shape (4, ) for RGBA values

# @dataclass
# class Light(struct.PyTreeNode):
#   pos: jnp.ndarray          # Position of the light (x, y, z)
#   dir: jnp.ndarray          # Direction of the light (for directional lights)
#   diffuse: jnp.ndarray      # Diffuse color (RGB)
#   specular: jnp.ndarray     # Specular color (RGB)
#   ambient: jnp.ndarray      # Ambient color (RGB)
#   intensity: jnp.float64    # Light intensity (0 to 1)
#   type: jnp.int32           # Light type (e.g., point, directional, etc.)

# @dataclass
# class Camera(struct.PyTreeNode):
#   position: jnp.ndarray  # Camera position in world coordinates (x, y, z)
#   lookat: jnp.ndarray    # Camera target point (x, y, z)
#   up: jnp.ndarray        # Up vector (orientation)
#   fovy: jnp.float64      # Field of view angle in radians
#   znear: jnp.float64     # Near plane distance for clipping
#   zfar: jnp.float64      # Far plane distance for clipping
#   projection: jnp.ndarray  # Projection matrix (4x4 matrix)
#   view: jnp.ndarray       # View matrix (4x4 matrix)

# @dataclass
# class Perturb(struct.PyTreeNode):
#   selectid: jnp.int32      # ID of the selected object
#   geomid: jnp.int32        # Geometry ID for selection
#   siteid: jnp.int32        # Site ID for selection
#   bodyid: jnp.int32        # Body ID for selection
#   selectpoint: jnp.ndarray # Selection point in world coordinates (x, y, z)
#   selectnormal: jnp.ndarray # Normal of the selection point (x, y, z)
#   selectdir: jnp.ndarray   # Direction vector for the selection (for raycasting)

# @dataclass
# class Scene(struct.PyTreeNode):
#   objects: jnp.ndarray    # Array of Object instances
#   camera: Camera
#   lights: jnp.ndarray     # Array of Light instances
#   nobjects: jnp.int32
#   nlights: jnp.int32
#   perturb: Perturb

# @dataclass
# class Option(struct.PyTreeNode):
#     # Camera options
#     camera_position: jnp.ndarray  # Shape (3,), camera position in world coordinates (x, y, z)
#     camera_lookat: jnp.ndarray    # Shape (3,), camera target point (x, y, z)
#     camera_up: jnp.ndarray        # Shape (3,), up vector (orientation)
#     fovy: jnp.float64             # Field of view angle in radians
#     znear: jnp.float64            # Near plane distance for clipping
#     zfar: jnp.float64             # Far plane distance for clipping

#     # Lighting options
#     light_intensity: jnp.float64  # Light intensity (0 to 1)
#     light_position: jnp.ndarray   # Shape (3,), position of the light (x, y, z)

#     # Rendering options
#     render_mode: jnp.int32

# class Renderer(struct.PyTreeNode):  # type: ignore[no-untyped-call]
#     # Basic config
#     model: mujoco.mjx._src.types.Model
#     height: int
#     width: int
#     max_geom: int

# @jax.jit
# def init(
#     model: mujoco.mjx._src.types.Model,
#     height: int = 240,    # Prone to changes
#     width: int = 320,
#     max_geom: int = 10000,

# ) -> Renderer:
#   return Renderer(model=model, height=height, width=width, max_geom=max_geom)


# def update_scene(
#     renderer: Renderer,
#     data: Data,
#     camera: Union[int, str, Camera] = -1,
#     scene_option: Optional[Option] = None
#   ) -> Renderer:

#   model: Model = renderer.model
#   ngeom = model.ngeom

#   class TileType(Enum):
#     CUBOID = 0
#     RAMP = 1

#   poses = []
#   sizes = []
#   colors = []
#   rotations = []
#   tile_types = []
#   for i in range(ngeom):
#     if model.geom_type[i] == GeomType.MESH:
#       # Hacky way, but use the floats in geom_pos or geom_size to determine between ramp or cuboids
#       pos = model.geom_pos[i]
#       size = model.geom_size[i]
#       color = model.geom_rgba[i][:3]  # Maybe change this in the future to include alpha
#       qw, qx, qy, qz = model.geom_quat[i]

#       if round(qw, 2) == 1:
#         # N or S
#         if qy >= 0:
#           # N
#           rotation = 0
#         else:
#           # S
#           rotation = 2
#       elif round (qw, 2) == 0.71:
#         # E or W
#         if qx < 0:
#           # W
#           rotation = 1
#         else:
#           # E
#           rotation = 3
#       else:
#         # Throw errors
#         print('Invalid quaternion values')

#       poses.append(pos)
#       sizes.append(size)
#       rotations.append(rotation)
#       colors.append(color)
#       tile_types.append(TileType.CUBOID.value)

#     else:
#       # Throw error (because everything is mesh except for agent)
#       pass

#   view_w, view_h = 640, 480   # 480p
#   camera_pos = [20.0, 25.0, 20.0]
#   pos0 = jnp.array(camera_pos, dtype=jnp.float32)
#   ray_dir = camera_rays(-pos0, view_size=(view_w, view_h))

#   print(f'ngeoms: {ngeom}')
#   print(f'poses: {len(poses)}')
#   world_w, world_h  = 10, 10
#   j_poses = jnp.reshape(jnp.array(poses), (world_w, world_h, -1))
#   j_sizes = jnp.reshape(jnp.array(sizes), (world_w, world_h, -1))
#   j_colors = jnp.reshape(jnp.array(colors), (world_w, world_h, -1))
#   j_rotations = jnp.reshape(jnp.array(rotations), (world_w, world_h, -1))
#   j_tile_types = jnp.reshape(jnp.array(tile_types), (world_w, world_h, -1))
#   tiles = Tiles(j_poses, j_sizes, j_colors, j_rotations, j_tile_types)
#   sdf = partial(_scene_sdf_from_tilemap, tiles)
#   hit_pos = jax.vmap(partial(raycast, sdf, pos0))(ray_dir)

#   def save_image(a, filename='output.png'):
#     # Ensure the array is in uint8 format
#     a = np.clip(a * 255, 0, 255).astype(np.uint8)
#     imageio.imwrite(filename, a)
#     print(f"Image saved as {filename}")

#   def process_image(a):
#       # Ensure the array is processed and clipped properly
#       a = np.array(a)
#       return np.clip(a * 255, 0, 255).astype(np.uint8)

#   hit_pos = process_image(hit_pos)

#   # Ensure the shape of 'hit_pos' is appropriate for image reshaping
#   # For example, it should be a flat array or correctly structured to reshape into 3 channels
#   a = hit_pos.reshape(view_h, view_w, 3) # Ensure view_w and view_h are correct dimensions

#   # Save the image
#   save_image(a, f"output_{camera_pos}.png")

# def render(renderer: Renderer) -> Tuple[Renderer, jnp.ndarray]:
#   pass


# mj_model = mujoco.MjModel.from_xml_path("meshes3/generated_mjcf_new.xml")
# # # Write MJ model to a txt file
# # with open("output.txt", "w") as text_file:
# #     text_file.write(str(mjx.put_model(mj_model)))
# mjx_model = mjx.put_model(mj_model)
# renderer = init(mjx_model)
# update_scene(renderer, None, None, None)
