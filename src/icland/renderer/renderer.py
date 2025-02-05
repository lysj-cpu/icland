import mujoco
from flax import struct
import jax
import jax.numpy as jnp
from types import MjxModelType, MjxStateType, MjxGeomType, MjxObjType
from typing import Tuple
from dataclasses import dataclass
from enum import Enum

@dataclass
class Object(struct.PyTreeNode): 
  objType: MjxObjType
  id: jnp.int32          # Unique idenifier for the object
  position: jnp.ndarray  # Shape (3, ) for position of object in 3D space (x, y, z)
  rotation: jnp.ndarray  # Shape (3, 3) for rotation matrix
  rgba: jnp.ndarray      # Shape (4, ) for RGBA values

@dataclass
class Light(struct.PyTreeNode):
  pos: jnp.ndarray          # Position of the light (x, y, z)
  dir: jnp.ndarray          # Direction of the light (for directional lights)
  diffuse: jnp.ndarray      # Diffuse color (RGB)
  specular: jnp.ndarray     # Specular color (RGB)
  ambient: jnp.ndarray      # Ambient color (RGB)
  intensity: jnp.float64    # Light intensity (0 to 1)
  type: jnp.int32           # Light type (e.g., point, directional, etc.)

@dataclass
class Camera(struct.PyTreeNode):
  position: jnp.ndarray  # Camera position in world coordinates (x, y, z)
  lookat: jnp.ndarray    # Camera target point (x, y, z)
  up: jnp.ndarray        # Up vector (orientation)
  fovy: jnp.float64      # Field of view angle in radians
  znear: jnp.float64     # Near plane distance for clipping
  zfar: jnp.float64      # Far plane distance for clipping
  projection: jnp.ndarray  # Projection matrix (4x4 matrix)
  view: jnp.ndarray       # View matrix (4x4 matrix)

@dataclass
class Perturb(struct.PyTreeNode):
  selectid: jnp.int32      # ID of the selected object
  geomid: jnp.int32        # Geometry ID for selection
  siteid: jnp.int32        # Site ID for selection
  bodyid: jnp.int32        # Body ID for selection
  selectpoint: jnp.ndarray # Selection point in world coordinates (x, y, z)
  selectnormal: jnp.ndarray # Normal of the selection point (x, y, z)
  selectdir: jnp.ndarray   # Direction vector for the selection (for raycasting)

@dataclass
class Scene(struct.PyTreeNode):
  objects: jnp.ndarray    # Array of Object instances
  camera: Camera
  lights: jnp.ndarray     # Array of Light instances
  nobjects: jnp.int32
  nlights: jnp.int32
  perturb: Perturb

@dataclass
class Option(struct.PyTreeNode):
    # Camera options
    camera_position: jnp.ndarray  # Shape (3,), camera position in world coordinates (x, y, z)
    camera_lookat: jnp.ndarray    # Shape (3,), camera target point (x, y, z)
    camera_up: jnp.ndarray        # Shape (3,), up vector (orientation)
    fovy: jnp.float64             # Field of view angle in radians
    znear: jnp.float64            # Near plane distance for clipping
    zfar: jnp.float64             # Far plane distance for clipping
    
    # Lighting options
    light_intensity: jnp.float64  # Light intensity (0 to 1)
    light_position: jnp.ndarray   # Shape (3,), position of the light (x, y, z)
    
    # Rendering options
    render_mode: jnp.int32   
  
class Renderer(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    # Basic config
    model: MjxModelType
    height: int
    width: int
    max_geom: int

@jax.jit
def init(
    model: mujoco.mjx._src.types.Model,
    height: int = 240,    # Prone to changes
    width: int = 320,
    max_geom: int = 10000,

) -> Renderer:
    buffer_width = model.vis.global_.offwidth
    buffer_height = model.vis.global_.offheight
    if width > buffer_width:
      raise ValueError(f"""
        Image width {width} > framebuffer width {buffer_width}. Either reduce the image
        width or specify a larger offscreen framebuffer in the model XML using the
        clause:
        <visual>
          <global offwidth="my_width"/>
        </visual>""".lstrip())

    if height > buffer_height:
      raise ValueError(f"""
Image height {height} > framebuffer height {buffer_height}. Either reduce the
image height or specify a larger offscreen framebuffer in the model XML using
the clause:
<visual>
  <global offheight="my_height"/>
</visual>""".lstrip())

    model._width = width
    model._height = height
    model._model = model

    model._scene = Scene(model=model, maxgeom=max_geom)
    model._scene_option = Option()

    model._rect = _render.MjrRect(0, 0, self._width, self._height)

    # Create render contexts.
    # TODO(nimrod): Figure out why pytype doesn't like gl_context.GLContext
    self._gl_context = None  # type: ignore
    if gl_context.GLContext is not None:
      self._gl_context = gl_context.GLContext(width, height)
    if self._gl_context:
      self._gl_context.make_current()
    self._mjr_context = _render.MjrContext(
        model, _enums.mjtFontScale.mjFONTSCALE_150.value
    )
    _render.mjr_setBuffer(
        _enums.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context
    )
    self._mjr_context.readDepthMap = _enums.mjtDepthMap.mjDEPTH_ZEROFAR

    # Default render flags.
    self._depth_rendering = False
    self._segmentation_rendering = False

    @property
    def model(self):
      return self._model

    @property
    def scene(self) -> Scene:
      return self._scene

    @property
    def height(self):
      return self._height

    @property
    def width(self):
      return self._width

def update_scene(
    renderer: Renderer, 
    data: MjxStateType, 
    camera: Union[int, str, Camera] = -1,
    scene_option: Optional[_structs.MjvOption] = None
  ) -> Renderer:

  pass

def render(renderer: Renderer) -> Tuple[Renderer, jnp.ndarray]:
  
  pass

