from icland import *
from typing import Iterable, NamedTuple, Optional
import re

import jax
from jax import numpy as jnp
import numpy as onp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image

import renderer
from renderer.types import Vec3f, Vec2f
from renderer import CameraParameters as Camera
from renderer import LightParameters as Light
from renderer import Model as RendererMesh
from renderer import ModelObject as Instance
from renderer import ShadowParameters as Shadow
from renderer import Renderer, merge_objects, transpose_for_display
from renderer.geometry import rotation_matrix

TEST_XML_STRING = """
<mujoco model="depth_render_test">
  <asset>
    <!-- Define a simple mesh -->
    <mesh name="test_mesh" file="./src/render/cube.obj"/>
  </asset>

  <worldbody>
    <light name="main_light" pos="0 0 1" dir="0 0 -1"
           diffuse="1 1 1" specular="0.1 0.1 0.1"/>

    <body name="agent1" pos="0 0 1">
      <joint type="slide" axis="1 0 0" />
      <joint type="slide" axis="0 1 0" />
      <joint type="slide" axis="0 0 1" />
      <joint type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="agent1_geom"
        type="capsule"
        size="0.06"
        fromto="0 0 0 0 0 -0.4"
        solimp="0.9 0.995 0.001 1 1000"
        friction="0.001 0.001 0.0001"
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        solimp="0.9 0.995 0.001 1 1000"
        friction="0.001 0.001 0.0001"
        mass="0.001"
      />
    </body>

    <!-- Ground plane -->
    <geom name="ground" type="plane" size="5 5 0.01" rgba="1 1 1 1"/>

    <!-- Static mesh object for depth rendering -->
    <body name="mesh_object" pos="1 0 0.5">
      <geom name="mesh_geom" type="mesh" mesh="test_mesh" rgba="0.6 0.6 0.6 1"/>
    </body>

    <!-- Additional box to add depth variation -->
    <geom type="box" size="1 1 1" pos="2 0 -0.5" euler="0 30 0" rgba="1 0.8 0.8 1" />

    <!-- Depth Camera -->
    <camera name="depth_camera" pos="2 3 1" xyaxes="1 0 0 0 1 0" fovy="45"/>
  </worldbody>
</mujoco>
"""

canvas_width: int = 1920 #@param {type:"integer"}
canvas_height: int = 1080 #@param {type:"integer"}
frames: int = 30 #@param {type:"slider", min:1, max:32, step:1}
rotation_axis = "Y" #@param ["X", "Y", "Z"]
rotation_degrees: float = 360.

rotation_axis = dict(
    X=(1., 0., 0.),
    Y=(0., 1., 0.),
    Z=(0., 0., 1.),
)[rotation_axis]

degrees = jax.lax.iota(float, frames) * rotation_degrees / frames

def compute_normals(vertices, faces):
  normals = jnp.zeros_like(vertices)
  
  for f in faces:
    v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
    normal = jnp.cross(v1 - v0, v2 - v0)
    normal = normal / jnp.linalg.norm(normal)
    normals = normals.at[f].add(normal)
  
  normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
  return normals

def extract_mesh(model: MjxModelType, mesh_id):
    vertices = jnp.array(model.mesh_vert[mesh_id].reshape(-1, 3), dtype=jnp.float32)
    faces = jnp.array(model.mesh_face[mesh_id].reshape(-1, 3), dtype=jnp.int32)
    norms = compute_normals(vertices, faces)
    uvs = jnp.zeros((vertices.shape[0], 2), dtype=jnp.float32)
    faces_norm = faces
    faces_uv = faces
    return vertices, faces, norms, uvs, faces_norm, faces_uv

def make_model(fileContent: list[str], diffuse_map, specular_map) -> RendererMesh:
  verts: list[Vec3f] = []
  norms: list[Vec3f] = []
  uv: list[Vec2f] = []
  faces: list[list[int]] = []
  faces_norm: list[list[int]] = []
  faces_uv: list[list[int]] = []

  _float = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)")
  _integer = re.compile(r"\d+")
  _one_vertex = re.compile(r"\d+/\d*/\d*")

  for line in fileContent:
    if line.startswith("v "):
      vert: Vec3f = tuple(map(float, _float.findall(line, 2)[:3]))
      verts.append(vert)
    elif line.startswith("vn "):
      norm: Vec3f = tuple(map(float, _float.findall(line, 2)[:3]))
      norms.append(norm)
    elif line.startswith("vt "):
      uv_coord: Vec2f = tuple(map(float, _float.findall(line, 2)[:2]))
      uv.append(uv_coord)
    elif line.startswith("f "):
      face: list[int] = []
      face_norm: list[int] = []
      face_uv: list[int] = []

      vertices: list[str] = _one_vertex.findall(line)
      assert len(vertices) == 3, ("Expected 3 vertices, "
                                  f"(got {len(vertices)}")
      for vertex in _one_vertex.findall(line):
          indices: list[int] = list(map(int, _integer.findall(vertex)))
          assert len(indices) == 3, ("Expected 3 indices (v/vt/vn), "
                                      f"got {len(indices)}")
          v, vt, vn = indices
          # indexed from 1 in Wavefront Obj
          face.append(v - 1)
          face_norm.append(vn - 1)
          face_uv.append(vt - 1)
        
      faces.append(face)
      faces_norm.append(face_norm)
      faces_uv.append(face_uv)

  return RendererMesh(
    verts=jnp.array(verts),
    norms=jnp.array(norms),
    uvs=jnp.array(uv),
    faces=jnp.array(faces),
    faces_norm=jnp.array(faces_norm),
    faces_uv=jnp.array(faces_uv),
    diffuse_map=None if not diffuse_map else jax.numpy.swapaxes(diffuse_map, 0, 1)[:, ::-1, :],
    specular_map=None if not specular_map else jax.numpy.swapaxes(specular_map, 0, 1)[:, ::-1],
  )



@jax.default_matmul_precision("float32")
def render_instances(
  instsances: list[Instance],
  width: int,
  height: int,
  camera: Camera,
  light: Optional[Light] = None,
  shadow: Optional[Shadow] = None,
  camera_target: Optional[jnp.ndarray] = None,
  enable_shadow: bool = True,
) -> jnp.ndarray:
  if light is None:
    direction = jnp.array([0.57735, -0.57735, 0.57735])
    light = Light(
        direction=direction,
        ambient=0.1,
        diffuse=0.85,
        specular=0.05,
    )
  if shadow is None and enable_shadow:
    assert camera_target is not None, 'camera_target is None'
    shadow = Shadow(centre=camera_target)
  elif not enable_shadow:
    shadow = None

  img = Renderer.get_camera_image(
    objects=instances,
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow,
    colour_default=jnp.zeros(3, dtype=jnp.single),
  )
  arr = jax.lax.clamp(0., img, 1.)

  return arr


def gen_compile_render(instances):
  eye = jnp.array((0, 0, 3.))
  center = jnp.array((0, 0, 0))
  up = jnp.array((0, 1, 0))
  camera: Camera = Camera(viewWidth=canvas_width, viewHeight=canvas_height, position=eye, target=center, up=up)

  def render(batched_instances) -> jnp.ndarray:
    def _render(instances) -> jnp.ndarray:
      _render = jax.jit(
        render_instances,
        static_arguments=("width", "height", "enable_shadow"),
        inline=True,
      )
      img = _render(instances=instances, 
                    width=canvas_width, 
                    height=canvas_height, 
                    camera=camera, 
                    camera_target=center)
      arr = transpose_for_display((img * 255).astype(jnp.uint8))
      return arr
    _render_batch = jax.jit(jax.vmap(_render))
    images = _render_batch(batched_instances)
    return images

def rotate(model: RendererMesh, rotation_axis: tuple[float, float, float], degree: float) -> Instance:
  instance = Instance(model=model)
  instance = instance.replace_with_orientation(rotation_matrix=rotation_matrix(rotation_axis, degree))

  return instance

# obj_path = "./src/render/cube.obj"
# model = make_model(open(obj_path, "r").readlines(), None, None)
# gen_model = lambda degree: rotate(model, rotation_axis, degree)
# batch_rotation = jax.jit(jax.vmap(gen_model)).lower(degrees).compile()
# instances = [batch_rotation(degrees)]
# render_with_states = gen_compile_render(instances)
# images = render_with_states(instances)


if __name__ == "__main__":
    mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_string(TEST_XML_STRING)
    key = jax.random.PRNGKey(30)
    params = ICLandParams(mj_model, None, 1)
    state = init(key, params)

    state = step(key, state, params, jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    mjx_model = state.mjx_model
    mjx_data = state.mjx_data
    print(mjx_model.tex_height)
    vs, fs, norms, uvs, faces_norm, faces_uv = extract_mesh(mjx_model, 0)


    diffuse_texture = jnp.ones((2, 2, 3), dtype=jnp.float32)
    specular_texture = jnp.zeros((2, 2), dtype=jnp.float32)

    renderer_model = renderer.Model(
        verts=vs,
        faces=fs,
        norms=norms,
        uvs=uvs,
        faces_norm=faces_norm,
        faces_uv=faces_uv,
        diffuse_map=jax.numpy.swapaxes(diffuse_texture, 0, 1)[:, ::-1, :],
        specular_map=jax.numpy.swapaxes(specular_texture, 0, 1)[:, ::-1],
    )

    camera = renderer.CameraParameters(
        viewWidth=640,
        viewHeight=480,
        position=jnp.array([2.0, 3.0, 1.0], dtype=jnp.float32),
        target=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        up=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
    )

    light = Light(
      direction=jnp.array([0.5, -0.5, 0.5]),
      ambient=0.1,
      diffuse=0.85,
      specular=0.05
    )

    image = renderer.Renderer.get_camera_image(
        objects=[renderer.ModelObject(model=renderer_model)],
        camera=camera,
        light=light,
        width=640,
        height=480,
    )

    image_np = jnp.array(image)


    print(image_np)
    plt.imshow(image_np)
    plt.axis('off')
    plt.title("Rendered MJX Model")
    plt.savefig("rendered_image.png", dpi=300, bbox_inches='tight')
    plt.show()


