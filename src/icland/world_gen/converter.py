from stl import mesh
from stl.base import RemoveDuplicates
import math
import numpy as np

# PACKAGE PRIVATE FUNCTIONS

# Package private helper function for making a cube by defining its faces
def __make_block_mesh():
  # Initialize the cube data with 12 triangles (2 triangles per face)
  cube_data = np.zeros(12, dtype=mesh.Mesh.dtype)

  # Bottom face
  cube_data["vectors"][0] = np.array([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0]
  ])
  cube_data["vectors"][1] = np.array([
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0]
  ])

  # Top face
  cube_data["vectors"][2] = np.array([
      [0, 0, 1],
      [0, 1, 1],
      [1, 0, 1]
  ])
  cube_data["vectors"][3] = np.array([
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1]
  ])

  # Front face
  cube_data["vectors"][4] = np.array([
      [0, 0, 0],
      [0, 0, 1],
      [1, 0, 0]
  ])
  cube_data["vectors"][5] = np.array([
      [1, 0, 0],
      [0, 0, 1],
      [1, 0, 1]
  ])

  # Back face
  cube_data["vectors"][6] = np.array([
      [0, 1, 0],
      [1, 1, 0],
      [0, 1, 1]
  ])
  cube_data["vectors"][7] = np.array([
      [1, 1, 0],
      [1, 1, 1],
      [0, 1, 1]
  ])

  # Left face
  cube_data["vectors"][8] = np.array([
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
  ])
  cube_data["vectors"][9] = np.array([
      [0, 1, 0],
      [0, 1, 1],
      [0, 0, 1]
  ])

  # Right face
  cube_data["vectors"][10] = np.array([
      [1, 0, 0],
      [1, 0, 1],
      [1, 1, 0]
  ])
  cube_data["vectors"][11] = np.array([
      [1, 1, 0],
      [1, 0, 1],
      [1, 1, 1]
  ])

  # Create the cube mesh
  cube = mesh.Mesh(cube_data.copy())
            
  return cube

# Package private helper function for making a ramp mesh
def __make_ramp_mesh():
  r_data = np.zeros(8, dtype=mesh.Mesh.dtype)

  # Bottom face
  r_data["vectors"][0] = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
  ])
  r_data["vectors"][1] = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0]
  ])

  # Side face
  r_data["vectors"][2] = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [0, 1, 0]
  ])
  r_data["vectors"][3] = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [0, 0, 1]
  ])

  # Right side
  r_data["vectors"][4] = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 0]
  ])

  # Left side
  r_data["vectors"][5] = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [0, 1, 0]
  ])

  # Top ramp face
  r_data["vectors"][6] = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0]
  ])
  r_data["vectors"][7] = np.array([
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 1]
  ])

  ramp = mesh.Mesh(r_data.copy())
  return ramp

# Ensure blocks are created correctly (adhering to our constraints)
def __block_assertions(x, y, level, rotation):
  return (
    # Rotation in units of 90 degrees counterclockwise
    rotation in [0, 1, 2, 3]
    
    # Positive world coordinates
    and x >= 0 and y >= 0
          
    # Level within [1, 6]
    and 1 <= level <= 6)

# MODULE PRIVATE FUNCTIONS

# Helper function to make a block
def _make_block(x, y, level) -> mesh.Mesh:
  # Assertion checks
  assert __block_assertions(x, y, level, 0)
  
  # Create a block
  cube = __make_block_mesh()
  
  # Move to designated X Y and Z coordinates
  cube.translate(np.array([x, y, level - 1]))
  
  return cube

# PUBLIC FUNCTIONS

def make_square(x, y, level) -> mesh.Mesh:
  """Make a square at x, y coordinates in the given level."""
  assert __block_assertions(x, y, level, 0)
  
  block_column = []
  for l in range(1, level + 1):
    c = _make_block(x, y, l)
    block_column.append(c.data.copy())
    
  combined = mesh.Mesh(np.concatenate(block_column))
  return combined

def make_vramp(x, y, from_level, to_level, rotation=0) -> mesh.Mesh:
  """Make a vertical ramp (cheese) at x, y coordinates.

    With lower level being from_level and the higher level
    being to_level with default rotation being 0.
  """
  # Assertion checks.
  assert __block_assertions(x, y, from_level, rotation)
  assert __block_assertions(0, 0, to_level, 0)

  block_column = []
  for l in range(from_level, to_level + 1):
    vramp = __make_ramp_mesh()
    vramp.x -= 0.5
    vramp.y -= 0.5
    vramp.z -= 0.5
    
    # TODO: Fix if broken
    vramp.rotate([1.0, 0.0, 0.0], math.radians(-90))
    vramp.rotate([0.0, 0.0, 1.0], math.radians(-90 * rotation))
    
    vramp.x += 0.5
    vramp.y += 0.5
    vramp.z += 0.5
  
    vramp.translate(np.array([x, y, l - 1]))
    block_column.append(vramp.data.copy())
  
  # Append cube blocks below
  for l in range(1, from_level):
    c = _make_block(x, y, l)
    block_column.append(c.data.copy())
  
  combined = mesh.Mesh(np.concatenate(block_column))
  return combined

def make_ramp(x, y, level, rotation=0) -> mesh.Mesh:
  """Make a ramp at x, y coordinates in the given level with default rotation being 0."""
  # Assertion checks.
  assert __block_assertions(x, y, level, rotation)

  ramp = __make_ramp_mesh()
  
  # Rotate about origin's Z axes
  ramp.x -= 0.5
  ramp.y -= 0.5
  ramp.rotate([0.0, 0.0, 1.0], math.radians(-90 * rotation))
  ramp.x += 0.5
  ramp.y += 0.5
  
  # Move to designated X Y and Z coordinates
  ramp.translate(np.array([x, y, level - 1]))
  
  block_column = [ramp.data.copy()]
  
  # Append cube blocks below
  for i in range(1, level):
    c = _make_block(x, y, i)
    block_column.append(c.data.copy())
  
  combined = mesh.Mesh(np.concatenate(block_column))
  
  return combined

# Flat world
world_size = 5 # 5x5 world
world = []
for i in range(world_size):
  for j in range(world_size):
    s = make_ramp(i, j, i + 2, 2)
    world.append(s.data.copy())
    
world_mesh = mesh.Mesh(np.concatenate(world), remove_duplicate_polygons=RemoveDuplicates.ALL)
world_mesh.save("world_r.stl")

from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Create a new plot
figure = plt.figure()
axes = figure.add_subplot(projection='3d')

# Load the STL files and add the vectors to the plot
poly_collection = mplot3d.art3d.Poly3DCollection(world_mesh.vectors)
poly_collection.set_color((0.7,0.7,0.7))  # play with color
axes.add_collection3d(poly_collection)

# Show the plot to the screen
plt.show()