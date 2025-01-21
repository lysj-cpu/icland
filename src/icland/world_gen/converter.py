import jax
import jax.numpy as jnp
from jax import random, vmap
from functools import partial
import numpy as np
from stl import mesh

# Previous constants (BLOCK_VECTORS, RAMP_VECTORS, ROTATION_MATRICES) remain the same...
# Optimization: Pre-compute block and ramp vectors as constants
BLOCK_VECTORS = jnp.array([
    # Bottom face
    [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
    # Top face
    [[0, 0, 1], [0, 1, 1], [1, 0, 1]],
    [[1, 0, 1], [0, 1, 1], [1, 1, 1]],
    # Front face
    [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
    [[1, 0, 0], [0, 0, 1], [1, 0, 1]],
    # Back face
    [[0, 1, 0], [1, 1, 0], [0, 1, 1]],
    [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
    # Left face
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
    # Right face
    [[1, 0, 0], [1, 0, 1], [1, 1, 0]],
    [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
])

RAMP_VECTORS = jnp.array([
    # Bottom face
    [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, 0, 0], [1, 1, 0]],
    # Side face
    [[0, 1, 1], [0, 0, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
    # Right side
    [[0, 0, 1], [1, 0, 0], [0, 0, 0]],
    # Left side
    [[1, 1, 0], [0, 1, 1], [0, 1, 0]],
    # Top ramp face
    [[0, 1, 1], [1, 0, 0], [1, 1, 0]],
    [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
])

# Optimization: Pre-compute rotation matrices
def get_rotation_matrix(rotation):
    angle = -jnp.pi * rotation / 2
    cos_t = jnp.cos(angle)
    sin_t = jnp.sin(angle)
    return jnp.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])

ROTATION_MATRICES = jnp.stack([get_rotation_matrix(r) for r in range(4)])

# Maximum number of triangles per column
MAX_TRIANGLES = 72  # 6 levels * 12 triangles per level

def pad_triangles(triangles: jnp.ndarray, max_triangles: int = MAX_TRIANGLES):
    """Pad triangle array to fixed size using dynamic padding."""
    triangles = triangles.astype("float32")
    current_triangles = triangles.shape[0]
    
    # Create full-size array of zeros
    result = jnp.zeros((max_triangles, 3, 3))
    
    # Use dynamic_update_slice to copy actual triangles
    result = jax.lax.dynamic_update_slice(result, triangles, (0, 0, 0))
    
    return result

def make_block_column(x: int, y: int, level: int):
    """Block column generation with fixed output size."""
    # Calculate actual number of triangles
    num_triangles = level * 12
    
    # Generate blocks for required levels
    offsets = jnp.arange(1, level + 1)[:, None, None, None]
    translation = jnp.array([x, y, -1])[None, None, None, :]
    blocks = BLOCK_VECTORS[None, :, :, :] + translation + jnp.array([0, 0, 1])[None, None, None, :] * offsets
    
    # Reshape and pad
    blocks = blocks.reshape(-1, 3, 3)
    return pad_triangles(blocks)

def make_ramp_column(x: int, y: int, level: int, rotation: int):
    """Ramp generation with fixed output size."""
    # Base blocks
    base_blocks = make_block_column(x, y, level - 1)[:((level-1) * 12)]
    
    # Ramp transformation
    centered = RAMP_VECTORS - 0.5
    rotated = jnp.einsum('ijk,kl->ijl', centered, ROTATION_MATRICES[rotation])
    final_ramp = rotated + 0.5 + jnp.array([x, y, level - 1])[None, None, :]
    
    # Concatenate and pad
    combined = jnp.concatenate([base_blocks, final_ramp])
    return pad_triangles(combined)

def make_vramp_column(x: int, y: int, from_level: int, to_level: int, rotation: int):
    """Vertical ramp generation with fixed output size."""
    # Base blocks
    base_blocks = make_block_column(x, y, from_level - 1)[:(from_level-1) * 12]
    
    # Vertical ramp transformation
    centered = RAMP_VECTORS - 0.5
    x_rotation = jnp.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    vertical = jnp.einsum('ijk,kl->ijl', centered, x_rotation)
    rotated = jnp.einsum('ijk,kl->ijl', vertical, ROTATION_MATRICES[rotation])
    
    # Generate vramps for each level
    vramp_count = to_level - from_level + 1
    offsets = jnp.arange(from_level, to_level + 1)[:, None, None, None]
    translation = jnp.array([x, y, -1])[None, None, None, :]
    vramps = (rotated[None, :, :, :] + translation + 0.5 +
             jnp.array([0, 0, 1])[None, None, None, :] * offsets)
    
    # Concatenate and pad
    combined = jnp.concatenate([base_blocks, vramps.reshape(-1, 3, 3)])
    return pad_triangles(combined)

@partial(jax.jit, static_argnums=[0, 1])
def create_world(world_size: int, pattern_fn, key: random.PRNGKey = None):
    """World generation with consistent shapes."""
    i_coords, j_coords = jnp.meshgrid(
        jnp.arange(world_size),
        jnp.arange(world_size),
        indexing='ij'
    )
    coords = jnp.stack([i_coords.flatten(), j_coords.flatten()], axis=1)
    
    if key is not None:
        keys = random.split(key, world_size * world_size)
        vectorized_pattern = vmap(lambda coord, subkey: pattern_fn(coord[0], coord[1], subkey))
        pieces = vectorized_pattern(coords, keys)
    else:
        vectorized_pattern = vmap(lambda coord: pattern_fn(coord[0], coord[1], None))
        pieces = vectorized_pattern(coords)

    return pieces

def prepare_for_stl(pieces):
    """Convert JAX array to numpy array in the correct format for STL export."""
    # Convert from JAX array to numpy
    triangles = np.array(pieces)
    
    # Ensure the array is contiguous and in float32 format
    # numpy-stl expects float32 data
    triangles = np.ascontiguousarray(triangles, dtype=np.float32)
    
    # Create the mesh data structure
    mesh_data = np.zeros(len(triangles), dtype=mesh.Mesh.dtype)
    
    # Assign vectors to the mesh
    mesh_data['vectors'] = triangles
    
    return mesh_data
  
# # Helper function to filter out padding after JIT compilation
# def filter_padding(pieces: jnp.ndarray) -> jnp.ndarray:
#     """Remove padding (zeros) from the generated pieces."""
#     mask = ~(jnp.abs(pieces) < 1e-7).all(axis=(2, 3))
#     return pieces[mask]

world_size = 5
key = random.PRNGKey(0)
# Create valley world
terrace_world = create_world(world_size, terrace_pattern_jax, key)
# filtered_world = filter_padding(valley_world)
# Convert to numpy and prepare data
mesh_data = prepare_for_stl(terrace_world.reshape(-1, *terrace_world.shape[-2:]))

# Create and save mesh
terrace_mesh = mesh.Mesh(mesh_data)
terrace_mesh.save("terrace.stl")