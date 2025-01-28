"""The code defines functions to generate block, ramp, and vertical ramp columns in a 3D world using JAX arrays and exports the generated mesh to an STL file."""

import jax
import jax.numpy as jnp
import numpy as np
from stl import mesh
from stl.base import RemoveDuplicates

from icland.world_gen.XMLReader import TileType

# Previous constants (BLOCK_VECTORS, RAMP_VECTORS, ROTATION_MATRICES) remain the same...
# Optimization: Pre-compute block and ramp vectors as constants
BLOCK_VECTORS = jnp.array(
    [
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
        [[1, 1, 0], [1, 0, 1], [1, 1, 1]],
    ]
)

RAMP_VECTORS = jnp.array(
    [
        # Bottom face
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 0], [1, 1, 0]],
        # Side face
        [[1, 1, 1], [1, 0, 0], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 1], [1, 0, 1]],
        # Right side
        [[1, 0, 1], [0, 0, 0], [1, 0, 0]],
        # Left side
        [[0, 1, 0], [1, 1, 1], [1, 1, 0]],
        # Top ramp face
        [[1, 1, 1], [0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [1, 0, 1]],
    ]
)


# Optimization: Pre-compute rotation matrices
def __get_rotation_matrix(rotation: jax.Array) -> jax.Array:
    angle = -jnp.pi * rotation / 2
    cos_t = jnp.cos(angle)
    sin_t = jnp.sin(angle)
    return jnp.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])


ROTATION_MATRICES = jnp.stack(
    [__get_rotation_matrix(jnp.array(r, dtype=jnp.int32)) for r in range(4)]
)

# Maximum number of triangles per column
MAX_TRIANGLES = 72  # 6 levels * 12 triangles per level


def pad_triangles(
    triangles: jax.Array,
    max_triangles: jax.Array = jnp.array(MAX_TRIANGLES, dtype=jnp.int32),
) -> jax.Array:
    """Pad triangle array to fixed size using dynamic padding."""
    triangles = triangles.astype("float32")
    current_triangles = triangles.shape[0]

    # Create full-size array of 0s
    result = jnp.zeros((max_triangles, 3, 3))

    # Use dynamic_update_slice to copy actual triangles
    result = jax.lax.dynamic_update_slice(result, triangles, (0, 0, 0))

    return result


def make_block_column(x: jax.Array, y: jax.Array, level: jax.Array) -> jax.Array:
    """Block column generation with fixed output size."""
    # Calculate actual number of triangles
    num_triangles = level * 12

    # Generate blocks for required levels
    offsets = jnp.arange(1, level + 1)[:, None, None, None]
    translation = jnp.array([x, y, -1])[None, None, None, :]
    blocks = (
        BLOCK_VECTORS[None, :, :, :]
        + translation
        + jnp.array([0, 0, 1])[None, None, None, :] * offsets
    )

    # Reshape and pad
    blocks = blocks.reshape(-1, 3, 3)
    return pad_triangles(blocks)


def make_ramp_column(
    x: jax.Array, y: jax.Array, level: jax.Array, rotation: jax.Array
) -> jax.Array:
    """Ramp generation with fixed output size."""
    # Base blocks
    base_blocks = make_block_column(x, y, level - 1)[: ((level - 1) * 12)]

    # Ramp transformation
    centered = RAMP_VECTORS - 0.5
    rotated = jnp.einsum("ijk,kl->ijl", centered, ROTATION_MATRICES[rotation])
    final_ramp = rotated + 0.5 + jnp.array([x, y, level - 1])[None, None, :]

    # Concatenate and pad
    combined = jnp.concatenate([base_blocks, final_ramp])
    return pad_triangles(combined)


def make_vramp_column(
    x: jax.Array,
    y: jax.Array,
    from_level: jax.Array,
    to_level: jax.Array,
    rotation: jax.Array,
) -> jax.Array:
    """Vertical ramp generation with fixed output size."""
    # Base blocks
    base_blocks = make_block_column(x, y, from_level)[: from_level * 12]

    # Vertical ramp transformation
    centered = RAMP_VECTORS - 0.5
    x_rotation = jnp.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    vertical = jnp.einsum("ijk,kl->ijl", centered, x_rotation)
    rotated = jnp.einsum("ijk,kl->ijl", vertical, ROTATION_MATRICES[rotation])

    # Generate vramps for each level
    vramp_count = to_level - from_level + 1
    offsets = jnp.arange(from_level + 1, to_level + 1)[:, None, None, None]
    translation = jnp.array([x, y, -1])[None, None, None, :]
    vramps = (
        rotated[None, :, :, :]
        + translation
        + 0.5
        + jnp.array([0, 0, 1])[None, None, None, :] * offsets
    )

    # Concatenate and pad
    combined = jnp.concatenate([base_blocks, vramps.reshape(-1, 3, 3)])
    return pad_triangles(combined)


def create_world(tile_map: jax.Array) -> jax.Array:
    """World generation with consistent shapes."""
    i_indices, j_indices = jnp.meshgrid(
        jnp.arange(tile_map.shape[0]), jnp.arange(tile_map.shape[1]), indexing="ij"
    )
    i_indices = i_indices[..., jnp.newaxis]  # Shape: (10, 10, 1)
    j_indices = j_indices[..., jnp.newaxis]  # Shape: (10, 10, 1)

    coords = jnp.concatenate([i_indices, j_indices, tile_map], axis=-1)

    def process_tile(entry: jax.Array) -> jax.Array:
        x, y, block, rotation, frm, to = entry
        if block == TileType.SQUARE.value:
            return make_block_column(x, y, to)
        elif block == TileType.RAMP.value:
            return make_ramp_column(x, y, to, rotation)
        elif block == TileType.VRAMP.value:
            return make_vramp_column(x, y, frm, to, rotation)
        else:
            raise RuntimeError("Unknown tile type. Please check XMLReader")

    pieces = jnp.zeros((tile_map.shape[0], tile_map.shape[1], MAX_TRIANGLES, 3, 3))
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            pieces = pieces.at[i, j].set(process_tile(coords.at[i, j].get()))

    return pieces


def export_stl(pieces: jax.Array, filename: str) -> mesh.Mesh:
    """Convert JAX array to numpy array in the correct format for STL export."""
    # # Helper function to filter out padding after JIT compilation
    pieces_reshaped = pieces.reshape(-1, *pieces.shape[-2:])
    # Convert from JAX array to numpy
    triangles = np.array(pieces_reshaped)
    # Invert the normals
    triangles = triangles[:, ::-1, :]

    # Ensure the array is contiguous and in float32 format
    # numpy-stl expects float32 data
    triangles = np.ascontiguousarray(triangles, dtype=np.float32)

    # Create the mesh data structure
    world_mesh = mesh.Mesh(
        np.zeros(len(triangles), dtype=mesh.Mesh.dtype),
        remove_duplicate_polygons=RemoveDuplicates.NONE,
    )

    # Assign vectors to the mesh
    world_mesh.vectors = triangles
    world_mesh.save(filename)

    return world_mesh
