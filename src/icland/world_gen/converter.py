"""The code defines functions to generate block, ramp, and vertical ramp columns in a 3D world using JAX arrays and exports the generated mesh to an STL file."""

import os

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
)  # pragma: no cover

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
)  # pragma: no cover


# Optimization: Pre-compute rotation matrices
def __get_rotation_matrix(rotation: jax.Array) -> jax.Array:
    angle = -jnp.pi * rotation / 2
    cos_t = jnp.cos(angle)
    sin_t = jnp.sin(angle)
    return jnp.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])


ROTATION_MATRICES = jnp.stack(
    [__get_rotation_matrix(jnp.array(r, dtype=jnp.int32)) for r in range(4)]
)  # pragma: no cover

# Maximum number of triangles per column
MAX_TRIANGLES = 72  # pragma: no cover


def __make_block_column(  # pragma: no cover
    x: jax.Array, y: jax.Array, level: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Block column generation with fixed output size."""
    return jnp.zeros((12, 3, 3)), (
        jnp.array(
            [
                # Bottom face
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
                # Top face
                [[0, 0, level], [0, 1, level], [1, 0, level]],
                [[1, 0, level], [0, 1, level], [1, 1, level]],
                # Front face
                [[0, 0, 0], [0, 0, level], [1, 0, 0]],
                [[1, 0, 0], [0, 0, level], [1, 0, level]],
                # Back face
                [[0, 1, 0], [1, 1, 0], [0, 1, level]],
                [[1, 1, 0], [1, 1, level], [0, 1, level]],
                # Left face
                [[0, 0, 0], [0, 1, 0], [0, 0, level]],
                [[0, 1, 0], [0, 1, level], [0, 0, level]],
                # Right face
                [[1, 0, 0], [1, 0, level], [1, 1, 0]],
                [[1, 1, 0], [1, 0, level], [1, 1, level]],
            ]
        )
        + jnp.array([x, y, 0])[None, None, :]
    )


def __make_ramp_column(  # pragma: no cover
    x: jax.Array, y: jax.Array, level: jax.Array, rotation: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Ramp generation with fixed output size."""
    centered = (
        jnp.array(
            [
                # Bottom face
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
                # Top face
                [[1, 1, level], [0, 0, level - 1], [0, 1, level - 1]],
                [[0, 0, level - 1], [1, 1, level], [1, 0, level]],
                # Front face
                [[0, 0, 0], [0, 0, level - 1], [1, 0, 0]],
                [[1, 0, 0], [0, 0, level - 1], [1, 0, level]],
                # Back face
                [[0, 1, 0], [1, 1, 0], [0, 1, level - 1]],
                [[1, 1, 0], [1, 1, level], [0, 1, level - 1]],
                # Left face
                [[0, 0, 0], [0, 1, 0], [0, 0, level - 1]],
                [[0, 1, 0], [0, 1, level - 1], [0, 0, level - 1]],
                # Right face
                [[1, 0, 0], [1, 0, level], [1, 1, 0]],
                [[1, 1, 0], [1, 0, level], [1, 1, level]],
            ]
        )
        - 0.5
    )
    rotated = jnp.einsum("ijk,kl->ijl", centered, ROTATION_MATRICES[rotation])
    final_ramp = rotated + 0.5
    return jnp.zeros((12, 3, 3)), (final_ramp + jnp.array([x, y, 0])[None, None, :])


def __make_vramp_column(  # pragma: no cover
    x: jax.Array,
    y: jax.Array,
    from_level: jax.Array,
    to_level: jax.Array,
    rotation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Vertical ramp generation with fixed output size."""
    vramp_count = to_level - from_level + 1
    column = __make_block_column(x, y, from_level)[1]
    centered = (
        jnp.array(
            [
                # Bottom face
                [[1, 0, 0], [0, vramp_count, 0], [0, 0, 0]],
                [[0, vramp_count, 0], [1, 0, 0], [1, vramp_count, 0]],
                # Side face
                [[1, vramp_count, 1], [1, 0, 0], [1, vramp_count, 0]],
                [[1, 0, 0], [1, vramp_count, 1], [1, 0, 1]],
                # Right side
                [[1, 0, 0], [0, 0, 0], [1, 0, 1]],
                # Left side
                [[1, vramp_count, 0], [1, vramp_count, 1], [0, vramp_count, 0]],
                # Top ramp face
                [[1, vramp_count, 1], [0, 0, 0], [0, vramp_count, 0]],
                [[0, 0, 0], [1, vramp_count, 1], [1, 0, 1]],
            ]
        )
        - 0.5
    )
    x_rotation = jnp.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    vertical = jnp.einsum("ijk,kl->ijl", centered, x_rotation)
    rotated = jnp.einsum("ijk,kl->ijl", vertical, ROTATION_MATRICES[(rotation - 1) % 4])
    vramp = (
        jnp.zeros((12, 3, 3))
        .at[:8]
        .set(rotated + 0.5 + jnp.array([x, y, from_level - 1])[None, None, :])
    )
    return vramp, column


def create_world(tile_map: jax.Array) -> jax.Array:  # pragma: no cover
    """World generation with consistent shapes."""
    i_indices, j_indices = jnp.meshgrid(
        jnp.arange(tile_map.shape[0]), jnp.arange(tile_map.shape[1]), indexing="ij"
    )
    i_indices = i_indices[..., jnp.newaxis]  # Shape: (w, h, 1)
    j_indices = j_indices[..., jnp.newaxis]  # Shape: (w, h, 1)

    coords = jnp.concatenate([i_indices, j_indices, tile_map], axis=-1)

    def process_tile(entry: jax.Array) -> jax.Array:
        x, y, block, rotation, frm, to = entry
        if block == TileType.SQUARE.value:
            return __make_block_column(x, y, to)
        elif block == TileType.RAMP.value:
            return __make_ramp_column(x, y, to, rotation)
        elif block == TileType.VRAMP.value:
            return __make_vramp_column(x, y, frm, to, rotation)
        else:
            raise RuntimeError("Unknown tile type. Please check XMLReader")

    pieces = jnp.zeros((tile_map.shape[0] * tile_map.shape[1] * 2, 12, 3, 3))
    tile_start, pieces_start = 0, 0
    while tile_start < tile_map.shape[0] * tile_map.shape[1]:
        i = tile_start // tile_map.shape[0]
        j = tile_start % tile_map.shape[1]
        a, b = process_tile(coords.at[i, j].get())
        pieces = pieces.at[pieces_start].set(a)
        pieces = pieces.at[pieces_start + 1].set(b)
        pieces_start += 2
        tile_start += 1

    return pieces


def export_stl(pieces: jax.Array, filename: str) -> mesh.Mesh:  # pragma: no cover
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


def export_stls(pieces: jax.Array, file_prefix: str) -> None:  # pragma: no cover
    """Export each piece as an stl."""
    # # Helper function to filter out padding after JIT compilation
    pieces_reshaped = pieces.reshape(-1, *pieces.shape[-2:])
    # Convert from JAX array to numpy
    triangles = np.array(pieces_reshaped)
    # Invert the normals
    triangles = triangles[:, ::-1, :]

    triangles = np.ascontiguousarray(triangles, dtype=np.float32)

    print(triangles.shape)

    n_pieces = pieces.shape[0] // 2
    n_triangles = len(triangles) // n_pieces
    for i in range(n_pieces):
        # Create the mesh data structure
        world_mesh = mesh.Mesh(
            np.zeros(n_triangles, dtype=mesh.Mesh.dtype),
            remove_duplicate_polygons=RemoveDuplicates.NONE,
        )
        world_mesh.vectors = triangles[
            (n_triangles * i) : (n_triangles * i + n_triangles)
        ]
        world_mesh.save(file_prefix + "_" + str(i) + ".stl")


def generate_mjcf_from_meshes(  # pragma: no cover
    tile_map: jax.Array, mesh_dir="meshes/", output_file="generated_mjcf.xml"
) -> None:
    """Generates MJCF file from column meshes that form the world."""
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".stl")]

    mjcf = """<mujoco model=\"generated_mesh_world\">
    <compiler meshdir=\"meshes/\"/>
    <default>
        <geom type=\"mesh\" contype=\"0\" conaffinity=\"0\"/>
    </default>
    
    <worldbody>\n"""

    for i, mesh_file in enumerate(mesh_files):
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += (
            f'        <geom name="{mesh_name}" mesh="{mesh_name}" pos="0 0 0"/>' + "\n"
        )

    mjcf += "    </worldbody>\n\n    <asset>\n"

    for mesh_file in mesh_files:
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += f'        <mesh name="{mesh_name}" file="{mesh_file}"/>' + "\n"

    mjcf += "    </asset>\n</mujoco>\n"

    with open(output_file, "w") as f:
        f.write(mjcf)

    print(f"MJCF file generated and saved as {output_file}")
