"""The code defines functions to generate block, ramp, and vertical ramp columns in a 3D world using JAX arrays and exports the generated mesh to an STL file."""

# import os
from functools import partial

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
    return (
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
    ), jnp.zeros((12, 3, 3))


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
    return (final_ramp + jnp.array([x, y, 0])[None, None, :]), jnp.zeros((12, 3, 3))


def __make_vramp_column(  # pragma: no cover
    x: jax.Array,
    y: jax.Array,
    from_level: jax.Array,
    to_level: jax.Array,
    rotation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Vertical ramp generation with fixed output size."""
    vramp_count = to_level - from_level + 1
    column = __make_block_column(x, y, from_level)[0]
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
    return column, vramp


def create_world(tile_map: jax.Array) -> jax.Array:  # pragma: no cover
    """World generation with consistent shapes."""
    i_indices, j_indices = jnp.meshgrid(
        jnp.arange(tile_map.shape[0]), jnp.arange(tile_map.shape[1]), indexing="ij"
    )
    i_indices = i_indices[..., jnp.newaxis]  # Shape: (w, h, 1)
    j_indices = j_indices[..., jnp.newaxis]  # Shape: (w, h, 1)

    coords = jnp.concatenate([i_indices, j_indices, tile_map], axis=-1)

    def process_tile(entry: jax.Array) -> tuple[jax.Array, jax.Array]:
        x, y, block, rotation, frm, to = entry
        if block == TileType.SQUARE.value:
            return __make_block_column(x, y, to)
        elif block == TileType.RAMP.value:
            return __make_ramp_column(x, y, to, rotation)
        elif block == TileType.VRAMP.value:
            return __make_vramp_column(x, y, frm, to, rotation)
        else:
            raise RuntimeError("Unknown tile type. Please check XMLReader")

    pieces = jnp.zeros((tile_map.shape[0] * tile_map.shape[1], 12, 3, 3))
    tile_start = 0
    while tile_start < tile_map.shape[0] * tile_map.shape[1]:
        i = tile_start // tile_map.shape[0]
        j = tile_start % tile_map.shape[1]
        a, _ = process_tile(coords.at[i, j].get())
        pieces = pieces.at[tile_start].set(a)
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

    n_pieces = pieces.shape[0]
    n_triangles = len(triangles) // n_pieces
    for i in range(0, n_pieces):
        # Create the mesh data structure
        world_mesh = mesh.Mesh(
            np.zeros(n_triangles, dtype=mesh.Mesh.dtype),
            remove_duplicate_polygons=RemoveDuplicates.NONE,
        )
        world_mesh.vectors = triangles[
            (n_triangles * i) : (n_triangles * i + n_triangles)
        ]
        world_mesh.save(file_prefix + "_" + str(i) + ".stl")


@partial(jax.jit, static_argnums=[2])
def sample_spawn_points(
    key: jax.Array, tilemap: jax.Array, num_objects: jnp.int32 = 1
) -> jax.Array:  # pragma: no cover
    """Sample num_objects spawn points from the tilemap."""
    TILE_DATA_HEIGHT_INDEX = 3
    spawn_map = __get_spawn_map(tilemap)
    flat_tilemap = spawn_map.flatten()
    nonzero_indices = jnp.where(
        flat_tilemap != 0, size=flat_tilemap.shape[0], fill_value=-1
    )[0]

    def run_once(key: jax.Array) -> jax.Array:
        def pick(
            item: tuple[jnp.int32, jax.Array],
        ) -> tuple[jnp.int32, jax.Array]:
            _, key = item
            key, subkey = jax.random.split(key)
            choice = jax.random.choice(subkey, nonzero_indices)
            return choice, key

        random_index, key = jax.lax.while_loop(lambda x: x[0] < 0, pick, (-1, key))

        # Convert the flat index back to 2D coordinates
        row = random_index // spawn_map.shape[0]
        col = random_index % spawn_map.shape[0]

        return jnp.array(
            [row + 0.5, col + 0.5, tilemap[row, col, TILE_DATA_HEIGHT_INDEX] + 1]
        )

    keys = jax.random.split(key, num_objects)

    return jax.vmap(run_once)(keys)


def __get_spawn_map(combined: jax.Array) -> jax.Array:  # pragma: no cover
    combined = combined.astype(jnp.int32)
    w, h = combined.shape[0], combined.shape[1]

    # Initialize arrays with JAX functions
    TILE_DATA_SIZE = 4
    NUM_ROTATIONS = 4
    NUM_COORDS = 2
    visited = jax.lax.full((w, h), False, dtype=jnp.bool)
    spawnable = jax.lax.full((w, h), 0, dtype=jnp.int32)

    def __adj_jit(i: jnp.int32, j: jnp.int32, combined: jax.Array) -> jax.Array:
        slots = jnp.full((TILE_DATA_SIZE, NUM_COORDS), -1)
        dx = jnp.array([-1, 0, 1, 0])
        dy = jnp.array([0, 1, 0, -1])

        def process_square(
            combined: jax.Array,
            i: jnp.int32,
            j: jnp.int32,
            slots: jax.Array,
            dx: jax.Array,
            dy: jax.Array,
        ) -> jax.Array:
            tile, r, f, level = combined[i, j]
            for d in range(TILE_DATA_SIZE):
                x = i + dx[d]
                y = j + dy[d]

                def process_square_inner(slots: jax.Array) -> jax.Array:
                    q, r2, f2, l = combined[x, y]
                    slots = jax.lax.cond(
                        jnp.any(
                            jnp.array(
                                [
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.RAMP.value,
                                                r2
                                                == (NUM_ROTATIONS - d) % NUM_ROTATIONS,
                                                f2 == level - 1,
                                                l == level,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.RAMP.value,
                                                r2
                                                == (NUM_ROTATIONS - 2 - d)
                                                % NUM_ROTATIONS,
                                                f2 == level,
                                                l == level + 1,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [q == TileType.SQUARE.value, l == level]
                                        )
                                    ),
                                ]
                            )
                        ),
                        lambda z: z.at[d].set(jnp.array([x, y])),
                        lambda z: z,
                        slots,
                    )
                    return slots

                slots = jax.lax.cond(
                    jnp.all(
                        jnp.array(
                            [
                                0 <= x,
                                x < combined.shape[0],
                                0 <= y,
                                y < combined.shape[1],
                            ]
                        )
                    ),
                    process_square_inner,
                    lambda x: x,
                    slots,
                )
            return slots

        def process_ramp(
            combined: jax.Array,
            i: jnp.int32,
            j: jnp.int32,
            slots: jax.Array,
            dx: jax.Array,
            dy: jax.Array,
        ) -> jax.Array:
            tile, r, f, level = combined[i, j]
            mask = jnp.where((r + 1) % 2 == 0, 1, 0)
            for d in range(NUM_ROTATIONS):
                x = i + dx[d]
                y = j + dy[d]

                def process_ramp_inner(slots: jax.Array) -> jax.Array:
                    q, r2, f2, l = combined[x, y]
                    slots = jax.lax.cond(
                        jnp.any(
                            jnp.array(
                                [
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.SQUARE.value,
                                                d
                                                == (NUM_ROTATIONS - 2 - r)
                                                % NUM_ROTATIONS,
                                                l == level,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.SQUARE.value,
                                                d
                                                == (NUM_ROTATIONS - r) % NUM_ROTATIONS,
                                                l == f,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.RAMP.value,
                                                d
                                                == (NUM_ROTATIONS - 2 - r)
                                                % NUM_ROTATIONS,
                                                r == r2,
                                                f2 == level,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.RAMP.value,
                                                d
                                                == (NUM_ROTATIONS - r) % NUM_ROTATIONS,
                                                r == r2,
                                                l == f,
                                            ]
                                        )
                                    ),
                                ]
                            )
                        ),
                        lambda z: z.at[d].set(jnp.array([x, y])),
                        lambda z: z,
                        slots,
                    )
                    return slots

                slots = jax.lax.cond(
                    jnp.all(
                        jnp.array(
                            [
                                0 <= x,
                                x < combined.shape[0],
                                0 <= y,
                                y < combined.shape[1],
                                (d + mask) % 2 == 0,
                            ]
                        )
                    ),
                    process_ramp_inner,
                    lambda x: x,
                    slots,
                )
            return slots

        slots = jax.lax.switch(
            combined[i, j, 0],
            [process_square, process_ramp, lambda a, b, c, s, d, e: s],
            combined,
            i,
            j,
            slots,
            dx,
            dy,
        )
        return slots

    def __bfs(
        i: jnp.int32,
        j: jnp.int32,
        ind: jnp.int32,
        visited: jax.Array,
        spawnable: jax.Array,
        combined: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        capacity = combined.shape[0] * combined.shape[1]
        queue = jnp.full((capacity, 2), -1)
        front, rear, size = 0, 0, 0

        def __enqueue(
            i: jnp.int32,
            j: jnp.int32,
            rear: jnp.int32,
            queue: jax.Array,
            size: jnp.int32,
        ) -> tuple[jnp.int32, jax.Array, jnp.int32]:
            queue = queue.at[rear].set(jnp.array([i, j]))
            rear = (rear + 1) % capacity
            size += 1
            return rear, queue, size

        def __dequeue(
            front: jnp.int32, queue: jax.Array, size: jnp.int32
        ) -> tuple[jax.Array, jax.Array, jnp.int32]:
            res = queue[front]
            front = (front + 1) % capacity
            size -= 1
            return res, front, size

        visited = visited.at[i, j].set(True)
        rear, queue, size = __enqueue(i, j, rear, queue, size)

        def body_fun(
            args: tuple[
                jax.Array, jnp.int32, jnp.int32, jnp.int32, jax.Array, jax.Array
            ],
        ) -> tuple[jax.Array, jnp.int32, jnp.int32, jnp.int32, jax.Array, jax.Array]:
            queue, front, rear, size, visited, spawnable = args
            item, front, size = __dequeue(front, queue, size)
            x, y = item.astype(jnp.int32)

            # PROCESS
            spawnable = spawnable.at[x, y].set(ind)

            # Find next nodes
            def process_adj(
                carry: tuple[jax.Array, jnp.int32, jax.Array, jnp.int32, jax.Array],
                node: jax.Array,
            ) -> tuple[
                tuple[jax.Array, jnp.int32, jax.Array, jnp.int32, jax.Array], None
            ]:
                p, q = node

                visited, rear, queue, size, combined = carry

                def process_node(
                    visited: jax.Array,
                    rear: jnp.int32,
                    queue: jax.Array,
                    size: jnp.int32,
                ) -> tuple[jax.Array, jnp.int32, jax.Array, jnp.int32]:
                    visited = visited.at[p, q].set(True)
                    rear, queue, size = __enqueue(p, q, rear, queue, size)
                    return visited, rear, queue, size

                def process_node_identity(
                    visited: jax.Array,
                    rear: jnp.int32,
                    queue: jax.Array,
                    size: jnp.int32,
                ) -> tuple[jax.Array, jnp.int32, jax.Array, jnp.int32]:
                    return visited, rear, queue, size

                visited, rear, queue, size = jax.lax.cond(
                    jnp.all(
                        jnp.array([p >= 0, q >= 0, jnp.logical_not(visited[p, q])])
                    ),
                    process_node,
                    process_node_identity,
                    visited,
                    rear,
                    queue,
                    size,
                )
                return (visited, rear, queue, size, combined), None

            (visited, rear, queue, size, _), _ = jax.lax.scan(
                process_adj,
                (visited, rear, queue, size, combined),
                __adj_jit(x, y, combined),
            )

            return queue, front, rear, size, visited, spawnable

        _, _, _, _, visited, spawnable = jax.lax.while_loop(
            lambda args: args[3] > 0,
            body_fun,
            (queue, front, rear, size, visited, spawnable),
        )

        return visited, spawnable

    def scan_body(
        carry: tuple[jax.Array, jax.Array], ind: jnp.int32
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        visited, spawnable = carry
        i = ind // w
        j = ind % w
        visited, spawnable = jax.lax.cond(
            jnp.logical_not(visited.at[i, j].get()),
            lambda x: __bfs(i, j, i * w + j, x[0], x[1], combined),
            lambda x: x,
            (visited, spawnable),
        )
        return (visited, spawnable), None

    (visited, spawnable), _ = jax.lax.scan(
        scan_body, (visited, spawnable), jnp.arange(w * h)
    )

    spawnable = jnp.where(
        spawnable == jnp.argmax(jnp.bincount(spawnable.flatten(), length=w * h)), 1, 0
    )
    return spawnable
