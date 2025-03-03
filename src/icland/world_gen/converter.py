"""The code defines functions to generate block, ramp, and vertical ramp columns in a 3D world using JAX arrays and exports the generated mesh to an STL file."""

# import os
from functools import partial

import jax
import jax.numpy as jnp

from icland.world_gen.XMLReader import TileType


@partial(jax.jit, static_argnums=[2])
def sample_spawn_points(
    key: jax.Array, tilemap: jax.Array, num_objects: int = 1
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
            item: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            _, key = item
            key, subkey = jax.random.split(key)
            choice = jax.random.choice(subkey, nonzero_indices)
            return choice, key

        random_index, key = jax.lax.while_loop(
            lambda x: x[0] < 0, pick, (jnp.array(-1), key)
        )

        # Convert the flat index back to 2D coordinates
        row = random_index // spawn_map.shape[0]
        col = random_index % spawn_map.shape[0]

        return jnp.array(
            [row + 0.5, col + 0.5, tilemap[row, col, TILE_DATA_HEIGHT_INDEX] + 1]
        )

    keys = jax.random.split(key, num_objects)
    spawnpoints = jax.vmap(run_once)(keys)
    return spawnpoints


def __get_spawn_map(combined: jax.Array) -> jax.Array:  # pragma: no cover
    combined = combined.astype(int)
    w, h = combined.shape[0], combined.shape[1]

    # Initialize arrays with JAX functions
    TILE_DATA_SIZE = 4
    NUM_ROTATIONS = 4
    NUM_COORDS = 2
    visited = jax.lax.full((w, h), False, dtype=jnp.bool)
    spawnable = jax.lax.full((w, h), 0, dtype=int)

    def __adj_jit(i: int, j: int, combined: jax.Array) -> jax.Array:
        slots = jnp.full((TILE_DATA_SIZE, NUM_COORDS), -1)
        dx = jnp.array([-1, 0, 1, 0])
        dy = jnp.array([0, 1, 0, -1])

        def process_square(
            combined: jax.Array,
            i: int,
            j: int,
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
            i: int,
            j: int,
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
        i: jax.Array,
        j: jax.Array,
        ind: jax.Array,
        visited: jax.Array,
        spawnable: jax.Array,
        combined: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        capacity = combined.shape[0] * combined.shape[1]
        queue = jnp.full((capacity, 2), -1)
        front, rear, size = 0, 0, 0

        def __enqueue(
            i: jax.Array,
            j: jax.Array,
            rear: int,
            queue: jax.Array,
            size: int,
        ) -> tuple[int, jax.Array, int]:
            queue = queue.at[rear].set(jnp.array([i, j]))
            rear = (rear + 1) % capacity
            size += 1
            return rear, queue, size

        def __dequeue(
            front: int, queue: jax.Array, size: int
        ) -> tuple[jax.Array, int, int]:
            res = queue[front]
            front = (front + 1) % capacity
            size -= 1
            return res, front, size

        visited = visited.at[i, j].set(True)
        rear, queue, size = __enqueue(i, j, rear, queue, size)

        def body_fun(
            args: tuple[jax.Array, int, int, int, jax.Array, jax.Array],
        ) -> tuple[jax.Array, int, int, int, jax.Array, jax.Array]:
            queue, front, rear, size, visited, spawnable = args
            item, front, size = __dequeue(front, queue, size)
            x, y = item.astype(int)

            # PROCESS
            spawnable = spawnable.at[x, y].set(ind)

            # Find next nodes
            def process_adj(
                carry: tuple[jax.Array, int, jax.Array, int, jax.Array],
                node: jax.Array,
            ) -> tuple[tuple[jax.Array, int, jax.Array, int, jax.Array], None]:
                p, q = node

                visited, rear, queue, size, combined = carry

                def process_node(
                    visited: jax.Array,
                    rear: int,
                    queue: jax.Array,
                    size: int,
                ) -> tuple[jax.Array, int, jax.Array, int]:
                    visited = visited.at[p, q].set(True)
                    rear, queue, size = __enqueue(p, q, rear, queue, size)
                    return visited, rear, queue, size

                def process_node_identity(
                    visited: jax.Array,
                    rear: int,
                    queue: jax.Array,
                    size: int,
                ) -> tuple[jax.Array, int, jax.Array, int]:
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
        carry: tuple[jax.Array, jax.Array], ind: jax.Array
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
