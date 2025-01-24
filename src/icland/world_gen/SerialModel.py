"""This file contains the base SerialModel class for WaveFunctionCollapse and helper functions."""

import math
import random
from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
from converter import create_world, export_stl
from tile_data import NUM_ACTIONS, PROPAGATOR, TILECODES, WEIGHTS
from XMLReader import TileType


def random_index_from_distribution(distribution, rand_value):
    """Select an index from 'distribution' proportionally to the values in 'distribution'.

    'rand_value' is a random float in [0,1).

    Returns:
        index (int) chosen according to the weights in 'distribution'.
        If the sum of 'distribution' is 0, returns -1 as an error code.
    """
    total = sum(distribution)
    if total <= 0:
        return -1
    cumulative = 0.0
    for i, w in enumerate(distribution):
        cumulative += w
        if rand_value * total <= cumulative:
            return i
    return len(distribution) - 1  # Fallback if floating-point issues occur


class Heuristic(Enum):
    """Enum for the heuristic selection in WaveFunctionCollapse."""

    ENTROPY = 1
    MRV = 2
    SCANLINE = 3


class SerialModel:
    """Base SerialModel class for WaveFunctionCollapse algorithm."""

    # Class-wide (static) arrays, analogous to the protected static int[] in C#
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    opposite = [2, 3, 0, 1]  # Opposite directions for 0->2, 1->3, 2->0, 3->1

    def __init__(self, width, height, N, periodic, heuristic):
        """Constructor for the SerialModel class."""
        self.heuristic = heuristic

        self.wave = None

        self.propagator = None
        self.compatible = None
        self.observed = None

        self.stack = None
        self.stacksize = 0
        self.observedSoFar = 0

        self.MX = width
        self.MY = height
        self.T = 0  # number of possible tile/pattern indices
        self.N = N

        self.periodic = periodic
        self.ground = False

        self.weights = None
        self.weightLogWeights = None
        self.distribution = None

        self.sumsOfOnes = None
        self.sumsOfWeights = None
        self.sumsOfWeightLogWeights = None
        self.startingEntropy = 0.0

        self.sumOfWeights = 0.0
        self.sumOfWeightLogWeights = 0.0
        self.entropies = None

    def init(self):
        """Initialise variables."""
        self.wave = [[False] * self.T for _ in range(self.MX * self.MY)]
        self.compatible = [
            [[0] * 4 for _ in range(self.T)] for _ in range(self.MX * self.MY)
        ]

        self.distribution = [0.0] * self.T

        self.observed = [-1] * (self.MX * self.MY)

        self.weightLogWeights = [0.0] * self.T
        self.sumOfWeights = 0.0
        self.sumOfWeightLogWeights = 0.0

        for t in range(self.T):
            self.weightLogWeights[t] = self.weights[t] * math.log(self.weights[t])
            self.sumOfWeights += self.weights[t]
            self.sumOfWeightLogWeights += self.weightLogWeights[t]

        self.startingEntropy = math.log(self.sumOfWeights) - (
            self.sumOfWeightLogWeights / self.sumOfWeights
        )

        self.sumsOfOnes = [0] * (self.MX * self.MY)
        self.sumsOfWeights = [0.0] * (self.MX * self.MY)
        self.sumsOfWeightLogWeights = [0.0] * (self.MX * self.MY)
        self.entropies = [0.0] * (self.MX * self.MY)

        self.stack = []
        self.stacksize = 0

    def run(self, seed, limit):
        """Run the WaveFunctionCollapse algorithm with the given seed and iteration limit."""
        if self.wave is None:
            self.init()

        self.clear()

        rng = random.Random(seed)

        steps = 0
        while True:
            if limit >= 0 and steps >= limit:
                break
            steps += 1

            node = self.next_unobserved_node(rng)
            if node >= 0:
                self.observe(node, rng)
                success = self.propagate()
                if not success:
                    return False
            else:
                for i in range(len(self.wave)):
                    for t in range(self.T):
                        if self.wave[i][t]:
                            self.observed[i] = t
                            break
                return True

        return True

    def next_unobserved_node(self, rng):
        """Selects the next cell to observe according to the chosen heuristic (Scanline, Entropy, or MRV).

        Returns:
            index (int) of the chosen cell in [0, MX*MY), or -1 if all cells are determined.
        """
        # Handle the SCANLINE heuristic
        if self.heuristic == Heuristic.SCANLINE:
            for i in range(self.observedSoFar, len(self.wave)):
                # skip if out of range (non-periodic boundary constraints)
                if (not self.periodic) and (
                    (i % self.MX) + self.N > self.MX
                    or (i // self.MX) + self.N > self.MY
                ):
                    continue
                if self.sumsOfOnes[i] > 1:
                    self.observedSoFar = i + 1
                    return i
            return -1

        # Handle ENTROPY or MRV
        min_entropy = 1e4
        argmin = -1
        for i in range(len(self.wave)):
            # skip out-of-range if non-periodic
            if (not self.periodic) and (
                (i % self.MX) + self.N > self.MX or (i // self.MX) + self.N > self.MY
            ):
                continue

            remaining_values = self.sumsOfOnes[i]

            # ENTROPY -> we look at entropies[i], MRV -> we look at remaining_values
            if self.heuristic == Heuristic.ENTROPY:
                entropy = self.entropies[i]
            else:  # MRV
                entropy = float(remaining_values)

            if remaining_values > 1 and entropy <= min_entropy:
                # small noise to break ties
                noise = 1e-6 * rng.random()
                if entropy + noise < min_entropy:
                    min_entropy = entropy + noise
                    argmin = i
        return argmin

    def observe(self, node, rng):
        """Collapses the wave at 'node' by picking a pattern index according to weights distribution.

        Then _bans all other patterns at that node.
        """
        w = self.wave[node]

        # Prepare distribution of patterns that are still possible
        for t in range(self.T):
            self.distribution[t] = self.weights[t] if w[t] else 0.0

        r = random_index_from_distribution(self.distribution, rng.random())
        # r = random_index_from_distribution(self.distribution, 0.07805848121643066)

        # Ban any pattern that isn't the chosen one
        for t in range(self.T):
            # If wave[node][t] != (t == r) => _ban it
            if w[t] != (t == r):
                self._ban(node, t)

    def propagate(self):
        """Propagate the wave function collapse constraints.

        Returns:
            True if propagation completed successfully (no contradictions found).
            (In the original snippet, it returns sumsOfOnes[0] > 0, which is suspicious,
            but we'll keep the same logic.)
        """
        while self.stacksize > 0:
            i1, t1 = self.stack[self.stacksize - 1]
            self.stacksize -= 1

            x1 = i1 % self.MX
            y1 = i1 // self.MX

            for d in range(4):
                x2 = x1 + self.dx[d]
                y2 = y1 + self.dy[d]

                # Boundary checks if not periodic
                if (not self.periodic) and (
                    x2 < 0 or y2 < 0 or x2 + self.N > self.MX or y2 + self.N > self.MY
                ):
                    continue

                # Wrap around if periodic
                if x2 < 0:
                    x2 += self.MX
                elif x2 >= self.MX:
                    x2 -= self.MX
                if y2 < 0:
                    y2 += self.MY
                elif y2 >= self.MY:
                    y2 -= self.MY

                i2 = x2 + y2 * self.MX
                p = self.propagator[d][t1]  # Patterns that can follow t1 in direction d
                compat = self.compatible[i2]

                # For each pattern t2 that can follow t1, decrease its compatibility count
                for t2 in p:
                    comp = compat[int(t2)]
                    comp[d] -= 1
                    if comp[d] == 0:
                        # If no compatibility remains in direction d, _ban this pattern
                        self._ban(i2, t2)

        # The original snippet returns sumsOfOnes[0] > 0.
        # This is not typical for full WFC, but we'll preserve it.
        return self.sumsOfOnes[0] > 0

    def _ban(self, i, t1):
        t = int(t1)
        """Bans pattern t at cell i. Updates wave, compatibility, sumsOfOnes, entropies, and stack."""
        # If wave[i][t] is already false, do nothing
        if not self.wave[i][t]:
            return

        self.wave[i][t] = False

        # Zero-out the compatibility in all directions for pattern t at cell i
        comp = self.compatible[i][t]
        for d in range(4):
            comp[d] = 0

        # Push (i, t) onto stack
        if self.stacksize < len(self.stack):
            self.stack[self.stacksize] = (i, t)
        else:
            self.stack.append((i, t))
        self.stacksize += 1

        # Update sumsOfOnes, sumsOfWeights, sumsOfWeightLogWeights, entropies
        self.sumsOfOnes[i] -= 1
        self.sumsOfWeights[i] -= self.weights[t]
        self.sumsOfWeightLogWeights[i] -= self.weightLogWeights[t]

        sum_w = self.sumsOfWeights[i]
        entropy_res = (
            math.log(sum_w) - (self.sumsOfWeightLogWeights[i] / sum_w)
            if sum_w > 0
            else 0.0
        )
        # print("Next entropy:", entropy_res)
        self.entropies[i] = entropy_res

    def clear(self):
        """Resets wave and compatibility to allow all patterns at all cells, then optionally applies 'ground' constraints."""
        for i in range(len(self.wave)):
            for t in range(self.T):
                self.wave[i][t] = True
                # Set each direction's compatibility count to the number of patterns
                # that can appear in the cell adjacent in that direction.
                # The code does: compatible[i][t][d] = propagator[opposite[d]][t].Length
                for d in range(4):
                    # Because we do not show how 'propagator' is fully set up,
                    # we assume it is an array-of-arrays of pattern-lists.
                    self.compatible[i][t][d] = len(self.propagator[self.opposite[d]][t])

            self.sumsOfOnes[i] = len(self.weights)
            self.sumsOfWeights[i] = self.sumOfWeights
            self.sumsOfWeightLogWeights[i] = self.sumOfWeightLogWeights
            self.entropies[i] = self.startingEntropy
            self.observed[i] = -1

        self.observedSoFar = 0

        if self.ground:
            # Force the bottom row to be the "ground" pattern
            for x in range(self.MX):
                # Ban all patterns except the last one (T-1) in the bottom row
                for t in range(self.T - 1):
                    self._ban(x + (self.MY - 1) * self.MX, t)
                # For other rows above the bottom, _ban the ground pattern
                for y in range(self.MY - 1):
                    self._ban(x + y * self.MX, self.T - 1)

            # Propagate after applying ground constraints
            self.propagate()

    def _export(self, tilemap):
        observed_reshaped = jnp.reshape(jnp.array(self.observed), (self.MX, self.MY))
        # Combine observed state and tile information using jax.vmap
        # Apply combine function using vmap for vectorization
        combined = jax.vmap(lambda x: tilemap.at[x].get())(observed_reshaped)
        return jnp.array(self.observed), combined


@partial(jax.jit, static_argnums=[2])
def _sample_spawn_points(key, spawn_map, num_objects):
    # Key will be consumed.
    flat_tilemap = spawn_map.flatten()
    nonzero_indices = jnp.where(
        flat_tilemap != 0, size=flat_tilemap.shape[0], fill_value=-1
    )[0]

    def run_once(key):
        def pick(item):
            _, key = item
            key, subkey = jax.random.split(key)
            choice = jax.random.choice(subkey, nonzero_indices)
            return choice, key

        random_index, key = jax.lax.while_loop(lambda x: x[0] < 0, pick, (-1, key))

        # Convert the flat index back to 2D coordinates
        row = random_index // spawn_map.shape[0]
        col = random_index % spawn_map.shape[0]

        return jnp.array([row, col])

    keys = jax.random.split(key, num_objects)

    return jax.vmap(run_once)(keys)


@jax.jit
def _get_spawn_map(combined):
    combined = combined.astype(jnp.int32)
    w, h = combined.shape[0], combined.shape[1]

    # Initialize arrays with JAX functions
    visited = jax.lax.full((w, h), False, dtype=jnp.bool)
    spawnable = jax.lax.full((w, h), 0, dtype=jnp.int32)

    @jax.jit
    def __adj_jit(i, j, combined):
        slots = jnp.full((4, 2), -1)
        dx = jnp.array([-1, 0, 1, 0])
        dy = jnp.array([0, 1, 0, -1])

        def process_square(combined, i, j, slots, dx, dy):
            tile, r, f, level = combined[i, j]
            for d in range(4):
                x = i + dx[d]
                y = j + dy[d]

                def process_square_inner(slots):
                    q, r2, f2, l = combined[x, y]
                    return jax.lax.cond(
                        jnp.any(
                            jnp.array(
                                [
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.RAMP.value,
                                                r2 == (4 - d) % 4,
                                                f2 == level - 1,
                                                l == level,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.RAMP.value,
                                                r2 == (2 - d) % 4,
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

        def process_ramp(combined, i, j, slots, dx, dy):
            tile, r, f, level = combined[i, j]
            mask = jnp.where((r + 1) % 2 == 0, 1, 0)
            for d in range(4):
                x = i + dx[d]
                y = j + dy[d]

                def process_ramp_inner(slots):
                    q, r2, f2, l = combined[x, y]
                    return jax.lax.cond(
                        jnp.any(
                            jnp.array(
                                [
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.SQUARE.value,
                                                d == (2 - r) % 4,
                                                l == level,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.SQUARE.value,
                                                d == (4 - r) % 4,
                                                l == f,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.RAMP.value,
                                                d == (2 - r) % 4,
                                                r == r2,
                                                f2 == level,
                                            ]
                                        )
                                    ),
                                    jnp.all(
                                        jnp.array(
                                            [
                                                q == TileType.RAMP.value,
                                                d == (4 - r) % 4,
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

        return jax.lax.switch(
            combined[i, j, 0],
            [process_square, process_ramp, lambda a, b, c, s, d, e: s],
            combined,
            i,
            j,
            slots,
            dx,
            dy,
        )

    @jax.jit
    def __bfs(i, j, ind, visited, spawnable, combined):
        capacity = combined.shape[0] * combined.shape[1]
        queue = jnp.full((capacity, 2), -1)
        front, rear, size = 0, 0, 0

        @jax.jit
        def __enqueue(i, j, rear, queue, size):
            queue = queue.at[rear].set(jnp.array([i, j]))
            rear = (rear + 1) % capacity
            size += 1
            return rear, queue, size

        @jax.jit
        def __dequeue(front, queue, size):
            res = queue[front]
            front = (front + 1) % capacity
            size -= 1
            return res, front, size

        visited = visited.at[i, j].set(True)
        rear, queue, size = __enqueue(i, j, rear, queue, size)

        def body_fun(args):
            queue, front, rear, size, visited, spawnable = args
            item, front, size = __dequeue(front, queue, size)
            x, y = item.astype(jnp.int32)

            # PROCESS
            spawnable = spawnable.at[x, y].set(ind)

            # Find next nodes
            def process_adj(carry, node):
                p, q = node

                visited, rear, queue, size, combined = carry

                def process_node(visited, rear, queue, size):
                    visited = visited.at[p, q].set(True)
                    rear, queue, size = __enqueue(p, q, rear, queue, size)
                    return visited, rear, queue, size

                def process_node_identity(visited, rear, queue, size):
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

    def scan_body(carry, ind):
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
        spawnable == jnp.argmax(jnp.bincount(spawnable.flatten(), length=1)), 1, 0
    )
    return spawnable

t, w, p, c = NUM_ACTIONS, WEIGHTS, PROPAGATOR, TILECODES
model = SerialModel(10, 10, 1, False, Heuristic.ENTROPY)
model.T = t
model.weights = w.tolist()
model.propagator = p.astype(int).tolist()
seed = 42
# model.init()
# model.clear()
# model.observe(0, 0.07805848121643066)
# model.propagate()
# print(model.wave)
# print(model.compatible)
# print(model.stack)
# print(model.stacksize)
# print(model.sumsOfOnes)
# print(model.sumsOfWeights)
# print(model.sumsOfWeightLogWeights)
# print(model.entropies)
# print(model.distribution)
model.run(seed, 1000)
obs, tilemap = model._export(c)

# reader.save(obs, model.MX, model.MY, "temp_2.png")
world = create_world(tilemap)
export_stl(world, "test.stl")
# spawnmap = _get_spawn_map(tilemap)
# res = _sample_spawn_points(jax.random.key(0), spawnmap, 10)
# print(res)
