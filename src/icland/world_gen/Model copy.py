"""This file contains the base Model class for WaveFunctionCollapse and helper functions."""

import math

# import random
from enum import Enum
import jax
import jax.numpy as jnp
from flax import struct
from typing import Callable, List, Any
import random


# def random_index_from_distribution(distribution, rand_value):
#     """Select an index from 'distribution' proportionally to the values in 'distribution'.
    
#     'rand_value' is a random float in [0,1).

#     Returns:
#         index (int) chosen according to the weights in 'distribution'.
#         If the sum of 'distribution' is 0, returns -1 as an error code.
#     """
#     total = sum(distribution)
#     if total <= 0:
#         return -1
#     cumulative = 0.0
#     for i, w in enumerate(distribution):
#         cumulative += w
#         if rand_value * total <= cumulative:
#             return i
#     return len(distribution) - 1  # Fallback if floating-point issues occur

@jax.jit
def random_index_from_distribution(distribution, rand_value):
    """Select an index from 'distribution' proportionally to the values in 'distribution'.

    Args:
        distribution: 1D array-like, weights for each index.
        rand_value: Random float in [0, 1).

    Returns:
        Index (int) chosen according to the weights in 'distribution'.
        If the sum of 'distribution' is 0, returns -1 as an error code.
    """
    # Compute the total of the distribution
    total = jnp.sum(distribution)
    
    # Handle the case where the total is zero
    def handle_zero_total(_):
        return -1

    def handle_nonzero_total(_):
        # Compute the cumulative sum
        cumulative_distribution = jnp.cumsum(distribution)

        # Find the index where the condition is satisfied
        condition = rand_value * total <= cumulative_distribution
        index = jnp.argmax(condition)  # First True index
        return index

    # Use lax.cond to handle the two cases
    result = jax.lax.cond(
        total <= 0,                    # Condition: total <= 0
        handle_zero_total,             # If True: return -1
        handle_nonzero_total,          # If False: compute the index
        None                           # Dummy argument for the functions
    )
    return result


class Heuristic(Enum):
    """Enum for the heuristic selection in WaveFunctionCollapse."""

    ENTROPY = 1
    MRV = 2
    SCANLINE = 3


@struct.dataclass
class Model:
    """Base Model class for WaveFunctionCollapse algorithm."""

    # Basic config
    MX: int
    MY: int
    N: int
    periodic: bool
    heuristic: Heuristic
    ground: bool

    T: int  # number of possible tile/pattern indices

    # Core arrays
    stacksize: int  # how many elements in the stack are currently valid
    propagator: tuple  # or a jax.Array of shape (4, T, ?) - depends on usage
    wave: jax.Array  # shape: (MX*MY, T), dtype=bool
    compatible: jax.Array  # shape: (MX*MY, T, 4), dtype=int
    observed: jax.Array  # shape: (MX*MY, ), dtype=int
    stack: jax.Array  # shape: (stack_capacity, 2), for (i, t)

    # Weights
    weights: jax.Array     # shape: (T,)
    weight_log_weights: jax.Array     # shape: (T,)

    # Summaries for each cell
    sums_of_ones: jax.Array  # shape: (MX*MY,)
    sums_of_weights: jax.Array  # shape: (MX*MY,)
    sums_of_weight_log_weights: jax.Array   # shape: (MX*MY,)
    entropies: jax.Array  # shape: (MX*MY,)

    # Precomputed sums for the entire set of patterns
    sum_of_weights: float
    sum_of_weight_log_weights: float
    starting_entropy: float

    # Because SCANLINE uses an incremental pointer
    observed_so_far: int

    propagator: jax.Array
    distribution: jax.Array

    dx: List[int] = struct.field(pytree_node=False, default=(-1, 0, 1, 0))
    dy: List[int] = struct.field(pytree_node=False, default=(0, 1, 0, -1))
    opposite: List[int] = struct.field(pytree_node=False, default=(2, 3, 0, 1))

    def run(self, seed, limit):
        """Run the WaveFunctionCollapse algorithm with the given seed and iteration limit."""
        if self.wave is None:
            self.init()

        self.clear()
        
        key = jax.random.PRNGKey(seed)
        steps = 0
        rng = random.Random(seed)

        while True:
            if limit >= 0 and steps >= limit:
                break
            steps += 1

            node = self.next_unobserved_node()
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
    
    @jax.jit
    def clear(self):
        """Resets wave and compatibility to allow all patterns at all cells, then optionally applies 'ground' constraints."""
        wave = jnp.ones((self.MX * self.MY, self.T), dtype=bool)
        compatible = jnp.zeros((self.MX * self.MY, self.T, 4), dtype=int)

        # Set each direction's compatibility count to the number of patterns
        # for i in range(self.MX * self.MY):
        #     for t in range(self.T):
        #         for d in range(4):
        #             compatible = compatible.at[i, t, d].set(
        #                 len(self.propagator[self.opposite[d]][t])
        #             )

        # Set each direction's compatibility count to the number of patterns
        def set_compatible(i, t):
            """ Helper function to calculate compatible values for each cell. """
            return jnp.array([len(self.propagator[self.opposite[d]][t]) for d in range(4)])

        # Use jax.vmap to vectorize the compatibility setting
        compatible = jax.vmap(lambda i, t: set_compatible(i, t))(jnp.arange(self.MX * self.MY), jnp.arange(self.T))

        sums_of_ones = jnp.full(self.MX * self.MY, self.T, dtype=int)
        sums_of_weights = jnp.full(self.MX * self.MY, self.sum_of_weights, dtype=float)
        sums_of_weight_log_weights = jnp.full(
            self.MX * self.MY, self.sum_of_weight_log_weights, dtype=float
        )
        entropies = jnp.full(self.MX * self.MY, self.starting_entropy, dtype=float)
        observed = jnp.full(self.MX * self.MY, -1, dtype=int)

        # Apply 'ground' constraints if needed
        if self.ground:
            # for x in range(self.MX):
            #     for t in range(self.T - 1):
            #         wave = wave.at[x + (self.MY - 1) * self.MX, t].set(False)
            #     for y in range(self.MY - 1):
            #         wave = wave.at[x + y * self.MX, self.T - 1].set(False)

            # Apply constraints to 'ground' values (last row and column)
            ground_wave = wave.at[self.MX * (self.MY - 1) + jnp.arange(self.T - 1)].set(False)
            ground_wave = ground_wave.at[jnp.arange(self.MX * (self.MY - 1)), self.T - 1].set(False)

            # Perform propagation after applying constraints
            new_model = self.replace(
                wave=ground_wave,
                compatible=compatible,
                sums_of_ones=sums_of_ones,
                sums_of_weights=sums_of_weights,
                sums_of_weight_log_weights=sums_of_weight_log_weights,
                entropies=entropies,
                observed=observed,
            ).propagate()
            return new_model

        return self.replace(
            wave=wave,
            compatible=compatible,
            sums_of_ones=sums_of_ones,
            sums_of_weights=sums_of_weights,
            sums_of_weight_log_weights=sums_of_weight_log_weights,
            entropies=entropies,
            observed=observed,
        )

    
    # Not yet jit-able
    def observe(self, node, rng):
        """Collapses the wave at 'node' by picking a pattern index according to weights distribution.

        Then bans all other patterns at that node.
        """
        w = self.wave[node]
        
        # Prepare distribution of patterns that are still possible
        for t in range(self.T):
            self.distribution[t] = self.weights[t] if w[t] else 0.0
        
        r = random_index_from_distribution(self.distribution, rng.random())

        # Ban any pattern that isn't the chosen one
        for t in range(self.T):
            # If wave[node][t] != (t == r) => ban it
            if w[t] != (t == r):
                self.ban(node, t)

    def next_unobserved_node(self):
        """Get the next unobserved node."""
        unobserved = jnp.where(self.wave.sum(axis=1) > 0)[0]
        # Scan bool, if all false reutnr -1, else return first
        return unobserved[0] if unobserved.size > 0 else -1
        # min_entropy = 1e4
        # argmin = -1
        # for i in range(len(self.wave)):
        #     if self.sumsOfOnes[i] <= 1:
        #         continue
        #     entropy = self.entropies[i]
        #     noise = 1e-6 * jax.random.uniform(rng_key)
        #     if entropy + noise < min_entropy:
        #         min_entropy = entropy + noise
        #         argmin = i
        # return argmin
        
    def propagate(self):
        """Propagates constraints across the wave."""
        
        while self.stacksize > 0:
            i1, t1 = self.stack[self.stacksize - 1]
            self.stacksize -= 1

            x1 = i1 % self.MX
            y1 = i1 // self.MX

            for d in range(4):
                x2 = x1 + self.dx[d]
                y2 = y1 + self.dy[d]

                if not self.periodic and (
                    x2 < 0 or y2 < 0 or x2 >= self.MX or y2 >= self.MY
                ):
                    continue

                if x2 < 0:
                    x2 += self.MX
                elif x2 >= self.MX:
                    x2 -= self.MX
                if y2 < 0:
                    y2 += self.MY
                elif y2 >= self.MY:
                    y2 -= self.MY

                i2 = x2 + y2 * self.MX
                p = self.propagator[d][t1]

                for t2 in p:
                    comp = self.compatible[i2, t2, d]
                    comp -= 1
                    if comp == 0:
                        self.ban(i2, t2)

        return self

    def ban(self, i, t):
        """Bans pattern t at cell i. Updates wave, compatibility, sums_of_ones, entropies, and stack."""
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

        # Update sums_of_ones, sums_of_weights, sums_of_weight_log_weights, entropies
        self.sums_of_ones[i] -= 1
        self.sums_of_weights[i] -= self.weights[t]
        self.sums_of_weight_log_weights[i] -= self.weightLogWeights[t]

        sum_w = self.sums_of_weights[i]
        self.entropies[i] = (
            math.log(sum_w) - (self.sums_of_weight_log_weights[i] / sum_w)
            if sum_w > 0
            else 0.0
        )

    def clear(self):
        """Resets wave and compatibility to allow all patterns at all cells, then optionally applies 'ground' constraints."""
        wave = jnp.ones((self.MX * self.MY, self.T), dtype=bool)
        compatible = jnp.zeros((self.MX * self.MY, self.T, 4), dtype=int)

        # Set each direction's compatibility count to the number of patterns
        for i in range(self.MX * self.MY):
            for t in range(self.T):
                for d in range(4):
                    compatible = compatible.at[i, t, d].set(
                        len(self.propagator[self.opposite[d]][t])
                    )

        sums_of_ones = jnp.full(self.MX * self.MY, self.T, dtype=int)
        sums_of_weights = jnp.full(self.MX * self.MY, self.sum_of_weights, dtype=float)
        sums_of_weight_log_weights = jnp.full(
            self.MX * self.MY, self.sum_of_weight_log_weights, dtype=float
        )
        entropies = jnp.full(self.MX * self.MY, self.starting_entropy, dtype=float)
        observed = jnp.full(self.MX * self.MY, -1, dtype=int)

        # Apply 'ground' constraints if needed
        if self.ground:
            for x in range(self.MX):
                for t in range(self.T - 1):
                    wave = wave.at[x + (self.MY - 1) * self.MX, t].set(False)
                for y in range(self.MY - 1):
                    wave = wave.at[x + y * self.MX, self.T - 1].set(False)

            # Perform propagation after applying constraints
            new_model = self.replace(
                wave=wave,
                compatible=compatible,
                sums_of_ones=sums_of_ones,
                sums_of_weights=sums_of_weights,
                sums_of_weight_log_weights=sums_of_weight_log_weights,
                entropies=entropies,
                observed=observed,
            ).propagate()
            return new_model

        return self.replace(
            wave=wave,
            compatible=compatible,
            sums_of_ones=sums_of_ones,
            sums_of_weights=sums_of_weights,
            sums_of_weight_log_weights=sums_of_weight_log_weights,
            entropies=entropies,
            observed=observed,
        )

    def save(self, filename):
        """Raises NotImplementedError to ensure its subclass will provide an implementation."""
        raise NotImplementedError("Must be implemented in a subclass.")

def init(
    width: int,
    height: int,
    N: int,
    periodic: bool,
    heuristic: Heuristic,
    weights_py: list[float],
    propagator_py,  # some Python structure describing adjacency
    ground: bool = False,
) -> Model:

    T = len(weights_py)
    wave_init = jnp.ones((width * height, T), dtype=bool)

    # Convert Python lists to jnp arrays:
    weights = jnp.array(weights_py, dtype=jnp.float32)

    weight_log_weights = weights * jnp.log(jnp.where(weights > 0, weights, 1e-9))
    sum_of_weights = jnp.sum(weights)
    sum_of_weight_log_weights = jnp.sum(weight_log_weights)
    starting_entropy = jnp.log(sum_of_weights) - sum_of_weight_log_weights / sum_of_weights

    # Example shape for 'compatible': (width*height, T, 4). Initialize to zero or some default.
    compatible_init = jnp.zeros((width * height, T, 4), dtype=jnp.int32)

    propagator_jax = jnp.zeros((4, T, 10)) # CHANGE

    # Observed array init
    observed_init = -jnp.ones((width * height,), dtype=jnp.int32)

    # Summaries
    sums_of_ones_init = jnp.full((width * height,), T, dtype=jnp.int32)
    sums_of_weights_init = jnp.full((width * height,), sum_of_weights, dtype=jnp.float32)
    sums_of_weight_log_weights_init = jnp.full(
        (width * height,), sum_of_weight_log_weights, dtype=jnp.float32
    )
    entropies_init = jnp.full((width * height,), starting_entropy, dtype=jnp.float32)\
        
    distribution_init = jnp.zeros((T, ), dtype=jnp.float32)

    # Initialize the stack array
    stack_init = jnp.zeros((len(wave_init) * T, 2), dtype=jnp.int32)
    stacksize_init = 0

    return Model(
        MX=width,
        MY=height,
        N=N,
        T=T,
        periodic=periodic,
        heuristic=heuristic,
        ground=ground,

        wave=wave_init,
        compatible=compatible_init,
        propagator=propagator_jax,
        observed=observed_init,
        stack=stack_init,
        stacksize=stacksize_init,

        weights=weights,
        weight_log_weights=weight_log_weights,
        sums_of_ones=sums_of_ones_init,
        sums_of_weights=sums_of_weights_init,
        sums_of_weight_log_weights=sums_of_weight_log_weights_init,
        entropies=entropies_init,

        sum_of_weights=float(sum_of_weights),
        sum_of_weight_log_weights=float(sum_of_weight_log_weights),
        starting_entropy=float(starting_entropy),
        distribution=distribution_init,
        observed_so_far=0
    )


model = init(3, 3, 2, False, Heuristic.ENTROPY, [1.0, 1.0], None)    
model.run(0, 1000)

    # def observe(self, rng_key):
    #     """Collapses the wave at a node by selecting a pattern index."""
    #     node = self.next_unobserved_node(rng_key)
    #     if node == -1:
    #         return self

    #     wave = self.wave.at[node].set(False)
    #     distribution = self.weights * wave[node]
    #     r = random_index_from_distribution(distribution, jax.random.uniform(rng_key))
    #     wave = wave.at[node, r].set(True)

    #     return self.replace(wave=wave)

    # def next_unobserved_node(self, rng_key):
    #     """Selects the next cell to observe based on heuristic."""
    #     min_entropy = 1e4
    #     argmin = -1
    #     for i in range(len(self.wave)):
    #         if self.sumsOfOnes[i] <= 1:
    #             continue
    #         entropy = self.entropies[i]
    #         noise = 1e-6 * jax.random.uniform(rng_key)
    #         if entropy + noise < min_entropy:
    #             min_entropy = entropy + noise
    #             argmin = i
    #     return argmin

    # def propagate(self):
    #     """Propagates constraints across the wave."""
    #     while self.stacksize > 0:
    #         i1, t1 = self.stack[self.stacksize - 1]
    #         self.stacksize -= 1

    #         x1 = i1 % self.MX
    #         y1 = i1 // self.MX

    #         for d in range(4):
    #             x2 = x1 + self.dx[d]
    #             y2 = y1 + self.dy[d]

    #             if not self.periodic and (
    #                 x2 < 0 or y2 < 0 or x2 >= self.MX or y2 >= self.MY
    #             ):
    #                 continue

    #             if x2 < 0:
    #                 x2 += self.MX
    #             elif x2 >= self.MX:
    #                 x2 -= self.MX
    #             if y2 < 0:
    #                 y2 += self.MY
    #             elif y2 >= self.MY:
    #                 y2 -= self.MY

    #             i2 = x2 + y2 * self.MX
    #             p = self.propagator[d][t1]

    #             for t2 in p:
    #                 comp = self.compatible[i2, t2, d]
    #                 comp -= 1
    #                 if comp == 0:
    #                     self.ban(i2, t2)

    #     return self

    # def ban(self, i, t):
    #     """Bans pattern t at cell i. Updates wave, compatibility, sumsOfOnes, entropies, and stack."""
    #     # If wave[i][t] is already false, do nothing
    #     if not self.wave[i][t]:
    #         return

    #     self.wave[i][t] = False

    #     # Zero-out the compatibility in all directions for pattern t at cell i
    #     comp = self.compatible[i][t]
    #     for d in range(4):
    #         comp[d] = 0

    #     # Push (i, t) onto stack
    #     if self.stacksize < len(self.stack):
    #         self.stack[self.stacksize] = (i, t)
    #     else:
    #         self.stack.append((i, t))
    #     self.stacksize += 1

    #     # Update sumsOfOnes, sumsOfWeights, sumsOfWeightLogWeights, entropies
    #     self.sumsOfOnes[i] -= 1
    #     self.sumsOfWeights[i] -= self.weights[t]
    #     self.sumsOfWeightLogWeights[i] -= self.weightLogWeights[t]

    #     sum_w = self.sumsOfWeights[i]
    #     self.entropies[i] = (
    #         math.log(sum_w) - (self.sumsOfWeightLogWeights[i] / sum_w)
    #         if sum_w > 0
    #         else 0.0
    #     )

    # def clear(self):
    #     """Resets wave and compatibility to allow all patterns at all cells, then optionally applies 'ground' constraints."""
    #     wave = jnp.ones((self.MX * self.MY, self.T), dtype=bool)
    #     compatible = jnp.zeros((self.MX * self.MY, self.T, 4), dtype=int)

    #     # Set each direction's compatibility count to the number of patterns
    #     for i in range(self.MX * self.MY):
    #         for t in range(self.T):
    #             for d in range(4):
    #                 compatible = compatible.at[i, t, d].set(
    #                     len(self.propagator[self.opposite[d]][t])
    #                 )

    #     sumsOfOnes = jnp.full(self.MX * self.MY, self.T, dtype=int)
    #     sumsOfWeights = jnp.full(self.MX * self.MY, self.sumOfWeights, dtype=float)
    #     sumsOfWeightLogWeights = jnp.full(
    #         self.MX * self.MY, self.sumOfWeightLogWeights, dtype=float
    #     )
    #     entropies = jnp.full(self.MX * self.MY, self.startingEntropy, dtype=float)
    #     observed = jnp.full(self.MX * self.MY, -1, dtype=int)

    #     # Apply 'ground' constraints if needed
    #     if self.ground:
    #         for x in range(self.MX):
    #             for t in range(self.T - 1):
    #                 wave = wave.at[x + (self.MY - 1) * self.MX, t].set(False)
    #             for y in range(self.MY - 1):
    #                 wave = wave.at[x + y * self.MX, self.T - 1].set(False)

    #         # Perform propagation after applying constraints
    #         new_model = self.replace(
    #             wave=wave,
    #             compatible=compatible,
    #             sumsOfOnes=sumsOfOnes,
    #             sumsOfWeights=sumsOfWeights,
    #             sumsOfWeightLogWeights=sumsOfWeightLogWeights,
    #             entropies=entropies,
    #             observed=observed,
    #         ).propagate()
    #         return new_model

    #     return self.replace(
    #         wave=wave,
    #         compatible=compatible,
    #         sumsOfOnes=sumsOfOnes,
    #         sumsOfWeights=sumsOfWeights,
    #         sumsOfWeightLogWeights=sumsOfWeightLogWeights,
    #         entropies=entropies,
    #         observed=observed,
    #     )

    # def save(self, filename):
    #     """Raises NotImplementedError to ensure its subclass will provide an implementation."""
    #     raise NotImplementedError("Must be implemented in a subclass.")