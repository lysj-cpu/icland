"""This file contains the base Model class for WaveFunctionCollapse and helper functions."""

import math

# import random
from enum import Enum
import jax
import jax.numpy as jnp
from typing import Callable, List, Any, NamedTuple
import random
from flax import struct

from functools import partial
from JITModel import XMLReader

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


class ModelParams(NamedTuple):
    MX: int
    MY: int
    T: int
    propagator_length: int
    periodic: bool
    heuristic: int
    key: jax.Array

@struct.dataclass
class ModelX:
    """Base Model class for WaveFunctionCollapse algorithm."""

    # Basic config
    MX: int
    MY: int
    periodic: bool
    heuristic: int

    T: int  # number of possible tile/pattern indices

    # Core arrays
    stacksize: int  # how many elements in the stack are currently valid
    max_stacksize: int
    wave: jax.Array  # shape: (MX*MY, T), dtype=bool
    compatible: jax.Array  # shape: (MX*MY, T, 4), dtype=int
    observed: jax.Array  # shape: (MX*MY, ), dtype=int
    stack: jax.Array  # shape: (MX*MY, 2), for (i, t)

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

    propagator: jax.Array # (4, T, propagator_length)
    distribution: jax.Array # shape: (T, )

    key: jax.Array

    dx: List[int] = struct.field(pytree_node=False, default=(-1, 0, 1, 0)) # 4
    dy: List[int] = struct.field(pytree_node=False, default=(0, 1, 0, -1)) # 4
    opposite: List[int] = struct.field(pytree_node=False, default=(2, 3, 0, 1)) # 4


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

@jax.jit
def observe(model, node, rng):
    """Collapses the wave at 'node' by picking a pattern index according to weights distribution.

    Then bans all other patterns at that node.
    """
    w = model.wave.at[node].get()
    
    # Prepare distribution of patterns that are still possible
    distribution = model.distribution
    distribution = jnp.where(w, model.weights, jnp.zeros_like(model.weights))
    
    r = random_index_from_distribution(model.distribution, rng.random())

    # Ban any pattern that isn't the chosen one
    # If wave[node][t] != (t == r) => ban it
    process_ban = lambda i, m: jax.lax.cond(w.at[i].get() != (i == r), lambda x: ban(x, node, i), lambda x: x, m)
    return jax.lax.fori_loop(0, self.T, process_ban, model)

@jax.jit
def ban(model, i, t1):
    # Ban
    # wave compatible stack stacksize sums_one sums_weights sums_weight_log_weights weights weight_log_weights
    """Bans pattern t at cell i. Updates wave, compatibility, sums_of_ones, entropies, and stack."""
    
    t = t1.astype(int)
    condition_1 = jnp.logical_not(model.wave.at[i, t].get())
    identity = lambda x: x
    
    def process_ban(model):
        wave = model.wave
        wave = wave.at[i, t].set(False)

        # Zero-out the compatibility in all directions for pattern t at cell i
        compatible = model.compatible
        compatible = compatible.at[i, t, :].set(0)

        stack = model.stack
        stacksize = model.stacksize
        stack.at[stacksize].set(jnp.array([i, t]))
        stacksize += 1

        # Update sums_of_ones, sums_of_weights, sums_of_weight_log_weights, entropies
        sums_of_ones = model.sums_of_ones
        sums_of_ones = sums_of_ones.at[i].subtract(1)
        
        sums_of_weights = model.sums_of_weights
        sums_of_weights = sums_of_weights.at[i].subtract(model.weights.at[t].get())
        
        sums_of_weight_log_weights = model.sums_of_weight_log_weights
        sums_of_weight_log_weights = sums_of_weight_log_weights.at[i].subtract(model.weight_log_weights.at[t].get())

        sum_w = sums_of_weights.at[i].get()
        entropies = model.entropies
        entropies.at[i].set(jnp.where(sum_w > 0, jnp.log(sum_w) - (sums_of_weight_log_weights.at[i].get() / sum_w), 0.0))
        
        return model.replace(
            wave=wave,
            compatible=compatible,
            stack=stack,
            stacksize=stacksize,
            sums_of_ones=sums_of_ones,
            sums_of_weights=sums_of_weights,
            sums_of_weight_log_weights=sums_of_weight_log_weights,
            entropies=entropies
        )
    
    return jax.lax.cond(condition_1, identity, process_ban, model)

    def save(self, filename):
        """Raises NotImplementedError to ensure its subclass will provide an implementation."""
        raise NotImplementedError("Must be implemented in a subclass.")

@partial(jax.jit, static_argnums=[0, 1, 2])
def init(
    width: int,
    height: int,
    T: int,
    periodic: bool,
    heuristic: int,
    weights: jax.Array,
    propagator: jax.Array,
    key: jax.Array
) -> ModelX:
    wave_init = jnp.ones((width * height, T), dtype=bool)

    weight_log_weights = weights * jnp.log(weights + 1e-9)
    
    sum_of_weights = jnp.sum(weights)
    sum_of_weight_log_weights = jnp.sum(weight_log_weights)
    starting_entropy = jnp.log(sum_of_weights) - sum_of_weight_log_weights / sum_of_weights
    
    # Example shape for 'compatible': (width*height, T, 4). Initialize to zero or some default.
    compatible_init = jnp.zeros((width * height, T, 4), dtype=jnp.int32)

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
    stack_init = jnp.zeros((width * height * T, 2), dtype=jnp.int32)
    stacksize_init = 0
    max_stacksize_init = width * height * T

    return ModelX(
        MX=width,
        MY=height,
        T=T,
        periodic=periodic,
        heuristic=heuristic,

        wave=wave_init,
        compatible=compatible_init,
        propagator=propagator,
        observed=observed_init,
        stack=stack_init,
        stacksize=stacksize_init,
        max_stacksize=max_stacksize_init,

        weights=weights,
        weight_log_weights=weight_log_weights,
        sums_of_ones=sums_of_ones_init,
        sums_of_weights=sums_of_weights_init,
        sums_of_weight_log_weights=sums_of_weight_log_weights_init,
        entropies=entropies_init,

        sum_of_weights=sum_of_weights,
        sum_of_weight_log_weights=sum_of_weight_log_weights,
        starting_entropy=starting_entropy,
        distribution=distribution_init,
        observed_so_far=0,

        key=key
    )

@jax.jit
def run(model, seed):
    """Run the WaveFunctionCollapse algorithm with the given seed and iteration limit."""
    # Pre: the model is freshly initialized

    steps = 0
    loop_condition = True
    condition = lambda x: x
    while True:
        # Runs until finish
        steps += 1

        node = next_unobserved_node(model, key)
        if node >= 0:
            model = observe(model, node, key)
            model, success = propagate(model)
            if not success:
                return False # False
        else:
            for i in range(len(model.wave)):
                for t in range(model.T):
                    if model.wave[i][t]:
                        model.observed[i] = t
                        break # True
            return True # False
        
    
    
    return True

@jax.jit
def next_unobserved_node(model):
    """Selects the next cell to observe according to the chosen heuristic (Scanline, Entropy, or MRV).
    
    Returns:
        index (int) of the chosen cell in [0, MX*MY), or -1 if all cells are determined.
    """
    # Handle the SCANLINE heuristic
    identity = lambda x: x
    
    observed_so_far = model.observed_so_far
    
    def process_scanline_body(i, t): 
        model, loop, val = t # Model, bool, int
        def process_scanline_body_inner(i, t):
            pred_1 = jnp.logical_not(model.periodic)
            pred_2 = jnp.any(jnp.array([(i % model.MX) + model.N > model.MX, (i // model.MX) + model.N > model.MY]))
            pred_3 = jnp.all(jnp.array([pred_1, pred_2]))
            # IF pred_3 holds, continue. Do not change val
            # Else check if sum holds. Then update model and set val, loop = false
            def update_model(model):
                observed
            
        return jax.lax.cond(loop, process_scanline_body_inner, lambda x, y: y, i, t)

    if model.heuristic == Heuristic.SCANLINE:
        for i in range(model.observed_so_far, len(model.wave)):
            # skip if out of range (non-periodic boundary constraints)
            if (not model.periodic) and ((i % model.MX) + model.N > model.MX or (i // model.MX) + model.N > model.MY):
                continue
            else:
                if model.sums_of_ones[i] > 1:
                    model.observed_so_far = i + 1
                    return i
        return -1
    else:
        # Handle ENTROPY or MRV
        min_entropy = 1e4
        argmin = -1
        for i in range(len(model.wave)):
            # skip out-of-range if non-periodic
            if (not model.periodic) and ((i % model.MX) + model.N > model.MX or (i // model.MX) + model.N > model.MY):
                continue

            remaining_values = model.sums_of_ones[i]

            # ENTROPY -> we look at entropies[i], MRV -> we look at remaining_values
            if model.heuristic == Heuristic.ENTROPY:
                entropy = model.entropies[i]
            else:  # MRV
                entropy = float(remaining_values)

            if remaining_values > 1 and entropy <= min_entropy:
                # small noise to break ties
                random_no = random.uniform(model.key)
                _key, subkey = random.uniform(model.key)
                noise = 1e-6 * random_no
                if entropy + noise < min_entropy:
                    min_entropy = entropy + noise
                    argmin = i
        
        return (model.replace(
            key=subkey
        ), argmin)


@jax.jit
def clear(model):
    """Resets wave and compatibility to allow all patterns at all cells."""
    wave = jnp.ones_like(model.wave)
    compatible = jnp.zeros_like(model.compatible)

    # Set each direction's compatibility count to the number of patterns
    for i in range(model.compatible.shape[0]):
        for t in range(model.compatible.shape[1]):
            for d in range(model.compatible.shape[2]):
                compatible = compatible.at[i, t, d].set(
                    len(model.propagator[model.opposite[d]][t])
                )

    sums_of_ones = jnp.full_like(model.sums_of_ones, t)
    sums_of_weights = jnp.full_like(model.sums_of_weights, model.sum_of_weights)
    sums_of_weight_log_weights = jnp.full_like(model.sums_of_weight_log_weights, model.sum_of_weight_log_weights)
    entropies = jnp.full_like(model.entropies, model.starting_entropy)
    observed = jnp.full_like(model.observed, -1)

    return model.replace(
        wave=wave,
        compatible=compatible,
        sums_of_ones=sums_of_ones,
        sums_of_weights=sums_of_weights,
        sums_of_weight_log_weights=sums_of_weight_log_weights,
        entropies=entropies,
        observed=observed,
    )
    
@jax.jit
def propagate(model):
    """Propagates constraints across the wave."""
    
    dx = jnp.array([-1, 0, 1, 0])
    dy = jnp.array([0, 1, 0, -1])
    
    condition_1 = lambda model: model.stacksize > 0
    
    identity = lambda m, x, y, d, t: m
    identity_2 = lambda m, i, t: m
    
    def proc_propagate_tile(model, x2, y2, d, t1):
        x2 = jax.lax.cond(x2 < 0, lambda y: y + model.MX, lambda y: y - model.MX, x2)
        y2 = jax.lax.cond(y2 < 0, lambda y: y + model.MY, lambda y: y - model.MY, y2)

        i2 = x2 + y2 * model.MX
        p = model.propagator[d][t1]

        for t2 in p:
            comp = model.compatible.at[i2, t2.astype(int), d].get()
            comp -= 1
            pred = comp == 0
            model = jax.lax.cond(pred, ban, identity_2, model, i2, t2)

        return model
    
    def proc_body(model):
        i1, t1 = model.stack[model.stacksize - 1]
        model.replace(stacksize=model.stacksize - 1)

        x1 = i1 % model.MX
        y1 = i1 // model.MX

        for d in range(4):
            x2 = x1 + dx[d]
            y2 = y1 + dy[d]
            pred_a = jnp.any(jnp.array([x2 < 0, y2 < 0, x2 >= model.MX, y2 >= model.MY]))
            pred_b = jnp.logical_not(model.periodic)
            pred = jnp.all(jnp.array([pred_a, pred_b]))
            model = jax.lax.cond(pred, identity, proc_propagate_tile, model, x2, y2, d, t1)
            
        return model
    
    model = jax.lax.while_loop(condition_1, proc_body, model)
    return model, model.sums_of_ones.at[0].get() > 0

# print("Initializing XML...")
# reader = XMLReader()
# t, w, p, c = reader.get_tilemap_data()
# # models = jax.vmap(lambda _: init(10, 10, t, False, 0, w, p, None))(jnp.array([0 for _ in range(10)]))
# # models = jax.vmap(clear)(models)
# print("Initializing model...")
# model = init(10, 10, t, False, 0, w, p, None)
# print("Propagating model...")
# # model = clear(model)
# model, res = propagate(model)
