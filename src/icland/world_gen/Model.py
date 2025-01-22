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

@struct.dataclass
class ModelX:
    """Base Model class for WaveFunctionCollapse algorithm."""

    # Basic config
    MX: int
    MY: int
    periodic: bool
    heuristic: int

    T: int  # number of possible tile/pattern indices
    N: int  # Sample size (N=1 for simple tiles)

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

@jax.jit
def observe(model, node):
    """Collapses the wave at 'node' by picking a pattern index according to weights distribution.

    Then bans all other patterns at that node.
    """
    w = model.wave.at[node].get()
    
    # Prepare distribution of patterns that are still possible
    distribution = model.distribution
    distribution = jnp.where(w, model.weights, jnp.zeros_like(model.weights))
    
    key, subkey = jax.random.split(model.key)
    r = random_index_from_distribution(model.distribution, jax.random.uniform(subkey))
    model = model.replace(key=key)

    # Ban any pattern that isn't the chosen one
    # If wave[node][t] != (t == r) => ban it
    process_ban = lambda i, m: jax.lax.cond(w.at[i].get() != (i == r), lambda x: ban(x, node, i), lambda x: x, m)
    return jax.lax.fori_loop(0, model.T, process_ban, model)

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
    N: int,
    periodic: bool,
    heuristic: int,
    weights: jax.Array,
    propagator: jax.Array,
    key: jax.Array
) -> ModelX:
    """Initialise variables for a new Model."""
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
        N=N,
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

@partial(jax.jit, static_argnames=['max_steps'])
def run(model, max_steps=1000):
    """Run the WaveFunctionCollapse algorithm with the given seed and iteration limit."""
    # Pre: the model is freshly initialized

    # Define the loop state
    init_state = {
        'model': model,
        'steps': 0,
        'done': False,
        'success': True
    }
    
    def cond_fun(state):
        """Condition function for the while loop."""
        return (~state['done']) & (state['steps'] < max_steps)
    
    def body_fun(state):
        """Body function for the while loop."""
        jax.debug.print("Body")
        model = state['model']
        steps = state['steps']
        
        # Generate new key for this iteration
        key, subkey = jax.random.split(model.key)
        model = model.replace(key=key)
        
        # Get next unobserved node
        model, node = next_unobserved_node(model)
        
        # Use lax.cond instead of if/else
        def handle_node(args):
            jax.debug.print("  Handle node")
            model, node, key = args
            # Observe and propagate
            model = observe(model, node)
            model, success = propagate(model)
            return model, False, success
        
        def handle_completion(args):
            model, node, key = args
            jax.debug.print("  Handle completion")
            # Final observation assignment
            def body_fun(i, model):
                def find_true(t):
                    return model.wave[i][t]
                t = jax.lax.map(find_true, jnp.arange(model.distribution.shape[0])).argmax()
                model = model.replace(observed=model.observed.at[i].set(t))
                return model
            
            model = jax.lax.fori_loop(0, model.wave.shape[0], body_fun, model)
            return model, True, True
        
        model, done, success = jax.lax.cond(
            node >= 0,
            handle_node,
            handle_completion,
            (model, node, subkey)
        )
        
        return {
            'model': model,
            'steps': steps + 1,
            'done': done,
            'success': success
        }
    
    # Run the while loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    
    return final_state['model'], final_state['success']

@jax.jit
def next_unobserved_node(model):
    """Selects the next cell to observe according to the chosen heuristic (Scanline, Entropy, or MRV).
    
    Returns:
        index (int) of the chosen cell in [0, MX*MY), or -1 if all cells are determined.
    """
    MX = model.MX
    MY = model.MY
    N = model.N
    periodic = model.periodic
    heuristic = model.heuristic
    sums_of_ones = model.sums_of_ones
    entropies = model.entropies
    observed_so_far = model.observed_so_far

    def within_bounds(i):
        x = i % MX
        y = i // MX
        return jnp.all(jnp.array([
            jnp.logical_or(model.periodic, x + N <= MX), 
            jnp.logical_or(model.periodic, y + N <= MY)
        ]), axis=0)

    all_indices = jnp.arange(model.wave.shape[0])
    valid_nodes_mask = jax.vmap(within_bounds)(all_indices)
    
    key = model.key

    def scanline_heuristic(_):
        observed_mask = jax.vmap(lambda x: x >= observed_so_far)(all_indices)
        sum_of_ones_mask = jax.vmap(lambda x: model.sums_of_ones.at[x].get() > 1)(all_indices)
        valid_scanline_nodes_with_choices = jnp.atleast_1d(jnp.all(jnp.array([valid_nodes_mask, observed_mask, sum_of_ones_mask])))
        # Use lax.dynamic_slice_in_dim to select the first element 
        def process_node(_):
            indices = jnp.nonzero(valid_scanline_nodes_with_choices, size=model.wave.shape[0], fill_value=-1)[0]
            next_node = indices[0]
            return (model.replace(observed_so_far=next_node + 1, key=jax.random.fold_in(key, next_node)), next_node)
            
        return jax.lax.cond(
            jnp.any(valid_scanline_nodes_with_choices),
            process_node,
            lambda _: (model, -1),
            operand=None  # No operand needed for this condition
        )

    def entropy_mrv_heuristic(_):
        node_entropies = jax.lax.cond(
            heuristic == Heuristic.ENTROPY.value,
            lambda _: entropies,
            lambda _: sums_of_ones.astype(float),
            None)
        sum_of_ones_mask = jax.vmap(lambda x: model.sums_of_ones.at[x].get() > 1)(all_indices)
        node_entropies_mask = jnp.atleast_1d(jnp.logical_and(valid_nodes_mask, sum_of_ones_mask))
        def process_node(_):
            valid_node_entropies = jnp.where(node_entropies_mask, node_entropies, jnp.full(node_entropies.shape, jnp.inf))
            min_entropy_idx = jnp.argmin(valid_node_entropies)
            return model.replace(key=jax.random.fold_in(key, min_entropy_idx)), min_entropy_idx
        
        return jax.lax.cond(
            jnp.any(node_entropies_mask),
            process_node,
            lambda _: (model, -1),
            operand=None  # No operand needed for this condition
        )

    return jax.lax.cond(
        heuristic == Heuristic.SCANLINE.value,
        scanline_heuristic, 
        entropy_mrv_heuristic, 
        operand=None  # No operand needed for this condition
    )


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

# Run the operations to be profiled
print("Initializing XML...")
reader = XMLReader()
t, w, p, c = reader.get_tilemap_data()
# models = jax.vmap(lambda _: init(10, 10, t, False, 0, w, p, None))(jnp.array([0 for _ in range(10)]))
# models = jax.vmap(clear)(models)
print("Initializing model...")
model = init(3, 3, t, 1, False, 1, w, p, jax.random.key(0))
print("Running model...")
# # model = clear(model)
# model, res = propagate(model)
# model = next_unobserved_node(model)
model, b = run(model)
