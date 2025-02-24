"""This file contains the base Model class for WaveFunctionCollapse and helper functions."""

from enum import Enum
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Bool, Float, Int

from icland.world_gen.converter import sample_spawn_points
from icland.world_gen.tile_data import NUM_ACTIONS, PROPAGATOR, TILECODES, WEIGHTS
from icland.world_gen.XMLReader import XMLReader


@jax.jit
def _random_index_from_distribution(
    distribution: jax.Array, rand_value: Float[Array, "1"] | Float[Array, ""] | float
) -> jax.Array:
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
    def handle_zero_total(_: None) -> jax.Array:
        return jnp.array(-1, dtype=jnp.int32)

    def handle_nonzero_total(_: None) -> jax.Array:
        # Compute the cumulative sum
        cumulative_distribution = jnp.cumsum(distribution)

        # Find the index where the condition is satisfied
        condition = rand_value * total <= cumulative_distribution
        index = jnp.argmax(condition)  # First True index
        return index

    # Use lax.cond to handle the two cases
    result = jax.lax.cond(
        total <= 0,  # Condition: total <= 0
        handle_zero_total,  # If True: return -1
        handle_nonzero_total,  # If False: compute the index
        None,  # Dummy argument for the functions
    )
    return cast(jax.Array, result)


class Heuristic(Enum):
    """Enum for the heuristic selection in WaveFunctionCollapse."""

    ENTROPY = 1
    MRV = 2
    SCANLINE = 3


class JITModel(struct.PyTreeNode):  # type: ignore[no-untyped-call]
    """Base Model class for WaveFunctionCollapse algorithm."""

    # Basic config
    MX: int | Int[Array, ""]
    MY: int | Int[Array, ""]
    periodic: bool | Bool[Array, ""]
    heuristic: int | Int[Array, ""]

    # number of possible tile/pattern indices
    T: int | Int[Array, ""]
    # Sample size (N=1 for simple tiles)
    N: int | Int[Array, ""]

    # Core arrays
    # how many elements in the stack are currently valid
    stacksize: int | Int[Array, ""]
    wave: jax.Array  # shape: (MX*MY, T), dtype=bool
    compatible: jax.Array  # shape: (MX*MY, T, 4), dtype=int
    observed: jax.Array  # shape: (MX*MY, ), dtype=int
    stack: jax.Array  # shape: (MX*MY, 2), for (i, t)

    # Weights
    weights: jax.Array  # shape: (T,)
    weight_log_weights: jax.Array  # shape: (T,)

    # Summaries for each cell
    sums_of_ones: jax.Array  # shape: (MX*MY,)
    sums_of_weights: jax.Array  # shape: (MX*MY,)
    sums_of_weight_log_weights: jax.Array  # shape: (MX*MY,)
    entropies: jax.Array  # shape: (MX*MY,)

    # Precomputed sums for the entire set of patterns
    sum_of_weights: Float[Array, ""]
    sum_of_weight_log_weights: Float[Array, ""]
    starting_entropy: Float[Array, ""]

    # Because SCANLINE uses an incremental pointer
    observed_so_far: int | Int[Array, ""]

    propagator: jax.Array  # (4, T, propagator_length)
    distribution: jax.Array  # shape: (T, )

    key: jax.Array


@partial(jax.jit, static_argnums=[0, 1, 2])
def _init(
    width: int,
    height: int,
    T: int,
    N: int,
    periodic: bool,
    heuristic: int,
    weights: jax.Array,
    propagator: jax.Array,
    key: jax.Array,
) -> JITModel:
    """Initialise variables for a new Model."""
    wave_init = jnp.ones((width * height, T), dtype=bool)

    weight_log_weights = weights * jnp.log(weights)

    sum_of_weights = jnp.sum(weights)
    sum_of_weight_log_weights = jnp.sum(weight_log_weights)
    starting_entropy = (
        jnp.log(sum_of_weights) - sum_of_weight_log_weights / sum_of_weights
    )

    # Example shape for 'compatible': (width*height, T, 4). Initialize to zero or some default.
    compatible_init = jnp.zeros((width * height, T, 4), dtype=jnp.int32)

    # Observed array init
    observed_init = -jnp.ones((width * height,), dtype=jnp.int32)

    # Summaries
    sums_of_ones_init = jnp.full((width * height,), T, dtype=jnp.int32)
    sums_of_weights_init = jnp.full(
        (width * height,), sum_of_weights, dtype=jnp.float32
    )
    sums_of_weight_log_weights_init = jnp.full(
        (width * height,), sum_of_weight_log_weights, dtype=jnp.float32
    )
    entropies_init = jnp.full((width * height,), starting_entropy, dtype=jnp.float32)
    distribution_init = jnp.zeros((T,), dtype=jnp.float32)

    # Initialize the stack array
    stack_init = jnp.zeros((width * height * T, 2), dtype=jnp.int32)
    stacksize_init = 0

    return JITModel(
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
        key=key,
    )


@jax.jit
def _observe(model: JITModel, node: Int[Array, ""]) -> JITModel:
    """Collapses the wave at 'node' by picking a pattern index according to weights distribution.

    Then bans all other patterns at that node.
    """
    w = model.wave.at[node].get()

    # Prepare distribution of patterns that are still possible
    distribution = model.distribution
    distribution = jnp.where(w, model.weights, jnp.zeros((model.weights.shape[0],)))

    key, subkey = jax.random.split(model.key)
    rand_val = jax.random.uniform(subkey)
    r = _random_index_from_distribution(distribution, rand_val)
    model = model.replace(key=key, distribution=distribution)

    # Ban any pattern that isn't the chosen one
    # If wave[node][t] != (t == r) => ban it
    process_ban = lambda i, m: jax.lax.cond(
        w.at[i].get() != (i == r), lambda x: _ban(x, node, i), lambda x: x, m
    )

    model = jax.lax.fori_loop(0, model.T, process_ban, model)
    return model


@jax.jit
def _ban(model: JITModel, i: Int[Array, ""], t1: jax.Array) -> JITModel:
    """Bans pattern t at cell i. Updates wave, compatibility, sums_of_ones, entropies, and stack."""
    t = jnp.int32(t1)
    condition_1 = jnp.logical_not(model.wave.at[i, t].get())
    identity = lambda x: x

    def process_ban(model: JITModel) -> JITModel:
        wave = model.wave.at[i, t].set(False)

        # Zero-out the compatibility in all directions for pattern t at cell i
        compatible = model.compatible.at[i, t, :].set(0)

        stack = model.stack
        stacksize = model.stacksize
        stack = stack.at[stacksize].set(jnp.array([i, t]))
        stacksize += 1

        # Update sums_of_ones, sums_of_weights, sums_of_weight_log_weights, entropies
        sums_of_ones = model.sums_of_ones.at[i].subtract(1)

        sums_of_weights = model.sums_of_weights.at[i].subtract(
            model.weights.at[t].get()
        )
        sums_of_weight_log_weights = model.sums_of_weight_log_weights.at[i].subtract(
            model.weight_log_weights.at[t].get()
        )

        sum_w = sums_of_weights.at[i].get()
        entropies = model.entropies.at[i].set(
            jnp.where(
                sum_w > 0,
                jnp.log(sum_w) - (sums_of_weight_log_weights.at[i].get() / sum_w),
                0.0,
            )
        )

        return model.replace(
            wave=wave,
            compatible=compatible,
            stack=stack,
            stacksize=stacksize,
            sums_of_ones=sums_of_ones,
            sums_of_weights=sums_of_weights,
            sums_of_weight_log_weights=sums_of_weight_log_weights,
            entropies=entropies,
        )

    return cast(JITModel, jax.lax.cond(condition_1, identity, process_ban, model))


def _run(
    model: JITModel, max_steps: jax.Array = jnp.array(1000, dtype=jnp.int32)
) -> tuple[JITModel, bool]:
    """Run the WaveFunctionCollapse algorithm with the given seed and iteration limit."""
    # Pre: the model is freshly initialized

    model = _clear(model)
    # Define the loop state
    init_state = (model, 0, False, True)

    def cond_fun(state: tuple[JITModel, Any, Any, Any]) -> jax.Array:
        """Condition function for the while loop."""
        _, steps, done, _ = state
        return jnp.bitwise_and(~done, (steps < max_steps))

    def body_fun(
        state: tuple[JITModel, jnp.int32, jnp.bool, jnp.bool],
    ) -> tuple[JITModel, jnp.int32, jnp.bool, jnp.bool]:
        """Body function for the while loop."""
        model, steps, done, success = state

        # Generate new key for this iteration
        key, _ = jax.random.split(model.key)
        model = model.replace(key=key)

        # Get next unobserved node
        model, node = _next_unobserved_node(model)

        # Use lax.cond instead of if/else
        def handle_node(
            args: tuple[JITModel, jax.Array],
        ) -> tuple[JITModel, jax.Array, jax.Array]:
            model, node = args
            # Observe and propagate
            model = _observe(model, node)
            model, success = _propagate(model)
            return model, jnp.logical_not(success), success

        def handle_completion(
            args: tuple[JITModel, jax.Array],
        ) -> tuple[JITModel, jax.Array, jax.Array]:
            model, node = args

            # Final observation assignment
            def handle_completion_inner(i: jax.Array, model: JITModel) -> JITModel:
                def find_true(t: jax.Array) -> jax.Array:
                    return model.wave[i][t]

                # collapsed: array of bools, where true are valid choices
                collapsed = jax.lax.map(
                    find_true, jnp.arange(model.distribution.shape[0])
                )

                # set the first argument that is true
                model = jax.lax.cond(
                    jnp.any(collapsed),
                    lambda x: x.replace(
                        observed=x.observed.at[i].set(collapsed.argmax())
                    ),
                    lambda x: x,
                    model,
                )
                return model

            model = jax.lax.fori_loop(
                0, model.wave.shape[0], handle_completion_inner, model
            )
            return model, jnp.array(True, dtype=bool), jnp.array(True, dtype=bool)

        model, done, success = jax.lax.cond(
            node >= 0, handle_node, handle_completion, (model, node)
        )

        return (model, steps + 1, done, success)

    # Run the while loop
    final_model, _, _, success = jax.lax.while_loop(cond_fun, body_fun, init_state)

    return final_model, success


@jax.jit
def _next_unobserved_node(model: JITModel) -> tuple[JITModel, jax.Array]:
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

    def within_bounds(i: jax.Array) -> jax.Array:
        x = i % MX
        y = i // MX
        return jnp.all(
            jnp.array(
                [
                    jnp.logical_or(model.periodic, x + N <= MX),
                    jnp.logical_or(model.periodic, y + N <= MY),
                ]
            ),
            axis=0,
        )

    all_indices = jnp.arange(model.wave.shape[0])
    valid_nodes_mask = jax.vmap(within_bounds)(all_indices)

    def scanline_heuristic(_: jax.Array) -> tuple[JITModel, jax.Array]:
        observed_mask = all_indices >= observed_so_far
        sum_of_ones_mask = model.sums_of_ones[all_indices] > 1

        valid_scanline_nodes_with_choices = jnp.atleast_1d(
            jnp.all(
                jnp.array([valid_nodes_mask, observed_mask, sum_of_ones_mask]), axis=0
            )
        )

        # Use lax.dynamic_slice_in_dim to select the first element
        def process_node(_: None) -> tuple[JITModel, jax.Array]:
            indices = jnp.nonzero(
                valid_scanline_nodes_with_choices,
                size=model.wave.shape[0],
                fill_value=-1,
            )[0]
            next_node = indices[0]
            return (
                model.replace(
                    observed_so_far=next_node + 1,
                ),
                next_node,
            )

        return cast(
            tuple[JITModel, jax.Array],
            jax.lax.cond(
                jnp.any(valid_scanline_nodes_with_choices),
                process_node,
                lambda _: (model, -1),
                operand=None,  # No operand needed for this condition
            ),
        )

    def entropy_mrv_heuristic(_: jax.Array) -> tuple[JITModel, jax.Array]:
        node_entropies = jax.lax.cond(
            heuristic == Heuristic.ENTROPY.value,
            lambda _: entropies,
            lambda _: sums_of_ones.astype(jnp.float32),
            None,
        )
        sum_of_ones_mask = jax.vmap(lambda x: model.sums_of_ones.at[x].get() > 1)(
            all_indices
        )
        node_entropies_mask = jnp.atleast_1d(
            jnp.logical_and(valid_nodes_mask, sum_of_ones_mask)
        )

        def process_node(node_entropies: jax.Array) -> tuple[JITModel, jax.Array]:
            key, subkey = jax.random.split(model.key)
            node_entropies = node_entropies + 1e-6 * jax.random.normal(
                subkey, shape=node_entropies.shape
            )
            valid_node_entropies = jnp.where(
                node_entropies_mask,
                node_entropies,
                jnp.full(node_entropies.shape, jnp.inf),
            )
            min_entropy_idx = jnp.argmin(valid_node_entropies)
            return model.replace(
                key=key,
            ), min_entropy_idx

        return cast(
            tuple[JITModel, jax.Array],
            jax.lax.cond(
                jnp.any(node_entropies_mask),
                process_node,
                lambda _: (model, -1),
                operand=node_entropies,
            ),
        )

    return cast(
        tuple[JITModel, jax.Array],
        jax.lax.cond(
            heuristic == Heuristic.SCANLINE.value,
            scanline_heuristic,
            entropy_mrv_heuristic,
            operand=entropies,  # No operand needed for this condition
        ),
    )


@jax.jit
def _clear(model: JITModel) -> JITModel:
    """Resets wave and compatibility to allow all patterns at all cells.

    Optimized version using vectorized operations.
    """
    # Initialize arrays directly to their final values
    wave = jnp.ones_like(model.wave, dtype=bool)  # All True
    observed = jnp.full_like(model.observed, -1)

    # Set all statistics arrays in one go
    sums_of_ones = jnp.full_like(model.sums_of_ones, model.weights.shape[0])
    sums_of_weights = jnp.full_like(model.sums_of_weights, model.sum_of_weights)
    sums_of_weight_log_weights = jnp.full_like(
        model.sums_of_weight_log_weights, model.sum_of_weight_log_weights
    )
    entropies = jnp.full_like(model.entropies, model.starting_entropy)

    # Vectorized computation of compatible array
    opposite = jnp.array([2, 3, 0, 1])

    # Compute all pattern compatibilities at once
    # Shape: (4, T) -> (T, 4)
    pattern_compatibilities = jnp.sum(
        model.propagator[opposite, :] >= 0,  # Using broadcasting
        axis=2,  # Sum over the patterns that can appear in each direction
    ).T

    # Broadcast pattern compatibilities to all positions
    # Shape: (MX * MY, T, 4)
    compatible = jnp.broadcast_to(
        pattern_compatibilities[None, :, :], model.compatible.shape
    )

    return model.replace(
        wave=wave,
        compatible=compatible,
        sums_of_ones=sums_of_ones,
        sums_of_weights=sums_of_weights,
        sums_of_weight_log_weights=sums_of_weight_log_weights,
        entropies=entropies,
        observed=observed,
        observed_so_far=0,
    )


@jax.jit
def _propagate(model: JITModel) -> tuple[JITModel, jax.Array]:
    """Propagates constraints across the wave."""
    dx = jnp.array([-1, 0, 1, 0])
    dy = jnp.array([0, 1, 0, -1])

    condition_1 = lambda model: model.stacksize > 0

    identity = lambda m, x, y, d, t: m
    identity_2 = lambda m, i, t: m

    def proc_propagate_tile(
        model: JITModel, x2: jax.Array, y2: jax.Array, d: jax.Array, t1: jax.Array
    ) -> JITModel:
        x2 = jax.lax.cond(
            x2 < 0,
            lambda y: y + model.MX,
            lambda y: jax.lax.cond(
                x2 >= model.MX, lambda z: z - model.MX, lambda w: w, y
            ),
            x2,
        )
        y2 = jax.lax.cond(
            y2 < 0,
            lambda y: y + model.MY,
            lambda y: jax.lax.cond(
                y2 >= model.MY, lambda z: z - model.MY, lambda w: w, y
            ),
            y2,
        )

        i2 = x2 + y2 * model.MX
        p = model.propagator[d][t1]

        for t2 in p:
            pred_2 = t2 >= 0

            def process_p(model: JITModel, i2: jax.Array, t2: jax.Array) -> JITModel:
                comp = model.compatible.at[i2, t2.astype(jnp.int32), d].subtract(1)
                model = model.replace(compatible=comp)
                pred = comp.at[i2, t2.astype(jnp.int32), d].get() == 0
                return cast(
                    JITModel, jax.lax.cond(pred, _ban, identity_2, model, i2, t2)
                )

            model = jax.lax.cond(pred_2, process_p, identity_2, model, i2, t2)

        return model

    def proc_body(model: JITModel) -> JITModel:
        i1, t1 = model.stack.at[model.stacksize - 1].get()
        model = model.replace(stacksize=model.stacksize - 1)

        x1 = i1 % model.MX
        y1 = i1 // model.MX

        def handle_fail(_: None) -> None:
            return None

        jax.lax.cond(jnp.any(model.sums_of_ones <= 0), handle_fail, lambda x: x, None)

        for d in range(4):
            x2 = x1 + dx[d]
            y2 = y1 + dy[d]
            pred_a = jnp.any(
                jnp.array(
                    [x2 < 0, y2 < 0, x2 + model.N > model.MX, y2 + model.N > model.MY]
                )
            )
            pred_b = jnp.logical_not(model.periodic)
            pred = jnp.all(jnp.array([pred_a, pred_b]))
            model = jax.lax.cond(
                pred, identity, proc_propagate_tile, model, x2, y2, d, t1
            )

        return model

    model = jax.lax.while_loop(condition_1, proc_body, model)
    return model, jnp.all(model.sums_of_ones > 0)


@partial(jax.jit, static_argnums=[2, 3])
def export(model: JITModel, tilemap: jax.Array, width: int, height: int) -> jax.Array:
    """Reshapes model data, combines it with tile info via vectorization, and generates a one-hot encoded state."""
    observed_reshaped = jnp.reshape(model.observed, (width, height))
    # Combine observed state and tile information using jax.vmap
    # Apply combine function using vmap for vectorization
    combined = jax.vmap(lambda x: tilemap.at[x].get())(observed_reshaped)

    return combined


@partial(jax.jit, static_argnames=["width", "height"])
def sample_world(
    width: jax.Array,
    height: jax.Array,
    key: jax.Array,
    periodic: jax.Array,
    heuristic: jax.Array,
) -> JITModel:
    """Samples a world such that its complete and has a playable area."""
    model = _init(
        width, height, NUM_ACTIONS, 1, periodic, heuristic, WEIGHTS, PROPAGATOR, key
    )
    condition = lambda x: jnp.logical_not(x[1])
    success = False

    def body_func(state: tuple[JITModel, bool]) -> tuple[JITModel, bool]:
        model, _ = state
        model, b = _run(model)
        key, _ = jax.random.split(model.key)
        model = model.replace(key=key)
        return (model, b)

    return cast(
        JITModel,
        jax.lax.while_loop(condition, body_func, (model, success))[0],
    )


if __name__ == "__main__":  # Drive code used for testing.
    xml_reader = XMLReader("src/icland/world_gen/tilemap/data.xml")
    key_no = 216
    key = jax.random.key(key_no)
    w = 10
    h = 10
    model = sample_world(w, h, 1000, key, True, 1)
    one_hot = export(model, TILECODES, w, h)
    xml_reader.save(model.observed, w, h, f"tilemap_{key_no}.png")
    print(one_hot.tolist())
    spawnable = sample_spawn_points(key, one_hot, num_objects=3)
    print(spawnable.tolist())
