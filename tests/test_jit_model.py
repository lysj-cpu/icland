"""Test for our JIT-able model in world generation."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from icland.world_gen.JITModel import (
    Heuristic,
    JITModel,
    _ban,
    _clear,
    _init,
    _next_unobserved_node,
    _observe,
    _propagate,
    _random_index_from_distribution,
    _run,
)
from icland.world_gen.XMLReader import XMLReader


@pytest.fixture
def xml_reader() -> XMLReader:
    """Fixture to create an XMLReader instance with our data XML file."""
    xml_path = "src/icland/world_gen/tilemap/data.xml"
    return XMLReader(xml_path=xml_path)


@pytest.fixture
def model(xml_reader: XMLReader) -> JITModel:
    """Fixture to create a JITModel instance."""
    t, w, p, _ = xml_reader.get_tilemap_data()
    model = _init(
        width=10,
        height=10,
        T=t,
        N=1,
        periodic=False,
        heuristic=1,
        weights=w,
        propagator=p,
        key=jax.random.key(0),
    )
    return cast(JITModel, model)


@pytest.fixture
def tilemap(xml_reader: XMLReader) -> jax.Array:
    """Fixture to create the corresponding tilemap from our XMLReader instance."""
    _, _, _, c = xml_reader.get_tilemap_data()
    return c


# Tests for random_index_from_distribution
@pytest.mark.parametrize(
    "distribution, rand_value, expected_result",
    [
        # Case 1: Normal case with a valid distribution
        (jnp.array([0.1, 0.3, 0.6]), 0.4, 1),
        (jnp.array([0.1, 0.2, 0.7]), 0.9, 2),
        # Case 2: Case where the distribution sums to 0 (should return -1)
        (jnp.array([0.0, 0.0, 0.0]), 0.5, -1),  # No valid index, should return -1
        # Case 3: Case with a single element distribution
        (jnp.array([1.0]), 0.5, 0),  # Only one index available, should return 0
        # Case 4: Case where rand_value is at the boundary (0 or 1)
        (jnp.array([0.1, 0.9]), 0.0, 0),  # rand_value=0 should select index 0
        (
            jnp.array([0.1, 0.9]),
            1.0,
            1,
        ),  # rand_value=1 should select index 1 (should never actually return 1, but we're testing edge cases)
    ],
)
def test_random_index_from_distribution(
    distribution: jax.Array, rand_value: jax.Array, expected_result: jax.Array
) -> None:
    """Test the random_index_from_distribution function."""
    # JIT compile the function
    jit_func = jax.jit(_random_index_from_distribution)

    # Call the function and check the result
    result = _random_index_from_distribution(distribution, rand_value)

    # Assert that the result matches the expected result
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


@pytest.mark.parametrize(
    "distribution",
    [
        jnp.array([0.0, 0.0, 0.0]),  # Edge case: all zeros
        jnp.array([1.0, 1.0, 1.0]),  # Edge case: equal distribution
        jnp.array([1000.0, 2000.0, 3000.0]),  # Edge case: large values
    ],
)
def test_edge_cases_for_random_index_from_distribution(distribution: jax.Array) -> None:
    """Test edge cases for random_index_from_distribution with edge cases in distribution."""
    # Test with random values between 0 and 1 for rand_value
    for rand_value in [0.0, 0.5, 0.999]:
        result = _random_index_from_distribution(distribution, rand_value)
        if jnp.sum(distribution) == 0:
            assert result == -1, f"Expected -1 for sum 0, but got {result}"
        else:
            assert result >= 0 and result < len(distribution), (
                f"Expected index between 0 and {len(distribution) - 1}, but got {result}"
            )


def test_model_init(model: JITModel) -> None:
    """Test the initialization of the ModelX."""
    assert model.MX == 10
    assert model.MY == 10
    assert model.T == 158
    assert model.N == 1
    assert not model.periodic
    assert model.heuristic == 1

    # Check that the wave is initialized to all True
    assert jnp.all(model.wave), "Wave should be initialized to all True"

    # Check that the observed array is initialized to -1
    # print("Observed array dtype:", model.observed.dtype)
    assert jnp.all(model.observed == -1), "Observed should be initialized to -1"

    # Check that the sums_of_ones is initialized correctly
    assert jnp.all(model.sums_of_ones == model.T), (
        "sums_of_ones should be initialized to T"
    )

    # Check that the sums_of_weights is initialized correctly
    assert jnp.all(model.sums_of_weights == jnp.sum(model.weights)), (
        "sums_of_weights should be initialized to the sum of weights"
    )

    # Check that the entropies are initialized correctly
    assert jnp.all(model.entropies == model.starting_entropy), (
        "Entropies should be initialized to starting_entropy"
    )


def test_model_observe(model: JITModel) -> None:
    """Test the observe function."""
    # - key changed
    # - and then could run ban
    # wave=wave,
    # compatible=compatible,
    # stack=stack,
    # stacksize=stacksize,
    # sums_of_ones=sums_of_ones,
    # sums_of_weights=sums_of_weights,
    # sums_of_weight_log_weights=sums_of_weight_log_weights,
    # entropies=entropies,
    # Initialize a dummy ModelX instance
    # Select a random node to observe

    node = 12

    # Run the `observe` function
    observed_model = _observe(model, jnp.array(node))
    assert model.key != observed_model.key, (
        "The random key should be updated after observation."
    )
    observed_wave = observed_model.wave.at[node,].get()
    assert jnp.sum(observed_wave) == 1, (
        "Only one pattern should remain possible at the observed node."
    )

    chosen_pattern = jnp.argmax(observed_wave)
    assert model.weights[chosen_pattern] > 0, (
        "The chosen pattern should have a non-zero weight."
    )

    # Verify that all other patterns at the observed node are banned
    for pattern in range(model.T):
        if pattern != chosen_pattern:
            assert not observed_model.wave[node, pattern], (
                f"Pattern {pattern} should be banned."
            )
        else:
            assert observed_model.wave[node, pattern], (
                f"Pattern {pattern} should be the chosen pattern."
            )


def test_model_ban(model: JITModel) -> None:
    """Test the ban function."""
    # Select a cell index and pattern to ban
    i = 5
    t = 1  # Pattern index to ban

    updated_model = _ban(model, jnp.array(i), jnp.array(t))

    assert not updated_model.wave.at[i, t].get(), (
        f"Pattern {t} at cell {i} should be banned."
    )

    # Assert compatibility is zeroed out
    assert jnp.all(updated_model.compatible.at[i, t, :].get() == 0), (
        f"Compatibility for pattern {t} at cell {i} should be zeroed out."
    )

    # Assert stack has been updated
    assert updated_model.stacksize == model.stacksize + 1, (
        "Stack size should have increased by 1."
    )
    assert updated_model.stack.at[model.stacksize].get().tolist() == [i, t], (
        "The stack should include the banned pattern and cell index."
    )

    # Assert sums_of_ones has been decremented
    assert (
        updated_model.sums_of_ones.at[i].get() == model.sums_of_ones.at[i].get() - 1
    ), f"sums_of_ones at cell {i} should have decremented by 1."

    # Assert sums_of_weights and sums_of_weight_log_weights have been updated
    assert (
        updated_model.sums_of_weights.at[i].get()
        == model.sums_of_weights.at[i].get() - model.weights.at[t].get()
    ), f"sums_of_weights at cell {i} should reflect the banned pattern."
    assert (
        updated_model.sums_of_weight_log_weights.at[i].get()
        == model.sums_of_weight_log_weights.at[i].get()
        - model.weight_log_weights.at[t].get()
    ), f"sums_of_weight_log_weights at cell {i} should reflect the banned pattern."

    # Assert entropy has been updated
    expected_entropy = jnp.where(
        updated_model.sums_of_weights.at[i].get() > 0,
        jnp.log(updated_model.sums_of_weights.at[i].get())
        - (
            updated_model.sums_of_weight_log_weights.at[i].get()
            / updated_model.sums_of_weights.at[i].get()
        ),
        0.0,
    )
    assert jnp.isclose(updated_model.entropies.at[i].get(), expected_entropy), (
        f"Entropy at cell {i} should be updated correctly."
    )


def test_model_run(model: JITModel) -> None:
    """Test the run function."""
    key = jax.random.PRNGKey(0)

    # Run the function
    for k in range(10):
        key, subkey = jax.random.split(model.key)
        model = _clear(model)
        model = model.replace(key=key)
        model, success = _run(model, max_steps=jnp.array(100, dtype=jnp.int32))
        if success:
            # Save result
            break

    # Verify model updates
    # print("observed: ", final_model.observed)
    assert jnp.all(model.observed >= 0), "Not all nodes were observed"
    assert jnp.count_nonzero(model.wave) == model.MX * model.MY, (
        "Not all nodes were collapsed"
    )

    # # Ensure no infinite loop (e.g., reached max_steps)
    assert model.key != key, "Algorithm did not run properly"


def test_model_propagate(model: JITModel) -> None:
    """Test the propagate function."""
    updated_model, has_non_zero_sum = _propagate(model)  # Propagate constraints

    assert updated_model.stacksize == 0, (
        "Stack size should be reduced after propagation."
    )
    assert has_non_zero_sum, (
        "The sum of ones should be greater than 0 after propagation."
    )
    assert updated_model.compatible[0, 0, 0] == 0, (
        "The compatible array should have been modified."
    )


def test_model_next_unobserved_node_scanline(model: JITModel) -> None:
    """Test for next unobserved node with scanline.

    SCANLINE picks the first valid cell that hasn't been observed and has sums_of_ones > 1.
    In our fixture, only index 0 is in-bounds if N=2 and periodic=False.
    sums_of_ones[0] = 2 (>1), so we expect index=0.
    """
    model = model.replace(heuristic=Heuristic.SCANLINE.value)
    model_1, chosen_index_1 = _next_unobserved_node(model)
    model_2, chosen_index_2 = _next_unobserved_node(model_1)
    assert chosen_index_1 == 0
    assert model_1.observed_so_far == 1
    assert chosen_index_2 == 1

    # Make sure observed_so_far was updated to 1 in the new model (if your code does that).
    # Adjust to match how your real code updates things.
    assert model_2.observed_so_far == 2


def test_next_unobserved_node_entropy(model: JITModel) -> None:
    """Test for next unobserved node with entropy.

    ENTROPY picks the valid cell with the minimum 'entropies' among those with sums_of_ones>1.
    Because only index=0 is truly in-bounds, we expect index=0 again (entropy=0.5).
    Even though index=2 has lower entropy (0.4), it's out-of-bounds with N=2 (periodic=False).
    """
    model = model.replace(heuristic=Heuristic.ENTROPY.value)
    new_model, chosen_index = _next_unobserved_node(model)
    assert chosen_index == 0
    assert new_model.observed_so_far == 0, (
        "Check if your code updates observed_so_far for ENTROPY"
    )


def test_next_unobserved_node_mrv(model: JITModel) -> None:
    """Test for next unobserved node with MRV.

    MRV picks the valid cell with the smallest sums_of_ones (>1).
    The only in-bounds index is 0, which has sums_of_ones=2 (>1), so expect 0 again.
    """
    model = model.replace(heuristic=Heuristic.MRV.value)
    new_model, chosen_index = _next_unobserved_node(model)
    assert chosen_index == 0


def test_next_unobserved_node_all_determined(model: JITModel) -> None:
    """Test for next unobserved node when all are determined.

    If all valid cells have sums_of_ones <= 1 (or if none are in-bounds),
    the function should return -1.
    """
    # Force sums_of_ones <= 1
    sums_of_ones = jnp.ones_like(model.sums_of_ones)
    model = model.replace(heuristic=Heuristic.ENTROPY.value, sums_of_ones=sums_of_ones)
    new_model, chosen_index = _next_unobserved_node(model)
    assert chosen_index == -1, "Expected -1 when all are determined"


# TODO: Not sure what chosen_index should be now, because it is a bit random
# def test_next_unobserved_node_periodic(model):
#     """Demonstrate what happens if periodic=True, so more cells might be in-bounds.

#     With periodic=True, index=2 might be considered valid, etc.
#     We'll see if it chooses index=2 due to lower entropy (0.4) under ENTROPY.
#     """
#     model = model.replace(periodic=True, heuristic=Heuristic.ENTROPY.value)
#     # Now index=0,1,2,3 might all be in-bounds since we wrap.
#     # Among them, which has sums_of_ones>1? Indices 0 (2) and 2 (3).
#     # Among those, the entropies are 0.5 and 0.4, so index=2 is the minimum.
#     new_model, chosen_index = next_unobserved_node(model)
#     assert chosen_index == 2, "Periodic + ENTROPY => picks index=2 with entropy=0.4"


def test_model_clear(model: JITModel) -> None:
    """Test the clear function to ensure it resets the model's attributes correctly."""
    # Call the clear function
    updated_model = _clear(model)

    # Test that 'wave' is reset to all True
    assert jnp.all(updated_model.wave), "Wave should be all True."

    # Test that 'observed' is reset to -1
    assert jnp.all(updated_model.observed == -1), "Observed should be all -1."

    # Test that 'sums_of_ones' is correctly set to the size of weights (158)
    size = 158
    assert jnp.all(updated_model.sums_of_ones == size), (
        "Sums of ones should match the number of weights."
    )

    # Test that 'sums_of_weight_log_weights' is set correctly to 0.5
    assert jnp.all(
        updated_model.sums_of_weight_log_weights == model.sum_of_weight_log_weights
    ), "Sums of weight log weights should match the initial value."

    # Test that 'entropies' is set to the starting entropy (1.0)
    assert jnp.all(updated_model.entropies == model.starting_entropy), (
        "Entropies should match the starting entropy."
    )

    # Test that 'compatible' is computed correctly
    expected_compatible_shape = (100, 158, 4)  # (MX * MY, T, 4)
    assert jnp.all(updated_model.compatible.shape == expected_compatible_shape), (
        "Compatible should match expected pattern compatibilities."
    )

    # Test that other attributes remain unchanged
    assert jnp.array_equal(updated_model.weights, model.weights), (
        "Weights should remain unchanged."
    )
    assert updated_model.sum_of_weights == model.sum_of_weights, (
        "Sum of weights should remain unchanged."
    )
    assert updated_model.starting_entropy == model.starting_entropy, (
        "Starting entropy should remain unchanged."
    )
