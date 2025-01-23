import jax
import jax.numpy as jnp
import pytest

from icland.world_gen.JITModel import (
    ban,
    clear,
    export,
    init,
    observe,
    propagate,
    random_index_from_distribution,
    run
)
from icland.world_gen.XMLReader import XMLReader


@pytest.fixture
def xml_reader():
    """Fixture to create an XMLReader instance with our data XML file."""
    xml_path = "src/icland/world_gen/tilemap/data.xml"
    return XMLReader(xml_path=xml_path)


@pytest.fixture
def model(xml_reader):
    """Fixture to create a JITModel instance."""
    t, w, p, _ = xml_reader.get_tilemap_data()
    model = init(
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
    return model


@pytest.fixture
def tilemap(xml_reader):
    """Fixture to create the corresponding tilemap from our XMLReader instance."""
    _, _, _, c = xml_reader.get_tilemap_data()
    return c


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
def test_random_index_from_distribution(distribution, rand_value, expected_result):
    """Test the random_index_from_distribution function."""
    # JIT compile the function
    jit_func = jax.jit(random_index_from_distribution)

    # Call the function and check the result
    result = jit_func(distribution, rand_value)

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
def test_edge_cases_for_random_index_from_distribution(distribution):
    """Test edge cases for random_index_from_distribution with edge cases in distribution."""
    # Test with random values between 0 and 1 for rand_value
    for rand_value in [0.0, 0.5, 0.999]:
        result = random_index_from_distribution(distribution, rand_value)
        if jnp.sum(distribution) == 0:
            assert result == -1, f"Expected -1 for sum 0, but got {result}"
        else:
            assert result >= 0 and result < len(distribution), (
                f"Expected index between 0 and {len(distribution) - 1}, but got {result}"
            )


def test_model_initialization(model):
    """Test the initialization of the ModelX."""
    assert model.MX == 10
    assert model.MY == 10
    assert model.T == 158
    assert model.N == 1
    assert model.periodic == False
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


def test_model_propagate(model):
    """Test the propagate function."""
    updated_model, has_non_zero_sum = propagate(model)  # Propagate constraints

    assert updated_model.stacksize == 0, (
        "Stack size should be reduced after propagation."
    )
    assert has_non_zero_sum, (
        "The sum of ones should be greater than 0 after propagation."
    )
    assert updated_model.compatible[0, 0, 0] == 0, (
        "The compatible array should have been modified."
    )


def test_model_observe(model):
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
    observed_model = observe(model, node)
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
            assert observed_model.wave[node, pattern] == False, (
                f"Pattern {pattern} should be banned."
            )
        else:
            assert observed_model.wave[node, pattern] == True, (
                f"Pattern {pattern} should be the chosen pattern."
            )


def test_model_ban(model):
    """Test the ban function."""
    # Select a cell index and pattern to ban
    i = 5
    t = 1  # Pattern index to ban

    updated_model = ban(model, i, jnp.array(t))

    assert updated_model.wave.at[i, t].get() == False, (
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

def test_model_run(model):
    key = jax.random.PRNGKey(0)

    # Run the function
    final_model, success = run(model, max_steps=100)

    # Assert the success flag
    assert success, "Algorithm did not complete successfully"

    # Verify model updates
    print('observed: ', final_model.observed)
    assert jnp.all(final_model.observed >= 0), "Not all nodes were observed"
    assert final_model.key is not None, "Final model has no key"

    # Ensure no infinite loop (e.g., reached max_steps)
    assert final_model.key != key, "Algorithm did not run properly"

def test_model_clear(model):
    """Test the clear function to ensure it resets the model's attributes correctly."""

    # Call the clear function
    updated_model = clear(model)

    # Test that 'wave' is reset to all True
    assert jnp.all(updated_model.wave == True), "Wave should be all True."

    # Test that 'observed' is reset to -1
    assert jnp.all(updated_model.observed == -1), "Observed should be all -1."

    # Test that 'sums_of_ones' is correctly set to the size of weights (4)
    assert jnp.all(updated_model.sums_of_ones == 158), "Sums of ones should match the number of weights."

    # Test that 'sums_of_weight_log_weights' is set correctly to 0.5
    assert jnp.all(updated_model.sums_of_weight_log_weights == 0.5), "Sums of weight log weights should match the initial value."

    # Test that 'entropies' is set to the starting entropy (1.0)
    assert jnp.all(updated_model.entropies == 1.0), "Entropies should match the starting entropy."

    # Test that 'compatible' is computed correctly
    expected_compatible = jnp.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    assert jnp.all(updated_model.compatible == expected_compatible), "Compatible should match expected pattern compatibilities."

    # Test that other attributes remain unchanged
    assert updated_model.weights is model.weights, "Weights should remain unchanged."
    assert updated_model.sum_of_weights == model.sum_of_weights, "Sum of weights should remain unchanged."
    assert updated_model.starting_entropy == model.starting_entropy, "Starting entropy should remain unchanged."


def test_model_export(model, tilemap):
    """Test the export function."""
    pass
