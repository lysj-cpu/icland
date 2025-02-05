"""ICLand Brax environment."""

from typing import Optional

import jax
from brax.envs.base import Env, ObservationSize
from brax.io import mjcf

import icland


class ICLand(Env):  # type: ignore[misc]
    """ICLand Brax environment."""

    def __init__(
        self, rng: jax.Array, params: Optional[icland.ICLandParams] = None
    ) -> None:
        """Initializes the environment with a random seed."""
        if params is None:
            params = icland.sample(rng)
        self.params = params
        self._sys = mjcf.load_model(self.params.model)

    def reset(self, rng: jax.Array) -> icland.ICLandBraxState:
        """Resets the environment to an initial state."""
        initial_state = icland.init(rng, self.params)
        return initial_state.to_brax_state()

    def step(
        self, state: icland.ICLandBraxState, action: jax.Array
    ) -> icland.ICLandBraxState:
        """Run one timestep of the environment's dynamics."""
        key = jax.random.PRNGKey(0)
        icland_state: icland.ICLandState = icland.step(
            key,
            state.to_icland_state(),
            self.params,
            action,
        )
        return icland_state.to_brax_state()

    @property
    def observation_size(self) -> ObservationSize:
        """The size of the observation vector returned in step and reset."""
        return icland.AGENT_OBSERVATION_DIM

    @property
    def action_size(self) -> int:
        """The size of the action vector expected by step."""
        return icland.AGENT_ACTION_SPACE_DIM

    @property
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""
        return "jax"
