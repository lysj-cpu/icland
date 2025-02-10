"""ICLand Brax environment."""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from brax import base
from brax.base import Motion, Transform
from brax.envs.base import Env, Observation, ObservationSize
from brax.io import mjcf
from brax.mjx.base import State as mjx_state
from brax.mjx.pipeline import _reformat_contact
from flax import struct

import icland
from icland.constants import AGENT_OBSERVATION_DIM
from icland.types import ICLandState, MjxModelType, MjxStateType


@struct.dataclass
class ICLandBraxState(base.Base):  # type: ignore
    """Environment state for training and inference."""

    ic_state: ICLandState
    pipeline_state: Optional[base.State]
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)  # type: ignore[no-untyped-call]
    info: Dict[str, Any] = struct.field(default_factory=dict)  # type: ignore[no-untyped-call]


class ICLand(Env):  # type: ignore
    """ICLand Brax environment."""

    def __init__(
        self, rng: jax.Array, params: Optional[icland.ICLandParams] = None
    ) -> None:
        """Initializes the environment with a random seed.

        Args:
            rng: A JAX random key.
            params: Optional ICLand parameters; if not provided they are sampled.
        """
        if params is None:
            params = icland.sample(rng)
        self.params = params
        self._sys = mjcf.load_model(self.params.model)
        self.reward_function = self.params.reward_function

    def reset(self, rng: jax.Array) -> ICLandBraxState:
        """Resets the environment to an initial state.

        Args:
            rng: A JAX random key for initialization.

        Returns:
            The initial environment state.
        """
        initial_state = icland.init(rng, self.params)
        model = initial_state.pipeline_state.mjx_model
        data = initial_state.pipeline_state.mjx_data

        pipeline_state = self._build_pipeline_state(model, data)

        # Initialize reward and done to zero.
        reward = jnp.array(0.0)
        done = jnp.array(0.0)

        return ICLandBraxState(
            ic_state=initial_state,
            pipeline_state=pipeline_state,
            obs=jnp.zeros(
                AGENT_OBSERVATION_DIM
            ),  # This is temporary while observation is not implemented.
            reward=reward,
            done=done,
        )

    def step(self, state: ICLandBraxState, action: jax.Array) -> ICLandBraxState:
        """Runs one timestep of the environment's dynamics.

        Args:
            state: The current environment state.
            action: The action to take.

        Returns:
            The updated environment state.
        """
        # NOTE: A constant key is used here. In a production setup, consider
        # passing the key from outside to ensure proper random behavior.
        key = jax.random.PRNGKey(0)

        # Step the internal ICLand simulation.
        ic_state = icland.step(key, state.ic_state, None, action)
        model = ic_state.pipeline_state.mjx_model
        data = ic_state.pipeline_state.mjx_data

        pipeline_state = self._build_pipeline_state(model, data)
        reward: float = self.reward_function(ic_state.data)[0][0]  # type: ignore

        nstate: ICLandBraxState = state.replace(
            ic_state=ic_state,
            pipeline_state=pipeline_state,
            obs=ic_state.obs,
            reward=jnp.array(reward),
            done=jnp.array(0.0),
        )

        return nstate

    def _build_pipeline_state(
        self, model: MjxModelType, data: MjxStateType
    ) -> base.State:
        """Helper method to create a pipeline state from the provided model and data.

        Args:
            model: The MJX model.
            data: The MJX data containing positions, velocities, etc.

        Returns:
            A Brax MJX state object.
        """
        q, qd = data.qpos, data.qvel
        x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[model.body_rootid[1:]]
        offset_transform = Transform.create(pos=offset)
        xd = offset_transform.vmap().do(cvel)

        # Reformat contact data, excluding keys that will be redefined.
        reformatted_data = _reformat_contact(model, data)
        reformatted_data = {
            k: v
            for k, v in reformatted_data.__dict__.items()
            if k not in {"q", "qd", "x", "xd"}
        }

        return mjx_state(q=q, qd=qd, x=x, xd=xd, **reformatted_data)

    @property
    def observation_size(self) -> ObservationSize:
        """The size of the observation vector returned in step and reset.

        Returns:
            The observation size.
        """
        return icland.AGENT_OBSERVATION_DIM

    @property
    def action_size(self) -> int:
        """The size of the action vector expected by step.

        Returns:
            The action size.
        """
        return icland.AGENT_ACTION_SPACE_DIM

    @property
    def backend(self) -> str:
        """The physics backend that this environment uses.

        Returns:
            A string representing the backend.
        """
        return "jax"
