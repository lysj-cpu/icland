"""ICLand Brax environment."""

from typing import Optional

import jax
from brax.base import Motion, System, Transform
from brax.envs.base import Env, ObservationSize, State
from brax.io import mjcf
from brax.mjx.base import State as mjx_state
from brax.mjx.pipeline import _reformat_contact

import icland


def icland_to_brax_adapter(icland_state: icland.ICLandState) -> State:
    """Converts an ICLandState to a Brax base State."""
    model = icland_state.pipeline_state.mjx_model
    data = icland_state.pipeline_state.mjx_data

    # Adapted from m_pipeline.init (brax/mjx/pipeline.py)
    # Its init function calls mjx.forward, which crashes
    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[model.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    reformated_data = _reformat_contact(model, data)

    return State(
        pipeline_state=mjx_state(q=q, qd=qd, x=x, xd=xd, **reformated_data.__dict__),
        obs=icland_state.observation,
        reward=icland_state.reward,
        done=icland_state.done,
        metrics=icland_state.metrics,
    )


def brax_to_icland_adapter(
    brax_state: State, model: System, agent_count: int
) -> icland.ICLandState:
    """Converts a Brax base State to an ICLandState."""
    # In m_pipeline.step, mjx.step is called directly on the brax state
    return icland.ICLandState(
        pipeline_state=icland.PipelineState(
            mjx_model=model,
            mjx_data=brax_state.pipeline_state,
            # TODO: Consider caching agent calculation
            component_ids=icland.collect_agent_components(model, agent_count),
        ),
        observation=brax_state.obs,
        reward=brax_state.reward,
        done=brax_state.done,
        metrics=brax_state.metrics,
        info=brax_state.info,
    )


class ICLand(Env):  # type: ignore[misc]
    """ICLand Brax environment."""

    def __init__(self, rng: jax.Array, params: Optional[icland.ICLandParams]) -> None:
        """Initializes the environment with a random seed."""
        if params is None:
            params = icland.sample(rng)
        self.params = params
        self._sys = mjcf.load_model(self.params.model)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        initial_state = icland.init(rng, self.params)
        return icland_to_brax_adapter(initial_state)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        key = jax.random.PRNGKey(0)
        return icland_to_brax_adapter(
            icland.step(
                key,
                brax_to_icland_adapter(state, self._sys, self.params.agent_count),
                self.params,
                action,
            )
        )

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
