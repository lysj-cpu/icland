"""ICLand Brax environment."""

import jax
from brax import base
from brax.envs.base import Env, ObservationSize, State
from brax.mjx import base as mjx_base
from brax.mjx.pipeline import _reformat_contact

import icland


def icland_to_brax_adapter(icland_state: icland.ICLandState) -> State:
    """Converts an ICLandState to a Brax base State."""
    model = icland_state.pipeline_state.mjx_model
    data = icland_state.pipeline_state.mjx_data

    # Adapted from m_pipeline.init (brax/mjx/pipeline.py)
    # Its init function calls mjx.forward, which crashes
    q, qd = data.qpos, data.qvel
    x = base.Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = base.Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[model.body_rootid[1:]]
    offset = base.Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    reformated_data = _reformat_contact(model, data)

    return State(
        pipeline_state=mjx_base.State(
            q=q, qd=qd, x=x, xd=xd, **reformated_data.__dict__
        ),
        obs=icland_state.observation,
        reward=icland_state.reward,
        done=icland_state.done,
        metrics=icland_state.metrics,
    )


class ICLand(Env):  # type: ignore[misc]
    """ICLand Brax environment."""

    def __init__(self, rng: jax.Array) -> None:
        """Initializes the environment with a random seed."""
        self.params = icland.sample(rng)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        initial_state = icland.init(rng, self.params)

        # self._sys = mjcf.load_model(self.params.model)
        # self._action_size = self._sys.act_size

        return icland_to_brax_adapter(initial_state)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        key = jax.random.PRNGKey(0)
        return icland_to_brax_adapter(icland.step(key, state, self.params, action))

    @property
    def observation_size(self) -> ObservationSize:
        """The size of the observation vector returned in step and reset."""
        return icland.AGENT_OBSERVATION_DIM
        # From https://github.com/google/brax/blob/296184a1c77818b1988ebfe07539affe9a07db18/brax/envs/base.py#L145
        # rng = jax.random.PRNGKey(0)
        # reset_state = self.unwrapped.reset(rng)
        # obs = reset_state.obs
        # if isinstance(obs, jax.Array):
        #     return obs.shape[-1]
        # return jax.tree_util.tree_map(lambda x: x.shape, obs)

    @property
    def action_size(self) -> int:
        """The size of the action vector expected by step."""
        return icland.AGENT_ACTION_SPACE_DIM
        # return self._sys.act_size()

    @property
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""
        return "jax"
