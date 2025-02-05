"""This module defines type aliases and type variables for the ICLand project.

It includes types for model parameters, state, and action sets used in the project.
"""

import inspect
from typing import Callable, Optional, Type, TypeAlias

import jax
import jax.numpy as jnp
import mujoco
from brax.base import Motion, System, Transform
from brax.envs.base import State
from brax.mjx.base import State as mjx_state
from brax.mjx.pipeline import _reformat_contact
from mujoco.mjx._src.dataclasses import PyTreeNode

"""Type variables from external modules."""

MjxStateType: TypeAlias = mujoco.mjx._src.types.Data
MjxModelType: TypeAlias = mujoco.mjx._src.types.Model

"""Type aliases for ICLand project."""


class PipelineState(PyTreeNode):  # type: ignore[misc]
    """State of the ICLand environment.

    Attributes:
        mjx_model: JAX-compatible Mujoco model.
        mjx_data: JAX-compatible Mujoco data.
        component_ids: Array of body and geometry IDs for agents. (agent_count, [body_ids, geom_ids])
    """

    mjx_model: MjxModelType
    mjx_data: MjxStateType
    component_ids: jnp.ndarray


type ICLandStateChildren = tuple[
    PipelineState,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
    dict[str, jax.Array],
]


class ICLandState:
    """Information regarding the current step.

    Attributes:
        pipeline_state: State of the ICLand environment.
        observation: Observation of the environment.
        reward: Reward of the environment.
        done: Flag indicating if the episode is done.
        metrics: Dictionary of metrics for the environment.
        info: Dictionary of additional information.
    """

    def __init__(
        self,
        pipeline_state: PipelineState,
        observation: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        metrics: dict[str, jax.Array],
        info: dict[str, jax.Array],
    ) -> None:
        """Initializes the state with the given values."""
        self.pipeline_state = pipeline_state
        self.observation = observation
        self.reward = reward
        self.done = done
        self.metrics = metrics
        self.info = info

    def to_brax_state(self) -> "ICLandBraxState":
        """Converts an ICLandState to a Brax base State."""
        model = self.pipeline_state.mjx_model
        data = self.pipeline_state.mjx_data

        # Adapted from m_pipeline.init (brax/mjx/pipeline.py)
        # Its init function calls mjx.forward, which crashes
        q, qd = data.qpos, data.qvel
        x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[model.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)
        reformated_data = _reformat_contact(model, data)

        return ICLandBraxState(
            pipeline_state=mjx_state(
                q=q, qd=qd, x=x, xd=xd, **reformated_data.__dict__
            ),
            obs=self.observation,
            reward=self.reward,
            done=self.done,
            metrics=self.metrics,
            model=model,
            component_ids=self.pipeline_state.component_ids,
        )

    def _tree_flatten(self) -> tuple[ICLandStateChildren, None]:
        return (
            self.pipeline_state,
            self.observation,
            self.reward,
            self.done,
            self.metrics,
            self.info,
        ), None

    @classmethod
    def _tree_unflatten(
        cls: Type["ICLandState"], _: None, children: ICLandStateChildren
    ) -> "ICLandState":
        return cls(*children)


jax.tree_util.register_pytree_node(
    ICLandState, ICLandState._tree_flatten, ICLandState._tree_unflatten
)


class ICLandParams(PyTreeNode):  # type: ignore[misc]
    """Parameters for the ICLand environment.

    Attributes:
        model: Mujoco model of the environment.
        game: Game string (placeholder, currently None).
        agent_count: Number of agents in the environment.
    """

    model: mujoco.MjModel
    game: Optional[str]
    agent_count: int
    reward_function: Callable[[ICLandState], jax.Array]

    # Without this, model is model=<mujoco._structs.MjModel object at 0x7b61fb18dc70>
    # For some arbitrary memory address. __repr__ provides cleaner output
    # for users and for testing.
    def __repr__(self) -> str:
        """Return a string representation of the ICLandParams object."""
        if (
            hasattr(self.reward_function, "__name__")
            and self.reward_function.__name__ != "<lambda>"
        ):
            reward_function_name = self.reward_function.__name__
        else:
            reward_function_name = "lambda function"

        reward_function_signature = str(inspect.signature(self.reward_function))

        return f"ICLandParams(model={type(self.model).__name__}, game={self.game}, agent_count={self.agent_count}, reward_function={reward_function_name}{reward_function_signature})"


class ICLandBraxState(State):  # type: ignore[misc]
    """A stricter version of the brax base State.

    pipeline_state isn't optional and is MJX specific
    """

    def __init__(
        self,
        pipeline_state: mjx_state,
        obs: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        metrics: dict[str, jax.Array],
        model: System,
        component_ids: jax.Array,
    ) -> None:
        """Initializes the state with the given values."""
        self.model = model
        self.component_ids = component_ids
        super().__init__(pipeline_state, obs, reward, done, metrics)

    def to_icland_state(self) -> ICLandState:
        """Converts a Brax State to an ICLandState."""
        # In m_pipeline.step, mjx.step is called directly on the brax state
        return ICLandState(
            pipeline_state=PipelineState(
                mjx_model=self.model,
                mjx_data=self.pipeline_state,
                component_ids=self.component_ids,
            ),
            observation=self.obs,
            reward=self.reward,
            done=self.done,
            metrics=self.metrics,
            info=self.info,
        )
