"""This module defines type aliases and type variables for the ICLand project.

It includes types for model parameters, state, and action sets used in the project.
"""

import inspect
from typing import Callable, Type, TypeAlias

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

    def __repr__(self) -> str:
        """Return a string representation of the PipelineState object."""
        return f"PipelineState(mjx_model={type(self.mjx_model).__name__}, mjx_data={type(self.mjx_data).__name__}, component_ids={self.component_ids})"


# Type variables for to define pytree functions of states
type ICLandStateChildren = tuple[
    PipelineState,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
    dict[str, jax.Array],
]

type ICLandBraxStateChildren = tuple[
    mjx_state,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
    System,
    jax.Array,
]


class ICLandInfo(PyTreeNode):  # type: ignore[misc]
    """Information about the ICLand environment.

    Attributes:
        agent_positions: [[x, y, z]] of agent positions, indexed by agent's body ID.
        agent_velocities: [[x, y, z]] of agent velocities, indexed by agent's body ID.
        agent_rotations: Quat of agent rotations, indexed by agent's body ID.
    """

    agent_positions: jax.Array
    agent_velocities: jax.Array
    agent_rotations: jax.Array


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

        reformated_data = {
            k: v
            for k, v in reformated_data.__dict__.items()
            if k not in {"q", "qd", "x", "xd"}
        }

        return ICLandBraxState(
            pipeline_state=mjx_state(q=q, qd=qd, x=x, xd=xd, **reformated_data),
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

    def __repr__(self) -> str:
        """Return a string representation of the ICLandState object."""
        return f"ICLandState(pipeline_state={self.pipeline_state}, observation={self.observation}, reward={self.reward}, done={self.done}, metrics={self.metrics}, info={self.info})"


jax.tree_util.register_pytree_node(
    ICLandState, ICLandState._tree_flatten, ICLandState._tree_unflatten
)


class ICLandParams(PyTreeNode):  # type: ignore[misc]
    """Parameters for the ICLand environment.

    Attributes:
        model: Mujoco model of the environment.
        reward_function: Reward function for the environment
        agent_count: Number of agents in the environment.
    """

    model: mujoco.MjModel
    reward_function: Callable[[ICLandInfo], jax.Array] | None
    agent_count: int

    # Without this, model is model=<mujoco._structs.MjModel object at 0x7b61fb18dc70>
    # For some arbitrary memory address. __repr__ provides cleaner output
    # for users and for testing.
    def __repr__(self) -> str:
        """Return a string representation of the ICLandParams object.

        Examples:
            >>> from icland.types import ICLandParams, ICLandState
            >>> import mujoco
            >>> import jax
            >>> mj_model = mujoco.MjModel.from_xml_string("<mujoco/>")
            >>> def example_reward_function(state: ICLandState) -> jax.Array:
            ...     return jax.numpy.array(0)
            >>> ICLandParams(mj_model, example_reward_function, 1)
            ICLandParams(model=MjModel, reward_function=example_reward_function(state: icland.types.ICLandState) -> jax.Array, agent_count=1)
            >>> ICLandParams(mj_model, lambda state: jax.numpy.array(0), 1)
            ICLandParams(model=MjModel, reward_function=lambda function(state), agent_count=1)
        """
        if (
            self.reward_function
            and hasattr(self.reward_function, "__name__")
            and self.reward_function.__name__ != "<lambda>"
        ):
            reward_function_name = self.reward_function.__name__
        else:
            reward_function_name = "lambda function"

        reward_function_signature = ""
        if self.reward_function is not None:
            reward_function_signature = str(inspect.signature(self.reward_function))

        return f"ICLandParams(model={type(self.model).__name__}, reward_function={reward_function_name}{reward_function_signature}, agent_count={self.agent_count})"


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

    def _tree_flatten(self) -> tuple[ICLandBraxStateChildren, None]:
        return (
            self.pipeline_state,
            self.obs,
            self.reward,
            self.done,
            self.metrics,
            self.model,
            self.component_ids,
        ), None

    @classmethod
    def _tree_unflatten(
        cls: Type["ICLandBraxState"], _: None, children: ICLandBraxStateChildren
    ) -> "ICLandBraxState":
        return cls(*children)


jax.tree_util.register_pytree_node(
    ICLandBraxState, ICLandBraxState._tree_flatten, ICLandBraxState._tree_unflatten
)
