"""Generates a game for the ICLand environment."""

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from icland.renderer.renderer import can_see_object

from .constants import *
from .types import *

POSITION_RANGE = (-5, 5)
ACCEPTABLE_DISTANCE = 0.5


# TODO: For this to work nicely
# Instead of generating new Entity/Agent objects on each step
# We need to replace the attributes of the existing objects
class RewardFunction:
    """Represents an atomic reward function with a callable action."""

    def __init__(
        self, action: Callable[[Entity, Entity], bool], obj1: Entity, obj2: Entity
    ):
        """Initialise the reward function with an action and two objects."""
        self.action = action
        self.obj1 = obj1
        self.obj2 = obj2

    def __call__(self):
        """Evaluate the reward function."""
        return self.action(self.obj1, self.obj2)

    def __repr__(self):
        """Return a string representation of the reward function."""
        return f"{self.action.__name__}({self.obj1}, {self.obj2})"

    def __and__(self, other):
        """Overload & operator to create a conjunction."""
        return Conjunction(self, other)

    def __or__(self, other):
        """Overload | operator to create a disjunction."""
        return Disjunction(self, other)

    def __invert__(self):
        """Overload ~ operator for negation."""
        return Negation(self)


class Conjunction(RewardFunction):
    """Represents a conjunction (A AND B)."""

    def __init__(self, *functions: RewardFunction) -> None:
        """Initialise the conjunction with a set of reward functions."""
        self.functions = set(functions)

    def __call__(self) -> bool:
        """Evaluate conjunction (all conditions must be True)."""
        return all(f() for f in self.functions)

    def __repr__(self) -> str:
        """Return a string representation of the conjunction."""
        return "(" + " AND ".join(map(str, self.functions)) + ")"

    def __and__(self, other) -> RewardFunction:
        """Overload & to add more conditions to the conjunction."""
        if isinstance(other, Conjunction):
            return Conjunction(*(self.functions | other.functions))
        return Conjunction(*self.functions, other)


class Disjunction(RewardFunction):
    """Represents a disjunction (A OR B)."""

    def __init__(self, *conjunctions: RewardFunction) -> None:
        """Initialise the disjunction with a set of conjunctions."""
        self.conjunctions = set(conjunctions)

    def __call__(self) -> bool:
        """Evaluate disjunction (at least one condition must be True)."""
        return any(c() for c in self.conjunctions)

    def __repr__(self) -> str:
        """Return a string representation of the disjunction."""
        return "(" + " OR ".join(map(str, self.conjunctions)) + ")"

    def __or__(self, other) -> RewardFunction:
        """Overload | to add more conditions to the disjunction."""
        if isinstance(other, Disjunction):
            return Disjunction(*(self.conjunctions | other.conjunctions))
        return Disjunction(*self.conjunctions, other)

    def __and__(self, other):
        """Overload & to form conjunctions within the disjunction."""
        return Disjunction(*(c & other for c in self.conjunctions))


class Negation(RewardFunction):
    """Represents a negated reward function (~A)."""

    def __init__(self, function: RewardFunction) -> None:
        """Initialise the negation with a reward function."""
        self.function = function

    def __call__(self) -> bool:
        """Evaluate negation (invert the condition)."""
        return not self.function()

    def __repr__(self) -> str:
        """Return a string representation of the negation."""
        return f"NOT({self.function})"

    def __invert__(self) -> RewardFunction:
        """Double negation cancels out."""
        return self.function


# N.B. In XLand: Agent height is 1.65m, near distance is 1m
# In ICLand: Agent height defined in AGENT_HEIGHT
def near(obj1: Entity, obj2: Entity, distance: float = 0.4):
    """Check if two objects are within a certain distance."""
    return jnp.linalg.norm(obj1.position - obj2.position) < distance


def see(obj1: Entity, obj2: Entity) -> bool:
    """Check if one object can see another."""
    match obj1, obj2:
        case Agent(), Agent():
            return False
        case Agent(), Prop():
            return bool(
                can_see_object(
                    player_pos=obj1.position,
                    player_dir=jnp.array(
                        [-jnp.cos(obj1.rotation), 0.0, jnp.sin(obj1.rotation)]
                    ),
                    obj_pos=obj2.position,
                    obj_sdf=obj2.sdf,
                    terrain_sdf=obj2.terrain_sdf,
                )
            )
        case Prop(), Prop():
            return False
        case Prop(), Agent():
            return see(obj2, obj1)
        case _:
            raise ValueError(f"Invalid object types: {obj1}, {obj2}")


T = TypeVar("T")
PREDICATES = [see, near]


# See: https://github.com/jax-ml/jax/discussions/17623
# Although that is for an arbitrary number of entities, this is a simpler case
# N.B. Currently, assuming linked issue holds here, this isn't jittable?
def _generate_two_unique(key: PRNGKeyArray, entities: list[T]) -> tuple[T, T]:
    """Sample two unique items from a list of unique items.

    Examples:
        >>> import jax
        >>> from icland.game import _generate_two_unique
        >>> key = jax.random.key(42)
        >>> entities = ['a', 'b', 'c', 'd']
        >>> _generate_two_unique(key, entities)
        ('c', 'd')
    """
    assert len(entities) >= 2
    indexes = jnp.arange(len(entities))
    shuffled_indexes = jax.random.permutation(key, indexes)
    return entities[shuffled_indexes[0]], entities[shuffled_indexes[1]]


def generate_reward_function(
    key: PRNGKeyArray,
    info: ICLandInfo,
    max_no_predicates: int = 5,
    neg_prob: float = 0.3,
) -> RewardFunction:
    """Randomly selects some entities from info.agents/props, and combines predicates to build a reward function."""
    entities = info.agents + info.props

    key, num_predicates_key = jax.random.split(key)

    num_predicates = jax.random.randint(
        num_predicates_key, (), 1, max_no_predicates + 1
    ).item()

    reward_funcs: list[RewardFunction] = []
    for _ in range(num_predicates):
        key, predicate_key, objects_key = jax.random.split(key, 3)
        predicate = PREDICATES[
            jax.random.randint(predicate_key, (), 0, len(PREDICATES))
        ]

        # TODO: Generate unique objects based on predicate
        # e.g. if "on", then one of the objects has to be a floor
        obj1, obj2 = _generate_two_unique(objects_key, entities)

        reward_func = RewardFunction(predicate, obj1, obj2)
        key, neg_key = jax.random.split(key)
        if jax.random.bernoulli(neg_key, p=neg_prob):
            reward_func = ~reward_func

        reward_funcs.append(reward_func)

    result = reward_funcs.pop()
    while reward_funcs:
        key, join_key = jax.random.split(key)
        clause_length = jax.random.randint(
            join_key, (), 1, len(reward_funcs) + 1
        ).item()

        if clause_length == 1:
            result |= reward_funcs.pop()
            continue

        clause = []
        for _ in range(clause_length):
            clause.append(reward_funcs.pop())

        conjunction = Conjunction(*clause)
        result |= conjunction

    return result


def generate_game(
    key: PRNGKeyArray, agent_count: int
) -> Callable[[ICLandInfo], jax.Array]:
    """Generate a game using the given random key and agent count.

    This function randomly selects one of two modes:
      - Translation mode: agents are rewarded when their (x, y) position is within
        ACCEPTABLE_DISTANCE of a target position.
      - Rotation mode: agents are rewarded when their rotation is within
        ACCEPTABLE_DISTANCE of a target rotation.

    The selection is made using the provided key.

    Args:
        key: Random key for generating the game.
        agent_count: Number of agents in the environment.

    Returns:
        A function that takes an ICLandInfo and returns a reward array.
    """
    # Split the key so that one part is used to decide the mode and another for sampling the target.
    key, mode_key, target_key = jax.random.split(key, 3)
    # Randomly decide the mode: True for translation mode, False for rotation mode.
    is_translation_mode = jax.random.bernoulli(mode_key)

    if is_translation_mode:
        # Translation mode: sample a target (x, y) position for each agent.
        target_position = jax.random.uniform(
            target_key,
            (agent_count, 2),
            minval=POSITION_RANGE[0],
            maxval=POSITION_RANGE[1],
        )

        def reward_function(info: ICLandInfo) -> jax.Array:
            """Compute reward based on agent position relative to a target position."""
            # Extract the first two coordinates (x, y) of agent positions.
            # agent_positions = info.agent_positions[:, :2]
            agent_positions = jnp.array([agent.position[:2] for agent in info.agents])
            # Compute Euclidean distance from each agent to its target.
            distance = jnp.linalg.norm(agent_positions - target_position, axis=1)
            # Reward is 1 if the distance is less than the acceptable threshold.
            reward = jnp.where(distance < ACCEPTABLE_DISTANCE, 1.0, 0.0)
            return reward

        return reward_function

    else:
        # Rotation mode: sample a target rotation for each agent.
        target_rotation = jax.random.uniform(
            target_key, (agent_count, 1), minval=0, maxval=3
        )

        def reward_function(info: ICLandInfo) -> jax.Array:
            """Compute reward based on agent rotation relative to a target rotation."""
            # Extract the agent rotations.
            # agent_rotation = info.agent_rotations
            agent_rotation = jnp.array([agent.rotation for agent in info.agents])
            # Compute the absolute difference between agent rotation and target.
            distance = jnp.abs(agent_rotation - target_rotation)
            # Reward is 1 if the rotation difference is within the acceptable threshold.
            reward = jnp.where(distance < ACCEPTABLE_DISTANCE, 1.0, 0.0)
            return reward

        return reward_function
