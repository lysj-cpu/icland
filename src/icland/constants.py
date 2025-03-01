"""This module defines constants used throughout the icland project.

These constants include debug camera settings, agent movement parameters,
and other small values.
"""

SMALL_VALUE = 1e-10

WORL_MIN_AGENT_COUNT = 1
WORLD_MAX_AGENT_COUNT = 5

AGENT_ACTION_SPACE_DIM = 4
AGENT_COMPONENT_IDS_DIM = 4
AGENT_DRIVING_FORCE = 100
AGENT_ROTATION_SPEED = 5e-3
AGENT_PITCH_SPEED = 0.1
AGENT_OBSERVATION_DIM = 4
AGENT_MAX_MOVEMENT_SPEED = 1.0
AGENT_ROTATIONAL_FRICTION_COEFFICIENT = 1.0
AGENT_MOVEMENT_FRICTION_COEFFICIENT = 0.1
AGENT_MAX_CLIMBABLE_STEEPNESS = 0.7
AGENT_MAX_ROTATION_SPEED = 1

WORLD_LEVEL = 6

BODY_OFFSET = 1
WALL_OFFSET = 5

PROP_DOF_OFFSET = 4
PROP_DOF_MULTIPLIER = 6
