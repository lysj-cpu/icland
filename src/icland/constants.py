"""This module defines constants used throughout the icland project.

These constants include debug camera settings, agent movement parameters,
and other small values.
"""

# Small Values
SMALL_VALUE = 1e-10

# Agent Parameters
AGENT_DRIVING_FORCE = 100
AGENT_ROTATION_SPEED = 5e-3
AGENT_PITCH_SPEED = 0.1
AGENT_MAX_MOVEMENT_SPEED = 1.0
AGENT_MAX_ROTATION_SPEED = 1
AGENT_ROTATIONAL_FRICTION_COEFFICIENT = 1.0
AGENT_MOVEMENT_FRICTION_COEFFICIENT = 0.1
AGENT_MAX_CLIMBABLE_STEEPNESS = 0.7
AGENT_HEIGHT = 0.4
AGENT_RADIUS = 0.06

# Dimensionality Constants
AGENT_VARIABLES_DIM = 2
PROP_VARIABLES_DIM = 1
ACTION_SPACE_DIM = 6

# World and Level Constants
WORLD_LEVEL = 6

# Offsets
BODY_OFFSET = 1
WALL_OFFSET = 5
AGENT_DOF_OFFSET = 4
PROP_DOF_OFFSET = 4
PROP_DOF_MULTIPLIER = 6

# Simulation Constants
FPS = 30
SPS = 150