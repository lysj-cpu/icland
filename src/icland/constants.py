"""This module defines constants used throughout the icland project.

These constants include debug camera settings, agent movement parameters,
and other small values.
"""

# Small Values
SMALL_VALUE = 1e-10

# Agent Parameters
AGENT_DRIVING_FORCE = 100
AGENT_ROTATION_SPEED = 5e-2
AGENT_PITCH_SPEED = 0.02
AGENT_MAX_MOVEMENT_SPEED = 1.0
AGENT_MAX_ROTATION_SPEED = 1.0
AGENT_ROTATIONAL_FRICTION_COEFFICIENT = 1.0
AGENT_MOVEMENT_FRICTION_COEFFICIENT = 0.1
AGENT_MAX_CLIMBABLE_STEEPNESS = 0.7
AGENT_HEIGHT = 0.4
AGENT_RADIUS = 0.06
AGENT_MAX_TAG_DISTANCE = 0.5
AGENT_TAG_SECS_OUT = 1
AGENT_GRAB_RANGE = 1
AGENT_GRAB_DURATION = 1

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
PHYS_PER_CTRL_STEP = 5
TIMESTEP = 0.01
