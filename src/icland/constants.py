"""This module defines constants used throughout the icland project.

These constants include debug camera settings, agent movement parameters,
and other small values.
"""

DEBUG_CAMERA_DISTANCE = 3.0
DEBUG_CAMERA_AZIMUTH = 90.0
DEBUG_CAMERA_ELEVATION = -40.0
DEBUG_CAMERA_VIS_FLAGS = ["mjVIS_JOINT", "mjVIS_CONTACTFORCE"]

AGENT_MAX_MOVEMENT_SPEED = 1.0
AGENT_MAX_ROTATION_SPEED = 5.0
AGENT_MOVEMENT_FRICTION_COEFFICIENT = 0.1
AGENT_MAX_CLIMBABLE_STEEPNESS = 0.7
SMALL_VALUE = 1e-10
