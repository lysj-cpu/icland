"""Prop generation code."""

from enum import Enum

import jax
import mujoco


class PropType(Enum):
    """Enum for the prop type selection for Props."""

    NONE = 0  # Do not render.
    CUBE = 1
    SPHERE = 2

    def _to_geom(cls) -> mujoco.mjtGeom:
        prop_to_geom = {
            PropType.NONE.value: mujoco.mjtGeom.mjGEOM_BOX,
            PropType.CUBE.value: mujoco.mjtGeom.mjGEOM_BOX,
            PropType.SPHERE.value: mujoco.mjtGeom.mjGEOM_SPHERE,
        }
        return prop_to_geom[cls.value]


def create_prop(
    id: int, pos: jax.Array, spec: mujoco.MjSpec, type: PropType
) -> mujoco.MjSpec:
    """Create an prop in the physics environment.

    Args:
        id: The ID of the prop.
        pos: The initial position of the prop.
        spec: The Mujoco specification object.
        type: The integer value of the prop type enum

    Returns:
        The updated Mujoco specification object.
    """
    prop = spec.worldbody.add_body(
        name=f"prop{id}",
        pos=pos[:3],
    )

    prop.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    prop.add_geom(
        name=f"prop{id}_geom",
        type=type._to_geom(),
        size=[0.1, 0.1, 0.1],
        mass=1,
    )

    return spec
