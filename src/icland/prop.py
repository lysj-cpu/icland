"""Prop generation code."""

import jax
import mujoco


class PropType(Enum):
    """Enum for the prop type selection for Props."""

    NONE = 0  # Do not render.
    CUBE = 1
    SPHERE = 2

    @classmethod
    def _to_geom(cls):
        prop_to_geom = {
            PropType.NONE: mujoco.mjtGeom.mjGEOM_NONE,
            PropType.CUBE: mujoco.mjtGeom.mjGEOM_BOX,
            PropType.SPHERE: mujoco.mjtGeom.mjGEOM_SPHERE,
        }
        return prop_to_geom[cls]


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
        pos=pos[: (AGENT_COMPONENT_IDS_DIM - 1)],
    )

    prop.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    prop.add_geom(
        name=f"prop{id}_geom",
        type=type._to_geom(),
        size=[0.1, 0.1, 0.1],
        mass=1,
    )

    return spec
