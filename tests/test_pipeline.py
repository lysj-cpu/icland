from typing import Any  # noqa: D100

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from icland.agent import create_agent
from icland.constants import *
from icland.presets import TEST_TILEMAP_FLAT
from icland.types import *
from icland.world_gen.converter import sample_spawn_points
from icland.world_gen.JITModel import export, sample_world
from icland.world_gen.tile_data import TILECODES


def __generate_mjcf_spec(
    tile_map: jax.Array,
    # agent_spawns: jax.Array,
    # prop_spawns: jax.Array,
) -> mujoco.MjSpec:
    """Generates MJCF spec from column meshes that form the world."""
    spec = mujoco.MjSpec()

    spec.compiler.degree = 1

    w, h = tile_map.shape[0], tile_map.shape[1]
    # Add assets
    # Columns: 1 to 6
    for i in range(1, WORLD_LEVEL + 1):
        spec.add_mesh(
            name=f"c{i}",  # ci for column level i
            uservert=[
                -0.5,
                -0.5,
                0,
                0.5,
                -0.5,
                0,
                0.5,
                0.5,
                0,
                -0.5,
                0.5,
                0,
                -0.5,
                -0.5,
                i,
                0.5,
                -0.5,
                i,
                0.5,
                0.5,
                i,
                -0.5,
                0.5,
                i,
            ],
        )

    # Ramps: 1-2 to 5-6
    for i in range(1, WORLD_LEVEL):
        spec.add_mesh(
            name=f"r{i + 1}",  # ri for ramp to i
            uservert=[
                -0.5,
                -0.5,
                0,
                0.5,
                -0.5,
                0,
                0.5,
                0.5,
                0,
                -0.5,
                0.5,
                0,
                -0.5,
                -0.5,
                i,
                0.5,
                -0.5,
                i + 1,
                0.5,
                0.5,
                i + 1,
                -0.5,
                0.5,
                i,
            ],
        )

    for i in range(w):
        for j in range(h):
            t_type, rot, _, to_h = tile_map[i, j]
            t_type_str = "r" if t_type == 1 else "c"
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                euler=[0, 0, 90 * rot],
                meshname=f"{t_type_str}{to_h}",
                pos=[i + 0.5, j + 0.5, 0],
            )

    # TODO: Add agents
    return spec


def compare_dataclass_instances(
    inst1: Any, inst2: Any, atol: float = 1e-8, rtol: float = 1e-5
) -> None:  # pragma: ignore
    """Compares two JAX/JIT-compatible dataclass instances and print the fields that differ.

    Parameters:
      inst1, inst2: Two instances of the same dataclass.
      atol: Absolute tolerance for numerical comparisons.
      rtol: Relative tolerance for numerical comparisons.
    """
    # Ensure both instances are of the same type.
    if type(inst1) != type(inst2):
        raise TypeError("Both instances must be of the same dataclass type.")

    # Retrieve field names.
    # For dataclasses (including flax.struct.dataclass), __dataclass_fields__ is available.
    field_names = (
        inst1.__dataclass_fields__.keys()
        if hasattr(inst1, "__dataclass_fields__")
        else vars(inst1).keys()
    )

    differences = {}
    for field in field_names:
        value1 = getattr(inst1, field)
        value2 = getattr(inst2, field)

        # If the field contains a JAX/NumPy array, compare numerically.
        if isinstance(value1, jnp.ndarray | np.ndarray):
            # Use jnp.allclose for floating point arrays.
            if not jnp.allclose(value1, value2, atol=atol, rtol=rtol):
                if value1.shape == value2.shape:
                    i, j = 0, 0
                    header1 = ""
                    header2 = ""
                    while i < len(value1) and j < len(value2):
                        if not jnp.allclose(value1[i], value2[j], atol=atol, rtol=rtol):
                            header1 += f"At {i}: {value1[i]}\n"
                            header2 += f"At {j}: {value2[j]}\n"
                        i += 1
                        j += 1
                    differences[field] = (header1, header2)
                else:
                    differences[field] = (
                        f"Shape {value1.shape}\n" + str(value1),
                        f"Shape {value2.shape}\n" + str(value2),
                    )
        elif isinstance(value1, int | bool | float):
            # For non-array values, use the normal equality check.
            # if not jnp.all(value1.data & value2.data):
            if value1 != value2:
                differences[field] = (str(value1), str(value2))
        else:
            differences[field] = (value1, value2)

    # Print only fields with differences.

    # Specify the file path where you want to save the differences
    file_path = "differences.txt"

    # Open the file in write mode
    with open(file_path, "w") as f:
        if differences:
            for field, (val1, val2) in differences.items():
                # Write the differences to the text file
                f.write(
                    f"Field '{field}' differs:\n  Instance 1: {val1}\n  Instance 2: {val2}\n\n"
                )
            print(f"Differences have been exported to {file_path}")

        else:
            print("No differences found.")


def __compare_two_models(tilemap_1: jax.Array, tilemap_2: jax.Array) -> None:
    spec_1 = __generate_mjcf_spec(tilemap_1)
    spec_2 = __generate_mjcf_spec(tilemap_2)

    mj_model_1 = spec_1.compile()
    mj_model_2 = spec_2.compile()

    mjx_model_1 = mjx.put_model(mj_model_1)
    mjx_model_2 = mjx.put_model(mj_model_2)

    compare_dataclass_instances(mjx_model_1, mjx_model_2)


def pipeline(
    key: jax.Array, height: int = 10, width: int = 10, num_agents: int = 1
) -> MjxModelType:  # pragma: no cover
    """Test pipeline."""
    # Sample the world and create tile map.
    MAX_STEPS = 100
    kn = key[1]
    # TODO: Vary the sample world using globally-defined config
    tile_map = export(
        sample_world(height, width, MAX_STEPS, key, True, 1), TILECODES, height, width
    )
    key, s = jax.random.split(key)
    # TODO: Change num_objs from num_agents to num_agents + num_props
    spawnpoints = sample_spawn_points(s, tile_map, num_objects=num_agents)

    # pieces = create_world(tile_map)
    # temp_dir = "temp"
    # export_stls(pieces, f"{temp_dir}/{temp_dir}")
    # xml_str = __generate_mjcf_string(tile_map, (1.5, 1, 4), f"{temp_dir}/")
    # mj_model = mujoco.MjModel.from_xml_string(xml_str)
    mj_spec = __generate_mjcf_spec(tile_map)
    # icland_params = ICLandParams(model=mj_model, game=None, agent_count=1)

    # icland_state = icland.init(key, icland_params)
    mj_model = mj_spec.compile()
    xml_string = mj_spec.to_xml()

    mjx_model = mjx.put_model(mj_model)

    # Save as a .txt file
    # with open(f"model_output_{kn}.txt", "w") as f:
    #     f.write(mjx_model)
    # print(xml_string)

    # output as .txt file
    return mjx_model


if __name__ == "__main__":
    t = TEST_TILEMAP_FLAT
    # for i in range(1, WORLD_LEVEL + 1):
    #     # for j in range(4):
    #     spec_1 = __generate_mjcf_spec(t.at[0, 0].set(jnp.array([0, 0, 0, i])))
    #     mj_model_1 = spec_1.compile()
    #     mjx_model_1 = mjx.put_model(mj_model_1)
    #     print(f"r{i}{i + 1}, rotation {0}: {round(mjx_model_1.geom_rbound[0], 6)}")
    # print(f"Indices for rotations for r{i}{i + 1} are", [(i - 1) * 10 + j for j in range(4)])
    # __compare_two_models(TEST_TILEMAP_FLAT.at[:, :].set(jnp.array([0, 0, 0, 1])), t)

    # t = TEST_TILEMAP_FLAT
    spec_1 = __generate_mjcf_spec(t)
    spec_2 = __generate_mjcf_spec(t)
    # spec_2 = create_agent(0, jnp.array([1, 1, 3.5]), spec_2)
    spec_1 = create_agent(0, jnp.array([8, 8, 3.5]), spec_1)
    spec_1 = create_agent(1, jnp.array([8, 9, 3.5]), spec_1)
    spec_2 = create_agent(0, jnp.array([0, 0, 0]), spec_2)
    spec_2 = create_agent(1, jnp.array([0, 0, 0]), spec_2)

    mj_model_1 = spec_1.compile()
    mj_model_2 = spec_2.compile()
    mjx_model_1 = mjx.put_model(mj_model_1)
    mjx_model_2 = mjx.put_model(mj_model_2)
    compare_dataclass_instances(mjx_model_1, mjx_model_2)
