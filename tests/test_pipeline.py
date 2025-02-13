import icland
from icland.world_gen.converter import *
from icland.types import *
import jax
import jax.numpy as jnp
import os

def __generate_mjcf_string(  # pragma: no cover
    tile_map: jax.Array,
    agent_spawns: jax.Array,
    mesh_dir: str = "meshes/",
) -> str:
    """Generates MJCF file from column meshes that form the world."""
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(".stl")]

    mjcf = f"""<mujoco model=\"generated_mesh_world\">
    <compiler meshdir=\"{mesh_dir}\"/>
    <default>
        <geom type=\"mesh\" />
    </default>
    
    <worldbody>\n"""

    for i, s in enumerate(agent_spawns):
        spawn_loc = s.tolist()
        spawn_loc[2] += 1
        mjcf += f'            <body name="agent{i}" pos="{" ".join([str(s) for s in spawn_loc])}">'
        mjcf += """
            <joint type="slide" axis="1 0 0" />
            <joint type="slide" axis="0 1 0" />
            <joint type="slide" axis="0 0 1" />
            <joint type="hinge" axis="0 0 1" stiffness="1"/>

            <geom"""
        mjcf += f'        name="agent{i}_geom"'
        mjcf += """        type="capsule"
                size="0.06"
                fromto="0 0 0 0 0 -0.4"
                mass="1"
            />

            <geom
                type="box"
                size="0.05 0.05 0.05"
                pos="0 0 0.2"
                mass="0"
            />
        </body>"""

    for i, mesh_file in enumerate(mesh_files):
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += (
            f'        <geom name="{mesh_name}" mesh="{mesh_name}" pos="0 0 0"/>' + "\n"
        )

    mjcf += "    </worldbody>\n\n    <asset>\n"

    for mesh_file in mesh_files:
        mesh_name = os.path.splitext(mesh_file)[0]
        mjcf += f'        <mesh name="{mesh_name}" file="{mesh_file}"/>' + "\n"

    mjcf += "    </asset>\n</mujoco>\n"

    return mjcf

def pipeline(key: int, tile_map: jnp.ndarray) -> MjxModelType:
    pieces = create_world(tile_map)
    temp_dir = "temp"
    export_stls(pieces, f"{temp_dir}/{temp_dir}")
    xml_str = __generate_mjcf_string(tile_map, (1.5, 1, 4), f"{temp_dir}/")
    mj_model = mujoco.MjModel.from_xml_string(xml_str)
    icland_params = ICLandParams(model=mj_model, game=None, agent_count=1)

    icland_state = icland.init(key, icland_params)
    return icland_state.pipeline_state.mjx_model

