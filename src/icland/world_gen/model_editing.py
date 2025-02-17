from typing import Any, cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from icland.agent import create_agent
from icland.constants import WORLD_HEIGHT
from icland.presets import TEST_TILEMAP_BUMP
from icland.types import MjxModelType


def generate_base_model(
    width: int,
    height: int,
    max_num_agents: int,
    max_height: int = 6,  # controls the BVH bounding box max height.
    # prop_spawns: jax.Array,
) -> MjxModelType:  # pragma: no cover
    """Generates base MJX model from column meshes that form the world."""
    # This code is run entirely on CPU
    spec = mujoco.MjSpec()

    spec.compiler.degree = 1

    # Add assets
    # Columns: 1 to 6
    for i in range(1, WORLD_HEIGHT + 1):
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
    for i in range(1, WORLD_HEIGHT):
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

    for i in range(width):
        for j in range(height):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="c6",
                pos=[i + 0.5, j + 0.5, 0],
                contype=1,
                conaffinity=2,
            )

    for a in range(max_num_agents):
        # Initialize them at placeholder position (0, 0, 0)
        spec = create_agent(a, jnp.array([0, 0, 0]), spec)

    mj_model = spec.compile()
    mjx_model = mujoco.mjx.put_model(mj_model)
    
    def numpy_to_jax(att: Any) -> Any:
        if isinstance(att, np.ndarray):
            return jnp.array(att)
        return att
    
    mjx_model_j = jax.tree_util.tree_map(numpy_to_jax, mjx_model)
    print(type(mjx_model_j.geom_dataid))
    return mjx_model_j


def edit_model(tilemap, base_model: MjxModelType, max_height: int = 6) -> MjxModelType:
    """Edit the base model such that the terrain matches that of the tilemap."""
    # Pre: the width and height of the tilemap MUST MATCH that of the base_model

    RAMP_ID = 1
    RAMP_OFFSET = 2
    RAMP_FRAME = 0
    COL_FRAME = 3

    # Extract geom information from base model

    b_geom_dataid = base_model.geom_dataid
    b_geom_sameframe = base_model.geom_sameframe
    b_geom_size = jnp.array(base_model.geom_size)
    b_geom_aabb = base_model.geom_aabb
    b_geom_rbound = base_model.geom_rbound
    b_geom_rbound_hfield = base_model.geom_rbound_hfield
    b_geom_pos = base_model.geom_pos
    b_geom_quat = base_model.geom_quat

    mesh_pos = base_model.mesh_pos
    mesh_pos = jax.lax.dynamic_update_slice_in_dim(
        mesh_pos,
        jnp.array(
            [
                [0.055556, 0.0, 0.777778],
                [0.033333, 0.0, 1.266667],
                [0.02381, 0.0, 1.761905],
                [0.018519, 0.0, 2.259259],
                [0.015152, 0.0, 2.757576],
            ]
        ),
        max_height,
        axis=0,
    )

    mesh_quat = base_model.mesh_quat
    mesh_quat = jax.lax.dynamic_update_slice_in_dim(
        mesh_quat,
        jnp.array(
            [
                [0.992309, 0.0, 0.123787, 0.0],
                [0.999101, 0.0, 0.0424, 0.0],
                [0.999778, 0.0, 0.021071, 0.0],
                [0.999921, 0.0, 0.012593, 0.0],
                [0.999965, 0.0, 0.008376, 0.0],
            ]
        ),
        max_height,
        axis=0,
    )

    w, h = tilemap.shape[0], tilemap.shape[1]

    def process_tile(i: int, tile):
        t_type, rot, _, to_h = tile

        # Check if tile type is ramp or column
        is_ramp = t_type % 2

        # geom_dataid = geom_dataid.at[i].set(
        #     is_ramp * max_height + to_h + is_ramp * -RAMP_OFFSET + (is_ramp - 1)
        # )
        dataid = is_ramp * max_height + to_h + is_ramp * -RAMP_OFFSET + (is_ramp - 1)
        # geom_sameframe = geom_sameframe.at[i].set(
        #     is_ramp * RAMP_FRAME + (1 - is_ramp) * COL_FRAME
        # )
        sameframe = jax.lax.convert_element_type(
            is_ramp * RAMP_FRAME + (1 - is_ramp) * COL_FRAME, jnp.uint8
        )

        def __process_column_size(h: float):
            return jnp.array([0.5, 0.5, jnp.round(h / 2, 1)])

        def __process_ramp_size(h: float):
            ramp_sizes = jnp.array(
                [
                    [0.62190086, 0.5, 1.2939521],
                    [0.59354615, 0.5, 1.7666386],
                    [0.57550883, 0.5, 2.2561712],
                    [0.56219345, 0.5, 2.7519972],
                    [0.5526448, 0.5, 3.2500916],
                ]
            )
            return ramp_sizes[h - RAMP_OFFSET]

        tile_size = jax.lax.cond(
            is_ramp, __process_ramp_size, __process_column_size, to_h
        )
        # geom_size = geom_size.at[i].set(tile_size)

        def __process_column_aabb(h: float):
            return jnp.array([0, 0, 0, 0.5, 0.5, jnp.round(h / 2, 1)])

        def __process_ramp_aabb(h: float):
            ramp_aabbs = jnp.array(
                [
                    [0.01438885, 0.0, 0.20176348, 0.60751198, 0.5, 1.09218866],
                    [-0.01062064, 0.0, 0.2296703, 0.58292548, 0.5, 1.53696839],
                    [-0.01275361, 0.0, 0.2368807, 0.56275523, 0.5, 2.01929044],
                    [-0.01198336, 0.0, 0.24019804, 0.55021008, 0.5, 2.51179923],
                    [-0.01083441, 0.0, 0.24213641, 0.54181035, 0.5, 3.00795512],
                ]
            )
            return ramp_aabbs[h - RAMP_OFFSET]

        tile_aabb = jax.lax.cond(
            is_ramp, __process_ramp_aabb, __process_column_aabb, to_h
        )
        # geom_aabb = geom_aabb.at[i].set(tile_aabb)

        def __process_column_rbound(h: int):
            col_rbound = jnp.array(
                [
                    0.8660249710083008,
                    1.2247450351715088,
                    1.6583119630813599,
                    2.1213200092315674,
                    2.598076105117798,
                    3.082206964492798,
                ]
            )
            return col_rbound[h]

        def __process_ramp_rbound(h: int):
            ramp_rbound = jnp.array(
                [
                    1.5202213525772095,
                    1.9295878410339355,
                    2.3814949989318848,
                    2.852989673614502,
                    3.3344430923461914,
                ]
            )
            return ramp_rbound[h - RAMP_OFFSET]

        tile_rbound = jax.lax.cond(
            is_ramp, __process_ramp_rbound, __process_column_rbound, to_h
        )
        # geom_rbound = geom_rbound.at[i].set(tile_rbound)
        # geom_rbound_hfield = geom_rbound_hfield.at[i].set(tile_rbound)

        def __process_column_position(h: int, rot: int):
            return jnp.array([(i // w) + 0.5, (i % w) + 0.5, jnp.round(h / 2, 1)])

        def __process_ramp_position(h: int, rot: int):
            ramp_pos = jnp.array(
                [
                    [
                        [0.5555555820465088, 0.5, 0.7777777910232544],
                        [0.5, 0.5555555820465088, 0.7777777910232544],
                        [0.4444444477558136, 0.5, 0.7777777910232544],
                        [0.5, 0.4444444477558136, 0.7777777910232544],
                    ],
                    [
                        [0.5333333611488342, 0.5, 1.2666666507720947],
                        [0.5, 0.5333333611488342, 1.2666666507720947],
                        [0.46666666865348816, 0.5, 1.2666666507720947],
                        [0.5, 0.46666666865348816, 1.2666666507720947],
                    ],
                    [
                        [0.523809552192688, 0.5, 1.7619047164916992],
                        [0.5, 0.523809552192688, 1.7619047164916992],
                        [0.4761904776096344, 0.5, 1.7619047164916992],
                        [0.5, 0.4761904776096344, 1.7619047164916992],
                    ],
                    [
                        [0.5185185074806213, 0.5, 2.2592592239379883],
                        [0.5, 0.5185185074806213, 2.2592592239379883],
                        [0.48148149251937866, 0.5, 2.2592592239379883],
                        [0.5, 0.48148149251937866, 2.2592592239379883],
                    ],
                    [
                        [0.5151515007019043, 0.5, 2.757575750350952],
                        [0.5, 0.5151515007019043, 2.757575750350952],
                        [0.4848484992980957, 0.5, 2.757575750350952],
                        [0.5, 0.4848484992980957, 2.757575750350952],
                    ],
                ]
            )
            return ramp_pos[h - RAMP_OFFSET, rot]

        tile_pos = jax.lax.cond(
            is_ramp, __process_ramp_position, __process_column_position, to_h, rot
        )
        # geom_pos = geom_pos.at[i].set(tile_pos)

        def __process_column_quat(h: int, rot: int):
            return jnp.array([1, 0, 0, 0], dtype=float)

        def __process_ramp_quat(h: int, rot: int):
            ramp_quats = jnp.array(
                [
                    [
                        [0.9923087954521179, 0.0, 0.12378735840320587, 0.0],
                        [
                            0.7016682624816895,
                            -0.08753088116645813,
                            0.08753088116645813,
                            0.7016682624816895,
                        ],
                        [
                            6.07613856931401e-17,
                            -0.12378735840320587,
                            7.579789306610251e-18,
                            0.9923087954521179,
                        ],
                        [
                            -0.7016682624816895,
                            -0.08753088116645813,
                            -0.08753088116645813,
                            0.7016682624816895,
                        ],
                    ],
                    [
                        [0.9991007447242737, 0.0, 0.042399730533361435, 0.0],
                        [
                            0.7064709067344666,
                            -0.029981136322021484,
                            0.029981136322021484,
                            0.7064709067344666,
                        ],
                        [
                            6.117727225279706e-17,
                            -0.042399730533361435,
                            2.5962347722877762e-18,
                            0.9991007447242737,
                        ],
                        [
                            -0.7064709067344666,
                            -0.029981136322021484,
                            -0.029981136322021484,
                            0.7064709067344666,
                        ],
                    ],
                    [
                        [0.9997779726982117, 0.0, 0.02107108198106289, 0.0],
                        [
                            0.706949770450592,
                            -0.014899504370987415,
                            0.014899504370987415,
                            0.706949770450592,
                        ],
                        [
                            6.121874377998802e-17,
                            -0.02107108198106289,
                            1.2902315631716904e-18,
                            0.9997779726982117,
                        ],
                        [
                            -0.706949770450592,
                            -0.014899504370987415,
                            -0.014899504370987415,
                            0.706949770450592,
                        ],
                    ],
                    [
                        [0.9999207258224487, 0.0, 0.012593165040016174, 0.0],
                        [
                            0.7070506811141968,
                            -0.008904712274670601,
                            0.008904712274670601,
                            0.7070506811141968,
                        ],
                        [
                            6.122748542470148e-17,
                            -0.012593165040016174,
                            7.711089339602019e-19,
                            0.9999207258224487,
                        ],
                        [
                            -0.7070506811141968,
                            -0.008904712274670601,
                            -0.008904712274670601,
                            0.7070506811141968,
                        ],
                    ],
                    [
                        [0.9999648928642273, 0.0, 0.008376396261155605, 0.0],
                        [
                            0.7070819735527039,
                            -0.005923006683588028,
                            0.005923006683588028,
                            0.7070819735527039,
                        ],
                        [
                            6.123019195966575e-17,
                            -0.008376396261155605,
                            5.129063669081525e-19,
                            0.9999648928642273,
                        ],
                        [
                            -0.7070819735527039,
                            -0.005923006683588028,
                            -0.005923006683588028,
                            0.7070819735527039,
                        ],
                    ],
                ]
            )
            return ramp_quats[h - RAMP_OFFSET, rot]

        tile_quat = jax.lax.cond(
            is_ramp, __process_ramp_quat, __process_column_quat, to_h, rot
        )
        # geom_quat = geom_quat.at[i].set(tile_quat)
        return (
            dataid,
            sameframe,
            tile_pos,
            tile_quat,
            tile_size,
            tile_aabb,
            tile_rbound,
        )

    (
        g_dataid,
        g_sameframe,
        g_pos,
        g_quat,
        g_size,
        g_aabb,
        g_rbound,
    ) = jax.vmap(process_tile, in_axes=(0, 0))(
        jnp.arange(w * h, dtype=int), jnp.reshape(tilemap, (w * h, -1))
    )

    # b_geom_dataid = b_geom_dataid.at[0].set(0)
    # print(type(b_geom_dataid))

    # return base_model.replace(
    #     geom_dataid = b_geom_dataid
    # )
    detaid = jax.lax.dynamic_update_slice_in_dim(b_geom_dataid, g_dataid, 0, axis=0)
    sameframe = jax.lax.dynamic_update_slice_in_dim(b_geom_sameframe, g_sameframe, 0, axis=0)
    pos = jax.lax.dynamic_update_slice_in_dim(b_geom_pos, g_pos, 0, axis=0)
    quat = jax.lax.dynamic_update_slice_in_dim(b_geom_quat, g_quat, 0, axis=0)
    size = jax.lax.dynamic_update_slice_in_dim(b_geom_size, g_size, 0, axis=0)
    aabb = jax.lax.dynamic_update_slice_in_dim(b_geom_aabb, g_aabb, 0, axis=0)
    rbound = jax.lax.dynamic_update_slice_in_dim(b_geom_rbound, g_rbound, 0, axis=0)

    model = base_model.replace(
        geom_dataid=detaid,
        geom_sameframe=sameframe,
        geom_pos=pos,
        geom_quat=quat,
        geom_size=size,
        geom_aabb=aabb,
        geom_rbound=rbound,
        mesh_pos=mesh_pos,
        mesh_quat=mesh_quat,
    )
    return model


if __name__ == "__main__":
    mjx_model_j = generate_base_model(10, 10, 1)
    jit_edit_model = jax.jit(edit_model)
    mjx_model_j = jit_edit_model(TEST_TILEMAP_BUMP, mjx_model_j)
