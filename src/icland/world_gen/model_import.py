"""Code to convert an initial model to MJX models."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from icland.types import MjxModelType


def create_initial_model(
    num_meshes: int = 100, tris_per_mesh: int = 24
) -> MjxModelType:  # pragma: no cover
    """Initialize model with placeholder meshes."""
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
            <worldbody>
                <!-- Meshes will be added programmatically -->
            </worldbody>
            <asset>
                <!-- Mesh assets will be added programmatically -->
            </asset>
        </mujoco>
    """)
    mj_model = mjx.put_model(model)

    # Preallocate mesh data arrays
    mj_model = mj_model.replace(
        mesh_vert=jnp.zeros((num_meshes * tris_per_mesh * 3, 3), dtype=jnp.float32),
        mesh_face=jnp.zeros((num_meshes * tris_per_mesh, 3), dtype=np.int32),
        mesh_vertadr=jnp.zeros(num_meshes, dtype=np.int32),
        mesh_vertnum=jnp.zeros(num_meshes, dtype=np.int32),
        mesh_faceadr=jnp.zeros(num_meshes, dtype=np.int32),
        nmesh=num_meshes,
    )

    # Configure mesh metadata
    for i in range(num_meshes):
        start_idx = i * tris_per_mesh * 3
        mj_model = mj_model.replace(
            mesh_vertadr=mj_model.mesh_vertadr.at[i].set(i * tris_per_mesh * 3),
            mesh_vertnum=mj_model.mesh_vertnum.at[i].set(tris_per_mesh * 3),
            mesh_faceadr=mj_model.mesh_faceadr.at[i].set(i * tris_per_mesh),
            mesh_face=jax.lax.dynamic_update_slice(
                mj_model.mesh_face,
                jnp.full(
                    (tris_per_mesh, 3),
                    jnp.array([start_idx, start_idx + 1, start_idx + 2]),
                ),
                (i * tris_per_mesh, 3),
            ),
        )

    return mj_model


def update_meshes(
    mj_model: MjxModelType, pieces: jax.Array
) -> MjxModelType:  # pragma: no cover
    """Update mesh vertices directly from JAX array."""
    # Convert JAX array to numpy and reverse vertex order for normals
    pieces_reshaped = pieces.reshape(-1, *pieces.shape[-2:])
    jax_triangles = np.array(pieces_reshaped)
    triangles = np.asarray(jax_triangles[::-1], dtype=np.float32)
    triangles = triangles.reshape(-1, 3, 3)  # (num_meshes*tris_per_mesh, 3, 3)

    # Update vertex positions in-place
    for mesh_idx in range(mj_model.nmesh):
        start = mj_model.mesh_vertadr[mesh_idx]
        end = start + mj_model.mesh_vertnum[mesh_idx]
        mesh_tris = triangles[mesh_idx :: mj_model.nmesh]
        mesh_vert = mj_model.mesh_vert
        mesh_vert = mesh_vert.at[start:end].set(mesh_tris.reshape(-1, 3))
        mj_model = mj_model.replace(mesh_vert=mesh_vert)

    return mj_model
