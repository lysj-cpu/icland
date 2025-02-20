from mujoco import mjx
import jax.numpy as jnp
import icland
from icland import *
from icland.presets import *

def test_collect_agent_components():
    # Load a sample MJX model (replace with your actual XML path)
    # key = jax.random.key(42) 
    # params = sample(key)
    # state = init(key, params)
    # mjx_model = state.pipeline_state.mjx_model
    # # print(mjx_model)
    # agent_count = 2  # Adjust based on the number of agents in the XML
    # print(collect_agent_components(params.model, agent_count))
    # print(mjx_model.name_bodyadr)
    # agent_components = collect_agent_components_mjx(mjx_model, agent_count)

    # print("Agent Components:\n", agent_components)
    xml_string = """
        <mujoco>
        <worldbody>
            <light name="main_light" pos="0 0 1" dir="0 0 -1"
                diffuse="1 1 1" specular="0.1 0.1 0.1"/>

            <body name="agent0" pos="0 0 1">
            <joint type="slide" axis="1 0 0" />
            <joint type="slide" axis="0 1 0" />
            <joint type="slide" axis="0 0 1" />
            <joint type="hinge" axis="0 0 1" stiffness="1"/>

            <geom
                name="agent0_geom"
                type="capsule"
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
            </body>

            <body name="prop0" pos="1 0 1">
                <freejoint />
                <geom
                    name="prop0_geom"
                    type="box"
                    size="0.1 0.1 0.1"
                    mass="1"
                />
                </body>

            <body name="prop1" pos="2 0 2">
                <freejoint />
                <geom
                    name="prop1_geom"
                    type="box"
                    size="0.1 0.1 0.1"
                    mass="1"
                />
                </body>

            <body name="prop2" pos="2 0 2">
                <freejoint />
                <geom
                    name="prop2_geom"
                    type="box"
                    size="0.1 0.1 0.1"
                    mass="1"
                />
                </body>

            <!-- Ground plane, also with low friction -->
            <geom
            name="ground"
            type="plane"
            size="0 0 0.01"
            rgba="1 1 1 1"
            />

        </worldbody>
        </mujoco>
        """
    key = jax.random.key(42) 
    mj_model = mujoco.MjModel.from_xml_string(xml_string)
    icland_params = ICLandParams(model=mj_model, reward_function=None, agent_count=1)

    icland_state = icland.init(key, icland_params)
    mjx_model = icland_state.pipeline_state.mjx_model
    res = collect_prop_components_mjx(mjx_model, 0, 0, 3, 1)
    print(res)
    res = collect_prop_components(mj_model, 3)
    print(res)
    
if __name__ == "__main__":
    test_collect_agent_components()