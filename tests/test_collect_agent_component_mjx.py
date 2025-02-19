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
    xml_string = EMPTY_WORLD
    key = jax.random.key(42) 
    mj_model = mujoco.MjModel.from_xml_string(xml_string)
    icland_params = ICLandParams(model=mj_model, reward_function=None, agent_count=1)

    icland_state = icland.init(key, icland_params)
    mjx_model = icland_state.pipeline_state.mjx_model
    res = collect_agent_components_mjx(mjx_model, 1)
    print(res)
    
if __name__ == "__main__":
    test_collect_agent_components()
