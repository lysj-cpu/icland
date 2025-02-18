import jax

from icland.game import generate_reward_function
from icland.types import ICLandInfo

key = jax.random.key(0)
info = ICLandInfo(agents=["agent1", "agent2", "agent3"], props=["prop1", "prop2"])

print(generate_reward_function(key, info))
