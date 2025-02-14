"""This module defines several policies and worlds for use in testing.

Policies:
    NOOP_POLICY: A policy that represents no operation, with all zeros.
    FORWARD_POLICY: A policy that represents moving forward, with a value of 1 in the first position.
    BACKWARD_POLICY: A policy that represents moving backward, with a value of -1 in the first position.
    LEFT_POLICY: A policy that represents moving left, with a value of -1 in the second position.
    RIGHT_POLICY: A policy that represents moving right, with a value of 1 in the second position.
    ANTI_CLOCKWISE_POLICY: A policy that represents turning anti-clockwise, with a value of -1 in the third position.
    CLOCKWISE_POLICY: A policy that represents turning clockwise, with a value of 1 in the third position.

Worlds:
    EMPTY_WORLD: An empty world with a single agent.
    TWO_AGENT_EMPTY_WORLD: An empty world with two agents.
    TWO_AGENT_EMPTY_WORLD_COLLIDE: An empty world with two agents that collide.
    RAMP_30: A world with a 30 degree ramp.
    RAMP_45: A world with a 45 degree ramp.
    RAMP_60: A world with a 60 degree ramp.
    WORLD_42_CONVEX: A pre-generated world using seed 42 with ramps and terrain.

Frames:
    TEST_TILEMAP_FLAT: A 10x10 flat world of height 3.
    TEST_TILEMAP_BUMP: A 10x10 flat world but with a bump of height 5 at z=5.
    TEST_FRAME: A 10x10 low-res frame of the rendered tilemap.
    TEST_FRAME_WITH_PROPS: A 10x10 low-res frame of the render with props.
"""

import jax.numpy as jnp

# ========================
# Policy Definitions
# ========================

NOOP_POLICY = jnp.array([0, 0, 0])
FORWARD_POLICY = jnp.array([1, 0, 0])
BACKWARD_POLICY = jnp.array([-1, 0, 0])
LEFT_POLICY = jnp.array([0, 1, 0])
RIGHT_POLICY = jnp.array([0, -1, 0])
ANTI_CLOCKWISE_POLICY = jnp.array([0, 0, -1])
CLOCKWISE_POLICY = jnp.array([0, 0, 1])


# ========================
# World Definitions
# ========================

EMPTY_WORLD = """
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

TWO_AGENT_EMPTY_WORLD = """
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

    <body name="agent1" pos="0 0.5 1">
      <joint type="slide" axis="1 0 0" />
      <joint type="slide" axis="0 1 0" />
      <joint type="slide" axis="0 0 1" />
      <joint type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="agent1_geom"
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

TWO_AGENT_EMPTY_WORLD_COLLIDE = """
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

    <body name="agent1" pos="1 0 1">
      <joint type="slide" axis="1 0 0" />
      <joint type="slide" axis="0 1 0" />
      <joint type="slide" axis="0 0 1" />
      <joint type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="agent1_geom"
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

RAMP_30 = """
<mujoco>
  <worldbody>
    <light name="main_light" pos="0 0 1" dir="0 0 -1"
           diffuse="1 1 1" specular="0.1 0.1 0.1"/>

    <body name="agent0" pos="0 0 1">
      <joint type="slide" axis="1 0 0" />
      <joint type="slide" axis="0 1 0" />
      <joint type="slide" axis="0 0 1" />
      <joint type="hinge" axis="0 0 1" />

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

    <!-- Ground plane, also with low friction -->
    <geom
      name="ground"
      type="plane"
      size="0 0 0.01"
      rgba="1 1 1 1"
    />

    <geom type="box" size="1 1 1" pos="2 0 -0.5" euler="0 60 0" rgba="1 0.8 0.8 1" />

  </worldbody>
</mujoco>
"""

RAMP_45 = """
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

    <!-- Ground plane, also with low friction -->
    <geom
      name="ground"
      type="plane"
      size="0 0 0.01"
      rgba="1 1 1 1"
    />

    <geom type="box" size="1 1 1" pos="2 0 -0.5" euler="0 45 0" rgba="1 0.8 0.8 1" />

  </worldbody>
</mujoco>
"""

RAMP_60 = """
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
    </body>

    <!-- Ground plane, also with low friction -->
    <geom
      name="ground"
      type="plane"
      size="0 0 0.01"
      rgba="1 1 1 1"
    />

    <geom type="box" size="1 1 1" pos="2 0 -0.5" euler="0 30 0" rgba="1 0.8 0.8 1" name="ramp" />

  </worldbody>
</mujoco>
"""

TEST_TILEMAP_FLAT = jnp.array([[[0, 0, 0, 3] for _ in range(10)] for _ in range(10)])
TEST_TILEMAP_BUMP = TEST_TILEMAP_FLAT.at[:, 5].set(jnp.array([0, 0, 0, 5]))
TEST_TILEMAP_MAX_HEIGHT = jnp.array([[[0, 0, 0, 6] for _ in range(10)] for _ in range(10)])
TEST_FRAME = jnp.array(
    [
        [
            [0.701171875, 0.701171875, 0.701171875],
            [0.63037109375, 0.63037109375, 0.63037109375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
        ],
        [
            [0.26025390625, 0.360107421875, 0.47216796875],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.260009765625, 0.360107421875, 0.47216796875],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.61474609375, 0.61474609375, 0.61474609375],
            [0.70458984375, 0.70458984375, 0.70458984375],
        ],
        [
            [0.268798828125, 0.372314453125, 0.488037109375],
            [0.265625, 0.367919921875, 0.482177734375],
            [0.2626953125, 0.36376953125, 0.47705078125],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.2587890625, 0.3583984375, 0.4697265625],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5927734375, 0.5927734375, 0.5927734375],
            [0.70068359375, 0.70068359375, 0.70068359375],
            [0.703125, 0.703125, 0.703125],
            [0.58642578125, 0.58642578125, 0.58642578125],
        ],
        [
            [0.260009765625, 0.360107421875, 0.47216796875],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.259765625, 0.359619140625, 0.4716796875],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.38134765625, 0.38134765625, 0.38134765625],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
        ],
        [
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.260009765625, 0.35986328125, 0.471923828125],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.26025390625, 0.360107421875, 0.47216796875],
            [0.26025390625, 0.3603515625, 0.472412109375],
            [0.38134765625, 0.38134765625, 0.38134765625],
            [0.62158203125, 0.62158203125, 0.62158203125],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.69677734375, 0.69677734375, 0.69677734375],
            [0.5859375, 0.5859375, 0.5859375],
        ],
        [
            [0.317138671875, 0.317138671875, 0.317138671875],
            [0.38134765625, 0.38134765625, 0.38134765625],
            [0.317138671875, 0.317138671875, 0.317138671875],
            [0.319580078125, 0.319580078125, 0.319580078125],
            [0.38134765625, 0.38134765625, 0.38134765625],
            [0.317138671875, 0.317138671875, 0.317138671875],
            [0.66748046875, 0.66748046875, 0.66748046875],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
        ],
        [
            [0.70703125, 0.70703125, 0.70703125],
            [0.587890625, 0.587890625, 0.587890625],
            [0.705078125, 0.705078125, 0.705078125],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
        ],
        [
            [0.6689453125, 0.6689453125, 0.6689453125],
            [0.6650390625, 0.6650390625, 0.6650390625],
            [0.65478515625, 0.65478515625, 0.65478515625],
            [0.6494140625, 0.6494140625, 0.6494140625],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
        ],
        [
            [0.7724609375, 0.7724609375, 0.7724609375],
            [0.6640625, 0.6640625, 0.6640625],
            [0.7314453125, 0.7314453125, 0.7314453125],
            [0.70849609375, 0.70849609375, 0.70849609375],
            [0.58837890625, 0.58837890625, 0.58837890625],
            [0.703125, 0.703125, 0.703125],
            [0.587890625, 0.587890625, 0.587890625],
            [0.587890625, 0.587890625, 0.587890625],
            [0.703125, 0.703125, 0.703125],
            [0.587890625, 0.587890625, 0.587890625],
        ],
        [
            [0.8427734375, 0.8427734375, 0.8427734375],
            [0.740234375, 0.740234375, 0.740234375],
            [0.76953125, 0.76953125, 0.76953125],
            [0.7080078125, 0.7080078125, 0.7080078125],
            [0.587890625, 0.587890625, 0.587890625],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.60107421875, 0.60107421875, 0.60107421875],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.58642578125, 0.58642578125, 0.58642578125],
        ],
    ]
)
TEST_FRAME_WITH_PROPS = jnp.array(
    [
        [
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.68994140625, 0.68994140625, 0.68994140625],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
        ],
        [
            [0.48095703125, 0.350830078125, 0.0],
            [0.35107421875, 0.48095703125, 0.35107421875],
            [0.48095703125, 0.35107421875, 0.0],
            [0.35107421875, 0.48095703125, 0.35107421875],
            [0.421875, 0.350830078125, 0.421875],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.61474609375, 0.61474609375, 0.61474609375],
            [0.70458984375, 0.70458984375, 0.70458984375],
        ],
        [
            [0.4970703125, 0.36279296875, 6.258487701416016e-06],
            [0.4912109375, 0.3583984375, 1.430511474609375e-06],
            [0.35791015625, 0.490478515625, 0.35791015625],
            [0.422119140625, 0.35107421875, 0.422119140625],
            [0.34912109375, 0.478515625, 0.34912109375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5927734375, 0.5927734375, 0.5927734375],
            [0.70068359375, 0.70068359375, 0.70068359375],
            [0.703125, 0.703125, 0.703125],
            [0.58642578125, 0.58642578125, 0.58642578125],
        ],
        [
            [0.350830078125, 0.480712890625, 0.350830078125],
            [0.422119140625, 0.35107421875, 0.422119140625],
            [0.35107421875, 0.48095703125, 0.35107421875],
            [0.0, 0.0, 0.350341796875],
            [0.48095703125, 0.35107421875, 0.0],
            [0.38134765625, 0.38134765625, 0.38134765625],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
        ],
        [
            [0.35107421875, 0.48095703125, 0.35107421875],
            [0.48046875, 0.3505859375, 0.0],
            [0.35107421875, 0.48095703125, 0.35107421875],
            [0.0, 0.0, 0.35107421875],
            [0.48095703125, 0.35107421875, 0.0],
            [0.38134765625, 0.38134765625, 0.38134765625],
            [0.62158203125, 0.62158203125, 0.62158203125],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.69677734375, 0.69677734375, 0.69677734375],
            [0.5859375, 0.5859375, 0.5859375],
        ],
        [
            [0.317138671875, 0.317138671875, 0.317138671875],
            [0.38134765625, 0.38134765625, 0.38134765625],
            [0.317138671875, 0.317138671875, 0.317138671875],
            [0.319580078125, 0.319580078125, 0.319580078125],
            [0.38134765625, 0.38134765625, 0.38134765625],
            [0.317138671875, 0.317138671875, 0.317138671875],
            [0.66748046875, 0.66748046875, 0.66748046875],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
        ],
        [
            [0.70703125, 0.70703125, 0.70703125],
            [0.587890625, 0.587890625, 0.587890625],
            [0.705078125, 0.705078125, 0.705078125],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.5859375, 0.5859375, 0.5859375],
        ],
        [
            [0.6689453125, 0.6689453125, 0.6689453125],
            [0.6650390625, 0.6650390625, 0.6650390625],
            [0.65478515625, 0.65478515625, 0.65478515625],
            [0.6494140625, 0.6494140625, 0.6494140625],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
            [0.6484375, 0.6484375, 0.6484375],
        ],
        [
            [0.7724609375, 0.7724609375, 0.7724609375],
            [0.6640625, 0.6640625, 0.6640625],
            [0.7314453125, 0.7314453125, 0.7314453125],
            [0.70849609375, 0.70849609375, 0.70849609375],
            [0.58837890625, 0.58837890625, 0.58837890625],
            [0.703125, 0.703125, 0.703125],
            [0.587890625, 0.587890625, 0.587890625],
            [0.587890625, 0.587890625, 0.587890625],
            [0.703125, 0.703125, 0.703125],
            [0.587890625, 0.587890625, 0.587890625],
        ],
        [
            [0.8427734375, 0.8427734375, 0.8427734375],
            [0.740234375, 0.740234375, 0.740234375],
            [0.76953125, 0.76953125, 0.76953125],
            [0.7080078125, 0.7080078125, 0.7080078125],
            [0.587890625, 0.587890625, 0.587890625],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.60107421875, 0.60107421875, 0.60107421875],
            [0.5859375, 0.5859375, 0.5859375],
            [0.70458984375, 0.70458984375, 0.70458984375],
            [0.58642578125, 0.58642578125, 0.58642578125],
        ],
    ]
)
