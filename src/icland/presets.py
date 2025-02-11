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
