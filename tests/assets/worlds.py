"""This module contains XML strings defining different Mujoco simulation environments.

Constants:
  EMPTY_WORLD (str): XML string defining an empty world with a single agent and a ground plane.
  RAMP_30 (str): XML string defining a world with a single agent, a ground plane, and a ramp inclined at 30 degrees.
  RAMP_45 (str): XML string defining a world with a single agent, a ground plane, and a ramp inclined at 45 degrees.
  RAMP_60 (str): XML string defining a world with a single agent, a ground plane, and a ramp inclined at 60 degrees.
"""

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
        
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        
        mass="0.001"
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
        
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        
        mass="0.001"
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
        
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        
        mass="0.001"
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
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        mass="0.001"
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
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        mass="0.001"
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
      <joint type="hinge" axis="0 0 1" stiffness="1"/>

      <geom
        name="agent0_geom"
        type="capsule"
        size="0.06"
        fromto="0 0 0 0 0 -0.4"
        
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        
        mass="0.001"
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
        
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        
        mass="0.001"
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
        
        mass="0.01"
      />

      <geom
        type="box"
        size="0.05 0.05 0.05"
        pos="0 0 0.2"
        
        mass="0.001"
      />
    </body>

    <!-- Ground plane, also with low friction -->
    <geom
      name="ground"
      type="plane"
      size="0 0 0.01"
      rgba="1 1 1 1"
    />

    <geom type="box" size="1 1 1" pos="2 0 -0.5" euler="0 30 0" rgba="1 0.8 0.8 1" />

  </worldbody>
</mujoco>
"""
