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

WORLD_42_CONVEX = """
<mujoco model="generated_mesh_world">
    <compiler meshdir="tests/assets/meshes/"/>
    <default>
        <geom type="mesh" />
    </default>
    
    <worldbody>
            <body name="agent0" pos="3 1 4">
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
        <geom name="temp_0" mesh="temp_0" pos="0 0 0"/>
        <geom name="temp_1" mesh="temp_1" pos="0 0 0"/>
        <geom name="temp_10" mesh="temp_10" pos="0 0 0"/>
        <geom name="temp_11" mesh="temp_11" pos="0 0 0"/>
        <geom name="temp_12" mesh="temp_12" pos="0 0 0"/>
        <geom name="temp_13" mesh="temp_13" pos="0 0 0"/>
        <geom name="temp_14" mesh="temp_14" pos="0 0 0"/>
        <geom name="temp_15" mesh="temp_15" pos="0 0 0"/>
        <geom name="temp_16" mesh="temp_16" pos="0 0 0"/>
        <geom name="temp_17" mesh="temp_17" pos="0 0 0"/>
        <geom name="temp_18" mesh="temp_18" pos="0 0 0"/>
        <geom name="temp_19" mesh="temp_19" pos="0 0 0"/>
        <geom name="temp_2" mesh="temp_2" pos="0 0 0"/>
        <geom name="temp_20" mesh="temp_20" pos="0 0 0"/>
        <geom name="temp_21" mesh="temp_21" pos="0 0 0"/>
        <geom name="temp_22" mesh="temp_22" pos="0 0 0"/>
        <geom name="temp_23" mesh="temp_23" pos="0 0 0"/>
        <geom name="temp_24" mesh="temp_24" pos="0 0 0"/>
        <geom name="temp_25" mesh="temp_25" pos="0 0 0"/>
        <geom name="temp_26" mesh="temp_26" pos="0 0 0"/>
        <geom name="temp_27" mesh="temp_27" pos="0 0 0"/>
        <geom name="temp_28" mesh="temp_28" pos="0 0 0"/>
        <geom name="temp_29" mesh="temp_29" pos="0 0 0"/>
        <geom name="temp_3" mesh="temp_3" pos="0 0 0"/>
        <geom name="temp_30" mesh="temp_30" pos="0 0 0"/>
        <geom name="temp_31" mesh="temp_31" pos="0 0 0"/>
        <geom name="temp_32" mesh="temp_32" pos="0 0 0"/>
        <geom name="temp_33" mesh="temp_33" pos="0 0 0"/>
        <geom name="temp_34" mesh="temp_34" pos="0 0 0"/>
        <geom name="temp_35" mesh="temp_35" pos="0 0 0"/>
        <geom name="temp_36" mesh="temp_36" pos="0 0 0"/>
        <geom name="temp_37" mesh="temp_37" pos="0 0 0"/>
        <geom name="temp_38" mesh="temp_38" pos="0 0 0"/>
        <geom name="temp_39" mesh="temp_39" pos="0 0 0"/>
        <geom name="temp_4" mesh="temp_4" pos="0 0 0"/>
        <geom name="temp_40" mesh="temp_40" pos="0 0 0"/>
        <geom name="temp_41" mesh="temp_41" pos="0 0 0"/>
        <geom name="temp_42" mesh="temp_42" pos="0 0 0"/>
        <geom name="temp_43" mesh="temp_43" pos="0 0 0"/>
        <geom name="temp_44" mesh="temp_44" pos="0 0 0"/>
        <geom name="temp_45" mesh="temp_45" pos="0 0 0"/>
        <geom name="temp_46" mesh="temp_46" pos="0 0 0"/>
        <geom name="temp_47" mesh="temp_47" pos="0 0 0"/>
        <geom name="temp_48" mesh="temp_48" pos="0 0 0"/>
        <geom name="temp_49" mesh="temp_49" pos="0 0 0"/>
        <geom name="temp_5" mesh="temp_5" pos="0 0 0"/>
        <geom name="temp_50" mesh="temp_50" pos="0 0 0"/>
        <geom name="temp_51" mesh="temp_51" pos="0 0 0"/>
        <geom name="temp_52" mesh="temp_52" pos="0 0 0"/>
        <geom name="temp_53" mesh="temp_53" pos="0 0 0"/>
        <geom name="temp_54" mesh="temp_54" pos="0 0 0"/>
        <geom name="temp_55" mesh="temp_55" pos="0 0 0"/>
        <geom name="temp_56" mesh="temp_56" pos="0 0 0"/>
        <geom name="temp_57" mesh="temp_57" pos="0 0 0"/>
        <geom name="temp_58" mesh="temp_58" pos="0 0 0"/>
        <geom name="temp_59" mesh="temp_59" pos="0 0 0"/>
        <geom name="temp_6" mesh="temp_6" pos="0 0 0"/>
        <geom name="temp_60" mesh="temp_60" pos="0 0 0"/>
        <geom name="temp_61" mesh="temp_61" pos="0 0 0"/>
        <geom name="temp_62" mesh="temp_62" pos="0 0 0"/>
        <geom name="temp_63" mesh="temp_63" pos="0 0 0"/>
        <geom name="temp_64" mesh="temp_64" pos="0 0 0"/>
        <geom name="temp_65" mesh="temp_65" pos="0 0 0"/>
        <geom name="temp_66" mesh="temp_66" pos="0 0 0"/>
        <geom name="temp_67" mesh="temp_67" pos="0 0 0"/>
        <geom name="temp_68" mesh="temp_68" pos="0 0 0"/>
        <geom name="temp_69" mesh="temp_69" pos="0 0 0"/>
        <geom name="temp_7" mesh="temp_7" pos="0 0 0"/>
        <geom name="temp_70" mesh="temp_70" pos="0 0 0"/>
        <geom name="temp_71" mesh="temp_71" pos="0 0 0"/>
        <geom name="temp_72" mesh="temp_72" pos="0 0 0"/>
        <geom name="temp_73" mesh="temp_73" pos="0 0 0"/>
        <geom name="temp_74" mesh="temp_74" pos="0 0 0"/>
        <geom name="temp_75" mesh="temp_75" pos="0 0 0"/>
        <geom name="temp_76" mesh="temp_76" pos="0 0 0"/>
        <geom name="temp_77" mesh="temp_77" pos="0 0 0"/>
        <geom name="temp_78" mesh="temp_78" pos="0 0 0"/>
        <geom name="temp_79" mesh="temp_79" pos="0 0 0"/>
        <geom name="temp_8" mesh="temp_8" pos="0 0 0"/>
        <geom name="temp_80" mesh="temp_80" pos="0 0 0"/>
        <geom name="temp_81" mesh="temp_81" pos="0 0 0"/>
        <geom name="temp_82" mesh="temp_82" pos="0 0 0"/>
        <geom name="temp_83" mesh="temp_83" pos="0 0 0"/>
        <geom name="temp_84" mesh="temp_84" pos="0 0 0"/>
        <geom name="temp_85" mesh="temp_85" pos="0 0 0"/>
        <geom name="temp_86" mesh="temp_86" pos="0 0 0"/>
        <geom name="temp_87" mesh="temp_87" pos="0 0 0"/>
        <geom name="temp_88" mesh="temp_88" pos="0 0 0"/>
        <geom name="temp_89" mesh="temp_89" pos="0 0 0"/>
        <geom name="temp_9" mesh="temp_9" pos="0 0 0"/>
        <geom name="temp_90" mesh="temp_90" pos="0 0 0"/>
        <geom name="temp_91" mesh="temp_91" pos="0 0 0"/>
        <geom name="temp_92" mesh="temp_92" pos="0 0 0"/>
        <geom name="temp_93" mesh="temp_93" pos="0 0 0"/>
        <geom name="temp_94" mesh="temp_94" pos="0 0 0"/>
        <geom name="temp_95" mesh="temp_95" pos="0 0 0"/>
        <geom name="temp_96" mesh="temp_96" pos="0 0 0"/>
        <geom name="temp_97" mesh="temp_97" pos="0 0 0"/>
        <geom name="temp_98" mesh="temp_98" pos="0 0 0"/>
        <geom name="temp_99" mesh="temp_99" pos="0 0 0"/>
        <geom name="east_wall" type="plane"
          pos="10 5 10"
          quat="0.5 -0.5 -0.5 0.5"
          size="5 10 0.01"
          rgba="1 0.819607843 0.859375 0.5" />

        <geom name="west_wall" type="plane"
          pos="0 5 10"
          quat="0.5 0.5 0.5 0.5"
          size="5 10 0.01"
          rgba="1 0.819607843 0.859375 0.5" />

        <geom name="north_wall" type="plane"
          pos="5 0 10"
          quat="0.5 -0.5 0.5 0.5"
          size="10 5 0.01"
          rgba="1 0.819607843 0.859375 0.5" />

        <geom name="south_wall" type="plane"
          pos="5 10 10"
          quat="0.5 0.5 -0.5 0.5"
          size="10 5 0.01"
          rgba="1 0.819607843 0.859375 0.5" />
    </worldbody>

    <asset>
        <mesh name="temp_0" file="temp_0.stl"/>
        <mesh name="temp_1" file="temp_1.stl"/>
        <mesh name="temp_10" file="temp_10.stl"/>
        <mesh name="temp_11" file="temp_11.stl"/>
        <mesh name="temp_12" file="temp_12.stl"/>
        <mesh name="temp_13" file="temp_13.stl"/>
        <mesh name="temp_14" file="temp_14.stl"/>
        <mesh name="temp_15" file="temp_15.stl"/>
        <mesh name="temp_16" file="temp_16.stl"/>
        <mesh name="temp_17" file="temp_17.stl"/>
        <mesh name="temp_18" file="temp_18.stl"/>
        <mesh name="temp_19" file="temp_19.stl"/>
        <mesh name="temp_2" file="temp_2.stl"/>
        <mesh name="temp_20" file="temp_20.stl"/>
        <mesh name="temp_21" file="temp_21.stl"/>
        <mesh name="temp_22" file="temp_22.stl"/>
        <mesh name="temp_23" file="temp_23.stl"/>
        <mesh name="temp_24" file="temp_24.stl"/>
        <mesh name="temp_25" file="temp_25.stl"/>
        <mesh name="temp_26" file="temp_26.stl"/>
        <mesh name="temp_27" file="temp_27.stl"/>
        <mesh name="temp_28" file="temp_28.stl"/>
        <mesh name="temp_29" file="temp_29.stl"/>
        <mesh name="temp_3" file="temp_3.stl"/>
        <mesh name="temp_30" file="temp_30.stl"/>
        <mesh name="temp_31" file="temp_31.stl"/>
        <mesh name="temp_32" file="temp_32.stl"/>
        <mesh name="temp_33" file="temp_33.stl"/>
        <mesh name="temp_34" file="temp_34.stl"/>
        <mesh name="temp_35" file="temp_35.stl"/>
        <mesh name="temp_36" file="temp_36.stl"/>
        <mesh name="temp_37" file="temp_37.stl"/>
        <mesh name="temp_38" file="temp_38.stl"/>
        <mesh name="temp_39" file="temp_39.stl"/>
        <mesh name="temp_4" file="temp_4.stl"/>
        <mesh name="temp_40" file="temp_40.stl"/>
        <mesh name="temp_41" file="temp_41.stl"/>
        <mesh name="temp_42" file="temp_42.stl"/>
        <mesh name="temp_43" file="temp_43.stl"/>
        <mesh name="temp_44" file="temp_44.stl"/>
        <mesh name="temp_45" file="temp_45.stl"/>
        <mesh name="temp_46" file="temp_46.stl"/>
        <mesh name="temp_47" file="temp_47.stl"/>
        <mesh name="temp_48" file="temp_48.stl"/>
        <mesh name="temp_49" file="temp_49.stl"/>
        <mesh name="temp_5" file="temp_5.stl"/>
        <mesh name="temp_50" file="temp_50.stl"/>
        <mesh name="temp_51" file="temp_51.stl"/>
        <mesh name="temp_52" file="temp_52.stl"/>
        <mesh name="temp_53" file="temp_53.stl"/>
        <mesh name="temp_54" file="temp_54.stl"/>
        <mesh name="temp_55" file="temp_55.stl"/>
        <mesh name="temp_56" file="temp_56.stl"/>
        <mesh name="temp_57" file="temp_57.stl"/>
        <mesh name="temp_58" file="temp_58.stl"/>
        <mesh name="temp_59" file="temp_59.stl"/>
        <mesh name="temp_6" file="temp_6.stl"/>
        <mesh name="temp_60" file="temp_60.stl"/>
        <mesh name="temp_61" file="temp_61.stl"/>
        <mesh name="temp_62" file="temp_62.stl"/>
        <mesh name="temp_63" file="temp_63.stl"/>
        <mesh name="temp_64" file="temp_64.stl"/>
        <mesh name="temp_65" file="temp_65.stl"/>
        <mesh name="temp_66" file="temp_66.stl"/>
        <mesh name="temp_67" file="temp_67.stl"/>
        <mesh name="temp_68" file="temp_68.stl"/>
        <mesh name="temp_69" file="temp_69.stl"/>
        <mesh name="temp_7" file="temp_7.stl"/>
        <mesh name="temp_70" file="temp_70.stl"/>
        <mesh name="temp_71" file="temp_71.stl"/>
        <mesh name="temp_72" file="temp_72.stl"/>
        <mesh name="temp_73" file="temp_73.stl"/>
        <mesh name="temp_74" file="temp_74.stl"/>
        <mesh name="temp_75" file="temp_75.stl"/>
        <mesh name="temp_76" file="temp_76.stl"/>
        <mesh name="temp_77" file="temp_77.stl"/>
        <mesh name="temp_78" file="temp_78.stl"/>
        <mesh name="temp_79" file="temp_79.stl"/>
        <mesh name="temp_8" file="temp_8.stl"/>
        <mesh name="temp_80" file="temp_80.stl"/>
        <mesh name="temp_81" file="temp_81.stl"/>
        <mesh name="temp_82" file="temp_82.stl"/>
        <mesh name="temp_83" file="temp_83.stl"/>
        <mesh name="temp_84" file="temp_84.stl"/>
        <mesh name="temp_85" file="temp_85.stl"/>
        <mesh name="temp_86" file="temp_86.stl"/>
        <mesh name="temp_87" file="temp_87.stl"/>
        <mesh name="temp_88" file="temp_88.stl"/>
        <mesh name="temp_89" file="temp_89.stl"/>
        <mesh name="temp_9" file="temp_9.stl"/>
        <mesh name="temp_90" file="temp_90.stl"/>
        <mesh name="temp_91" file="temp_91.stl"/>
        <mesh name="temp_92" file="temp_92.stl"/>
        <mesh name="temp_93" file="temp_93.stl"/>
        <mesh name="temp_94" file="temp_94.stl"/>
        <mesh name="temp_95" file="temp_95.stl"/>
        <mesh name="temp_96" file="temp_96.stl"/>
        <mesh name="temp_97" file="temp_97.stl"/>
        <mesh name="temp_98" file="temp_98.stl"/>
        <mesh name="temp_99" file="temp_99.stl"/>
    </asset>
</mujoco>

"""

TWO_AGENT_EMPTY_temp_COLLIDE = """
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

temp_42_CONVEX = """
<mujoco model="generated_mesh_world">
    <compiler meshdir="tests/assets/meshes/"/>
    <default>
        <geom type="mesh" />
    </default>
    
    <worldbody>
            <body name="agent0" pos="1.5 1 4">
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
        <geom name="temp_0" mesh="temp_0" pos="0 0 0"/>
        <geom name="temp_1" mesh="temp_1" pos="0 0 0"/>
        <geom name="temp_10" mesh="temp_10" pos="0 0 0"/>
        <geom name="temp_11" mesh="temp_11" pos="0 0 0"/>
        <geom name="temp_12" mesh="temp_12" pos="0 0 0"/>
        <geom name="temp_13" mesh="temp_13" pos="0 0 0"/>
        <geom name="temp_14" mesh="temp_14" pos="0 0 0"/>
        <geom name="temp_15" mesh="temp_15" pos="0 0 0"/>
        <geom name="temp_16" mesh="temp_16" pos="0 0 0"/>
        <geom name="temp_17" mesh="temp_17" pos="0 0 0"/>
        <geom name="temp_18" mesh="temp_18" pos="0 0 0"/>
        <geom name="temp_19" mesh="temp_19" pos="0 0 0"/>
        <geom name="temp_2" mesh="temp_2" pos="0 0 0"/>
        <geom name="temp_20" mesh="temp_20" pos="0 0 0"/>
        <geom name="temp_21" mesh="temp_21" pos="0 0 0"/>
        <geom name="temp_22" mesh="temp_22" pos="0 0 0"/>
        <geom name="temp_23" mesh="temp_23" pos="0 0 0"/>
        <geom name="temp_24" mesh="temp_24" pos="0 0 0"/>
        <geom name="temp_25" mesh="temp_25" pos="0 0 0"/>
        <geom name="temp_26" mesh="temp_26" pos="0 0 0"/>
        <geom name="temp_27" mesh="temp_27" pos="0 0 0"/>
        <geom name="temp_28" mesh="temp_28" pos="0 0 0"/>
        <geom name="temp_29" mesh="temp_29" pos="0 0 0"/>
        <geom name="temp_3" mesh="temp_3" pos="0 0 0"/>
        <geom name="temp_30" mesh="temp_30" pos="0 0 0"/>
        <geom name="temp_31" mesh="temp_31" pos="0 0 0"/>
        <geom name="temp_32" mesh="temp_32" pos="0 0 0"/>
        <geom name="temp_33" mesh="temp_33" pos="0 0 0"/>
        <geom name="temp_34" mesh="temp_34" pos="0 0 0"/>
        <geom name="temp_35" mesh="temp_35" pos="0 0 0"/>
        <geom name="temp_36" mesh="temp_36" pos="0 0 0"/>
        <geom name="temp_37" mesh="temp_37" pos="0 0 0"/>
        <geom name="temp_38" mesh="temp_38" pos="0 0 0"/>
        <geom name="temp_39" mesh="temp_39" pos="0 0 0"/>
        <geom name="temp_4" mesh="temp_4" pos="0 0 0"/>
        <geom name="temp_40" mesh="temp_40" pos="0 0 0"/>
        <geom name="temp_41" mesh="temp_41" pos="0 0 0"/>
        <geom name="temp_42" mesh="temp_42" pos="0 0 0"/>
        <geom name="temp_43" mesh="temp_43" pos="0 0 0"/>
        <geom name="temp_44" mesh="temp_44" pos="0 0 0"/>
        <geom name="temp_45" mesh="temp_45" pos="0 0 0"/>
        <geom name="temp_46" mesh="temp_46" pos="0 0 0"/>
        <geom name="temp_47" mesh="temp_47" pos="0 0 0"/>
        <geom name="temp_48" mesh="temp_48" pos="0 0 0"/>
        <geom name="temp_49" mesh="temp_49" pos="0 0 0"/>
        <geom name="temp_5" mesh="temp_5" pos="0 0 0"/>
        <geom name="temp_50" mesh="temp_50" pos="0 0 0"/>
        <geom name="temp_51" mesh="temp_51" pos="0 0 0"/>
        <geom name="temp_52" mesh="temp_52" pos="0 0 0"/>
        <geom name="temp_53" mesh="temp_53" pos="0 0 0"/>
        <geom name="temp_54" mesh="temp_54" pos="0 0 0"/>
        <geom name="temp_55" mesh="temp_55" pos="0 0 0"/>
        <geom name="temp_56" mesh="temp_56" pos="0 0 0"/>
        <geom name="temp_57" mesh="temp_57" pos="0 0 0"/>
        <geom name="temp_58" mesh="temp_58" pos="0 0 0"/>
        <geom name="temp_59" mesh="temp_59" pos="0 0 0"/>
        <geom name="temp_6" mesh="temp_6" pos="0 0 0"/>
        <geom name="temp_60" mesh="temp_60" pos="0 0 0"/>
        <geom name="temp_61" mesh="temp_61" pos="0 0 0"/>
        <geom name="temp_62" mesh="temp_62" pos="0 0 0"/>
        <geom name="temp_63" mesh="temp_63" pos="0 0 0"/>
        <geom name="temp_64" mesh="temp_64" pos="0 0 0"/>
        <geom name="temp_65" mesh="temp_65" pos="0 0 0"/>
        <geom name="temp_66" mesh="temp_66" pos="0 0 0"/>
        <geom name="temp_67" mesh="temp_67" pos="0 0 0"/>
        <geom name="temp_68" mesh="temp_68" pos="0 0 0"/>
        <geom name="temp_69" mesh="temp_69" pos="0 0 0"/>
        <geom name="temp_7" mesh="temp_7" pos="0 0 0"/>
        <geom name="temp_70" mesh="temp_70" pos="0 0 0"/>
        <geom name="temp_71" mesh="temp_71" pos="0 0 0"/>
        <geom name="temp_72" mesh="temp_72" pos="0 0 0"/>
        <geom name="temp_73" mesh="temp_73" pos="0 0 0"/>
        <geom name="temp_74" mesh="temp_74" pos="0 0 0"/>
        <geom name="temp_75" mesh="temp_75" pos="0 0 0"/>
        <geom name="temp_76" mesh="temp_76" pos="0 0 0"/>
        <geom name="temp_77" mesh="temp_77" pos="0 0 0"/>
        <geom name="temp_78" mesh="temp_78" pos="0 0 0"/>
        <geom name="temp_79" mesh="temp_79" pos="0 0 0"/>
        <geom name="temp_8" mesh="temp_8" pos="0 0 0"/>
        <geom name="temp_80" mesh="temp_80" pos="0 0 0"/>
        <geom name="temp_81" mesh="temp_81" pos="0 0 0"/>
        <geom name="temp_82" mesh="temp_82" pos="0 0 0"/>
        <geom name="temp_83" mesh="temp_83" pos="0 0 0"/>
        <geom name="temp_84" mesh="temp_84" pos="0 0 0"/>
        <geom name="temp_85" mesh="temp_85" pos="0 0 0"/>
        <geom name="temp_86" mesh="temp_86" pos="0 0 0"/>
        <geom name="temp_87" mesh="temp_87" pos="0 0 0"/>
        <geom name="temp_88" mesh="temp_88" pos="0 0 0"/>
        <geom name="temp_89" mesh="temp_89" pos="0 0 0"/>
        <geom name="temp_9" mesh="temp_9" pos="0 0 0"/>
        <geom name="temp_90" mesh="temp_90" pos="0 0 0"/>
        <geom name="temp_91" mesh="temp_91" pos="0 0 0"/>
        <geom name="temp_92" mesh="temp_92" pos="0 0 0"/>
        <geom name="temp_93" mesh="temp_93" pos="0 0 0"/>
        <geom name="temp_94" mesh="temp_94" pos="0 0 0"/>
        <geom name="temp_95" mesh="temp_95" pos="0 0 0"/>
        <geom name="temp_96" mesh="temp_96" pos="0 0 0"/>
        <geom name="temp_97" mesh="temp_97" pos="0 0 0"/>
        <geom name="temp_98" mesh="temp_98" pos="0 0 0"/>
        <geom name="temp_99" mesh="temp_99" pos="0 0 0"/>
    </worldbody>

    <asset>
        <mesh name="temp_0" file="temp_0.stl"/>
        <mesh name="temp_1" file="temp_1.stl"/>
        <mesh name="temp_10" file="temp_10.stl"/>
        <mesh name="temp_11" file="temp_11.stl"/>
        <mesh name="temp_12" file="temp_12.stl"/>
        <mesh name="temp_13" file="temp_13.stl"/>
        <mesh name="temp_14" file="temp_14.stl"/>
        <mesh name="temp_15" file="temp_15.stl"/>
        <mesh name="temp_16" file="temp_16.stl"/>
        <mesh name="temp_17" file="temp_17.stl"/>
        <mesh name="temp_18" file="temp_18.stl"/>
        <mesh name="temp_19" file="temp_19.stl"/>
        <mesh name="temp_2" file="temp_2.stl"/>
        <mesh name="temp_20" file="temp_20.stl"/>
        <mesh name="temp_21" file="temp_21.stl"/>
        <mesh name="temp_22" file="temp_22.stl"/>
        <mesh name="temp_23" file="temp_23.stl"/>
        <mesh name="temp_24" file="temp_24.stl"/>
        <mesh name="temp_25" file="temp_25.stl"/>
        <mesh name="temp_26" file="temp_26.stl"/>
        <mesh name="temp_27" file="temp_27.stl"/>
        <mesh name="temp_28" file="temp_28.stl"/>
        <mesh name="temp_29" file="temp_29.stl"/>
        <mesh name="temp_3" file="temp_3.stl"/>
        <mesh name="temp_30" file="temp_30.stl"/>
        <mesh name="temp_31" file="temp_31.stl"/>
        <mesh name="temp_32" file="temp_32.stl"/>
        <mesh name="temp_33" file="temp_33.stl"/>
        <mesh name="temp_34" file="temp_34.stl"/>
        <mesh name="temp_35" file="temp_35.stl"/>
        <mesh name="temp_36" file="temp_36.stl"/>
        <mesh name="temp_37" file="temp_37.stl"/>
        <mesh name="temp_38" file="temp_38.stl"/>
        <mesh name="temp_39" file="temp_39.stl"/>
        <mesh name="temp_4" file="temp_4.stl"/>
        <mesh name="temp_40" file="temp_40.stl"/>
        <mesh name="temp_41" file="temp_41.stl"/>
        <mesh name="temp_42" file="temp_42.stl"/>
        <mesh name="temp_43" file="temp_43.stl"/>
        <mesh name="temp_44" file="temp_44.stl"/>
        <mesh name="temp_45" file="temp_45.stl"/>
        <mesh name="temp_46" file="temp_46.stl"/>
        <mesh name="temp_47" file="temp_47.stl"/>
        <mesh name="temp_48" file="temp_48.stl"/>
        <mesh name="temp_49" file="temp_49.stl"/>
        <mesh name="temp_5" file="temp_5.stl"/>
        <mesh name="temp_50" file="temp_50.stl"/>
        <mesh name="temp_51" file="temp_51.stl"/>
        <mesh name="temp_52" file="temp_52.stl"/>
        <mesh name="temp_53" file="temp_53.stl"/>
        <mesh name="temp_54" file="temp_54.stl"/>
        <mesh name="temp_55" file="temp_55.stl"/>
        <mesh name="temp_56" file="temp_56.stl"/>
        <mesh name="temp_57" file="temp_57.stl"/>
        <mesh name="temp_58" file="temp_58.stl"/>
        <mesh name="temp_59" file="temp_59.stl"/>
        <mesh name="temp_6" file="temp_6.stl"/>
        <mesh name="temp_60" file="temp_60.stl"/>
        <mesh name="temp_61" file="temp_61.stl"/>
        <mesh name="temp_62" file="temp_62.stl"/>
        <mesh name="temp_63" file="temp_63.stl"/>
        <mesh name="temp_64" file="temp_64.stl"/>
        <mesh name="temp_65" file="temp_65.stl"/>
        <mesh name="temp_66" file="temp_66.stl"/>
        <mesh name="temp_67" file="temp_67.stl"/>
        <mesh name="temp_68" file="temp_68.stl"/>
        <mesh name="temp_69" file="temp_69.stl"/>
        <mesh name="temp_7" file="temp_7.stl"/>
        <mesh name="temp_70" file="temp_70.stl"/>
        <mesh name="temp_71" file="temp_71.stl"/>
        <mesh name="temp_72" file="temp_72.stl"/>
        <mesh name="temp_73" file="temp_73.stl"/>
        <mesh name="temp_74" file="temp_74.stl"/>
        <mesh name="temp_75" file="temp_75.stl"/>
        <mesh name="temp_76" file="temp_76.stl"/>
        <mesh name="temp_77" file="temp_77.stl"/>
        <mesh name="temp_78" file="temp_78.stl"/>
        <mesh name="temp_79" file="temp_79.stl"/>
        <mesh name="temp_8" file="temp_8.stl"/>
        <mesh name="temp_80" file="temp_80.stl"/>
        <mesh name="temp_81" file="temp_81.stl"/>
        <mesh name="temp_82" file="temp_82.stl"/>
        <mesh name="temp_83" file="temp_83.stl"/>
        <mesh name="temp_84" file="temp_84.stl"/>
        <mesh name="temp_85" file="temp_85.stl"/>
        <mesh name="temp_86" file="temp_86.stl"/>
        <mesh name="temp_87" file="temp_87.stl"/>
        <mesh name="temp_88" file="temp_88.stl"/>
        <mesh name="temp_89" file="temp_89.stl"/>
        <mesh name="temp_9" file="temp_9.stl"/>
        <mesh name="temp_90" file="temp_90.stl"/>
        <mesh name="temp_91" file="temp_91.stl"/>
        <mesh name="temp_92" file="temp_92.stl"/>
        <mesh name="temp_93" file="temp_93.stl"/>
        <mesh name="temp_94" file="temp_94.stl"/>
        <mesh name="temp_95" file="temp_95.stl"/>
        <mesh name="temp_96" file="temp_96.stl"/>
        <mesh name="temp_97" file="temp_97.stl"/>
        <mesh name="temp_98" file="temp_98.stl"/>
        <mesh name="temp_99" file="temp_99.stl"/>
    </asset>
</mujoco>

"""
