# Copyright (c) 2024, RoArm Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the RoArm-M3-Pro robot arm.

RoArm-M3-Pro is a 4-DOF robot arm with a gripper.
USD model has 4 revolute joints:
- base_link_to_link1: Base rotation
- link1_to_link2: Shoulder
- link2_to_link3: Elbow
- link3_to_gripper_link: Gripper

Reference: https://www.waveshare.com/roarm-m3-pro.htm
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# RoArm USD file path (fixed version with single ArticulationRoot)
# Original file has ArticulationRootAPI on both base_link and root_joint
# The fixed version has ArticulationRootAPI only on base_link
ROARM_USD_PATH = r"E:\RoArm_Project\RoARM_PRO_M3_fixed.usd"

##
# Configuration
##

ROARM_M3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROARM_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,  # Fix robot base to world frame
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.70711, 0.0, 0.0, 0.70711),  # 90 degree rotation to stand upright
        joint_pos={
            "base_link_to_link1": 0.0,
            "link1_to_link2": 0.0,
            "link2_to_link3": 0.0,
            "link3_to_gripper_link": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["base_link_to_link1", "link1_to_link2", "link2_to_link3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=1000.0,
            damping=100.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["link3_to_gripper_link"],
            effort_limit_sim=50.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=50.0,
        ),
    },
)
"""Configuration for RoArm-M3-Pro (4-DOF) arm with gripper."""
