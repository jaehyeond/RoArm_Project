# Copyright (c) 2024, RoArm Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RoArm-M3-Pro reach environment.

This environment trains the robot to reach target positions with its end-effector.
"""

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from .roarm_cfg import ROARM_M3_CFG


@configclass
class RoArmReachEnvCfg(ReachEnvCfg):
    """Configuration for RoArm-M3-Pro reach environment."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Switch robot to RoArm
        self.scene.robot = ROARM_M3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Override events - reset joint positions
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # Override rewards - end-effector body name
        # RoArm's end-effector is gripper_link
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]

        # Position-Only Reach: Disable orientation tracking (3-DOF cannot achieve 6-DOF task)
        # This makes the task physically solvable for our 3-joint arm
        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # Increase position tracking weight for better convergence
        self.rewards.end_effector_position_tracking.weight = -0.5
        self.rewards.end_effector_position_tracking_fine_grained.weight = -0.2

        # Override actions - joint position control for arm (3 joints, excluding gripper)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["base_link_to_link1", "link1_to_link2", "link2_to_link3"],
            scale=0.5,
            use_default_offset=True,
        )

        # Override command generator body
        # End-effector position command configuration
        self.commands.ee_pose.body_name = "gripper_link"
        self.commands.ee_pose.ranges.pos_x = (0.1, 0.3)  # Adjusted for RoArm workspace
        self.commands.ee_pose.ranges.pos_y = (-0.15, 0.15)
        self.commands.ee_pose.ranges.pos_z = (0.05, 0.25)
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Simulation settings
        self.scene.num_envs = 1024  # Reduced for smaller robot
        self.episode_length_s = 8.0


@configclass
class RoArmReachEnvCfg_PLAY(RoArmReachEnvCfg):
    """Configuration for RoArm-M3-Pro reach environment (play/evaluation mode)."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play
        self.observations.policy.enable_corruption = False
