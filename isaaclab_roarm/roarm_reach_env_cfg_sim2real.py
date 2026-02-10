# Copyright (c) 2024, RoArm Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RoArm-M3-Pro reach environment with Sim2Real Domain Randomization.

This environment extends the base reach environment with:
1. Domain Randomization (DR) - joint stiffness/damping, friction, mass
2. Action Noise - motor noise simulation
3. Observation Noise - sensor noise simulation
4. Longer training for better convergence

Reference:
- https://www.reinforcementlearningpath.com/sim2real/
- Isaac Lab Domain Randomization Guide
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, EventCfg

from .roarm_cfg import ROARM_M3_CFG


##
# Domain Randomization Events for Sim2Real
##


@configclass
class Sim2RealEventCfg(EventCfg):
    """Extended event configuration with Domain Randomization for Sim2Real transfer."""

    # Inherit reset_robot_joints from parent

    # Randomize actuator gains (stiffness and damping)
    # Real servos have manufacturing tolerances and wear
    # CONSERVATIVE: Reduced from ±20% to ±10% for stability
    randomize_actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),  # ±10% variation (was ±20%)
            "damping_distribution_params": (0.9, 1.1),    # ±10% variation (was ±20%)
            "operation": "scale",
            "distribution": "uniform",
        },
    )


##
# Sim2Real Environment Configuration
##


@configclass
class RoArmReachEnvCfg_Sim2Real(ReachEnvCfg):
    """Configuration for RoArm-M3-Pro reach environment with Sim2Real improvements.

    Key improvements over base config:
    1. Domain Randomization: actuator gains, joint friction, body mass
    2. Action Noise: simulates motor inaccuracies
    3. Observation Noise: simulates sensor noise
    4. Increased observation noise range
    """

    # Override events with Sim2Real version
    events: Sim2RealEventCfg = Sim2RealEventCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Switch robot to RoArm
        self.scene.robot = ROARM_M3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Override events - reset joint positions with wider range
        self.events.reset_robot_joints.params["position_range"] = (0.6, 1.4)  # Wider than base (0.75, 1.25)

        # Override rewards - end-effector body name
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]

        # Position-Only Reach: Disable orientation tracking
        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # Increase position tracking weight for better convergence
        self.rewards.end_effector_position_tracking.weight = -0.5
        self.rewards.end_effector_position_tracking_fine_grained.weight = -0.2

        # Add action penalty for smoother movements (helps Sim2Real)
        # CONSERVATIVE: Reduced penalty to prevent training instability
        self.rewards.action_rate.weight = -0.0005  # Reduced from -0.001
        self.rewards.joint_vel.weight = -0.0005    # Reduced from -0.001

        # Override actions - joint position control for arm
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["base_link_to_link1", "link1_to_link2", "link2_to_link3"],
            scale=0.5,
            use_default_offset=True,
        )

        # Override command generator
        self.commands.ee_pose.body_name = "gripper_link"
        self.commands.ee_pose.ranges.pos_x = (0.1, 0.3)
        self.commands.ee_pose.ranges.pos_y = (-0.15, 0.15)
        self.commands.ee_pose.ranges.pos_z = (0.05, 0.25)
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Simulation settings
        self.scene.num_envs = 2048  # More envs for better sample diversity
        self.episode_length_s = 8.0

        # === Sim2Real: Action Noise ===
        # Simulates motor inaccuracies and control delays
        # CONSERVATIVE: Reduced noise for stability (was std=0.03)
        self.action_noise_model = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),  # Reduced from 0.03
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="abs"),  # Reduced from 0.01
        )

        # === Sim2Real: Observation Noise ===
        # Simulates sensor noise (encoders, etc.)
        self.observation_noise_model = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),  # Reduced from 0.005
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),  # Reduced from 0.002
        )


@configclass
class RoArmReachEnvCfg_Sim2Real_PLAY(RoArmReachEnvCfg_Sim2Real):
    """Configuration for RoArm-M3-Pro reach environment (play/evaluation mode)."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play
        self.observations.policy.enable_corruption = False

        # Disable action/observation noise for evaluation
        self.action_noise_model = None
        self.observation_noise_model = None
