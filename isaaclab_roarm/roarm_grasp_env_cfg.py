# Copyright (c) 2024, RoArm Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RoArm-M3-Pro grasping environment.

RoArm uses a 'Sticky Gripper' approach since the physical gripper mechanism
is not modeled in the URDF. When the end-effector is close enough to an object,
the policy should learn to 'grasp' by maintaining proximity and lifting.

Key differences from standard grasping:
1. No gripper action (no physical gripper fingers modeled)
2. Grasping is achieved by reaching close to object + lifting motion
3. Reward shaping guides: approach → contact → lift sequence
"""

import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

from .roarm_cfg import ROARM_M3_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# Custom MDP functions for RoArm grasping
# Using robot body data directly (not FrameTransformer) for reliability
##


def object_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object_asset.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def _get_ee_position(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Helper to get end-effector position from robot body data."""
    robot: Articulation = env.scene[robot_cfg.name]
    # Get the gripper_link body index
    ee_body_idx = robot.find_bodies("gripper_link")[0][0]
    # Get world position of the body
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]
    return ee_pos_w


def object_ee_distance_reward(
    env: "ManagerBasedRLEnv",
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    # Object position
    cube_pos_w = object_asset.data.root_pos_w
    # End-effector position from robot body
    ee_w = _get_ee_position(env, robot_cfg)
    # Distance
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_is_lifted_reward(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward if the object is above minimal height."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    return torch.where(object_asset.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def sticky_grasp_reward(
    env: "ManagerBasedRLEnv",
    grasp_distance: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for maintaining a 'sticky grasp' - being very close to object while lifting.

    This simulates a gripper by rewarding the agent for:
    1. Being very close to the object (within grasp_distance)
    2. Lifting the object upward
    """
    object_asset: RigidObject = env.scene[object_cfg.name]

    # Object position
    object_pos_w = object_asset.data.root_pos_w
    # End-effector position from robot body
    ee_w = _get_ee_position(env, robot_cfg)
    # Distance to object
    distance = torch.norm(object_pos_w - ee_w, dim=1)

    # Check if close enough to 'grasp'
    is_grasping = distance < grasp_distance

    # Check if object is being lifted
    object_height = object_pos_w[:, 2]
    is_lifting = object_height > 0.15  # Above 15cm (object starts at ~12cm)

    # Reward: 1.0 if grasping AND lifting, 0.3 if just grasping, 0.0 otherwise
    reward = torch.where(is_grasping & is_lifting, 1.0, torch.where(is_grasping, 0.3, 0.0))
    return reward


##
# Scene Configuration
##


@configclass
class RoArmGraspSceneCfg(InteractiveSceneCfg):
    """Scene configuration for RoArm grasping."""

    # Robot
    robot: ArticulationCfg = MISSING

    # Target object (cube)
    object: RigidObjectCfg = MISSING

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP Settings
##


@configclass
class ActionsCfg:
    """Action specifications - only arm control, no gripper."""

    arm_action: reach_mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)

        # Object state (position relative to robot)
        object_position = ObsTerm(func=object_position_in_robot_root_frame)

        # Last action
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),  # Less randomization for stability
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_object_position = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Phase 1: Reach the object (dense reward)
    # std=0.3 allows gradient even at 0.5m+ distances
    reaching_object = RewTerm(
        func=object_ee_distance_reward,
        params={"std": 0.3},
        weight=1.0
    )

    # Phase 2: Get very close (prepare to grasp)
    close_to_object = RewTerm(
        func=object_ee_distance_reward,
        params={"std": 0.05},
        weight=2.0
    )

    # Phase 3: Lift the object (object starts at z=0.12, lift to z>0.18)
    lifting_object = RewTerm(
        func=object_is_lifted_reward,
        params={"minimal_height": 0.18},
        weight=10.0
    )

    # Phase 4: Sticky grasp (maintain proximity while lifting)
    sticky_grasp = RewTerm(
        func=sticky_grasp_reward,
        params={"grasp_distance": 0.08},
        weight=15.0
    )

    # Action penalties for smooth motion
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.0005)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # Object falls below ground
    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


##
# Environment Configuration
##


@configclass
class RoArmGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for RoArm grasping environment."""

    # Scene settings
    scene: RoArmGraspSceneCfg = RoArmGraspSceneCfg(num_envs=2048, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 8.0

        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # Physics settings for stable grasping
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Set RoArm as robot with adjusted initial pose for grasping
        # EE starts pointing forward and slightly down, closer to object
        # Object at [0.08, 0, 0.18], so EE should start near this height
        self.scene.robot = ROARM_M3_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ROARM_M3_CFG.init_state.replace(
                joint_pos={
                    "base_link_to_link1": 0.0,
                    "link1_to_link2": 0.8,   # Shoulder forward ~45 degrees
                    "link2_to_link3": 0.8,   # Elbow bent ~45 degrees
                    "link3_to_gripper_link": 0.0,
                },
            ),
        )

        # Set arm action (4 joints, including gripper_link rotation for extended reach)
        # 4-DOF allows the robot to orient the end-effector for better object approach
        self.actions.arm_action = reach_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_gripper_link"],
            scale=0.5,
            use_default_offset=True,
        )

        # Set object (small cube) - positioned closer and higher for easier reaching
        # Previous position [0.15, 0, 0.12] was too far/low for the robot to reach
        # New position: closer (x=0.08) and higher (z=0.18) within confirmed workspace
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.08, 0.0, 0.18], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.4, 0.4, 0.4),  # Small cube (~2cm)
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )


@configclass
class RoArmGraspEnvCfg_PLAY(RoArmGraspEnvCfg):
    """Play/evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable observation noise
        self.observations.policy.enable_corruption = False
