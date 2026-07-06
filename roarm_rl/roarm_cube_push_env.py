"""RoArmCubePushEnv: no-attach 3cm cube push task.

This env is separate from the existing Pick/Stack tasks. Its purpose is to
turn the professor's "known endpoint, push/tap the 3cm cube first" request into
a small RL task without using grasp attach or object pose writes during rollout.

Default action semantics remain the project standard 6D normalized joint-delta
command: robot_dof_targets += action_scale * action, clipped to joint limits.
Candidate6 DiffIK residual control is an explicit opt-in mode. The target
residual mode is intentionally a 3D policy action space: forward, lateral,
height target waypoint residual before DiffIK. The tap push primitive is a
default-off non-policy action mode that executes a bounded tool/object push
target and terminates by holding the current joint state.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, sample_uniform, subtract_frame_transforms

from sim_scripts.roarm_kinematics import clip_joints, fk_tcp, ik_dls
from roarm_rl.roarm_stack_env import (
    HOME_RAD,
    TABLE_Z,
    RoArmStackEnv,
    RoArmStackEnvCfg,
)


CUBE_SIZE_M = 0.030
CUBE_CENTER_Z = TABLE_Z + CUBE_SIZE_M / 2.0
CUBE10CM_SIZE_M = 0.100
CUBE10CM_MASS_KG = 0.720
CUBE10CM_CENTER_Z = TABLE_Z + CUBE10CM_SIZE_M / 2.0
TAP_TABLE_SIZE_X_M = 1.000
TAP_TABLE_SIZE_Y_M = 1.000
TAP_TABLE_THICKNESS_M = 0.020
TAP_TABLE_CENTER_Z = TABLE_Z - TAP_TABLE_THICKNESS_M / 2.0
LINK5_COLLISION_BBOX_MIN_M = (-0.03099808120727539, -0.01774634552001953, -0.0007495112419128418)
LINK5_COLLISION_BBOX_MAX_M = (0.015497934341430665, 0.01777365303039551, 0.11988562011718751)

AUDIT_SPEED_P95_MPS = 1.302103193
AUDIT_SPEED_P99_MPS = 1.733444051
AUDIT_TIP_P95_DEG = 141.181661216
AUDIT_TIP_P99_DEG = 150.399799770
AUDIT_DISP_XY_P99_M = 0.133549188


def _make_bc_teacher_model(in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
    layers = []
    last = int(in_dim)
    for _ in range(int(hidden_layers)):
        layers.append(torch.nn.Linear(last, int(hidden_dim)))
        layers.append(torch.nn.ReLU())
        last = int(hidden_dim)
    layers.append(torch.nn.Linear(last, int(out_dim)))
    return torch.nn.Sequential(*layers)


@configclass
class RoArmCubePushEnvCfg(RoArmStackEnvCfg):
    """Config for the no-attach cube push task."""

    episode_length_s = 1.2
    action_scale: float = 0.04
    reward_phase: int = 8

    sponge: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sponge",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=2,
                max_angular_velocity=10.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.020),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.72, 0.60),
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.00, CUBE_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    cube_x_min: float = 0.205
    cube_x_max: float = 0.360
    cube_y_min: float = -0.125
    cube_y_max: float = 0.125
    cube_size_x_m: float = CUBE_SIZE_M
    cube_size_y_m: float = CUBE_SIZE_M
    cube_size_z_m: float = CUBE_SIZE_M
    fixed_push_dir_x: float = math.nan
    fixed_push_dir_y: float = math.nan
    cube_push_target_disp_m: float = 0.040
    cube_success_disp_m: float = 0.030
    cube_success_target_tol_m: float = 0.050
    cube_success_speed_max_mps: float = 0.500
    cube_low_motion_disp_m: float = 0.005
    cube_far_target_terminate_m: float = 0.120
    cube_impact_terminate_disp_m: float = 0.150
    ik_endpoint_reset: bool = False
    ik_precontact_clearance_m: float = 0.010
    ik_tcp_top_margin_m: float = 0.003
    ik_max_iter: int = 250
    ik_tol_mm: float = 1.5
    ik_accept_m: float = 0.004
    ik_reset_jitter_rad: float = 0.003
    action_smoothing_alpha: float = 0.25
    max_joint_delta_per_step_rad: float = 0.010
    contact_slowdown_tcp_dist_m: float = 0.055
    contact_joint_delta_scale: float = 0.35
    fast_cube_joint_delta_scale: float = 0.20
    joint_target_lead_limit_rad: float = 0.060
    joint_delta_reference: str = "target"
    policy_obs_target_mode: str = "push_target"
    scripted_teacher_blend: float = 0.0
    scripted_teacher_horizon_frac: float = 0.55
    scripted_teacher_goal_push_m: float = 0.055
    rl_action_mode: str = "joint_delta"
    candidate6_diffik_goal_push_m: float = 0.006
    candidate6_diffik_push_steps: int = 580
    candidate6_diffik_step_clip_rad: float = 0.010
    candidate6_diffik_lambda: float = 0.010
    candidate6_diffik_residual_scale_rad: float = 0.002
    candidate6_diffik_hold_after_tap_success: bool = True
    candidate6_diffik_target_base_mode: str = "previous_joint_target"
    candidate6_diffik_target_path_mode: str = "near_face_goal"
    candidate6_diffik_cube_reference_mode: str = "start_pose"
    candidate8_diffik_target_residual_forward_m: float = 0.004
    candidate8_diffik_target_residual_lateral_m: float = 0.012
    candidate8_diffik_target_residual_height_m: float = 0.004
    candidate8_diffik_target_residual_zero_after_contact: bool = False
    candidate8_diffik_target_residual_zero_after_reaction: bool = False
    candidate8_diffik_target_residual_zero_after_disp_m: float = 0.0
    tap_push_primitive_stop_disp_m: float = 0.003
    tap_push_primitive_speed_stop_mps: float = 0.200
    tap_push_primitive_speed_stop_min_disp_m: float = 0.001
    tap_push_primitive_stop_on_overshoot: bool = True
    bc_teacher_checkpoint_path: str = ""
    bc_teacher_blend: float = 0.0
    bc_teacher_imitation_reward_scale: float = 0.0
    bc_teacher_feature_target_mode: str = "tcp_target"
    bc_teacher_policy_delta_clip_rad: float = 0.040
    bc_teacher_policy_delta_scale: float = 1.0
    bc_teacher_posx_policy_delta_scale: float = 1.0
    bc_teacher_lowx_policy_delta_scale: float = 0.85
    bc_teacher_highx_policy_delta_scale: float = 0.80
    bc_teacher_delta_smoothing_alpha: float = 0.85
    bc_teacher_x_bucket_edge0_m: float = 0.257
    bc_teacher_x_bucket_edge1_m: float = 0.308
    bc_teacher_precontact_clearance_m: float = 0.020
    bc_teacher_push_through_m: float = 0.030
    bc_teacher_tcp_top_margin_m: float = 0.003
    bc_teacher_approach_steps: int = 220
    bc_teacher_push_steps: int = 90
    bc_teacher_post_steps: int = 40
    bc_teacher_posx_precontact_clearance_m: float = 0.014
    bc_teacher_posx_push_through_m: float = 0.020
    bc_teacher_posx_tcp_top_margin_m: float = -0.011
    bc_teacher_posx_approach_steps: int = 300
    bc_teacher_posx_push_steps: int = 220
    bc_teacher_posx_post_steps: int = 60
    bc_teacher_lowx_threshold_m: float = 0.240
    bc_teacher_lowx_precontact_clearance_m: float = 0.020
    bc_teacher_lowx_push_through_m: float = 0.030
    bc_teacher_lowx_tcp_top_margin_m: float = 0.003
    bc_teacher_lowx_approach_steps: int = 300
    bc_teacher_lowx_push_steps: int = 220
    bc_teacher_lowx_post_steps: int = 60
    bc_teacher_midx_push_through_m: float = -1.0
    bc_teacher_highx_push_through_m: float = -1.0
    bc_teacher_phase_timing: str = "episode_scaled"
    bc_teacher_linear_phase_steps: int = 579
    d256_reset_csv_path: str = ""
    d256_reset_frame_index: int = 0
    d256_reset_sample_mode: str = "random"
    d256_reset_episode_min: int = -1
    d256_reset_episode_max: int = -1

    push_progress_reward_scale: float = 60.0
    push_displacement_reward_scale: float = 18.0
    push_target_reward_scale: float = 10.0
    tcp_cube_distance_penalty_scale: float = 0.8
    gripper_close_penalty_scale: float = 0.4
    impact_penalty_scale: float = 12.0
    low_motion_penalty_scale: float = 6.0
    reverse_push_penalty_scale: float = 8.0
    controlled_bonus_scale: float = 1.0
    target_distance_penalty_scale: float = 30.0
    lateral_penalty_scale: float = 6.0
    overshoot_penalty_scale: float = 16.0
    speed_penalty_scale: float = 3.0
    speed_penalty_start_mps: float = 0.500
    impact_terminal_penalty: float = 10.0
    success_bonus: float = 12.0


@configclass
class RoArmCubeTap10cmEnvCfg(RoArmCubePushEnvCfg):
    """Default-off config for the professor 10cm/0.72kg tap/reaction task."""

    tap_table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TapTable",
        spawn=sim_utils.CuboidCfg(
            size=(TAP_TABLE_SIZE_X_M, TAP_TABLE_SIZE_Y_M, TAP_TABLE_THICKNESS_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.42, 0.42, 0.42),
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.00, TAP_TABLE_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    sponge: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sponge",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE10CM_SIZE_M, CUBE10CM_SIZE_M, CUBE10CM_SIZE_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=2,
                max_angular_velocity=10.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE10CM_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.32, 0.58, 0.86),
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.00, CUBE10CM_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    cube_size_x_m: float = CUBE10CM_SIZE_M
    cube_size_y_m: float = CUBE10CM_SIZE_M
    cube_size_z_m: float = CUBE10CM_SIZE_M
    cube_push_target_disp_m: float = 0.001
    cube_success_disp_m: float = 0.001
    cube_success_target_tol_m: float = 1.0
    cube_success_speed_max_mps: float = 10.0
    cube_low_motion_disp_m: float = 0.0005
    cube_far_target_terminate_m: float = 0.200
    cube_impact_terminate_disp_m: float = 0.050
    success_bonus: float = 0.0

    tap_objective_name: str = "tap_reaction_contact_not_final_relocation"
    tap_final_relocation_required: bool = False
    tap_contact_proxy_mode: str = "tcp_point"
    tap_contact_face_band_m: float = 0.010
    tap_contact_lateral_margin_m: float = 0.015
    tap_contact_vertical_margin_m: float = 0.020
    tap_reaction_disp_m: float = 0.001
    tap_reaction_z_delta_m: float = 0.002
    tap_reaction_speed_mps: float = 0.020
    tap_reaction_tip_angle_deg: float = 1.0
    tap_useful_min_disp_m: float = 0.001
    tap_overshoot_disp_m: float = 0.020
    tap_target_disp_tolerance_m: float = 0.003
    tap_overshoot_terminate: bool = False
    tap_success_terminate: bool = False
    tap_useful_terminate: bool = False
    tap_stop_after_useful_seen: bool = False
    tap_stop_after_disp_m: float = 0.0
    tap_contact_slowdown_use_proxy: bool = False
    tap_action_governor_mode: str = "off"
    tap_action_governor_target_disp_m: float = 0.003
    tap_action_governor_predict_horizon_s: float = 0.020
    tap_action_governor_speed_stop_mps: float = 0.200
    tap_action_governor_min_contact_steps: int = 1
    tap_action_governor_push_scale: float = 1.0
    tap_action_governor_brake_scale: float = 0.35
    tap_action_governor_brake_steps: int = 2
    professor_physical_reaction_disp_m: float = 0.0005
    professor_physical_reaction_speed_mps: float = 0.005
    professor_physical_reaction_z_delta_m: float = 0.0005

    tap_contact_reward_scale: float = 1.0
    tap_reaction_reward_scale: float = 4.0
    tap_transient_disp_reward_scale: float = 40.0
    tap_contact_proximity_reward_scale: float = 0.8
    tap_overshoot_penalty_scale: float = 12.0
    tap_tip_penalty_scale: float = 0.02


class RoArmCubePushEnv(RoArmStackEnv):
    """No-attach cube pushing task with the Stack env's robot/action scaffold."""

    cfg: RoArmCubePushEnvCfg

    def __init__(self, cfg: RoArmCubePushEnvCfg, render_mode: str | None = None, **kwargs):
        action_mode = str(getattr(cfg, "rl_action_mode", "joint_delta"))
        if action_mode == "candidate8_diffik_target_residual" and int(cfg.action_space) != 3:
            raise ValueError(
                "candidate8_diffik_target_residual requires cfg.action_space=3; "
                "do not run it through the inherited 6D joint action scaffold"
            )
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_push_buffers()

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        dof_pos_scaled = (
            2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        env_origins = self.scene.env_origins
        sponge_pos_local = self._sponge_pos_w - env_origins
        tcp_pos_local = self._tcp_pos_w - env_origins
        target_world = self._target_world
        mode = str(getattr(self.cfg, "policy_obs_target_mode", "push_target"))
        if mode == "bc_teacher_tcp_target":
            traj = self._bc_teacher_traj()
            alpha = self._bc_teacher_phase_alpha(traj)
            target_world = self._bc_teacher_tcp_target(alpha, traj)
        elif mode != "push_target":
            raise ValueError(f"unsupported policy_obs_target_mode={mode!r}")
        target_pos_local = target_world - env_origins
        tcp_to_sponge = sponge_pos_local - tcp_pos_local
        sponge_to_target = target_pos_local - sponge_pos_local

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                sponge_pos_local,
                self._sponge_quat_w,
                tcp_to_sponge,
                target_pos_local,
                sponge_to_target,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _ensure_push_buffers(self) -> None:
        if hasattr(self, "_cube_start_w"):
            return
        self._cube_start_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._push_dir_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._prev_disp_along = torch.zeros(self.num_envs, device=self.device)
        self._push_success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ik_reset_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ik_reset_err_mm = torch.zeros(self.num_envs, device=self.device)
        self._smoothed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._last_joint_delta_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self._last_joint_delta_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._last_joint_delta_cap_rate = torch.zeros(self.num_envs, device=self.device)
        self._last_action_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self._last_action_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._last_target_lead_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self._last_target_lead_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._last_target_lead_limit_rate = torch.zeros(self.num_envs, device=self.device)
        self._last_contact_slowdown = torch.ones(self.num_envs, device=self.device)
        joint_dim = int(self._robot.num_joints)
        self._teacher_start_joints = torch.zeros((self.num_envs, joint_dim), device=self.device)
        self._teacher_goal_joints = torch.zeros((self.num_envs, joint_dim), device=self.device)
        self._teacher_goal_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_teacher_blend = torch.zeros(self.num_envs, device=self.device)
        self._last_bc_teacher_blend = torch.zeros(self.num_envs, device=self.device)
        self._last_bc_teacher_imitation_mse = torch.zeros(self.num_envs, device=self.device)
        self._last_bc_teacher_action_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self._last_d256_reset_active = torch.zeros(self.num_envs, device=self.device)
        self._last_d256_reset_episode_index = torch.full((self.num_envs,), -1.0, device=self.device)
        self._bc_prev_teacher_delta = torch.zeros((self.num_envs, 5), device=self.device)
        self._candidate6_prev_arm_joint_target = torch.zeros((self.num_envs, 5), device=self.device)
        self._candidate6_prev_arm_joint_target_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_candidate6_diffik_active = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate6_diffik_numeric_ok = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate6_diffik_raw_delta_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate6_diffik_clipped_delta_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate6_diffik_step_clip_rate = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate6_diffik_residual_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate6_diffik_residual_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate6_diffik_hold_success_rate = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate8_diffik_target_residual_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate8_diffik_target_residual_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate8_diffik_target_residual_forward_abs = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate8_diffik_target_residual_lateral_abs = torch.zeros(self.num_envs, device=self.device)
        self._last_candidate8_diffik_target_residual_height_abs = torch.zeros(self.num_envs, device=self.device)
        self._tap_push_primitive_stop_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tap_push_primitive_stop_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._tap_push_primitive_hold_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self._last_tap_push_primitive_stop_latched = torch.zeros(self.num_envs, device=self.device)
        self._last_tap_push_primitive_target_delta_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self._last_tap_push_primitive_target_delta_abs_max = torch.zeros(self.num_envs, device=self.device)
        self._candidate6_diffik_controller = None
        if not hasattr(self, "_bc_arm_joint_ids"):
            arm_joint_ids, _arm_joint_names = self._robot.find_joints(
                [
                    "base_link_to_link1",
                    "link1_to_link2",
                    "link2_to_link3",
                    "link3_to_link4",
                    "link4_to_link5",
                ],
                preserve_order=True,
            )
            self._bc_arm_joint_ids = arm_joint_ids
        self._load_bc_teacher_if_needed()

    def _load_d256_reset_table_if_needed(self) -> dict[str, torch.Tensor] | None:
        path = str(getattr(self.cfg, "d256_reset_csv_path", "") or "")
        if not path:
            return None
        if hasattr(self, "_d256_reset_table"):
            return self._d256_reset_table

        csv_path = Path(path).expanduser()
        required = [
            "episode_index",
            "frame_index_t",
            "cube_local_x_m",
            "cube_local_y_m",
            "cube_local_z_m",
            "target_local_x_m",
            "target_local_y_m",
            "target_local_z_m",
            "push_dx",
            "push_dy",
            "arm_joint_0_rad",
            "arm_joint_1_rad",
            "arm_joint_2_rad",
            "arm_joint_3_rad",
            "arm_joint_4_rad",
            "gripper_joint_rad",
        ]
        frame_index = int(getattr(self.cfg, "d256_reset_frame_index", 0))
        episode_min = int(getattr(self.cfg, "d256_reset_episode_min", -1))
        episode_max = int(getattr(self.cfg, "d256_reset_episode_max", -1))
        if episode_min >= 0 and episode_max >= 0 and episode_min > episode_max:
            raise ValueError(
                "d256_reset_episode_min must be <= d256_reset_episode_max "
                f"when both are set, got {episode_min}>{episode_max}"
            )
        arm_rows: list[list[float]] = []
        gripper_rows: list[float] = []
        cube_rows: list[list[float]] = []
        target_rows: list[list[float]] = []
        push_rows: list[list[float]] = []
        episode_rows: list[float] = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"empty D256 reset csv: {csv_path}")
            missing = [c for c in required if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"missing D256 reset columns in {csv_path}: {missing}")
            for row in reader:
                if int(float(row["frame_index_t"])) != frame_index:
                    continue
                episode_index = int(float(row["episode_index"]))
                if episode_min >= 0 and episode_index < episode_min:
                    continue
                if episode_max >= 0 and episode_index > episode_max:
                    continue
                arm_rows.append([float(row[f"arm_joint_{idx}_rad"]) for idx in range(5)])
                gripper_rows.append(float(row["gripper_joint_rad"]))
                cube_rows.append(
                    [
                        float(row["cube_local_x_m"]),
                        float(row["cube_local_y_m"]),
                        float(row["cube_local_z_m"]),
                    ]
                )
                target_rows.append(
                    [
                        float(row["target_local_x_m"]),
                        float(row["target_local_y_m"]),
                        float(row["target_local_z_m"]),
                    ]
                )
                push_rows.append([float(row["push_dx"]), float(row["push_dy"])])
                episode_rows.append(float(episode_index))
        if not arm_rows:
            raise ValueError(
                f"no D256 reset rows for frame_index_t={frame_index} "
                f"episode_min={episode_min} episode_max={episode_max} in {csv_path}"
            )

        self._d256_reset_table = {
            "arm": torch.tensor(arm_rows, device=self.device, dtype=torch.float32),
            "gripper": torch.tensor(gripper_rows, device=self.device, dtype=torch.float32),
            "cube_local": torch.tensor(cube_rows, device=self.device, dtype=torch.float32),
            "target_local": torch.tensor(target_rows, device=self.device, dtype=torch.float32),
            "push_dir": torch.tensor(push_rows, device=self.device, dtype=torch.float32),
            "episode_index": torch.tensor(episode_rows, device=self.device, dtype=torch.float32),
        }
        return self._d256_reset_table

    def _sample_d256_reset_table(self, n: int) -> dict[str, torch.Tensor] | None:
        table = self._load_d256_reset_table_if_needed()
        if table is None:
            return None
        total = int(table["arm"].shape[0])
        mode = str(getattr(self.cfg, "d256_reset_sample_mode", "random"))
        if mode == "random":
            idx = torch.randint(0, total, (n,), device=self.device)
        elif mode == "linspace":
            idx = torch.linspace(0, total - 1, n, device=self.device).round().long()
        else:
            raise ValueError(f"unsupported d256_reset_sample_mode={mode!r}")
        push_dir = table["push_dir"][idx]
        push_dir = push_dir / torch.clamp(torch.linalg.norm(push_dir, dim=-1, keepdim=True), min=1.0e-6)
        return {
            "arm": table["arm"][idx],
            "gripper": table["gripper"][idx],
            "cube_local": table["cube_local"][idx],
            "target_local": table["target_local"][idx],
            "push_dir": push_dir,
            "episode_index": table["episode_index"][idx],
        }

    def _load_bc_teacher_if_needed(self) -> None:
        if hasattr(self, "_bc_teacher_load_attempted"):
            return
        self._bc_teacher_load_attempted = True
        self._bc_teacher_model = None
        self._bc_teacher_feature_columns: list[str] = []
        self._bc_teacher_target_columns: list[str] = []
        self._bc_teacher_ready = False
        path = str(getattr(self.cfg, "bc_teacher_checkpoint_path", "") or "")
        if not path:
            return
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        feature_columns = list(checkpoint["feature_columns"])
        target_columns = list(checkpoint["target_columns"])
        if len(target_columns) != 5:
            raise ValueError(f"BC teacher target columns must be 5 arm deltas, got {target_columns}")
        model = _make_bc_teacher_model(
            len(feature_columns),
            len(target_columns),
            int(checkpoint["hidden_dim"]),
            int(checkpoint["hidden_layers"]),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._bc_teacher_model = model
        self._bc_teacher_feature_columns = feature_columns
        self._bc_teacher_target_columns = target_columns
        self._bc_teacher_x_mean = checkpoint["x_mean"].to(device=self.device, dtype=torch.float32).view(1, -1)
        self._bc_teacher_x_std = checkpoint["x_std"].to(device=self.device, dtype=torch.float32).view(1, -1)
        self._bc_teacher_y_mean = checkpoint["y_mean"].to(device=self.device, dtype=torch.float32).view(1, -1)
        self._bc_teacher_y_std = checkpoint["y_std"].to(device=self.device, dtype=torch.float32).view(1, -1)
        self._bc_teacher_ready = True

    def _ensure_candidate6_diffik_controller(self) -> None:
        if self._candidate6_diffik_controller is not None:
            return
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        diffik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(self.cfg.candidate6_diffik_lambda)},
        )
        self._candidate6_diffik_controller = DifferentialIKController(
            diffik_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._candidate6_link5_body_idx = self.link5_idx
        self._candidate6_jacobi_body_idx = self.link5_idx - 1 if self._robot.is_fixed_base else self.link5_idx
        self._candidate6_jacobi_joint_ids = (
            self._bc_arm_joint_ids if self._robot.is_fixed_base else [idx + 6 for idx in self._bc_arm_joint_ids]
        )

    def _candidate6_diffik_base_joint_target(
        self, tcp_target_residual_w: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._ensure_candidate6_diffik_controller()
        self._compute_intermediate_values()

        cube_reference_mode = str(getattr(self.cfg, "candidate6_diffik_cube_reference_mode", "start_pose"))
        if cube_reference_mode == "start_pose":
            cube_w = self._cube_start_w
        elif cube_reference_mode == "current_pose":
            cube_w = self._sponge_pos_w
        else:
            raise ValueError(f"unsupported candidate6_diffik_cube_reference_mode={cube_reference_mode!r}")
        push_dir = self._push_dir_xy
        half_xy = torch.tensor(
            (float(self.cfg.cube_size_x_m) * 0.5, float(self.cfg.cube_size_y_m) * 0.5),
            device=self.device,
            dtype=torch.float32,
        )
        half_z = float(self.cfg.cube_size_z_m) * 0.5
        half_along = torch.sum(torch.abs(push_dir) * half_xy.unsqueeze(0), dim=-1)
        horizon = max(1.0, float(self.cfg.candidate6_diffik_push_steps))
        alpha = torch.clamp((self.episode_length_buf.float() + 1.0) / horizon, 0.0, 1.0)

        pre_w = cube_w.clone()
        through_w = cube_w.clone()
        pre_w[:, 0:2] = cube_w[:, 0:2] - push_dir * (
            half_along + float(self.cfg.ik_precontact_clearance_m)
        ).unsqueeze(-1)
        target_path_mode = str(self.cfg.candidate6_diffik_target_path_mode)
        if target_path_mode == "near_face_goal":
            through_w[:, 0:2] = cube_w[:, 0:2] + push_dir * (
                float(self.cfg.candidate6_diffik_goal_push_m) - half_along
            ).unsqueeze(-1)
        elif target_path_mode == "legacy_far_face_through":
            through_w[:, 0:2] = cube_w[:, 0:2] + push_dir * (
                half_along + float(self.cfg.candidate6_diffik_goal_push_m)
            ).unsqueeze(-1)
        else:
            raise ValueError(f"unsupported candidate6_diffik_target_path_mode={target_path_mode!r}")
        z = cube_w[:, 2] + half_z + float(self.cfg.ik_tcp_top_margin_m)
        pre_w[:, 2] = z
        through_w[:, 2] = z
        tcp_target_w = pre_w + alpha.unsqueeze(-1) * (through_w - pre_w)
        if tcp_target_residual_w is not None:
            if tcp_target_residual_w.shape != tcp_target_w.shape:
                raise ValueError(
                    "tcp_target_residual_w must have shape "
                    f"{tuple(tcp_target_w.shape)}, got {tuple(tcp_target_residual_w.shape)}"
                )
            tcp_target_w = tcp_target_w + tcp_target_residual_w

        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        link5_pos_w = self._robot.data.body_pos_w[:, self._candidate6_link5_body_idx].clone()
        link5_quat_w = self._robot.data.body_quat_w[:, self._candidate6_link5_body_idx].clone()
        link5_pos_b, link5_quat_b = subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            link5_pos_w,
            link5_quat_w,
        )
        tool_proxy_offset_w = quat_apply(link5_quat_w, self._tcp_local.unsqueeze(0).repeat(self.num_envs, 1))
        link5_target_w = tcp_target_w - tool_proxy_offset_w
        link5_target_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, link5_target_w, link5_quat_w)

        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._candidate6_jacobi_body_idx, :, self._candidate6_jacobi_joint_ids
        ]
        base_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
        jacobian = jacobian.clone()
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        arm_joint_ids = self._bc_arm_joint_ids
        joint_pos = self._robot.data.joint_pos
        joint_pos_arm = joint_pos[:, arm_joint_ids]
        diffik = self._candidate6_diffik_controller
        diffik.set_command(link5_target_b, ee_pos=link5_pos_b, ee_quat=link5_quat_b)
        joint_pos_des = diffik.compute(link5_pos_b, link5_quat_b, jacobian, joint_pos_arm)
        numeric_ok = torch.isfinite(joint_pos_des).all(dim=-1)

        raw_delta_arm = joint_pos_des - joint_pos_arm
        raw_delta_arm = torch.where(numeric_ok.unsqueeze(-1), raw_delta_arm, torch.zeros_like(raw_delta_arm))
        step_clip = float(self.cfg.candidate6_diffik_step_clip_rad)
        clipped_delta_arm = torch.clamp(raw_delta_arm, -step_clip, step_clip)
        step_clip_rate = (torch.abs(raw_delta_arm) >= step_clip - 1.0e-9).float().mean(dim=-1)

        target_base_mode = str(self.cfg.candidate6_diffik_target_base_mode)
        if target_base_mode == "actual_joint_pos":
            target_base_arm = joint_pos_arm
        elif target_base_mode == "previous_joint_target":
            target_base_arm = torch.where(
                self._candidate6_prev_arm_joint_target_valid.unsqueeze(-1),
                self._candidate6_prev_arm_joint_target.detach(),
                joint_pos_arm,
            )
        else:
            raise ValueError(f"unsupported candidate6_diffik_target_base_mode={target_base_mode!r}")

        target_arm = target_base_arm + clipped_delta_arm
        lead = float(self.cfg.joint_target_lead_limit_rad)
        target_arm = torch.maximum(torch.minimum(target_arm, joint_pos_arm + lead), joint_pos_arm - lead)
        lower_arm = (
            self.robot_dof_lower_limits[:, arm_joint_ids]
            if self.robot_dof_lower_limits.ndim == 2
            else self.robot_dof_lower_limits[arm_joint_ids]
        )
        upper_arm = (
            self.robot_dof_upper_limits[:, arm_joint_ids]
            if self.robot_dof_upper_limits.ndim == 2
            else self.robot_dof_upper_limits[arm_joint_ids]
        )
        target_arm = torch.maximum(torch.minimum(target_arm, upper_arm), lower_arm)
        hold_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if bool(getattr(self.cfg, "candidate6_diffik_hold_after_tap_success", True)) and hasattr(
            self, "_tap_success_flag"
        ):
            hold_success = self._tap_success_flag & self._candidate6_prev_arm_joint_target_valid
            target_arm = torch.where(
                hold_success.unsqueeze(-1),
                self._candidate6_prev_arm_joint_target.detach(),
                target_arm,
            )

        target_full = joint_pos.clone()
        target_full[:, arm_joint_ids] = target_arm
        target_full[:, self.gripper_joint_idx] = 0.0
        self._candidate6_prev_arm_joint_target[:] = target_arm.detach()
        self._candidate6_prev_arm_joint_target_valid[:] = True
        self._last_candidate6_diffik_active[:] = 1.0
        self._last_candidate6_diffik_numeric_ok[:] = numeric_ok.float()
        self._last_candidate6_diffik_raw_delta_abs_max[:] = torch.max(torch.abs(raw_delta_arm), dim=-1).values
        self._last_candidate6_diffik_clipped_delta_abs_max[:] = torch.max(torch.abs(clipped_delta_arm), dim=-1).values
        self._last_candidate6_diffik_step_clip_rate[:] = step_clip_rate
        self._last_candidate6_diffik_hold_success_rate[:] = hold_success.float()
        return target_full

    def _tap_push_primitive_joint_target(self) -> torch.Tensor:
        stop_disp_m = float(getattr(self.cfg, "tap_push_primitive_stop_disp_m", 0.003))
        speed_stop_mps = float(getattr(self.cfg, "tap_push_primitive_speed_stop_mps", 0.200))
        speed_stop_min_disp_m = float(getattr(self.cfg, "tap_push_primitive_speed_stop_min_disp_m", 0.0))
        if stop_disp_m <= 0.0:
            raise ValueError("tap_push_primitive_stop_disp_m must be positive")
        if speed_stop_mps < 0.0:
            raise ValueError("tap_push_primitive_speed_stop_mps must be non-negative")
        if speed_stop_min_disp_m < 0.0:
            raise ValueError("tap_push_primitive_speed_stop_min_disp_m must be non-negative")

        pre_terms = self._tap_terms()
        speed_stop_now = (pre_terms["speed"] >= speed_stop_mps) & (
            pre_terms["disp_xy"] >= speed_stop_min_disp_m
        )
        stop_now = (pre_terms["disp_xy"] >= stop_disp_m) | speed_stop_now
        if bool(getattr(self.cfg, "tap_push_primitive_stop_on_overshoot", True)):
            stop_now = stop_now | pre_terms["tap_overshoot_now"]
        newly_stopped = stop_now & ~self._tap_push_primitive_stop_latched

        joint_pos = self._robot.data.joint_pos
        self._tap_push_primitive_hold_targets[:] = torch.where(
            newly_stopped.unsqueeze(-1),
            joint_pos.detach(),
            self._tap_push_primitive_hold_targets,
        )
        self._tap_push_primitive_stop_step[:] = torch.where(
            newly_stopped,
            self.episode_length_buf.to(dtype=torch.long),
            self._tap_push_primitive_stop_step,
        )
        self._tap_push_primitive_stop_latched |= stop_now

        push_targets = self._candidate6_diffik_base_joint_target()
        targets = torch.where(
            self._tap_push_primitive_stop_latched.unsqueeze(-1),
            self._tap_push_primitive_hold_targets,
            push_targets,
        )
        targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        targets[:, self.gripper_joint_idx] = 0.0

        target_delta = targets - joint_pos
        self._last_tap_push_primitive_stop_latched[:] = self._tap_push_primitive_stop_latched.float()
        self._last_tap_push_primitive_target_delta_abs_mean[:] = torch.mean(torch.abs(target_delta), dim=-1)
        self._last_tap_push_primitive_target_delta_abs_max[:] = torch.max(torch.abs(target_delta), dim=-1).values
        return targets

    def _ik_precontact_joints(self, cube_local: torch.Tensor, push_dir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cube_np = cube_local.detach().cpu().numpy().astype(np.float64)
        dir_np = push_dir.detach().cpu().numpy().astype(np.float64)
        home_deg = np.degrees(np.asarray(HOME_RAD, dtype=np.float64))
        q_out = np.tile(home_deg, (cube_np.shape[0], 1))
        ok = np.zeros(cube_np.shape[0], dtype=bool)
        err_mm_out = np.full(cube_np.shape[0], np.inf, dtype=np.float64)
        half_xy = np.asarray(
            [float(self.cfg.cube_size_x_m) * 0.5, float(self.cfg.cube_size_y_m) * 0.5],
            dtype=np.float64,
        )
        half_z = float(self.cfg.cube_size_z_m) * 0.5
        for idx in range(cube_np.shape[0]):
            tcp_target = cube_np[idx].copy()
            half_along = float(np.sum(np.abs(dir_np[idx, :2]) * half_xy))
            tcp_target[:2] -= dir_np[idx] * (half_along + float(self.cfg.ik_precontact_clearance_m))
            tcp_target[2] = cube_np[idx, 2] + half_z + float(self.cfg.ik_tcp_top_margin_m)
            q_deg, converged, err_mm, _iters = ik_dls(
                tcp_target,
                home_deg,
                max_iter=int(self.cfg.ik_max_iter),
                tol_mm=float(self.cfg.ik_tol_mm),
            )
            q_deg[5] = 0.0
            q_deg = clip_joints(q_deg)
            reached = np.linalg.norm(fk_tcp(q_deg) - tcp_target) <= float(self.cfg.ik_accept_m)
            if converged and reached:
                ok[idx] = True
                q_out[idx] = q_deg
            err_mm_out[idx] = err_mm
        q_rad = torch.tensor(np.radians(q_out), device=self.device, dtype=torch.float32)
        ok_t = torch.tensor(ok, device=self.device, dtype=torch.bool)
        err_t = torch.tensor(err_mm_out, device=self.device, dtype=torch.float32)
        return q_rad, ok_t, err_t

    def _ik_teacher_goal_joints(
        self,
        cube_local: torch.Tensor,
        push_dir: torch.Tensor,
        seed_rad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cube_np = cube_local.detach().cpu().numpy().astype(np.float64)
        dir_np = push_dir.detach().cpu().numpy().astype(np.float64)
        seed_deg = np.degrees(seed_rad.detach().cpu().numpy().astype(np.float64))
        q_out = seed_deg.copy()
        ok = np.zeros(cube_np.shape[0], dtype=bool)
        half_xy = np.asarray(
            [float(self.cfg.cube_size_x_m) * 0.5, float(self.cfg.cube_size_y_m) * 0.5],
            dtype=np.float64,
        )
        half_z = float(self.cfg.cube_size_z_m) * 0.5
        for idx in range(cube_np.shape[0]):
            tcp_target = cube_np[idx].copy()
            half_along = float(np.sum(np.abs(dir_np[idx, :2]) * half_xy))
            tcp_target[:2] += dir_np[idx] * (half_along + float(self.cfg.scripted_teacher_goal_push_m))
            tcp_target[2] = cube_np[idx, 2] + half_z + float(self.cfg.ik_tcp_top_margin_m)
            q_deg, converged, _err_mm, _iters = ik_dls(
                tcp_target,
                seed_deg[idx],
                max_iter=int(self.cfg.ik_max_iter),
                tol_mm=float(self.cfg.ik_tol_mm),
            )
            q_deg[5] = 0.0
            q_deg = clip_joints(q_deg)
            reached = np.linalg.norm(fk_tcp(q_deg) - tcp_target) <= float(self.cfg.ik_accept_m)
            if converged and reached:
                ok[idx] = True
                q_out[idx] = q_deg
        q_rad = torch.tensor(np.radians(q_out), device=self.device, dtype=torch.float32)
        ok_t = torch.tensor(ok, device=self.device, dtype=torch.bool)
        return q_rad, ok_t

    def _bc_teacher_traj(self) -> dict[str, torch.Tensor]:
        cube = self._cube_start_w
        push_dir = self._push_dir_xy
        posx = (push_dir[:, 0] > 0.5) & (torch.abs(push_dir[:, 1]) < 0.5)
        cube_x_local = cube[:, 0] - self.scene.env_origins[:, 0]
        edge0 = float(self.cfg.bc_teacher_x_bucket_edge0_m)
        edge1 = float(self.cfg.bc_teacher_x_bucket_edge1_m)
        posx_low_bucket = posx & (cube_x_local < edge0)
        posx_mid_bucket = posx & (cube_x_local >= edge0) & (cube_x_local < edge1)
        posx_high_bucket = posx & (cube_x_local >= edge1)
        lowx = posx & (cube_x_local <= float(self.cfg.bc_teacher_lowx_threshold_m))
        n = self.num_envs
        approach_steps = torch.full((n,), int(self.cfg.bc_teacher_approach_steps), dtype=torch.float32, device=self.device)
        push_steps = torch.full((n,), int(self.cfg.bc_teacher_push_steps), dtype=torch.float32, device=self.device)
        post_steps = torch.full((n,), int(self.cfg.bc_teacher_post_steps), dtype=torch.float32, device=self.device)
        precontact = torch.full((n,), float(self.cfg.bc_teacher_precontact_clearance_m), dtype=torch.float32, device=self.device)
        push_through = torch.full((n,), float(self.cfg.bc_teacher_push_through_m), dtype=torch.float32, device=self.device)
        tcp_top_margin = torch.full((n,), float(self.cfg.bc_teacher_tcp_top_margin_m), dtype=torch.float32, device=self.device)
        approach_steps[posx] = float(self.cfg.bc_teacher_posx_approach_steps)
        push_steps[posx] = float(self.cfg.bc_teacher_posx_push_steps)
        post_steps[posx] = float(self.cfg.bc_teacher_posx_post_steps)
        precontact[posx] = float(self.cfg.bc_teacher_posx_precontact_clearance_m)
        push_through[posx] = float(self.cfg.bc_teacher_posx_push_through_m)
        tcp_top_margin[posx] = float(self.cfg.bc_teacher_posx_tcp_top_margin_m)
        approach_steps[lowx] = float(self.cfg.bc_teacher_lowx_approach_steps)
        push_steps[lowx] = float(self.cfg.bc_teacher_lowx_push_steps)
        post_steps[lowx] = float(self.cfg.bc_teacher_lowx_post_steps)
        precontact[lowx] = float(self.cfg.bc_teacher_lowx_precontact_clearance_m)
        push_through[lowx] = float(self.cfg.bc_teacher_lowx_push_through_m)
        tcp_top_margin[lowx] = float(self.cfg.bc_teacher_lowx_tcp_top_margin_m)
        if float(self.cfg.bc_teacher_midx_push_through_m) >= 0.0:
            push_through[posx_mid_bucket] = float(self.cfg.bc_teacher_midx_push_through_m)
        if float(self.cfg.bc_teacher_highx_push_through_m) >= 0.0:
            push_through[posx_high_bucket] = float(self.cfg.bc_teacher_highx_push_through_m)
        return {
            "posx": posx,
            "posx_low_bucket": posx_low_bucket,
            "posx_mid_bucket": posx_mid_bucket,
            "posx_high_bucket": posx_high_bucket,
            "lowx": lowx,
            "approach_steps": approach_steps,
            "push_steps": push_steps,
            "post_steps": post_steps,
            "precontact": precontact,
            "push_through": push_through,
            "tcp_top_margin": tcp_top_margin,
        }

    def _bc_teacher_phase_alpha(self, traj: dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.clamp(traj["approach_steps"] + traj["push_steps"] + traj["post_steps"], min=1.0)
        timing = str(getattr(self.cfg, "bc_teacher_phase_timing", "episode_scaled"))
        if timing == "episode_scaled":
            denom = max(float(self.max_episode_length - 1), 1.0)
            step_v = self.episode_length_buf.float() / denom * total
        elif timing == "direct_steps":
            step_v = self.episode_length_buf.float()
        elif timing == "linear_episode":
            denom = max(float(self.max_episode_length - 1), 1.0)
            return torch.clamp(self.episode_length_buf.float() / denom, min=0.0, max=1.0)
        elif timing == "linear_steps":
            denom = max(float(getattr(self.cfg, "bc_teacher_linear_phase_steps", 579)), 1.0)
            return torch.clamp(self.episode_length_buf.float() / denom, min=0.0, max=1.0)
        else:
            raise ValueError(f"unsupported bc_teacher_phase_timing={timing!r}")
        push = torch.clamp(traj["push_steps"], min=1.0)
        raw_alpha = (step_v - traj["approach_steps"] + 1.0) / push
        alpha = torch.where(step_v < traj["approach_steps"], torch.zeros_like(raw_alpha), raw_alpha)
        alpha = torch.where(step_v >= traj["approach_steps"] + push, torch.ones_like(alpha), alpha)
        return torch.clamp(alpha, min=0.0, max=1.0)

    def _bc_teacher_tcp_target(self, alpha: torch.Tensor, traj: dict[str, torch.Tensor]) -> torch.Tensor:
        half_xy = torch.tensor(
            (float(self.cfg.cube_size_x_m) * 0.5, float(self.cfg.cube_size_y_m) * 0.5),
            device=self.device,
            dtype=torch.float32,
        )
        half_z = float(self.cfg.cube_size_z_m) * 0.5
        cube = self._cube_start_w
        push_dir = self._push_dir_xy
        half_along = torch.sum(torch.abs(push_dir) * half_xy.unsqueeze(0), dim=-1)
        pre = cube.clone()
        through = cube.clone()
        z = cube[:, 2] + half_z + traj["tcp_top_margin"]
        pre[:, 0:2] = cube[:, 0:2] - push_dir * (half_along + traj["precontact"]).unsqueeze(-1)
        through[:, 0:2] = cube[:, 0:2] + push_dir * (half_along + traj["push_through"]).unsqueeze(-1)
        pre[:, 2] = z
        through[:, 2] = z
        return pre + alpha.unsqueeze(-1) * (through - pre)

    def _bc_teacher_feature_tensor(self, alpha: torch.Tensor, tcp_target_w: torch.Tensor) -> torch.Tensor:
        origin = self.scene.env_origins
        cube_local = self._sponge_pos_w - origin
        tcp_local = self._tcp_pos_w - origin
        target_mode = str(getattr(self.cfg, "bc_teacher_feature_target_mode", "tcp_target"))
        if target_mode == "tcp_target":
            target_w = tcp_target_w
        elif target_mode == "env_target":
            target_w = self._target_world
        else:
            raise ValueError(f"unsupported bc_teacher_feature_target_mode={target_mode!r}")
        target_local = target_w - origin
        tcp_to_cube = cube_local - tcp_local
        target_to_tcp = target_local - tcp_local
        target_to_cube = target_local - cube_local
        joints = self._robot.data.joint_pos[:, self._bc_arm_joint_ids]
        gripper = self._robot.data.joint_pos[:, self.gripper_joint_idx]
        values = {
            "push_dx": self._push_dir_xy[:, 0],
            "push_dy": self._push_dir_xy[:, 1],
            "phase_alpha": alpha,
            "cube_local_x_m": cube_local[:, 0],
            "cube_local_y_m": cube_local[:, 1],
            "cube_local_z_m": cube_local[:, 2],
            "tcp_local_x_m": tcp_local[:, 0],
            "tcp_local_y_m": tcp_local[:, 1],
            "tcp_local_z_m": tcp_local[:, 2],
            "target_local_x_m": target_local[:, 0],
            "target_local_y_m": target_local[:, 1],
            "target_local_z_m": target_local[:, 2],
            "tcp_to_cube_x_m": tcp_to_cube[:, 0],
            "tcp_to_cube_y_m": tcp_to_cube[:, 1],
            "tcp_to_cube_z_m": tcp_to_cube[:, 2],
            "target_to_tcp_x_m": target_to_tcp[:, 0],
            "target_to_tcp_y_m": target_to_tcp[:, 1],
            "target_to_tcp_z_m": target_to_tcp[:, 2],
            "target_to_cube_x_m": target_to_cube[:, 0],
            "target_to_cube_y_m": target_to_cube[:, 1],
            "target_to_cube_z_m": target_to_cube[:, 2],
            "arm_joint_0_rad": joints[:, 0],
            "arm_joint_1_rad": joints[:, 1],
            "arm_joint_2_rad": joints[:, 2],
            "arm_joint_3_rad": joints[:, 3],
            "arm_joint_4_rad": joints[:, 4],
            "gripper_joint_rad": gripper,
        }
        missing = [col for col in self._bc_teacher_feature_columns if col not in values]
        if missing:
            raise KeyError(f"unsupported BC teacher feature columns: {missing}")
        return torch.stack([values[col] for col in self._bc_teacher_feature_columns], dim=-1).to(dtype=torch.float32)

    def _bc_teacher_actions(self) -> torch.Tensor:
        out = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device, dtype=torch.float32)
        if not getattr(self, "_bc_teacher_ready", False):
            return out
        self._compute_intermediate_values()
        traj = self._bc_teacher_traj()
        alpha = self._bc_teacher_phase_alpha(traj)
        tcp_target_w = self._bc_teacher_tcp_target(alpha, traj)
        with torch.no_grad():
            x = self._bc_teacher_feature_tensor(alpha, tcp_target_w)
            pred_n = self._bc_teacher_model((x - self._bc_teacher_x_mean) / self._bc_teacher_x_std)
            delta = pred_n * self._bc_teacher_y_std + self._bc_teacher_y_mean
            delta = torch.clamp(
                delta,
                -float(self.cfg.bc_teacher_policy_delta_clip_rad),
                float(self.cfg.bc_teacher_policy_delta_clip_rad),
            )
            scale = torch.full((self.num_envs,), float(self.cfg.bc_teacher_policy_delta_scale), device=self.device)
            scale = torch.where(traj["posx"], scale * float(self.cfg.bc_teacher_posx_policy_delta_scale), scale)
            scale = torch.where(traj["posx_low_bucket"], scale * float(self.cfg.bc_teacher_lowx_policy_delta_scale), scale)
            scale = torch.where(traj["posx_high_bucket"], scale * float(self.cfg.bc_teacher_highx_policy_delta_scale), scale)
            delta = delta * scale.unsqueeze(-1)
            smooth_alpha = max(0.0, min(1.0, float(self.cfg.bc_teacher_delta_smoothing_alpha)))
            if smooth_alpha < 1.0:
                delta = smooth_alpha * delta + (1.0 - smooth_alpha) * self._bc_prev_teacher_delta
            self._bc_prev_teacher_delta[:] = delta.detach()
            out[:, self._bc_arm_joint_ids] = delta / max(float(self.cfg.action_scale), 1.0e-6)
            out[:, self.gripper_joint_idx] = 0.0
            out = torch.clamp(out, -1.0, 1.0)
        return out

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ensure_push_buffers()
        override_targets = getattr(self, "_external_joint_targets_override", None)
        if override_targets is not None:
            self.actions = actions.clone().clamp(-1.0, 1.0)
            targets = torch.clamp(override_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            targets[:, self.gripper_joint_idx] = 0.0
            self.robot_dof_targets[:] = targets
            self._external_joint_targets_override = None

            joint_pos = self._robot.data.joint_pos
            direct_delta = self.robot_dof_targets - joint_pos
            self._last_action_abs_mean[:] = torch.mean(torch.abs(self.actions), dim=-1)
            self._last_action_abs_max[:] = torch.max(torch.abs(self.actions), dim=-1).values
            self._last_joint_delta_abs_mean[:] = torch.mean(torch.abs(direct_delta), dim=-1)
            self._last_joint_delta_abs_max[:] = torch.max(torch.abs(direct_delta), dim=-1).values
            self._last_joint_delta_cap_rate.zero_()
            self._last_target_lead_abs_mean[:] = self._last_joint_delta_abs_mean
            self._last_target_lead_abs_max[:] = self._last_joint_delta_abs_max
            self._last_target_lead_limit_rate.zero_()
            self._last_contact_slowdown[:] = 1.0
            self._last_teacher_blend.zero_()
            self._last_bc_teacher_blend.zero_()
            self._last_bc_teacher_imitation_mse.zero_()
            self._last_bc_teacher_action_abs_mean.zero_()
            self._last_candidate6_diffik_hold_success_rate.zero_()
            self._last_candidate8_diffik_target_residual_abs_mean.zero_()
            self._last_candidate8_diffik_target_residual_abs_max.zero_()
            self._last_candidate8_diffik_target_residual_forward_abs.zero_()
            self._last_candidate8_diffik_target_residual_lateral_abs.zero_()
            self._last_candidate8_diffik_target_residual_height_abs.zero_()
            self._last_tap_push_primitive_stop_latched.zero_()
            self._last_tap_push_primitive_target_delta_abs_mean.zero_()
            self._last_tap_push_primitive_target_delta_abs_max.zero_()
            return

        action_mode = str(getattr(self.cfg, "rl_action_mode", "joint_delta"))
        if action_mode != "tap_push_primitive":
            self._last_tap_push_primitive_stop_latched.zero_()
            self._last_tap_push_primitive_target_delta_abs_mean.zero_()
            self._last_tap_push_primitive_target_delta_abs_max.zero_()
        if action_mode == "tap_push_primitive":
            policy_actions = actions.clone().clamp(-1.0, 1.0)
            targets = self._tap_push_primitive_joint_target()

            joint_pos = self._robot.data.joint_pos
            applied_delta = targets - joint_pos
            self.actions = policy_actions
            self.robot_dof_targets[:] = targets
            self._last_action_abs_mean[:] = torch.mean(torch.abs(self.actions), dim=-1)
            self._last_action_abs_max[:] = torch.max(torch.abs(self.actions), dim=-1).values
            self._last_joint_delta_abs_mean[:] = torch.mean(torch.abs(applied_delta), dim=-1)
            self._last_joint_delta_abs_max[:] = torch.max(torch.abs(applied_delta), dim=-1).values
            self._last_joint_delta_cap_rate.zero_()
            self._last_target_lead_abs_mean[:] = self._last_joint_delta_abs_mean
            self._last_target_lead_abs_max[:] = self._last_joint_delta_abs_max
            lead = float(self.cfg.joint_target_lead_limit_rad)
            self._last_target_lead_limit_rate[:] = (
                torch.abs(targets - joint_pos) > lead + 1.0e-9
            ).float().mean(dim=-1)
            self._last_contact_slowdown[:] = 1.0
            self._last_teacher_blend.zero_()
            self._last_bc_teacher_blend.zero_()
            self._last_bc_teacher_imitation_mse.zero_()
            self._last_bc_teacher_action_abs_mean.zero_()
            self._last_candidate6_diffik_residual_abs_mean.zero_()
            self._last_candidate6_diffik_residual_abs_max.zero_()
            self._last_candidate6_diffik_hold_success_rate.zero_()
            self._last_candidate8_diffik_target_residual_abs_mean.zero_()
            self._last_candidate8_diffik_target_residual_abs_max.zero_()
            self._last_candidate8_diffik_target_residual_forward_abs.zero_()
            self._last_candidate8_diffik_target_residual_lateral_abs.zero_()
            self._last_candidate8_diffik_target_residual_height_abs.zero_()
            if hasattr(self, "_last_tap_stop_after_useful_hold"):
                self._last_tap_stop_after_useful_hold.zero_()
            if hasattr(self, "_last_tap_stop_after_disp_hold"):
                self._last_tap_stop_after_disp_hold.zero_()
            return
        if action_mode == "candidate6_diffik_residual_joint":
            policy_actions = actions.clone().clamp(-1.0, 1.0)
            base_targets = self._candidate6_diffik_base_joint_target()
            residual = torch.zeros_like(base_targets)
            arm_joint_ids = self._bc_arm_joint_ids
            residual[:, arm_joint_ids] = (
                policy_actions[:, arm_joint_ids] * float(self.cfg.candidate6_diffik_residual_scale_rad)
            )
            residual[:, self.gripper_joint_idx] = 0.0
            targets_unclamped = base_targets + residual

            joint_pos = self._robot.data.joint_pos
            lead_before = torch.abs(targets_unclamped - joint_pos)
            self._last_target_lead_abs_mean[:] = torch.mean(lead_before, dim=-1)
            self._last_target_lead_abs_max[:] = torch.max(lead_before, dim=-1).values
            lead = float(self.cfg.joint_target_lead_limit_rad)
            self._last_target_lead_limit_rate[:] = (
                torch.abs(targets_unclamped - joint_pos) > lead + 1.0e-9
            ).float().mean(dim=-1)
            targets = torch.maximum(torch.minimum(targets_unclamped, joint_pos + lead), joint_pos - lead)
            targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            targets[:, self.gripper_joint_idx] = 0.0

            applied_delta = targets - joint_pos
            self.actions = policy_actions
            self.robot_dof_targets[:] = targets
            self._last_action_abs_mean[:] = torch.mean(torch.abs(self.actions), dim=-1)
            self._last_action_abs_max[:] = torch.max(torch.abs(self.actions), dim=-1).values
            self._last_joint_delta_abs_mean[:] = torch.mean(torch.abs(applied_delta), dim=-1)
            self._last_joint_delta_abs_max[:] = torch.max(torch.abs(applied_delta), dim=-1).values
            self._last_joint_delta_cap_rate.zero_()
            self._last_contact_slowdown[:] = 1.0
            self._last_teacher_blend.zero_()
            self._last_bc_teacher_blend.zero_()
            self._last_bc_teacher_imitation_mse.zero_()
            self._last_bc_teacher_action_abs_mean.zero_()
            self._last_candidate6_diffik_residual_abs_mean[:] = torch.mean(torch.abs(residual), dim=-1)
            self._last_candidate6_diffik_residual_abs_max[:] = torch.max(torch.abs(residual), dim=-1).values
            self._last_candidate8_diffik_target_residual_abs_mean.zero_()
            self._last_candidate8_diffik_target_residual_abs_max.zero_()
            self._last_candidate8_diffik_target_residual_forward_abs.zero_()
            self._last_candidate8_diffik_target_residual_lateral_abs.zero_()
            self._last_candidate8_diffik_target_residual_height_abs.zero_()
            return
        if action_mode == "candidate8_diffik_target_residual":
            policy_actions = actions.clone().clamp(-1.0, 1.0)
            if policy_actions.shape[-1] != 3:
                raise ValueError(
                    "candidate8_diffik_target_residual expects exactly 3 policy "
                    f"actions (forward, lateral, height), got {policy_actions.shape[-1]}"
                )
            if (
                bool(getattr(self.cfg, "candidate8_diffik_target_residual_zero_after_contact", False))
                or bool(getattr(self.cfg, "candidate8_diffik_target_residual_zero_after_reaction", False))
                or float(getattr(self.cfg, "candidate8_diffik_target_residual_zero_after_disp_m", 0.0)) > 0.0
            ):
                raise ValueError(
                    "candidate8_diffik_target_residual is the clean 3D target-residual "
                    "action space; post-contact/reaction/displacement gates are disabled"
                )
            push_dir = self._push_dir_xy
            lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=-1)
            forward_m = policy_actions[:, 0] * float(self.cfg.candidate8_diffik_target_residual_forward_m)
            lateral_m = policy_actions[:, 1] * float(self.cfg.candidate8_diffik_target_residual_lateral_m)
            height_m = policy_actions[:, 2] * float(self.cfg.candidate8_diffik_target_residual_height_m)
            target_residual_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
            target_residual_w[:, 0:2] = (
                push_dir * forward_m.unsqueeze(-1)
                + lateral_dir * lateral_m.unsqueeze(-1)
            )
            target_residual_w[:, 2] = height_m

            targets = self._candidate6_diffik_base_joint_target(tcp_target_residual_w=target_residual_w)
            joint_pos = self._robot.data.joint_pos
            applied_delta = targets - joint_pos
            self.actions = policy_actions
            self.robot_dof_targets[:] = targets
            self._last_action_abs_mean[:] = torch.mean(torch.abs(self.actions), dim=-1)
            self._last_action_abs_max[:] = torch.max(torch.abs(self.actions), dim=-1).values
            self._last_joint_delta_abs_mean[:] = torch.mean(torch.abs(applied_delta), dim=-1)
            self._last_joint_delta_abs_max[:] = torch.max(torch.abs(applied_delta), dim=-1).values
            self._last_joint_delta_cap_rate.zero_()
            self._last_target_lead_abs_mean[:] = self._last_joint_delta_abs_mean
            self._last_target_lead_abs_max[:] = self._last_joint_delta_abs_max
            lead = float(self.cfg.joint_target_lead_limit_rad)
            self._last_target_lead_limit_rate[:] = (
                torch.abs(targets - joint_pos) > lead + 1.0e-9
            ).float().mean(dim=-1)
            self._last_contact_slowdown[:] = 1.0
            self._last_teacher_blend.zero_()
            self._last_bc_teacher_blend.zero_()
            self._last_bc_teacher_imitation_mse.zero_()
            self._last_bc_teacher_action_abs_mean.zero_()
            self._last_candidate6_diffik_residual_abs_mean.zero_()
            self._last_candidate6_diffik_residual_abs_max.zero_()
            self._last_candidate8_diffik_target_residual_abs_mean[:] = torch.mean(
                torch.abs(target_residual_w), dim=-1
            )
            self._last_candidate8_diffik_target_residual_abs_max[:] = torch.max(
                torch.abs(target_residual_w), dim=-1
            ).values
            self._last_candidate8_diffik_target_residual_forward_abs[:] = torch.abs(forward_m)
            self._last_candidate8_diffik_target_residual_lateral_abs[:] = torch.abs(lateral_m)
            self._last_candidate8_diffik_target_residual_height_abs[:] = torch.abs(height_m)
            return
        if action_mode != "joint_delta":
            raise ValueError(f"unsupported rl_action_mode={action_mode!r}")

        policy_actions = actions.clone().clamp(-1.0, 1.0)
        self._last_candidate6_diffik_active.zero_()
        self._last_candidate6_diffik_numeric_ok.zero_()
        self._last_candidate6_diffik_raw_delta_abs_max.zero_()
        self._last_candidate6_diffik_clipped_delta_abs_max.zero_()
        self._last_candidate6_diffik_step_clip_rate.zero_()
        self._last_candidate6_diffik_residual_abs_mean.zero_()
        self._last_candidate6_diffik_residual_abs_max.zero_()
        self._last_candidate6_diffik_hold_success_rate.zero_()
        self._last_candidate8_diffik_target_residual_abs_mean.zero_()
        self._last_candidate8_diffik_target_residual_abs_max.zero_()
        self._last_candidate8_diffik_target_residual_forward_abs.zero_()
        self._last_candidate8_diffik_target_residual_lateral_abs.zero_()
        self._last_candidate8_diffik_target_residual_height_abs.zero_()
        teacher_blend = torch.zeros(self.num_envs, device=self.device)
        if float(self.cfg.scripted_teacher_blend) > 0.0:
            horizon = max(1, int(float(self.cfg.scripted_teacher_horizon_frac) * float(self.max_episode_length)))
            phase = torch.clamp((self.episode_length_buf.float() + 1.0) / float(horizon), 0.0, 1.0)
            teacher_target = self._teacher_start_joints + phase.unsqueeze(-1) * (
                self._teacher_goal_joints - self._teacher_start_joints
            )
            denom = max(float(self.cfg.action_scale), 1.0e-6)
            teacher_actions = torch.clamp((teacher_target - self.robot_dof_targets) / denom, -1.0, 1.0)
            teacher_actions[:, self.gripper_joint_idx] = 0.0
            teacher_blend = torch.where(
                self._teacher_goal_ok,
                torch.full_like(teacher_blend, float(self.cfg.scripted_teacher_blend)),
                teacher_blend,
            )
            teacher_blend = torch.where(phase < 1.0, teacher_blend, torch.zeros_like(teacher_blend))
            self.actions = (1.0 - teacher_blend.unsqueeze(-1)) * policy_actions + teacher_blend.unsqueeze(-1) * teacher_actions
        else:
            self.actions = policy_actions
        self._last_teacher_blend[:] = teacher_blend

        bc_blend_value = float(self.cfg.bc_teacher_blend)
        bc_reward_scale = float(self.cfg.bc_teacher_imitation_reward_scale)
        self._last_bc_teacher_blend.zero_()
        self._last_bc_teacher_imitation_mse.zero_()
        self._last_bc_teacher_action_abs_mean.zero_()
        if hasattr(self, "_last_tap_stop_after_useful_hold"):
            self._last_tap_stop_after_useful_hold.zero_()
        if hasattr(self, "_last_tap_stop_after_disp_hold"):
            self._last_tap_stop_after_disp_hold.zero_()
        if getattr(self, "_bc_teacher_ready", False) and (bc_blend_value > 0.0 or bc_reward_scale > 0.0):
            bc_teacher_actions = self._bc_teacher_actions()
            bc_blend = torch.full((self.num_envs,), max(0.0, min(1.0, bc_blend_value)), device=self.device)
            if bc_blend_value > 0.0:
                self.actions = (1.0 - bc_blend.unsqueeze(-1)) * self.actions + bc_blend.unsqueeze(-1) * bc_teacher_actions
            self._last_bc_teacher_blend[:] = bc_blend
            self._last_bc_teacher_imitation_mse[:] = torch.mean((policy_actions - bc_teacher_actions) ** 2, dim=-1)
            self._last_bc_teacher_action_abs_mean[:] = torch.mean(torch.abs(bc_teacher_actions), dim=-1)
        if bool(getattr(self.cfg, "tap_stop_after_useful_seen", False)) and hasattr(self, "_tap_contact_seen"):
            useful_min_disp_m = max(float(getattr(self.cfg, "tap_useful_min_disp_m", 0.001)), 0.0)
            useful_hold = (
                self._tap_contact_seen
                & self._tap_reaction_seen
                & (self._tap_max_disp_xy >= useful_min_disp_m)
                & ~self._tap_overshoot_seen
            )
            self.actions = torch.where(useful_hold.unsqueeze(-1), torch.zeros_like(self.actions), self.actions)
            if hasattr(self, "_last_tap_stop_after_useful_hold"):
                self._last_tap_stop_after_useful_hold[:] = useful_hold.float()
        tap_stop_after_disp_m = float(getattr(self.cfg, "tap_stop_after_disp_m", 0.0))
        if tap_stop_after_disp_m > 0.0 and hasattr(self, "_tap_max_disp_xy"):
            disp_hold = (self._tap_max_disp_xy >= tap_stop_after_disp_m) & ~self._tap_overshoot_seen
            self.actions = torch.where(disp_hold.unsqueeze(-1), torch.zeros_like(self.actions), self.actions)
            if hasattr(self, "_last_tap_stop_after_disp_hold"):
                self._last_tap_stop_after_disp_hold[:] = disp_hold.float()
        self._apply_tap_action_governor_if_enabled()

        alpha = float(self.cfg.action_smoothing_alpha)
        self._last_action_abs_mean[:] = torch.mean(torch.abs(self.actions), dim=-1)
        self._last_action_abs_max[:] = torch.max(torch.abs(self.actions), dim=-1).values
        self._smoothed_actions[:] = (1.0 - alpha) * self._smoothed_actions + alpha * self.actions
        raw_delta = self.cfg.action_scale * self._smoothed_actions
        max_delta = float(self.cfg.max_joint_delta_per_step_rad)
        self._last_joint_delta_cap_rate[:] = (torch.abs(raw_delta) >= max_delta - 1.0e-9).float().mean(dim=-1)
        delta = torch.clamp(
            raw_delta,
            -max_delta,
            max_delta,
        )

        self._compute_intermediate_values()
        terms = self._push_terms()
        slowdown = torch.ones(self.num_envs, device=self.device)
        contact_slowdown_mask = terms["tcp_cube_dist"] < float(self.cfg.contact_slowdown_tcp_dist_m)
        if bool(getattr(self.cfg, "tap_contact_slowdown_use_proxy", False)) and hasattr(self, "_tap_terms"):
            tap_terms = self._tap_terms()
            contact_slowdown_mask = contact_slowdown_mask | tap_terms["tap_contact_proxy"]
        slowdown = torch.where(
            contact_slowdown_mask,
            torch.full_like(slowdown, float(self.cfg.contact_joint_delta_scale)),
            slowdown,
        )
        slowdown = torch.where(
            terms["speed"] > float(self.cfg.speed_penalty_start_mps),
            torch.minimum(slowdown, torch.full_like(slowdown, float(self.cfg.fast_cube_joint_delta_scale))),
            slowdown,
        )
        delta = delta * slowdown.unsqueeze(-1)
        delta[:, self.gripper_joint_idx] = 0.0

        joint_pos = self._robot.data.joint_pos
        lead_before = torch.abs(self.robot_dof_targets - joint_pos)
        self._last_target_lead_abs_mean[:] = torch.mean(lead_before, dim=-1)
        self._last_target_lead_abs_max[:] = torch.max(lead_before, dim=-1).values
        reference = str(getattr(self.cfg, "joint_delta_reference", "target"))
        if reference == "target":
            target_base = self.robot_dof_targets
        elif reference == "joint_pos":
            target_base = joint_pos
        else:
            raise ValueError(f"unsupported joint_delta_reference={reference!r}")
        targets_unclamped = target_base + delta
        lead = float(self.cfg.joint_target_lead_limit_rad)
        self._last_target_lead_limit_rate[:] = (
            torch.abs(targets_unclamped - joint_pos) > lead + 1.0e-9
        ).float().mean(dim=-1)
        targets = targets_unclamped
        targets = torch.maximum(torch.minimum(targets, joint_pos + lead), joint_pos - lead)
        targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        targets[:, self.gripper_joint_idx] = 0.0

        self.robot_dof_targets[:] = targets
        self._last_joint_delta_abs_mean[:] = torch.mean(torch.abs(delta), dim=-1)
        self._last_joint_delta_abs_max[:] = torch.max(torch.abs(delta), dim=-1).values
        self._last_contact_slowdown[:] = slowdown

    def _apply_action(self):
        # No attach path in this task: robot joint target writes only.
        self._grasped[:] = False
        self._was_grasped[:] = False
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _apply_tap_action_governor_if_enabled(self) -> None:
        mode = str(getattr(self.cfg, "tap_action_governor_mode", "off"))
        if mode == "off":
            if hasattr(self, "_last_tap_action_governor_stop_latched"):
                self._last_tap_action_governor_stop_latched.zero_()
                self._last_tap_action_governor_brake_active.zero_()
                self._last_tap_action_governor_projected_disp.zero_()
                self._last_tap_action_governor_contact_age.zero_()
            return
        if mode not in {"predict_stop", "predict_brake"}:
            raise ValueError(f"unsupported tap_action_governor_mode={mode!r}")
        if not hasattr(self, "_tap_action_governor_stop_latched"):
            return

        target_disp = float(getattr(self.cfg, "tap_action_governor_target_disp_m", 0.003))
        horizon_s = float(getattr(self.cfg, "tap_action_governor_predict_horizon_s", 0.020))
        speed_stop = float(getattr(self.cfg, "tap_action_governor_speed_stop_mps", 0.200))
        min_contact_steps = int(getattr(self.cfg, "tap_action_governor_min_contact_steps", 1))
        push_scale = float(getattr(self.cfg, "tap_action_governor_push_scale", 1.0))
        brake_scale = float(getattr(self.cfg, "tap_action_governor_brake_scale", 0.35))
        brake_steps = int(getattr(self.cfg, "tap_action_governor_brake_steps", 2))
        if target_disp <= 0.0:
            raise ValueError("tap_action_governor_target_disp_m must be positive")
        if horizon_s < 0.0:
            raise ValueError("tap_action_governor_predict_horizon_s must be non-negative")
        if speed_stop < 0.0:
            raise ValueError("tap_action_governor_speed_stop_mps must be non-negative")
        if min_contact_steps < 0:
            raise ValueError("tap_action_governor_min_contact_steps must be non-negative")
        if not (0.0 <= push_scale <= 1.0):
            raise ValueError("tap_action_governor_push_scale must be in [0, 1]")
        if not (0.0 <= brake_scale <= 1.0):
            raise ValueError("tap_action_governor_brake_scale must be in [0, 1]")
        if brake_steps < 0:
            raise ValueError("tap_action_governor_brake_steps must be non-negative")

        pre_terms = self._tap_terms()
        self._tap_action_governor_contact_age[:] = torch.where(
            pre_terms["tap_contact_proxy"] | self._tap_contact_seen,
            self._tap_action_governor_contact_age + 1,
            self._tap_action_governor_contact_age,
        )
        projected_disp = pre_terms["disp_xy"] + pre_terms["speed"] * horizon_s
        can_stop = self._tap_action_governor_contact_age >= min_contact_steps
        stop_now = can_stop & (
            (pre_terms["disp_xy"] >= target_disp)
            | (projected_disp >= target_disp)
            | (pre_terms["speed"] >= speed_stop)
        )
        newly_stopped = stop_now & ~self._tap_action_governor_stop_latched
        self._tap_action_governor_stop_step[:] = torch.where(
            newly_stopped,
            self.episode_length_buf.to(dtype=torch.long),
            self._tap_action_governor_stop_step,
        )
        self._tap_action_governor_stop_latched |= stop_now

        brake_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if mode == "predict_brake":
            self._tap_action_governor_brake_source_actions[:] = torch.where(
                newly_stopped.unsqueeze(-1),
                self._tap_action_governor_prev_actions,
                self._tap_action_governor_brake_source_actions,
            )
            self._tap_action_governor_brake_remaining[:] = torch.where(
                newly_stopped,
                torch.full_like(self._tap_action_governor_brake_remaining, brake_steps),
                self._tap_action_governor_brake_remaining,
            )
            brake_mask = self._tap_action_governor_brake_remaining > 0
            brake_actions = -brake_scale * self._tap_action_governor_brake_source_actions
            self.actions = torch.where(
                self._tap_action_governor_stop_latched.unsqueeze(-1),
                torch.zeros_like(self.actions),
                self.actions,
            )
            self.actions = torch.where(brake_mask.unsqueeze(-1), brake_actions, self.actions)
            self._tap_action_governor_brake_remaining[:] = torch.clamp(
                self._tap_action_governor_brake_remaining - brake_mask.long(),
                min=0,
            )
        else:
            self.actions = torch.where(
                self._tap_action_governor_stop_latched.unsqueeze(-1),
                torch.zeros_like(self.actions),
                self.actions,
            )
        self.actions = torch.clamp(self.actions * push_scale, -1.0, 1.0)
        self._tap_action_governor_prev_actions[:] = self.actions.detach()
        self._last_tap_action_governor_stop_latched[:] = self._tap_action_governor_stop_latched.float()
        self._last_tap_action_governor_brake_active[:] = brake_mask.float()
        self._last_tap_action_governor_projected_disp[:] = projected_disp.detach()
        self._last_tap_action_governor_contact_age[:] = self._tap_action_governor_contact_age.float()

    def _update_grasp_attach(self):
        # Defense-in-depth: even if a future code path calls this, object pose writes
        # remain disabled for cube-push learning.
        return

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        super()._compute_intermediate_values(env_ids)
        self._grasped[:] = False
        self._was_grasped[:] = False

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self._ensure_push_buffers()
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)
        d256_reset = self._sample_d256_reset_table(n)

        if d256_reset is None:
            sx = sample_uniform(self.cfg.cube_x_min, self.cfg.cube_x_max, (n,), self.device)
            sy = sample_uniform(self.cfg.cube_y_min, self.cfg.cube_y_max, (n,), self.device)
            cube_center_z = TABLE_Z + 0.5 * float(self.cfg.cube_size_z_m)
            sz = torch.full((n,), cube_center_z, device=self.device)

            dirs = torch.tensor(
                ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
                device=self.device,
                dtype=torch.float32,
            )
            if math.isfinite(float(self.cfg.fixed_push_dir_x)) and math.isfinite(float(self.cfg.fixed_push_dir_y)):
                fixed_dir = torch.tensor(
                    [float(self.cfg.fixed_push_dir_x), float(self.cfg.fixed_push_dir_y)],
                    device=self.device,
                    dtype=torch.float32,
                )
                fixed_norm = torch.linalg.norm(fixed_dir)
                if float(fixed_norm.detach().cpu().item()) <= 1.0e-6:
                    raise ValueError("fixed push direction must be nonzero")
                push_dir = fixed_dir.unsqueeze(0).repeat(n, 1) / fixed_norm
            else:
                dir_idx = torch.randint(0, 4, (n,), device=self.device)
                push_dir = dirs[dir_idx]

            cube_local = torch.stack([sx, sy, sz], dim=-1)
            target_local = cube_local.clone()
            target_local[:, 0:2] = target_local[:, 0:2] + push_dir * self.cfg.cube_push_target_disp_m
        else:
            cube_local = d256_reset["cube_local"]
            target_local = d256_reset["target_local"]
            push_dir = d256_reset["push_dir"]

        if d256_reset is not None:
            joint_pos = self._robot.data.joint_pos.detach().clone()[env_ids]
            joint_pos[:, self._bc_arm_joint_ids] = d256_reset["arm"]
            joint_pos[:, self.gripper_joint_idx] = d256_reset["gripper"]
            ik_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
            ik_err_mm = torch.zeros(n, device=self.device)
        elif self.cfg.ik_endpoint_reset:
            joint_pos, ik_ok, ik_err_mm = self._ik_precontact_joints(cube_local, push_dir)
            if self.cfg.ik_reset_jitter_rad > 0.0:
                jitter = sample_uniform(
                    -self.cfg.ik_reset_jitter_rad,
                    self.cfg.ik_reset_jitter_rad,
                    (n, self._robot.num_joints),
                    self.device,
                )
                jitter[:, self.gripper_joint_idx] = 0.0
                joint_pos = joint_pos + jitter
            home_pos = torch.tensor(HOME_RAD, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
            joint_pos = torch.where(ik_ok.unsqueeze(-1), joint_pos, home_pos)
        else:
            joint_pos = torch.tensor(HOME_RAD, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
            joint_pos = joint_pos + sample_uniform(-0.02, 0.02, (n, self._robot.num_joints), self.device)
            ik_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
            ik_err_mm = torch.zeros(n, device=self.device)
        if d256_reset is None:
            joint_pos[:, self.gripper_joint_idx] = 0.0
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        env_origins = self.scene.env_origins[env_ids]
        cube_world = env_origins + cube_local
        cube_quat = torch.zeros((n, 4), device=self.device)
        cube_quat[:, 0] = 1.0
        cube_state = torch.zeros((n, 13), device=self.device)
        cube_state[:, 0:3] = cube_world
        cube_state[:, 3:7] = cube_quat
        self._sponge.write_root_pose_to_sim(cube_state[:, 0:7], env_ids=env_ids)
        self._sponge.write_root_velocity_to_sim(cube_state[:, 7:13], env_ids=env_ids)

        self._target_world[env_ids] = env_origins + target_local

        self._cube_start_w[env_ids] = cube_world
        self._push_dir_xy[env_ids] = push_dir
        self._ik_reset_ok[env_ids] = ik_ok
        self._ik_reset_err_mm[env_ids] = ik_err_mm
        self._prev_disp_along[env_ids] = 0.0
        self._push_success_flag[env_ids] = False
        self._smoothed_actions[env_ids] = 0.0
        self._last_joint_delta_abs_mean[env_ids] = 0.0
        self._last_joint_delta_abs_max[env_ids] = 0.0
        self._last_joint_delta_cap_rate[env_ids] = 0.0
        self._last_action_abs_mean[env_ids] = 0.0
        self._last_action_abs_max[env_ids] = 0.0
        self._last_target_lead_abs_mean[env_ids] = 0.0
        self._last_target_lead_abs_max[env_ids] = 0.0
        self._last_target_lead_limit_rate[env_ids] = 0.0
        self._last_contact_slowdown[env_ids] = 1.0
        self._last_teacher_blend[env_ids] = 0.0
        self._last_bc_teacher_blend[env_ids] = 0.0
        self._last_bc_teacher_imitation_mse[env_ids] = 0.0
        self._last_bc_teacher_action_abs_mean[env_ids] = 0.0
        self._last_d256_reset_active[env_ids] = 1.0 if d256_reset is not None else 0.0
        if d256_reset is None:
            self._last_d256_reset_episode_index[env_ids] = -1.0
        else:
            self._last_d256_reset_episode_index[env_ids] = d256_reset["episode_index"]
        self._bc_prev_teacher_delta[env_ids] = 0.0
        self._candidate6_prev_arm_joint_target[env_ids] = joint_pos[:, self._bc_arm_joint_ids]
        self._candidate6_prev_arm_joint_target_valid[env_ids] = False
        self._last_candidate6_diffik_active[env_ids] = 0.0
        self._last_candidate6_diffik_numeric_ok[env_ids] = 0.0
        self._last_candidate6_diffik_raw_delta_abs_max[env_ids] = 0.0
        self._last_candidate6_diffik_clipped_delta_abs_max[env_ids] = 0.0
        self._last_candidate6_diffik_step_clip_rate[env_ids] = 0.0
        self._last_candidate6_diffik_residual_abs_mean[env_ids] = 0.0
        self._last_candidate6_diffik_residual_abs_max[env_ids] = 0.0
        self._last_candidate6_diffik_hold_success_rate[env_ids] = 0.0
        self._last_candidate8_diffik_target_residual_abs_mean[env_ids] = 0.0
        self._last_candidate8_diffik_target_residual_abs_max[env_ids] = 0.0
        self._last_candidate8_diffik_target_residual_forward_abs[env_ids] = 0.0
        self._last_candidate8_diffik_target_residual_lateral_abs[env_ids] = 0.0
        self._last_candidate8_diffik_target_residual_height_abs[env_ids] = 0.0
        self._tap_push_primitive_stop_latched[env_ids] = False
        self._tap_push_primitive_stop_step[env_ids] = -1
        self._tap_push_primitive_hold_targets[env_ids] = joint_pos
        self._last_tap_push_primitive_stop_latched[env_ids] = 0.0
        self._last_tap_push_primitive_target_delta_abs_mean[env_ids] = 0.0
        self._last_tap_push_primitive_target_delta_abs_max[env_ids] = 0.0
        self._teacher_start_joints[env_ids] = joint_pos
        if float(self.cfg.scripted_teacher_blend) > 0.0:
            teacher_goal, teacher_ok = self._ik_teacher_goal_joints(cube_local, push_dir, joint_pos)
        else:
            teacher_goal = joint_pos
            teacher_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._teacher_goal_joints[env_ids] = teacher_goal
        self._teacher_goal_ok[env_ids] = teacher_ok
        self._grasped[env_ids] = False
        self._was_grasped[env_ids] = False
        self._lift_counter[env_ids] = 0
        self._lift_success_flag[env_ids] = False
        self._lift_bonus_paid[env_ids] = False
        self._place_counter[env_ids] = 0
        self._place_success_flag[env_ids] = False
        self._place_bonus_paid[env_ids] = False
        self._stage3_fired[env_ids] = False

        self._compute_intermediate_values(env_ids)

    def _push_terms(self) -> dict[str, torch.Tensor]:
        self._ensure_push_buffers()
        disp_xy_vec = self._sponge_pos_w[:, 0:2] - self._cube_start_w[:, 0:2]
        disp_xy = torch.norm(disp_xy_vec, p=2, dim=-1)
        disp_along = torch.sum(disp_xy_vec * self._push_dir_xy, dim=-1)
        lateral_vec = disp_xy_vec - disp_along.unsqueeze(-1) * self._push_dir_xy
        lateral_abs = torch.norm(lateral_vec, p=2, dim=-1)
        target_xy_dist = torch.norm(self._target_world[:, 0:2] - self._sponge_pos_w[:, 0:2], p=2, dim=-1)
        tcp_cube_dist = torch.norm(self._tcp_pos_w - self._sponge_pos_w, p=2, dim=-1)
        speed = torch.norm(self._sponge.data.root_lin_vel_w, p=2, dim=-1)
        quat_w = torch.clamp(torch.abs(self._sponge_quat_w[:, 0]), 0.0, 1.0)
        tip_angle_deg = torch.rad2deg(2.0 * torch.acos(quat_w))
        controlled = (
            (disp_along >= 0.001)
            & (speed <= AUDIT_SPEED_P95_MPS)
            & (tip_angle_deg <= AUDIT_TIP_P95_DEG)
            & (disp_xy <= AUDIT_DISP_XY_P99_M)
        )
        impact = (
            (speed > AUDIT_SPEED_P99_MPS)
            | (disp_xy > AUDIT_DISP_XY_P99_M)
            | (tip_angle_deg > AUDIT_TIP_P99_DEG)
        )
        low_motion = disp_xy < self.cfg.cube_low_motion_disp_m
        far_target = target_xy_dist > self.cfg.cube_far_target_terminate_m
        terminal_impact = impact | far_target | (disp_xy > self.cfg.cube_impact_terminate_disp_m)
        return {
            "disp_xy": disp_xy,
            "disp_along": disp_along,
            "lateral_abs": lateral_abs,
            "target_xy_dist": target_xy_dist,
            "tcp_cube_dist": tcp_cube_dist,
            "speed": speed,
            "tip_angle_deg": tip_angle_deg,
            "controlled": controlled,
            "impact": impact,
            "low_motion": low_motion,
            "far_target": far_target,
            "terminal_impact": terminal_impact,
        }

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        terms = self._push_terms()

        progress = torch.clamp(terms["disp_along"] - self._prev_disp_along, min=-0.02, max=0.02)
        self._prev_disp_along[:] = terms["disp_along"].detach()

        target_score = 1.0 - torch.tanh(20.0 * terms["target_xy_dist"])
        displacement_score = torch.clamp(terms["disp_along"], min=-0.03, max=self.cfg.cube_push_target_disp_m)
        reverse_push = torch.clamp(-terms["disp_along"], min=0.0, max=0.05)
        target_distance_penalty = torch.clamp(
            terms["target_xy_dist"] - self.cfg.cube_success_target_tol_m,
            min=0.0,
            max=0.50,
        )
        speed_penalty = torch.clamp(
            terms["speed"] - self.cfg.speed_penalty_start_mps,
            min=0.0,
            max=5.0,
        )
        overshoot_penalty = torch.clamp(
            terms["disp_xy"] - self.cfg.cube_push_target_disp_m * 1.5,
            min=0.0,
            max=0.50,
        )
        elapsed_gate = (self.episode_length_buf > 5).float()
        gripper_q = torch.clamp(self._robot.data.joint_pos[:, self.gripper_joint_idx], min=0.0)

        action_penalty = -torch.sum(self.actions ** 2, dim=-1) * self.cfg.action_penalty_scale
        bc_imitation_penalty = -float(self.cfg.bc_teacher_imitation_reward_scale) * self._last_bc_teacher_imitation_mse
        rewards = (
            self.cfg.push_progress_reward_scale * progress
            + self.cfg.push_displacement_reward_scale * displacement_score
            + self.cfg.push_target_reward_scale * target_score
            - self.cfg.tcp_cube_distance_penalty_scale * terms["tcp_cube_dist"]
            - self.cfg.gripper_close_penalty_scale * gripper_q
            - self.cfg.impact_penalty_scale * terms["impact"].float()
            - self.cfg.low_motion_penalty_scale * terms["low_motion"].float() * elapsed_gate
            - self.cfg.reverse_push_penalty_scale * reverse_push
            - self.cfg.target_distance_penalty_scale * target_distance_penalty
            - self.cfg.lateral_penalty_scale * terms["lateral_abs"]
            - self.cfg.overshoot_penalty_scale * overshoot_penalty
            - self.cfg.speed_penalty_scale * speed_penalty
            - self.cfg.impact_terminal_penalty * terms["terminal_impact"].float()
            + self.cfg.controlled_bonus_scale * terms["controlled"].float()
            + action_penalty
            + bc_imitation_penalty
        )

        success_now = (
            terms["controlled"]
            & ~terms["impact"]
            & (terms["disp_along"] >= self.cfg.cube_success_disp_m)
            & (terms["target_xy_dist"] <= self.cfg.cube_success_target_tol_m)
            & (terms["speed"] <= self.cfg.cube_success_speed_max_mps)
        )
        just_succeeded = success_now & ~self._push_success_flag
        self._push_success_flag = self._push_success_flag | success_now
        rewards = rewards + self.cfg.success_bonus * just_succeeded.float()

        self.extras["log"] = {
            "cube_push_disp_along_m": terms["disp_along"].mean().detach(),
            "cube_push_disp_xy_m": terms["disp_xy"].mean().detach(),
            "cube_push_target_xy_dist_m": terms["target_xy_dist"].mean().detach(),
            "cube_push_tcp_cube_dist_m": terms["tcp_cube_dist"].mean().detach(),
            "cube_push_speed_mps": terms["speed"].mean().detach(),
            "cube_push_speed_over_0p5_rate": (terms["speed"] > self.cfg.speed_penalty_start_mps).float().mean().detach(),
            "cube_push_tip_angle_deg": terms["tip_angle_deg"].mean().detach(),
            "cube_push_controlled_rate": terms["controlled"].float().mean().detach(),
            "cube_push_impact_rate": terms["impact"].float().mean().detach(),
            "cube_push_low_motion_rate": terms["low_motion"].float().mean().detach(),
            "cube_push_far_target_rate": terms["far_target"].float().mean().detach(),
            "cube_push_terminal_impact_rate": terms["terminal_impact"].float().mean().detach(),
            "cube_push_success_rate": self._push_success_flag.float().mean().detach(),
            "cube_push_grasped_marker_rate": self._grasped.float().mean().detach(),
            "cube_push_ik_endpoint_reset_rate": self._ik_reset_ok.float().mean().detach(),
            "cube_push_ik_reset_err_mm": self._ik_reset_err_mm.mean().detach(),
            "cube_push_joint_delta_abs_mean": self._last_joint_delta_abs_mean.mean().detach(),
            "cube_push_joint_delta_abs_max": self._last_joint_delta_abs_max.mean().detach(),
            "cube_push_joint_delta_cap_rate": self._last_joint_delta_cap_rate.mean().detach(),
            "cube_push_action_abs_mean": self._last_action_abs_mean.mean().detach(),
            "cube_push_action_abs_max": self._last_action_abs_max.mean().detach(),
            "cube_push_target_lead_abs_mean": self._last_target_lead_abs_mean.mean().detach(),
            "cube_push_target_lead_abs_max": self._last_target_lead_abs_max.mean().detach(),
            "cube_push_target_lead_limit_rate": self._last_target_lead_limit_rate.mean().detach(),
            "cube_push_contact_slowdown_mean": self._last_contact_slowdown.mean().detach(),
            "cube_push_teacher_blend_mean": self._last_teacher_blend.mean().detach(),
            "cube_push_teacher_goal_ok_rate": self._teacher_goal_ok.float().mean().detach(),
            "cube_push_bc_teacher_blend_mean": self._last_bc_teacher_blend.mean().detach(),
            "cube_push_bc_teacher_imitation_mse": self._last_bc_teacher_imitation_mse.mean().detach(),
            "cube_push_bc_teacher_action_abs_mean": self._last_bc_teacher_action_abs_mean.mean().detach(),
            "cube_push_d256_reset_active_rate": self._last_d256_reset_active.mean().detach(),
            "cube_push_d256_reset_episode_index_mean": self._last_d256_reset_episode_index.mean().detach(),
            "cube_push_candidate6_diffik_active_rate": self._last_candidate6_diffik_active.mean().detach(),
            "cube_push_candidate6_diffik_numeric_ok_rate": self._last_candidate6_diffik_numeric_ok.mean().detach(),
            "cube_push_candidate6_diffik_raw_delta_abs_max": self._last_candidate6_diffik_raw_delta_abs_max.mean().detach(),
            "cube_push_candidate6_diffik_clipped_delta_abs_max": self._last_candidate6_diffik_clipped_delta_abs_max.mean().detach(),
            "cube_push_candidate6_diffik_step_clip_rate": self._last_candidate6_diffik_step_clip_rate.mean().detach(),
            "cube_push_candidate6_diffik_residual_abs_mean": self._last_candidate6_diffik_residual_abs_mean.mean().detach(),
            "cube_push_candidate6_diffik_residual_abs_max": self._last_candidate6_diffik_residual_abs_max.mean().detach(),
            "cube_push_candidate6_diffik_hold_success_rate": self._last_candidate6_diffik_hold_success_rate.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_abs_mean": self._last_candidate8_diffik_target_residual_abs_mean.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_abs_max": self._last_candidate8_diffik_target_residual_abs_max.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_forward_abs": self._last_candidate8_diffik_target_residual_forward_abs.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_lateral_abs": self._last_candidate8_diffik_target_residual_lateral_abs.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_height_abs": self._last_candidate8_diffik_target_residual_height_abs.mean().detach(),
            "cube_push_tap_push_primitive_enabled": torch.tensor(
                float(str(getattr(self.cfg, "rl_action_mode", "joint_delta")) == "tap_push_primitive"),
                device=self.device,
            ),
            "cube_push_tap_push_primitive_stop_latched_rate": self._last_tap_push_primitive_stop_latched.mean().detach(),
            "cube_push_tap_push_primitive_target_delta_abs_mean": self._last_tap_push_primitive_target_delta_abs_mean.mean().detach(),
            "cube_push_tap_push_primitive_target_delta_abs_max": self._last_tap_push_primitive_target_delta_abs_max.mean().detach(),
            "bc_teacher_imitation_penalty": bc_imitation_penalty.mean().detach(),
            "action_penalty": action_penalty.mean().detach(),
        }
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        terms = self._push_terms()
        success_now = (
            terms["controlled"]
            & ~terms["impact"]
            & (terms["disp_along"] >= self.cfg.cube_success_disp_m)
            & (terms["target_xy_dist"] <= self.cfg.cube_success_target_tol_m)
            & (terms["speed"] <= self.cfg.cube_success_speed_max_mps)
        )
        self._push_success_flag = self._push_success_flag | success_now
        terminated = success_now | terms["terminal_impact"]
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated


class RoArmCubeTap10cmEnv(RoArmCubePushEnv):
    """Default-off 10cm/0.72kg tap/reaction env with event-first metrics."""

    cfg: RoArmCubeTap10cmEnvCfg

    def __init__(self, cfg: RoArmCubeTap10cmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_tap_buffers()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._sponge = RigidObject(self.cfg.sponge)
        self._tap_table = RigidObject(self.cfg.tap_table)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["sponge"] = self._sponge
        self.scene.rigid_objects["tap_table"] = self._tap_table

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _ensure_tap_buffers(self) -> None:
        self._ensure_push_buffers()
        if hasattr(self, "_tap_contact_seen"):
            return
        self._tap_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tap_reaction_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._professor_physical_reaction_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tap_overshoot_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tap_success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tap_just_succeeded_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tap_max_disp_along = torch.zeros(self.num_envs, device=self.device)
        self._tap_max_disp_xy = torch.zeros(self.num_envs, device=self.device)
        self._tap_max_z_delta = torch.zeros(self.num_envs, device=self.device)
        self._tap_max_speed = torch.zeros(self.num_envs, device=self.device)
        self._tap_max_tip_angle_deg = torch.zeros(self.num_envs, device=self.device)
        self._tap_min_contact_vertical_offset = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._last_tap_stop_after_useful_hold = torch.zeros(self.num_envs, device=self.device)
        self._last_tap_stop_after_disp_hold = torch.zeros(self.num_envs, device=self.device)
        self._tap_action_governor_contact_age = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._tap_action_governor_stop_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tap_action_governor_brake_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._tap_action_governor_prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._tap_action_governor_brake_source_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space),
            device=self.device,
        )
        self._tap_action_governor_stop_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self._last_tap_action_governor_stop_latched = torch.zeros(self.num_envs, device=self.device)
        self._last_tap_action_governor_brake_active = torch.zeros(self.num_envs, device=self.device)
        self._last_tap_action_governor_projected_disp = torch.zeros(self.num_envs, device=self.device)
        self._last_tap_action_governor_contact_age = torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self._ensure_tap_buffers()
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._tap_contact_seen[env_ids] = False
        self._tap_reaction_seen[env_ids] = False
        self._professor_physical_reaction_seen[env_ids] = False
        self._tap_overshoot_seen[env_ids] = False
        self._tap_success_flag[env_ids] = False
        self._tap_just_succeeded_pending[env_ids] = False
        self._tap_max_disp_along[env_ids] = 0.0
        self._tap_max_disp_xy[env_ids] = 0.0
        self._tap_max_z_delta[env_ids] = 0.0
        self._tap_max_speed[env_ids] = 0.0
        self._tap_max_tip_angle_deg[env_ids] = 0.0
        self._tap_min_contact_vertical_offset[env_ids] = torch.inf
        self._last_tap_stop_after_useful_hold[env_ids] = 0.0
        self._last_tap_stop_after_disp_hold[env_ids] = 0.0
        self._tap_action_governor_contact_age[env_ids] = 0
        self._tap_action_governor_stop_latched[env_ids] = False
        self._tap_action_governor_brake_remaining[env_ids] = 0
        self._tap_action_governor_prev_actions[env_ids] = 0.0
        self._tap_action_governor_brake_source_actions[env_ids] = 0.0
        self._tap_action_governor_stop_step[env_ids] = -1
        self._last_tap_action_governor_stop_latched[env_ids] = 0.0
        self._last_tap_action_governor_brake_active[env_ids] = 0.0
        self._last_tap_action_governor_projected_disp[env_ids] = 0.0
        self._last_tap_action_governor_contact_age[env_ids] = 0.0

    def _link5_collision_aabb_contact_terms(
        self,
        half_xy: torch.Tensor,
        half_z: float,
        half_along: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        corners = torch.tensor(
            [
                (x, y, z)
                for x in (LINK5_COLLISION_BBOX_MIN_M[0], LINK5_COLLISION_BBOX_MAX_M[0])
                for y in (LINK5_COLLISION_BBOX_MIN_M[1], LINK5_COLLISION_BBOX_MAX_M[1])
                for z in (LINK5_COLLISION_BBOX_MIN_M[2], LINK5_COLLISION_BBOX_MAX_M[2])
            ],
            device=self.device,
            dtype=torch.float32,
        )
        n = int(self.num_envs)
        link5_pos_w = self._robot.data.body_pos_w[:, self.link5_idx]
        link5_quat_w = self._robot.data.body_quat_w[:, self.link5_idx]
        local = corners.unsqueeze(0).expand(n, -1, -1)
        quat = link5_quat_w.unsqueeze(1).expand(n, corners.shape[0], -1)
        offset_w = quat_apply(quat.reshape(-1, 4), local.reshape(-1, 3)).reshape(n, corners.shape[0], 3)
        corners_w = link5_pos_w.unsqueeze(1) + offset_w

        push_dir = self._push_dir_xy
        lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=-1)
        half_lateral = torch.sum(torch.abs(lateral_dir) * half_xy.unsqueeze(0), dim=-1)
        rel_xy = corners_w[:, :, 0:2] - self._sponge_pos_w[:, None, 0:2]
        along = torch.sum(rel_xy * push_dir[:, None, :], dim=-1)
        lateral_coord = torch.sum(rel_xy * lateral_dir[:, None, :], dim=-1)
        z_coord = corners_w[:, :, 2]

        along_min = along.min(dim=-1).values
        along_max = along.max(dim=-1).values
        lateral_min = lateral_coord.min(dim=-1).values
        lateral_max = lateral_coord.max(dim=-1).values
        z_min = z_coord.min(dim=-1).values
        z_max = z_coord.max(dim=-1).values

        band = float(self.cfg.tap_contact_face_band_m)
        lateral_margin = float(self.cfg.tap_contact_lateral_margin_m)
        vertical_margin = float(self.cfg.tap_contact_vertical_margin_m)
        face_lower = -half_along - band
        face_upper = -half_along + band
        lateral_lower = -half_lateral - lateral_margin
        lateral_upper = half_lateral + lateral_margin
        z_lower = self._sponge_pos_w[:, 2] - half_z - vertical_margin
        z_upper = self._sponge_pos_w[:, 2] + half_z + vertical_margin

        face_overlap = (along_max >= face_lower) & (along_min <= face_upper)
        lateral_overlap = (lateral_max >= lateral_lower) & (lateral_min <= lateral_upper)
        vertical_overlap = (z_max >= z_lower) & (z_min <= z_upper)
        contact_proxy = face_overlap & lateral_overlap & vertical_overlap

        zeros = torch.zeros_like(along_max)
        face_gap = torch.where(
            along_max < face_lower,
            along_max + half_along,
            torch.where(along_min > face_upper, along_min + half_along, zeros),
        )
        lateral = torch.where(
            lateral_max < lateral_lower,
            lateral_lower - lateral_max,
            torch.where(lateral_min > lateral_upper, lateral_min - lateral_upper, zeros),
        )
        vertical_offset = torch.where(
            z_max < z_lower,
            z_lower - z_max,
            torch.where(z_min > z_upper, z_min - z_upper, zeros),
        )
        contact_proximity = 1.0 - torch.clamp(torch.abs(face_gap) / max(band, 1.0e-6), min=0.0, max=1.0)
        return face_gap, lateral, vertical_offset, contact_proxy, contact_proximity

    def _tap_terms(self) -> dict[str, torch.Tensor]:
        self._ensure_tap_buffers()
        terms = self._push_terms()

        half_xy = torch.tensor(
            (0.5 * float(self.cfg.cube_size_x_m), 0.5 * float(self.cfg.cube_size_y_m)),
            device=self.device,
            dtype=torch.float32,
        )
        half_z = 0.5 * float(self.cfg.cube_size_z_m)
        half_along = torch.sum(torch.abs(self._push_dir_xy) * half_xy.unsqueeze(0), dim=-1)
        contact_proxy_mode = str(getattr(self.cfg, "tap_contact_proxy_mode", "tcp_point"))
        if contact_proxy_mode == "tcp_point":
            rel_xy = self._tcp_pos_w[:, 0:2] - self._sponge_pos_w[:, 0:2]
            along = torch.sum(rel_xy * self._push_dir_xy, dim=-1)
            lateral = torch.norm(rel_xy - along.unsqueeze(-1) * self._push_dir_xy, p=2, dim=-1)
            face_gap = along + half_along
            vertical_offset = torch.abs(self._tcp_pos_w[:, 2] - self._sponge_pos_w[:, 2])
            contact_proxy = (
                (face_gap >= -float(self.cfg.tap_contact_face_band_m))
                & (face_gap <= float(self.cfg.tap_contact_face_band_m))
                & (lateral <= half_along + float(self.cfg.tap_contact_lateral_margin_m))
                & (vertical_offset <= half_z + float(self.cfg.tap_contact_vertical_margin_m))
            )
            contact_proximity = 1.0 - torch.clamp(
                torch.abs(face_gap) / max(float(self.cfg.tap_contact_face_band_m), 1.0e-6),
                min=0.0,
                max=1.0,
            )
        elif contact_proxy_mode == "link5_collision_aabb":
            face_gap, lateral, vertical_offset, contact_proxy, contact_proximity = (
                self._link5_collision_aabb_contact_terms(half_xy, half_z, half_along)
            )
        else:
            raise ValueError(f"unsupported tap_contact_proxy_mode={contact_proxy_mode!r}")

        z_delta = torch.abs(self._sponge_pos_w[:, 2] - self._cube_start_w[:, 2])
        disp_reaction = terms["disp_along"] >= float(self.cfg.tap_reaction_disp_m)
        z_reaction = z_delta >= float(self.cfg.tap_reaction_z_delta_m)
        speed_reaction = terms["speed"] >= float(self.cfg.tap_reaction_speed_mps)
        tip_reaction = contact_proxy & (terms["tip_angle_deg"] >= float(self.cfg.tap_reaction_tip_angle_deg))
        reaction_signal_now = disp_reaction | z_reaction | speed_reaction | tip_reaction
        contact_context = contact_proxy | self._tap_contact_seen
        reaction_now = contact_context & reaction_signal_now
        overshoot_now = terms["disp_xy"] >= float(self.cfg.tap_overshoot_disp_m)
        useful_min_disp_m = max(float(getattr(self.cfg, "tap_useful_min_disp_m", 0.001)), 0.0)
        target_disp_m = max(float(self.cfg.cube_push_target_disp_m), 1.0e-6)
        target_tol_m = max(float(self.cfg.tap_target_disp_tolerance_m), 1.0e-6)
        disp_along_pos = torch.clamp(terms["disp_along"], min=0.0)
        useful_min_disp_now = terms["disp_xy"] >= useful_min_disp_m
        target_disp_error = torch.abs(disp_along_pos - target_disp_m)
        target_total_ok = terms["disp_xy"] <= target_disp_m + target_tol_m
        target_band_now = (target_disp_error <= target_tol_m) & target_total_ok & useful_min_disp_now
        target_band_reward_m = torch.clamp(target_tol_m - target_disp_error, min=0.0, max=target_tol_m)
        target_band_reward_m = torch.where(
            target_total_ok & useful_min_disp_now,
            target_band_reward_m,
            torch.zeros_like(target_band_reward_m),
        )
        target_excess_m = torch.clamp(terms["disp_xy"] - (target_disp_m + target_tol_m), min=0.0)
        target_excess_ratio = target_excess_m / target_tol_m
        professor_physical_reaction_signal = (
            (torch.clamp(terms["disp_along"], min=0.0) >= float(self.cfg.professor_physical_reaction_disp_m))
            | (terms["disp_xy"] >= float(self.cfg.professor_physical_reaction_disp_m))
            | (z_delta >= float(self.cfg.professor_physical_reaction_z_delta_m))
            | (terms["speed"] >= float(self.cfg.professor_physical_reaction_speed_mps))
        )
        professor_physical_reaction_now = professor_physical_reaction_signal & ~overshoot_now
        success_now = (contact_proxy | self._tap_contact_seen) & reaction_now & target_band_now & ~overshoot_now

        terms.update(
            {
                "tap_contact_proxy": contact_proxy,
                "tap_contact_proximity": contact_proximity,
                "tap_contact_face_gap_m": face_gap,
                "tap_contact_lateral_m": lateral,
                "tap_contact_vertical_offset_m": vertical_offset,
                "tap_z_delta_m": z_delta,
                "tap_reaction_signal_now": reaction_signal_now,
                "tap_reaction_contact_context": contact_context,
                "tap_reaction_now": reaction_now,
                "tap_overshoot_now": overshoot_now,
                "tap_useful_min_disp_now": useful_min_disp_now,
                "tap_target_disp_error_m": target_disp_error,
                "tap_target_band_now": target_band_now,
                "tap_target_band_reward_m": target_band_reward_m,
                "tap_target_excess_m": target_excess_m,
                "tap_target_excess_ratio": target_excess_ratio,
                "professor_physical_reaction_signal": professor_physical_reaction_signal,
                "professor_physical_reaction_now": professor_physical_reaction_now,
                "tap_success_now": success_now,
            }
        )
        return terms

    def _update_tap_buffers(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        self._tap_contact_seen |= terms["tap_contact_proxy"]
        self._tap_reaction_seen |= terms["tap_reaction_now"]
        self._professor_physical_reaction_seen |= terms["professor_physical_reaction_now"]
        self._tap_overshoot_seen |= terms["tap_overshoot_now"]
        self._tap_min_contact_vertical_offset[:] = torch.where(
            terms["tap_contact_proxy"],
            torch.minimum(self._tap_min_contact_vertical_offset, terms["tap_contact_vertical_offset_m"]),
            self._tap_min_contact_vertical_offset,
        )
        self._tap_max_disp_along[:] = torch.maximum(self._tap_max_disp_along, torch.clamp(terms["disp_along"], min=0.0))
        self._tap_max_disp_xy[:] = torch.maximum(self._tap_max_disp_xy, terms["disp_xy"])
        self._tap_max_z_delta[:] = torch.maximum(self._tap_max_z_delta, terms["tap_z_delta_m"])
        self._tap_max_speed[:] = torch.maximum(self._tap_max_speed, terms["speed"])
        self._tap_max_tip_angle_deg[:] = torch.maximum(self._tap_max_tip_angle_deg, terms["tip_angle_deg"])
        useful_min_disp_seen = self._tap_max_disp_xy >= float(getattr(self.cfg, "tap_useful_min_disp_m", 0.001))
        success_now = (
            self._tap_contact_seen
            & self._tap_reaction_seen
            & useful_min_disp_seen
            & terms["tap_target_band_now"]
            & ~self._tap_overshoot_seen
        )
        just_succeeded = success_now & ~self._tap_success_flag
        self._tap_success_flag |= success_now
        self._tap_just_succeeded_pending |= just_succeeded
        return just_succeeded

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        terms = self._tap_terms()
        self._update_tap_buffers(terms)
        just_succeeded = self._tap_just_succeeded_pending.clone()

        target_disp_m = max(float(self.cfg.cube_push_target_disp_m), 1.0e-6)
        useful_min_disp_m = max(float(getattr(self.cfg, "tap_useful_min_disp_m", 0.001)), 0.0)
        useful_min_disp_seen = self._tap_max_disp_xy >= useful_min_disp_m
        useful_now = (
            (terms["tap_contact_proxy"] | self._tap_contact_seen)
            & terms["tap_reaction_now"]
            & terms["tap_useful_min_disp_now"]
            & ~terms["tap_overshoot_now"]
        )
        contact_reaction_seen = self._tap_contact_seen & self._tap_reaction_seen
        useful_seen = contact_reaction_seen & useful_min_disp_seen & ~self._tap_overshoot_seen
        no_overshoot_seen = ~self._tap_overshoot_seen
        min_contact_vertical_finite = torch.isfinite(self._tap_min_contact_vertical_offset)
        min_contact_vertical = torch.where(
            min_contact_vertical_finite,
            self._tap_min_contact_vertical_offset,
            torch.zeros_like(self._tap_min_contact_vertical_offset),
        )
        tap_max_disp_along_ge_1mm = self._tap_max_disp_along >= 0.001
        tap_max_disp_xy_ge_1mm = self._tap_max_disp_xy >= 0.001
        tap_max_disp_along_ge_3mm = self._tap_max_disp_along >= 0.003
        tap_max_disp_xy_ge_3mm = self._tap_max_disp_xy >= 0.003
        prev_target_error = torch.abs(torch.clamp(self._prev_disp_along, min=0.0) - target_disp_m)
        progress = torch.clamp(prev_target_error - terms["tap_target_disp_error_m"], min=-0.005, max=0.005)
        self._prev_disp_along[:] = terms["disp_along"].detach()
        action_penalty = -torch.sum(self.actions ** 2, dim=-1) * self.cfg.action_penalty_scale
        bc_imitation_penalty = -float(self.cfg.bc_teacher_imitation_reward_scale) * self._last_bc_teacher_imitation_mse
        rewards = (
            self.cfg.push_progress_reward_scale * progress
            + self.cfg.tap_contact_reward_scale * terms["tap_contact_proxy"].float()
            + self.cfg.tap_contact_proximity_reward_scale * terms["tap_contact_proximity"]
            + self.cfg.tap_reaction_reward_scale * just_succeeded.float()
            + self.cfg.tap_transient_disp_reward_scale * terms["tap_target_band_reward_m"]
            - self.cfg.tap_overshoot_penalty_scale * terms["tap_target_excess_ratio"]
            - self.cfg.tap_tip_penalty_scale * terms["tip_angle_deg"]
            + action_penalty
            + bc_imitation_penalty
        )

        self.extras["log"] = {
            "cube_tap_objective_final_relocation_required": torch.zeros((), device=self.device),
            "cube_tap_object_size_m": torch.tensor(float(self.cfg.cube_size_x_m), device=self.device),
            "cube_tap_object_mass_kg": torch.tensor(CUBE10CM_MASS_KG, device=self.device),
            "cube_tap_disp_along_m": terms["disp_along"].mean().detach(),
            "cube_tap_disp_xy_m": terms["disp_xy"].mean().detach(),
            "cube_tap_speed_mps": terms["speed"].mean().detach(),
            "cube_tap_tip_angle_deg": terms["tip_angle_deg"].mean().detach(),
            "cube_tap_contact_proxy_rate": terms["tap_contact_proxy"].float().mean().detach(),
            "cube_tap_contact_seen_rate": self._tap_contact_seen.float().mean().detach(),
            "cube_tap_reaction_signal_now_rate": terms["tap_reaction_signal_now"].float().mean().detach(),
            "cube_tap_reaction_contact_context_rate": terms["tap_reaction_contact_context"].float().mean().detach(),
            "cube_tap_reaction_now_rate": terms["tap_reaction_now"].float().mean().detach(),
            "cube_tap_reaction_seen_rate": self._tap_reaction_seen.float().mean().detach(),
            "cube_tap_useful_now_rate": useful_now.float().mean().detach(),
            "cube_tap_useful_seen_rate": useful_seen.float().mean().detach(),
            "cube_tap_useful_min_disp_m": torch.tensor(useful_min_disp_m, device=self.device),
            "cube_tap_useful_min_disp_seen_rate": useful_min_disp_seen.float().mean().detach(),
            "cube_tap_contact_reaction_seen_rate": contact_reaction_seen.float().mean().detach(),
            "cube_tap_no_overshoot_seen_rate": no_overshoot_seen.float().mean().detach(),
            "cube_tap_target_disp_m": torch.tensor(float(self.cfg.cube_push_target_disp_m), device=self.device),
            "cube_tap_target_disp_tolerance_m": torch.tensor(
                float(self.cfg.tap_target_disp_tolerance_m), device=self.device
            ),
            "cube_tap_target_disp_error_m": terms["tap_target_disp_error_m"].mean().detach(),
            "cube_tap_target_band_rate": terms["tap_target_band_now"].float().mean().detach(),
            "cube_tap_target_excess_m": terms["tap_target_excess_m"].mean().detach(),
            "cube_tap_target_excess_ratio": terms["tap_target_excess_ratio"].mean().detach(),
            "cube_tap_target_band_reward_m": terms["tap_target_band_reward_m"].mean().detach(),
            "cube_tap_professor_physical_reaction_signal_now_rate": terms[
                "professor_physical_reaction_signal"
            ].float().mean().detach(),
            "cube_tap_professor_physical_reaction_now_rate": terms[
                "professor_physical_reaction_now"
            ].float().mean().detach(),
            "cube_tap_professor_physical_reaction_seen_rate": self._professor_physical_reaction_seen.float()
            .mean()
            .detach(),
            "cube_tap_professor_physical_reaction_disp_threshold_m": torch.tensor(
                float(self.cfg.professor_physical_reaction_disp_m), device=self.device
            ),
            "cube_tap_professor_physical_reaction_speed_threshold_mps": torch.tensor(
                float(self.cfg.professor_physical_reaction_speed_mps), device=self.device
            ),
            "cube_tap_overshoot_now_rate": terms["tap_overshoot_now"].float().mean().detach(),
            "cube_tap_overshoot_seen_rate": self._tap_overshoot_seen.float().mean().detach(),
            "cube_tap_just_succeeded_rate": just_succeeded.float().mean().detach(),
            "cube_tap_just_succeeded_count": just_succeeded.float().sum().detach(),
            "cube_tap_success_rate": self._tap_success_flag.float().mean().detach(),
            "cube_tap_max_disp_along_m": self._tap_max_disp_along.mean().detach(),
            "cube_tap_max_disp_xy_m": self._tap_max_disp_xy.mean().detach(),
            "cube_tap_max_disp_along_ge_1mm_rate": tap_max_disp_along_ge_1mm.float().mean().detach(),
            "cube_tap_max_disp_xy_ge_1mm_rate": tap_max_disp_xy_ge_1mm.float().mean().detach(),
            "cube_tap_max_disp_along_ge_3mm_rate": tap_max_disp_along_ge_3mm.float().mean().detach(),
            "cube_tap_max_disp_xy_ge_3mm_rate": tap_max_disp_xy_ge_3mm.float().mean().detach(),
            "cube_tap_max_z_delta_m": self._tap_max_z_delta.mean().detach(),
            "cube_tap_max_speed_mps": self._tap_max_speed.mean().detach(),
            "cube_tap_contact_face_gap_m": terms["tap_contact_face_gap_m"].mean().detach(),
            "cube_tap_contact_lateral_m": terms["tap_contact_lateral_m"].mean().detach(),
            "cube_tap_contact_vertical_offset_m": terms["tap_contact_vertical_offset_m"].mean().detach(),
            "cube_tap_min_contact_vertical_offset_m": min_contact_vertical.mean().detach(),
            "cube_tap_min_contact_vertical_finite_rate": min_contact_vertical_finite.float().mean().detach(),
            "cube_tap_stop_after_useful_hold_rate": self._last_tap_stop_after_useful_hold.mean().detach(),
            "cube_tap_stop_after_disp_hold_rate": self._last_tap_stop_after_disp_hold.mean().detach(),
            "cube_tap_stop_after_disp_m": torch.tensor(float(self.cfg.tap_stop_after_disp_m), device=self.device),
            "cube_tap_action_governor_enabled": torch.tensor(
                float(str(getattr(self.cfg, "tap_action_governor_mode", "off")) != "off"),
                device=self.device,
            ),
            "cube_tap_action_governor_stop_latched_rate": self._last_tap_action_governor_stop_latched.mean().detach(),
            "cube_tap_action_governor_brake_active_rate": self._last_tap_action_governor_brake_active.mean().detach(),
            "cube_tap_action_governor_projected_disp_m": self._last_tap_action_governor_projected_disp.mean().detach(),
            "cube_tap_action_governor_contact_age_steps": self._last_tap_action_governor_contact_age.mean().detach(),
            "cube_tap_contact_slowdown_use_proxy": torch.tensor(
                float(bool(self.cfg.tap_contact_slowdown_use_proxy)), device=self.device
            ),
            "cube_tap_push_primitive_enabled": torch.tensor(
                float(str(getattr(self.cfg, "rl_action_mode", "joint_delta")) == "tap_push_primitive"),
                device=self.device,
            ),
            "cube_tap_push_primitive_stop_latched_rate": self._last_tap_push_primitive_stop_latched.mean().detach(),
            "cube_tap_push_primitive_target_delta_abs_mean": self._last_tap_push_primitive_target_delta_abs_mean.mean().detach(),
            "cube_tap_push_primitive_target_delta_abs_max": self._last_tap_push_primitive_target_delta_abs_max.mean().detach(),
            "cube_push_tcp_cube_dist_m": terms["tcp_cube_dist"].mean().detach(),
            "cube_push_joint_delta_abs_mean": self._last_joint_delta_abs_mean.mean().detach(),
            "cube_push_joint_delta_abs_max": self._last_joint_delta_abs_max.mean().detach(),
            "cube_push_joint_delta_cap_rate": self._last_joint_delta_cap_rate.mean().detach(),
            "cube_push_action_abs_mean": self._last_action_abs_mean.mean().detach(),
            "cube_push_action_abs_max": self._last_action_abs_max.mean().detach(),
            "cube_push_target_lead_abs_mean": self._last_target_lead_abs_mean.mean().detach(),
            "cube_push_target_lead_abs_max": self._last_target_lead_abs_max.mean().detach(),
            "cube_push_target_lead_limit_rate": self._last_target_lead_limit_rate.mean().detach(),
            "cube_push_contact_slowdown_mean": self._last_contact_slowdown.mean().detach(),
            "cube_push_ik_endpoint_reset_rate": self._ik_reset_ok.float().mean().detach(),
            "cube_push_ik_reset_err_mm": self._ik_reset_err_mm.mean().detach(),
            "cube_push_teacher_blend_mean": self._last_teacher_blend.mean().detach(),
            "cube_push_bc_teacher_blend_mean": self._last_bc_teacher_blend.mean().detach(),
            "cube_push_bc_teacher_imitation_mse": self._last_bc_teacher_imitation_mse.mean().detach(),
            "cube_push_bc_teacher_action_abs_mean": self._last_bc_teacher_action_abs_mean.mean().detach(),
            "cube_tap_bc_teacher_blend_mean": self._last_bc_teacher_blend.mean().detach(),
            "cube_tap_bc_teacher_imitation_mse": self._last_bc_teacher_imitation_mse.mean().detach(),
            "cube_tap_bc_teacher_action_abs_mean": self._last_bc_teacher_action_abs_mean.mean().detach(),
            "cube_push_d256_reset_active_rate": self._last_d256_reset_active.mean().detach(),
            "cube_push_d256_reset_episode_index_mean": self._last_d256_reset_episode_index.mean().detach(),
            "cube_tap_d256_reset_active_rate": self._last_d256_reset_active.mean().detach(),
            "cube_tap_d256_reset_episode_index_mean": self._last_d256_reset_episode_index.mean().detach(),
            "cube_push_candidate6_diffik_active_rate": self._last_candidate6_diffik_active.mean().detach(),
            "cube_push_candidate6_diffik_numeric_ok_rate": self._last_candidate6_diffik_numeric_ok.mean().detach(),
            "cube_push_candidate6_diffik_raw_delta_abs_max": self._last_candidate6_diffik_raw_delta_abs_max.mean().detach(),
            "cube_push_candidate6_diffik_clipped_delta_abs_max": self._last_candidate6_diffik_clipped_delta_abs_max.mean().detach(),
            "cube_push_candidate6_diffik_step_clip_rate": self._last_candidate6_diffik_step_clip_rate.mean().detach(),
            "cube_push_candidate6_diffik_residual_abs_mean": self._last_candidate6_diffik_residual_abs_mean.mean().detach(),
            "cube_push_candidate6_diffik_residual_abs_max": self._last_candidate6_diffik_residual_abs_max.mean().detach(),
            "cube_push_candidate6_diffik_hold_success_rate": self._last_candidate6_diffik_hold_success_rate.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_abs_mean": self._last_candidate8_diffik_target_residual_abs_mean.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_abs_max": self._last_candidate8_diffik_target_residual_abs_max.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_forward_abs": self._last_candidate8_diffik_target_residual_forward_abs.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_lateral_abs": self._last_candidate8_diffik_target_residual_lateral_abs.mean().detach(),
            "cube_push_candidate8_diffik_target_residual_height_abs": self._last_candidate8_diffik_target_residual_height_abs.mean().detach(),
            "cube_push_grasped_marker_rate": self._grasped.float().mean().detach(),
            "bc_teacher_imitation_penalty": bc_imitation_penalty.mean().detach(),
            "action_penalty": action_penalty.mean().detach(),
        }
        self._tap_just_succeeded_pending.zero_()
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        terms = self._tap_terms()
        self._update_tap_buffers(terms)
        terminated = (
            self._tap_overshoot_seen
            if bool(self.cfg.tap_overshoot_terminate)
            else torch.zeros_like(self._tap_overshoot_seen)
        )
        if bool(self.cfg.tap_success_terminate):
            terminated = terminated | self._tap_success_flag
        if bool(getattr(self.cfg, "tap_useful_terminate", False)):
            useful_min_disp_m = max(float(getattr(self.cfg, "tap_useful_min_disp_m", 0.001)), 0.0)
            useful_seen = (
                self._tap_contact_seen
                & self._tap_reaction_seen
                & (self._tap_max_disp_xy >= useful_min_disp_m)
                & ~self._tap_overshoot_seen
            )
            terminated = terminated | useful_seen
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated
