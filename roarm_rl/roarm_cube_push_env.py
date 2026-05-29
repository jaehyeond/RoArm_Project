"""RoArmCubePushEnv: no-attach 3cm cube push task.

This env is separate from the existing Pick/Stack tasks. Its purpose is to
turn the professor's "known endpoint, push/tap the 3cm cube first" request into
a small RL task without using grasp attach or object pose writes during rollout.

Action semantics remain the project standard 6D normalized joint-delta command:
robot_dof_targets += action_scale * action, clipped to joint limits.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from sim_scripts.roarm_kinematics import clip_joints, fk_tcp, ik_dls
from roarm_rl.roarm_stack_env import (
    HOME_RAD,
    TABLE_Z,
    RoArmStackEnv,
    RoArmStackEnvCfg,
)


CUBE_SIZE_M = 0.030
CUBE_CENTER_Z = TABLE_Z + CUBE_SIZE_M / 2.0

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
    scripted_teacher_blend: float = 0.0
    scripted_teacher_horizon_frac: float = 0.55
    scripted_teacher_goal_push_m: float = 0.055
    bc_teacher_checkpoint_path: str = ""
    bc_teacher_blend: float = 0.0
    bc_teacher_imitation_reward_scale: float = 0.0
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
    bc_teacher_phase_timing: str = "episode_scaled"

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


class RoArmCubePushEnv(RoArmStackEnv):
    """No-attach cube pushing task with the Stack env's robot/action scaffold."""

    cfg: RoArmCubePushEnvCfg

    def __init__(self, cfg: RoArmCubePushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ensure_push_buffers()

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
        self._last_contact_slowdown = torch.ones(self.num_envs, device=self.device)
        self._teacher_start_joints = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._teacher_goal_joints = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._teacher_goal_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_teacher_blend = torch.zeros(self.num_envs, device=self.device)
        self._last_bc_teacher_blend = torch.zeros(self.num_envs, device=self.device)
        self._last_bc_teacher_imitation_mse = torch.zeros(self.num_envs, device=self.device)
        self._last_bc_teacher_action_abs_mean = torch.zeros(self.num_envs, device=self.device)
        self._bc_prev_teacher_delta = torch.zeros((self.num_envs, 5), device=self.device)
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

    def _ik_precontact_joints(self, cube_local: torch.Tensor, push_dir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cube_np = cube_local.detach().cpu().numpy().astype(np.float64)
        dir_np = push_dir.detach().cpu().numpy().astype(np.float64)
        home_deg = np.degrees(np.asarray(HOME_RAD, dtype=np.float64))
        q_out = np.tile(home_deg, (cube_np.shape[0], 1))
        ok = np.zeros(cube_np.shape[0], dtype=bool)
        err_mm_out = np.full(cube_np.shape[0], np.inf, dtype=np.float64)
        half_xy = 0.5 * CUBE_SIZE_M
        for idx in range(cube_np.shape[0]):
            tcp_target = cube_np[idx].copy()
            tcp_target[:2] -= dir_np[idx] * (half_xy + float(self.cfg.ik_precontact_clearance_m))
            tcp_target[2] = cube_np[idx, 2] + half_xy + float(self.cfg.ik_tcp_top_margin_m)
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
        half_xy = 0.5 * CUBE_SIZE_M
        for idx in range(cube_np.shape[0]):
            tcp_target = cube_np[idx].copy()
            tcp_target[:2] += dir_np[idx] * (half_xy + float(self.cfg.scripted_teacher_goal_push_m))
            tcp_target[2] = cube_np[idx, 2] + half_xy + float(self.cfg.ik_tcp_top_margin_m)
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
        return {
            "posx": posx,
            "posx_low_bucket": posx_low_bucket,
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
        else:
            raise ValueError(f"unsupported bc_teacher_phase_timing={timing!r}")
        push = torch.clamp(traj["push_steps"], min=1.0)
        raw_alpha = (step_v - traj["approach_steps"] + 1.0) / push
        alpha = torch.where(step_v < traj["approach_steps"], torch.zeros_like(raw_alpha), raw_alpha)
        alpha = torch.where(step_v >= traj["approach_steps"] + push, torch.ones_like(alpha), alpha)
        return torch.clamp(alpha, min=0.0, max=1.0)

    def _bc_teacher_tcp_target(self, alpha: torch.Tensor, traj: dict[str, torch.Tensor]) -> torch.Tensor:
        half = 0.5 * float(CUBE_SIZE_M)
        cube = self._cube_start_w
        push_dir = self._push_dir_xy
        pre = cube.clone()
        through = cube.clone()
        z = cube[:, 2] + half + traj["tcp_top_margin"]
        pre[:, 0:2] = cube[:, 0:2] - push_dir * (half + traj["precontact"]).unsqueeze(-1)
        through[:, 0:2] = cube[:, 0:2] + push_dir * (half + traj["push_through"]).unsqueeze(-1)
        pre[:, 2] = z
        through[:, 2] = z
        return pre + alpha.unsqueeze(-1) * (through - pre)

    def _bc_teacher_feature_tensor(self, alpha: torch.Tensor, tcp_target_w: torch.Tensor) -> torch.Tensor:
        origin = self.scene.env_origins
        cube_local = self._sponge_pos_w - origin
        tcp_local = self._tcp_pos_w - origin
        target_local = tcp_target_w - origin
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
        policy_actions = actions.clone().clamp(-1.0, 1.0)
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
        if getattr(self, "_bc_teacher_ready", False) and (bc_blend_value > 0.0 or bc_reward_scale > 0.0):
            bc_teacher_actions = self._bc_teacher_actions()
            bc_blend = torch.full((self.num_envs,), max(0.0, min(1.0, bc_blend_value)), device=self.device)
            if bc_blend_value > 0.0:
                self.actions = (1.0 - bc_blend.unsqueeze(-1)) * self.actions + bc_blend.unsqueeze(-1) * bc_teacher_actions
            self._last_bc_teacher_blend[:] = bc_blend
            self._last_bc_teacher_imitation_mse[:] = torch.mean((policy_actions - bc_teacher_actions) ** 2, dim=-1)
            self._last_bc_teacher_action_abs_mean[:] = torch.mean(torch.abs(bc_teacher_actions), dim=-1)

        alpha = float(self.cfg.action_smoothing_alpha)
        self._smoothed_actions[:] = (1.0 - alpha) * self._smoothed_actions + alpha * self.actions
        delta = torch.clamp(
            self.cfg.action_scale * self._smoothed_actions,
            -float(self.cfg.max_joint_delta_per_step_rad),
            float(self.cfg.max_joint_delta_per_step_rad),
        )

        self._compute_intermediate_values()
        terms = self._push_terms()
        slowdown = torch.ones(self.num_envs, device=self.device)
        slowdown = torch.where(
            terms["tcp_cube_dist"] < float(self.cfg.contact_slowdown_tcp_dist_m),
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
        reference = str(getattr(self.cfg, "joint_delta_reference", "target"))
        if reference == "target":
            target_base = self.robot_dof_targets
        elif reference == "joint_pos":
            target_base = joint_pos
        else:
            raise ValueError(f"unsupported joint_delta_reference={reference!r}")
        targets = target_base + delta
        lead = float(self.cfg.joint_target_lead_limit_rad)
        targets = torch.maximum(torch.minimum(targets, joint_pos + lead), joint_pos - lead)
        targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        targets[:, self.gripper_joint_idx] = 0.0

        self.robot_dof_targets[:] = targets
        self._last_joint_delta_abs_mean[:] = torch.mean(torch.abs(delta), dim=-1)
        self._last_contact_slowdown[:] = slowdown

    def _apply_action(self):
        # No attach path in this task: robot joint target writes only.
        self._grasped[:] = False
        self._was_grasped[:] = False
        self._robot.set_joint_position_target(self.robot_dof_targets)

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

        sx = sample_uniform(self.cfg.cube_x_min, self.cfg.cube_x_max, (n,), self.device)
        sy = sample_uniform(self.cfg.cube_y_min, self.cfg.cube_y_max, (n,), self.device)
        sz = torch.full((n,), CUBE_CENTER_Z, device=self.device)

        dirs = torch.tensor(
            ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
            device=self.device,
            dtype=torch.float32,
        )
        dir_idx = torch.randint(0, 4, (n,), device=self.device)
        push_dir = dirs[dir_idx]

        cube_local = torch.stack([sx, sy, sz], dim=-1)
        if self.cfg.ik_endpoint_reset:
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

        target_local = cube_local.clone()
        target_local[:, 0:2] = target_local[:, 0:2] + push_dir * self.cfg.cube_push_target_disp_m
        self._target_world[env_ids] = env_origins + target_local

        self._cube_start_w[env_ids] = cube_world
        self._push_dir_xy[env_ids] = push_dir
        self._ik_reset_ok[env_ids] = ik_ok
        self._ik_reset_err_mm[env_ids] = ik_err_mm
        self._prev_disp_along[env_ids] = 0.0
        self._push_success_flag[env_ids] = False
        self._smoothed_actions[env_ids] = 0.0
        self._last_joint_delta_abs_mean[env_ids] = 0.0
        self._last_contact_slowdown[env_ids] = 1.0
        self._last_teacher_blend[env_ids] = 0.0
        self._last_bc_teacher_blend[env_ids] = 0.0
        self._last_bc_teacher_imitation_mse[env_ids] = 0.0
        self._last_bc_teacher_action_abs_mean[env_ids] = 0.0
        self._bc_prev_teacher_delta[env_ids] = 0.0
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
            "cube_push_contact_slowdown_mean": self._last_contact_slowdown.mean().detach(),
            "cube_push_teacher_blend_mean": self._last_teacher_blend.mean().detach(),
            "cube_push_teacher_goal_ok_rate": self._teacher_goal_ok.float().mean().detach(),
            "cube_push_bc_teacher_blend_mean": self._last_bc_teacher_blend.mean().detach(),
            "cube_push_bc_teacher_imitation_mse": self._last_bc_teacher_imitation_mse.mean().detach(),
            "cube_push_bc_teacher_action_abs_mean": self._last_bc_teacher_action_abs_mean.mean().detach(),
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
