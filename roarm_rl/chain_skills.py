"""Hierarchical chain skills — scripted approach/grasp/transport + learned release.

Architecture (HARD RULE #18: structure locked-in 2026-05-13 after P6v1-v14c +
PATH D BC FAIL evidence):

  Skill 0: HOME -> hover above sponge (scripted IK + linear joint interp)
  Skill 1: hover -> grasp pose + gripper close (scripted IK + gripper cmd)
  Skill 2: grasp pose -> place-target hover (scripted IK + linear joint interp)
  Skill 3: place hover -> descend + release (LEARNED, P6v14a/model_499.pt)

Each skill emits env actions in [-1, 1]^6 scaled by env.cfg.action_scale (=0.1 rad/step).

This file has two parts:
  Part 1: TrajectoryPlanner — local-runnable (numpy only). Computes IK waypoints
          + action sequences for skill 0/1/2. Used in dry-run sanity (no Isaac Sim).
  Part 2: ChainRunner — B200-only (isaaclab dep). Runs full chain in Isaac Sim:
          env reset, override sponge spawn, scripted skill 0/1/2 actions, load
          model_499.pt for skill 3, run until success/timeout.

Run:
  Local dry-run (Part 1 only):
    python roarm_rl/chain_skills.py --dry-run
  B200 full chain (Part 2):
    python -m roarm_rl.chain_skills --episode 1 --sponge_xy 0.25 -0.04
  B200 (alpha') Skill 3 basin-of-attraction sweep — skips scripted Skill 0/1/2,
  force-sets env at P6v14a training entry + perturbation, runs Skill 3 inference:
    python -m roarm_rl.chain_skills --basin_sweep
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import fk_tcp, ik_dls, clip_joints, JOINT_LIMITS_DEG  # noqa: E402


# =====================================================================
# Geometry constants (HARD RULE #19/#20)
# =====================================================================
TABLE_Z = -0.012117
SPONGE_HEIGHT_EDGE = 0.047
SPONGE_LEN_LONG = 0.125
SPONGE_WIDTH = 0.022
SPONGE_CENTER_Z = TABLE_Z + SPONGE_HEIGHT_EDGE / 2.0
TCP_GRASP_Z = +0.033
HOVER_OFFSET_Z = 0.030      # 30mm above grasp z (legacy intermediate waypoint)
# (delta) Top-down approach (5/14): start descent from HIGH z = +150mm world,
# 115mm above sponge top (+35mm world). Forces top-down trajectory; gripper
# fingers approach DOWNWARD instead of side-sweep. Mitigates 5/13 Skill 1 fail
# (descent collision: tcp_err 10mm + sponge 22mm wide → side collision pushed sponge).
HIGH_OFFSET_Z = 0.117       # TCP_GRASP_Z + 0.117 = +0.150 world

L1_SP1 = (+0.280, -0.0435, SPONGE_CENTER_Z)
HOME_DEG = np.array([0.0, 0.0, 90.0, 0.0, 0.0, 0.0])
HOME_RAD = np.array([0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0])

# Gripper convention (deg).
# CRITICAL (verified roarm_stack_env L314 + L920): sim env's grasp_gripper_thresh = 0.4 rad = 22.9 deg.
# gripper_q < 22.9 deg => "open" (no grasp latch), >= 22.9 deg => "closed" (grasp candidate).
# P6v14a training entry gripper q = 0.8 rad = 45.84 deg (HARD-coded in pregrasp_joints_rad).
# Chain MUST match P6v14a init: close to 45.84 deg so (1) _grasped triggers in sim, (2) actor
# obs distribution matches training (gripper q ~0.8 rad).
GRIPPER_OPEN_DEG = 0.0       # 5/14 (δ.4): revert -10→0. URDF link5_to_gripper_link lower=0 clamps any negative target to 0 actual (D006). Negative target also inflates max_joint_err by gripper joint, blocking Skill 0/1a/1b break-out (run_skill_closed_loop max_err = max(abs(err)) over all 6 joints). Keep 0.0; future scripted skills MUST NOT use negative gripper open target.
GRIPPER_CLOSE_DEG = 45.84    # match P6v14a training entry (0.8 rad).

# Action scale in env (rad/step). roarm_stack_env.py L199.
ACTION_SCALE_RAD = 0.1


def _quat_rotate_np(q_wxyz: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate Nx3 points by a wxyz quaternion."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    w, x, y, z = q
    qvec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qvec, pts)
    uuv = np.cross(qvec, uv)
    return pts + 2.0 * (w * uv + uuv)


def _candidate_gripper_mesh_paths() -> list[Path]:
    paths: list[Path] = []
    root = os.environ.get("ROARM_B200_ROOT")
    if root:
        paths.append(Path(root) / "assets/roarm_m3/urdf/meshes/gripper_link.stl")
    paths.extend([
        REPO / "local_assets/roarm_m3/urdf/meshes/gripper_link.stl",
        REPO / "assets/roarm_m3/urdf/meshes/gripper_link.stl",
    ])
    return paths


def _load_binary_stl_vertices_m(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL too small: {path}")
    n_tri = struct.unpack("<I", data[80:84])[0]
    expected = 84 + 50 * n_tri
    if expected > len(data):
        raise ValueError(f"STL size mismatch: {path} expected {expected}, got {len(data)}")
    verts = []
    off = 84
    for _ in range(n_tri):
        vals = struct.unpack("<12fH", data[off:off + 50])
        verts.extend((vals[3:6], vals[6:9], vals[9:12]))
        off += 50
    return np.asarray(verts, dtype=np.float64) * 0.001


def _load_gripper_mesh_vertices_m() -> tuple[Optional[np.ndarray], Optional[Path]]:
    for path in _candidate_gripper_mesh_paths():
        if path.exists():
            return _load_binary_stl_vertices_m(path), path
    return None, None


# =====================================================================
# Part 1: TrajectoryPlanner (local-runnable)
# =====================================================================
class TrajectoryPlanner:
    """Computes IK joint waypoints + env actions for skill 0/1/2 (scripted)."""

    def __init__(self,
                 sponge_xyz: tuple,
                 place_xyz: tuple = L1_SP1,
                 hover_offset_z: float = HOVER_OFFSET_Z,
                 high_offset_z: float = HIGH_OFFSET_Z,
                 transport_offset_z: float = 0.030):
        # transport_offset_z=0.030: chain end TCP z = +33+30 = +63mm matches P6v14a
        # training entry TCP z=+61.6mm (verified via FK on pregrasp_joints_rad).
        # high_offset_z=0.117: TCP at +150mm world for top-down approach (delta 5/14).
        self.sponge_xyz = np.asarray(sponge_xyz)
        self.place_xyz = np.asarray(place_xyz)
        self.hover_offset_z = hover_offset_z
        self.high_offset_z = high_offset_z
        self.transport_offset_z = transport_offset_z

        # IK waypoints (all in deg, joint 0-5; gripper kept at HOME=0.0).
        # Warm-start chain HOME → high → hover → grasp keeps wrist_p posture consistent
        # across waypoints (verified locally — pregrasp_q wrist_p +12.7°, IK natural
        # consistency within ~5° across all sponge_xy in grid).
        self.q_home_deg = HOME_DEG.copy()
        self.q_high_deg = self._ik_with_gripper(
            (self.sponge_xyz[0], self.sponge_xyz[1], TCP_GRASP_Z + high_offset_z),
            self.q_home_deg,
        )
        self.q_hover_deg = self._ik_with_gripper(
            (self.sponge_xyz[0], self.sponge_xyz[1], TCP_GRASP_Z + hover_offset_z),
            self.q_high_deg,
        )
        # 5/14 (δ.4) Skill 1b multi-stage descent: hover(+63) → +50 → +40 → +33.
        # Each intermediate IK warm-started from previous to preserve wrist_p posture.
        # Purpose = DIAGNOSTIC, not tuning: locate exact z where descent stalls
        # (gripper-state independent per D006 — δ.1 and δ.2 both stuck at +51.9mm).
        self.q_1b1_deg = self._ik_with_gripper(
            (self.sponge_xyz[0], self.sponge_xyz[1], TCP_GRASP_Z + 0.017),  # +50mm
            self.q_hover_deg,
        )
        self.q_1b2_deg = self._ik_with_gripper(
            (self.sponge_xyz[0], self.sponge_xyz[1], TCP_GRASP_Z + 0.007),  # +40mm
            self.q_1b1_deg,
        )
        self.q_grasp_deg = self._ik_with_gripper(
            (self.sponge_xyz[0], self.sponge_xyz[1], TCP_GRASP_Z),          # +33mm
            self.q_1b2_deg,
        )
        self.q_transport_deg = self._ik_with_gripper(
            (self.place_xyz[0], self.place_xyz[1], TCP_GRASP_Z + transport_offset_z),
            self.q_grasp_deg,
        )

        # Check IK convergence.
        self._verify_ik()

    def _ik_with_gripper(self, target_xyz, q0_deg, gripper_deg: float = 0.0):
        q, conv, err_mm, n_iter = ik_dls(target_xyz, q0_deg, max_iter=200, tol_mm=1.0)
        if not conv:
            print(f"  [WARN] IK not converged for target={target_xyz}, err={err_mm:.2f}mm, iter={n_iter}")
        q_out = q.copy()
        q_out[5] = gripper_deg
        return clip_joints(q_out)

    def _verify_ik(self):
        # Cross-verify wrist_p consistency: all 3 descent waypoints (high/hover/grasp)
        # should have similar wrist_p (joint 3, deg) to ensure vertical descent profile.
        # If wrist_p varies >5° across high/hover/grasp, IK chose different postures and
        # joint-space interpolation will tilt the gripper during descent.
        wrist_p_vals = []
        for name, q in [
            ("high",            self.q_high_deg),
            ("hover",           self.q_hover_deg),
            ("1b1_z50",         self.q_1b1_deg),
            ("1b2_z40",         self.q_1b2_deg),
            ("grasp",           self.q_grasp_deg),
            ("transport_hover", self.q_transport_deg),
        ]:
            tcp = fk_tcp(q)
            print(f"  Waypoint {name:20s}: q_deg={[f'{x:+6.1f}' for x in q[:5]]}  "
                  f"TCP=({tcp[0]*1000:+6.1f},{tcp[1]*1000:+6.1f},{tcp[2]*1000:+6.1f})mm")
            if name in ("high", "hover", "1b1_z50", "1b2_z40", "grasp"):
                wrist_p_vals.append(q[3])
        wp_range = max(wrist_p_vals) - min(wrist_p_vals)
        wp_ok = wp_range < 5.0
        print(f"  wrist_p range across high/hover/1b1/1b2/grasp: {wp_range:.2f}deg  ({'OK' if wp_ok else 'WARN: >5deg, descent may tilt'})")

    def deg_to_rad(self, q_deg):
        return np.radians(q_deg)

    # =================================================================
    # Skill targets exposed for closed-loop control (real sim feedback)
    # (delta 5/14) Top-down chain:
    #   Skill 0 = HOME -> q_high (TCP +150mm world, gripper OPEN)
    #   Skill 1a = q_high -> q_hover (TCP +63mm, gripper OPEN) — descent stage 1
    #   Skill 1b = q_hover -> q_grasp (TCP +33mm, gripper OPEN) — descent stage 2, TIGHT tol
    #   Skill 1c = close gripper (q_grasp + gripper q=45.84deg)
    #   Skill 2  = q_grasp -> q_transport (TCP at place +63mm)
    #   Skill 3  = P6v14a inference (release)
    #   Skill 4  = q_transport -> q_high(place) — retreat to avoid sponge knock-away
    # =================================================================
    def target_q_skill0_high(self):
        """HOME -> high (TCP +150mm above grasp). gripper OPEN (q=0)."""
        t = self.deg_to_rad(self.q_high_deg.copy())
        t[5] = math.radians(GRIPPER_OPEN_DEG)
        return t

    def target_q_skill1a_to_hover(self):
        """high -> hover (TCP from +150 to +63mm). gripper OPEN."""
        t = self.deg_to_rad(self.q_hover_deg.copy())
        t[5] = math.radians(GRIPPER_OPEN_DEG)
        return t

    def target_q_skill1b_to_grasp(self):
        """hover -> grasp (TCP from +63 to +33mm). gripper OPEN. TIGHT tol required.
        LEGACY single-shot; 5/14 (δ.4) chain uses 1b1/1b2/1b3 multi-stage."""
        t = self.deg_to_rad(self.q_grasp_deg.copy())
        t[5] = math.radians(GRIPPER_OPEN_DEG)
        return t

    def target_q_skill1b1_z50(self):
        """δ.4 stage 1: hover(+63) -> +50mm. gripper OPEN. Diagnostic stop."""
        t = self.deg_to_rad(self.q_1b1_deg.copy())
        t[5] = math.radians(GRIPPER_OPEN_DEG)
        return t

    def target_q_skill1b2_z40(self):
        """δ.4 stage 2: +50 -> +40mm. gripper OPEN. Diagnostic stop."""
        t = self.deg_to_rad(self.q_1b2_deg.copy())
        t[5] = math.radians(GRIPPER_OPEN_DEG)
        return t

    def target_q_skill1b3_z33(self):
        """δ.4 stage 3: +40 -> +33mm (q_grasp). gripper OPEN. Final descent stop."""
        t = self.deg_to_rad(self.q_grasp_deg.copy())
        t[5] = math.radians(GRIPPER_OPEN_DEG)
        return t

    def target_q_skill1c_close(self):
        """Close gripper at grasp pose (gripper q OPEN -> 45.84deg)."""
        t = self.deg_to_rad(self.q_grasp_deg.copy())
        t[5] = math.radians(GRIPPER_CLOSE_DEG)
        return t

    def target_q_skill2(self):
        """grasp -> transport_hover (TCP at place +63mm). gripper CLOSED."""
        t = self.deg_to_rad(self.q_transport_deg.copy())
        t[5] = math.radians(GRIPPER_CLOSE_DEG)
        return t

    # ---- Legacy aliases (5/13 chain still uses these; keep for backward compat) ----
    def target_q_skill0(self):
        """LEGACY: same as skill0_high for new chain. Kept for backward compat."""
        return self.target_q_skill0_high()

    def target_q_skill1a_descend(self):
        """LEGACY: same as skill1b_to_grasp (direct descent from current pose to grasp)."""
        return self.target_q_skill1b_to_grasp()

    def target_q_skill1b_close(self):
        """LEGACY: same as skill1c_close."""
        return self.target_q_skill1c_close()

    @staticmethod
    def compute_action(current_q_rad, target_q_rad):
        """Compute env action[6] in [-1, 1] to move current_q toward target_q
        with the env's action_scale = 0.1 rad/step.

        Returns: (action[6] np.float32, max_abs_err_rad)
        """
        err = np.asarray(target_q_rad) - np.asarray(current_q_rad)
        action = np.clip(err / ACTION_SCALE_RAD, -1.0, 1.0).astype(np.float32)
        return action, float(np.max(np.abs(err)))

    # =================================================================
    # Open-loop generators (kept for local dry-run summary only)
    # =================================================================
    def actions_skill0(self, current_q_rad, max_steps: int = 40):
        target = self.target_q_skill0()
        return self._move_to_open_loop(current_q_rad, target, max_steps)

    def actions_skill1(self, current_q_rad, max_steps_descend: int = 20,
                       max_steps_close: int = 15):
        target_descend = self.target_q_skill1a_descend()
        descend_actions = list(self._move_to_open_loop(current_q_rad, target_descend, max_steps_descend))
        q_after_descend = descend_actions[-1][1] if descend_actions else current_q_rad
        target_close = self.target_q_skill1b_close()
        close_actions = list(self._move_to_open_loop(q_after_descend, target_close, max_steps_close))
        return descend_actions + close_actions

    def actions_skill2(self, current_q_rad, max_steps: int = 40):
        target = self.target_q_skill2()
        return self._move_to_open_loop(current_q_rad, target, max_steps)

    def _move_to_open_loop(self, current_q_rad, target_q_rad, max_steps: int,
                           tol_rad: float = 0.005):
        q = np.asarray(current_q_rad, dtype=np.float64).copy()
        target = np.asarray(target_q_rad, dtype=np.float64)
        for _ in range(max_steps):
            err = target - q
            if np.max(np.abs(err)) < tol_rad:
                break
            action = np.clip(err / ACTION_SCALE_RAD, -1.0, 1.0)
            q = q + ACTION_SCALE_RAD * action
            yield action.astype(np.float32), q.copy()

    # =================================================================
    # Dry-run summary (local, no Isaac Sim)
    # =================================================================
    def dry_run_summary(self):
        print(f"\nSponge:        ({self.sponge_xyz[0]*1000:+6.1f},{self.sponge_xyz[1]*1000:+6.1f},{self.sponge_xyz[2]*1000:+6.1f})mm")
        print(f"Place target:  ({self.place_xyz[0]*1000:+6.1f},{self.place_xyz[1]*1000:+6.1f},{self.place_xyz[2]*1000:+6.1f})mm")

        q = HOME_RAD.copy()
        a0 = list(self.actions_skill0(q));     q = a0[-1][1] if a0 else q
        a1 = list(self.actions_skill1(q));     q = a1[-1][1] if a1 else q
        a2 = list(self.actions_skill2(q));     q = a2[-1][1] if a2 else q

        n_total = len(a0) + len(a1) + len(a2)
        # P6v14a max_episode_length = 200 step (episode_length_s=2.0, dt=1/200, decimation=2)
        budget_for_skill3 = 200 - n_total
        print(f"\nSkill 0 (HOME->hover):      {len(a0):3d} steps")
        print(f"Skill 1 (hover->grasp+close): {len(a1):3d} steps")
        print(f"Skill 2 (grasp->place hover): {len(a2):3d} steps")
        print(f"Total scripted (0+1+2):       {n_total:3d} steps")
        print(f"Budget remaining for skill3:  {budget_for_skill3:3d} steps (P6v14a max=200)")

        # Verify final scripted q matches transport_hover waypoint
        q_target_rad = self.deg_to_rad(self.q_transport_deg)
        q_target_rad[5] = math.radians(GRIPPER_CLOSE_DEG)
        max_residual = np.max(np.abs(q - q_target_rad))
        print(f"Final scripted q residual:    {math.degrees(max_residual):.3f} deg "
              f"({'OK' if max_residual < math.radians(1.0) else 'FAIL'})")

        # FK of final q (should be near transport hover TCP)
        tcp_final = fk_tcp(np.degrees(q))
        tcp_target = (self.place_xyz[0], self.place_xyz[1], TCP_GRASP_Z + self.transport_offset_z)
        d_mm = np.linalg.norm(np.array(tcp_target) - tcp_final) * 1000.0
        print(f"Final TCP:                   ({tcp_final[0]*1000:+.1f},{tcp_final[1]*1000:+.1f},{tcp_final[2]*1000:+.1f})mm")
        print(f"Target TCP:                  ({tcp_target[0]*1000:+.1f},{tcp_target[1]*1000:+.1f},{tcp_target[2]*1000:+.1f})mm")
        print(f"Residual TCP error:          {d_mm:.2f}mm")

        return n_total < 200 and max_residual < math.radians(1.0)


# =====================================================================
# Part 2: ChainRunner (B200 only, isaaclab dep — imported lazy)
# =====================================================================
def run_chain_isaac(sponge_xy: tuple, episode_idx: int, model_path: str,
                    place_xyz: tuple = L1_SP1, num_envs: int = 1, headless: bool = True):
    """Full chain in Isaac Sim. B200 only. Loads model_499.pt for skill 3.

    Returns dict of metrics: success, final_sponge_z, n_steps_per_skill, etc.
    """
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=headless, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl                              # registers env
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.reward_phase = 6
    # Turn OFF all curriculum (we control state externally)
    cfg.curriculum_pregrasp = False
    cfg.curriculum_pregrasp_hover = False
    cfg.curriculum_post_grasp_cap = False
    cfg.curriculum_disable_nearzone_cap = False
    cfg.curriculum_spawn_min_r = 0.0
    cfg.curriculum_spawn_max_r = 0.0
    # Long episode so chain doesn't auto-truncate mid-skill (200 step default = 2s).
    # (delta 5/14): chain extended to Skill 0/1a/1b/1c/2/3/4 = max ~1100 step (200+150+
    # 200+80+120+200+150). Real wall ~300-600 step (early convergence). 15s = 1500 step.
    cfg.episode_length_s = 15.0

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Load PPO runner just to access the actor (model_499.pt was trained with this cfg)
    ppo_cfg = RoArmPickPPORunnerCfg()
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir="/tmp/chain_runner", device=env.unwrapped.device)
    state = torch.load(model_path, map_location=env.unwrapped.device, weights_only=False)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    target = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    target.load_state_dict(sd, strict=False)
    target.eval()
    print(f"[chain] loaded P6v14a actor from {model_path}")

    base_env = env.unwrapped       # access internals for fixed sponge spawn
    device = base_env.device

    # === Reset and override sponge pose ===
    obs, _ = env.reset()

    sponge_xyz = torch.tensor(
        [[sponge_xy[0], sponge_xy[1], SPONGE_CENTER_Z]], device=device
    ).repeat(num_envs, 1)
    sponge_quat = torch.zeros((num_envs, 4), device=device); sponge_quat[:, 0] = 1.0     # identity
    sponge_pose_w = torch.cat([sponge_xyz + base_env.scene.env_origins, sponge_quat], dim=-1)
    base_env._sponge.write_root_pose_to_sim(sponge_pose_w)
    base_env._sponge.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=device))

    # Set robot to HOME
    home_rad = torch.tensor(HOME_RAD, device=device, dtype=torch.float32).repeat(num_envs, 1)
    base_env._robot.write_joint_state_to_sim(home_rad, torch.zeros_like(home_rad))
    base_env.robot_dof_targets[:] = home_rad

    # === Plan scripted actions for skills 0/1/2 ===
    planner = TrajectoryPlanner(
        sponge_xyz=(sponge_xy[0], sponge_xy[1], SPONGE_CENTER_Z),
        place_xyz=place_xyz,
    )

    metrics = {"skill_steps": {0: 0, 1: 0, 2: 0, 3: 0},
               "skill_tcp_err_mm": {0: None, 1: None, 2: None},
               "success_step": -1, "final_sponge_z": None,
               "final_d_xy": None, "final_d_z": None,
               "grasped_at_skill1_end": None}

    current_q_np = base_env._robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    gripper_mesh_vertices_m, gripper_mesh_path = _load_gripper_mesh_vertices_m()
    if gripper_mesh_vertices_m is not None:
        mesh_min = gripper_mesh_vertices_m.min(axis=0) * 1000.0
        mesh_max = gripper_mesh_vertices_m.max(axis=0) * 1000.0
        print(f"[diag 5-F/proxy] gripper_link mesh={gripper_mesh_path}", flush=True)
        print(f"[diag 5-F/proxy] gripper_link local_bbox_mm "
              f"min=({mesh_min[0]:+.1f},{mesh_min[1]:+.1f},{mesh_min[2]:+.1f}) "
              f"max=({mesh_max[0]:+.1f},{mesh_max[1]:+.1f},{mesh_max[2]:+.1f}) "
              f"span=({mesh_max[0]-mesh_min[0]:.1f},{mesh_max[1]-mesh_min[1]:.1f},{mesh_max[2]-mesh_min[2]:.1f}) "
              f"NOTE: single gripper_link collision mesh; true inner jaw gap is not directly represented.",
              flush=True)
    else:
        print("[diag 5-F/proxy] WARN gripper_link mesh not found; skip mesh bbox diagnostics", flush=True)

    def _sponge_diag(label: str):
        env_origin = base_env.scene.env_origins[0].detach().cpu().numpy()
        raw_local = base_env._sponge.data.root_pos_w[0].detach().cpu().numpy() - env_origin
        cache_local = base_env._sponge_pos_w[0].detach().cpu().numpy() - env_origin
        top_z = raw_local[2] + SPONGE_HEIGHT_EDGE / 2.0
        bottom_z = raw_local[2] - SPONGE_HEIGHT_EDGE / 2.0
        tcp_local = base_env._tcp_pos_w[0].detach().cpu().numpy() - env_origin
        d_tcp_sponge = float(np.linalg.norm(tcp_local - raw_local))
        print(f"[diag 5-D/{label}] raw_root=({raw_local[0]*1000:+.1f},{raw_local[1]*1000:+.1f},{raw_local[2]*1000:+.1f})mm "
              f"cache_root=({cache_local[0]*1000:+.1f},{cache_local[1]*1000:+.1f},{cache_local[2]*1000:+.1f})mm "
              f"assumed_center_z={SPONGE_CENTER_Z*1000:+.1f}mm "
              f"bottom_z={bottom_z*1000:+.1f}mm top_z={top_z*1000:+.1f}mm "
              f"tcp_z={tcp_local[2]*1000:+.1f}mm tcp_minus_top={(tcp_local[2]-top_z)*1000:+.1f}mm "
              f"d_tcp_sponge={d_tcp_sponge*1000:.1f}mm "
              f"grasped={bool(base_env._grasped[0].item())} was_grasped={bool(base_env._was_grasped[0].item())}",
              flush=True)
        return raw_local, top_z

    def _gripper_mesh_diag(label: str, sponge_local: np.ndarray, sponge_top_z: float):
        if gripper_mesh_vertices_m is None:
            return
        env_origin = base_env.scene.env_origins[0].detach().cpu().numpy()
        grip_pos = base_env._robot.data.body_pos_w[0, base_env.gripper_link_idx].detach().cpu().numpy()
        grip_quat = base_env._robot.data.body_quat_w[0, base_env.gripper_link_idx].detach().cpu().numpy()
        verts_world_local = _quat_rotate_np(grip_quat, gripper_mesh_vertices_m) + grip_pos - env_origin
        bb_min = verts_world_local.min(axis=0)
        bb_max = verts_world_local.max(axis=0)
        sponge_half = np.array([SPONGE_LEN_LONG / 2.0, SPONGE_WIDTH / 2.0, SPONGE_HEIGHT_EDGE / 2.0])
        sponge_min = sponge_local - sponge_half
        sponge_max = sponge_local + sponge_half
        overlap_x = max(0.0, min(bb_max[0], sponge_max[0]) - max(bb_min[0], sponge_min[0]))
        overlap_y = max(0.0, min(bb_max[1], sponge_max[1]) - max(bb_min[1], sponge_min[1]))
        top_penetration = sponge_top_z - bb_min[2]
        print(f"[diag 5-E/{label}] gripper_link_origin=({(grip_pos[0]-env_origin[0])*1000:+.1f},"
              f"{(grip_pos[1]-env_origin[1])*1000:+.1f},{(grip_pos[2]-env_origin[2])*1000:+.1f})mm "
              f"mesh_world_bbox_min=({bb_min[0]*1000:+.1f},{bb_min[1]*1000:+.1f},{bb_min[2]*1000:+.1f})mm "
              f"max=({bb_max[0]*1000:+.1f},{bb_max[1]*1000:+.1f},{bb_max[2]*1000:+.1f})mm "
              f"mesh_min_z_minus_sponge_top={(bb_min[2]-sponge_top_z)*1000:+.1f}mm "
              f"top_penetration_if_positive={top_penetration*1000:+.1f}mm "
              f"xy_aabb_overlap=({overlap_x*1000:.1f},{overlap_y*1000:.1f})mm "
              f"world_y_span={(bb_max[1]-bb_min[1])*1000:.1f}mm vs sponge_width={SPONGE_WIDTH*1000:.1f}mm",
              flush=True)

    print(f"[chain] init current_q_deg = {[f'{math.degrees(x):+.2f}' for x in current_q_np]}", flush=True)
    sponge_pos_init = base_env._sponge.data.root_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    print(f"[chain] init sponge_pos_local = ({sponge_pos_init[0]*1000:+.1f}, {sponge_pos_init[1]*1000:+.1f}, {sponge_pos_init[2]*1000:+.1f})mm "
          f"(expected ({sponge_xy[0]*1000:+.1f}, {sponge_xy[1]*1000:+.1f}, {SPONGE_CENTER_Z*1000:+.1f})mm)", flush=True)
    _sponge_diag("after_write_before_step")
    total_step = 0

    def step_action(action_np):
        # RslRlVecEnvWrapper.step returns (obs_TensorDict, rewards, dones, extras) — 4-tuple
        nonlocal total_step, current_q_np
        action = torch.tensor(action_np, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        obs_t, rew, dones, extras = env.step(action)
        # CLOSED-LOOP: read actual sim joint state (not the open-loop predicted q)
        current_q_np[:] = base_env._robot.data.joint_pos[0].detach().cpu().numpy()
        total_step += 1
        return obs_t, dones

    def run_skill_closed_loop(name, target_q_rad, max_steps, tol_rad=0.03, log_every=0,
                              exclude_gripper: bool = False):
        """Scripted skill execution — bypasses PPO action interface.

        Mechanism: set robot_dof_targets directly to IK waypoint (force-set every step,
        env._pre_physics_step adds null action so target unchanged). PD controller drives
        robot toward target over many sim steps. Loop terminates when actual joint state
        is within tol_rad of target.

        Rationale: closed-loop via env action_scale=0.1 caused limit cycle
        (saturated ±1 action accumulates robot_dof_targets to joint limit, then
        robot follows that limit instead of IK target — verified run #6 elbow 128°
        overshoot for 200 step). Direct target-set is clean alternative.

        5/14 (δ.4) exclude_gripper: D006 found that any gripper target/actual mismatch
        (e.g. GRIPPER_OPEN_DEG=-10 clamped by URDF lower=0 → actual=0 → err=10°) pollutes
        max_joint_err and blocks tol break-out (Skill 0/1a forced to max_steps in δ.2).
        With exclude_gripper=True, tol/break uses max over joint 0-4 only; gripper q is
        still logged separately. Use True for any scripted skill whose objective is arm
        pose (Skill 0/1a/1b{1,2,3}/2/4). Use False for Skill 1c (gripper close is the
        objective itself).

        Returns dict: {steps, max_arm_err_rad, max_full_err_rad, gripper_q_rad,
                       gripper_err_rad, final_q_rad}.
        """
        target_t = torch.tensor(target_q_rad, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
        null_action = np.zeros(6, dtype=np.float32)
        skill_steps = 0
        last_arm_err = 0.0
        last_full_err = 0.0
        last_gripper_err = 0.0
        target_arr = np.asarray(target_q_rad)
        for s in range(max_steps):
            err = target_arr - current_q_np
            arm_err = float(np.max(np.abs(err[:5])))   # joint 0-4 (base, shoulder, elbow, wrist_p, wrist_r)
            full_err = float(np.max(np.abs(err)))      # joint 0-5 (incl. gripper)
            gripper_err = float(err[5])
            last_arm_err = arm_err
            last_full_err = full_err
            last_gripper_err = gripper_err
            break_err = arm_err if exclude_gripper else full_err
            if log_every and (s % log_every == 0):
                print(f"    [{name} s={s:3d}] max_arm_err_deg={math.degrees(arm_err):+.2f} "
                      f"gripper_q_deg={math.degrees(current_q_np[5]):+.2f} "
                      f"q_deg=[{','.join(f'{math.degrees(x):+.1f}' for x in current_q_np)}]", flush=True)
            if break_err < tol_rad:
                break
            # Force-set targets BEFORE env.step. env._pre_physics_step will compute
            # targets += action_scale * null_action = targets (unchanged).
            base_env.robot_dof_targets[:] = target_t
            step_action(null_action)
            skill_steps += 1
        return {
            "steps": skill_steps,
            "max_arm_err_rad": last_arm_err,
            "max_full_err_rad": last_full_err,
            "gripper_q_rad": float(current_q_np[5]),
            "gripper_err_rad": last_gripper_err,
            "final_q_rad": current_q_np.copy(),
        }

    # 5/14 (δ.4) diagnostic helper — log per-substage actual vs target TCP + sponge.
    def _diag_log(label, target_z_m, result):
        tcp_local = base_env._tcp_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
        sponge_local = base_env._sponge_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
        sponge_top_z = sponge_local[2] + SPONGE_HEIGHT_EDGE / 2.0
        tcp_target = np.array([sponge_xy[0], sponge_xy[1], target_z_m])
        tcp_err_mm = float(np.linalg.norm(tcp_target - tcp_local) * 1000.0)
        print(f"  [{label}] steps={result['steps']} "
              f"arm_err={math.degrees(result['max_arm_err_rad']):.2f}deg "
              f"gripper_q={math.degrees(result['gripper_q_rad']):+.2f}deg "
              f"TCP_actual_z={tcp_local[2]*1000:+.2f}mm "
              f"TCP_target_z={target_z_m*1000:+.1f}mm "
              f"TCP_minus_sponge_top={(tcp_local[2]-sponge_top_z)*1000:+.2f}mm "
              f"TCP_xyz_err={tcp_err_mm:.2f}mm  "
              f"TCP=({tcp_local[0]*1000:+.1f},{tcp_local[1]*1000:+.1f},{tcp_local[2]*1000:+.1f})mm  "
              f"sponge=({sponge_local[0]*1000:+.1f},{sponge_local[1]*1000:+.1f},{sponge_local[2]*1000:+.1f})mm "
              f"sponge_top={sponge_top_z*1000:+.1f}mm "
              f"grasped={bool(base_env._grasped[0].item())} was_grasped={bool(base_env._was_grasped[0].item())}",
              flush=True)
        if label.startswith("skill1b"):
            _gripper_mesh_diag(label, sponge_local, sponge_top_z)
        return tcp_local, sponge_local, tcp_err_mm

    # === Skill 0: HOME -> q_high (TCP +150mm world, top-down approach 5/14) ===
    print(f"\n[chain] Skill 0: HOME -> q_high (TCP +150mm above grasp, top-down approach)", flush=True)
    r0 = run_skill_closed_loop("skill0", planner.target_q_skill0_high(),
                                max_steps=200, log_every=40, exclude_gripper=True)
    metrics["skill_steps"][0] = r0["steps"]
    _diag_log("skill0_done", TCP_GRASP_Z + HIGH_OFFSET_Z, r0)
    tcp_now = base_env._tcp_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    target_high = (sponge_xy[0], sponge_xy[1], TCP_GRASP_Z + HIGH_OFFSET_Z)
    metrics["skill_tcp_err_mm"][0] = float(np.linalg.norm(np.array(target_high) - tcp_now) * 1000.0)

    # === Skill 1a: q_high -> q_hover (TCP from +150 to +63mm, gripper open) ===
    # Descent stage 1: bulk of vertical motion. Normal tol.
    print(f"\n[chain] Skill 1a: q_high -> q_hover (descent stage 1, TCP +150 -> +63mm)", flush=True)
    r1a = run_skill_closed_loop("skill1a_to_hover", planner.target_q_skill1a_to_hover(),
                                 max_steps=150, tol_rad=0.03, log_every=40, exclude_gripper=True)
    _diag_log("skill1a_done", TCP_GRASP_Z + HOVER_OFFSET_Z, r1a)

    # === Skill 1b multi-stage descent (δ.4 diagnostic): hover(+63) → +50 → +40 → +33 ===
    # Purpose: locate exact z where PD stalls. δ.1 and δ.2 both stalled at +51.9mm
    # (gripper-state independent per D006). 4-waypoint split lets us see whether the
    # stall starts already at +50mm (sponge-contact reaction at hover) or only at +33mm
    # (final penetration). Each substage uses TIGHT tol (0.005 rad) with exclude_gripper.
    print(f"\n[chain] Skill 1b multi-stage descent (δ.4): hover(+63) -> +50 -> +40 -> +33mm", flush=True)
    r1b1 = run_skill_closed_loop("skill1b1_to_z50", planner.target_q_skill1b1_z50(),
                                  max_steps=200, tol_rad=0.005, log_every=40, exclude_gripper=True)
    _diag_log("skill1b1_done_target+50mm", TCP_GRASP_Z + 0.017, r1b1)
    r1b2 = run_skill_closed_loop("skill1b2_to_z40", planner.target_q_skill1b2_z40(),
                                  max_steps=200, tol_rad=0.005, log_every=40, exclude_gripper=True)
    _diag_log("skill1b2_done_target+40mm", TCP_GRASP_Z + 0.007, r1b2)
    r1b3 = run_skill_closed_loop("skill1b3_to_z33", planner.target_q_skill1b3_z33(),
                                  max_steps=200, tol_rad=0.005, log_every=40, exclude_gripper=True)
    _diag_log("skill1b3_done_target+33mm", TCP_GRASP_Z, r1b3)

    tcp_now = base_env._tcp_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    target_grasp = (sponge_xy[0], sponge_xy[1], TCP_GRASP_Z)
    tcp_err_pre_close_mm = float(np.linalg.norm(np.array(target_grasp) - tcp_now) * 1000.0)
    steps1b = r1b1["steps"] + r1b2["steps"] + r1b3["steps"]
    print(f"  [skill1b summary] total_steps={steps1b} "
          f"per_stage=({r1b1['steps']},{r1b2['steps']},{r1b3['steps']}) "
          f"final_tcp_err_pre_close={tcp_err_pre_close_mm:.2f}mm  "
          f"stall_signature={'TRUE' if r1b3['steps']==200 else 'FALSE_at_b3'}", flush=True)

    # === Skill 1c: close gripper at grasp pose ===
    print(f"\n[chain] Skill 1c: close gripper (q -> {GRIPPER_CLOSE_DEG}deg)", flush=True)
    r1c = run_skill_closed_loop("skill1c_close", planner.target_q_skill1c_close(),
                                 max_steps=80, tol_rad=0.03, exclude_gripper=False)
    metrics["skill_steps"][1] = r1a["steps"] + steps1b + r1c["steps"]
    tcp_now = base_env._tcp_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    metrics["skill_tcp_err_mm"][1] = float(np.linalg.norm(np.array(target_grasp) - tcp_now) * 1000.0)
    metrics["grasped_at_skill1_end"] = bool(base_env._grasped[0].item())
    gripper_q_now = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].item())
    d_sponge_tcp = float(torch.norm(base_env._sponge_pos_w[0] - base_env._tcp_pos_w[0]).item())
    sponge_pos_after1 = base_env._sponge_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    tcp_pos_after1 = base_env._tcp_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    print(f"  close steps={r1c['steps']} max_arm_err={math.degrees(r1c['max_arm_err_rad']):.2f}deg "
          f"tcp_err={metrics['skill_tcp_err_mm'][1]:.1f}mm "
          f"gripper_q={math.degrees(gripper_q_now):.2f}deg "
          f"d_sponge_tcp={d_sponge_tcp*1000:.1f}mm "
          f"grasped={metrics['grasped_at_skill1_end']}", flush=True)
    print(f"    tcp_after1=({tcp_pos_after1[0]*1000:+.1f},{tcp_pos_after1[1]*1000:+.1f},{tcp_pos_after1[2]*1000:+.1f})mm  "
          f"sponge_after1=({sponge_pos_after1[0]*1000:+.1f},{sponge_pos_after1[1]*1000:+.1f},{sponge_pos_after1[2]*1000:+.1f})mm", flush=True)

    # === Skill 2: grasp -> place hover ===
    print(f"\n[chain] Skill 2: grasp -> place hover (closed-loop)", flush=True)
    r2 = run_skill_closed_loop("skill2", planner.target_q_skill2(), max_steps=120, exclude_gripper=True)
    metrics["skill_steps"][2] = r2["steps"]
    tcp_now = base_env._tcp_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    target_transport = (place_xyz[0], place_xyz[1], TCP_GRASP_Z + planner.transport_offset_z)
    metrics["skill_tcp_err_mm"][2] = float(np.linalg.norm(np.array(target_transport) - tcp_now) * 1000.0)
    grasped_skill2_end = bool(base_env._grasped[0].item())
    sponge_z_now = float(base_env._sponge_pos_w[0, 2].item() - base_env.scene.env_origins[0, 2].item())
    print(f"  done. steps={r2['steps']} max_arm_err={math.degrees(r2['max_arm_err_rad']):.2f}deg "
          f"tcp_err={metrics['skill_tcp_err_mm'][2]:.1f}mm "
          f"grasped={grasped_skill2_end} sponge_z={sponge_z_now*1000:.1f}mm", flush=True)

    # === Skill 3: place + release (LEARNED) ===
    print(f"\n[chain] Skill 3: place + release (P6v14a/model_499 inference)", flush=True)
    obs_t = env.get_observations()  # rsl_rl wrapper returns TensorDict {"policy": tensor}
    # P6v14a was trained with 200-step horizon; give it that many steps for release.
    # (delta 5/14): early-terminate when gripper opens (release detected) + buffer N steps
    # for sponge to fall and settle. Avoids basin-sweep observation: 200-step P6v14a
    # inference DOES release at step ~13 but then continues moving robot, knocking
    # sponge to final d_xy 147-227mm. Terminate at release_step + 15 keeps sponge close.
    skill3_budget = 200
    release_detected_step = -1
    post_release_buffer = 15  # steps to let sponge fall + settle after gripper opens
    grasp_thresh = base_env.cfg.grasp_gripper_thresh  # 0.4 rad
    for s in range(skill3_budget):
        with torch.inference_mode():
            action = target.act_inference(obs_t)
        obs_t, rew, dones, extras = env.step(action)
        metrics["skill_steps"][3] += 1
        sponge_pos_now = base_env._sponge_pos_w[0] - base_env.scene.env_origins[0]
        target_world = base_env._target_world[0] - base_env.scene.env_origins[0]
        d_xy = float(torch.norm(sponge_pos_now[:2] - target_world[:2]).item())
        d_z = float(torch.abs(sponge_pos_now[2] - target_world[2]).item())
        gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].item())
        # Detect release (gripper opens past threshold).
        if release_detected_step < 0 and gripper_q < grasp_thresh:
            release_detected_step = s
            print(f"  [skill3] release detected at step {s} (gripper_q={math.degrees(gripper_q):.2f}deg)", flush=True)
        if d_xy < 0.030 and d_z < 0.025 and metrics["success_step"] < 0:
            metrics["success_step"] = metrics["skill_steps"][3]
            metrics["final_d_xy"] = d_xy
            metrics["final_d_z"] = d_z
            metrics["final_sponge_z"] = float(sponge_pos_now[2].item())
            print(f"  [skill3] SUCCESS at step {s+1}: d_xy={d_xy*1000:.1f}mm d_z={d_z*1000:.1f}mm")
        # Early-terminate Skill 3: release + buffer reached. Hand off to Skill 4.
        if release_detected_step >= 0 and (s - release_detected_step) >= post_release_buffer:
            print(f"  [skill3] terminating Skill 3 at step {s} (release_step={release_detected_step} + {post_release_buffer} buffer)", flush=True)
            break
        if dones.any():
            break

    # Capture final state at end of Skill 3 (this is what matters for chain success — TRUE
    # settled position, not single-step pass-through).
    sponge_pos_now = base_env._sponge_pos_w[0] - base_env.scene.env_origins[0]
    target_world = base_env._target_world[0] - base_env.scene.env_origins[0]
    metrics["final_d_xy"] = float(torch.norm(sponge_pos_now[:2] - target_world[:2]).item())
    metrics["final_d_z"] = float(torch.abs(sponge_pos_now[2] - target_world[2]).item())
    metrics["final_sponge_z"] = float(sponge_pos_now[2].item())
    metrics["release_step"] = release_detected_step
    chain_success_settled = metrics["final_d_xy"] < 0.030 and metrics["final_d_z"] < 0.025
    print(f"  [skill3] post-Skill3 settled state: d_xy={metrics['final_d_xy']*1000:.1f}mm "
          f"d_z={metrics['final_d_z']*1000:.1f}mm  CHAIN_SETTLED={'YES' if chain_success_settled else 'NO'}", flush=True)

    # === Skill 4: retreat to q_high(place) to keep sponge undisturbed ===
    # Use place_xyz + HIGH_OFFSET_Z for retreat target (above place, not original sponge).
    print(f"\n[chain] Skill 4: retreat to TCP +150mm above place (avoid sponge knock-away)", flush=True)
    place_high_q_deg = planner._ik_with_gripper(
        (planner.place_xyz[0], planner.place_xyz[1], TCP_GRASP_Z + HIGH_OFFSET_Z),
        planner.q_transport_deg.tolist(),
    )
    retreat_target = planner.deg_to_rad(place_high_q_deg)
    retreat_target[5] = math.radians(GRIPPER_OPEN_DEG)
    r4 = run_skill_closed_loop("skill4_retreat", retreat_target,
                                max_steps=150, tol_rad=0.03, exclude_gripper=True)
    metrics.setdefault("skill_steps", {})[4] = r4["steps"]
    sponge_pos_final = base_env._sponge_pos_w[0] - base_env.scene.env_origins[0]
    metrics["final_d_xy_after_retreat"] = float(torch.norm(sponge_pos_final[:2] - target_world[:2]).item())
    metrics["final_d_z_after_retreat"] = float(torch.abs(sponge_pos_final[2] - target_world[2]).item())
    chain_success_after_retreat = (metrics["final_d_xy_after_retreat"] < 0.030
                                    and metrics["final_d_z_after_retreat"] < 0.025)
    print(f"  done. steps={r4['steps']} max_arm_err={math.degrees(r4['max_arm_err_rad']):.2f}deg "
          f"final_d_xy={metrics['final_d_xy_after_retreat']*1000:.1f}mm "
          f"final_d_z={metrics['final_d_z_after_retreat']*1000:.1f}mm "
          f"CHAIN_FINAL_SUCCESS={'YES' if chain_success_after_retreat else 'NO'}", flush=True)

    if metrics["success_step"] < 0:
        print(f"  TIMEOUT (no single-step success within Skill 3 budget)")

    env.close()
    sim_app.close()
    return metrics


# =====================================================================
# Part 3 (alpha'): Skill 3 basin-of-attraction test (B200 only)
# =====================================================================
# Goal: bypass scripted Skill 0/1/2 entirely. Force-set env to (P6v14a training
# entry + perturbation). Run Skill 3 (P6v14a model_499) inference. Measure where
# P6v14a's release behavior generalizes.
#
# P6v14a training entry (verified roarm_stack_env L808-863, curriculum_pregrasp):
#   - robot q = pregrasp_joints_rad = (-0.1541, +0.4109, +2.0177, +0.2213, 0.0, 0.8)
#   - TCP = (0.280, -0.0435, +0.0614)  [IK target = target + 5cm above]
#   - sponge spawn at (0.280, -0.0435, +0.0614)  [pinned to TCP via _update_grasp_attach]
#   - _grasped = True, _was_grasped = True (both latched at reset)
#   - target_world (env-local) = (0.280, -0.0435, +0.011383)
#
# Perturbation: shift IK target by (dx_mm, 0, dz_mm) → solve IK → robot at perturbed
# pose → sponge pinned to perturbed TCP via _update_grasp_attach.
#
# Critical sensitivity in obs (28-dim, state-only HARD RULE #17):
#   sponge_to_target = target - sponge_pos.
#   When grasped, sponge = TCP. So sponge_to_target = (-dx, 0, -0.05-dz).
#   P6v14a learned: at this z-offset, open gripper → sponge drops to target.
#
# Success metric (same as run_chain_isaac, L423): d_xy<30mm AND d_z<25mm.
# After gripper opens, sponge drops via gravity. Sponge xy ≈ TCP_xy. If TCP_xy
# is dx off target, d_xy ≈ dx after drop. So dx<30mm should pass, dx>=45mm fail.


def _force_set_p6v14a_entry(base_env, q_rad_np, tcp_local_target_np, device, num_envs: int = 1):
    """Force env state to (q_rad, sponge_at_tcp_local_target, _grasped=True).

    Uses caller-supplied TCP local target (computed via roarm_kinematics.fk_tcp
    on q_rad) — avoids reliance on _robot.data.body_pos_w cache (which may be
    stale right after write_joint_state_to_sim before next sim step).

    Sponge is placed directly at TCP world position (= env_origin + tcp_local_target).
    Even if there is a small mismatch with the actual TCP after physics propagation,
    _update_grasp_attach (called in env.step._apply_action when _grasped=True) will
    re-pin sponge to current TCP — self-correcting on first env.step.
    """
    import torch
    q_t = torch.tensor(q_rad_np, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    base_env._robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t))
    base_env._robot.set_joint_position_target(q_t)
    base_env.robot_dof_targets[:] = q_t

    # Sponge force-set at TCP local target (independent of body_pos_w cache).
    tcp_local_t = torch.tensor(tcp_local_target_np, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    env_origins = base_env.scene.env_origins[:num_envs]
    sponge_world_xyz = env_origins + tcp_local_t
    sponge_quat = torch.zeros((num_envs, 4), device=device); sponge_quat[:, 0] = 1.0
    pose7 = torch.cat([sponge_world_xyz, sponge_quat], dim=-1)
    base_env._sponge.write_root_pose_to_sim(pose7)
    base_env._sponge.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=device))

    # Latch grasp flags (matches curriculum_pregrasp behavior in _reset_idx L890-891).
    base_env._grasped[:] = True
    base_env._was_grasped[:] = True
    base_env._lift_counter[:] = 0
    base_env._lift_success_flag[:] = False
    base_env._lift_bonus_paid[:] = False
    base_env._place_counter[:] = 0
    base_env._place_success_flag[:] = False
    base_env._place_bonus_paid[:] = False
    base_env._stage3_fired[:] = False


def _run_single_basin(env, base_env, actor, device, target_xyz, dx_mm, dz_mm,
                      num_steps, log_every: int = 0):
    """One perturbation run. Returns metrics dict + per-step log list.

    Perturbation: IK target = target_xyz + (dx_mm/1000, 0, 0.050 + dz_mm/1000).
    Sponge pinned to TCP via _grasped=True latch. P6v14a inference for num_steps.
    """
    import torch

    # Solve IK for perturbed TCP position.
    pregrasp_q_rad = np.array([-0.1541, +0.4109, +2.0177, +0.2213, 0.0, 0.8])
    pregrasp_q_deg = np.degrees(pregrasp_q_rad)
    # Gripper not used in IK (joint 5 = wrist_r, joint 6 = gripper, ik_dls uses joints 0-4).
    tcp_perturbed = (
        target_xyz[0] + dx_mm / 1000.0,
        target_xyz[1],
        target_xyz[2] + 0.050 + dz_mm / 1000.0,
    )
    q_ik, conv, err_mm, n_iter = ik_dls(tcp_perturbed, pregrasp_q_deg[:6].tolist(),
                                         max_iter=300, tol_mm=0.5)
    if not conv:
        print(f"  [WARN] IK not converged for perturbed target={tcp_perturbed}, err={err_mm:.2f}mm")
    # Match P6v14a training: gripper q forced to 0.8 rad (closed).
    q_rad_out = np.radians(q_ik).copy()
    q_rad_out[5] = 0.8

    # FK on perturbed q to compute exact TCP local position (sanity check vs IK input).
    tcp_fk = fk_tcp(np.degrees(q_rad_out))
    fk_vs_target_mm = float(np.linalg.norm(tcp_fk - np.asarray(tcp_perturbed))) * 1000.0

    # Reset env (fresh state), then force-set perturbed entry.
    obs, _ = env.reset()
    _force_set_p6v14a_entry(base_env, q_rad_out, tcp_fk, device, num_envs=1)
    # Settle: one null-action env.step to propagate physics + run _update_grasp_attach
    # (sponge re-pinned to TCP after physics body_pos_w refresh).
    null_action_t = torch.zeros((1, 6), device=device, dtype=torch.float32)
    env.step(null_action_t)

    # Diagnostic: print initial state.
    # (read AFTER settling step → body_pos_w refreshed → _tcp_pos_w cache valid).
    init_tcp = base_env._tcp_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    init_sponge = base_env._sponge_pos_w[0].detach().cpu().numpy() - base_env.scene.env_origins[0].detach().cpu().numpy()
    target_local = np.asarray(target_xyz)
    print(f"  [init] target_pert_xyz=({tcp_perturbed[0]*1000:+6.1f},{tcp_perturbed[1]*1000:+6.1f},{tcp_perturbed[2]*1000:+6.1f})mm  "
          f"IK_err={err_mm:.2f}mm  conv={conv}  FK_vs_IKtarget={fk_vs_target_mm:.2f}mm")
    print(f"  [init] TCP_after_settle=({init_tcp[0]*1000:+6.1f},{init_tcp[1]*1000:+6.1f},{init_tcp[2]*1000:+6.1f})mm  "
          f"sponge_after_settle=({init_sponge[0]*1000:+6.1f},{init_sponge[1]*1000:+6.1f},{init_sponge[2]*1000:+6.1f})mm  "
          f"target=({target_local[0]*1000:+6.1f},{target_local[1]*1000:+6.1f},{target_local[2]*1000:+6.1f})mm")
    init_d_xy_mm = float(np.linalg.norm(init_sponge[:2] - target_local[:2])) * 1000.0
    init_d_z_mm = float(abs(init_sponge[2] - target_local[2])) * 1000.0
    print(f"  [init] init_d_xy={init_d_xy_mm:.1f}mm  init_d_z={init_d_z_mm:.1f}mm  "
          f"_grasped={bool(base_env._grasped[0].item())} _was_grasped={bool(base_env._was_grasped[0].item())}  "
          f"gripper_q={math.degrees(float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].item())):.2f}deg")

    # Skill 3 inference loop.
    obs_t = env.get_observations()
    metrics = {
        "dx_mm": dx_mm, "dz_mm": dz_mm,
        "ik_err_mm": err_mm, "ik_conv": bool(conv),
        "success_step": -1, "release_step": -1,
        "final_d_xy_mm": None, "final_d_z_mm": None,
        "final_sponge_z_mm": None, "final_gripper_q_deg": None,
        "min_d_xy_mm": None, "min_d_z_mm": None,
        "max_steps_run": 0,
    }
    log = []
    min_d_xy = float("inf"); min_d_z = float("inf")

    for s in range(num_steps):
        with torch.inference_mode():
            action = actor.act_inference(obs_t)
        obs_t, rew, dones, extras = env.step(action)

        sponge_pos_now = base_env._sponge_pos_w[0] - base_env.scene.env_origins[0]
        target_world = base_env._target_world[0] - base_env.scene.env_origins[0]
        d_xy = float(torch.norm(sponge_pos_now[:2] - target_world[:2]).item())
        d_z = float(torch.abs(sponge_pos_now[2] - target_world[2]).item())
        sponge_z_local = float(sponge_pos_now[2].item())
        gripper_q = float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].item())
        grasped_now = bool(base_env._grasped[0].item())

        min_d_xy = min(min_d_xy, d_xy * 1000.0)
        min_d_z = min(min_d_z, d_z * 1000.0)

        # Release event: gripper drops below threshold for first time.
        if metrics["release_step"] < 0 and gripper_q < 0.4:
            metrics["release_step"] = s

        # Success: d_xy<30mm AND d_z<25mm.
        if d_xy < 0.030 and d_z < 0.025 and metrics["success_step"] < 0:
            metrics["success_step"] = s

        if log_every and (s % log_every == 0 or s == num_steps - 1):
            log.append({
                "step": s, "d_xy_mm": d_xy * 1000.0, "d_z_mm": d_z * 1000.0,
                "sponge_z_mm": sponge_z_local * 1000.0,
                "gripper_q_deg": math.degrees(gripper_q),
                "grasped": grasped_now,
            })
        if dones.any():
            metrics["max_steps_run"] = s + 1
            break
    else:
        metrics["max_steps_run"] = num_steps

    # Final state.
    sponge_pos_final = base_env._sponge_pos_w[0] - base_env.scene.env_origins[0]
    target_world = base_env._target_world[0] - base_env.scene.env_origins[0]
    metrics["final_d_xy_mm"] = float(torch.norm(sponge_pos_final[:2] - target_world[:2]).item()) * 1000.0
    metrics["final_d_z_mm"] = float(torch.abs(sponge_pos_final[2] - target_world[2]).item()) * 1000.0
    metrics["final_sponge_z_mm"] = float(sponge_pos_final[2].item()) * 1000.0
    metrics["final_gripper_q_deg"] = math.degrees(
        float(base_env._robot.data.joint_pos[0, base_env.gripper_joint_idx].item())
    )
    metrics["min_d_xy_mm"] = min_d_xy
    metrics["min_d_z_mm"] = min_d_z

    verdict = "SUCCESS" if metrics["success_step"] >= 0 else (
        "RELEASE_NO_LAND" if metrics["release_step"] >= 0 else "NO_RELEASE"
    )
    print(f"  [done] verdict={verdict}  success_step={metrics['success_step']}  "
          f"release_step={metrics['release_step']}  "
          f"final_d_xy={metrics['final_d_xy_mm']:.1f}mm  final_d_z={metrics['final_d_z_mm']:.1f}mm  "
          f"final_sponge_z={metrics['final_sponge_z_mm']:.1f}mm  "
          f"final_gripper_q={metrics['final_gripper_q_deg']:.2f}deg")
    return metrics, log, verdict


def run_basin_sweep_isaac(target_xyz, perturb_grid, model_path: str,
                          num_envs: int = 1, headless: bool = True,
                          num_steps: int = 200, log_every: int = 25):
    """Skip scripted Skill 0/1/2. For each (dx_mm, dz_mm) in perturb_grid:
      1. env.reset() (default)
      2. force-set robot at IK(target + (dx, 0, +0.05+dz)), gripper closed q=0.8 rad
      3. force-set sponge at TCP, _grasped=True, _was_grasped=True
      4. Skill 3 (P6v14a model_499) inference for num_steps
      5. log per-step metrics

    Returns list of (perturb, metrics, log) tuples + summary printed.
    """
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=headless, enable_cameras=False)
    sim_app = app_launcher.app

    import torch
    import gymnasium as gym
    import roarm_rl
    from roarm_rl.roarm_stack_env import RoArmStackEnvCfg
    from roarm_rl.agents.rsl_rl_ppo_cfg import RoArmPickPPORunnerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    cfg = RoArmStackEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.reward_phase = 6
    cfg.curriculum_pregrasp = False
    cfg.curriculum_pregrasp_hover = False
    cfg.curriculum_post_grasp_cap = False
    cfg.curriculum_disable_nearzone_cap = False
    cfg.curriculum_spawn_min_r = 0.0
    cfg.curriculum_spawn_max_r = 0.0
    # 2s = 200 step. Matches P6v14a training horizon. No need to extend.
    cfg.episode_length_s = 2.0

    env = gym.make("RoArm-Stack-Direct-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    ppo_cfg = RoArmPickPPORunnerCfg()
    runner = OnPolicyRunner(env, ppo_cfg.to_dict(), log_dir="/tmp/basin_runner",
                            device=env.unwrapped.device)
    state = torch.load(model_path, map_location=env.unwrapped.device, weights_only=False)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    actor = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
    actor.load_state_dict(sd, strict=False)
    actor.eval()
    print(f"[basin] loaded P6v14a actor from {model_path}")

    base_env = env.unwrapped
    device = base_env.device

    results = []
    print(f"\n[basin] sweep target={target_xyz}, {len(perturb_grid)} perturbations, "
          f"num_steps={num_steps}, num_envs={num_envs}")
    print("=" * 100)
    for i, (dx_mm, dz_mm) in enumerate(perturb_grid):
        print(f"\n--- Run {i+1}/{len(perturb_grid)}: dx={dx_mm:+d}mm dz={dz_mm:+d}mm ---")
        m, log, verdict = _run_single_basin(env, base_env, actor, device,
                                            target_xyz, dx_mm, dz_mm,
                                            num_steps=num_steps, log_every=log_every)
        results.append({"dx_mm": dx_mm, "dz_mm": dz_mm, "verdict": verdict,
                        "metrics": m, "log": log})

    # Summary
    print("\n" + "=" * 100)
    print("[basin] SWEEP SUMMARY")
    print("=" * 100)
    print(f"{'dx_mm':>7} {'dz_mm':>7} {'verdict':>16} {'succ_s':>7} {'rel_s':>7} "
          f"{'fin_dxy':>8} {'fin_dz':>8} {'min_dxy':>8} {'fin_grip':>9}")
    print("-" * 100)
    for r in results:
        m = r["metrics"]
        print(f"{r['dx_mm']:>+7d} {r['dz_mm']:>+7d} {r['verdict']:>16s} "
              f"{m['success_step']:>+7d} {m['release_step']:>+7d} "
              f"{m['final_d_xy_mm']:>7.1f}mm {m['final_d_z_mm']:>7.1f}mm "
              f"{m['min_d_xy_mm']:>7.1f}mm {m['final_gripper_q_deg']:>+8.1f}d")
    print("=" * 100)

    env.close()
    sim_app.close()
    return results


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Local: skip Isaac Sim, only run TrajectoryPlanner dry-run")
    ap.add_argument("--sponge_xy", nargs=2, type=float, default=[0.25, -0.04],
                    help="Sponge spawn XY (m). Default (0.25, -0.04) inside source region.")
    ap.add_argument("--model_path", type=str,
                    default="logs/roarm_rl/p6v14a_pregrasp_resumeP6v14/model_499.pt")
    ap.add_argument("--episode", type=int, default=1, help="Episode count for B200 chain run")
    ap.add_argument("--basin_sweep", action="store_true",
                    help="B200: skip Skill 0/1/2, run Skill 3 basin-of-attraction sweep")
    ap.add_argument("--basin_dx", type=int, default=None,
                    help="Single perturbation dx_mm (overrides default grid). Use with --basin_dz.")
    ap.add_argument("--basin_dz", type=int, default=None,
                    help="Single perturbation dz_mm.")
    ap.add_argument("--basin_steps", type=int, default=200,
                    help="Skill 3 inference horizon per run (default 200 = P6v14a training horizon).")
    args = ap.parse_args()

    sponge_xyz = (args.sponge_xy[0], args.sponge_xy[1], SPONGE_CENTER_Z)

    if args.dry_run:
        print("=" * 80); print("Dry-run: TrajectoryPlanner only (no Isaac Sim)"); print("=" * 80)
        planner = TrajectoryPlanner(sponge_xyz=sponge_xyz, place_xyz=L1_SP1)
        ok = planner.dry_run_summary()
        print()
        print("=" * 80)
        print(f"Result: {'PASS' if ok else 'FAIL'}")
        print("=" * 80)
        return 0 if ok else 1

    if args.basin_sweep:
        # 6-point grid: (dx_mm, dz_mm).
        # (0, 0): memorization baseline. MUST pass else setup broken.
        # (+15, 0), (+30, 0), (+45, 0): xy sweep (success thresh = 30mm).
        # (0, +20): z higher (chain end TCP +63 vs training +61.4).
        # (+30, +20): combined.
        if args.basin_dx is not None and args.basin_dz is not None:
            grid = [(args.basin_dx, args.basin_dz)]
        else:
            grid = [(0, 0), (+15, 0), (+30, 0), (+45, 0), (0, +20), (+30, +20)]
        results = run_basin_sweep_isaac(
            target_xyz=L1_SP1,
            perturb_grid=grid,
            model_path=args.model_path,
            num_steps=args.basin_steps,
        )
        # Verdict aggregation.
        n_success = sum(1 for r in results if r["verdict"] == "SUCCESS")
        print(f"\n[basin] {n_success}/{len(results)} SUCCESS")
        return 0

    # B200 full-chain path
    print(f"\n[chain] Isaac Sim chain run, episodes={args.episode}, sponge_xy={args.sponge_xy}")
    for ep in range(args.episode):
        m = run_chain_isaac(args.sponge_xy, ep, args.model_path)
        print(f"\n[ep {ep}] metrics:")
        for k, v in m.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
