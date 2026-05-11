"""RoArmStackEnv — Phase 1.B-alpha (1 sponge -> L1.spot1 fixed target stacking).

DirectRLEnv subclass. State-only obs (HARD RULE #17 visual RL forbidden).

Geometry constants (HARD RULE #19/#20):
  TABLE_Z = -0.012117
  SPONGE: 47mm tall x 22mm wide x 125mm long (edge-stand)
  SPONGE_CENTER_Z = TABLE_Z + 47/2 = +0.011383

Target (HARD RULE #20, L1.spot1, decision recorded 2026-05-08):
  L1 Y center-to-center = 87mm -> spot1 Y = -0.0435m
  Layout center X = +0.280m (HARD RULE #21 A layout)
  -> target sponge center = (+0.280, -0.0435, +0.011383) world == base coord
  HOME = [0, 0, pi/2, 0, 0, 0]

Reward curriculum (cfg.reward_phase in {4, 5, 6}):
  P4 stabilize : reach + lift + grasp + lift_success_bonus
                 (Phase 1.A P3 reproduced; target ignored — for warm-start anneal)
  P5 navigate  : P4 + nav_reward (only when grasped, weighted by -|sponge - target|)
  P6 place     : P5 + place_bonus + place_success_bonus (single-shot)

Termination: NEVER on success (HARD RULE / Phase 1.A lesson — collapse cause).
             Episode ends only on truncation (max_episode_length).

Place release mode: gravity (NOT kinematic-pin).
  Decision: 1.B-alpha is single sponge, no other layer -> physical realism preferred.
  Decision doc: claudedocs/phase1_balpha_design_decisions_20260508.md

Observation 28-dim (Phase 1.A 22 + 6 new):
  joint_pos[6] + joint_vel[6] + sponge_pos[3] + sponge_quat[4] + tcp_to_sponge[3]
  + target_pos_local[3] + sponge_to_target[3]
"""
from __future__ import annotations

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


# =====================================================================
# Geometry constants (HARD RULE #19/#20)
# =====================================================================
TABLE_Z = -0.012117
SPONGE_HEIGHT_EDGE = 0.047
SPONGE_LEN_LONG = 0.125
SPONGE_WIDTH = 0.022
SPONGE_CENTER_Z = TABLE_Z + SPONGE_HEIGHT_EDGE / 2.0  # +0.011383

# L1.spot1 target (sponge center coord, world)
TARGET_L1_SPOT1 = (0.280, -0.0435, SPONGE_CENTER_Z)

HOME_RAD = (0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0)

# TCP offset from link5 (URDF link5_to_hand_tcp xyz="0 0 0.115428" rpy="1.5708 -1.5708 0")
TCP_LOCAL_OFFSET_M = (0.0, 0.0, 0.115428)

# Sponge spawn regions R1-R4 (m, world)
SOURCE_REGIONS = (
    (0.150, 0.250, -0.220, -0.130),   # R1 left-back
    (0.150, 0.250,  0.070,  0.200),   # R2 left-front
    (0.330, 0.430, -0.220, -0.100),   # R3 right-back
    (0.330, 0.430,  0.050,  0.200),   # R4 right-front
)

USD_PATH = os.environ.get(
    "ROARM_M3_USD_PATH",
    "/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/assets/roarm_m3/usd/roarm_m3.usd",
)


# =====================================================================
# Config
# =====================================================================
@configclass
class RoArmStackEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 4.0
    action_space = 6
    observation_space = 28   # was 22 in Phase 1.A
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "base_link_to_link1": 0.0,
                "link1_to_link2": 0.0,
                "link2_to_link3": math.pi / 2,
                "link3_to_link4": 0.0,
                "link4_to_link5": 0.0,
                "link5_to_gripper_link": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["base_link_to_link1", "link1_to_link2", "link2_to_link3",
                                   "link3_to_link4", "link4_to_link5"],
                stiffness=80.0,
                damping=4.0,
                effort_limit_sim=2.5,
                velocity_limit_sim=3.14,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["link5_to_gripper_link"],
                stiffness=80.0,
                damping=4.0,
                effort_limit_sim=2.5,
                velocity_limit_sim=3.14,
            ),
        },
    )

    sponge: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sponge",
        spawn=sim_utils.CuboidCfg(
            size=(SPONGE_LEN_LONG, SPONGE_WIDTH, SPONGE_HEIGHT_EDGE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_angular_velocity=10.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.2, restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.55, 0.55), metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.00, SPONGE_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Action / scaling (Phase 1.A unchanged)
    action_scale: float = 0.1
    dof_velocity_scale: float = 0.1

    # Reward curriculum phase (4=stabilize, 5=navigate, 6=place)
    reward_phase: int = 4

    # P4 reward weights (mirrors Phase 1.A P3)
    reach_reward_scale: float = 1.0
    action_penalty_scale: float = 0.005
    lift_reward_scale: float = 5.0
    grasp_bonus_scale: float = 2.0
    lift_success_bonus: float = 10.0       # Phase 1.A "success" -> here renamed for clarity

    # P5 nav reward (new) — bumped 1.0 -> 5.0 (Phase 1.B-α P5 v1 nav-stalled diagnosis)
    nav_reward_scale: float = 5.0          # weight for -|sponge - target|, only when grasped

    # P6 place reward (new)
    # Phase 1.B-α P6 v2 (2026-05-08): 25mm → 100mm (chicken-and-egg fix).
    # P6 v1 plateau d=91mm > 25mm thresh → place_cond never fired (place_success_rate=0.0000).
    # 100mm = curriculum start; squeeze to 50→25mm in subsequent runs after place learned.
    place_dist_thresh: float = 0.100       # 100mm (was 25mm)
    place_bonus_scale: float = 5.0         # per-step bonus while sponge near target & grounded & stable
    # P6 v3 (5/08 late, A2 #1): separate gripper-open-when-near bonus.
    # Removed gripper_open from _place_condition (was chicken-and-egg with close-fit grasp policy).
    # P6 v4 (5/09): bumped 2.0 -> 10.0 to overpower hold-path (chicken-and-egg #2 fix).
    gripper_open_bonus_scale: float = 10.0
    # P6 v4 (5/09): per-step penalty proportional to sponge height above table,
    # gated by (sponge_near AND grasped). Encourages descent before release so sponge
    # reaches grounded zone (place_cond: z<TABLE_Z+30mm). Was hovering ~10cm in P6v3.
    lower_reward_scale: float = 5.0
    place_success_bonus: float = 50.0      # single-shot when stable for N steps
    place_success_steps: int = 50          # consecutive steps required for place success

    # grasp condition (Phase 1.A unchanged)
    grasp_distance_thresh: float = 0.025
    grasp_gripper_thresh: float = 0.4

    # Lift (P4 success, kept for backward-compat reward)
    lift_success_height: float = 0.10
    lift_success_steps: int = 50

    # Target (HARD RULE #20 L1.spot1 sponge center)
    target_pos: tuple = TARGET_L1_SPOT1


# =====================================================================
# Env
# =====================================================================
class RoArmStackEnv(DirectRLEnv):
    """Phase 1.B-alpha: 1 sponge -> L1.spot1 fixed-target stacking."""

    cfg: RoArmStackEnvCfg

    def __init__(self, cfg: RoArmStackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        # body / joint indices
        self.link5_idx = self._robot.find_bodies("link5")[0][0]
        self.gripper_link_idx = self._robot.find_bodies("gripper_link")[0][0]
        self.gripper_joint_idx = self._robot.find_joints("link5_to_gripper_link")[0][0]

        # HOME pose for reset
        self._home_q = torch.tensor(
            HOME_RAD, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self._tcp_local = torch.tensor(TCP_LOCAL_OFFSET_M, device=self.device, dtype=torch.float32)
        self._regions = torch.tensor(SOURCE_REGIONS, device=self.device, dtype=torch.float32)

        # Target is per-env: env_origin + cfg.target_pos (local offset).
        # Otherwise all envs share env-0's world coord (only env 0 reachable).
        target_offset = torch.tensor(self.cfg.target_pos, device=self.device, dtype=torch.float32)
        self._target_world = self.scene.env_origins + target_offset.unsqueeze(0)

        # State trackers
        self._grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lift_success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._place_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._place_success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._place_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Cached intermediate values
        self._sponge_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._sponge_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._tcp_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

    # =================================================================
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._sponge = RigidObject(self.cfg.sponge)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["sponge"] = self._sponge

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    # =================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.cfg.action_scale * self.actions
        self.robot_dof_targets[:] = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)
        # NOTE: gravity place mode -> NO kinematic pin even when grasped.
        # Phase 1.A used kinematic-attach for grasp simplification; Phase 1.B-alpha
        # also keeps that for reach/lift consistency BUT releases physically when
        # gripper opens (so sponge falls naturally).
        if self._grasped.any():
            self._update_grasp_attach()

    # =================================================================
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        dof_pos_scaled = (
            2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        env_origins = self.scene.env_origins  # (num_envs, 3)
        sponge_pos_local = self._sponge_pos_w - env_origins
        tcp_pos_local = self._tcp_pos_w - env_origins
        target_pos_local = self._target_world - env_origins
        tcp_to_sponge = sponge_pos_local - tcp_pos_local
        sponge_to_target = target_pos_local - sponge_pos_local

        obs = torch.cat(
            (
                dof_pos_scaled,                                                # 6
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,      # 6
                sponge_pos_local,                                              # 3
                self._sponge_quat_w,                                           # 4
                tcp_to_sponge,                                                 # 3
                target_pos_local,                                              # 3 (NEW)
                sponge_to_target,                                              # 3 (NEW)
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # =================================================================
    def _get_rewards(self) -> torch.Tensor:
        # post_place_gate = 1 before place_success, 0 after.
        # Stops reach_reward / lift_reward / nav_reward from firing AFTER place_success
        # (would otherwise lure policy to re-grasp the placed sponge).
        post_place_gate = (~self._place_success_flag).float()

        # === reach: Phase 1.A P3 mirror, gated post-place ===
        tcp_to_sponge = self._sponge_pos_w - self._tcp_pos_w
        d_tcp_sponge = torch.norm(tcp_to_sponge, p=2, dim=-1)
        reach_reward = -d_tcp_sponge * self.cfg.reach_reward_scale * post_place_gate

        action_penalty = -torch.sum(self.actions ** 2, dim=-1) * self.cfg.action_penalty_scale

        rewards = reach_reward + action_penalty

        # P6v4 (5/09): compute d_sponge_target / sponge_near up-front for ~near gate.
        # Hold path (lift+grasp_bonus) faded out near target so release path dominates.
        d_sponge_target = torch.norm(self._target_world - self._sponge_pos_w, p=2, dim=-1)
        sponge_near = d_sponge_target < self.cfg.place_dist_thresh
        near_gate = (~sponge_near).float()  # 1 when far from target, 0 when near

        # P4 (always on for phase >= 4): lift, grasp, lift_success.
        # P6v4: lift_reward + grasp_bonus gated by ~sponge_near so they fade to 0 at target;
        # this lets release_path (gripper_open_when_near + lower_when_near) overpower the
        # hold_path (which was +6.5/step in P6v3 and prevented descent — chicken-and-egg #2).
        lift = torch.clamp(
            self._sponge_pos_w[:, 2] - TABLE_Z,
            min=0.0,
            max=self.cfg.lift_success_height,
        )
        rewards = rewards + self.cfg.lift_reward_scale * lift * post_place_gate * near_gate

        grasp_cond = self._grasp_condition()
        rewards = rewards + grasp_cond.float() * self.cfg.grasp_bonus_scale * post_place_gate * near_gate

        should_pay_lift = self._lift_success_flag & ~self._lift_bonus_paid
        rewards = rewards + should_pay_lift.float() * self.cfg.lift_success_bonus
        self._lift_bonus_paid = self._lift_bonus_paid | should_pay_lift

        # === P5: nav reward (only when grasped, gated post-place) ===
        if self.cfg.reward_phase >= 5:
            nav_reward = -d_sponge_target * self.cfg.nav_reward_scale
            rewards = rewards + nav_reward * self._grasped.float() * post_place_gate

        # === P6: place ===
        if self.cfg.reward_phase >= 6:
            place_cond = self._place_condition(d_sponge_target)
            rewards = rewards + place_cond.float() * self.cfg.place_bonus_scale

            # P6 v3 (5/08 late, A2 #1): separate gripper-open-when-near bonus.
            # P6 v4 (5/09): scale bumped 2.0 -> 10.0 to overpower hold-path.
            gripper_q = self._robot.data.joint_pos[:, self.gripper_joint_idx]
            gripper_open = gripper_q < self.cfg.grasp_gripper_thresh
            rewards = rewards + (gripper_open & sponge_near).float() * self.cfg.gripper_open_bonus_scale

            # P6 v4 (5/09): lower_when_near — descent incentive once sponge is near target.
            # P6v3 final: sponge hovered at z≈+0.10m above table, place_cond (grounded
            # z<table+30mm) never fired. This penalty (gated by sponge_near AND grasped)
            # makes lowering strictly profitable while approaching the goal.
            sponge_height_above = self._sponge_pos_w[:, 2] - TABLE_Z
            lower_reward = -sponge_height_above * self.cfg.lower_reward_scale
            rewards = rewards + lower_reward * sponge_near.float() * self._grasped.float()

            should_pay_place = self._place_success_flag & ~self._place_bonus_paid
            rewards = rewards + should_pay_place.float() * self.cfg.place_success_bonus
            self._place_bonus_paid = self._place_bonus_paid | should_pay_place

        # === Logging ===
        self.extras["log"] = {
            "reach_reward": reach_reward.mean().detach(),
            "tcp_sponge_dist_m": d_tcp_sponge.mean().detach(),
            "sponge_target_dist_m": d_sponge_target.mean().detach(),
            "sponge_height_m": (self._sponge_pos_w[:, 2] - TABLE_Z).mean().detach(),
            "grasped_frac": self._grasped.float().mean().detach(),
            "lift_success_rate": self._lift_success_flag.float().mean().detach(),
            "place_success_rate": self._place_success_flag.float().mean().detach(),
            "action_penalty": action_penalty.mean().detach(),
        }

        return rewards

    # =================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # --- lift latch (sponge held aloft >= N steps) ---
        is_aloft = (self._sponge_pos_w[:, 2] - TABLE_Z) > self.cfg.lift_success_height
        self._lift_counter = torch.where(
            is_aloft, self._lift_counter + 1, torch.zeros_like(self._lift_counter)
        )
        new_lift = (self._lift_counter >= self.cfg.lift_success_steps) & (~self._lift_success_flag)
        self._lift_success_flag = self._lift_success_flag | new_lift

        # --- place latch (sponge near target + gripper open + stable >= N steps) ---
        d_sponge_target = torch.norm(self._target_world - self._sponge_pos_w, p=2, dim=-1)
        place_cond = self._place_condition(d_sponge_target)

        self._place_counter = torch.where(
            place_cond, self._place_counter + 1, torch.zeros_like(self._place_counter)
        )
        new_place = (self._place_counter >= self.cfg.place_success_steps) & (~self._place_success_flag)
        self._place_success_flag = self._place_success_flag | new_place

        # NEVER terminate on success (HARD RULE / Phase 1.A lesson)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # =================================================================
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)

        # Robot HOME with jitter
        joint_pos = self._home_q[env_ids] + sample_uniform(
            -0.02, 0.02, (n, self._robot.num_joints), self.device
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # Sponge spawn (uniform on R1-R4)
        region_idx = torch.randint(0, 4, (n,), device=self.device)
        regions = self._regions[region_idx]
        ux = torch.rand(n, device=self.device)
        uy = torch.rand(n, device=self.device)
        sx = regions[:, 0] + ux * (regions[:, 1] - regions[:, 0])
        sy = regions[:, 2] + uy * (regions[:, 3] - regions[:, 2])
        sz = torch.full((n,), SPONGE_CENTER_Z, device=self.device)

        env_origins = self.scene.env_origins[env_ids]
        sponge_pos = env_origins + torch.stack([sx, sy, sz], dim=-1)

        yaw = sample_uniform(-math.pi, math.pi, (n,), self.device)
        cy = torch.cos(yaw / 2)
        sy_q = torch.sin(yaw / 2)
        zeros = torch.zeros_like(yaw)
        sponge_quat = torch.stack([cy, zeros, zeros, sy_q], dim=-1)

        sponge_state = torch.zeros((n, 13), device=self.device)
        sponge_state[:, 0:3] = sponge_pos
        sponge_state[:, 3:7] = sponge_quat
        self._sponge.write_root_pose_to_sim(sponge_state[:, 0:7], env_ids=env_ids)
        self._sponge.write_root_velocity_to_sim(sponge_state[:, 7:13], env_ids=env_ids)

        # State resets
        self._grasped[env_ids] = False
        self._lift_counter[env_ids] = 0
        self._lift_success_flag[env_ids] = False
        self._lift_bonus_paid[env_ids] = False
        self._place_counter[env_ids] = 0
        self._place_success_flag[env_ids] = False
        self._place_bonus_paid[env_ids] = False

        self._compute_intermediate_values(env_ids)

    # =================================================================
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)

        link5_pos = self._robot.data.body_pos_w[env_ids, self.link5_idx]
        link5_quat = self._robot.data.body_quat_w[env_ids, self.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, self._tcp_local.expand(link5_pos.shape[0], 3))
        self._tcp_pos_w[env_ids] = link5_pos + tcp_offset_world

        self._sponge_pos_w[env_ids] = self._sponge.data.root_pos_w[env_ids]
        self._sponge_quat_w[env_ids] = self._sponge.data.root_quat_w[env_ids]

        # Grasp latch (Phase 1.A logic: latch on, release when gripper opens)
        cond = self._grasp_condition()
        gripper_open = self._robot.data.joint_pos[:, self.gripper_joint_idx] < self.cfg.grasp_gripper_thresh
        self._grasped = (self._grasped & ~gripper_open) | (cond & ~gripper_open)

    def _grasp_condition(self) -> torch.Tensor:
        d = torch.norm(self._sponge_pos_w - self._tcp_pos_w, p=2, dim=-1)
        gripper_q = self._robot.data.joint_pos[:, self.gripper_joint_idx]
        return (d < self.cfg.grasp_distance_thresh) & (gripper_q >= self.cfg.grasp_gripper_thresh)

    def _place_condition(self, d_sponge_target: torch.Tensor) -> torch.Tensor:
        """Place success per-step condition.

        Phase 1.B-α P6 v3 (5/08 late, A2 #1): removed gripper_open from AND-condition.
        Reason: P5v2 close-fit grasp policy (actor.6.bias[5]=+0.798, std≈1.5,
        action_scale=0.1) saturates gripper joint to closed → gripper_open never fires →
        place_cond never fires → place_success_rate=0.0000 for 1000 iter.
        Replaced with sponge_grounded (z<table+30mm) so place can fire while sponge is
        being released. gripper_open is incentivized separately via gripper_open_when_near
        bonus in _get_rewards.

        sponge near target (≤place_dist_thresh) ∧ grounded (z<table+30mm) ∧ stable (vel<5cm/s).
        """
        sponge_lin_vel = self._sponge.data.root_lin_vel_w
        sponge_stable = torch.norm(sponge_lin_vel, p=2, dim=-1) < 0.05
        sponge_near = d_sponge_target < self.cfg.place_dist_thresh
        sponge_grounded = self._sponge_pos_w[:, 2] < (TABLE_Z + 0.030)
        return sponge_near & sponge_stable & sponge_grounded

    def _update_grasp_attach(self):
        env_ids = torch.where(self._grasped)[0]
        if len(env_ids) == 0:
            return
        link5_pos = self._robot.data.body_pos_w[env_ids, self.link5_idx]
        link5_quat = self._robot.data.body_quat_w[env_ids, self.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, self._tcp_local.expand(link5_pos.shape[0], 3))
        tcp_pos = link5_pos + tcp_offset_world

        pose7 = torch.zeros((len(env_ids), 7), device=self.device)
        pose7[:, 0:3] = tcp_pos
        pose7[:, 3:7] = self._sponge.data.root_quat_w[env_ids]
        self._sponge.write_root_pose_to_sim(pose7, env_ids=env_ids)
        zeros = torch.zeros((len(env_ids), 6), device=self.device)
        self._sponge.write_root_velocity_to_sim(zeros, env_ids=env_ids)


# =====================================================================
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by quaternion q=(w,x,y,z). Both batched on dim 0."""
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)
