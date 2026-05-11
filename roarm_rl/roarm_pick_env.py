"""RoArmPickEnv — Phase 1.A 1-sponge pick task (state-only, HARD RULE #17).

DirectRLEnv subclass. State-only obs (NO image — Annotator/TiledCamera 미사용).

Geometry constants per HARD RULE #19/#20 (v3 edge-stand):
  TABLE_Z = -0.012117
  SPONGE: 47mm tall × 22mm wide × 125mm long (edge-stand)
  Z_TCP_GRASP_L1 = 0.033 (TCP world z at grasp = +33mm)
  HOME = [0, 0, π/2, 0, 0, 0]

Reward curriculum (set via cfg.reward_phase ∈ {1,2,3}):
  P1 reach: −‖tcp − sponge‖ + small action penalty
  P2 lift:  P1 + λ_lift · max(0, sponge_z − table_z)
  P3 grasp: P2 + grasp_bonus(when condition met) + success_bonus

Termination:
  success = sponge held aloft (z > 0.10m world) for ≥50 consecutive steps
  truncate = episode_length_buf >= max_episode_length

Grasp impl (kinematic attach):
  When (‖tcp − sponge‖ < 0.025) ∧ (gripper_joint > 0.4 rad):
    Set sponge root pose to follow TCP each step (rigid attach).
  When gripper opens (< 0.4 rad): release.

Reference: Isaac Lab v2.3.2 franka_cabinet_env.py.
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
# v3 Geometry constants (HARD RULE #19/#20)
# =====================================================================
TABLE_Z = -0.012117
SPONGE_HEIGHT_EDGE = 0.047
SPONGE_LEN_LONG = 0.125
SPONGE_WIDTH = 0.022
Z_TCP_GRASP_L1 = 0.033

HOME_RAD = (0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0)

# TCP offset from link5 (URDF: link5_to_hand_tcp xyz="0 0 0.115428" rpy="1.5708 -1.5708 0", fixed)
# Merged into link5 by URDF importer; we apply offset manually in env-local.
TCP_LOCAL_OFFSET_M = (0.0, 0.0, 0.115428)

# Source spawn regions (R1-R4 union, m world).
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


@configclass
class RoArmPickEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 4.0   # 200 timesteps at dt=1/100*decimation=1/50 effective
    action_space = 6
    observation_space = 22
    state_space = 0

    # simulation (200Hz physics, control step at 100Hz with decimation 2)
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

    # scene (default 4096; override at env init via gym.make("...", num_envs=N))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # ground plane at world z=TABLE_Z
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

    # robot
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

    # sponge (rigid body cuboid, edge-stand)
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
            pos=(0.30, 0.00, TABLE_Z + SPONGE_HEIGHT_EDGE / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # action / reward scales
    action_scale: float = 0.1                  # rad/step delta (with decimation 2 + dt 1/200)
    dof_velocity_scale: float = 0.1

    # reward curriculum phase: 1=reach, 2=lift, 3=grasp+success
    reward_phase: int = 1

    # reward weights
    reach_reward_scale: float = 1.0
    action_penalty_scale: float = 0.005
    lift_reward_scale: float = 5.0             # P2/P3
    grasp_bonus_scale: float = 2.0             # P3
    success_bonus: float = 10.0                # P3 single-shot

    # success / grasp thresholds
    grasp_distance_thresh: float = 0.025       # 25 mm
    grasp_gripper_thresh: float = 0.4          # rad (gripper joint)
    success_height: float = 0.10               # sponge_z > +0.10m world
    success_steps_required: int = 50           # consecutive steps held aloft


class RoArmPickEnv(DirectRLEnv):
    """RoArm M3 1-sponge pick env. State-only obs, kinematic-attach grasp."""

    cfg: RoArmPickEnvCfg

    def __init__(self, cfg: RoArmPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # joint limits (per env, broadcast first env -> shape (num_joints,))
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # body indices
        # Phase 1 USD: hand_tcp merged into link5 via --merge-joints.
        # We use link5 body world pose + TCP_LOCAL_OFFSET to compute TCP world pose.
        self.link5_idx = self._robot.find_bodies("link5")[0][0]
        self.gripper_link_idx = self._robot.find_bodies("gripper_link")[0][0]

        # gripper joint index (for grasp condition)
        self.gripper_joint_idx = self._robot.find_joints("link5_to_gripper_link")[0][0]

        # HOME pose for episode reset (broadcast)
        self._home_q = torch.tensor(HOME_RAD, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(
            self.num_envs, 1
        )

        # tcp offset (link5 local frame)
        self._tcp_local = torch.tensor(TCP_LOCAL_OFFSET_M, device=self.device, dtype=torch.float32)

        # source region tensor (4, 4): xmin, xmax, ymin, ymax
        self._regions = torch.tensor(SOURCE_REGIONS, device=self.device, dtype=torch.float32)

        # grasp / success state
        self._grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # tracks whether success bonus already paid this episode (true single-shot)
        self._success_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # sponge env-origin local target (for obs computation)
        self._sponge_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._sponge_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._tcp_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

    # =================================================================
    # scene
    # =================================================================
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._sponge = RigidObject(self.cfg.sponge)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["sponge"] = self._sponge

        # ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    # =================================================================
    # actions
    # =================================================================
    def _pre_physics_step(self, actions: torch.Tensor):
        # actions in [-1, 1]; map to delta-target (rad) scaled by cfg.action_scale.
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.cfg.action_scale * self.actions
        self.robot_dof_targets[:] = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)
        # write_data_to_sim is called by base DirectRLEnv after _apply_action returns.
        # If grasped, also pin sponge to TCP each physics sub-step.
        if self._grasped.any():
            self._update_grasp_attach()

    # =================================================================
    # observations
    # =================================================================
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        # joint pos scaled to [-1, 1]
        dof_pos_scaled = (
            2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        # env-local positions (subtract env origin)
        env_origins = self.scene.env_origins  # (num_envs, 3)
        sponge_pos_local = self._sponge_pos_w - env_origins
        tcp_pos_local = self._tcp_pos_w - env_origins
        tcp_to_sponge = sponge_pos_local - tcp_pos_local

        obs = torch.cat(
            (
                dof_pos_scaled,                                                # 6
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,      # 6
                sponge_pos_local,                                              # 3
                self._sponge_quat_w,                                           # 4
                tcp_to_sponge,                                                 # 3
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # =================================================================
    # rewards
    # =================================================================
    def _get_rewards(self) -> torch.Tensor:
        # Distances (computed in _compute_intermediate_values, called by _get_observations earlier this step)
        tcp_to_sponge = self._sponge_pos_w - self._tcp_pos_w
        d = torch.norm(tcp_to_sponge, p=2, dim=-1)
        reach_reward = -d * self.cfg.reach_reward_scale

        action_penalty = -torch.sum(self.actions ** 2, dim=-1) * self.cfg.action_penalty_scale

        rewards = reach_reward + action_penalty

        if self.cfg.reward_phase >= 2:
            # lift reward: how much above table (clamped)
            lift = torch.clamp(self._sponge_pos_w[:, 2] - TABLE_Z, min=0.0)
            rewards = rewards + self.cfg.lift_reward_scale * lift

        if self.cfg.reward_phase >= 3:
            grasp_cond = self._grasp_condition()
            grasp_bonus = grasp_cond.float() * self.cfg.grasp_bonus_scale
            rewards = rewards + grasp_bonus

            # success bonus (true single-shot per episode; previously masked by
            # terminated=True ending the trajectory immediately. With terminated=False
            # we explicitly track whether bonus was paid this episode.)
            should_pay = self._success_flag & ~self._success_bonus_paid
            success_now = should_pay.float() * self.cfg.success_bonus
            self._success_bonus_paid = self._success_bonus_paid | should_pay
            rewards = rewards + success_now

        # episode log (created here BEFORE _get_dones reads it next step)
        self.extras["log"] = {
            "reach_reward": reach_reward.mean().detach(),
            "tcp_sponge_dist_m": d.mean().detach(),
            "sponge_height_m": (self._sponge_pos_w[:, 2] - TABLE_Z).mean().detach(),
            "grasped_frac": self._grasped.float().mean().detach(),
            "action_penalty": action_penalty.mean().detach(),
            "success_rate": self._success_flag.float().mean().detach(),
        }

        return rewards

    # =================================================================
    # done / reset
    # =================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # success: sponge held aloft for ≥ N consecutive steps
        is_aloft = (self._sponge_pos_w[:, 2] - TABLE_Z) > self.cfg.success_height
        self._success_counter = torch.where(is_aloft, self._success_counter + 1,
                                             torch.zeros_like(self._success_counter))
        new_success = (self._success_counter >= self.cfg.success_steps_required) & (~self._success_flag)
        self._success_flag = self._success_flag | new_success

        # Never terminate on success (let timeout end episodes).
        # Rationale: terminating-on-success creates a local-min where policy learns to
        # briefly lift then "end the trial" with high return; value/advantage become
        # myopic and entropy collapse + KL blow-up follow.
        # success_flag is still latched for reward + logging.
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)

        # reset arm to HOME (with small random jitter to break symmetry)
        joint_pos = self._home_q[env_ids] + sample_uniform(
            -0.02, 0.02, (n, self._robot.num_joints), self.device
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # sample sponge spawn from one of 4 regions per env
        region_idx = torch.randint(0, 4, (n,), device=self.device)
        regions = self._regions[region_idx]  # (n, 4)
        ux = torch.rand(n, device=self.device)
        uy = torch.rand(n, device=self.device)
        sx = regions[:, 0] + ux * (regions[:, 1] - regions[:, 0])
        sy = regions[:, 2] + uy * (regions[:, 3] - regions[:, 2])
        sz = torch.full((n,), TABLE_Z + SPONGE_HEIGHT_EDGE / 2.0, device=self.device)

        env_origins = self.scene.env_origins[env_ids]
        sponge_pos = env_origins + torch.stack([sx, sy, sz], dim=-1)

        # random yaw orientation (4 quadrants)
        yaw = sample_uniform(-math.pi, math.pi, (n,), self.device)
        cy = torch.cos(yaw / 2)
        sy_q = torch.sin(yaw / 2)
        zeros = torch.zeros_like(yaw)
        sponge_quat = torch.stack([cy, zeros, zeros, sy_q], dim=-1)  # (w, x, y, z)

        sponge_state = torch.zeros((n, 13), device=self.device)
        sponge_state[:, 0:3] = sponge_pos
        sponge_state[:, 3:7] = sponge_quat
        # zero linear & angular velocity
        self._sponge.write_root_pose_to_sim(sponge_state[:, 0:7], env_ids=env_ids)
        self._sponge.write_root_velocity_to_sim(sponge_state[:, 7:13], env_ids=env_ids)

        # reset success / grasp counters
        self._grasped[env_ids] = False
        self._success_counter[env_ids] = 0
        self._success_flag[env_ids] = False
        self._success_bonus_paid[env_ids] = False

        self._compute_intermediate_values(env_ids)

    # =================================================================
    # auxiliary
    # =================================================================
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)

        link5_pos = self._robot.data.body_pos_w[env_ids, self.link5_idx]
        link5_quat = self._robot.data.body_quat_w[env_ids, self.link5_idx]

        # TCP world = link5_pos + R(link5_quat) @ tcp_local_offset
        tcp_offset_world = _quat_rotate(link5_quat, self._tcp_local.expand(link5_pos.shape[0], 3))
        self._tcp_pos_w[env_ids] = link5_pos + tcp_offset_world

        self._sponge_pos_w[env_ids] = self._sponge.data.root_pos_w[env_ids]
        self._sponge_quat_w[env_ids] = self._sponge.data.root_quat_w[env_ids]

        # update grasp state
        cond = self._grasp_condition()
        # latch grasp once acquired; release if gripper opens
        gripper_open = self._robot.data.joint_pos[:, self.gripper_joint_idx] < self.cfg.grasp_gripper_thresh
        self._grasped = (self._grasped & ~gripper_open) | (cond & ~gripper_open)

    def _grasp_condition(self) -> torch.Tensor:
        d = torch.norm(self._sponge_pos_w - self._tcp_pos_w, p=2, dim=-1)
        gripper_q = self._robot.data.joint_pos[:, self.gripper_joint_idx]
        return (d < self.cfg.grasp_distance_thresh) & (gripper_q >= self.cfg.grasp_gripper_thresh)

    def _update_grasp_attach(self):
        """For envs where grasped==True, set sponge pose to follow TCP each step."""
        env_ids = torch.where(self._grasped)[0]
        if len(env_ids) == 0:
            return
        # snapshot current TCP world pose
        link5_pos = self._robot.data.body_pos_w[env_ids, self.link5_idx]
        link5_quat = self._robot.data.body_quat_w[env_ids, self.link5_idx]
        tcp_offset_world = _quat_rotate(link5_quat, self._tcp_local.expand(link5_pos.shape[0], 3))
        tcp_pos = link5_pos + tcp_offset_world

        pose7 = torch.zeros((len(env_ids), 7), device=self.device)
        pose7[:, 0:3] = tcp_pos
        # keep current sponge orientation (or could lock to TCP — keep simpler)
        pose7[:, 3:7] = self._sponge.data.root_quat_w[env_ids]

        self._sponge.write_root_pose_to_sim(pose7, env_ids=env_ids)
        zeros = torch.zeros((len(env_ids), 6), device=self.device)
        self._sponge.write_root_velocity_to_sim(zeros, env_ids=env_ids)


# =====================================================================
# helpers
# =====================================================================
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (w, x, y, z). Both batched on dim 0."""
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)
