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

    Reward curriculum (cfg.reward_phase in {4, 5, 6, 7}):
  P4 stabilize : reach + lift + grasp + lift_success_bonus
                 (Phase 1.A P3 reproduced; target ignored — for warm-start anneal)
  P5 navigate  : P4 + nav_reward (only when grasped, weighted by -|sponge - target|)
	  P6 place     : P5 + place_bonus + place_success_bonus (single-shot)
	  P7 transport/release : G2-A attached transport + release-only tower

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
TARGET_L1_SPOT2 = (0.280, +0.0435, SPONGE_CENTER_Z)

# G2-A v11 seed0 four-source attached handoff distribution.
# These are post-Skill-1c latch poses from the same top-down pick planner that
# fixed the gripper_link top-contact failure.  The policy starts already
# attached, so B200 time is spent on the current failing surface: transport and
# release, not re-learning Skill 1b/1c.
G2A_SEED0_ATTACHED_Q_RAD = (
    (-0.6916619, 0.4990615, 1.9663698, 0.1758003, math.pi / 2.0, 0.4537856),
    (+0.8242815, 0.3933379, 2.1345673, 0.2568303, math.pi / 2.0, 0.4537856),
    (-0.3026116, 0.8245730, 1.4823227, -0.0719685, math.pi / 2.0, 0.4537856),
    (+0.3845700, 0.9971872, 1.2305230, -0.1880081, math.pi / 2.0, 0.4537856),
)
G2A_SEED0_ATTACHED_TCP = (
    (+0.229508, -0.190064, +0.047238),
    (+0.167471, +0.181029, +0.047368),
    (+0.405889, -0.126718, +0.047577),
    (+0.439077, +0.177704, +0.047155),
)
G2A_SEED0_TARGETS = (TARGET_L1_SPOT1, TARGET_L1_SPOT2, TARGET_L1_SPOT1, TARGET_L1_SPOT2)

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
    episode_length_s = 2.0   # P6v8 α (5/14): 4.0→2.0 (400→200 step). ManiSkill 50 + DrS 3-5× 절충. Reduces stage 3 hover dominance.
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

    # P6v14 (5/12) Curriculum (Option B) — bootstrap stage-4 release signal.
    # P6v13 diag: stage 4 joint AND prob ≈ 0 → jackpot fire 0 / 800M steps → release never
    # learned. 7 reward-shape iterations (v6→v13) created 5 farming local opts but failed
    # to produce one release event. Curriculum tackles exploration, not shape.
    # Three traps addressed by Phase 0 simultaneously (see roarm_stack_env design notes):
    #   (1) spawn-at-target: min_r > xy_thresh avoids iter-0 trivial jackpot
    #   (2) near-zone cap conflict: short-transport curriculum needs cap disabled
    #   (3) tight thresholds: random π release prob ≈ 0 → relaxed Phase 0 xy/z
    # Phase 0 defaults below. Phase 1/2 via CLI override on resume.
    curriculum_spawn_min_r: float = 0.0     # 0 = legacy R1-R4 region sampling.
    curriculum_spawn_max_r: float = 0.0     # >0 = annulus around target_xy.
    curriculum_xy_thresh: float = 0.0       # 0 = use on_target_xy_thresh (production).
    curriculum_z_thresh: float = 0.0        # 0 = use on_target_z_thresh.
    curriculum_disable_nearzone_cap: bool = False  # True = remove stage 2 d<0.1 cap.

    # P6v14a (5/12) Phase 0a — pre-grasp init (Option α).
    # Episode start: TCP positioned via IK above target (+5cm), gripper closed (q>0.4),
    # sponge spawned at that position, _grasped/_was_grasped latched True. Agent's only
    # task: open gripper → sponge falls 5cm → stage 4 success_now fires (xy<0.05 AND
    # z<0.04 AND gripper_open AND stable). Bootstrap signal guaranteed without exploration.
    # Numerical: with near-zone cap KEPT (d<0.1, stage 2 = 2.0), hover at start gives
    # 2×200=400 reward vs release 5 + 8×190 = 1525. Release dominates +281%.
    # Joint values from roarm_kinematics.ik_dls on (0.280, -0.0435, +0.0614), err=0.30mm.
    # Gripper q=0.8 rad override (IK gave 0.524=30°; 0.8 ensures > grasp_thresh 0.4).
    curriculum_pregrasp: bool = False
    pregrasp_joints_rad: tuple = (-0.1541, +0.4109, +2.0177, +0.2213, 0.0, 0.8)

    # P6v14c (5/13 evening) Phase 0a' — pre-grasp HOVER init.
    # Bridge between P6v14a (sponge in hand at target → release only) and P6v14b
    # (cold-start full chain → catastrophic forgetting). P6v14b iter 999 stage4=0.0 /
    # gripper_open 0.578→0.066 within 5 iters (8th farming = stage 2 grasp-hold).
    # Phase 0a' design:
    #   - TCP at P6v14a IK pose (5cm above target, gripper OPEN q=0.0)
    #   - Sponge on table near target via annulus 5-7cm (>on_target_xy_thresh 0.05 to
    #     avoid iter-0 trivial jackpot, inside d<0.1 cap zone for post_grasp_cap design)
    #   - _grasped/_was_grasped=False (sponge NOT in hand at start)
    #   - Agent task: descend → close gripper → grasp → release at target
    # P6v14a's release-aware policy resumed; descent+grasp = new ~5-step skill.
    curriculum_pregrasp_hover: bool = False

    # P6v14c (5/13 evening) — post-grasp unconditional stage 2 cap.
    # Default stage 2 r = 4 + 3*place_progress capped to 2.0 ONLY when d_sponge_target<0.1.
    # P6v14b proved this cap insufficient: agent grasps + moves sponge to d=0.131 (outside
    # cap), earns stage 2 = 5.28/step × 178 = 940 reward without ever releasing (8th farm).
    # post_grasp_cap=True: stage 2 = post_grasp_cap_value ALWAYS when is_grasped (any d).
    # Kills "grasp + move away" farm.
    #
    # CAP VALUE = 3.0 (NOT 2.0). Critical: stage 1 reach_r max = 2.0 (at d_tcp_sponge=0).
    # If cap=2.0, stage 1 max == stage 2 cap → no PPO gradient toward grasp transition.
    # cap=3.0 gives +1.0 reward jump on grasp = positive gradient stage1→stage2.
    #
    # Margin (cap=3.0): Path A'' (grasp+hold) = 22 + 178×3 = 556 vs Path B (release) =
    # 22 + 42 + 16.5 + 168×8 + jackpot 150 = 1574 → +183% margin SAFE (C1 protocol pass).
    #
    # Trade-off: no stage 2 gradient toward target. Relies on stage 3 transient (16.5
    # first-fire at d_xy<thresh & gripper open) for release signal. Works when sponge
    # spawn already near target (d=0.05-0.07 → minimal drag needed).
    # Disable after Phase 0a' converges (Phase 0b transitions back to d<0.1 cap).
    curriculum_post_grasp_cap: bool = False
    curriculum_post_grasp_cap_value: float = 3.0  # Stage 2 r when post_grasp_cap=True. Must > stage 1 max (2.0).

    # P6v17 (5/15) — G2-A attached transport/release curriculum.
    # Starts from the stable post-pick handoff distribution instead of HOME:
    # wrist_r +90 deg, gripper latch ~26 deg, _grasped/_was_grasped=True, sponge at
    # TCP, target sampled from the v11 four-source layout.  This tests whether PPO
    # can learn a stable source-to-target attached transport and physical release
    # under the current _update_grasp_attach model. It is explicitly not a Skill
    # 1b/1c tuning path and not a release-only curriculum.
    curriculum_attached_transport_release: bool = False
    curriculum_attached_start_jitter_rad: float = 0.01

    # Attach pose-write semantics.
    # Defaults preserve the original behavior exactly: sponge xyz is pinned to
    # TCP, current sponge quaternion is preserved, and root velocity is zeroed.
    # P7 diagnostics showed quaternion preservation amplifies attached tipping,
    # so non-default modes are gated mechanics experiments, not baseline changes.
    attach_quat_mode: str = "preserve"      # preserve | identity
    attach_velocity_mode: str = "zero"      # zero | keep

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
    place_dist_thresh: float = 0.100       # 100mm (was 25mm). Stage 3 (near_target) zone.
    # P6 v8 (5/14, Fix A): split stage 3 (100mm) from stage 4 success (50mm).
    # P6 v9 (5/15): Fix A jackpot disabled — root cause was deeper:
    # ManiSkill StackCube stage 3 = is_cubeA_on_cubeB (xy_flag AND z_flag, separated)
    # but we used 3D-Euclidean d<100mm → sponge hover at z=+88mm (z_offset=77mm)
    # still fires stage 3 if xy<63mm → hover trap baked in. P6v8 jackpot fire 0/1000
    # because success_now (50mm Euclidean & gripper_open & stable) has near-zero
    # joint-probability at hover policy.
    # Fix (γ ManiSkill-strict): separate xy/z thresholds (matches ManiSkill 30mm xy
    # + 5mm z, adapted to our 47mm edge-stand sponge with z_tol 25mm to capture
    # TCP-release (z=+33mm) → free-fall → anchor (z=+11mm) transient).
    success_dist_thresh: float = 0.050     # 50mm. UNUSED in P6v9 (kept for backward log).
    # P6v11 (5/12): β jackpot 5.0 re-enabled with C (β+δ+γ) combo. P6v10 strict gate 통과
    # 0.91% × gripper_open 0.061 × stable 0.137 = joint AND ≈ 0 → jackpot 단독 fire 0 산수.
    # δ bias reset (--reset_actor_bias_idx 5) 결합으로 gripper_open exploration 강제 →
    # jackpot fire 활성화 → release path EV +5 one-time = stage 4 EV margin 확보.
    # 5.0 = P6v8 20.0보다 보수 (당시 zone 진입 0.88%라 fire 안 됨, P6v10 zone 46%로 fire 가능).
    success_jackpot: float = 5.0           # P6v11: 0 → 5.0 (β re-enable, combined with δ bias reset).
    on_target_xy_thresh: float = 0.030     # 30mm. ManiSkill xy_flag analog (P6v9, 5/15).
    on_target_z_thresh: float = 0.025      # 25mm. P6v9 z_flag analog. Captures TCP-release transient.
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
        if self.cfg.attach_quat_mode not in ("preserve", "identity"):
            raise ValueError(
                f"attach_quat_mode must be 'preserve' or 'identity' "
                f"(got {self.cfg.attach_quat_mode!r})"
            )
        if self.cfg.attach_velocity_mode not in ("zero", "keep"):
            raise ValueError(
                f"attach_velocity_mode must be 'zero' or 'keep' "
                f"(got {self.cfg.attach_velocity_mode!r})"
            )

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
        # P6 v5 (5/11): permanent latch — True once first grasp fires, never resets within episode.
        # Used to gate nav_reward and lower_reward so release (gripper_open) does not lose
        # reward signal (chicken-and-egg #3 root cause: _grasped flips to False on release,
        # nav+lower instantly cut → release becomes negative advantage → policy never releases).
        # _grasped retains its physics-attach role (kinematic pin, released on gripper_open).
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lift_success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lift_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._place_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._place_success_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._place_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # P6v12 (5/12): rising-edge latch for stage 3 transient +10 bonus (η fix). Fires once
        # per env per episode on first is_on_target=True; resets in _reset_idx. Drives release
        # path by giving open the 1-step PPO advantage over stage 2 near-zone hold (cap 2.0).
        self._stage3_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        # P6 v6 (5/12): if reward_phase == 6, use ManiSkill StackCube REPLACE tower
        # (root-cause fix for hold-path globally optimal misspecification — 5/12 doc).
        # P4/P5 keep legacy ADD-with-gating logic (proven Phase 1.A reach + nav warmup).
        if self.cfg.reward_phase == 7:
            return self._p7_transport_release_tower()
        if self.cfg.reward_phase == 6:
            return self._p6v6_replace_tower()

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

        # === P5: nav reward (gated by _was_grasped, NOT _grasped) ===
        # P6 v5 (5/11): _grasped → _was_grasped. Reason: when policy opens gripper to release,
        # _grasped flips to False, nav_reward instantly cuts → release becomes negative advantage.
        # _was_grasped is permanent latch → nav reward continues after release, so policy can
        # learn to lower & release without losing nav signal. _grasped retains physics-attach role.
        if self.cfg.reward_phase >= 5:
            nav_reward = -d_sponge_target * self.cfg.nav_reward_scale
            rewards = rewards + nav_reward * self._was_grasped.float() * post_place_gate

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
            # P6 v5 (5/11): _grasped → _was_grasped. Same rationale as nav_reward (above):
            # release must not cut the lowering signal. Otherwise policy learns to hover
            # (P6v4: sponge_height=0.13m, place_success=0). lower_reward stays on through release.
            sponge_height_above = self._sponge_pos_w[:, 2] - TABLE_Z
            lower_reward = -sponge_height_above * self.cfg.lower_reward_scale
            rewards = rewards + lower_reward * sponge_near.float() * self._was_grasped.float()

            should_pay_place = self._place_success_flag & ~self._place_bonus_paid
            rewards = rewards + should_pay_place.float() * self.cfg.place_success_bonus
            self._place_bonus_paid = self._place_bonus_paid | should_pay_place

        # === Logging ===
        # P6 v5 (5/11): added gripper_open_rate / sponge_grounded_rate / was_grasped_rate /
        # place_cond_fire_rate. Diagnoses where policy gets stuck:
        #   - gripper_open_rate ≈ 0 → actor bias still saturated (B.2 reset failed)
        #   - sponge_grounded_rate ≈ 0 → policy never descends (lower_reward insufficient)
        #   - was_grasped_rate < grasped_frac → grasp never fires (regression)
        #   - place_cond_fire_rate ≈ 0 → place_cond too strict OR upstream block
        gripper_q_log = self._robot.data.joint_pos[:, self.gripper_joint_idx]
        gripper_open_log = gripper_q_log < self.cfg.grasp_gripper_thresh
        sponge_grounded_log = self._sponge_pos_w[:, 2] < (TABLE_Z + 0.030)

        log_dict = {
            "reach_reward": reach_reward.mean().detach(),
            "tcp_sponge_dist_m": d_tcp_sponge.mean().detach(),
            "sponge_target_dist_m": d_sponge_target.mean().detach(),
            "sponge_height_m": (self._sponge_pos_w[:, 2] - TABLE_Z).mean().detach(),
            "grasped_frac": self._grasped.float().mean().detach(),
            "was_grasped_rate": self._was_grasped.float().mean().detach(),
            "gripper_open_rate": gripper_open_log.float().mean().detach(),
            "sponge_grounded_rate": sponge_grounded_log.float().mean().detach(),
            "lift_success_rate": self._lift_success_flag.float().mean().detach(),
            "place_success_rate": self._place_success_flag.float().mean().detach(),
            "action_penalty": action_penalty.mean().detach(),
        }
        if self.cfg.reward_phase >= 6:
            log_dict["place_cond_fire_rate"] = place_cond.float().mean().detach()
        self.extras["log"] = log_dict

        return rewards

    # =================================================================
    def _p6v6_replace_tower(self) -> torch.Tensor:
        """ManiSkill StackCube REPLACE tower (P6 v6, 5/12).

        Why: P6 v1-v5 used ADD-of-all-stages, which made hold-path globally optimal
        (lift+grasp +7/step × 400 step = +2800 ≫ place_bonus +5). PPO learned to
        hover forever. Confirmed via P6v5 iter 0 (gripper_open 54%) -> iter 50 (3%):
        50 iter to re-saturate hold-path despite bias reset.

        Fix: ManiSkill (https://github.com/haosulab/ManiSkill/blob/main/mani_skill/
        envs/tasks/tabletop/stack_cube.py) verified PPO baseline uses stage REPLACE:
            stage 1 reach 0-2 (default)
            stage 2 (is_grasped):       4 + (1-tanh(5*d_sponge_target))    -> 4-5
            stage 3 (sponge near target): 6 + 0.5*ungrasp + 0.5*static     -> 6-7
            stage 4 (success):            8                                  -> 8
        Max reward = 8/step capped. Hold-path mathematically cannot be globally
        optimal. SOTA recipe (Toru Lin/Yuke Zhu CoRL 2025 arXiv 2502.20396 also
        endorses 1-tanh kernel + stage-replace for sim-to-real RL).

        Adaptations to our task:
        - is_grasped = self._grasped (physics-attach state; release sets False).
          Release directly drops to stage 1 (transient ~100ms while sponge falls)
          but immediately jumps to stage 3 if sponge near target — same as ManiSkill.
        - sponge_near_target excludes stability (release transient ~100ms otherwise
          falls back to stage 2). Stability only modulates stage 3 reward via
          static_signal so transient still receives ~6.0 base.
        - place_dist_thresh kept at 100mm (curriculum start; squeeze later).
        - success requires (near_target AND gripper_open AND stable) — once latched
          via _place_success_flag, reward stays at 8 indefinitely.
        - No post_place_gate, lift_success_bonus, gripper_open_bonus, lower_reward,
          nav_reward, lift_reward, grasp_bonus. All replaced by tower structure.
        - action_penalty kept (small, throughout).
        """
        # ============ Conditions ============
        d_tcp_sponge = torch.norm(self._sponge_pos_w - self._tcp_pos_w, p=2, dim=-1)
        d_sponge_target = torch.norm(self._target_world - self._sponge_pos_w, p=2, dim=-1)

        is_grasped = self._grasped  # physics-attach state (closed gripper + sponge close)
        is_near_target = d_sponge_target < self.cfg.place_dist_thresh  # loose 3D Euclidean (log only in P6v9)

        # P6v9 (5/15): ManiSkill-strict stage 3 condition. Reference:
        # ManiSkill stack_cube.py L107-115: is_cubeA_on_cubeB = xy_flag AND z_flag.
        # We split sponge_target distance into xy and z components and gate them
        # separately. This blocks the hover trap (P6v6/v7/v8 sponge hover at z=+88mm
        # had z_offset=77mm yet xy_offset<63mm → 3D Euclidean d<100mm still fired stage 3).
        sponge_target_xy = self._target_world[:, :2] - self._sponge_pos_w[:, :2]
        sponge_target_z = self._target_world[:, 2] - self._sponge_pos_w[:, 2]
        xy_offset = torch.norm(sponge_target_xy, p=2, dim=-1)
        z_offset = torch.abs(sponge_target_z)
        # P6v14 (5/12) Curriculum: override thresholds if set (>0). Phase 0 relaxes
        # xy/z to make stage-4 success_now reachable from near-random policy.
        xy_thresh = self.cfg.curriculum_xy_thresh if self.cfg.curriculum_xy_thresh > 0.0 \
            else self.cfg.on_target_xy_thresh
        z_thresh = self.cfg.curriculum_z_thresh if self.cfg.curriculum_z_thresh > 0.0 \
            else self.cfg.on_target_z_thresh
        is_on_target = (xy_offset < xy_thresh) & (z_offset < z_thresh)

        gripper_q = self._robot.data.joint_pos[:, self.gripper_joint_idx]
        gripper_open = gripper_q < self.cfg.grasp_gripper_thresh
        gripper_low = self.robot_dof_lower_limits[self.gripper_joint_idx]
        gripper_high = self.robot_dof_upper_limits[self.gripper_joint_idx]
        # P6 v7 (5/12 ε fix): RoArm sim convention is q LOW = OPEN, q HIGH = CLOSED
        # (verified via L689 _grasp_condition: grasp = q >= grasp_gripper_thresh).
        # ManiSkill panda convention is opposite (q HIGH = OPEN). Direct copy of their
        # (q - low) / (high - low) formula was inverted on RoArm → ungrasp_signal saturated
        # at 1.0 when gripper CLOSED → stage 3 incentivized CLOSE (P6v6 root cause).
        # Fix: invert to (high - q) / (high - low) so signal=1 when fully OPEN.
        ungrasp_signal = torch.clamp(
            (gripper_high - gripper_q) / (gripper_high - gripper_low + 1e-6), 0.0, 1.0
        )  # 0~1, 1 = fully open (q at low end)
        # P6v9 (5/15) ManiSkill force-set: when not grasping, signal stays at max (1.0).
        # Reference: ManiSkill stack_cube.py L150 `ungrasp_reward[~is_cubeA_grasped] = 1.0`.
        # Why: once release occurs, stage 3 ungrasp contribution must be max regardless of
        # gripper joint state — agent shouldn't lose stage 3 reward during release transient
        # (gripper joint moving from closed→open over ~10 steps) or after release (gripper
        # state irrelevant once cube placed). Our P6v6-v8 omitted this force-set: ungrasp_signal
        # depended purely on joint q → release transient stage 3 reward 6.5 vs ManiSkill 6.75 (+0.25).
        # Cumulative: +0.25/step × ~130 steps × 0.65 fire = +21 reward for release path.
        # Combined with stage 3 def fix (xy/z separated → no hover stage 3 fire), this gives
        # release path the EV margin to outcompete hover.
        ungrasp_signal = torch.where(~is_grasped, torch.ones_like(ungrasp_signal), ungrasp_signal)

        sponge_lin_vel = self._sponge.data.root_lin_vel_w
        sponge_vel_mag = torch.norm(sponge_lin_vel, p=2, dim=-1)
        # P6v13 (5/12): V3 fix — relax vel threshold 0.05→0.10 to allow stage 4 fire during
        # release transient. P6v12 release sequence (open jaw 10 steps) → sponge bounce-down
        # vel peak ~0.07 m/s in sim → sponge_stable=False during exact release window → stage 4
        # success_now=False → jackpot fire 0. 0.10 m/s captures bounce-down within ~150ms.
        sponge_stable = sponge_vel_mag < 0.10
        static_signal = 1.0 - torch.tanh(10.0 * sponge_vel_mag)  # 0~1, 1 = static

        # P6v14b (5/13 evening): Bug #2 fix — upright check.
        # Without this, tipped sponge (90°) center z drops to table → z_offset shrinks below
        # thresh → stage 3/4 fire on tipping, would create 8th reward farming pattern.
        # sponge z-axis (body frame) projected onto world z: 1 - 2(qx²+qy²) for wxyz quat.
        # Upright = 1.0, fully tipped (90°) = ~0.0. Threshold 0.90 → ~25.8° tipping cutoff.
        qw, qx, qy, qz = (self._sponge_quat_w[:, 0], self._sponge_quat_w[:, 1],
                          self._sponge_quat_w[:, 2], self._sponge_quat_w[:, 3])
        sz_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        upright = sz_world_z > 0.90

        # ============ REPLACE tower ============
        # Stage 1: reach (default; sponge not yet grasped)
        reach_r = 2.0 * (1.0 - torch.tanh(5.0 * d_tcp_sponge))   # 0~2
        rewards = reach_r

        # Stage 2: grasped -> stage 1 reach REPLACED by 4 + 3*place_progress (P6v10 γ transport shaping, 5/12)
        # P6v9 diagnosis: hover policy reward farms stage 2 at d=120mm (4.48/step) ≈ transport
        # d=30mm (4.85/step), margin only +9%. PPO chose safe hover. γ fix: triple place_progress
        # weight so hover d=120mm → 5.39/step vs transport d=30mm → 6.55/step (margin +23%, 3× P6v9
        # transport gradient). Stage 2 max 7 (was 5), still below stage 4 cap 8.
        place_progress = 1.0 - torch.tanh(5.0 * d_sponge_target)  # 0~1
        stage2_r = 4.0 + 3.0 * place_progress  # P6v10: 4~7 (was 4~5)
        # P6v12 (5/12): η fix — near-zone cap to 2.0. P6v11 diag: stage 2 near-zone hold reward
        # 7.0/step >> stage 3 transient 6.5/step → PPO 1-step margin close+0.5 over open → close
        # hold re-saturates within 1 iter (50× faster than P5 phase, 2nd BIAS RE-SAT confirmed).
        # Cap stops at d<100mm only → γ transport shaping (sponge_far d>100mm) retained.
        # Path A close hold near-zone 199·0.856·2.0 ≈ 340 vs Path B (transport+release) ≈ 1769.
        # P6v14 (5/12) Curriculum: skip cap when Phase 0 enabled. Short-transport
        # bootstrap requires unblocked gradient through d<0.1 zone toward stage 3.
        # Cap re-enabled in Phase 1/2 (production) after release path learned.
        # P6v14c (5/13 evening) Phase 0a': post_grasp_cap=True → stage 2 = cap_value unconditionally.
        # Overrides nearzone_cap (stricter). Kills P6v14b's "grasp + move away" 8th farming.
        # cap_value default 3.0 (> stage 1 max 2.0 for PPO grasp gradient; cap=2.0 would tie).
        # Trade-off: no stage 2 gradient toward target. Relies on stage 3 transient 16.5
        # first-fire at d<thresh for release signal. Works when sponge spawned near target.
        if self.cfg.curriculum_post_grasp_cap:
            stage2_r = torch.full_like(stage2_r, self.cfg.curriculum_post_grasp_cap_value)
        elif not self.cfg.curriculum_disable_nearzone_cap:
            stage2_r = torch.where(d_sponge_target < 0.1, torch.full_like(stage2_r, 2.0), stage2_r)
        rewards = torch.where(is_grasped, stage2_r, rewards)

        # Stage 3: sponge on target (xy AND z, ManiSkill-strict, P6v9 5/15)
        # -> stage 2 REPLACED by 6 + ungrasp + static
        # P6v13 (5/12): V2 (Plan B) — η-v2 fix gating + close-hover cap.
        # P6v12 η-v1 diag: transient fired regardless of gripper state → 93% close path
        # claimed the +10 bonus (gradient mass 15.81 vs open 1.19) → release 학습 0,
        # on_target=40.6% but stage4=0.02% FLAT. Cause: transient gate had no gripper_open
        # condition. V2 fix:
        #   (1) transient AND-gate gripper_open → first-fire reward only when releasing
        #   (2) stage3_r_close cap = 3.0 → close-hover sustained 3.0/step << open 6.0+/step
        #   (3) gripper-conditional reward via torch.where(gripper_open, open_r, close_r)
        # 1-step PPO margin: close-hover 3.0 vs open-transition (first fire 16.5 / sustained 6.5+)
        # → +13.5 first / +3.5 sustained (P6v12 was +0.5 marginal). _stage3_fired latches only
        # on (is_on_target & gripper_open) so close-hover entry doesn't burn the bonus latch.
        # P6v14b (5/13 evening): Bug #2 fix — gate stage 3 by upright as well.
        # Tipped sponge with z_offset < thresh would farm stage 3 transient bonus +10
        # without ever placing the sponge correctly.
        on_target_upright = is_on_target & upright
        just_on_target = on_target_upright & gripper_open & ~self._stage3_fired
        self._stage3_fired = self._stage3_fired | (on_target_upright & gripper_open)
        stage3_r_open = 6.0 + 0.5 * ungrasp_signal + 0.5 * static_signal + 10.0 * just_on_target.float()
        stage3_r_close = torch.full_like(stage3_r_open, 3.0)
        stage3_r = torch.where(gripper_open, stage3_r_open, stage3_r_close)
        rewards = torch.where(on_target_upright, stage3_r, rewards)

        # Stage 4: success = on_target AND gripper_open AND stable AND upright (latched permanently)
        # P6v9 (5/15): success_zone (50mm Euclidean) → is_on_target (xy 30mm AND z 25mm).
        # P6v14b (5/13 evening): + upright (Bug #2 fix — block 8th farming via tipping).
        # Mirrors ManiSkill StackCube success_now = is_cubeA_on_cubeB & is_static & ~is_grasped.
        is_success_zone = d_sponge_target < self.cfg.success_dist_thresh  # kept for log only
        success_now = on_target_upright & gripper_open & sponge_stable
        # Rising edge BEFORE updating flag so jackpot fires exactly once per env per episode.
        just_succeeded = success_now & ~self._place_success_flag
        self._place_success_flag = self._place_success_flag | success_now
        rewards = torch.where(self._place_success_flag, torch.full_like(rewards, 8.0), rewards)
        # P6v8 Jackpot: one-time +success_jackpot on first stage 4 entry, counters release cliff.
        rewards = rewards + self.cfg.success_jackpot * just_succeeded.float()

        # Action penalty (small, throughout)
        action_penalty = -torch.sum(self.actions ** 2, dim=-1) * self.cfg.action_penalty_scale
        rewards = rewards + action_penalty

        # ============ Logging ============
        # Stage fractions: where is policy spending time?
        # NOTE: stage4 supersedes stage3 supersedes stage2 supersedes stage1 (priority).
        # We log mutually-exclusive stage residence by precedence.
        in_stage4 = self._place_success_flag
        # P6v14b (5/13 evening): stage 3 residence requires upright (Bug #2 fix consistency)
        in_stage3 = on_target_upright & ~in_stage4
        in_stage2 = is_grasped & ~in_stage3 & ~in_stage4
        in_stage1 = ~in_stage2 & ~in_stage3 & ~in_stage4

        sponge_grounded_log = self._sponge_pos_w[:, 2] < (TABLE_Z + 0.030)

        self.extras["log"] = {
            "reach_reward_p6v6": reach_r.mean().detach(),
            "tcp_sponge_dist_m": d_tcp_sponge.mean().detach(),
            "sponge_target_dist_m": d_sponge_target.mean().detach(),
            "sponge_height_m": (self._sponge_pos_w[:, 2] - TABLE_Z).mean().detach(),
            "grasped_frac": self._grasped.float().mean().detach(),
            "was_grasped_rate": self._was_grasped.float().mean().detach(),
            "gripper_open_rate": gripper_open.float().mean().detach(),
            "sponge_grounded_rate": sponge_grounded_log.float().mean().detach(),
            "sponge_stable_rate": sponge_stable.float().mean().detach(),
            "near_target_rate": is_near_target.float().mean().detach(),         # loose 100mm 3D (P6v9 log only)
            "is_success_zone_rate": is_success_zone.float().mean().detach(),    # 50mm 3D (P6v9 log only)
            "is_on_target_rate": is_on_target.float().mean().detach(),          # P6v9 strict xy AND z (stage 3/4 gate)
            "upright_rate": upright.float().mean().detach(),                    # P6v14b (5/13e) Bug #2 — anti-tipping diagnostic
            "sponge_z_axis_world_z_mean": sz_world_z.mean().detach(),           # P6v14b raw upright signal (1=upright, 0=tipped)
            "xy_offset_mean": xy_offset.mean().detach(),                        # P6v9 horizontal transport gap
            "z_offset_mean": z_offset.mean().detach(),                          # P6v9 vertical drop gap (hover diagnostic)
            "jackpot_fire_rate": just_succeeded.float().mean().detach(),
            "stage1_reach_frac": in_stage1.float().mean().detach(),
            "stage2_grasp_frac": in_stage2.float().mean().detach(),
            "stage3_neartgt_frac": in_stage3.float().mean().detach(),
            "stage4_success_frac": in_stage4.float().mean().detach(),
            "place_success_rate": self._place_success_flag.float().mean().detach(),
            "ungrasp_signal_mean": ungrasp_signal.mean().detach(),
            "static_signal_mean": static_signal.mean().detach(),
            "action_penalty": action_penalty.mean().detach(),
        }

        return rewards

    # =================================================================
    def _p7_transport_release_tower(self) -> torch.Tensor:
        """Transport/release-only reward for G2-A attached starts.

        This avoids the P6 failure mode where attached policies farm grasp-hold
        reward while lifting the sponge away from the target.  The only useful
        attached behavior here is to move the sponge/TCP to a release entry above
        the sampled target, then open and settle.
        """
        target_xy = self._target_world[:, :2]
        sponge_xy = self._sponge_pos_w[:, :2]
        xy_offset = torch.norm(target_xy - sponge_xy, p=2, dim=-1)
        settled_z_offset = torch.abs(self._target_world[:, 2] - self._sponge_pos_w[:, 2])
        release_entry_z = self._target_world[:, 2] + 0.029
        release_z_offset = torch.abs(release_entry_z - self._sponge_pos_w[:, 2])

        gripper_q = self._robot.data.joint_pos[:, self.gripper_joint_idx]
        gripper_open = gripper_q < self.cfg.grasp_gripper_thresh
        is_grasped = self._grasped
        was_grasped = self._was_grasped

        sponge_lin_vel = self._sponge.data.root_lin_vel_w
        sponge_vel_mag = torch.norm(sponge_lin_vel, p=2, dim=-1)
        sponge_stable = sponge_vel_mag < 0.10
        static_signal = 1.0 - torch.tanh(10.0 * sponge_vel_mag)

        qw, qx, qy, qz = (self._sponge_quat_w[:, 0], self._sponge_quat_w[:, 1],
                          self._sponge_quat_w[:, 2], self._sponge_quat_w[:, 3])
        sz_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        upright = sz_world_z > 0.90

        xy_score = 1.0 - torch.tanh(8.0 * xy_offset)
        release_z_score = 1.0 - torch.tanh(35.0 * release_z_offset)
        settled_z_score = 1.0 - torch.tanh(45.0 * settled_z_offset)

        near_release = (xy_offset < 0.040) & (release_z_offset < 0.020)
        on_target = (xy_offset < self.cfg.on_target_xy_thresh) & (settled_z_offset < self.cfg.on_target_z_thresh) & upright

        episode_progress = self.episode_length_buf.float() / max(float(self.max_episode_length), 1.0)

        # Attached transport: reward progress to the release entry, not survival.
        # Far-from-target closed holding must be low/negative, otherwise PPO farms
        # _grasped=True episodes without ever creating release attempts.
        transport_r = -3.0 + 5.0 * xy_score + 2.0 * release_z_score - 2.0 * episode_progress
        high_penalty = torch.relu(self._sponge_pos_w[:, 2] - (self._target_world[:, 2] + 0.080)) * 10.0
        transport_r = transport_r - high_penalty
        # Once at release entry, closed-hold must not dominate opening.
        transport_r = torch.where(near_release & ~gripper_open, torch.zeros_like(transport_r), transport_r)

        # Released/settling: keep reward alive only if object is actually near the target.
        released_r = -3.0 + 6.0 * xy_score + 2.0 * settled_z_score + 1.0 * static_signal
        released_r = torch.where(xy_offset < 0.080, released_r, torch.full_like(released_r, -3.0))

        success_now = on_target & gripper_open & sponge_stable & was_grasped
        just_succeeded = success_now & ~self._place_success_flag
        self._place_success_flag = self._place_success_flag | success_now

        rewards = torch.where(is_grasped, transport_r, released_r)
        rewards = torch.where(self._place_success_flag, torch.full_like(rewards, 10.0), rewards)
        rewards = rewards + 10.0 * just_succeeded.float()

        action_penalty = -torch.sum(self.actions ** 2, dim=-1) * self.cfg.action_penalty_scale
        rewards = rewards + action_penalty

        self.extras["log"] = {
            "p7_xy_offset_mean": xy_offset.mean().detach(),
            "p7_release_z_offset_mean": release_z_offset.mean().detach(),
            "p7_settled_z_offset_mean": settled_z_offset.mean().detach(),
            "p7_grasped_frac": is_grasped.float().mean().detach(),
            "p7_was_grasped_rate": was_grasped.float().mean().detach(),
            "p7_gripper_open_rate": gripper_open.float().mean().detach(),
            "p7_near_release_rate": near_release.float().mean().detach(),
            "p7_on_target_rate": on_target.float().mean().detach(),
            "p7_sponge_stable_rate": sponge_stable.float().mean().detach(),
            "p7_upright_rate": upright.float().mean().detach(),
            "p7_place_success_rate": self._place_success_flag.float().mean().detach(),
            "p7_jackpot_fire_rate": just_succeeded.float().mean().detach(),
            "p7_sponge_height_m": (self._sponge_pos_w[:, 2] - TABLE_Z).mean().detach(),
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
        # P7 has its own stricter xy/z/upright/open success latch in the reward tower.
        # Do not let the legacy P6 3D-distance place condition contaminate P7 metrics
        # or rewards.
        if self.cfg.reward_phase != 7:
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

        attached_idx = None
        if self.cfg.curriculum_attached_transport_release:
            attached_idx = torch.randint(0, 4, (n,), device=self.device)
            q_table = torch.tensor(G2A_SEED0_ATTACHED_Q_RAD, device=self.device, dtype=torch.float32)
            joint_pos = q_table[attached_idx].clone()
            jitter_amp = self.cfg.curriculum_attached_start_jitter_rad
            if jitter_amp > 0.0:
                jitter = sample_uniform(-jitter_amp, jitter_amp, (n, self._robot.num_joints), self.device)
                jitter[:, self.gripper_joint_idx] = 0.0
                joint_pos = joint_pos + jitter
        # Robot init: HOME (default) / pre-grasp closed (P6v14a Phase 0a) / pre-grasp hover (P6v14c Phase 0a').
        elif self.cfg.curriculum_pregrasp or self.cfg.curriculum_pregrasp_hover:
            pre_q = torch.tensor(self.cfg.pregrasp_joints_rad, device=self.device,
                                 dtype=torch.float32).unsqueeze(0).repeat(n, 1)
            jitter = sample_uniform(-0.02, 0.02, (n, self._robot.num_joints), self.device)
            jitter[:, self.gripper_joint_idx] = 0.0  # NO jitter on gripper (Phase 0a: must stay > 0.4; Phase 0a': must stay < 0.4)
            joint_pos = pre_q + jitter
            if self.cfg.curriculum_pregrasp_hover:
                # P6v14c Phase 0a': override gripper to OPEN (q=0.0). cfg.pregrasp_joints_rad
                # last element is 0.8 (closed for Phase 0a); we override to 0.0 (open) so
                # agent's task is descend → close → grasp → release (not "open to release only").
                joint_pos[:, self.gripper_joint_idx] = 0.0
        else:
            joint_pos = self._home_q[env_ids] + sample_uniform(
                -0.02, 0.02, (n, self._robot.num_joints), self.device
            )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.robot_dof_targets[env_ids] = joint_pos

        # Sponge spawn
        # P6v14a (5/12) Phase 0a pre-grasp: spawn at target + 5cm above. _grasped=True
        # below will cause attach in step-1 to teleport sponge to TCP (which IK places
        # at target +5cm above). Initial pose matches TCP within IK error ~0.3mm.
        # P6v14 (5/12) Curriculum: if curriculum_spawn_max_r > 0, sample annulus around
        # target_xy with min_r/max_r. Else legacy uniform on R1-R4.
        # min_r > on_target_xy_thresh required to prevent iter-0 trivial jackpot (sponge
        # spawned within target zone + HOME gripper open + zero vel → success_now=True).
        if self.cfg.curriculum_attached_transport_release:
            tcp_table = torch.tensor(G2A_SEED0_ATTACHED_TCP, device=self.device, dtype=torch.float32)
            target_table = torch.tensor(G2A_SEED0_TARGETS, device=self.device, dtype=torch.float32)
            tcp_local = tcp_table[attached_idx]
            target_local = target_table[attached_idx]
            self._target_world[env_ids] = self.scene.env_origins[env_ids] + target_local
            sx = tcp_local[:, 0]
            sy = tcp_local[:, 1]
        elif self.cfg.curriculum_pregrasp:
            tgt = torch.tensor(self.cfg.target_pos, device=self.device, dtype=torch.float32)
            sx = torch.full((n,), tgt[0].item(), device=self.device)
            sy = torch.full((n,), tgt[1].item(), device=self.device)
            # z = target_z + 0.05 (5cm above target = matches IK-pre-computed TCP pose)
        elif self.cfg.curriculum_spawn_max_r > 0.0:
            tgt_xy = torch.tensor(self.cfg.target_pos[:2], device=self.device)
            min_r = self.cfg.curriculum_spawn_min_r
            max_r = self.cfg.curriculum_spawn_max_r
            # Uniform in annulus: r = sqrt(u * (max² − min²) + min²), θ uniform [0, 2π)
            u = torch.rand(n, device=self.device)
            r = torch.sqrt(u * (max_r * max_r - min_r * min_r) + min_r * min_r)
            theta = sample_uniform(-math.pi, math.pi, (n,), self.device)
            sx = tgt_xy[0] + r * torch.cos(theta)
            sy = tgt_xy[1] + r * torch.sin(theta)
        else:
            region_idx = torch.randint(0, 4, (n,), device=self.device)
            regions = self._regions[region_idx]
            ux = torch.rand(n, device=self.device)
            uy = torch.rand(n, device=self.device)
            sx = regions[:, 0] + ux * (regions[:, 1] - regions[:, 0])
            sy = regions[:, 2] + uy * (regions[:, 3] - regions[:, 2])
        if self.cfg.curriculum_attached_transport_release:
            sz = tcp_local[:, 2]
        elif self.cfg.curriculum_pregrasp:
            sz = torch.full((n,), self.cfg.target_pos[2] + 0.050, device=self.device)
        else:
            sz = torch.full((n,), SPONGE_CENTER_Z, device=self.device)

        env_origins = self.scene.env_origins[env_ids]
        sponge_pos = env_origins + torch.stack([sx, sy, sz], dim=-1)

        if self.cfg.curriculum_attached_transport_release:
            sponge_quat = torch.zeros((n, 4), device=self.device)
            sponge_quat[:, 0] = 1.0
        elif self.cfg.curriculum_pregrasp or self.cfg.curriculum_pregrasp_hover:
            # Identity quaternion — no yaw rand. Phase 0a/0a' clean experiment.
            sponge_quat = torch.zeros((n, 4), device=self.device)
            sponge_quat[:, 0] = 1.0  # w=1 (identity)
        else:
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
        # P6v14a Phase 0a: latch grasp True at start (pre-grasp init).
        # _update_grasp_attach in next step's _apply_action will pin sponge to TCP
        # (which IK pose places at target +5cm). Agent's task: open gripper → release.
        if self.cfg.curriculum_attached_transport_release:
            self._grasped[env_ids] = True
            self._was_grasped[env_ids] = True
        elif self.cfg.curriculum_pregrasp:
            self._grasped[env_ids] = True
            self._was_grasped[env_ids] = True
        else:
            self._grasped[env_ids] = False
            self._was_grasped[env_ids] = False  # P6 v5 (5/11): permanent latch resets only at episode boundary
        self._lift_counter[env_ids] = 0
        self._lift_success_flag[env_ids] = False
        self._lift_bonus_paid[env_ids] = False
        self._place_counter[env_ids] = 0
        self._place_success_flag[env_ids] = False
        self._place_bonus_paid[env_ids] = False
        self._stage3_fired[env_ids] = False  # P6v12 (5/12): η fix — rising-edge stage 3 transient bonus latch

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
        # P6 v5 (5/11): permanent grasp latch — True after first grasp, persists through release.
        # Decouples reward gating (nav, lower) from physics attach (_grasped).
        self._was_grasped = self._was_grasped | cond

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
        if self.cfg.attach_quat_mode == "preserve":
            pose7[:, 3:7] = self._sponge.data.root_quat_w[env_ids]
        elif self.cfg.attach_quat_mode == "identity":
            pose7[:, 3] = 1.0
        else:
            raise RuntimeError(f"Unexpected attach_quat_mode={self.cfg.attach_quat_mode!r}")
        self._sponge.write_root_pose_to_sim(pose7, env_ids=env_ids)
        if self.cfg.attach_velocity_mode == "zero":
            zeros = torch.zeros((len(env_ids), 6), device=self.device)
            self._sponge.write_root_velocity_to_sim(zeros, env_ids=env_ids)


# =====================================================================
def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by quaternion q=(w,x,y,z). Both batched on dim 0."""
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)
