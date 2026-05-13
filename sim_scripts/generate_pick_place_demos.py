"""Generate procedural pick-place demos in Isaac Sim for BC pretraining (B1 backup plan).

Phase 0b backup: if P6v14b PPO from scratch fails (catastrophic forgetting from P6v14a
release-only policy), warm-start MLP actor via BC on these procedural demos. Effective
horizon for PPO shrinks from 200 step → ~50 step (place+release only), matching what
P6v14a pre-grasp init achieved through scaffolding.

Demo structure (single sponge → L1.spot1, matches Phase 0b annulus distribution):
  Waypoint 1: pre-grasp  — TCP at (sponge_xy, +5cm above sponge_z), gripper open
  Waypoint 2: grasp      — TCP at (sponge_xy, sponge_z + 1cm), gripper close (q=0.8)
  Waypoint 3: lift       — TCP at (sponge_xy, z=+0.12m), gripper closed
  Waypoint 4: transport  — TCP at (target_xy, z=+0.12m), gripper closed
  Waypoint 5: place      — TCP at (target_xy, target_z + 1cm), gripper closed
  Waypoint 6: release    — TCP at (target_xy, target_z + 5cm), gripper open
  Waypoint 7: retreat    — TCP back to HOME, gripper open

Each waypoint resolved via roarm_kinematics.ik_dls. Joint trajectory interpolated
with sin-profile (ManiSkill style smooth motion).

Per-demo recording (matches 28-dim obs for direct BC compatibility with PPO actor):
  obs[28]: joint_pos[6] + joint_vel[6] + sponge_pos[3] + sponge_quat[4]
          + tcp_to_sponge[3] + target_pos_local[3] + sponge_to_target[3]
  action[6]: delta_joint_target (mapped to [-1, +1] via cfg.action_scale)

Sponge spawn: SOURCE_REGIONS R1-R4 + yaw rand (matches Phase 1+ distribution; Phase 0b
annulus spawn 0.08-0.15m can be approximated as subset of R1-R4 close to L1.spot1).

Run (4090, isaaclab env):
    conda run -n isaaclab python sim_scripts/generate_pick_place_demos.py \\
        --n_demos 500 --output sim_demos_pickplace_v1/ --seed 0

⚠️ STATUS: SKELETON ONLY. Implementation deferred until Phase 0b result confirms backup needed.
   If Phase 0b sanity gate fails at iter 50 → complete this file + implement bc_pretrain.py.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_scripts"))


# Geometry (HARD RULE #19/#20)
TABLE_Z = -0.012117
SPONGE_HEIGHT_EDGE = 0.047
SPONGE_CENTER_Z = TABLE_Z + SPONGE_HEIGHT_EDGE / 2.0  # +0.011383
TARGET_L1_SPOT1 = (0.280, -0.0435, SPONGE_CENTER_Z)

# Sponge spawn regions (matches roarm_stack_env.py SOURCE_REGIONS)
SOURCE_REGIONS = [
    (0.150, 0.250, -0.220, -0.130),   # R1
    (0.150, 0.250,  0.070,  0.200),   # R2
    (0.330, 0.430, -0.220, -0.100),   # R3
    (0.330, 0.430,  0.050,  0.200),   # R4
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_demos", type=int, default=500)
    p.add_argument("--output", type=str, default=str(REPO / "sim_demos_pickplace_v1"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--annulus_only", action="store_true",
                   help="Sample sponge spawn from annulus 0.08-0.15m around L1.spot1 (Phase 0b match)")
    p.add_argument("--steps_per_wp", type=int, default=40,
                   help="Sim steps to reach each waypoint (smooth interpolation)")
    return p.parse_args()


def waypoint_sequence(sponge_xy, target_xyz):
    """Return list of 7 (tcp_xyz, gripper_q) waypoints for one demo.

    sponge_xy: (x, y) spawn position
    target_xyz: (x, y, z) place target
    """
    sx, sy = sponge_xy
    tx, ty, tz = target_xyz
    return [
        # (tcp_xyz, gripper_q)
        ((sx, sy, SPONGE_CENTER_Z + 0.05), 0.0),   # 1 pre-grasp open
        ((sx, sy, SPONGE_CENTER_Z + 0.01), 0.8),   # 2 grasp close
        ((sx, sy, 0.12),                   0.8),   # 3 lift up
        ((tx, ty, 0.12),                   0.8),   # 4 transport
        ((tx, ty, tz + 0.01),              0.8),   # 5 place down
        ((tx, ty, tz + 0.05),              0.0),   # 6 release open
        # 7 retreat to HOME handled separately (joint-space, not TCP)
    ]


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[demo_gen] n_demos={args.n_demos} output={out} seed={args.seed}")
    print(f"[demo_gen] ⚠️ SKELETON — implementation deferred per Phase 0b result")
    print(f"[demo_gen] Estimated full implementation: 1-2 days")
    print(f"[demo_gen] Components needed:")
    print(f"  1. Isaac Sim launch + RoArmStackEnv (single env or batched)")
    print(f"  2. IK resolver per waypoint (roarm_kinematics.ik_dls)")
    print(f"  3. Joint-space interpolation (sin-profile)")
    print(f"  4. Action delta computation (current_target - prev_target, mapped to [-1, 1])")
    print(f"  5. 28-dim obs recording at each sim step")
    print(f"  6. Save to torch dataset format: {{obs: (N, 28), action: (N, 6)}}")
    print(f"[demo_gen] Trigger condition: Phase 0b sanity gate iter 50 FAIL")
    print(f"[demo_gen] Companion script needed: roarm_rl/bc_pretrain.py")


if __name__ == "__main__":
    main()
