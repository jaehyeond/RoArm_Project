"""Sanity check: IK waypoints for hierarchical chain (Skill 0-2).

Verifies that for every key waypoint in the 4-skill chain, ik_dls converges
within tolerance and produces joints inside v6-derived limits.

Run (local, no Isaac Sim required):
    cd /home/cgxr/Documents/Robotics/RoArm_Project
    python roarm_rl/sanity_chain_ik.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_scripts"))

from roarm_kinematics import fk_tcp, ik_dls, clip_joints, JOINT_LIMITS_DEG  # noqa: E402


# ---------------------------------------------------------------
# Geometry constants (HARD RULE #19/#20)
# ---------------------------------------------------------------
TABLE_Z = -0.012117
SPONGE_HEIGHT_EDGE = 0.047
SPONGE_CENTER_Z = TABLE_Z + SPONGE_HEIGHT_EDGE / 2.0    # +0.011383 m
TCP_GRASP_Z = +0.033                                   # m world (above sponge bottom +45mm)
HOVER_OFFSET = 0.030                                   # m, 30mm above grasp

# Targets (HARD RULE #20, base coord)
L1_SP1 = (+0.280, -0.0435, SPONGE_CENTER_Z)
L1_SP2 = (+0.280, +0.0435, SPONGE_CENTER_Z)
L2_Z_CENTER = TABLE_Z + 1.5 * SPONGE_HEIGHT_EDGE       # +0.058383 m
L2_SP3 = (+0.2465, 0.0, L2_Z_CENTER)
L2_SP4 = (+0.3135, 0.0, L2_Z_CENTER)

HOME_DEG = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]


def tcp_above(sponge_xyz, offset_z=HOVER_OFFSET):
    return (sponge_xyz[0], sponge_xyz[1], TCP_GRASP_Z + offset_z)


def tcp_grasp(sponge_xyz):
    return (sponge_xyz[0], sponge_xyz[1], TCP_GRASP_Z)


def joints_within_limits(q_deg):
    names = ["base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper"]
    out = []
    for i, n in enumerate(names):
        lo, hi = JOINT_LIMITS_DEG[n]
        margin = min(q_deg[i] - lo, hi - q_deg[i])
        out.append((n, q_deg[i], lo, hi, margin))
    return out


def run_waypoint(name, target_xyz, q0_deg, verbose=False):
    """Solve IK and report. Returns (q_deg, ok)."""
    q, conv, err_mm, n_iter = ik_dls(target_xyz, q0_deg, max_iter=200, tol_mm=1.0)
    fk_back = fk_tcp(q)
    fk_err_mm = np.linalg.norm(np.array(target_xyz) - fk_back) * 1000.0

    lim = joints_within_limits(q)
    min_margin = min(l[4] for l in lim[:5])    # ignore gripper
    in_limits = min_margin > 0.0

    status = "OK " if (conv and in_limits) else "FAIL"
    print(f"  [{status}] {name:28s}  target=({target_xyz[0]*1000:+6.1f},"
          f"{target_xyz[1]*1000:+6.1f},{target_xyz[2]*1000:+6.1f})mm  "
          f"err={err_mm:5.2f}mm  iter={n_iter:3d}  "
          f"q_deg=[{q[0]:+6.1f},{q[1]:+6.1f},{q[2]:+6.1f},{q[3]:+6.1f},{q[4]:+6.1f}]  "
          f"min_margin={min_margin:+5.1f}deg")

    if verbose:
        for n, v, lo, hi, m in lim[:5]:
            flag = "✓" if m > 0 else "✗"
            print(f"        {flag} {n:9s}  {v:+7.2f}  in [{lo:+6.1f}, {hi:+6.1f}]  margin {m:+5.1f}")

    return q, conv and in_limits, err_mm


def main():
    print("=" * 100)
    print("Sanity check: IK for hierarchical chain waypoints")
    print("=" * 100)

    # ------------------------------------------------------------
    # Self-test: HOME FK
    # ------------------------------------------------------------
    p_home = fk_tcp(HOME_DEG)
    print(f"\nHOME q={HOME_DEG} -> TCP=({p_home[0]*1000:+.1f},"
          f"{p_home[1]*1000:+.1f},{p_home[2]*1000:+.1f})mm")
    print()

    fails = 0
    total = 0

    # ------------------------------------------------------------
    # SOURCE REGION targets (4 corners of source spawn region)
    # ------------------------------------------------------------
    SOURCE_CORNERS = [
        ("R1_inner_back",  (+0.150, -0.220, SPONGE_CENTER_Z)),
        ("R1_outer_back",  (+0.250, -0.130, SPONGE_CENTER_Z)),
        ("R2_inner_front", (+0.150, +0.070, SPONGE_CENTER_Z)),
        ("R2_outer_front", (+0.250, +0.200, SPONGE_CENTER_Z)),
        ("R3_inner_back",  (+0.330, -0.220, SPONGE_CENTER_Z)),
        ("R3_outer_back",  (+0.430, -0.100, SPONGE_CENTER_Z)),
        ("R4_inner_front", (+0.330, +0.050, SPONGE_CENTER_Z)),
        ("R4_outer_front", (+0.430, +0.200, SPONGE_CENTER_Z)),
    ]

    print("[A] Source region — TCP HOVER above sponge (+30mm)")
    for name, sponge_xyz in SOURCE_CORNERS:
        total += 1
        target = tcp_above(sponge_xyz)
        q, ok, err = run_waypoint(name + "_hover", target, HOME_DEG)
        if not ok: fails += 1

    print("\n[B] Source region — TCP GRASP pose (z=+33mm)")
    for name, sponge_xyz in SOURCE_CORNERS:
        total += 1
        target = tcp_grasp(sponge_xyz)
        # warm-start from hover IK
        q_hover, _, _, _ = ik_dls(tcp_above(sponge_xyz), HOME_DEG, max_iter=100)
        q, ok, err = run_waypoint(name + "_grasp", target, q_hover.tolist())
        if not ok: fails += 1

    # ------------------------------------------------------------
    # PLACE targets (L1.sp1, L1.sp2, L2.sp3, L2.sp4)
    # ------------------------------------------------------------
    print("\n[C] Place hover (50mm above place target)")
    PLACE_TARGETS = [
        ("L1_sp1", L1_SP1),
        ("L1_sp2", L1_SP2),
        ("L2_sp3", L2_SP3),
        ("L2_sp4", L2_SP4),
    ]
    for name, place_xyz in PLACE_TARGETS:
        total += 1
        target = (place_xyz[0], place_xyz[1], place_xyz[2] + 0.050 + SPONGE_HEIGHT_EDGE / 2.0)
        q, ok, err = run_waypoint(name + "_hover", target, HOME_DEG)
        if not ok: fails += 1

    print("\n[D] Place final pose (TCP at +33mm above sponge center for L1; +80mm for L2)")
    PLACE_FINAL = [
        ("L1_sp1_place", (L1_SP1[0], L1_SP1[1], +0.033)),
        ("L1_sp2_place", (L1_SP2[0], L1_SP2[1], +0.033)),
        ("L2_sp3_place", (L2_SP3[0], L2_SP3[1], +0.080)),
        ("L2_sp4_place", (L2_SP4[0], L2_SP4[1], +0.080)),
    ]
    for name, target in PLACE_FINAL:
        total += 1
        q_hover, _, _, _ = ik_dls(
            (target[0], target[1], target[2] + 0.050), HOME_DEG, max_iter=100
        )
        q, ok, err = run_waypoint(name, target, q_hover.tolist(), verbose=(not ok))
        if not ok: fails += 1

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print()
    print("=" * 100)
    print(f"Result: {total - fails}/{total} waypoints PASS  ({fails} FAIL)")
    print("=" * 100)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
