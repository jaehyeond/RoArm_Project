"""Numeric preflight for the audited-and-repaired p9 (numpy-only, no Isaac).

Replicates run_resampled_path's exact IK chain (chain seeding, per-waypoint
gating) for seed0_S1 at the SETTLED height (ground z=0 -> center z = H/2) and
asserts:
  1. plan targets (approach/descend/lift) pass the full vertical gate (REACH),
  2. HOME->approach transit passes position-only gating on every waypoint and
     the ARRIVAL waypoint passes the full vertical gate (FATAL-2 repair),
  3. descend + lift corridors pass the full vertical gate on every waypoint,
  4. worst-case control-step budget < 6000 (episode_length_s=60, MAJOR-c).
"""
import importlib.util
import math
import sys
from types import SimpleNamespace

import numpy as np

P9 = "/home/cgxr/Documents/Robotics/RoArm_Project/sim_scripts/p9_g0b_t3_cyld29h50_top_center_vertical_close_sweep_grasp_probe.py"
spec = importlib.util.spec_from_file_location("p9", P9)
p9 = importlib.util.module_from_spec(spec)
sys.modules["p9"] = p9
spec.loader.exec_module(p9)

args = SimpleNamespace(
    object_size_m=[0.029, 0.029, 0.050],
    approach_clearance_m=0.040,
    grasp_surface_margin_m=0.0005,
    lift_delta_m=0.010,
    close_deg=[p9.Q5_OPEN_DEG, 60.0, 45.0, 41.40, 39.0, 37.0, 35.0, 33.0, 31.65, 28.0, 24.0],
    target_error_gate_m=0.003,
    plan_tilt_gate_deg=5.0,
    max_tcp_step_m=0.010,
    command_resample_fraction=0.80,
)

failures = []

# Settled center: ground plane z=0, upright cylinder -> center z = H/2.
settled_center = np.array([0.21369616873214542, -0.19571919576125169, 0.025])
plan = p9._build_plan_from_center(args, settled_center, "seed0_S1_settled")
print(f"plan approach_tcp={plan.approach_tcp} descend_tcp={plan.descend_tcp} lift_tcp={plan.lift_tcp}")
print(
    f"plan ik_ok=({plan.approach_ik_ok},{plan.descend_ik_ok},{plan.lift_ik_ok}) "
    f"err_mm=({plan.approach_ik_err_mm:.3f},{plan.descend_ik_err_mm:.3f},{plan.lift_ik_err_mm:.3f}) "
    f"tilt=({plan.approach_tilt_deg:.3f},{plan.descend_tilt_deg:.3f},{plan.lift_tilt_deg:.3f})"
)
if not (plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok):
    failures.append("REACH: plan target IK failed")

home_tcp, _, _ = p9.fk_points(p9.HOME_ARM_DEG)
print(f"home_tcp={home_tcp}")


def waypoints(start, end):
    delta = np.asarray(end) - np.asarray(start)
    gap = float(np.linalg.norm(delta))
    max_cmd_gap = args.max_tcp_step_m * args.command_resample_fraction
    count = max(1, int(math.ceil(gap / max_cmd_gap)))
    return [np.asarray(start) + delta * (i / count) for i in range(1, count + 1)]


def run_chain(name, start, end, seed_q6, gripper_deg, scope):
    q_seed = seed_q6.copy()
    wps = waypoints(start, end)
    worst_transit_tilt = 0.0
    for idx, wp in enumerate(wps, start=1):
        require_tilt = scope == "all" or idx == len(wps)
        q_step, ok, pe, tl = p9._solve_q_vertical(wp, q_seed, gripper_deg, args, require_tilt=require_tilt)
        if not require_tilt:
            worst_transit_tilt = max(worst_transit_tilt, tl)
        if not ok:
            failures.append(f"{name}: wp{idx:03d}/{len(wps)} FAIL pe={pe:.3f}mm tilt={tl:.3f} require_tilt={require_tilt}")
        if idx == len(wps):
            print(f"{name}: {len(wps)} wps, arrival pe={pe:.3f}mm tilt={tl:.3f}deg, worst transit tilt={worst_transit_tilt:.2f}deg")
        q_seed = q_step
    return q_seed, len(wps)


home_q6 = np.array([*p9.HOME_ARM_DEG, 0.0, p9.Q5_OPEN_DEG])
q_after_approach, n_appr = run_chain("approach(arrival)", home_tcp, plan.approach_tcp, home_q6, p9.Q5_OPEN_DEG, "arrival")
q_after_descend, n_desc = run_chain("descend(all)", plan.approach_tcp, plan.descend_tcp, q_after_approach, p9.Q5_OPEN_DEG, "all")
q_after_lift, n_lift = run_chain("lift(all)", plan.descend_tcp, plan.lift_tcp, q_after_descend, 24.0, "all")

budget = 30 + n_appr * 60 + n_desc * 60 + 11 * 45 + 30 + n_lift * 60
print(f"worst-case budget = 30 + {n_appr}*60 + {n_desc}*60 + 11*45 + 30 + {n_lift}*60 = {budget} (limit 6000)")
if budget >= 6000:
    failures.append(f"budget {budget} >= 6000")

# Also check the PLANNED (unsettled) center path, which runs before settle replan.
planned_center = np.array([0.21369616873214542, -0.19571919576125169, p9.TABLE_Z + 0.025])
plan0 = p9._build_plan_from_center(args, planned_center, "seed0_S1_planned")
if not (plan0.approach_ik_ok and plan0.descend_ik_ok and plan0.lift_ik_ok):
    failures.append("REACH: planned-height target IK failed")
print(f"planned-height ik_ok=({plan0.approach_ik_ok},{plan0.descend_ik_ok},{plan0.lift_ik_ok})")

if failures:
    print("PREFLIGHT_FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PREFLIGHT_PASS")
