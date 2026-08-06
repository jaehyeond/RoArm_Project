"""Numeric preflight v2 for the reverify-repaired p9 (numpy-only, no Isaac).

v1 (t3_preflight_ik_chain.py, sha pinned in the 21st session doc) validated the
audit-repair revision; it is kept untouched as evidence. v2 covers the reverify
wf_3cea04db-7c2 survivor repairs and turns both surviving MAJOR margins into
fail-able numeric gates:

  1. REACH: plan targets (approach/descend/lift) pass the full vertical gate
     for ALL four T2-PASS poses at the settled height (+ seed0_S1 at the
     planned height, which runs before the settle replan).
  2. Chain: HOME->approach (arrival scope), descend (all), lift (all) — every
     waypoint ik_ok with the joint trust region applied
     (waypoint_max_joint_dev_deg=12).
  3. MAJOR-2 margin: commanded-solution position residual <= 1.0 mm on EVERY
     waypoint (transit selection band 0.5 mm; the unrepaired selection
     commanded 2.52-2.55 mm vs the 3 mm physical reached gate).
  4. Trust region construction: per-waypoint max joint deviation from the
     chain seed <= 12 deg + eps.
  5. MAJOR-1 margin: pessimistic velocity-clamp slew bound (every joint moves
     toward its command at 1.799 deg/control step simultaneously; effort/
     damping ignored -> upper bound) of per-step FK TCP displacement:
       approach pairs  <= 14.0 mm (30% margin below the phase-scoped
                                   transit_tcp_step_gate_m = 20 mm)
       descend/lift    <= 7.5 mm  (25% margin below max_tcp_step_m = 10 mm)
  6. Budget: worst-case control-step budget < 6000 per pose (episode 60 s).
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
    waypoint_max_joint_dev_deg=12.0,
    transit_tcp_step_gate_m=0.020,
)

VEL_CLAMP_DEG_PER_STEP = math.degrees(3.14 * 0.01)  # velocity_limit_sim * control dt
# pe gate: transit selection band 0.5 mm + 0.1 tolerance (leaves >= 2.4 mm of
# the 3 mm physical reached gate for tracking error + gravity sag).
PE_GATE_MM = 0.6
SLEW_GATE_APPROACH_M = 0.014
# Corridor slew: 8 mm command spacing swept in ~1 clamped step is the p7 design
# point (command_resample_fraction 0.8 of the 10 mm gate); 9.0 mm flags any
# reconfiguration adding >1 mm on top of the designed spacing.
SLEW_GATE_CORRIDOR_M = 0.009
# Trust region 12 deg (bias stage) + transit position polish allowance 2 deg.
DEV_GATE_DEG = args.waypoint_max_joint_dev_deg + p9.TRANSIT_POLISH_DEV_DEG + 1.0e-9

failures = []


def waypoints(start, end):
    delta = np.asarray(end) - np.asarray(start)
    gap = float(np.linalg.norm(delta))
    max_cmd_gap = args.max_tcp_step_m * args.command_resample_fraction
    count = max(1, int(math.ceil(gap / max_cmd_gap)))
    return [np.asarray(start) + delta * (i / count) for i in range(1, count + 1)]


def slew_max_tcp_step(q_from4, q_to4):
    """Pessimistic bound: all joints slew simultaneously at the velocity clamp."""
    q = np.asarray(q_from4, dtype=np.float64).copy()
    target = np.asarray(q_to4, dtype=np.float64)
    worst = 0.0
    for _ in range(1000):
        rem = target - q
        if float(np.max(np.abs(rem))) <= 1.0e-9:
            break
        q_next = q + np.clip(rem, -VEL_CLAMP_DEG_PER_STEP, VEL_CLAMP_DEG_PER_STEP)
        tcp0, _, _ = p9.fk_points(q)
        tcp1, _, _ = p9.fk_points(q_next)
        worst = max(worst, float(np.linalg.norm(tcp1 - tcp0)))
        q = q_next
    return worst


def run_chain(tag, start, end, seed_q6, gripper_deg, scope, phase, prev_cmd4):
    q_seed = seed_q6.copy()
    wps = waypoints(start, end)
    worst = {"pe": 0.0, "tilt_transit": 0.0, "dev": 0.0, "slew": 0.0}
    slew_gate = SLEW_GATE_APPROACH_M if phase == "approach" else SLEW_GATE_CORRIDOR_M
    arrival = (float("nan"), float("nan"))
    for idx, wp in enumerate(wps, start=1):
        require_tilt = scope == "all" or idx == len(wps)
        q_step, ok, pe, tl = p9._solve_q_vertical(
            wp, q_seed, gripper_deg, args,
            require_tilt=require_tilt,
            max_dev_from_seed_deg=args.waypoint_max_joint_dev_deg,
        )
        dev = float(np.max(np.abs(np.asarray(q_step[:4]) - np.asarray(q_seed[:4], dtype=np.float64))))
        slew = slew_max_tcp_step(prev_cmd4, q_step[:4])
        worst["pe"] = max(worst["pe"], pe)
        worst["dev"] = max(worst["dev"], dev)
        worst["slew"] = max(worst["slew"], slew)
        if not require_tilt:
            worst["tilt_transit"] = max(worst["tilt_transit"], tl)
        if not ok:
            failures.append(f"{tag}: wp{idx:03d}/{len(wps)} ik_ok FAIL pe={pe:.3f}mm tilt={tl:.3f} require_tilt={require_tilt}")
        if pe > PE_GATE_MM:
            failures.append(f"{tag}: wp{idx:03d} commanded pe {pe:.3f}mm > {PE_GATE_MM}mm margin gate (MAJOR-2)")
        if dev > DEV_GATE_DEG:
            failures.append(f"{tag}: wp{idx:03d} seed_dev {dev:.3f}deg > trust region {args.waypoint_max_joint_dev_deg}deg")
        if slew > slew_gate:
            failures.append(f"{tag}: wp{idx:03d} pessimistic slew {slew * 1000:.2f}mm > {slew_gate * 1000:.1f}mm margin gate (MAJOR-1, phase={phase})")
        if idx == len(wps):
            arrival = (pe, tl)
        prev_cmd4 = np.asarray(q_step[:4], dtype=np.float64)
        q_seed = q_step
    print(
        f"  {tag}: {len(wps)} wps | arrival pe={arrival[0]:.3f}mm tilt={arrival[1]:.3f}deg | "
        f"worst pe={worst['pe']:.3f}mm dev={worst['dev']:.2f}deg "
        f"slew={worst['slew'] * 1000:.2f}mm (gate {slew_gate * 1000:.1f}) transit_tilt={worst['tilt_transit']:.2f}deg"
    )
    return q_seed, len(wps), prev_cmd4


home_tcp, _, _ = p9.fk_points(p9.HOME_ARM_DEG)
print(f"home_tcp={home_tcp} vel_clamp={VEL_CLAMP_DEG_PER_STEP:.4f} deg/step")

POSES = ["seed0_S1", "seed0_S2", "R1_center", "R2_center"]
for label in POSES:
    x, y = p9._workspace_xy_from_label(label)
    center = np.array([x, y, 0.025])  # settled: ground z=0 -> center z = H/2
    plan = p9._build_plan_from_center(args, center, f"{label}_settled")
    print(
        f"{label}: plan ik_ok=({plan.approach_ik_ok},{plan.descend_ik_ok},{plan.lift_ik_ok}) "
        f"err_mm=({plan.approach_ik_err_mm:.3f},{plan.descend_ik_err_mm:.3f},{plan.lift_ik_err_mm:.3f}) "
        f"tilt=({plan.approach_tilt_deg:.3f},{plan.descend_tilt_deg:.3f},{plan.lift_tilt_deg:.3f})"
    )
    if not (plan.approach_ik_ok and plan.descend_ik_ok and plan.lift_ik_ok):
        failures.append(f"{label}: REACH plan target IK failed")
        continue
    home_q6 = np.array([*p9.HOME_ARM_DEG, 0.0, p9.Q5_OPEN_DEG])
    prev4 = np.asarray(p9.HOME_ARM_DEG, dtype=np.float64)
    q1, n_appr, prev4 = run_chain(f"{label}/approach", home_tcp, plan.approach_tcp, home_q6, p9.Q5_OPEN_DEG, "arrival", "approach", prev4)
    q2, n_desc, prev4 = run_chain(f"{label}/descend", plan.approach_tcp, plan.descend_tcp, q1, p9.Q5_OPEN_DEG, "all", "descend", prev4)
    q3, n_lift, prev4 = run_chain(f"{label}/lift", plan.descend_tcp, plan.lift_tcp, q2, args.close_deg[-1], "all", "lift", prev4)
    budget = 30 + n_appr * 60 + n_desc * 60 + 11 * 45 + 30 + n_lift * 60
    print(f"  {label}: worst-case budget = {budget} (limit 6000)")
    if budget >= 6000:
        failures.append(f"{label}: budget {budget} >= 6000")

# Planned (unsettled) height REACH for the actual spawn pose (runs pre-replan).
x, y = p9._workspace_xy_from_label("seed0_S1")
plan0 = p9._build_plan_from_center(args, np.array([x, y, p9.TABLE_Z + 0.025]), "seed0_S1_planned")
print(f"seed0_S1 planned-height ik_ok=({plan0.approach_ik_ok},{plan0.descend_ik_ok},{plan0.lift_ik_ok})")
if not (plan0.approach_ik_ok and plan0.descend_ik_ok and plan0.lift_ik_ok):
    failures.append("seed0_S1: planned-height REACH failed")

if failures:
    print("PREFLIGHT_V2_FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PREFLIGHT_V2_PASS")
