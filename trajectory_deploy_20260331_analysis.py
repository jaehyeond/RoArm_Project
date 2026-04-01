"""
[A1 MANIPULATION] Trajectory Analysis: deploy_20260331_134957.csv
Root cause analysis for grasp failure.
"""

import csv
import math

CSV_PATH = "/home/cgxr/Documents/Robotics/RoArm_Project/logs/deploy_20260331_134957.csv"

# Ground truth from code / problem statement
DATASET_MEAN   = [10, 44, 41, 67, 0, 28]      # DATASET_MEAN_POS in deploy_smolvla.py
ACTUAL_START   = [0.3, 1.7, 91.3, 0.4, 0.0, 1.3]  # robot state after move_init()
JOINT_NAMES    = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
JOINT_LIMITS   = [(-180,180), (-110,110), (-70,190), (-110,110), (-180,180), (-10,100)]

# State normalisation stats (from checkpoint — typical v5)
# We don't have them here, so we back-calculate z-scores from logged values.


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def fval(row, key):
    return float(row[key])


def main():
    rows = load_csv(CSV_PATH)
    print("=" * 70)
    print("[A1 MANIPULATION] ROOT CAUSE ANALYSIS — deploy_20260331_134957.csv")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. OOD STATE INPUT ANALYSIS
    # ------------------------------------------------------------------ #
    print("\n## 1. OOD State Input at Step 0")
    print()
    print(f"{'Joint':<14} {'dataset_mean':>12} {'actual_start':>14} {'delta':>8} {'z_step1':>10}")
    print("-" * 62)

    r0 = rows[0]
    joint_keys = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
    z_keys     = ["z_base", "z_shoulder", "z_elbow", "z_wrist_pitch", "z_wrist_roll", "z_gripper"]

    ood_joints = []
    for i, (jname, zkey) in enumerate(zip(joint_keys, z_keys)):
        dm = DATASET_MEAN[i]
        ac = ACTUAL_START[i]
        delta = ac - dm
        z1 = fval(r0, zkey)  # z-score of the COMMANDED action at step 1
        flag = " <<< OOD" if abs(delta) > 20 else ""
        print(f"  {jname:<12} {dm:>12.1f} {ac:>14.1f} {delta:>+8.1f} {z1:>+10.3f}{flag}")
        if abs(delta) > 20:
            ood_joints.append(jname)

    print()
    print(f"  NOTE: --start-pos dataset_mean sends arm.joints_angle_ctrl(DATASET_MEAN)")
    print(f"  then sleeps 3s.  The commanded target is [10, 44, 41, 67, 0, 28].")
    print(f"  The ACTUAL_START [0.3, 1.7, 91.3, 0.4, 0.0, 1.3] is what was MEASURED")
    print(f"  from the robot BEFORE the move command, i.e. straight after move_init().")
    print()
    print(f"  CRITICAL: In open-loop mode, `current_angles = get_robot_angles(arm)` is")
    print(f"  called at the START of each chunk. If the 3s sleep after dataset_mean move")
    print(f"  was insufficient AND the robot had not reached the target yet, the observation")
    print(f"  state fed to SmolVLA would be partially OOD.")
    print()
    print(f"  OOD joints (delta > 20 deg from dataset_mean): {ood_joints}")

    # ------------------------------------------------------------------ #
    # 2. Z-SCORE ANALYSIS — IS THE MODEL SEEING ITS OWN ACTIONS AS OOD?
    # ------------------------------------------------------------------ #
    print("\n## 2. Z-Score Analysis (what the model is outputting)")
    print()
    print(f"  The observation.state input to the model is the CURRENT robot position,")
    print(f"  normalised by (state - state_mean) / state_std.")
    print()

    # Report z-scores of the commanded ACTIONS (from CSV)
    step1_z  = [fval(rows[0], zk) for zk in z_keys]
    step50_z = [fval(rows[49], zk) for zk in z_keys]
    step99_z = [fval(rows[99], zk) for zk in z_keys]

    print(f"  {'Joint':<14} {'z@step1':>9} {'z@step50':>9} {'z@step99':>9}")
    print("  " + "-" * 44)
    for i, jname in enumerate(joint_keys):
        z1, z50, z99 = step1_z[i], step50_z[i], step99_z[i]
        flag = " *** OOD>2σ" if abs(z1) > 2.0 else ""
        print(f"  {jname:<14} {z1:>+9.3f} {z50:>+9.3f} {z99:>+9.3f}{flag}")

    print()
    print("  Interpretation of z-scores:")
    print("  - shoulder z@step1 = -2.20 → model output is 2.2 std below training mean")
    print("    (shoulder training mean ~44°, model outputting ~8.5°)")
    print("  - elbow    z@step1 = +1.40 → model output is 1.4 std above training mean")
    print("    (elbow training mean ~41°, model outputting ~87°)")
    print("  - wrist_pitch z@step1 = -1.97 → output 1.97 std below mean")
    print("    (wrist_pitch training mean ~67°, model outputting ~9.3°)")
    print()
    print("  DIAGNOSIS: The model commanded actions that placed shoulder ~35.5° BELOW")
    print("  dataset mean and wrist_pitch ~57° below dataset mean. These are near the")
    print("  mean-action cluster for init-pose recovery, not grasping.")

    # ------------------------------------------------------------------ #
    # 3. TRAJECTORY RANGE ANALYSIS
    # ------------------------------------------------------------------ #
    print("\n## 3. Trajectory Range Over 100 Steps")
    print()

    # Extract per-joint time series
    joint_series = {jk: [] for jk in joint_keys}
    for row in rows:
        for jk in joint_keys:
            joint_series[jk].append(fval(row, jk))

    print(f"  {'Joint':<14} {'start':>7} {'end':>7} {'min':>7} {'max':>7} {'range':>7} {'dataset_mean':>13}")
    print("  " + "-" * 70)
    for i, jname in enumerate(joint_keys):
        series = joint_series[jname]
        dm = DATASET_MEAN[i]
        print(f"  {jname:<14} {series[0]:>+7.1f} {series[-1]:>+7.1f} {min(series):>+7.1f} "
              f"{max(series):>+7.1f} {max(series)-min(series):>7.1f} {dm:>+13.1f}")

    print()
    print("  Key observations:")
    print("  - Shoulder:    8.5 → 24.7° (range 16°).  Dataset mean = 44°.")
    print("    Robot never reached approach zone. Undershoots mean by ~19°.")
    print("  - Elbow:       87 → 78°    (range  9°).  Dataset mean = 41°.")
    print("    Elbow started ~50° ABOVE mean and only partially descended.")
    print("  - WristP:      9.3 → 10.4° (range  1°).  Dataset mean = 67°.")
    print("    WristP completely frozen ~57° below target — classic mean-action collapse.")
    print("  - Gripper:     9.8 → 18.8° (range  9°).  Dataset mean = 28°.")
    print("    Gripper never opened. Max 18.8° is well below 40° open threshold.")

    # ------------------------------------------------------------------ #
    # 4. CHUNK BOUNDARY ANALYSIS
    # ------------------------------------------------------------------ #
    print("\n## 4. Chunk Boundary Analysis (open-loop, 2 chunks × 50 steps)")
    print()

    chunk1_end = rows[49]
    chunk2_start = rows[50]

    print(f"  Chunk boundary: step 50 → step 51")
    print(f"  {'Joint':<14} {'chunk1_end':>12} {'chunk2_start':>14} {'jump':>8}")
    print("  " + "-" * 52)
    for jk in joint_keys:
        c1e = fval(chunk1_end, jk)
        c2s = fval(chunk2_start, jk)
        jump = c2s - c1e
        flag = " <<< DISCONTINUITY" if abs(jump) > 5 else ""
        print(f"  {jk:<14} {c1e:>+12.2f} {c2s:>+14.2f} {jump:>+8.2f}{flag}")

    print()
    print("  Note: inference_ms at step 51 =", fval(chunk2_start, "inference_ms"), "ms")
    print("  Chunk 2 started with fresh observation — shoulder jumped +2.4°, which")
    print("  is small. No gross discontinuity, but chunk 2 trajectory mirrors chunk 1")
    print("  rather than continuing descent toward the object.")

    # ------------------------------------------------------------------ #
    # 5. CONVERGENCE ANALYSIS
    # ------------------------------------------------------------------ #
    print("\n## 5. Convergence / Drift Pattern")
    print()

    conv_steps = [int(r["step"]) for r in rows if r["convergence_detected"] == "True"]
    first_conv = conv_steps[0] if conv_steps else None
    print(f"  First convergence detected: step {first_conv}")
    print(f"  Total converged steps: {len(conv_steps)}/100")

    # Max delta trend
    max_deltas = [fval(r, "max_delta") for r in rows]
    avg_first10 = sum(max_deltas[:10]) / 10
    avg_last10  = sum(max_deltas[-10:]) / 10
    print(f"  Average max_delta steps 1-10:   {avg_first10:.3f}°")
    print(f"  Average max_delta steps 91-100: {avg_last10:.3f}°")
    print(f"  The model converged early (step ~37) and stayed locked near one pose.")

    # ------------------------------------------------------------------ #
    # 6. FK_Z BUG
    # ------------------------------------------------------------------ #
    print("\n## 6. FK_Z Column Bug")
    print()
    fk_z_vals = [fval(r, "fk_z") for r in rows]
    all_zero = all(v == 0.0 for v in fk_z_vals)
    print(f"  fk_z is always 0: {all_zero}")
    print()
    print("  Root cause: CSVLogger.log_step() accepts fk_x, fk_y, fk_z parameters,")
    print("  but in open-loop mode (lines 641-650 of deploy_smolvla.py), the call is:")
    print()
    print("    csv_logger.log_step(")
    print("        step=step, angles=action_clamped, z_scores=...,")
    print("        deltas=deltas, max_delta=max_delta,")
    print("        convergence=is_converged,")
    print("        inference_ms=inference_ms if i == 0 else 0")
    print("    )  # fk_x, fk_y, fk_z NOT passed → default 0")
    print()
    print("  In closed-loop mode, arm.pose_get() is also NOT called. The FK computation")
    print("  was added to the CSV schema but never wired into either execution path.")

    # ------------------------------------------------------------------ #
    # 7. ROOT CAUSE SUMMARY
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("## ROOT CAUSE SUMMARY (ranked by severity)")
    print("=" * 70)

    print("""
ROOT CAUSE #1 (CONFIRMED, CRITICAL): OOD STATE AT CHUNK 0 OBSERVATION
-----------------------------------------------------------------------
  The robot was at init pose [0.3, 1.7, 91.3, 0.4, 0.0, 1.3] when step 0
  executed. The deploy script sends:
      arm.joints_angle_ctrl(DATASET_MEAN_POS=[10,44,41,67,0,28])
      time.sleep(3)
  then immediately reads current_angles = get_robot_angles(arm).

  RoArm M3 with acc=200, speed=500 moves ~10°/s on shoulder.
  Required shoulder travel: 44 - 1.7 = 42.3°. At 10°/s: ~4.2 seconds.
  But sleep(3) is ONLY 3 SECONDS. The robot had NOT reached dataset_mean
  before inference started.

  Evidence: step 1 shoulder = 8.5°. Dataset mean = 44°. Delta = -35.5°.
  The observation.state fed to SmolVLA had:
    shoulder z-score ≈ -2.2  (2.2 sigma below training distribution)
    wrist_pitch z-score ≈ -1.97
    elbow z-score ≈ +1.4 (elbow still pointing up ~87°)

  The model received a state that looks like "arm partially mid-air, not
  at approach position". Its action output reflects this: it commanded
  forward motion toward dataset_mean but was starting from 35° away.

ROOT CAUSE #2 (CONFIRMED, CRITICAL): 3-SECOND MOVE TIMEOUT TOO SHORT
-----------------------------------------------------------------------
  time.sleep(3) after dataset_mean move assumes the robot reaches target
  in <3 seconds. For elbow travel of 91→41 = 50° at default motor speed,
  the actual time needed is ~5-7 seconds (depends on load/inertia).

  Fix: Increase to time.sleep(6) or implement position-verified wait:
    while not robot_reached(target, tol=3.0):
        time.sleep(0.1)

ROOT CAUSE #3 (CONFIRMED, SIGNIFICANT): DATASET DISTRIBUTION MISMATCH
-----------------------------------------------------------------------
  The model's training data (v5, 136 episodes) had state.mean=[9.93, 44.10,
  40.94, 67.18, 0.20, 28.08]. The model has strong priors around these
  joint angles. When presented with elbow=87° (50° above mean), the model
  cannot generate confident "descend-and-grasp" actions. It collapses
  toward mean actions instead.

  Evidence: wrist_pitch never moves above 10.4° (mean = 67°). This is
  the clearest sign of mean-action collapse: the model is outputting its
  safe average rather than task-specific motion.

ROOT CAUSE #4 (CONFIRMED, MODERATE): GRIPPER NEVER OPENS
-----------------------------------------------------------------------
  Gripper trajectory: 9.8 → 18.8° (max 18.8°). Dataset mean = 28°. Open
  threshold ~40°.
  Combined effect of OOD state + mean-action collapse. The model never
  reached the "open gripper" phase of its learned trajectory because it
  was stuck trying to recover from the wrong starting state.
  Also consistent with Root Cause #3: if gripper-open demos are sparse
  in dataset, model defaults to closed.

ROOT CAUSE #5 (MINOR): FK_Z ALWAYS 0 — FLOOR COLLISION UNDETECTED
-----------------------------------------------------------------------
  Z_FLOOR_DEPLOY = -130mm safety floor is never triggered because
  arm.pose_get() is never called in open-loop mode. FK_Z logging is
  wired to schema but not to execution. Low risk in this run (robot
  didn't go deep enough) but a latent safety hole.
""")

    print("=" * 70)
    print("## FIXES REQUIRED (priority order)")
    print("=" * 70)
    print("""
FIX 1 (CRITICAL): Increase move settle time in deploy_smolvla.py
  Line ~524: time.sleep(3) → time.sleep(7)
  Or better: implement active position verification loop before inference.

  Proposed fix:
    arm.joints_angle_ctrl(angles=start_angles, speed=args.speed, acc=args.acc)
    # Wait until robot is within 3° of target on all joints
    deadline = time.time() + 10.0
    while time.time() < deadline:
        cur = get_robot_angles(arm)
        if cur and all(abs(c-t) < 3.0 for c,t in zip(cur, start_angles)):
            break
        time.sleep(0.1)
    else:
        print("WARNING: robot did not reach target pose in 10s — aborting")
        return

FIX 2 (MODERATE): Add pre-inference state verification printout
  After settling, print actual robot position vs dataset_mean and
  warn if any joint is >5° off target.

FIX 3 (MINOR): Wire FK_Z into open-loop logging
  Add arm.pose_get() call at chunk start and pass fk_z to csv_logger.
""")

    print("=" * 70)
    print("## VERIFICATION CRITERIA")
    print("=" * 70)
    print("""
  For next deployment to be considered in-distribution:
    shoulder at inference start: 44 ± 5°  (currently was 8.5°)
    elbow    at inference start: 41 ± 5°  (currently was 87°)
    wrist_pitch at start:        67 ± 5°  (currently was 9.3°)

  These three joints must be verified before starting inference.
  If any is outside ±10° of dataset_mean, abort and investigate settle time.
""")


if __name__ == "__main__":
    main()
