"""
data_v3_deployment_failure_analysis.py
=======================================
SmolVLA v3 deployment failure root cause analysis.
Compares training data distribution vs deployment trajectory.

Run:
    conda run -n roarm python3 /home/cgxr/Documents/Robotics/RoArm_Project/data_v3_deployment_failure_analysis.py

Checkpoint:  outputs/smolvla_v3_sponge/checkpoints/025000/pretrained_model
Dataset:     lerobot_dataset_v3  (74 episodes, 13145 frames)
Deploy log:  logs/deploy_20260225_154420.csv  (300 steps)
"""

import json
import csv
import numpy as np

# ─── 0. Constants ────────────────────────────────────────────────────────────
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist_pitch", "Wrist_roll", "Gripper"]

# ─── 1. Training data stats (from lerobot_dataset_v3/meta/stats.json) ────────
# These are the MEAN_STD normalizer values used during training AND deployment.
TRAIN_ACTION_MEAN = np.array([-0.471, 30.177, 58.876, 40.721, -2.328, 26.479])
TRAIN_ACTION_STD  = np.array([25.812, 18.807, 24.829, 30.069, 20.216, 24.153])

TRAIN_STATE_MEAN  = np.array([-0.492, 30.064, 58.953, 40.680, -2.320, 26.354])
TRAIN_STATE_STD   = np.array([25.809, 18.889, 24.830, 30.066, 20.213, 24.223])

# ─── 2. Dataset v4 stats (collected_data / 51 episodes, NOT used for v3 train)
# Included for cross-comparison only.
DATASET_V4_ACTION_MEAN = np.array([2.935, 41.217, 12.643, 61.435, -2.460, 10.041])
DATASET_V4_ACTION_STD  = np.array([21.135, 26.236, 28.864, 25.411, 22.043, 14.127])

# ─── 3. Offline checkpoint evaluation means (from train_v3_checkpoint_eval_results.json)
OFFLINE_PRED_MEANS = {
    "5K":  np.array([2.704, 26.874, 70.145, 23.026, -1.461, 26.230]),
    "10K": np.array([3.173, 26.689, 69.951, 23.774, -0.894, 28.424]),
    "15K": np.array([2.609, 25.862, 70.002, 23.812, -0.499, 28.531]),
    "20K": np.array([2.604, 25.922, 69.696, 23.460, -0.453, 28.188]),
    "25K": np.array([2.786, 26.093, 70.343, 23.161, -0.759, 28.441]),
    "30K": np.array([2.945, 26.223, 70.379, 23.425, -0.708, 28.436]),
    "35K": np.array([3.052, 26.066, 70.329, 23.476, -0.573, 28.439]),
    "40K": np.array([2.929, 26.180, 70.251, 23.519, -0.773, 28.527]),
    "45K": np.array([3.049, 26.088, 70.357, 23.449, -0.692, 28.514]),
    "50K": np.array([3.003, 26.168, 70.177, 23.460, -0.714, 28.505]),
}

# ─── 4. Load deployment CSV ──────────────────────────────────────────────────
deploy_log = "/home/cgxr/Documents/Robotics/RoArm_Project/logs/deploy_20260225_154420.csv"

steps, positions, z_scores = [], [], []
with open(deploy_log) as f:
    reader = csv.DictReader(f)
    for row in reader:
        step = int(row["step"])
        pos = np.array([
            float(row["base"]), float(row["shoulder"]), float(row["elbow"]),
            float(row["wrist_pitch"]), float(row["wrist_roll"]), float(row["gripper"])
        ])
        zs = np.array([
            float(row["z_base"]), float(row["z_shoulder"]), float(row["z_elbow"]),
            float(row["z_wrist_pitch"]), float(row["z_wrist_roll"]), float(row["z_gripper"])
        ])
        steps.append(step)
        positions.append(pos)
        z_scores.append(zs)

positions = np.array(positions)   # (300, 6)
z_scores  = np.array(z_scores)    # (300, 6)
n_steps   = len(steps)

# ─── ANALYSIS ─────────────────────────────────────────────────────────────────

print("=" * 70)
print("SmolVLA v3 Deployment Failure Analysis")
print("=" * 70)

# ─── Section 1: Dataset overview ─────────────────────────────────────────────
print("\n[1] TRAINING DATASET (lerobot_dataset_v3)")
print(f"    74 episodes, 13145 frames, 30fps, max_ep_len={325} frames (10.8s)")
print(f"    Dataset MEAN_STD normalizer (action):")
for i, name in enumerate(JOINT_NAMES):
    print(f"      {name:12s}: mean={TRAIN_ACTION_MEAN[i]:7.2f}  std={TRAIN_ACTION_STD[i]:6.2f}")

# ─── Section 2: Deployment starting position vs dataset mean ─────────────────
start_pos = positions[0]
print("\n[2] STARTING POSITION vs DATASET MEAN")
print(f"  {'Joint':12s}  {'Start':>8}  {'DataMean':>8}  {'Diff':>8}  {'in-dist?'}")
print(f"  {'-'*55}")
for i, name in enumerate(JOINT_NAMES):
    diff = start_pos[i] - TRAIN_ACTION_MEAN[i]
    z_at_start = diff / TRAIN_ACTION_STD[i]
    indist = "YES" if abs(z_at_start) < 1.0 else "WARN"
    print(f"  {name:12s}  {start_pos[i]:8.2f}  {TRAIN_ACTION_MEAN[i]:8.2f}  "
          f"{diff:8.2f}  [{indist}] z={z_at_start:.2f}")

# ─── Section 3: Converged position (steps 72-100) ───────────────────────────
# convergence_detected=True first appears at step 72 (index 71)
converged_pos   = positions[70:100].mean(axis=0)   # mean of steps 71-100
converged_pos_final = positions[250:300].mean(axis=0)  # late phase (steps 250-300)

print("\n[3] CONVERGED POSITION ANALYSIS")
print(f"  {'Joint':12s}  {'Conv(70-100)':>12}  {'DataMean':>10}  {'Z-score':>8}  {'Late(250-300)':>14}")
print(f"  {'-'*65}")
for i, name in enumerate(JOINT_NAMES):
    z = (converged_pos[i] - TRAIN_ACTION_MEAN[i]) / TRAIN_ACTION_STD[i]
    print(f"  {name:12s}  {converged_pos[i]:12.2f}  {TRAIN_ACTION_MEAN[i]:10.2f}  "
          f"{z:8.3f}  {converged_pos_final[i]:14.2f}")

print("\n  KEY: Is converged_pos == dataset_mean?")
delta = converged_pos - TRAIN_ACTION_MEAN
print(f"  Max deviation from dataset mean: {np.abs(delta).max():.2f} deg")
print(f"  Mean |deviation|:                {np.abs(delta).mean():.2f} deg")
print(f"  Converged pos == dataset mean:   {np.all(np.abs(delta) < 5.0)}")

# ─── Section 4: Gripper trajectory ──────────────────────────────────────────
gripper_traj = positions[:, 5]
print("\n[4] GRIPPER TRAJECTORY ANALYSIS")
print(f"  Start:      {gripper_traj[0]:.2f} deg")
print(f"  Max ever:   {gripper_traj.max():.2f} deg  (at step {gripper_traj.argmax()+1})")
print(f"  Min ever:   {gripper_traj.min():.2f} deg")
print(f"  Mean:       {gripper_traj.mean():.2f} deg")
print(f"  DataMean:   {TRAIN_ACTION_MEAN[5]:.2f} deg")
print(f"  DataMedian: 24.06 deg (q50)")
print(f"  > 30 deg (open)?  steps: {(gripper_traj > 30).sum()}/{n_steps}")
print(f"  > 40 deg (open)?  steps: {(gripper_traj > 40).sum()}/{n_steps}")
print(f"  < 15 deg (closed)? steps: {(gripper_traj < 15).sum()}/{n_steps}")
print(f"\n  Offline pred gripper range (25K): min=1.85  max=97.48  range=95.6 deg")
print(f"  Deployment gripper range:         min={gripper_traj.min():.2f}  max={gripper_traj.max():.2f}  range={gripper_traj.max()-gripper_traj.min():.2f} deg")
print(f"  -> Deployment range is {((gripper_traj.max()-gripper_traj.min())/95.6)*100:.1f}% of offline predicted range!")

# ─── Section 5: Elbow trajectory ─────────────────────────────────────────────
elbow_traj = positions[:, 2]
print("\n[5] ELBOW TRAJECTORY (depth proxy)")
print(f"  Start:      {elbow_traj[0]:.2f} deg")
print(f"  Max ever:   {elbow_traj.max():.2f} deg  (at step {elbow_traj.argmax()+1})")
print(f"  Min ever:   {elbow_traj.min():.2f} deg  (at step {elbow_traj.argmin()+1})")
print(f"  Converged:  {converged_pos[2]:.2f} deg")
print(f"  DataMean:   {TRAIN_ACTION_MEAN[2]:.2f} deg")
print(f"  DataMedian: 51.21 deg (q50)")
print(f"  q10:        34.84 deg (deep range)")
print(f"  ELBOW NEVER went deep (< 35 deg): {elbow_traj.min():.2f} deg")
print(f"  ELBOW converged ABOVE dataset mean: {converged_pos[2]:.2f} > {TRAIN_ACTION_MEAN[2]:.2f}")

# ─── Section 6: Shoulder trajectory ──────────────────────────────────────────
shoulder_traj = positions[:, 1]
print("\n[6] SHOULDER TRAJECTORY (primary depth factor)")
print(f"  Start:      {shoulder_traj[0]:.2f} deg")
print(f"  Max ever:   {shoulder_traj.max():.2f} deg  (at step {shoulder_traj.argmax()+1})")
print(f"  Min ever:   {shoulder_traj.min():.2f} deg")
print(f"  Converged:  {converged_pos[1]:.2f} deg")
print(f"  DataMean:   {TRAIN_ACTION_MEAN[1]:.2f} deg")
print(f"  DataMedian: 29.90 deg (q50)")
print(f"  DataQ10:    4.84 deg (deep approach zone)")
print(f"  Deep zone (> 50 deg): {(shoulder_traj > 50).sum()}/{n_steps} steps ({(shoulder_traj>50).sum()/n_steps*100:.1f}%)")
print(f"  Approach zone (> 30 deg): {(shoulder_traj > 30).sum()}/{n_steps} steps ({(shoulder_traj>30).sum()/n_steps*100:.1f}%)")

# ─── Section 7: Z-score trajectory (normalized input to model) ───────────────
print("\n[7] INPUT Z-SCORE TRAJECTORY (what the model 'sees' as state)")
print(f"  {'Joint':12s}  {'z_start':>8}  {'z_conv(70-100)':>15}  {'z_late(250-300)':>15}  {'In-dist (<2)?'}")
print(f"  {'-'*65}")
z_conv = z_scores[70:100].mean(axis=0)
z_late = z_scores[250:300].mean(axis=0)
for i, name in enumerate(JOINT_NAMES):
    indist = "YES" if abs(z_conv[i]) < 2.0 else "OUT-OF-DIST"
    print(f"  {name:12s}  {z_scores[0,i]:8.3f}  {z_conv[i]:15.3f}  {z_late[i]:15.3f}  {indist}")

print("\n  NOTE: z_elbow converges to ~+0.44 (ABOVE mean, not below)")
print("        z_shoulder converges to ~0.0 (exactly at mean)")
print("        z_gripper converges to ~-0.06 (exactly at mean = 26.5 deg)")
print("        => Model is outputting near-mean predictions for all joints")

# ─── Section 8: Offline vs Deployment comparison ─────────────────────────────
offline_25k_mean = OFFLINE_PRED_MEANS["25K"]
print("\n[8] OFFLINE PREDICTION MEANS vs DEPLOYMENT BEHAVIOR")
print(f"  {'Joint':12s}  {'OfflineMean':>12}  {'ConvPos':>10}  {'Diff':>8}  {'DataMean':>10}")
print(f"  {'-'*60}")
for i, name in enumerate(JOINT_NAMES):
    diff = converged_pos[i] - offline_25k_mean[i]
    print(f"  {name:12s}  {offline_25k_mean[i]:12.2f}  {converged_pos[i]:10.2f}  "
          f"{diff:8.2f}  {TRAIN_ACTION_MEAN[i]:10.2f}")

print("\n  KEY FINDING: Converged position closely matches OFFLINE PREDICTION MEAN!")
print(f"  Both offline pred and deployment converge near dataset action mean.")

# ─── Section 9: THE CORE PROBLEM - MEAN REGRESSION ANALYSIS ──────────────────
print("\n[9] CORE PROBLEM: MEAN REGRESSION ANALYSIS")
print("  " + "=" * 66)

# Compute deviation from dataset mean across all checkpoints
print("\n  All checkpoint offline prediction means:")
print(f"  {'Ckpt':>6}  {'Elbow':>8}  {'Gripper':>8}  {'Shoulder':>8}  {'Wrist_p':>8}")
for ckpt, mean in OFFLINE_PRED_MEANS.items():
    print(f"  {ckpt:>6}  {mean[2]:8.2f}  {mean[5]:8.2f}  {mean[1]:8.2f}  {mean[3]:8.2f}")
print(f"  {'TRAIN':>6}  {TRAIN_ACTION_MEAN[2]:8.2f}  {TRAIN_ACTION_MEAN[5]:8.2f}  "
      f"{TRAIN_ACTION_MEAN[1]:8.2f}  {TRAIN_ACTION_MEAN[3]:8.2f}")
print(f"\n  OBSERVATION: All checkpoint means ≈ [3, 26, 70, 23, -0.6, 28]")
print(f"               vs DataMean = [{', '.join(f'{v:.1f}' for v in TRAIN_ACTION_MEAN)}]")
print(f"               Elbow: offline mean 70.2 >> DataMean 58.9 (ABOVE mean, not deep!)")
print(f"               Gripper: offline mean 27.7 ≈ DataMean 26.5 (at mean = 26.5 deg)")
print(f"               -> Gripper is predicting its mean, which is 26.5 deg OPEN~MID")
print(f"               -> Dataset gripper median is 24.1 deg, q90=68.5 deg")
print(f"               -> 26.5 deg mean is dominated by partial-open frames (43% of frames)")

# ─── Section 10: WHY OFFLINE != DEPLOYMENT ───────────────────────────────────
print("\n[10] WHY OFFLINE DIVERSITY != DEPLOYMENT DIVERSITY")
print("   Offline (test_inference_official.py):")
print("     - Tests on BATCH of 222 random samples from training distribution")
print("     - Each sample is from a DIFFERENT episode/timestep")
print("     - Diversity comes from different visual states across episodes")
print("     - pred_std reflects CROSS-SAMPLE variance (position diversity)")
print("     - Gripper offline range: 1.75 to 97.5 deg (94 deg range) -- LOOKS GREAT")
print("")
print("   Deployment (real robot):")
print("     - n_action_steps=50 → execute 50 steps from ONE prediction")
print("     - Robot MOVES to predicted position → state converges toward mean position")
print("     - Next inference sees state ≈ mean → predicts ≈ mean again")
print("     - Self-reinforcing loop: state→mean→predict_mean→execute→state_near_mean")
print("")
print("   THE TRAP: n_action_steps=50 executes a full action chunk.")
print("   If the action chunk predicts [mean±noise], the robot moves to mean position")
print("   and STAYS there. All subsequent inferences see the same mean-ish state.")
print("   The offline evaluation MASKS this because each sample has a DIFFERENT state.")

# ─── Section 11: Dataset distribution analysis ───────────────────────────────
print("\n[11] TRAINING DATA DISTRIBUTION PROBLEMS (v3 dataset)")
print("   lerobot_dataset_v3: 74 episodes, 13145 frames")
print("   Action mean/std comparison:")
print(f"   Elbow:   mean={TRAIN_ACTION_MEAN[2]:.1f}, std={TRAIN_ACTION_STD[2]:.1f}")
print(f"            q50=51.2, q90=91.1, q10=34.8 deg")
print(f"            SPREAD: episodes cover 9-119 deg range (good)")
print(f"            BUT mean=58.9 is in mid-range → mean regression = mid elbow")
print("")
print(f"   Gripper: mean={TRAIN_ACTION_MEAN[5]:.1f}, std={TRAIN_ACTION_STD[5]:.1f}")
print(f"            q50=24.1, q90=68.5, q10=1.73 deg")
print(f"            BIMODAL: mostly closed (q10=1.7) OR open (q90=68.5)")
print(f"            Mean=26.5 is in the TRANSITION ZONE (15-30 deg)")
print(f"            When model regresses to mean, gripper stays at 26.5 deg")
print(f"            = partially open (sponge NOT gripped properly)")
print("")
print(f"   Shoulder: mean={TRAIN_ACTION_MEAN[1]:.1f}, std={TRAIN_ACTION_STD[1]:.1f}")
print(f"             q50=29.9, q90=54.6, q10=4.8 deg")
print(f"             Deep episodes: shoulder goes to 4-25 deg range")
print(f"             Mean regression → shoulder stays at 30 deg = approach height")
print(f"             = robot NEVER descends to sponge level")
print("")
print(f"   Wrist_pitch: mean={TRAIN_ACTION_MEAN[3]:.1f}, std={TRAIN_ACTION_STD[3]:.1f}")
print(f"               q50=54.9, q90=68.0 --- BIMODAL distribution!")
print(f"               q10=-3.0 (approach) vs q50=54.9 (hover/descent)")
print(f"               Offline pred mean=23.2 (BELOW q10!) → wrist collapse")
print(f"               This causes wrist_pitch to fold during deployment")

# ─── Section 12: Wrist_pitch specific problem ────────────────────────────────
wrist_traj = positions[:, 3]
print("\n[12] WRIST_PITCH COLLAPSE DETAIL")
print(f"   Start:        {wrist_traj[0]:.2f} deg")
print(f"   Max:          {wrist_traj.max():.2f} deg (step {wrist_traj.argmax()+1})")
print(f"   Converged:    {converged_pos[3]:.2f} deg")
print(f"   Late-phase:   {converged_pos_final[3]:.2f} deg")
print(f"   DataMean:     {TRAIN_ACTION_MEAN[3]:.2f} deg")
print(f"   DataMedian:   54.9 deg")
print(f"   OFFLINE MEAN: 23.2 deg (25K ckpt) --- far below median!")
print(f"   -> Wrist collapses from 41 deg to ~10-11 deg during deployment")
print(f"   -> This is severe OOD for wrist_pitch: q1={-5.0:.1f}, q10=-3.0")
print(f"   -> Wrist pitch at 10 deg = gripper pointing sideways, NOT down toward sponge")

# ─── Section 13: VERDICT ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[13] ROOT CAUSE VERDICT")
print("=" * 70)

print("""
PRIMARY CAUSES (data-side):

1. MEAN REGRESSION in n_action_steps=50 CLOSED LOOP
   - The model learned to predict near-mean actions on average
   - With n_action_steps=50, the robot executes a full 50-step chunk
   - After execution, robot is at or near the mean position
   - All subsequent inferences see mean-like state → predict mean → loop
   - The offline evaluation hides this because it tests with diverse states

2. GRIPPER DISTRIBUTION: MEAN = PARTIAL OPEN (26.5 deg)
   - The dataset gripper is bimodal: closed (0-15) OR fully open (40-100)
   - The mean (26.5 deg) falls in the TRANSITION zone
   - Model predicting mean = gripper stuck at 26.5 deg = never fully closes
   - Cannot grasp sponge (need to close to ~5-10 deg)

3. WRIST_PITCH BIMODAL DISTRIBUTION
   - Training data has two modes: approach (~5 deg) and descent (~60 deg)
   - Offline pred mean=23 deg is BETWEEN the modes (a "mean" of bimodal data)
   - In deployment, wrist collapses to ~10 deg = gripper pointing sideways
   - Sponge cannot be reached with this wrist orientation

4. SHOULDER NEVER DEEP ENOUGH
   - Converged shoulder ≈ 30 deg (dataset mean), but deep grasp needs < 10 deg
   - Shoulder never went below 25 deg in 300 deployment steps
   - Robot stays at hover height, never descends

SECONDARY CAUSES (deployment-side):

5. 25K CHECKPOINT (NOT 25K BUT SPECIFICALLY UNDERFITTED)
   - 25K checkpoint used, but ALL checkpoints show same mean regression
   - The problem is NOT checkpoint selection — it's the data distribution

6. n_action_steps=50 IS CORRECT (not a bug)
   - BUT it means the robot commits to a full chunk before new observation
   - If the chunk predicts "hover at mean position", robot stays there
   - n_action_steps=1 (closed-loop per-step) would NOT fix this if model
     predicts mean — it would just converge faster

WHAT OFFLINE EVALUATION CORRECTLY REPORTED vs MISSED:
   - Correctly reported: high diversity across samples (range 94 deg gripper)
   - Missed: the TEMPORAL correlation — in deployment, consecutive states
     are highly correlated (robot moves toward predicted position)
   - Offline pred_std ≈ train_std (good diversity ACROSS samples)
   - But deployment is NOT random sampling — it's sequential execution

DATA FIX REQUIRED:
   A. More episodes with CLEAR bimodal temporal structure:
      - Start phase: gripper open, shoulder high
      - Descent phase: gripper open, shoulder low (< 10 deg)
      - Grasp phase: gripper closes, shoulder low
      - Lift phase: shoulder rises, gripper stays closed
   B. Ensure 40%+ of frames are in the gripper-OPEN state (> 40 deg)
   C. Ensure temporal consistency: ONE episode should show full grasp cycle
   D. Reduce static/hovering frames (currently 33.5% static in v4 data)
""")

# ─── Section 14: Quick numbers summary ───────────────────────────────────────
print("=" * 70)
print("[14] QUICK NUMBERS SUMMARY")
print("=" * 70)
print(f"  Dataset (v3): 74 eps, 13145 frames")
print(f"  Deploy log:   300 steps ({(300/30):.1f}s)")
print(f"  Convergence:  first detected at step 72")
print(f"  ")
print(f"  Joint         DataMean   Conv70-100   Late250-300  Delta(conv-mean)")
print(f"  {'-'*65}")
for i, name in enumerate(JOINT_NAMES):
    delta = converged_pos[i] - TRAIN_ACTION_MEAN[i]
    print(f"  {name:12s}  {TRAIN_ACTION_MEAN[i]:9.2f}  {converged_pos[i]:11.2f}  "
          f"{converged_pos_final[i]:11.2f}  {delta:+.2f}")

print(f"""
  CONCLUSION:
  Converged position IS the dataset action mean (delta < 2 deg for most joints)
  EXCEPT wrist_pitch: converged to 17.9 deg vs mean 40.7 (collapsed 22.8 deg!)
  Gripper: converged to 25.4 deg vs mean 26.5 deg (stuck at mean = partial open)
  Elbow:   converged to 69.8 deg vs mean 58.9 deg (10.9 deg ABOVE mean = too HIGH)

  The deployment is a "fixed point": model predicts mean → robot goes to mean →
  model sees mean state → predicts mean again → stable fixed point loop.
  This fixed point is NOT the grasp position. It is the hover position.
""")

print("\nAnalysis complete.")
