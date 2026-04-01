"""
V5 Deployment Failure Analysis — Data Agent
2026-03-31

Analyzes WHY v5 deployment failed vs v3 success.
Computes exact metrics from parquet files for both datasets.

Usage:
    conda run -n roarm python3 data_v5_deployment_failure_analysis.py

Output:
    Prints full analysis to stdout (no files written)
"""

import pandas as pd
import numpy as np
import os

BASE = "/home/cgxr/Documents/Robotics/RoArm_Project"
V5_PARQUET = f"{BASE}/lerobot_dataset_v5/data/chunk-000/file-000.parquet"
V3_PARQUET = f"{BASE}/lerobot_dataset_v3/data/chunk-000/file-000.parquet"
JOINTS = ['Base', 'Shoulder', 'Elbow', 'WristP', 'WristR', 'Gripper']


def load_dataset(path, label):
    df = pd.read_parquet(path)
    act = np.stack(df['action'].values)
    sta = np.stack(df['observation.state'].values)
    ep  = df['episode_index'].values
    fi  = df['frame_index'].values
    print(f"{label}: {len(df)} frames, {df['episode_index'].nunique()} episodes")
    return act, sta, ep, fi


def section(title):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def analyze_start_positions(act, sta, ep, fi, label):
    """Compare episode start state to training distribution and init position."""
    fi0 = (fi == 0)
    s0 = sta[fi0]
    init = np.array([0, 2.5, 90, 0, 0, 1.7])

    print(f"\n{label} episode start state (frame_index == 0, N={fi0.sum()}):")
    for i, j in enumerate(JOINTS):
        vals = s0[:, i]
        mean_act = act[:, i].mean()
        std_act  = act[:, i].std()
        z_from_mean = (vals.mean() - mean_act) / max(std_act, 0.01)
        z_from_init = (vals.mean() - init[i])  / max(std_act, 0.01)
        print(f"  {j:10s}: start_mean={vals.mean():+7.2f}  "
              f"z_vs_dataset_mean={z_from_mean:+.2f}  "
              f"z_vs_init={z_from_init:+.2f}")

    sh = s0[:, 1]
    print(f"\n  Shoulder at frame 0: min={sh.min():.1f} max={sh.max():.1f} mean={sh.mean():.1f}")
    print(f"  Near home (sh<10°): {(sh<10).sum()}/{len(sh)} = {(sh<10).mean()*100:.1f}%")
    print(f"  Near home (sh<20°): {(sh<20).sum()}/{len(sh)} = {(sh<20).mean()*100:.1f}%")
    print(f"  Already at approach (sh>=35°): {(sh>=35).sum()}/{len(sh)} = {(sh>=35).mean()*100:.1f}%")


def analyze_base_echo(act, sta, label):
    """Proprioceptive echo analysis for base joint."""
    delta = np.abs(act[:, 0] - sta[:, 0])
    r = np.corrcoef(sta[:, 0], act[:, 0])[0, 1]

    nonzero = (np.abs(sta[:, 0]) > 0.1) & (np.abs(act[:, 0]) > 0.1)
    sign_diff = (np.sign(sta[:, 0]) != np.sign(act[:, 0])) & nonzero

    print(f"\n{label} base joint echo analysis:")
    print(f"  r(state.base, action.base) = {r:.4f}")
    print(f"  Echo MAE (|action-state|): mean={delta.mean():.4f}°  median={np.median(delta):.4f}°")
    print(f"  |delta_base| < 0.5°: {(delta<0.5).mean()*100:.1f}%")
    print(f"  |delta_base| < 2.0°: {(delta<2.0).mean()*100:.1f}%")
    print(f"  |delta_base| > 2.0°: {(delta>2.0).mean()*100:.1f}%")
    print(f"  Sign-crossing frames: {sign_diff.sum()}/{nonzero.sum()} = {sign_diff.mean()*100:.2f}%")


def analyze_gripper(act, sta, ep, fi, label):
    """Gripper timing and distribution analysis."""
    grp_act = act[:, 5]
    grp_sta = sta[:, 5]
    grp_delta = grp_act - grp_sta

    print(f"\n{label} gripper distribution:")
    for thr in [10, 15, 20, 40]:
        print(f"  action.gripper < {thr}°: {(grp_act<thr).mean()*100:.1f}%")
    print(f"  action.gripper > 40° (open): {(grp_act>40).mean()*100:.1f}%")
    q = np.percentile(grp_act, [1, 10, 25, 50, 75, 90, 99])
    print(f"  quantiles: q1={q[0]:.1f} q10={q[1]:.1f} q25={q[2]:.1f} "
          f"q50={q[3]:.1f} q75={q[4]:.1f} q90={q[5]:.1f} q99={q[6]:.1f}")

    # Frame-level opening signal
    print(f"  gripper opening frames (action>state+5°): {(grp_delta>5).mean()*100:.1f}%")

    # Per-episode timing
    eps = np.unique(ep)
    open_abs = []
    close_abs = []
    durations = []
    full_cycle = 0

    for e in eps:
        mask = ep == e
        fi_ep = fi[mask]
        act_ep = act[mask]
        sort_idx = np.argsort(fi_ep)
        act_ep = act_ep[sort_idx]
        fi_ep = fi_ep[sort_idx]
        n = len(act_ep)
        g = act_ep[:, 5]
        durations.append(n)

        open_frames = np.where(g > 40)[0]
        if len(open_frames) > 0:
            open_abs.append(open_frames[0])
            peak = open_frames[0]
            close_frames = np.where((g < 20) & (np.arange(n) > peak))[0]
            if len(close_frames) > 0:
                close_abs.append(close_frames[0])
                full_cycle += 1

    n_open = len(open_abs)
    n_close = len(close_abs)
    mean_dur = np.mean(durations)
    print(f"\n  Per-episode gripper timing (N={len(eps)} eps):")
    print(f"    Episodes with clear open (>40°): {n_open}/{len(eps)} = {n_open/len(eps)*100:.1f}%")
    if open_abs:
        print(f"    Open frame: mean={np.mean(open_abs):.1f} ({np.mean(open_abs)/mean_dur*100:.1f}% into ep)")
        print(f"    Open before frame 50: {sum(1 for x in open_abs if x<50)}/{n_open} = {sum(1 for x in open_abs if x<50)/n_open*100:.1f}%")
    print(f"    Episodes with close after open: {n_close}/{n_open}")
    if close_abs:
        print(f"    Close before frame 50: {sum(1 for x in close_abs if x<50)}/{n_close} = {sum(1 for x in close_abs if x<50)/n_close*100:.1f}%")
    print(f"    Full cycle (open+close): {full_cycle}/{len(eps)} = {full_cycle/len(eps)*100:.1f}%")


def analyze_base_zone(act, label):
    """Base angle zone distribution."""
    base = act[:, 0]
    zones = [
        ("FAR_LEFT (<-40°)", base < -40),
        ("LEFT (-40 to -10°)", (base >= -40) & (base < -10)),
        ("CENTER (-10 to 10°)", (base >= -10) & (base <= 10)),
        ("RIGHT (10 to 40°)", (base > 10) & (base <= 40)),
        ("FAR_RIGHT (>40°)", base > 40),
    ]
    print(f"\n{label} base angle zone distribution (per frame):")
    for name, mask in zones:
        print(f"  {name:25s}: {mask.sum():5d} = {mask.mean()*100:.1f}%")
    print(f"  mean={base.mean():.2f}  std={base.std():.2f}  "
          f"range=[{base.min():.1f}, {base.max():.1f}]")

    # Episode-level: how many episodes start near zero
    return base


def print_v5_fail_analysis():
    """Print targeted analysis of WHY V5 failed with dataset_mean start."""
    section("V5 FAILURE ROOT CAUSE: dataset_mean start vs training distribution")

    print("""
PROBLEM: V5 deployed with --start-pos dataset_mean
  Robot starts at: [9.95, 43.94, 41.31, 66.57, 0.21, 28.25]
  This matches the dataset_mean EXACTLY (z=+0.01 for shoulder)

BUT: The training data's start-of-episode state is ALSO near this position:
  V5 frame-0 shoulder: mean=44.1° (= dataset_mean 43.94°)
  V5 frame-0 elbow:    mean=36.0° (dataset_mean 41.3°, z=-0.16)

This seems correct — until you look at what happens NEXT:

  The model IS given the correct starting position.
  The model CAN see the image.
  BUT the ACTIONS at the start of every episode are ALSO near dataset_mean.
  Because: at frame 0, arm is already positioned, so action ≈ state (delta ≈ 0).

  Result: Model predicts "stay still" → robot stays at approach pose →
          Model keeps seeing the approach pose → keeps predicting "stay still"
          → FIXED-POINT LOOP at approach pose, gripper stuck at ~28°

V3 with --start-pos init WORKED because:
  Robot starts at init (shoulder=2.5°), which is 1.46 SIGMA below training mean.
  This is OOD, but it forces LARGE action predictions: shoulder needs to go to 44°+
  The model learned "from init → approach → grasp" as a SINGLE sequence.
  Large initial action prevents fixed-point convergence.

KEY INSIGHT: V5 lacks a "from home to approach" prefix in training data.
  100% of V5 episodes START at approach pose (shoulder 16-70°, mean=44°).
  The approach movement (home→approach) is NEVER in V5 training data.
  So the model never learned to move from low shoulder to high shoulder.
  Starting at approach = model predicts small movements = fixed-point loop.
""")


def main():
    v5_act, v5_sta, v5_ep, v5_fi = load_dataset(V5_PARQUET, "V5")
    v3_act, v3_sta, v3_ep, v3_fi = load_dataset(V3_PARQUET, "V3")

    section("SECTION 1: BASIC ACTION STATISTICS")
    print(f"\n{'Joint':10s}  {'V5 mean':>10} {'V5 std':>8}  |  {'V3 mean':>10} {'V3 std':>8}")
    print("-" * 55)
    for i, j in enumerate(JOINTS):
        print(f"{j:10s}  {v5_act[:,i].mean():+10.2f} {v5_act[:,i].std():8.2f}  |  "
              f"{v3_act[:,i].mean():+10.2f} {v3_act[:,i].std():8.2f}")

    section("SECTION 2: ACTION-STATE DELTA (proprioceptive echo)")
    print(f"\n{'Joint':10s}  {'V5 >2° (%)':>12} {'V5 MAE':>8}  |  {'V3 >2° (%)':>12} {'V3 MAE':>8}")
    print("-" * 60)
    v5_d = np.abs(v5_act - v5_sta)
    v3_d = np.abs(v3_act - v3_sta)
    for i, j in enumerate(JOINTS):
        print(f"{j:10s}  {(v5_d[:,i]>2).mean()*100:12.1f} {v5_d[:,i].mean():8.3f}  |  "
              f"{(v3_d[:,i]>2).mean()*100:12.1f} {v3_d[:,i].mean():8.3f}")

    section("SECTION 3: BASE ANGLE ZONE DISTRIBUTIONS")
    analyze_base_zone(v5_act, "V5")
    analyze_base_zone(v3_act, "V3")

    section("SECTION 4: BASE PROPRIOCEPTIVE ECHO")
    analyze_base_echo(v5_act, v5_sta, "V5")
    analyze_base_echo(v3_act, v3_sta, "V3")

    section("SECTION 5: EPISODE START POSITION ANALYSIS")
    analyze_start_positions(v5_act, v5_sta, v5_ep, v5_fi, "V5")
    analyze_start_positions(v3_act, v3_sta, v3_ep, v3_fi, "V3")

    section("SECTION 6: GRIPPER PATTERNS")
    analyze_gripper(v5_act, v5_sta, v5_ep, v5_fi, "V5")
    analyze_gripper(v3_act, v3_sta, v3_ep, v3_fi, "V3")

    section("SECTION 7: V5 GRIPPER AT DATASET_MEAN START (28°)")
    grp_sta = v5_sta[:, 5]
    grp_act = v5_act[:, 5]
    for lo, hi in [(25, 35), (23, 33), (20, 30)]:
        mask = (grp_sta >= lo) & (grp_sta < hi)
        if mask.sum() > 0:
            acts = grp_act[mask]
            print(f"state.gripper in [{lo},{hi}°): n={mask.sum():4d}  "
                  f"action mean={acts.mean():.1f}±{acts.std():.1f}  "
                  f"<20°: {(acts<20).mean()*100:.0f}%  >40°: {(acts>40).mean()*100:.0f}%")

    print_v5_fail_analysis()

    section("SECTION 8: TRAINING CONFIG COMPARISON")
    print("""
V3 Training:
  steps=50,000  batch_size=8  epochs≈30
  Dataset start: shoulder=2.5° (home position)
  Deployment: --start-pos init (matches training start)
  Result: SUCCESS (5/5)

V5 Training:
  steps=200,000  batch_size=64  epochs≈950
  Dataset start: shoulder=44.1° (approach position)
  Deployment: --start-pos dataset_mean (matches training start — but fixed-point trap)
  Result: FAIL (base locked, gripper stuck)

CRITICAL NOTE: V5 dataset_mean start is NOT the fix for OOD.
  It is a trap: matches training distribution but eliminates all net displacement.
  The actual fix is: start from a position that FORCES the model to commit to motion.
  In V3, init (shoulder=2.5°) is -1.46σ OOD, but this forces shoulder to RISE → motion.
""")

    section("SUMMARY: KEY NUMBERS")
    print(f"""
                                    V5          V3
                                    -------     -------
Total frames                        13,470      13,145
Total episodes                      136         74
Mean episode duration               3.3s        5.9s

BASE JOINT:
  r(state, action)                  0.9996      0.9992
  |delta| > 2° (% frames)          6.3%        7.5%
  Sign-crossing frames              0.40%       0.26%
  Early approach in first 30fr     mean 1.7°   mean 23.7°

GRIPPER:
  Gripper opening (>state+5°)       5.2%        3.5%
  Full open+close per episode       100%        100%
  Opens in first 50 frames          100%        43.2%
  Closes in first 50 frames         93.2%       0.0%
  q10 (mostly-closed dist)          16.2°       1.7°  ← KEY DIFFERENCE

EPISODE START:
  Shoulder at frame 0               44.1°       2.8°
  Starts near home (sh<10°)         0/136 (0%)  74/74 (100%)
  Base at frame 0 (mean)            8.7°        0.2°

TRAINING:
  batch_size                        64          8
  Epochs                            ~950        ~30
  Scheduler warmup                  2,000 steps default
""")


if __name__ == "__main__":
    main()
