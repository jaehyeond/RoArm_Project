"""
data_training_quality_analysis.py
Comprehensive quantitative analysis of training data quality.
Analyzes: episode variety, base joint distribution, gripper timing,
elbow depth distribution, and parquet-level dataset stats.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
COLLECTED = PROJECT / "collected_data"
DATASET = PROJECT / "lerobot_dataset_v4"

# ── helpers ────────────────────────────────────────────────────────────────
def load_episode_metadata(ep_dir: Path):
    """Load metadata.json, return dict with frames list."""
    meta_path = ep_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def extract_angles_from_meta(meta: dict):
    """Return numpy array shape (N, 6): [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]."""
    frames = meta.get("frames", [])
    if not frames:
        return None
    angles = np.array([f["angles"] for f in frames], dtype=np.float32)
    return angles


def percentile_str(arr, percentiles=(5, 25, 50, 75, 95)):
    ps = np.percentile(arr, percentiles)
    return "  ".join(f"p{p}={v:.1f}" for p, v in zip(percentiles, ps))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Load all episode data from collected_data/
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: EPISODE INVENTORY")
print("=" * 70)

ep_dirs = sorted([d for d in COLLECTED.iterdir()
                  if d.is_dir() and d.name.startswith("episode_")])

print(f"Episode directories found: {len(ep_dirs)}")
print(f"Episodes: {[e.name for e in ep_dirs]}")

# Load all metadata
all_episodes = {}
for ep_dir in ep_dirs:
    meta = load_episode_metadata(ep_dir)
    if meta is not None:
        angles = extract_angles_from_meta(meta)
        if angles is not None and len(angles) > 0:
            all_episodes[ep_dir.name] = {
                "meta": meta,
                "angles": angles,
                "num_frames": len(angles),
            }

print(f"\nEpisodes with valid metadata: {len(all_episodes)}")
total_frames = sum(v["num_frames"] for v in all_episodes.values())
print(f"Total frames across all episodes: {total_frames}")
print(f"Mean frames per episode: {total_frames / len(all_episodes):.1f}")
print(f"Min frames: {min(v['num_frames'] for v in all_episodes.values())}")
print(f"Max frames: {max(v['num_frames'] for v in all_episodes.values())}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: BASE JOINT ANGLE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: BASE JOINT ANGLE DISTRIBUTION")
print("=" * 70)

# Per-episode base angle summary
base_means = []
base_mins = []
base_maxs = []

print(f"\n{'Episode':15s} {'frames':>6s}  {'base_mean':>10s}  {'base_min':>9s}  {'base_max':>9s}  {'base_range':>10s}")
print("-" * 70)
for ep_name, data in sorted(all_episodes.items()):
    angles = data["angles"]
    base = angles[:, 0]  # joint 0 = base
    mean_b = base.mean()
    min_b = base.min()
    max_b = base.max()
    range_b = max_b - min_b
    base_means.append(mean_b)
    base_mins.append(min_b)
    base_maxs.append(max_b)
    print(f"{ep_name:15s} {data['num_frames']:>6d}  {mean_b:>10.2f}  {min_b:>9.2f}  {max_b:>9.2f}  {range_b:>10.2f}")

# Global statistics
all_base = np.concatenate([v["angles"][:, 0] for v in all_episodes.values()])
print("\n--- Global base angle statistics (all frames) ---")
print(f"Mean:  {all_base.mean():.2f} deg")
print(f"Std:   {all_base.std():.2f} deg")
print(f"Min:   {all_base.min():.2f} deg")
print(f"Max:   {all_base.max():.2f} deg")
print(f"Distribution: {percentile_str(all_base)}")

# Histogram of per-episode base means
print(f"\nPer-episode mean base angle spread:")
print(f"  Min mean: {min(base_means):.2f}  Max mean: {max(base_means):.2f}  Std of means: {np.std(base_means):.2f}")

# How many episodes have base mean near 0 (within ±5 degrees)?
near_zero = sum(1 for b in base_means if abs(b) < 5)
print(f"  Episodes with |base_mean| < 5°: {near_zero}/{len(base_means)} ({100*near_zero/len(base_means):.0f}%)")
near_zero_10 = sum(1 for b in base_means if abs(b) < 10)
print(f"  Episodes with |base_mean| < 10°: {near_zero_10}/{len(base_means)} ({100*near_zero_10/len(base_means):.0f}%)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: OBJECT POSITION VARIATION INDICATORS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: OBJECT POSITION VARIATION (inferred from joint angles)")
print("=" * 70)
print("Note: If box always at same position, base+shoulder should cluster tightly at grasp moment")

# Look at per-episode final grasp configuration (last 20% of frames or when gripper closes)
print(f"\n{'Episode':15s} {'grasp_base':>10s}  {'grasp_shoulder':>14s}  {'grasp_elbow':>12s}  {'grasp_gripper':>13s}")
print("-" * 75)
grasp_bases = []
grasp_elbows = []
for ep_name, data in sorted(all_episodes.items()):
    angles = data["angles"]
    n = len(angles)
    # "grasp window" = last 30% of episode (assume grasp happens near end)
    window_start = int(n * 0.6)
    grasp_window = angles[window_start:]
    # Also find the frame where gripper is most closed (min gripper angle)
    gripper_col = angles[:, 5]
    grasp_frame_idx = np.argmin(gripper_col)  # most closed
    grasp_frame = angles[grasp_frame_idx]

    # Use grasp window mean as the grasp configuration
    gw_mean = grasp_window.mean(axis=0)
    grasp_bases.append(gw_mean[0])
    grasp_elbows.append(gw_mean[2])
    print(f"{ep_name:15s} {gw_mean[0]:>10.2f}  {gw_mean[1]:>14.2f}  {gw_mean[2]:>12.2f}  {gw_mean[5]:>13.2f}")

print(f"\nGrasp-phase base angle spread (last 40% of episodes):")
print(f"  Mean: {np.mean(grasp_bases):.2f}  Std: {np.std(grasp_bases):.2f}  Min: {np.min(grasp_bases):.2f}  Max: {np.max(grasp_bases):.2f}")
print(f"  => High std means box at varied positions; low std means fixed position")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: GRIPPER TIMING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: GRIPPER TIMING ANALYSIS")
print("=" * 70)

GRIPPER_OPEN_THRESH = 20.0   # degrees: gripper "open" if > this
GRIPPER_CLOSED_THRESH = 5.0  # degrees: gripper "closed" if < this

print(f"\nThresholds: open > {GRIPPER_OPEN_THRESH}°, closed < {GRIPPER_CLOSED_THRESH}°")

all_gripper = np.concatenate([v["angles"][:, 5] for v in all_episodes.values()])
pct_open = 100 * np.mean(all_gripper > GRIPPER_OPEN_THRESH)
pct_closed = 100 * np.mean(all_gripper < GRIPPER_CLOSED_THRESH)
pct_mid = 100 - pct_open - pct_closed
print(f"\nGlobal gripper distribution (all {len(all_gripper)} frames):")
print(f"  Open  (> {GRIPPER_OPEN_THRESH}°): {pct_open:.1f}%")
print(f"  Mid   ({GRIPPER_CLOSED_THRESH}-{GRIPPER_OPEN_THRESH}°): {pct_mid:.1f}%")
print(f"  Closed(< {GRIPPER_CLOSED_THRESH}°): {pct_closed:.1f}%")
print(f"  Gripper > 50°: {100*np.mean(all_gripper > 50):.1f}%")
print(f"  Distribution: {percentile_str(all_gripper)}")

# Per-episode gripper analysis
print(f"\n{'Episode':15s} {'max_grip':>8s}  {'pct_open>20':>11s}  {'pct_open>50':>11s}  {'open_start%':>11s}  {'open_end%':>9s}  {'opens':>6s}")
print("-" * 90)

ep_open_counts = []
ep_max_grippers = []
ep_has_open = []

for ep_name, data in sorted(all_episodes.items()):
    angles = data["angles"]
    gripper = angles[:, 5]
    n = len(gripper)

    max_grip = gripper.max()
    pct_open_20 = 100 * np.mean(gripper > GRIPPER_OPEN_THRESH)
    pct_open_50 = 100 * np.mean(gripper > 50.0)

    # Find first and last frame where gripper > 20 (open)
    open_frames = np.where(gripper > GRIPPER_OPEN_THRESH)[0]
    if len(open_frames) > 0:
        first_open_pct = 100 * open_frames[0] / n
        last_open_pct = 100 * open_frames[-1] / n
        n_open_transitions = 0
        # Count open→close transitions
        is_open = gripper > GRIPPER_OPEN_THRESH
        transitions = np.diff(is_open.astype(int))
        n_opens = np.sum(transitions == 1)
        has_open = True
    else:
        first_open_pct = float('nan')
        last_open_pct = float('nan')
        n_opens = 0
        has_open = False

    ep_max_grippers.append(max_grip)
    ep_has_open.append(has_open)
    ep_open_counts.append(n_opens)

    print(f"{ep_name:15s} {max_grip:>8.1f}  {pct_open_20:>11.1f}  {pct_open_50:>11.1f}  {first_open_pct:>11.1f}  {last_open_pct:>9.1f}  {n_opens:>6d}")

print(f"\nSummary:")
print(f"  Episodes with ANY gripper > 20°: {sum(ep_has_open)}/{len(ep_has_open)} ({100*sum(ep_has_open)/len(ep_has_open):.0f}%)")
print(f"  Episodes with max gripper > 50°: {sum(1 for g in ep_max_grippers if g > 50)}/{len(ep_max_grippers)}")
print(f"  Episodes where gripper opens (transition): {sum(1 for n in ep_open_counts if n > 0)}/{len(ep_open_counts)}")
print(f"  Mean max gripper angle: {np.mean(ep_max_grippers):.1f}°  Std: {np.std(ep_max_grippers):.1f}°")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: ELBOW DEPTH DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: ELBOW DEPTH DISTRIBUTION")
print("=" * 70)

ELBOW_DEEP_THRESH = -30.0     # degrees: deep grasp
ELBOW_APPROACH_THRESH = -10.0 # degrees: approach

all_elbow = np.concatenate([v["angles"][:, 2] for v in all_episodes.values()])
pct_deep = 100 * np.mean(all_elbow < ELBOW_DEEP_THRESH)
pct_approach = 100 * np.mean((all_elbow >= ELBOW_DEEP_THRESH) & (all_elbow < ELBOW_APPROACH_THRESH))
pct_shallow = 100 * np.mean(all_elbow >= ELBOW_APPROACH_THRESH)

print(f"\nGlobal elbow distribution (all frames):")
print(f"  DEEP     (< {ELBOW_DEEP_THRESH}°):   {pct_deep:.1f}%")
print(f"  APPROACH ({ELBOW_DEEP_THRESH} to {ELBOW_APPROACH_THRESH}°): {pct_approach:.1f}%")
print(f"  SHALLOW  (> {ELBOW_APPROACH_THRESH}°): {pct_shallow:.1f}%")
print(f"  Distribution: {percentile_str(all_elbow)}")

# Per-episode elbow classification
print(f"\n{'Episode':15s} {'min_elbow':>9s}  {'mean_elbow':>10s}  {'pct_deep':>8s}  {'class':>8s}")
print("-" * 65)
ep_classes = {"DEEP": 0, "APPROACH": 0, "SHALLOW": 0}
for ep_name, data in sorted(all_episodes.items()):
    angles = data["angles"]
    elbow = angles[:, 2]
    min_e = elbow.min()
    mean_e = elbow.mean()
    pct_d = 100 * np.mean(elbow < ELBOW_DEEP_THRESH)

    if min_e < ELBOW_DEEP_THRESH:
        cls = "DEEP"
    elif min_e < ELBOW_APPROACH_THRESH:
        cls = "APPROACH"
    else:
        cls = "SHALLOW"
    ep_classes[cls] += 1

    print(f"{ep_name:15s} {min_e:>9.2f}  {mean_e:>10.2f}  {pct_d:>8.1f}  {cls:>8s}")

print(f"\nEpisode-level classification:")
for cls, cnt in ep_classes.items():
    print(f"  {cls}: {cnt} episodes ({100*cnt/len(all_episodes):.0f}%)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: PARQUET DATASET STATS (lerobot_dataset_v4)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: LEROBOT DATASET V4 PARQUET STATS")
print("=" * 70)

try:
    import pandas as pd
    parquet_path = DATASET / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"\nParquet shape: {df.shape} (rows x cols)")
    print(f"Columns: {list(df.columns)}")
    print(f"Episodes in parquet: {df['episode_index'].nunique()}")
    print(f"Total frames in parquet: {len(df)}")

    # Action columns (joint angles)
    action_cols = [c for c in df.columns if c.startswith("action")]
    state_cols = [c for c in df.columns if c.startswith("observation.state")]
    print(f"\nAction columns: {action_cols}")
    print(f"State columns: {state_cols}")

    # Base angle (action[0])
    if "action" in df.columns:
        # action is stored as array
        actions = np.vstack(df["action"].values)
        states = np.vstack(df["observation.state"].values)
        joint_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]

        print("\n--- Parquet action statistics ---")
        print(f"{'Joint':12s}  {'mean':>7s}  {'std':>7s}  {'min':>7s}  {'max':>7s}  {'p10':>7s}  {'p90':>7s}")
        print("-" * 70)
        for i, jname in enumerate(joint_names):
            col = actions[:, i]
            print(f"{jname:12s}  {col.mean():>7.2f}  {col.std():>7.2f}  {col.min():>7.2f}  {col.max():>7.2f}  {np.percentile(col,10):>7.2f}  {np.percentile(col,90):>7.2f}")

        print("\n--- Parquet base angle (action[0]) distribution ---")
        base_a = actions[:, 0]
        print(f"  Mean: {base_a.mean():.2f}°  Std: {base_a.std():.2f}°")
        print(f"  |base| < 5°: {100*np.mean(np.abs(base_a) < 5):.1f}% of frames")
        print(f"  |base| < 10°: {100*np.mean(np.abs(base_a) < 10):.1f}% of frames")
        print(f"  |base| > 20°: {100*np.mean(np.abs(base_a) > 20):.1f}% of frames")

        print("\n--- Parquet gripper distribution ---")
        grip = actions[:, 5]
        print(f"  Mean: {grip.mean():.2f}°  Std: {grip.std():.2f}°")
        print(f"  > 20° (open): {100*np.mean(grip > 20):.1f}%")
        print(f"  > 50° (wide open): {100*np.mean(grip > 50):.1f}%")
        print(f"  < 5° (closed): {100*np.mean(grip < 5):.1f}%")

        print("\n--- Parquet elbow distribution ---")
        elbow_a = actions[:, 2]
        print(f"  Mean: {elbow_a.mean():.2f}°  Std: {elbow_a.std():.2f}°")
        print(f"  < -30° (DEEP): {100*np.mean(elbow_a < -30):.1f}%")
        print(f"  < -10° (approach+): {100*np.mean(elbow_a < -10):.1f}%")
        print(f"  > 0° (up): {100*np.mean(elbow_a > 0):.1f}%")

        # Per-episode parquet analysis
        print("\n--- Per-episode parquet base angle ---")
        print(f"{'ep_idx':>6s}  {'frames':>6s}  {'base_mean':>9s}  {'base_std':>9s}  {'elbow_min':>9s}  {'grip_max':>8s}")
        print("-" * 60)
        for ep_idx in sorted(df["episode_index"].unique()):
            ep_mask = df["episode_index"] == ep_idx
            ep_actions = actions[ep_mask.values]
            ep_frames = ep_actions.shape[0]
            b_mean = ep_actions[:, 0].mean()
            b_std = ep_actions[:, 0].std()
            e_min = ep_actions[:, 2].min()
            g_max = ep_actions[:, 5].max()
            print(f"{ep_idx:>6d}  {ep_frames:>6d}  {b_mean:>9.2f}  {b_std:>9.2f}  {e_min:>9.2f}  {g_max:>8.2f}")

except ImportError:
    print("pandas not available — skipping parquet analysis")
except Exception as e:
    print(f"Error reading parquet: {e}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: STATIC / REDUNDANT FRAME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: STATIC FRAME ANALYSIS (motion < 0.5° per frame)")
print("=" * 70)

total_static = 0
total_all = 0
STATIC_THRESH = 0.5  # degrees

for ep_name, data in sorted(all_episodes.items()):
    angles = data["angles"]
    n = len(angles)
    if n < 2:
        continue
    diffs = np.abs(np.diff(angles, axis=0))  # (N-1, 6)
    max_diff_per_frame = diffs.max(axis=1)    # (N-1,)
    n_static = np.sum(max_diff_per_frame < STATIC_THRESH)
    pct_static = 100 * n_static / (n - 1)
    total_static += n_static
    total_all += (n - 1)

print(f"\nOverall static frame fraction: {total_static}/{total_all} = {100*total_static/total_all:.1f}%")
print(f"(Static = no joint moved more than {STATIC_THRESH}° from previous frame)")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: EPISODE QUALITY SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: EPISODE QUALITY SUMMARY")
print("=" * 70)
print(f"\n{'Episode':15s} {'frames':>6s}  {'base_var':>8s}  {'elbow_min':>9s}  {'elbow_cls':>9s}  {'grip_max':>8s}  {'has_open':>8s}  {'static%':>7s}")
print("-" * 85)

for ep_name, data in sorted(all_episodes.items()):
    angles = data["angles"]
    n = len(angles)
    base_var = angles[:, 0].std()
    elbow_min = angles[:, 2].min()
    grip_max = angles[:, 5].max()
    has_open_flag = grip_max > GRIPPER_OPEN_THRESH

    if elbow_min < ELBOW_DEEP_THRESH:
        elbow_cls = "DEEP"
    elif elbow_min < ELBOW_APPROACH_THRESH:
        elbow_cls = "APPROACH"
    else:
        elbow_cls = "SHALLOW"

    if n > 1:
        diffs = np.abs(np.diff(angles, axis=0)).max(axis=1)
        pct_static_ep = 100 * np.mean(diffs < STATIC_THRESH)
    else:
        pct_static_ep = 100.0

    print(f"{ep_name:15s} {n:>6d}  {base_var:>8.2f}  {elbow_min:>9.2f}  {elbow_cls:>9s}  {grip_max:>8.1f}  {str(has_open_flag):>8s}  {pct_static_ep:>7.1f}")

print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
print(f"Total episodes (collected_data):  {len(all_episodes)}")
print(f"Total frames:                     {total_frames}")
print(f"Elbow DEEP (<-30°):               {ep_classes['DEEP']} eps ({100*ep_classes['DEEP']/len(all_episodes):.0f}%)")
print(f"Elbow APPROACH (-30 to -10°):     {ep_classes['APPROACH']} eps ({100*ep_classes['APPROACH']/len(all_episodes):.0f}%)")
print(f"Elbow SHALLOW (>-10°):            {ep_classes['SHALLOW']} eps ({100*ep_classes['SHALLOW']/len(all_episodes):.0f}%)")
has_open_count = sum(ep_has_open)
print(f"Episodes with gripper open >20°:  {has_open_count} eps ({100*has_open_count/len(ep_has_open):.0f}%)")
print(f"Global base angle std:            {all_base.std():.2f}°")
print(f"Per-episode base mean std:        {np.std(base_means):.2f}°")
print(f"Frames with |base| < 5°:          {100*np.mean(np.abs(all_base) < 5):.0f}%")
print(f"Static frame fraction:            {100*total_static/total_all:.1f}%")
