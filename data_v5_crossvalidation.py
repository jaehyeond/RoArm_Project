"""
data_v5_crossvalidation.py
V5 Dataset Cross-Validation Analysis — 136 episodes
Reads collected_data_v5/ metadata.json files for quantitative analysis
against VLA training best practices.
Output: claudedocs/DATASET_V5_ANALYSIS.md
"""

import json
import os
import sys
import math
from collections import defaultdict

BASE_DIR = "/home/cgxr/Documents/Robotics/RoArm_Project/collected_data_v5"
OUTPUT_PATH = "/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/DATASET_V5_ANALYSIS.md"

JOINT_NAMES = ["Base", "Shoulder", "Elbow", "WristP", "WristR", "Gripper"]

# ─── helper stats ────────────────────────────────────────────────────────────

def stats(vals):
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(var)
    sorted_v = sorted(vals)
    def pct(p):
        idx = int(p / 100.0 * (n - 1))
        return sorted_v[idx]
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": sorted_v[0],
        "p10": pct(10),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "max": sorted_v[-1],
        "range": sorted_v[-1] - sorted_v[0],
    }

def fmt(v):
    return f"{v:.2f}"

def fmt_s(s, key):
    return fmt(s[key]) if key in s else "N/A"

def histogram(vals, bins, lo, hi):
    """Return count per bin (equal width from lo to hi)."""
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in vals:
        idx = int((v - lo) / width)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    return counts

def ascii_bar(count, max_count, width=30):
    if max_count == 0:
        return ""
    n = int(count / max_count * width)
    return "#" * n

# ─── load all episodes ───────────────────────────────────────────────────────

print("Loading episodes...", flush=True)

episode_dirs = sorted([
    d for d in os.listdir(BASE_DIR)
    if d.startswith("episode_")
])

episodes = []
for ep_dir in episode_dirs:
    meta_path = os.path.join(BASE_DIR, ep_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"  WARNING: {ep_dir} has no metadata.json", flush=True)
        continue
    with open(meta_path) as f:
        m = json.load(f)
    episodes.append(m)

N_EPISODES = len(episodes)
print(f"Loaded {N_EPISODES} episodes", flush=True)

# ─── aggregate all frames ────────────────────────────────────────────────────

all_angles = [[] for _ in range(6)]   # per-joint angle lists (all frames)
all_deltas = [[] for _ in range(6)]   # per-joint frame-to-frame delta (abs)
total_frames = 0
static_frame_count = 0
total_transition_count = 0

episode_stats = []  # per-episode summary dicts

for m in episodes:
    frames = m.get("frames", [])
    nf = len(frames)
    if nf == 0:
        continue
    total_frames += nf

    ep_angles = [[] for _ in range(6)]
    for fr in frames:
        angs = fr["angles"]
        for j in range(6):
            ep_angles[j].append(angs[j])
            all_angles[j].append(angs[j])

    # per-episode frame-to-frame deltas
    ep_static = 0
    for i in range(1, nf):
        max_delta = 0.0
        for j in range(6):
            d = abs(ep_angles[j][i] - ep_angles[j][i - 1])
            all_deltas[j].append(d)
            if d > max_delta:
                max_delta = d
        if max_delta < 0.5:
            ep_static += 1
    total_transition_count += (nf - 1)
    static_frame_count += ep_static

    static_ratio = ep_static / (nf - 1) if nf > 1 else 0.0

    # Gripper open/close metrics
    gripper_vals = ep_angles[5]
    gripper_max = max(gripper_vals)
    gripper_min = min(gripper_vals)
    pct_open = sum(1 for v in gripper_vals if v > 40) / nf
    pct_closed = sum(1 for v in gripper_vals if v < 15) / nf

    # Starting/ending positions
    start_angles = frames[0]["angles"]
    end_angles = frames[-1]["angles"]

    episode_stats.append({
        "ep_id": m.get("episode_id"),
        "zone": m.get("zone", "UNKNOWN"),
        "object": m.get("object", "UNKNOWN"),
        "num_frames": nf,
        "duration_s": nf / m.get("fps", 30),
        "static_ratio": static_ratio,
        "gripper_max": m.get("gripper_max", gripper_max),
        "gripper_min": m.get("gripper_min", gripper_min),
        "gripper_range": m.get("gripper_range", gripper_max - gripper_min),
        "grip_open_frame": m.get("grip_open_frame"),
        "grip_close_frame": m.get("grip_close_frame"),
        "max_shoulder": m.get("max_shoulder"),
        "shoulder_at_grip_close": m.get("shoulder_at_grip_close"),
        "min_z": m.get("min_z"),
        "z_at_grip_close": m.get("z_at_grip_close"),
        "min_elbow": m.get("min_elbow"),
        "max_elbow": m.get("max_elbow"),
        "elbow_range": m.get("elbow_range"),
        "pct_open": pct_open,
        "pct_closed": pct_closed,
        "start": start_angles,
        "end": end_angles,
        "angle_means": [sum(ep_angles[j]) / nf for j in range(6)],
    })

# ─── per-joint global stats ──────────────────────────────────────────────────

joint_stats = [stats(all_angles[j]) for j in range(6)]
delta_stats = [stats(all_deltas[j]) for j in range(6)]

# ─── zone breakdown ──────────────────────────────────────────────────────────

zones = defaultdict(list)
for es in episode_stats:
    zones[es["zone"]].append(es)

zone_names = sorted(zones.keys())

# ─── start / end position variance ──────────────────────────────────────────

start_by_joint = [[es["start"][j] for es in episode_stats] for j in range(6)]
end_by_joint = [[es["end"][j] for es in episode_stats] for j in range(6)]
start_stats = [stats(start_by_joint[j]) for j in range(6)]
end_stats = [stats(end_by_joint[j]) for j in range(6)]

# ─── dataset mean (for deployment start) ────────────────────────────────────

dataset_mean = [joint_stats[j]["mean"] for j in range(6)]
dataset_std = [joint_stats[j]["std"] for j in range(6)]

# ─── zone-specific start OOD from dataset_mean ──────────────────────────────

def zone_start_ood(zone_name):
    """Max z-score of zone's mean start position vs dataset_mean."""
    es_list = zones[zone_name]
    if not es_list:
        return 0.0
    zone_mean_start = [
        sum(es["start"][j] for es in es_list) / len(es_list)
        for j in range(6)
    ]
    zscores = []
    for j in range(6):
        if dataset_std[j] > 0:
            zscores.append(abs(zone_mean_start[j] - dataset_mean[j]) / dataset_std[j])
    return max(zscores) if zscores else 0.0

# ─── gripper bimodality check ────────────────────────────────────────────────

gripper_all = all_angles[5]
n_closed = sum(1 for v in gripper_all if v < 15)
n_mid = sum(1 for v in gripper_all if 15 <= v < 40)
n_open = sum(1 for v in gripper_all if v >= 40)
n_total = len(gripper_all)

# ─── MEAN_STD normalization check ────────────────────────────────────────────

norm_issues = []
for j, jn in enumerate(JOINT_NAMES):
    if dataset_std[j] < 5.0:
        norm_issues.append(f"{jn}: std={dataset_std[j]:.2f} (very low, normalization may amplify noise)")

# ─── flag episodes with quality issues ──────────────────────────────────────

FLAG_SHORT = 50         # < 50 frames
FLAG_LONG = 400         # > 400 frames
FLAG_STATIC_HIGH = 0.6  # > 60% static frames
FLAG_GRIP_MAX_LOW = 35  # gripper never opened wide enough
FLAG_NO_GRIP_CLOSE = None  # grip_close_frame is None = no grasp detected

flagged = []
for es in episode_stats:
    flags = []
    if es["num_frames"] < FLAG_SHORT:
        flags.append(f"SHORT({es['num_frames']}f)")
    if es["num_frames"] > FLAG_LONG:
        flags.append(f"LONG({es['num_frames']}f)")
    if es["static_ratio"] > FLAG_STATIC_HIGH:
        flags.append(f"STATIC({es['static_ratio']:.0%})")
    if es["gripper_max"] is not None and es["gripper_max"] < FLAG_GRIP_MAX_LOW:
        flags.append(f"NO_OPEN(max={es['gripper_max']:.1f})")
    if es.get("grip_close_frame") is None:
        flags.append("NO_CLOSE_DETECTED")
    if flags:
        flagged.append((es["ep_id"], es["zone"], es["num_frames"], flags))

# ─── zone-level gripper and trajectory quality ──────────────────────────────

zone_summary = {}
for zn in zone_names:
    es_list = zones[zn]
    n = len(es_list)
    if n == 0:
        continue

    def zone_mean(key):
        vals = [es[key] for es in es_list if es.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def zone_vals(key):
        return [es[key] for es in es_list if es.get(key) is not None]

    grip_close_count = sum(1 for es in es_list if es.get("grip_close_frame") is not None)

    zone_summary[zn] = {
        "n": n,
        "mean_frames": zone_mean("num_frames"),
        "mean_duration": zone_mean("duration_s"),
        "mean_gripper_max": zone_mean("gripper_max"),
        "mean_gripper_range": zone_mean("gripper_range"),
        "mean_shoulder_at_close": zone_mean("shoulder_at_grip_close"),
        "mean_z_at_close": zone_mean("z_at_grip_close"),
        "mean_min_z": zone_mean("min_z"),
        "mean_static": zone_mean("static_ratio"),
        "grip_close_rate": grip_close_count / n,
        "mean_pct_open": zone_mean("pct_open"),
        "mean_pct_closed": zone_mean("pct_closed"),
    }

# ─── temporal delta distribution per joint ──────────────────────────────────

# Check for episodes with near-zero deltas (robot not moving)
per_ep_mean_delta = []
for m in episodes:
    frames = m.get("frames", [])
    nf = len(frames)
    if nf < 2:
        per_ep_mean_delta.append(0.0)
        continue
    total_d = 0.0
    for i in range(1, nf):
        d = sum(abs(frames[i]["angles"][j] - frames[i-1]["angles"][j]) for j in range(6))
        total_d += d
    per_ep_mean_delta.append(total_d / (nf - 1))

low_delta_eps = [(episodes[i].get("episode_id"), per_ep_mean_delta[i])
                 for i in range(len(episodes)) if per_ep_mean_delta[i] < 2.0]

# ─── 7-phase grasp cycle check ──────────────────────────────────────────────

# Phase criteria (from v3 requirements):
# 1. gripper_max > 40 (gripper opened wide enough)
# 2. grip_close_frame is not None (gripper closed after opening)
# 3. max_shoulder > 40 (arm descended enough)
# 4. min_z < 0 (deep grasp)

phases_ok = []
phases_fail = []
for es in episode_stats:
    fails = []
    if es["gripper_max"] is None or es["gripper_max"] < 40:
        fails.append(f"GRIP_OPEN_FAIL(max={es['gripper_max']})")
    if es.get("grip_close_frame") is None:
        fails.append("GRIP_CLOSE_FAIL")
    if es.get("max_shoulder") is None or es["max_shoulder"] < 40:
        fails.append(f"SHOULDER_FAIL(max={es.get('max_shoulder')})")
    if es.get("min_z") is None or es["min_z"] > 0:
        fails.append(f"DEPTH_FAIL(min_z={es.get('min_z')})")
    if fails:
        phases_fail.append((es["ep_id"], es["zone"], fails))
    else:
        phases_ok.append(es["ep_id"])

# ─── generate report ─────────────────────────────────────────────────────────

lines = []
A = lines.append

A("# V5 Dataset Cross-Validation Analysis")
A(f"**Date**: 2026-03-26")
A(f"**Dataset**: `collected_data_v5/`")
A(f"**Episodes analyzed**: {N_EPISODES}")
A(f"**Total frames**: {total_frames}")
A(f"**Mean frames/ep**: {total_frames/N_EPISODES:.1f}")
A(f"**Mean duration/ep**: {total_frames/N_EPISODES/30:.1f}s @ 30 FPS")
A("")

A("---")
A("")

# ── SECTION 1: Action Distribution ───────────────────────────────────────────
A("## 1. Action Distribution Analysis")
A("")
A("### 1.1 Per-Joint Global Statistics (all frames)")
A("")
A("| Joint | Mean | Std | Min | P10 | P50 | P90 | Max | Range |")
A("|-------|------|-----|-----|-----|-----|-----|-----|-------|")
for j, jn in enumerate(JOINT_NAMES):
    s = joint_stats[j]
    A(f"| {jn} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']:.2f} | "
      f"{s['p10']:.2f} | {s['p50']:.2f} | {s['p90']:.2f} | {s['max']:.2f} | {s['range']:.2f} |")
A("")

A("### 1.2 ASCII Histograms (joint angle distribution)")
A("")
HIST_BINS = 20
joint_lo = [-60, 0, 0, -30, -60, 0]
joint_hi = [60, 90, 120, 120, 60, 110]
for j, jn in enumerate(JOINT_NAMES):
    lo, hi = joint_lo[j], joint_hi[j]
    counts = histogram(all_angles[j], HIST_BINS, lo, hi)
    max_c = max(counts)
    width = (hi - lo) / HIST_BINS
    A(f"**{jn}** (range [{lo}, {hi}] deg)")
    for b, c in enumerate(counts):
        label = f"{lo + b*width:+6.1f}°"
        bar = ascii_bar(c, max_c)
        pct = 100.0 * c / len(all_angles[j])
        A(f"  {label} | {bar:<30} {c:5d} ({pct:.1f}%)")
    A("")

A("### 1.3 Bimodality / Dead Zone Assessment")
A("")
for j, jn in enumerate(JOINT_NAMES):
    s = joint_stats[j]
    # Check mean vs median gap (bimodality indicator)
    mean_med_gap = abs(s["mean"] - s["p50"])
    bimodal_flag = " [BIMODAL WARNING: mean-median gap > 10 deg]" if mean_med_gap > 10 else ""
    # Check dead zones (bins with < 0.5% of data)
    lo, hi = joint_lo[j], joint_hi[j]
    counts = histogram(all_angles[j], HIST_BINS, lo, hi)
    total_count = len(all_angles[j])
    dead_zones = [(b, lo + b*(hi-lo)/HIST_BINS) for b, c in enumerate(counts) if c < 0.005 * total_count]
    dead_str = f" | Dead zones at bins: {[f'{v:.1f}' for _, v in dead_zones[:4]]}" if dead_zones else " | No dead zones"
    A(f"- **{jn}**: mean={s['mean']:.1f}, median={s['p50']:.1f}, gap={mean_med_gap:.1f}°{bimodal_flag}{dead_str}")
A("")

A("---")
A("")

# ── SECTION 2: Start/End Position Consistency ─────────────────────────────────
A("## 2. Start/End Position Consistency")
A("")
A("### 2.1 Starting Position Statistics (frame 0 of each episode)")
A("")
A("| Joint | Mean | Std | Min | Max | Notes |")
A("|-------|------|-----|-----|-----|-------|")
for j, jn in enumerate(JOINT_NAMES):
    s = start_stats[j]
    note = ""
    if s["std"] > 15:
        note = "HIGH VAR — starts not consistent"
    elif s["std"] > 5:
        note = "moderate variance"
    else:
        note = "consistent"
    A(f"| {jn} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']:.2f} | {s['max']:.2f} | {note} |")
A("")

A("### 2.2 Ending Position Statistics (last frame of each episode)")
A("")
A("| Joint | Mean | Std | Min | Max | Notes |")
A("|-------|------|-----|-----|-----|-------|")
for j, jn in enumerate(JOINT_NAMES):
    s = end_stats[j]
    note = ""
    if s["std"] > 20:
        note = "HIGH VAR — endings not consistent"
    elif s["std"] > 8:
        note = "moderate variance"
    else:
        note = "consistent"
    A(f"| {jn} | {s['mean']:.2f} | {s['std']:.2f} | {s['min']:.2f} | {s['max']:.2f} | {note} |")
A("")

A("### 2.3 dataset_mean as Deployment Starting Position")
A("")
A(f"**dataset_mean** = [{', '.join(f'{v:.2f}' for v in dataset_mean)}]")
A(f"**dataset_std**  = [{', '.join(f'{v:.2f}' for v in dataset_std)}]")
A("")

# Compare dataset_mean to start_means
A("**Z-score of start_mean vs dataset_mean** (how OOD is init position from mean?):")
A("")
A("| Joint | start_mean | dataset_mean | Z-score | Assessment |")
A("|-------|-----------|-------------|---------|------------|")
for j, jn in enumerate(JOINT_NAMES):
    sm = start_stats[j]["mean"]
    dm = dataset_mean[j]
    ds = dataset_std[j]
    z = abs(sm - dm) / ds if ds > 0 else 0
    flag = "OK" if z < 1.5 else ("WARNING" if z < 2.5 else "SEVERE OOD")
    A(f"| {jn} | {sm:.2f} | {dm:.2f} | {z:.2f} | {flag} |")
A("")

A("### 2.4 Zone Start OOD from dataset_mean")
A("")
A("| Zone | Eps | Max Z-score vs dataset_mean | Assessment |")
A("|------|-----|----------------------------|------------|")
for zn in zone_names:
    n = len(zones[zn])
    ood = zone_start_ood(zn)
    flag = "OK" if ood < 1.5 else ("WARNING" if ood < 2.5 else "SEVERE OOD")
    A(f"| {zn} | {n} | {ood:.2f} | {flag} |")
A("")

A("---")
A("")

# ── SECTION 3: Zone-Specific Quality ─────────────────────────────────────────
A("## 3. Zone-Specific Quality")
A("")
A("### 3.1 Zone Distribution")
A("")
A("| Zone | Episodes | % of Total |")
A("|------|----------|------------|")
for zn in zone_names:
    n = len(zones[zn])
    pct = 100.0 * n / N_EPISODES
    A(f"| {zn} | {n} | {pct:.1f}% |")
A("")

A("### 3.2 Zone-Level Trajectory Quality")
A("")
header = ("| Zone | N | Frames | Dur(s) | GripMax | GripRange | "
          "Sh@Close | Z@Close | StaticRatio | GripCloseRate |")
A(header)
A("|------|---|--------|--------|---------|-----------|----------|---------|-------------|---------------|")
for zn in zone_names:
    zs = zone_summary[zn]
    def f(k):
        v = zs.get(k)
        return f"{v:.2f}" if v is not None else "N/A"
    A(f"| {zn} | {zs['n']} | {f('mean_frames')} | {f('mean_duration')} | "
      f"{f('mean_gripper_max')} | {f('mean_gripper_range')} | "
      f"{f('mean_shoulder_at_close')} | {f('mean_z_at_close')} | "
      f"{f('mean_static')} | {zs['grip_close_rate']:.2f} |")
A("")

A("### 3.3 Zone-Level Gripper Open/Close Ratio")
A("")
A("| Zone | % Open (>40°) | % Closed (<15°) | Assessment |")
A("|------|--------------|-----------------|------------|")
for zn in zone_names:
    zs = zone_summary[zn]
    po = zs.get("mean_pct_open", 0) or 0
    pc = zs.get("mean_pct_closed", 0) or 0
    if po > 0.3 and pc > 0.2:
        flag = "GOOD — clear open/close signal"
    elif po < 0.15:
        flag = "WARNING — gripper barely opens"
    elif pc < 0.15:
        flag = "WARNING — gripper barely closes"
    else:
        flag = "OK"
    A(f"| {zn} | {po:.1%} | {pc:.1%} | {flag} |")
A("")

A("### 3.4 Zone Anomalies")
A("")
for zn in zone_names:
    es_list = zones[zn]
    anomalies = []

    # Check elbow range anomalies (OVERHEAD zone might have near-zero elbow)
    low_elbow_range = [es for es in es_list if es.get("elbow_range", 99) < 5]
    if low_elbow_range:
        anomalies.append(f"{len(low_elbow_range)} eps with elbow_range < 5° (arm barely uses elbow)")

    # Check for no deep grasp
    shallow = [es for es in es_list if es.get("min_z", -999) > 50]
    if shallow:
        anomalies.append(f"{len(shallow)} eps with min_z > 50mm (NOT deep grasp)")

    # Check for very short episodes
    short = [es for es in es_list if es["num_frames"] < 60]
    if short:
        anomalies.append(f"{len(short)} eps with < 60 frames (potentially incomplete)")

    if anomalies:
        A(f"**{zn}**: {'; '.join(anomalies)}")
    else:
        A(f"**{zn}**: No anomalies detected")
A("")

A("---")
A("")

# ── SECTION 4: Temporal Quality ───────────────────────────────────────────────
A("## 4. Temporal Quality")
A("")

static_ratio_global = static_frame_count / total_transition_count if total_transition_count > 0 else 0
A(f"**Global static frame ratio**: {static_ratio_global:.1%} ({static_frame_count}/{total_transition_count} transitions with max_joint_delta < 0.5°)")
A("")

A("### 4.1 Frame-to-Frame Delta Statistics (per joint)")
A("")
A("| Joint | Mean Delta | Std Delta | P50 Delta | P90 Delta | P99 Delta |")
A("|-------|-----------|-----------|-----------|-----------|-----------|")
for j, jn in enumerate(JOINT_NAMES):
    s = delta_stats[j]
    p99_idx = int(0.99 * (len(all_deltas[j]) - 1))
    p99 = sorted(all_deltas[j])[p99_idx] if all_deltas[j] else 0
    A(f"| {jn} | {s['mean']:.3f} | {s['std']:.3f} | {s['p50']:.3f} | {s['p90']:.3f} | {p99:.3f} |")
A("")

A("### 4.2 Per-Episode Mean Total Delta (sum of abs deltas / frame)")
A("")
ep_delta_stats = stats(per_ep_mean_delta)
A(f"- Mean: {ep_delta_stats['mean']:.2f}°/frame (total joint movement)")
A(f"- Std: {ep_delta_stats['std']:.2f}")
A(f"- Min: {ep_delta_stats['min']:.2f}")
A(f"- Max: {ep_delta_stats['max']:.2f}")
A("")

if low_delta_eps:
    A("### 4.3 Episodes with Very Low Motion (mean_delta < 2°/frame)")
    A("")
    A("| Episode ID | Mean Delta |")
    A("|-----------|-----------|")
    for ep_id, md in sorted(low_delta_eps):
        A(f"| {ep_id} | {md:.3f} |")
    A("")
else:
    A("### 4.3 Episodes with Very Low Motion")
    A("")
    A("None — all episodes have sufficient motion (mean_delta >= 2°/frame)")
    A("")

A("### 4.4 Episode Duration Distribution")
A("")
ep_durations = [es["duration_s"] for es in episode_stats]
dur_stats = stats(ep_durations)
A(f"- Mean: {dur_stats['mean']:.1f}s")
A(f"- Std: {dur_stats['std']:.1f}s")
A(f"- Min: {dur_stats['min']:.1f}s")
A(f"- Max: {dur_stats['max']:.1f}s")
A(f"- P10: {dur_stats['p10']:.1f}s")
A(f"- P90: {dur_stats['p90']:.1f}s")
short_eps = sum(1 for d in ep_durations if d < 1.5)
long_eps = sum(1 for d in ep_durations if d > 15.0)
A(f"- Too short (< 1.5s): {short_eps} episodes")
A(f"- Too long (> 15.0s): {long_eps} episodes")
A("")

A("---")
A("")

# ── SECTION 5: Training Readiness ─────────────────────────────────────────────
A("## 5. Training Readiness Check")
A("")

A("### 5.1 MEAN_STD Normalization Feasibility")
A("")
A("For SmolVLA's MEAN_STD preprocessor, each joint needs sufficient std so normalization")
A("doesn't amplify noise. Rule of thumb: std > 5° = safe.")
A("")
A("| Joint | Mean | Std | Normalized Range | Assessment |")
A("|-------|------|-----|-----------------|------------|")
for j, jn in enumerate(JOINT_NAMES):
    s = joint_stats[j]
    if s["std"] > 0:
        norm_range = s["range"] / s["std"]
    else:
        norm_range = 999
    if s["std"] < 5:
        flag = "DANGER: std too low"
    elif s["std"] < 10:
        flag = "WARNING: std marginal"
    else:
        flag = "OK"
    A(f"| {jn} | {s['mean']:.2f} | {s['std']:.2f} | {norm_range:.1f}σ | {flag} |")
A("")

if norm_issues:
    A("**ISSUES FOUND:**")
    for issue in norm_issues:
        A(f"- {issue}")
else:
    A("**No normalization issues** — all joints have std > 5°")
A("")

A("### 5.2 Gripper Bimodality (critical for VLA learning)")
A("")
A("Historical issue (v3): bimodal gripper (closed + briefly open) caused mean regression to ~26°,")
A("preventing the model from learning open→grasp→close transitions.")
A("")
A(f"| State | Frame Count | % of Total |")
A(f"|-------|------------|------------|")
A(f"| Closed (< 15°) | {n_closed} | {100*n_closed/n_total:.1f}% |")
A(f"| Mid (15–40°) | {n_mid} | {100*n_mid/n_total:.1f}% |")
A(f"| Open (≥ 40°) | {n_open} | {100*n_open/n_total:.1f}% |")
A("")
if n_open / n_total < 0.20:
    A("**WARNING**: Fewer than 20% of frames have gripper open. Model may fail to learn open phase.")
elif n_open / n_total > 0.30:
    A("**GOOD**: >30% of frames have gripper open — sufficient open phase representation.")
else:
    A("**OK**: Open phase representation is acceptable (20-30%).")
A("")
if n_mid / n_total > 0.30:
    A("**WARNING**: >30% of frames are in the ambiguous mid zone (15-40°). This is the bimodal danger zone.")
else:
    A("**OK**: Mid-zone (15-40°) is below 30% of frames.")
A("")

A("### 5.3 7-Phase Grasp Cycle Completeness")
A("")
A("Each episode should complete: start → approach → open → descend → grasp → lift → return")
A("")
A(f"- Episodes passing all 4 phase criteria: {len(phases_ok)} / {N_EPISODES} ({100*len(phases_ok)/N_EPISODES:.1f}%)")
A(f"- Episodes with phase failures: {len(phases_fail)}")
A("")
if phases_fail:
    A("**Phase Failure Summary:**")
    A("")
    # Group by failure type
    fail_types = defaultdict(list)
    for ep_id, zone, fails in phases_fail:
        for f in fails:
            fail_type = f.split("(")[0]
            fail_types[fail_type].append(ep_id)
    for ft, ep_ids in sorted(fail_types.items()):
        A(f"- `{ft}`: {len(ep_ids)} episodes — IDs: {sorted(ep_ids)[:10]}{'...' if len(ep_ids) > 10 else ''}")
    A("")

A("### 5.4 Episode Quality Flags")
A("")
if flagged:
    A("| Episode ID | Zone | Frames | Flags |")
    A("|-----------|------|--------|-------|")
    for ep_id, zone, nf, flags in sorted(flagged):
        A(f"| {ep_id} | {zone} | {nf} | {', '.join(flags)} |")
else:
    A("No episodes with quality flags.")
A("")

A("### 5.5 Diversity Assessment for Multi-Zone Training")
A("")
A("For the model to generalize across zones, each zone needs:")
A("- Minimum ~20 episodes")
A("- Base joint std > 5° within zone (spatial diversity)")
A("- Consistent gripper pattern (reliable open→close sequence)")
A("")
A("| Zone | N | Enough Eps? | Grip Close Rate | Verdict |")
A("|------|---|------------|-----------------|---------|")
for zn in zone_names:
    zs = zone_summary[zn]
    n = zs["n"]
    gcr = zs["grip_close_rate"]
    enough = "YES" if n >= 20 else ("MARGINAL" if n >= 10 else "NO")
    gcr_ok = "OK" if gcr >= 0.80 else ("WARNING" if gcr >= 0.60 else "FAIL")
    if n >= 20 and gcr >= 0.80:
        verdict = "READY"
    elif n >= 10 and gcr >= 0.60:
        verdict = "MARGINAL"
    else:
        verdict = "NOT READY"
    A(f"| {zn} | {n} | {enough} | {gcr:.0%} | {verdict} |")
A("")

A("---")
A("")

# ── SECTION 6: Summary and Recommendations ───────────────────────────────────
A("## 6. Summary and Recommendations")
A("")

A("### 6.1 Key Findings")
A("")
A(f"1. **Dataset Size**: {N_EPISODES} episodes, {total_frames} frames, {total_frames/N_EPISODES:.0f} frames/ep mean")
A(f"2. **Gripper bimodality**: {100*n_open/n_total:.1f}% open, {100*n_mid/n_total:.1f}% mid, {100*n_closed/n_total:.1f}% closed")
A(f"3. **Phase completion**: {100*len(phases_ok)/N_EPISODES:.1f}% of episodes complete full grasp cycle")
A(f"4. **Static frames**: {static_ratio_global:.1%} globally")
A(f"5. **Zone coverage**: {', '.join(f'{zn}={len(zones[zn])}ep' for zn in zone_names)}")
A("")

A("### 6.2 Recommendations")
A("")

# Dynamic recommendations based on findings
recs = []

if len(phases_fail) > 0.15 * N_EPISODES:
    recs.append(f"CRITICAL: {len(phases_fail)} episodes ({100*len(phases_fail)/N_EPISODES:.1f}%) "
                f"fail the 7-phase grasp cycle check. Review and potentially re-collect.")

if n_open / n_total < 0.20:
    recs.append("CRITICAL: Gripper open ratio is below 20%. Model will struggle to learn open phase. "
                "Ensure collectors fully open gripper during approach.")

for zn in zone_names:
    zs = zone_summary[zn]
    if zs["n"] < 15:
        recs.append(f"WARNING: Zone {zn} has only {zs['n']} episodes. "
                    f"Recommend collecting to 20+ for reliable generalization.")
    if zs["grip_close_rate"] < 0.70:
        recs.append(f"WARNING: Zone {zn} grip-close detection rate = {zs['grip_close_rate']:.0%}. "
                    f"Check if grasp is being completed in these episodes.")

if static_ratio_global > 0.45:
    recs.append(f"WARNING: Global static frame ratio {static_ratio_global:.1%} is high. "
                f"Consider faster, more fluid demonstrations.")

if not norm_issues:
    recs.append("GOOD: All joint std values are adequate for MEAN_STD normalization.")

for j, jn in enumerate(JOINT_NAMES):
    s = joint_stats[j]
    if abs(s["mean"] - s["p50"]) > 15:
        recs.append(f"WARNING: {jn} has large mean-median gap ({abs(s['mean']-s['p50']):.1f}°), "
                    f"indicating strong bimodality or skew that may cause mean regression.")

if not recs:
    recs.append("Dataset looks good overall — no critical issues detected.")

for i, rec in enumerate(recs, 1):
    A(f"{i}. {rec}")

A("")
A("### 6.3 Training Readiness Score")
A("")

score = 10
score_notes = []

# Phase completion
phase_rate = len(phases_ok) / N_EPISODES
if phase_rate < 0.70:
    score -= 2
    score_notes.append(f"-2: Phase completion {phase_rate:.0%} < 70%")
elif phase_rate < 0.85:
    score -= 1
    score_notes.append(f"-1: Phase completion {phase_rate:.0%} < 85%")

# Gripper open ratio
if n_open / n_total < 0.15:
    score -= 2
    score_notes.append(f"-2: Gripper open ratio {n_open/n_total:.1%} < 15%")
elif n_open / n_total < 0.25:
    score -= 1
    score_notes.append(f"-1: Gripper open ratio {n_open/n_total:.1%} < 25%")

# Zone coverage
missing_zones = [zn for zn in zone_names if len(zones[zn]) < 10]
if missing_zones:
    score -= 1
    score_notes.append(f"-1: Zones with < 10 eps: {missing_zones}")

# Static ratio
if static_ratio_global > 0.50:
    score -= 1
    score_notes.append(f"-1: Static ratio {static_ratio_global:.1%} > 50%")

# Normalization
if norm_issues:
    score -= 1
    score_notes.append(f"-1: Normalization issues on {len(norm_issues)} joints")

# Flagged episodes
flag_rate = len(flagged) / N_EPISODES
if flag_rate > 0.10:
    score -= 1
    score_notes.append(f"-1: {len(flagged)} flagged episodes ({flag_rate:.1%})")

A(f"**Training Readiness Score: {score}/10**")
A("")
if score_notes:
    A("Deductions:")
    for note in score_notes:
        A(f"- {note}")
else:
    A("No deductions — dataset is fully ready.")
A("")

A("---")
A("")
A(f"*Generated by data_v5_crossvalidation.py on 2026-03-26*")
A(f"*Total frames analyzed: {total_frames} across {N_EPISODES} episodes*")

# ─── write output ─────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(lines))

print(f"\nReport written to: {OUTPUT_PATH}", flush=True)
print(f"Lines: {len(lines)}", flush=True)
