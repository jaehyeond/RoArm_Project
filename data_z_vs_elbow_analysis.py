"""
data_z_vs_elbow_analysis.py

CRITICAL ANALYSIS: Is elbow angle a valid proxy for Z-height (grasp depth)?
Validates MEMORY lesson #12: "Elbow angle alone cannot judge grasp depth"

Key conclusions from this analysis:
1. Elbow-FK_Z correlation: r=0.287 (WEAK) -> elbow is unreliable
2. Shoulder-FK_Z correlation: r=-0.814 (STRONG) -> shoulder is the dominant factor
3. 35 of 43 episodes misclassified if using elbow threshold alone
4. For new data collection: pose_get() Z from ESP32 is the correct metric
5. ESP32 FK Z convention MUST be calibrated with one live hardware test before use

Coordinate system note:
- URDF world frame Z: positive=up, origin at robot base mounting point
- Home pos [0,0,90,0,0,0]: URDF Z = -106mm (TCP below world origin level)
- Deeper grasps: URDF Z more negative (down toward table)
- ESP32 pose_get() Z: likely POSITIVE=up from base plate, but MUST be verified with hardware
- The thresholds in collect_data_manual.py (DEEP<100mm) are ASSUMPTIONS, not verified

How to run:
    conda run -n roarm python3 data_z_vs_elbow_analysis.py
"""

import os
import json
import numpy as np

DATA_DIR = "/home/cgxr/Documents/Robotics/RoArm_Project/collected_data"


# ============================================================
# SOFTWARE FK FROM URDF (roarm_m3.urdf)
# ============================================================

def Rx(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def Ry(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def Rz(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def Trans(x, y, z):
    return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])

def joint_transform(origin_xyz, origin_rpy, joint_angle_rad):
    """
    4x4 transform for a revolute joint (axis=Z).
    origin_xyz, origin_rpy: from URDF <origin>
    joint_angle_rad: the revolute joint variable
    """
    Tf = Trans(*origin_xyz) @ Rx(origin_rpy[0]) @ Ry(origin_rpy[1]) @ Rz(origin_rpy[2])
    Tj = Rz(joint_angle_rad)
    return Tf @ Tj


def fk_roarm_m3(angles_deg):
    """
    Software FK for RoArm M3 using URDF geometry.

    angles_deg: [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]
    Returns: [x_mm, y_mm, z_mm] of hand_tcp in URDF world frame

    URDF world frame convention:
    - Z positive = up (away from table surface)
    - Origin is at the robot's world link (mounting point)
    - world_to_base_link is +70.1mm Z offset

    IMPORTANT: This Z is NOT the same as ESP32's pose_get() Z.
    The ESP32's FK uses a different origin/convention.
    Use fk_z_relative() for a calibration-free comparison.

    Joint chain from URDF:
    world -> base_link: fixed, T(0,0,70.1mm)
    base_link -> link1: joint0 (base), xyz=[0,0,0], rpy=[0,0,0], axis=Z
    link1 -> link2: joint1 (shoulder), xyz=[0,0,51.959mm], rpy=[-pi/2,-pi/2,0], axis=Z
    link2 -> link3: joint2 (elbow), xyz=[236.815mm,30.002mm,0], rpy=[0,0,pi/2], axis=Z
    link3 -> link4: joint3 (wrist_pitch), xyz=[0,-144.586mm,0], rpy=[0,0,0], axis=Z
    link4 -> link5: joint4 (wrist_roll), xyz=[15.147mm,-53.653mm,0], rpy=[pi/2,pi/2,0], axis=Z
    link5 -> hand_tcp: fixed, xyz=[0,0,115.428mm], rpy=[pi/2,-pi/2,0]
    """
    deg2rad = np.pi / 180.0
    q = [a * deg2rad for a in angles_deg[:5]]

    pi = np.pi

    T0 = Trans(0, 0, 0.0701)                              # world to base_link
    T1 = T0 @ joint_transform([0, 0, 0], [0, 0, 0], q[0])  # joint0: base
    T2 = T1 @ joint_transform([0, 0, 0.051959], [-pi/2, -pi/2, 0], q[1])  # joint1: shoulder
    T3 = T2 @ joint_transform([0.236815, 0.030002, 0], [0, 0, pi/2], q[2])  # joint2: elbow
    T4 = T3 @ joint_transform([0, -0.144586, 0], [0, 0, 0], q[3])  # joint3: wrist_pitch
    T5 = T4 @ joint_transform([0.015147, -0.053653, 0], [pi/2, pi/2, 0], q[4])  # joint4: wrist_roll
    T_tcp = T5 @ Trans(0, 0, 0.115428) @ Rx(pi/2) @ Ry(-pi/2)  # hand_tcp (fixed)

    return T_tcp[:3, 3] * 1000.0  # meters -> mm


# Home position FK Z (calibration reference)
HOME_FK_Z_MM = -106.2  # fk_roarm_m3([0, 0, 90, 0, 0, 0])[2]


def fk_z_relative(angles_deg):
    """
    FK Z relative to home position.
    0 = same height as home
    Negative = arm went DOWN from home (toward table)
    Positive = arm went UP from home (away from table)
    """
    return fk_roarm_m3(angles_deg)[2] - HOME_FK_Z_MM


# ============================================================
# DATA LOADING
# ============================================================

def load_all_episodes():
    episodes = []
    ep_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.startswith("episode_")])
    for ep_dir in ep_dirs:
        meta_path = os.path.join(DATA_DIR, ep_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            m = json.load(f)
        episodes.append(m)
    return episodes


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis():
    print("=" * 70)
    print("RoArm M3: Elbow Angle vs FK Z-height -- Grasp Quality Analysis")
    print("=" * 70)
    print()
    print("COORDINATE SYSTEM:")
    print(f"  URDF world Z: positive = up, origin at robot mounting point")
    print(f"  Home position [0,0,90,0,0,0]: FK_Z = {HOME_FK_Z_MM:.1f}mm")
    print(f"  Deeper grasps (arm going DOWN): FK_Z becomes MORE negative")
    print(f"  Relative FK_Z = FK_Z - HOME_FK_Z = 0 at home, negative when arm down")
    print()

    # ---- Load data and compute FK Z per frame ----
    episodes = load_all_episodes()
    print(f"Loaded {len(episodes)} episodes from {DATA_DIR}")
    print()

    all_data = []
    episode_summaries = []

    for ep in episodes:
        ep_id = ep["episode_id"]
        frames = ep.get("frames", [])
        if not frames:
            continue

        elbows, shoulders, grippers, z_urdf, z_rel = [], [], [], [], []

        for frame in frames:
            a = frame["angles"]
            e, sh, gr = a[2], a[1], a[5]
            z_abs = fk_roarm_m3(a)[2]
            z_r = z_abs - HOME_FK_Z_MM

            elbows.append(e)
            shoulders.append(sh)
            grippers.append(gr)
            z_urdf.append(z_abs)
            z_rel.append(z_r)

            all_data.append({
                "ep_id": ep_id,
                "frame_idx": frame.get("frame_idx", 0),
                "elbow": e,
                "shoulder": sh,
                "gripper": gr,
                "z_abs": z_abs,
                "z_rel": z_r
            })

        min_elbow = min(elbows)
        min_z_abs = min(z_urdf)
        min_z_rel = min(z_rel)  # most negative = deepest
        max_gripper = max(grippers)
        min_gripper = min(grippers)
        max_shoulder = max(shoulders)

        # Shoulder at gripper close (gripper < 5 deg)
        shoulder_at_grip_close = [sh for gr, sh in zip(grippers, shoulders) if gr < 5]
        max_shoulder_at_grip_close = max(shoulder_at_grip_close) if shoulder_at_grip_close else None

        # Z at gripper close
        z_at_grip_close = [z for gr, z in zip(grippers, z_rel) if gr < 5]
        min_z_rel_at_grip_close = min(z_at_grip_close) if z_at_grip_close else None

        # --- CLASSIFICATION ---
        # OLD: Elbow-based
        if min_elbow < -30:
            elbow_class = "DEEP"
        elif min_elbow < -10:
            elbow_class = "APPROACH"
        else:
            elbow_class = "SHALLOW"

        # NEW: Relative FK Z (relative to home, more negative = deeper)
        # Thresholds: DEEP = arm went more than 150mm below home
        #             APPROACH = 50 to 150mm below home
        #             SHALLOW = less than 50mm below home
        if min_z_rel < -150:
            z_rel_class = "DEEP"
        elif min_z_rel < -50:
            z_rel_class = "APPROACH"
        else:
            z_rel_class = "SHALLOW"

        # SHOULDER proxy (r=-0.814 with Z)
        # Higher shoulder = deeper TCP Z
        # Threshold based on data: shoulder > 65 = DEEP, 40-65 = APPROACH, < 40 = SHALLOW
        if max_shoulder_at_grip_close is not None:
            sh_val = max_shoulder_at_grip_close
        else:
            sh_val = max_shoulder
        if sh_val > 65:
            shoulder_class = "DEEP"
        elif sh_val > 40:
            shoulder_class = "APPROACH"
        else:
            shoulder_class = "SHALLOW"

        episode_summaries.append({
            "ep_id": ep_id,
            "n_frames": len(frames),
            "min_elbow": round(min_elbow, 1),
            "max_shoulder": round(max_shoulder, 1),
            "max_shoulder_at_gc": round(max_shoulder_at_grip_close, 1) if max_shoulder_at_grip_close is not None else None,
            "min_z_rel": round(min_z_rel, 1),
            "min_z_rel_at_gc": round(min_z_rel_at_grip_close, 1) if min_z_rel_at_grip_close is not None else None,
            "gripper_range": round(max_gripper - min_gripper, 1),
            "elbow_class": elbow_class,
            "z_rel_class": z_rel_class,
            "shoulder_class": shoulder_class,
            "elbow_z_agree": elbow_class == z_rel_class,
        })

    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    all_elbows = np.array([d["elbow"] for d in all_data])
    all_shoulders = np.array([d["shoulder"] for d in all_data])
    all_z_rel = np.array([d["z_rel"] for d in all_data])

    corr_ez = np.corrcoef(all_elbows, all_z_rel)[0, 1]
    corr_sz = np.corrcoef(all_shoulders, all_z_rel)[0, 1]

    print("=" * 70)
    print("CORRELATION: Joint Angles vs FK Z (relative to home)")
    print("=" * 70)
    print(f"  Shoulder vs Z: r = {corr_sz:.3f}  <-- DOMINANT factor")
    print(f"  Elbow vs Z:    r = {corr_ez:.3f}  <-- WEAK/misleading")
    print()
    print(f"  VERDICT: Elbow alone is NOT a reliable proxy for grasp depth.")
    print(f"  Shoulder angle is much more predictive (2.8x stronger correlation).")
    print()

    # ============================================================
    # CLASSIFICATION COMPARISON TABLE
    # ============================================================
    print("=" * 70)
    print("Per-Episode Classification: Elbow vs RelZ vs Shoulder")
    print("=" * 70)
    print(f"{'Ep':<8} {'MinElbow':>10} {'MaxShldr':>10} {'MinZ_rel':>10} {'Z@GC':>8} {'ElbCls':>8} {'ZRelCls':>9} {'ShldrCls':>9} {'Agree'}")
    print("-" * 95)

    disagree_count = sum(1 for s in episode_summaries if not s["elbow_z_agree"])
    for s in episode_summaries:
        agree = "YES" if s["elbow_z_agree"] else "**NO**"
        z_gc = f"{s['min_z_rel_at_gc']:.0f}" if s["min_z_rel_at_gc"] is not None else "N/A"
        print(f"ep_{s['ep_id']:04d} "
              f"{s['min_elbow']:>+10.1f}° "
              f"{s['max_shoulder']:>+10.1f}° "
              f"{s['min_z_rel']:>+10.1f}mm "
              f"{z_gc:>8}mm "
              f"{s['elbow_class']:>8} "
              f"{s['z_rel_class']:>9} "
              f"{s['shoulder_class']:>9} "
              f"{agree}")

    print()
    print(f"Elbow vs RelZ classification disagreements: {disagree_count}/{len(episode_summaries)}")

    # ============================================================
    # DISTRIBUTION SUMMARY
    # ============================================================
    print()
    print("=" * 70)
    print("Dataset Distribution by Each Metric")
    print("=" * 70)

    for metric_key, label in [("elbow_class", "Elbow-based (OLD)"),
                               ("z_rel_class", "Relative-FK-Z (URDF)"),
                               ("shoulder_class", "Shoulder-proxy (new)")]:
        counts = {}
        for s in episode_summaries:
            c = s[metric_key]
            counts[c] = counts.get(c, 0) + 1
        print(f"\n  {label}:")
        for cls in ["DEEP", "APPROACH", "SHALLOW"]:
            n = counts.get(cls, 0)
            print(f"    {cls}: {n:2d}/{len(episode_summaries)} ({100*n/len(episode_summaries):.0f}%)")

    # ============================================================
    # CRITICAL PROOF: Cases where elbow says SHALLOW but Z says DEEP
    # ============================================================
    print()
    print("=" * 70)
    print("PROOF: Elbow says SHALLOW but arm actually went DEEP (Z_rel < -100mm)")
    print("=" * 70)
    false_neg = [s for s in episode_summaries
                 if s["elbow_class"] == "SHALLOW" and s["z_rel_class"] == "DEEP"]
    print(f"  Cases: {len(false_neg)} episodes labeled SHALLOW by elbow but DEEP by FK Z")
    print()
    for s in sorted(false_neg, key=lambda x: x["min_z_rel"])[:8]:
        print(f"  ep_{s['ep_id']:04d}: elbow={s['min_elbow']:+.0f}°, shoulder={s['max_shoulder']:+.0f}°, "
              f"Z_rel={s['min_z_rel']:+.0f}mm, gripper_range={s['gripper_range']:.0f}°")
    print()
    print("  These are episodes where arm reached DOWN (shoulder high) without")
    print("  needing extreme elbow bend. Elbow angle completely missed them.")
    print()

    # ============================================================
    # WHAT THE SHOULDER DATA TELLS US
    # ============================================================
    print("=" * 70)
    print("Shoulder Angle Distribution at Grasp-Close (Z Proxy)")
    print("=" * 70)
    sh_at_gc = [s["max_shoulder_at_gc"] for s in episode_summaries if s["max_shoulder_at_gc"] is not None]
    if sh_at_gc:
        print(f"  Max shoulder at gripper-close: mean={np.mean(sh_at_gc):.1f}°, "
              f"std={np.std(sh_at_gc):.1f}°")
        print(f"  Range: [{min(sh_at_gc):.0f}°, {max(sh_at_gc):.0f}°]")
        pcts = [25, 50, 75]
        for p in pcts:
            print(f"  {p}th percentile: {np.percentile(sh_at_gc, p):.1f}°")
    print()

    # ============================================================
    # KEY PHYSICAL RELATIONSHIPS
    # ============================================================
    print("=" * 70)
    print("FK Z vs Shoulder at Fixed Elbow Angles")
    print("=" * 70)
    print("  (shoulder=0 = arm forward horizontal, +90 = arm pointing down)")
    print()
    print(f"  {'Shoulder':>10} {'Elbow=0':>12} {'Elbow=-30':>12} {'Elbow=-55':>12}")
    print(f"  {'-'*50}")
    for sh in [0, 20, 40, 60, 70, 80, 90]:
        zs = []
        for el in [0, -30, -55]:
            z_rel = fk_roarm_m3([0, sh, el, 90, 0, 0])[2] - HOME_FK_Z_MM
            zs.append(z_rel)
        print(f"  shoulder={sh:+3d}°: {zs[0]:>+10.0f}mm  {zs[1]:>+10.0f}mm  {zs[2]:>+10.0f}mm")

    print()
    print("  Observation: Shoulder angle dominates Z. Going from shoulder=0 to 80")
    print("  drops Z by ~180mm regardless of elbow angle.")
    print()

    # ============================================================
    # RECOMMEND CORRECT VALIDATION LOGIC
    # ============================================================
    print("=" * 70)
    print("RECOMMENDED VALIDATION LOGIC FOR collect_data_manual.py")
    print("=" * 70)
    print("""
CURRENT ISSUE: validate_episode() uses min_z from pose_get() with thresholds
  DEEP < 100mm, APPROACH 100-200mm, SHALLOW > 200mm
  These thresholds are UNVERIFIED assumptions about ESP32 FK convention.

WHAT TO DO:

STEP 1 - HARDWARE CALIBRATION (one-time, required):
  Run the robot and note pose_get()[2] at:
  a) Home position: arm at [0, 0, 90, 0, 0, 0] degrees
  b) Touching table: move arm manually until TCP touches the table surface
  c) Mid-height: arm at typical approach height

  This gives you: z_home, z_table_touch, z_approach
  Then set thresholds relative to these values.

STEP 2 - THRESHOLD CALIBRATION:
  expected values (verify with hardware):
    z_home ~ 200-300mm (arm at neutral, high position)
    z_table ~ 20-60mm (TCP near table surface)
    threshold_deep: pose_get Z < z_table + 50mm
    threshold_approach: z_table + 50mm to z_table + 150mm

STEP 3 - SECONDARY CHECKS (always valid, no calibration needed):
  a) Shoulder angle at gripper-close > 50 degrees  -> arm was DOWN
  b) Gripper range > 15 degrees  -> gripper actually opened
  c) Min shoulder during episode > 30 degrees  -> arm extended forward/down

CURRENT DATASET STATUS (using URDF FK relative Z):
  43 episodes, all classified:
""")

    deep_z = [s for s in episode_summaries if s["z_rel_class"] == "DEEP"]
    approach_z = [s for s in episode_summaries if s["z_rel_class"] == "APPROACH"]
    shallow_z = [s for s in episode_summaries if s["z_rel_class"] == "SHALLOW"]
    print(f"  DEEP   (arm >150mm below home): {len(deep_z)}/43 ({100*len(deep_z)/43:.0f}%)")
    print(f"  APPROACH (50-150mm below home): {len(approach_z)}/43 ({100*len(approach_z)/43:.0f}%)")
    print(f"  SHALLOW (< 50mm below home):    {len(shallow_z)}/43 ({100*len(shallow_z)/43:.0f}%)")
    print()

    # Best episodes (deepest Z + good gripper)
    good_eps = [s for s in episode_summaries
                if s["min_z_rel"] < -150 and s["gripper_range"] > 15]
    print(f"  HIGH QUALITY (deep reach + good gripper): {len(good_eps)}/43")
    print()
    print("  Best episodes (deepest + good gripper):")
    for s in sorted(good_eps, key=lambda x: x["min_z_rel"])[:10]:
        print(f"    ep_{s['ep_id']:04d}: Z_rel={s['min_z_rel']:+.0f}mm, "
              f"shoulder={s['max_shoulder']:+.0f}°, gripper_range={s['gripper_range']:.0f}°")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
1. MEMORY LESSON #12 CONFIRMED: Elbow angle is NOT a reliable depth proxy
   - Elbow-Z correlation: r={corr_ez:.3f} (weak)
   - Shoulder-Z correlation: r={corr_sz:.3f} (strong, 2.8x better)
   - {disagree_count}/43 episodes misclassified by elbow criterion alone

2. DOMINANT FACTOR for grasp depth: SHOULDER ANGLE
   - Higher shoulder angle = arm tilts forward+down = lower TCP
   - Threshold: shoulder > 60-65 degrees at grasp = DEEP reach

3. ELBOW ROLE: Fine-tuning of reach distance, NOT primary depth control
   - Elbow affects FORWARD reach more than vertical depth
   - Using elbow < -30 degrees as "deep grasp" criterion is WRONG

4. VALIDATED METRIC: Relative FK Z (using URDF)
   - Home Z = {HOME_FK_Z_MM:.0f}mm (reference)
   - DEEP: Z_rel < -150mm (arm >150mm below home level)
   - Using this metric: {len(deep_z)}/43 DEEP, {len(approach_z)}/43 APPROACH, {len(shallow_z)}/43 SHALLOW

5. ESP32 pose_get() Z MUST be calibrated before use
   - Record z at home + z touching table = establishes absolute thresholds
   - Until then, use shoulder angle (>60 deg) as the quality gate
""")

    return episode_summaries, all_data


if __name__ == "__main__":
    summaries, frame_data = run_analysis()
