"""
data_official_pipeline_analysis.py

Compare our collect_data_manual.py validation rules against the official
LeRobot/SmolVLA reference pipeline.

Findings:
1. Official lerobot-record has ZERO episode validation
2. svla_so100_pickplace: 50ep, 5 positions, 10ep/pos, ~393fr/ep (13s), 30fps
3. Our script has 6 checks — classify as: JUSTIFIED / REDUNDANT / OVERCONSTRAINED

Run: python data_official_pipeline_analysis.py
"""

import json
import os

# ── Reference dataset numbers (from SmolVLA paper / smolvla.mdx) ──────────────
REFERENCE = {
    "dataset":          "lerobot/svla_so100_pickplace",
    "num_episodes":     50,
    "num_positions":    5,
    "reps_per_pos":     10,
    "fps":              30,
    "episode_frames":   393,        # ~13 seconds (from memory analysis)
    "episode_time_s":   13.1,
    "training_steps":   20000,
    "batch_size":       64,
    "hardware":         "SO100 (5-DOF)",
    "joint_limits":     "Feetech STS3215 (360° servo)",
    "camera":           "OpenCV (640x480 or similar)",
}

# ── Our v5 dataset numbers (from agent memory) ─────────────────────────────────
V5_STATS = {
    "num_episodes":     136,
    "fps":              30,
    "mean_frames":      99,         # 3.3s/ep
    "mean_time_s":      3.3,
    "training_steps":   200000,     # 10x the reference
}

# ── Our current validation checks in collect_data_manual.py ───────────────────
VALIDATION_CHECKS = [
    {
        "id": "C0a",
        "name": "HOME start distance < 30deg",
        "code": "home_dist > 30 → FAIL (blocks save)",
        "in_official": False,
        "verdict": "JUSTIFIED",
        "reason": (
            "Official pipeline has leader-follower: operator naturally returns arm to rest "
            "before starting next episode. Our manual torque-OFF mode has no such guarantee. "
            "V5 failure proved this: 136ep all started at approach pose → proprioceptive echo. "
            "This check is necessary for OUR specific setup."
        ),
    },
    {
        "id": "C0b",
        "name": "Approach phase (base OR shoulder >5deg travel by midpoint)",
        "code": "base_travel < 5 AND shoulder_travel < 5 → FAIL",
        "in_official": False,
        "verdict": "OVERCONSTRAINED",
        "reason": (
            "5 degree threshold is too strict. CENTER zone episodes naturally have base near 0. "
            "If C0a (HOME start) passes, approach phase is guaranteed because HOME shoulder=0 "
            "and grasp requires shoulder>40deg → travel >40deg always. "
            "This check is REDUNDANT if C0a passes. "
            "RECOMMENDATION: Remove C0b, or relax to shoulder_travel < 10."
        ),
    },
    {
        "id": "C1",
        "name": "Gripper must open >40deg",
        "code": "max_gripper < 40 → FAIL",
        "in_official": False,
        "verdict": "JUSTIFIED",
        "reason": (
            "Our torque-OFF mode has a common failure mode where operator forgets to open "
            "gripper. Without this check, closed-gripper episodes pollute training data. "
            "Official leader-follower setup naturally demonstrates open→grasp→close. "
            "HOWEVER: threshold of 40deg may be too strict. Reference sponge grasps may use "
            "softer open. RECOMMENDATION: Reduce to 30deg to be less restrictive."
        ),
    },
    {
        "id": "C2",
        "name": "Gripper range >15deg",
        "code": "gripper_range < 15 → FAIL",
        "in_official": False,
        "verdict": "REDUNDANT",
        "reason": (
            "If C1 (max_gripper > 40) passes AND min_gripper is typically 0-5deg at HOME, "
            "range will automatically be 35-40deg. This check only triggers if max=40 and min=25+. "
            "RECOMMENDATION: Remove this check — it adds no protection beyond C1."
        ),
    },
    {
        "id": "C3",
        "name": "Shoulder at grip close >40deg",
        "code": "shoulder_at_grip_close < 40 → FAIL",
        "in_official": False,
        "verdict": "OVERCONSTRAINED",
        "reason": (
            "This requires DEEP grasp. But the reference protocol says nothing about "
            "minimum grasp depth. V3 success included both DEEP (shoulder>60) and APPROACH (40-60). "
            "The real requirement is: gripper closes while arm is low enough to reach object. "
            "RECOMMENDATION: Relax to shoulder_at_grip_close < 30 (currently catches arm-at-home grasps). "
            "Or convert to WARNING only."
        ),
    },
    {
        "id": "C4",
        "name": "Frame count >=120 (4 seconds)",
        "code": "num_frames < 120 → FAIL",
        "in_official": False,
        "verdict": "OVERCONSTRAINED",
        "reason": (
            "Reference SO100 dataset: 393 frames (13s). OUR v3 success: ~178 frames (5.9s). "
            "Our v5 failure: 99 frames (3.3s) — but failure was HOME issue, not length. "
            "120 frame minimum (4s) is reasonable BUT: the reference protocol does not "
            "impose minimum. A fast, clean 90-frame (3s) episode HOME→approach→grasp→return "
            "should be VALID. RECOMMENDATION: Reduce to 90 frames (3s) minimum, "
            "or make this a WARNING not FAIL."
        ),
    },
    {
        "id": "C5",
        "name": "Z height at grasp <130mm",
        "code": "z_at_grip_close > 130 → FAIL",
        "in_official": False,
        "verdict": "JUSTIFIED",
        "reason": (
            "Prevents recording where operator grasps air (arm never descends). "
            "Z=130mm corresponds to arm above the sponge height. "
            "This is a robot-specific physical check — sensible for our hardware."
        ),
    },
]


def print_analysis():
    print("=" * 70)
    print("OFFICIAL PIPELINE vs OUR VALIDATION CHECKS")
    print("=" * 70)

    print("\n[1] OFFICIAL PIPELINE (lerobot-record)")
    print("-" * 40)
    print("  Validation checks: ZERO")
    print("  The only quality gate is the OPERATOR pressing Left Arrow to re-record.")
    print("  No gripper check, no frame count check, no joint travel check.")
    print("  Protocol: Leader-Follower, operator re-records bad episodes manually.")
    print()

    print("[2] REFERENCE DATASET (svla_so100_pickplace)")
    print("-" * 40)
    for k, v in REFERENCE.items():
        print(f"  {k:20s}: {v}")
    print()
    print("  KEY NUMBERS:")
    print(f"    Episode length: 393 frames = 13.1 seconds @ 30fps")
    print(f"    Training: 20K steps, batch=64")
    print(f"    Our v5 episodes: 99 frames = 3.3s (33% of reference)")
    print(f"    Our v5 training: 200K steps (10x over-training)")
    print()

    print("[3] OUR VALIDATION CHECKS — VERDICT")
    print("-" * 40)
    counts = {"JUSTIFIED": 0, "OVERCONSTRAINED": 0, "REDUNDANT": 0}
    for c in VALIDATION_CHECKS:
        verdict = c["verdict"]
        counts[verdict] += 1
        marker = {"JUSTIFIED": "OK", "OVERCONSTRAINED": "!!", "REDUNDANT": "--"}[verdict]
        print(f"\n  [{marker}] {c['id']}: {c['name']}")
        print(f"       Code   : {c['code']}")
        print(f"       Verdict: {verdict}")
        print(f"       Reason : {c['reason'][:200]}")
        if len(c['reason']) > 200:
            print(f"                {c['reason'][200:]}")

    print()
    print("[4] SUMMARY")
    print("-" * 40)
    for verdict, count in counts.items():
        print(f"  {verdict}: {count}")

    print()
    print("[5] RECOMMENDED CHANGES TO collect_data_manual.py")
    print("-" * 40)
    changes = [
        ("KEEP",   "C0a", "HOME start check — unique to our torque-OFF setup"),
        ("REMOVE", "C0b", "Approach phase check — redundant if C0a passes"),
        ("RELAX",  "C1",  "Gripper open: 40deg → 30deg threshold"),
        ("REMOVE", "C2",  "Gripper range check — redundant with C1"),
        ("RELAX",  "C3",  "Shoulder at grasp: FAIL→WARNING, threshold 40→30deg"),
        ("RELAX",  "C4",  "Frame count: 120 FAIL → 90 FAIL, 150 WARNING"),
        ("KEEP",   "C5",  "Z height at grasp — robot-specific physical check"),
    ]
    for action, check_id, note in changes:
        marker = {"KEEP": " OK", "REMOVE": "DEL", "RELAX": "MOD"}[action]
        print(f"  [{marker}] {check_id}: {note}")

    print()
    print("[6] ROOT CAUSE OF v5 FAILURE (confirmed)")
    print("-" * 40)
    print("  The validations were NOT the problem.")
    print("  The problem was that ALL 136 episodes PASSED our checks")
    print("  but were collected starting at APPROACH POSE, not HOME.")
    print("  → C0a (HOME start) check was missing in v5 collection.")
    print("  → C0a was added AFTER v5 failure. Correct.")
    print()
    print("  REAL DANGER: Over-aggressive checks (C0b, C2, C3) may cause")
    print("  GOOD episodes to be rejected, reducing effective dataset size.")
    print("  The official pipeline rejects ZERO episodes automatically.")
    print("  Quality is maintained by the operator, not the code.")

    print()
    print("[7] ACTION ITEMS")
    print("-" * 40)
    print("  1. Remove C0b (approach check) — causes false rejects in CENTER zone")
    print("  2. Relax C1 gripper threshold: 40deg → 30deg")
    print("  3. Remove C2 (gripper range) — redundant")
    print("  4. Convert C3 (shoulder at grasp) to WARNING only")
    print("  5. Relax C4 frame count FAIL threshold: 120 → 90")
    print("  6. Keep C0a and C5 as FAIL-level checks")
    print()
    print("  MOST IMPORTANT: Focus on getting 50 clean episodes")
    print("  (5 zones × 10 reps) rather than perfecting validation logic.")


def check_v5_collected_data():
    """Check what actually exists in collected_data directories."""
    dirs_to_check = [
        "collected_data",
        "collected_data_v5",
        "collected_data_sponge",
    ]
    base = os.path.dirname(os.path.abspath(__file__))

    print("\n[8] EXISTING COLLECTED DATA")
    print("-" * 40)
    for d in dirs_to_check:
        full_path = os.path.join(base, d)
        if os.path.exists(full_path):
            episodes = [x for x in os.listdir(full_path) if x.startswith("episode_")]
            print(f"  {d}: {len(episodes)} episodes")
            # Read zone distribution from metadata
            zone_counts = {}
            home_start_fails = 0
            for ep in sorted(episodes):
                meta_path = os.path.join(full_path, ep, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        zone = meta.get("zone", "UNKNOWN")
                        zone_counts[zone] = zone_counts.get(zone, 0) + 1
                        # Check if shoulder start is near 0 (HOME)
                        frames = meta.get("frames", [])
                        if frames:
                            first_angles = frames[0].get("angles", None)
                            if first_angles and first_angles[1] > 20:
                                home_start_fails += 1
                    except Exception:
                        pass
            if zone_counts:
                print(f"    Zone distribution: {dict(sorted(zone_counts.items()))}")
            if home_start_fails > 0:
                print(f"    WARNING: {home_start_fails} episodes may NOT start from HOME (shoulder>20 at frame 0)")
        else:
            print(f"  {d}: NOT FOUND")


if __name__ == "__main__":
    print_analysis()
    check_v5_collected_data()
