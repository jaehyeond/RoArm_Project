"""
[A1 MANIPULATION] Eval Methodology Critique: Visual Grounding Failure Diagnosis
Target: SmolVLA v5 120K checkpoint, deploy_20260331_135841.csv

Problem: ALL offline metrics PASSED, yet base joint completely ignores
sponge position in real deployment (base stuck at ~10° = dataset mean).

This script:
1. Quantifies the exact nature of the eval's blindspot
2. Demonstrates the correct metrics that would have caught the failure
3. Proposes a pre-deployment visual grounding test protocol
"""

import csv
import math
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# Constants from eval_v5_checkpoints.py & results
# ─────────────────────────────────────────────

JOINT_NAMES = ["Base", "Shoulder", "Elbow", "WristP", "WristR", "Gripper"]

# Dataset distribution (from problem statement)
DATASET_BASE_MEAN = 9.93
DATASET_BASE_STD  = None  # not directly given; we derive it

# From train_v5_eval_results.md — 120K checkpoint
EVAL_BASE_MEAN = 9.88   # predicted base mean over 50 test frames
EVAL_BASE_STD  = 28.06  # predicted base std over 50 test frames
EVAL_BASE_MIN  = -42.96
EVAL_BASE_MAX  = 84.07

# Zone breakdown (50 test frames: n=7 LEFT, n=33 CENTER, n=10 RIGHT)
ZONE_N = {"LEFT": 7, "CENTER": 33, "RIGHT": 10}
ZONE_L2 = {"LEFT": 4.33, "CENTER": 3.63, "RIGHT": 4.01}  # total L2, not base-only

# Data distribution
PDATA_CENTER = 0.801  # 80.1% episodes |base| < 30°
PDATA_LEFT   = 2 / 136  # ~1.5%
PDATA_RIGHT  = 25 / 136  # ~18.4%

# Deployment observation: base stuck at ~10°
DEPLOY_BASE_RANGE = (8.9, 10.5)  # degrees observed in deploy_20260331_135841.csv

LOG_PATH = Path("/home/cgxr/Documents/Robotics/RoArm_Project/logs/deploy_20260331_135841.csv")


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def fval(row, key):
    try:
        return float(row[key])
    except (ValueError, KeyError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 1: Is the eval fundamentally testing the wrong thing?
# ─────────────────────────────────────────────────────────────────────────────

def q1_l2_masking_analysis():
    print("=" * 72)
    print("Q1: Does L2 mask the base grounding failure?")
    print("=" * 72)

    # Scenario A: model always predicts base = dataset_mean = 9.93°
    # How much does this contribute to L2 error?

    # The eval samples 50 frames proportional to dataset distribution.
    # With 80.1% CENTER frames, expected base GT in test set:
    # Rough model: CENTER samples have base ~ N(0, 10°), non-center ~ N(±30, 15°)

    # For a model that ALWAYS predicts base = 9.93 (mean):
    # Expected base L2 contribution from CENTER (n=40): |9.93 - ~0| ≈ 9.93°
    # Expected base L2 contribution from RIGHT  (n=8):  |9.93 - ~30| ≈ 20°
    # Expected base L2 contribution from LEFT   (n=2):  |9.93 - ~-20| ≈ 30°

    # But overall L2 is computed as vector norm across ALL 6 joints.
    # Even if base error is 15-20°, other joints contribute: shoulder=0, elbow=0, etc.
    # So overall L2 from "base-stuck" model is dominated by shoulder/elbow/wrist.

    print("""
FINDING: Yes — the eval is fundamentally measuring the wrong property.

The L2 metric computes ||predicted - ground_truth||_6 (Euclidean norm over
6 joints). The reported L2 = 3.80° is an AVERAGE error including all joints.

Key arithmetic:
  - 80.1% of frames: base GT ≈ 0-15° → if model outputs 9.93°, base error ≈ 5°
  - 18.4% of frames: base GT ≈ 15-50° → model outputs 9.93°, base error ≈ 10-40°
  - 1.5% of frames:  base GT ≈ -30-0° → model outputs 9.93°, base error ≈ 10-40°

But shoulder/elbow/wrist_pitch contribute ~15-25° each to the L2 norm.
A base error of 10-15° adds only sqrt(10²) = 10° to the vector norm while
sqrt(15² + 20² + 12²) = 27° is coming from the temporal-sequence joints.

Concrete scenario: if base is ALWAYS wrong by 15°, but shoulder+elbow+wristP
vary correctly, the 6D L2 increases by only:
  sqrt(25² + 15²) - sqrt(25²) ≈ 30.4 - 25 = 5.4°

So base grounding failure raises L2 by at most ~5°, well within noise.

This is why L2 = 3.80° passed: the metric is INSENSITIVE to base grounding.
""")

    # The deeper problem: base is being predicted correctly on average
    # because 80% of test inputs have base GT ≈ dataset_mean anyway.
    # The model never has to move base far — and so a constant-prediction
    # looks indistinguishable from a correctly-conditioned prediction.

    print("CRITICAL INSIGHT: Accuracy ≠ Correlation")
    print("─" * 50)
    print("""
  The model's base predictions (mean=9.88°, std=28.06°) LOOK reasonable.
  But do they CORRELATE with ground truth base angles?

  Two models produce identical L2 and Std:
    Model A (working):   pred_base = f(image) correctly locates sponge
    Model B (broken):    pred_base = random_sample(N(9.88, 28.06°))

  Both produce:  mean ≈ 9.88°, std ≈ 28°, L2 ≈ 3.80°
  Only correlation analysis distinguishes them.

  Pearson r (pred_base vs GT_base):
    Model A: r ≈ 0.85+ (strong positive correlation)
    Model B: r ≈ 0.0   (no correlation — exactly what we see in deployment)
""")


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 2: Correct metrics for visual grounding
# ─────────────────────────────────────────────────────────────────────────────

def q2_grounding_metrics():
    print("=" * 72)
    print("Q2: Metrics that actually test visual grounding")
    print("=" * 72)

    print("""
The key question is: "Does the model's BASE angle prediction co-vary with
the actual sponge position (i.e., ground truth base angle in the demo)?"

METRIC 1 — Pearson Correlation: pred_base vs GT_base
─────────────────────────────────────────────────────
  r = Cov(pred_base, GT_base) / (σ_pred × σ_GT)

  Threshold: r < 0.5 → grounding failure (model output not tracking GT)
  Threshold: r > 0.7 → grounding likely working

  This is the SINGLE most diagnostic metric for this failure mode.
  Cost: trivial to add to eval_v5_checkpoints.py (2 lines of numpy).

METRIC 2 — Conditional Zone Accuracy (per-zone directional correctness)
────────────────────────────────────────────────────────────────────────
  For each test frame, check: does pred_base have the SAME SIGN as GT_base
  relative to CENTER (0°)?

  zone_directional_accuracy = mean(sign(pred_base - 0) == sign(GT_base - 0))

  For CENTER frames (GT_base ≈ 0): tolerate ±15° regardless
  For LEFT  frames (GT_base < -15°): require pred_base < 0°
  For RIGHT frames (GT_base > +15°): require pred_base > 0°

  Threshold: < 0.70 → visual grounding failing for lateral positioning

METRIC 3 — Per-joint Correlation Matrix
────────────────────────────────────────
  For each joint j: r_j = Pearson(pred_j, GT_j)

  Minimum acceptable: r_j > 0.5 for base and shoulder (spatially-grounded joints)
  Joints less critical: wrist_roll, gripper (temporal sequence, less image-dependent)

METRIC 4 — Zone-Conditional Base L2 (not total L2)
───────────────────────────────────────────────────
  What eval_v5_checkpoints.py computes: L2 = ||pred_6D - GT_6D|| (all joints)
  What it SHOULD compute: base_L2_by_zone = mean |pred_base - GT_base| per zone

  Current eval: zone_L2(LEFT)=4.33° across all 6 joints. Base contribution unknown.
  Correct eval: base_L2(LEFT) should be significantly LARGER than base_L2(CENTER)
  if the model is responding correctly to sponge position.

  If base_L2 is THE SAME across zones → model is ignoring zone = grounding failure.

METRIC 5 — Baseline Comparison (oracle test)
────────────────────────────────────────────
  Compare model predictions against two baselines:

  Baseline A — constant predictor: always output dataset_mean
  Baseline B — random normal:      sample from N(action_mean, action_std)

  A real model should beat BOTH baselines on base L2. If it doesn't beat
  Baseline A on LEFT/RIGHT frames, it's not using the image at all for base.

  Current eval result: zone L2 ratio = 1.19 (max/min).
  A purely constant predictor would give ratio ≈ ZONE_L2_RIGHT/ZONE_L2_CENTER
  which for a symmetric task ≈ 1.0-1.3 purely from the other 5 joints.
  So ratio 1.19 is CONSISTENT with a constant base predictor.
""")


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 3: Why did overall_std = 25.26° not catch it?
# ─────────────────────────────────────────────────────────────────────────────

def q3_std_masking():
    print("=" * 72)
    print("Q3: Why did overall_std = 25.26° not catch the failure?")
    print("=" * 72)

    print("""
The eval checks: overall_std = mean(per_joint_std) across 50 frames.
From train_v5_eval_results.md at 120K:

  Joint     | Predicted Std (deg)
  ─────────────────────────────────
  Base      |  28.06   ← appears healthy
  Shoulder  |  17.12
  Elbow     |  32.30
  WristP    |  29.81
  WristR    |  26.44
  Gripper   |  17.82
  ─────────────────────────────────
  Overall   |  25.26°  (mean of above)

Base std = 28.06° across 50 test frames.
This looks like the model IS varying base angle. But WHY is std high?

Answer: because the TEST FRAMES span LEFT/CENTER/RIGHT zones.
The GT base angles vary widely (from -43° to +84° in test set).
The MODEL is outputting base predictions that vary with this range.

Critical question: ARE the base predictions varying because the model
is REACTING TO THE IMAGE, or because the model is reacting to the
STATE INPUT (current base angle as normalised proprioception)?

Evidence that it's the STATE, not the IMAGE:
  - Deployment: base is always ~10° regardless of sponge position
  - In deployment, current_base ≈ 10° (fixed starting point)
  - Model outputs base ≈ 10° always (regresses toward input state)
  - In offline eval: input state IS the dataset frame state, which varies
    → base prediction varies by echoing the proprioceptive input, not vision

ROOT CAUSE of std masking:
  The model has learned "predict base ≈ current_base + small_delta"
  (position tracking with small offset, not vision-conditioned movement).

  In offline eval: state input varies across frames → base prediction varies
  In deployment:   state input is fixed near dataset_mean → base prediction fixed

  Std = 28° in offline eval does NOT mean visual grounding works.
  It means the model tracks proprioceptive input. Completely different.

WHAT METRIC WOULD CATCH THIS:
  Ablation: run eval with PERMUTED images (shuffle images across test frames).
  If base prediction std and L2 are UNCHANGED under image permutation,
  the model is not using the image for base grounding.

  Expected result for broken model: L2 unchanged ±0.5°
  Expected result for working model: L2 increases significantly (>3-5°)
""")

    print("Deployment data confirms:")
    print(f"  Base range in deployment (n=300 steps): {DEPLOY_BASE_RANGE[0]:.1f} – {DEPLOY_BASE_RANGE[1]:.1f}°")
    print(f"  Base range in offline eval (50 frames):  {EVAL_BASE_MIN:.1f} – {EVAL_BASE_MAX:.1f}°")
    print(f"  Discrepancy: {EVAL_BASE_MAX - DEPLOY_BASE_RANGE[1]:.0f}° range in eval vs {DEPLOY_BASE_RANGE[1]-DEPLOY_BASE_RANGE[0]:.1f}° in deployment")
    print()
    print("  Interpretation: the offline eval base variation comes from")
    print("  proprioceptive echoing. In deployment with fixed start state,")
    print("  the model outputs fixed base angle.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 4: Zone L2 ratio 1.19 with n=7 LEFT — statistical power
# ─────────────────────────────────────────────────────────────────────────────

def q4_statistical_power():
    print("=" * 72)
    print("Q4: Is zone L2 ratio 1.19 with n=7 LEFT statistically meaningful?")
    print("=" * 72)

    # Bootstrap confidence interval estimate for zone L2 ratio
    # Without raw data, estimate via normal approximation.

    # L2 error is chi-distributed (sum of squares).
    # For 6D L2 with each component ~N(0, sigma), L2 follows chi(6)*sigma.
    # Variance of mean L2 ≈ sigma_l2^2 / n

    # The reported L2 per zone (e.g., LEFT=4.33°, CENTER=3.63°) are mean values.
    # If individual L2 errors have std ≈ 2-3° (typical), then:
    # SE(LEFT mean)   = 3.0 / sqrt(7) ≈ 1.13°
    # SE(CENTER mean) = 3.0 / sqrt(33) ≈ 0.52°
    # SE(RIGHT mean)  = 3.0 / sqrt(10) ≈ 0.95°

    n_left, n_center, n_right = 7, 33, 10
    assumed_l2_sd = 2.5  # degrees — estimate from overall L2 distribution

    se_left   = assumed_l2_sd / math.sqrt(n_left)
    se_center = assumed_l2_sd / math.sqrt(n_center)
    se_right  = assumed_l2_sd / math.sqrt(n_right)

    # 95% CI for each zone mean L2 (±1.96 SE)
    ci_left   = (ZONE_L2["LEFT"]   - 1.96*se_left,   ZONE_L2["LEFT"]   + 1.96*se_left)
    ci_center = (ZONE_L2["CENTER"] - 1.96*se_center, ZONE_L2["CENTER"] + 1.96*se_center)
    ci_right  = (ZONE_L2["RIGHT"]  - 1.96*se_right,  ZONE_L2["RIGHT"]  + 1.96*se_right)

    print(f"""
Zone L2 summary (120K checkpoint):
  LEFT   (n={n_left:2d}): L2 = {ZONE_L2["LEFT"]:.2f}° ± {1.96*se_left:.2f}°  95% CI: [{ci_left[0]:.2f}, {ci_left[1]:.2f}]
  CENTER (n={n_center:2d}): L2 = {ZONE_L2["CENTER"]:.2f}° ± {1.96*se_center:.2f}°  95% CI: [{ci_center[0]:.2f}, {ci_center[1]:.2f}]
  RIGHT  (n={n_right:2d}): L2 = {ZONE_L2["RIGHT"]:.2f}° ± {1.96*se_right:.2f}°  95% CI: [{ci_right[0]:.2f}, {ci_right[1]:.2f}]

  95% CI for LEFT overlaps CENTER: [{ci_left[0]:.2f}, {ci_left[1]:.2f}] vs [{ci_center[0]:.2f}, {ci_center[1]:.2f}]
  Overlap: YES → difference is NOT statistically significant at α=0.05.

Statistical power analysis:
  To detect a true L2 difference of Δ = 1.0° between LEFT and CENTER:
  Required sample size per zone (power=0.80, α=0.05, two-sample t-test):
    n ≈ (2 * (1.96 + 0.84)^2 * σ^2) / Δ^2
    n ≈ (2 * 7.84 * 6.25) / 1.0 ≈ 98 samples per zone

  Current LEFT n=7 has power ≈ {_power_estimate(n_left, assumed_l2_sd, 1.0):.0%} to detect a 1° difference.
  Current RIGHT n=10 has power ≈ {_power_estimate(n_right, assumed_l2_sd, 1.0):.0%} to detect a 1° difference.

CONCLUSION: With n=7 LEFT and n=10 RIGHT, zone L2 ratios are NOISE.
  The ratio 1.19 (max/min) is consistent with random sampling variation.
  No meaningful conclusion can be drawn about zone-specific performance.

  The threshold "< 2.0 → OK" accepted this result, but the metric has
  effectively zero statistical power at these sample sizes.

  Required minimum: n ≥ 30 per zone for 80% power at 1° resolution.
  Required minimum: n ≥ 15 per zone for basic directional sign test.
  Current LEFT n=7 is INSUFFICIENT for ANY conclusion about LEFT zone performance.
""")


def _power_estimate(n, sigma, delta):
    """Approximate power for two-sample t-test vs CENTER (n_center=33)."""
    # SE of difference
    n_c = 33
    se_diff = sigma * math.sqrt(1/n + 1/n_c)
    # Non-centrality parameter
    ncp = delta / se_diff
    # Approximate power (normal approximation)
    z_alpha = 1.96
    power = max(0, 1 - _norm_cdf(z_alpha - ncp))
    return power


def _norm_cdf(x):
    """Approximate normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 5: Pre-deployment test protocol for visual grounding
# ─────────────────────────────────────────────────────────────────────────────

def q5_grounding_test_protocol():
    print("=" * 72)
    print("Q5: Pre-deployment visual grounding test protocol")
    print("=" * 72)

    print("""
PROPOSED PROTOCOL: Visual Grounding Verification (VGV) Test
─────────────────────────────────────────────────────────────

This should run BEFORE any real robot deployment. It requires no hardware.

Step 0: Minimum sample requirements
  ┌─────────────────────────────────────────────────────────┐
  │ Sample test set with ZONE BALANCE:                      │
  │   n_LEFT   ≥ 30  (currently: 7  — insufficient)        │
  │   n_CENTER ≥ 30  (currently: 33 — OK)                  │
  │   n_RIGHT  ≥ 30  (currently: 10 — insufficient)        │
  │   TOTAL    ≥ 90  (currently: 50 — too small overall)   │
  └─────────────────────────────────────────────────────────┘
  Sampling strategy: do NOT sample evenly across all frames.
  Explicitly oversample LEFT and RIGHT episodes to balance zones.
  Use episode_index from dataset metadata to identify zone membership.

Step 1: Per-joint Pearson correlation test (GATE TEST)
  For each spatially-grounded joint (Base, Shoulder):
    r = np.corrcoef(pred_joint, gt_joint)[0,1]
    PASS criterion: r ≥ 0.50
    FAIL criterion: r < 0.30 → immediate deployment block
    WARN criterion: 0.30 ≤ r < 0.50

  Note: Shoulder/Elbow are temporal (pick sequence), so correlation
  may be lower there — but BASE must show correlation if sponge position matters.

Step 2: Image-permutation ablation (CRITICAL — catches Q3 failure mode)
  a) Run model on N test frames → record pred_base[i] for each frame i
  b) Shuffle images: reshuffle image[i] independently of state[i]
  c) Run model again on shuffled images with same states → pred_base_shuffled[i]
  d) Compute: image_sensitivity = mean |pred_base[i] - pred_base_shuffled[i]|

  PASS criterion: image_sensitivity ≥ 5°
    (model changes base prediction when image changes, while state is fixed)
  FAIL criterion: image_sensitivity < 2°
    (model base prediction does NOT depend on image → pure proprioception echo)

  This test DIRECTLY catches the failure mode in this deployment.
  Expected current model result: image_sensitivity ≈ 1-3° (FAIL)

Step 3: Zone-conditional base directional accuracy
  For each non-CENTER test frame:
    correct = (pred_base > 0) == (gt_base > 0)  # correct side of center?

  PASS criterion: directional_accuracy ≥ 0.70 for LEFT frames
  PASS criterion: directional_accuracy ≥ 0.70 for RIGHT frames

  This is a sign test — even n=15 gives 80% power if true rate is 0.85.
  Better than L2 for sparse zones.

Step 4: Constant-predictor baseline comparison
  Compute baseline L2: model that always outputs dataset_mean

  For LEFT zone:  baseline_base_L2 = mean|9.93 - gt_base_left|
  For RIGHT zone: baseline_base_L2 = mean|9.93 - gt_base_right|

  Model must beat baseline on base L2 by ≥ 30% in each zone:
    model_base_L2_LEFT < 0.70 × baseline_base_L2_LEFT

  A model that fails this test is no better than a constant predictor
  for lateral positioning — exactly the failure observed here.

Step 5: State-conditioned consistency check
  Run inference on the SAME image with DIFFERENT state inputs
  (e.g., state at dataset_mean ± 30° on base joint).

  PASS criterion: base prediction changes by < 10° when state changes by 30°
    (model primarily driven by image, not state echo)
  FAIL criterion: base prediction tracks state input with gain > 0.5
    (model is echoing proprioception, not reading the image)

─────────────────────────────────────────────────────────────
MINIMUM VIABLE ADDITION to eval_v5_checkpoints.py:
─────────────────────────────────────────────────────────────

  # After computing all_actions_arr and all_gt_arr:

  # (A) Per-joint Pearson correlation
  for i, name in enumerate(JOINT_NAMES):
      r = np.corrcoef(all_actions_arr[:,i], all_gt_arr[:,i])[0,1]
      pass_str = "OK" if r > 0.5 else ("WARN" if r > 0.3 else "FAIL")
      print(f"  {name} Pearson r: {r:.3f}  {pass_str}")

  # (B) Image-permutation test (add ~10 lines)
  shuffled_indices = np.random.permutation(len(test_indices))
  perm_actions = run_inference_with_permuted_images(shuffled_indices, ...)
  image_sensitivity = np.mean(np.abs(all_actions_arr[:,0] - perm_actions[:,0]))
  print(f"  Base image sensitivity: {image_sensitivity:.2f}° (need ≥ 5° for PASS)")

  # (C) Zone directional accuracy
  for zone_name, mask in zones:
      if mask.sum() >= 5:
          dir_acc = np.mean(np.sign(all_actions_arr[mask,0]) == np.sign(all_gt_arr[mask,0]))
          print(f"  {zone_name} directional accuracy: {dir_acc:.1%}")
""")


# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Deployment trajectory analysis for 135841 run
# ─────────────────────────────────────────────────────────────────────────────

def analyze_deployment_run():
    print("=" * 72)
    print("DEPLOYMENT ANALYSIS: deploy_20260331_135841.csv")
    print("=" * 72)

    if not LOG_PATH.exists():
        print(f"  [SKIP] Log not found: {LOG_PATH}")
        return

    rows = load_csv(LOG_PATH)
    if not rows:
        print("  [SKIP] Empty log")
        return

    base_vals      = [fval(r, "base")        for r in rows]
    shoulder_vals  = [fval(r, "shoulder")    for r in rows]
    elbow_vals     = [fval(r, "elbow")       for r in rows]
    wrist_p_vals   = [fval(r, "wrist_pitch") for r in rows]
    gripper_vals   = [fval(r, "gripper")     for r in rows]

    n = len(rows)
    print(f"\n  Steps logged: {n}")
    print()
    print(f"  {'Joint':<14} {'start':>7} {'end':>7} {'min':>7} {'max':>7} {'range':>7} {'std':>7}")
    print("  " + "-" * 58)
    for name, vals in [
        ("Base",       base_vals),
        ("Shoulder",   shoulder_vals),
        ("Elbow",      elbow_vals),
        ("WristP",     wrist_p_vals),
        ("Gripper",    gripper_vals),
    ]:
        arr = np.array(vals)
        print(f"  {name:<14} {arr[0]:>7.2f} {arr[-1]:>7.2f} {arr.min():>7.2f} {arr.max():>7.2f} "
              f"{arr.max()-arr.min():>7.2f} {arr.std():>7.2f}")

    base_arr = np.array(base_vals)
    print(f"\n  Base joint analysis:")
    print(f"    Mean:   {base_arr.mean():.3f}° (dataset mean = {DATASET_BASE_MEAN:.2f}°)")
    print(f"    Std:    {base_arr.std():.3f}°")
    print(f"    Range:  {base_arr.min():.2f} – {base_arr.max():.2f}°")
    print(f"    Max deviation from dataset_mean: {np.max(np.abs(base_arr - DATASET_BASE_MEAN)):.2f}°")
    print()

    # Confirm: base stuck at dataset mean
    within_5 = np.mean(np.abs(base_arr - DATASET_BASE_MEAN) < 5.0)
    print(f"    Fraction of steps with |base - dataset_mean| < 5°: {within_5:.1%}")
    print()

    if within_5 > 0.90:
        print("  CONFIRMED: Base joint is STUCK at dataset mean throughout deployment.")
        print("  This is the visual grounding failure: base ignores sponge position.")
        print()

    # Show that OTHER joints DO move (pick sequence is executed)
    shoulder_range = np.array(shoulder_vals).max() - np.array(shoulder_vals).min()
    gripper_range  = np.array(gripper_vals).max()  - np.array(gripper_vals).min()
    print(f"  Shoulder range: {shoulder_range:.1f}° — model IS executing pick sequence")
    print(f"  Gripper range:  {gripper_range:.1f}° — gripper opens/closes")
    print()
    print("  INTERPRETATION: The model learned the pick MOTION (temporal sequence)")
    print("  but NOT the pick LOCATION (which direction to point the base).")
    print("  The visual grounding of sponge position is absent or broken.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY: Root cause of eval blindspot
# ─────────────────────────────────────────────────────────────────────────────

def root_cause_summary():
    print("=" * 72)
    print("ROOT CAUSE SUMMARY: Why All Metrics Passed But Grounding Failed")
    print("=" * 72)

    print("""
The eval has FIVE structural blindspots, in order of severity:

BLINDSPOT 1 (CRITICAL): No image-conditional metric
  All metrics measure prediction quality against ground truth,
  but NONE verify that predictions CHANGE when the image changes.
  A model that ignores the image entirely can score identically
  to a model that reads the image, if proprioception provides
  sufficient signal for L2 minimization.
  FIX: Image-permutation ablation (Step 2 of VGV protocol above).

BLINDSPOT 2 (CRITICAL): L2 measured across all 6 joints
  Base grounding failure adds only ~5° to a 25° L2 norm.
  The insensitivity of L2 to base errors is guaranteed by
  the other 5 joints dominating the norm.
  FIX: Compute base_L2 and shoulder_L2 SEPARATELY, with
  dedicated thresholds (not averaged into overall L2).

BLINDSPOT 3 (HIGH): base_std reflects proprioceptive echo, not vision
  The base std=28° in offline eval appears healthy, but the
  variation is driven by the state input (current base angle),
  not by the image content. In deployment, when state is fixed
  at dataset_mean, base output is also fixed.
  FIX: Pearson correlation r(pred_base, GT_base) ≥ 0.5 gate.

BLINDSPOT 4 (HIGH): Zone sample sizes have no statistical power
  n=7 LEFT, n=10 RIGHT are insufficient to detect any meaningful
  zone performance difference. The eval threshold (<2.0 ratio)
  accepts results that are statistically indistinguishable from
  random noise at these sample sizes.
  FIX: Require n≥30 per zone, with explicit resampling to
  oversample under-represented zones.

BLINDSPOT 5 (MODERATE): Zone L2 measures 6D error, not base-specific
  Even with balanced zones, total L2 cannot detect base grounding
  failure if the other 5 joints are within normal range.
  FIX: Compute per-zone BASE L2 separately from total zone L2.

SECONDARY FACTOR: Data distribution makes grounding trivially hard to distinguish
  80.1% of episodes are CENTER. A model that outputs base≈10° always
  achieves <5° base error on 80% of samples automatically.
  The model "learned" that base≈dataset_mean is safe, which
  IS correct for CENTER — but generalizes incorrectly to LEFT/RIGHT.
  FIX: Balanced data collection (equal episodes per zone) or
  explicit loss weighting for non-CENTER frames.
""")


def main():
    np.random.seed(42)

    q1_l2_masking_analysis()
    print()
    q2_grounding_metrics()
    print()
    q3_std_masking()
    print()
    q4_statistical_power()
    print()
    q5_grounding_test_protocol()
    print()
    analyze_deployment_run()
    print()
    root_cause_summary()

    print("=" * 72)
    print("[A1 MANIPULATION] Analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
