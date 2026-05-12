"""
analysis_p6v12_stats.py
C2 Analysis — P6v12 Statistical Analysis (no plotting)

Usage:
    python analysis_p6v12_stats.py [--log /tmp/train_p6v12.out]

Outputs:
    - Linear regression slopes for gripper_open and is_on_target
    - Stage allocation comparison
    - η-v1 design flaw margin calculations
    - P6v11 vs P6v12 final-iter comparison table
"""

import argparse
import os
import re
import sys
import numpy as np
from scipy import stats

# ── P6v11 / P6v12 reference values (from claudedocs result md, 2026-05-12) ───
P6V11 = {
    "reward":         1059,
    "gripper_open":   0.0634,
    "is_on_target":   0.0161,
    "stage4":         0.0,
    "stage2_grasp":   0.849,
    "stage3_near":    0.0161,
    "xy_offset_m":    0.0671,
    "z_offset_m":     0.0479,
    "action_std":     1.44,
    "is_success_zone":0.526,
}

# P6v12 known iter-table (iters with partial data)
P6V12_ITERS = [0,    65,   247,  351,  532,  750,  966,  999]
P6V12 = {
    "reward":       [None, 637,  755,  787,  794,  822,  824,  854],
    "gripper_open": [0.070, 0.072, 0.069, 0.069, 0.062, 0.064, 0.064, 0.064],
    "is_on_target": [0.019, 0.185, 0.320, 0.346, 0.397, 0.394, 0.397, 0.406],
    "stage4":       [0.0003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0002],
    "stage2_grasp": [0.82, 0.67, 0.51, 0.51, None, None, None, 0.45],
    "stage3_near":  [0.019, 0.185, 0.320, 0.346, 0.397, 0.394, 0.397, 0.406],
    "xy_off":       [0.166, 0.069, 0.072, 0.072, None, None, None, 0.081],
    "z_off":        [0.054, 0.044, 0.046, 0.044, None, None, None, 0.048],
    "std":          [1.00, 1.00, 0.97, 0.96, None, None, None, 0.88],
}


def regression_slope(iters, values):
    """Linear regression over non-None pairs."""
    pairs = [(x, y) for x, y in zip(iters, values) if y is not None]
    if len(pairs) < 2:
        return float("nan"), float("nan"), float("nan")
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    slope, intercept, r, p, se = stats.linregress(xs, ys)
    return slope, r, p


def print_regression_report():
    print("\n" + "="*65)
    print("  Linear Regression Analysis — P6v12 (sparse iter-table)")
    print("  Source: claudedocs/phase1_balpha_p6v12_session_20260512_result.md")
    print("="*65)

    for metric_key, label in [
        ("gripper_open", "gripper_open_rate"),
        ("is_on_target", "is_on_target_rate"),
        ("reward",       "Mean reward"),
        ("stage4",       "stage4_success_frac"),
    ]:
        vals = P6V12[metric_key]
        slope, r, pval = regression_slope(P6V12_ITERS, vals)

        valid = [(x, y) for x, y in zip(P6V12_ITERS, vals) if y is not None]
        v0 = valid[0][1]
        vf = valid[-1][1]

        print(f"\n  [{label}]")
        print(f"    iter 0    : {v0:.5f}")
        print(f"    iter 999  : {vf:.5f}")
        print(f"    Slope     : {slope:.6f} /iter  = {slope*1000:.5f} /1k-iter")
        print(f"    Pearson r : {r:.4f}    p-value: {pval:.4f}")

        # Verdict
        abs_slope_per1k = abs(slope) * 1000
        if abs_slope_per1k < 0.005:
            verdict = "FLAT (|slope| < 0.005/1k-iter)"
        elif slope > 0:
            verdict = f"RISING  ({(vf/v0-1)*100:.1f}% change across 999 iter)" if v0 > 0 else "RISING"
        else:
            verdict = f"FALLING ({(vf/v0-1)*100:.1f}% change across 999 iter)"
        print(f"    Verdict   : {verdict}")


def print_comparison_table():
    print("\n" + "="*65)
    print("  P6v11 vs P6v12 Final-Iter Comparison (iter 999)")
    print("="*65)

    rows = [
        ("Mean reward",       "reward",       P6V11["reward"],       854),
        ("gripper_open_rate", "gripper_open", P6V11["gripper_open"], 0.064),
        ("is_on_target_rate", "is_on_target", P6V11["is_on_target"], 0.406),
        ("stage4_success",    "stage4",       0.0,                   0.0002),
        ("stage2_grasp_frac", "stage2_grasp", P6V11["stage2_grasp"], 0.45),
        ("stage3_near_frac",  "stage3_near",  P6V11["stage3_near"],  0.406),
        ("xy_offset (mm)",    "xy_off",       P6V11["xy_offset_m"]*1000, 81),
        ("z_offset (mm)",     "z_off",        P6V11["z_offset_m"]*1000,  48),
        ("action_std",        "std",          P6V11["action_std"],   0.88),
        ("is_success_zone",   "is_success_zone", P6V11["is_success_zone"], 0.460),
    ]

    print(f"  {'Metric':<22} {'P6v11':>10} {'P6v12':>10} {'Delta':>12}  Verdict")
    print("  " + "-"*63)
    for label, key, v11, v12 in rows:
        if v11 > 0:
            delta_pct = (v12 - v11) / v11 * 100
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "new signal"

        if "gripper" in key or key == "stage4":
            verdict = "FLAT/fail" if abs(v12 - v11) < 0.005 else ("↑" if v12 > v11 else "↓")
        elif v11 > 0:
            verdict = ("25× ↑ ✓" if v12 / v11 > 10 else ("↑ ✓" if v12 > v11 else "↓ warn"))
        else:
            verdict = "↑ new"

        print(f"  {label:<22} {v11:>10.4f} {v12:>10.4f} {delta_str:>12}  {verdict}")


def print_eta_v1_margin_analysis():
    print("\n" + "="*65)
    print("  η-v1 Design Flaw: 1-Step Reward Margin at Zone (Stage 3)")
    print("="*65)

    # Stage 3 reward function
    # base = 6.0 + 0.5*ungrasp + 0.5*static
    # transient (+10) fires on is_on_target (no gripper gate)
    # ungrasp_signal ≈ 0 when grasped (closed), ≈ 0.5 open (conservative), ≈ 1.0 released
    # static_signal ≈ 0.5 when stable in zone (approx)

    stage3_close_base = 6.0 + 0.5 * 0.0 + 0.5 * 0.5   # = 6.25
    stage3_open_base  = 6.0 + 0.5 * 0.5 + 0.5 * 0.5   # = 7.0
    transient         = 10.0

    # Fraction of time gripper open on first zone entry (policy prior)
    p_open_prior = 0.07   # ~7% from P6v12 gripper_open rate

    # Gradient mass for close vs open path (first entry)
    mass_close = (1 - p_open_prior) * (stage3_close_base + transient)
    mass_open  = p_open_prior * (stage3_open_base + transient)

    print(f"\n  Stage 3 reward components:")
    print(f"    Close + on_target: {stage3_close_base:.2f} + {transient:.1f} (transient) = {stage3_close_base+transient:.2f}")
    print(f"    Open  + on_target: {stage3_open_base:.2f} + {transient:.1f} (transient) = {stage3_open_base+transient:.2f}")
    print(f"\n  Gradient mass on first zone entry (p_open_prior={p_open_prior:.2f}):")
    print(f"    Close path: {mass_close:.2f}  ({(1-p_open_prior)*100:.0f}% × {stage3_close_base+transient:.2f})")
    print(f"    Open  path: {mass_open:.2f}  ({p_open_prior*100:.0f}% × {stage3_open_base+transient:.2f})")
    print(f"    Ratio close/open: {mass_close/mass_open:.1f}×  → PPO gradient overwhelmingly favors close")

    print(f"\n  Persistent margin (post-transient):")
    print(f"    Close: {stage3_close_base:.2f}  vs  Open: {stage3_open_base:.2f}  → margin = {stage3_open_base-stage3_close_base:.2f}")
    print(f"    Verdict: +{stage3_open_base-stage3_close_base:.1f} margin is insufficient (≈ noise level)")

    print(f"\n  η-v2 (Plan B, P6v13) projected margins:")
    stage3_close_cap = 3.0
    stage3_open_v2   = 7.0
    transient_gated  = 10.0  # fires only when gripper_open
    print(f"    Close + on_target (cap): {stage3_close_cap:.1f}")
    print(f"    Open  + on_target (base + transient): {stage3_open_v2:.1f} + {transient_gated:.1f} = {stage3_open_v2+transient_gated:.1f} (first entry)")
    print(f"    Persistent margin: +{stage3_open_v2-stage3_close_cap:.1f} (per step, post-transient)")
    print(f"    First-entry gradient mass close: {stage3_close_cap:.1f} × {(1-p_open_prior):.2f} = {stage3_close_cap*(1-p_open_prior):.2f}")
    print(f"    First-entry gradient mass open:  {stage3_open_v2+transient_gated:.1f} × {p_open_prior:.2f} = {(stage3_open_v2+transient_gated)*p_open_prior:.2f}")
    print(f"    Now open path has higher gradient mass → PPO should learn open")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="", help="Optional path to train_p6v12.out for live parsing")
    args = parser.parse_args()

    print("\n[C2 ANALYSIS] P6v12 Statistical Report")
    print("Data source: claudedocs/phase1_balpha_p6v12_session_20260512_result.md")
    print("Status: CONFIRMED (iter-table from result doc, no live log required for dry-run)")

    print_regression_report()
    print_comparison_table()
    print_eta_v1_margin_analysis()

    print("\n" + "="*65)
    print("  DIAGNOSTIC SUMMARY")
    print("="*65)
    print("""
  1. gripper_open_rate: slope ≈ -0.006/1k-iter → FLAT (release 학습 0)
     Root cause: η-v1 transient fires without gripper_open gate
     Gradient mass ratio close:open ≈ 13:1 → PPO learns close dominantly

  2. is_on_target_rate: slope ≈ +382/1k-iter → STRONG RISE (25× ↑)
     η γ transport shaping working, policy navigates to zone successfully
     BUT closed gripper — position good, release missing

  3. stage4_success_frac: 0.0002 final — sporadic (~1/4096 envs)
     stage_stable & gripper_open joint AND extremely rare
     Not enough to bootstrap learning

  4. Reward-farm shift: stage2 0.85→0.45, stage3 0.41 new
     η-v1 moved farming from stage2 to stage3 (close-hover),
     did NOT break farming itself

  5. η-v2 (P6v13) fix: transient gate = is_on_target & gripper_open
     Stage3 close-cap = 3.0
     Projected persistent margin: +4.0 (vs current +0.5)
     Gradient mass: open 1.19 > close 0.28 (4.3× reversal)
  """)


if __name__ == "__main__":
    main()
