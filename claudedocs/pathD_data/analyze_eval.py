"""Critical analysis of D.3 v1 eval results.

Question: 68.36% success — counter-path artifact OR genuine release?
Evidence to gather (from current eval_metrics.pt — limited):
  - success_step distribution split (early vs late)
  - gripper_open_rate per env: high for success vs low for failure?
  - If success envs have higher GOR → evidence of real release
  - If similar/lower → counter-path artifact dominant
"""
import torch
import numpy as np

d = torch.load("eval_metrics.pt", weights_only=False)
succ_step = d["success_step"]  # [256], -1 = no success
gor = d["gripper_open_rate"]   # [256], episode mean open fraction
N = succ_step.shape[0]

succ_mask = succ_step >= 0
n_succ = int(succ_mask.sum().item())
n_fail = N - n_succ

print(f"=== EVAL METRICS ANALYSIS ===")
print(f"Trials  : {N}")
print(f"Success : {n_succ} ({100*n_succ/N:.2f}%)")
print(f"Failure : {n_fail} ({100*n_fail/N:.2f}%)")

succ_step_succ = succ_step[succ_mask].float()
print(f"\nsuccess_step (only success envs):")
print(f"  mean={succ_step_succ.mean():.1f}  median={succ_step_succ.median():.0f}  "
      f"std={succ_step_succ.std():.1f}  min={succ_step_succ.min():.0f}  max={succ_step_succ.max():.0f}")

# Split: early (s < 50) vs counter-path-suspect (s >= 50)
# Rationale: COUNTER path needs ≥50 consecutive near-target steps. So earliest possible
# COUNTER-path fire is around t=50 (give or take depending on counter logic).
# DIRECT path can fire any time once gripper opens (no counter requirement).
COUNTER_THRESH = 50
early_mask = succ_mask & (succ_step < COUNTER_THRESH)
late_mask = succ_mask & (succ_step >= COUNTER_THRESH)
n_early = int(early_mask.sum().item())
n_late = int(late_mask.sum().item())
print(f"\nSplit by success_step:")
print(f"  Early (s < {COUNTER_THRESH}, DIRECT-path-only possible) : {n_early}/{n_succ} ({100*n_early/n_succ:.1f}%)")
print(f"  Late  (s >= {COUNTER_THRESH}, COUNTER-path candidate)    : {n_late}/{n_succ} ({100*n_late/n_succ:.1f}%)")

# Gripper open rate comparison
gor_succ = gor[succ_mask]
gor_fail = gor[~succ_mask]
gor_early = gor[early_mask]
gor_late = gor[late_mask]
print(f"\nGripper open rate (episode mean):")
print(f"  All success     : mean={gor_succ.mean():.3f}  median={gor_succ.median():.3f}")
print(f"  All failure     : mean={gor_fail.mean():.3f}  median={gor_fail.median():.3f}")
print(f"  Early success   : mean={gor_early.mean():.3f}  median={gor_early.median():.3f}")
print(f"  Late success    : mean={gor_late.mean():.3f}  median={gor_late.median():.3f}")

# Interpretation:
# - If gor_succ > gor_fail: success correlates with more open → likely real release
# - If gor_early >> gor_late: late successes are counter-path (gripper irrelevant)
print(f"\n=== CRITICAL INTERPRETATION ===")
if gor_succ.mean() > gor_fail.mean() * 1.5:
    print(f"  gor_succ ({gor_succ.mean():.3f}) >> gor_fail ({gor_fail.mean():.3f}) "
          f"→ Success correlated with open → REAL RELEASE evidence")
elif gor_succ.mean() < gor_fail.mean() * 0.8:
    print(f"  gor_succ ({gor_succ.mean():.3f}) << gor_fail ({gor_fail.mean():.3f}) "
          f"→ Inverse! Counter-path more likely")
else:
    print(f"  gor_succ ≈ gor_fail → AMBIGUOUS, gripper not differential")

if n_late / n_succ > 0.5:
    print(f"  {100*n_late/n_succ:.0f}% late success → COUNTER-PATH SUSPECT (dominant)")
elif n_early / n_succ > 0.7:
    print(f"  {100*n_early/n_succ:.0f}% early success → DIRECT-PATH LIKELY (dominant)")
else:
    print(f"  Mixed early/late ({n_early}/{n_late}) → mixed nature")

# success_step histogram
print(f"\n=== success_step histogram ===")
bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
hist, _ = np.histogram(succ_step[succ_mask].numpy(), bins=bins)
for h, e0, e1 in zip(hist, bins[:-1], bins[1:]):
    bar = "#" * int(h / 2)  # scale down
    print(f"  [{e0:>3}, {e1:>3}): {h:>3} {bar}")

# Failure analysis
print(f"\n=== Failure gor distribution ===")
print(f"  gor_fail stats: min={gor_fail.min():.3f}  max={gor_fail.max():.3f}  std={gor_fail.std():.3f}")
n_low_gor = int((gor_fail < 0.05).sum())
print(f"  Failure envs with gor < 0.05 (gripper always closed): {n_low_gor}/{n_fail}")
