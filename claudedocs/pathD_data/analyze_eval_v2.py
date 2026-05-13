"""Path D Phase D.3 v2 — Critical analysis of release_bc eval with per-env gripper_q@s.

v1 → v2 delta: eval now captures gripper_q at the EXACT step `_place_success_flag` fires.
We can now split nominal success into CLEAN (gripper actually open at success) vs
counter-path artifact (gripper still closed but env counted success via _place_counter≥50).

Decision matrix (user-specified):
  Clean ≥30% → PATH D real PASS → subskill expansion
  Clean 10-30% → BC capacity or demo source improvement
  Clean <10% → PATH D FAIL → procedural release demos (SkillGen/MimicGen pivot)
"""
import torch
import numpy as np

d = torch.load("eval_metrics_v2.pt", weights_only=False)
succ_step = d["success_step"]              # [256], -1 = no success
gq_at_s = d["gripper_q_at_success"]        # [256], NaN where no success
gor = d["gripper_open_rate"]                # [256], episode mean open fraction
N = succ_step.shape[0]
CLEAN_TH = float(d["clean_thresh_rad"])
GRASP_TH = float(d["grasp_thresh_rad"])

succ_mask = succ_step >= 0
clean_mask = succ_mask & (gq_at_s < CLEAN_TH)
n_succ = int(succ_mask.sum().item())
n_clean = int(clean_mask.sum().item())
n_dirty = n_succ - n_clean
n_fail = N - n_succ

print("=" * 72)
print("PATH D PHASE D.3 v2 — RELEASE_BC EVAL ANALYSIS")
print("=" * 72)
print(f"Trials                    : {N}")
print(f"Nominal success           : {n_succ} ({100*n_succ/N:.2f}%)")
print(f"  └─ CLEAN (gq<{CLEAN_TH:.2f}rad)   : {n_clean} ({100*n_clean/N:.2f}%)  [TRUE direct-path]")
print(f"  └─ DIRTY (gq>={CLEAN_TH:.2f}rad)  : {n_dirty} ({100*n_dirty/N:.2f}%)  [counter-path artifact]")
print(f"Failure                   : {n_fail} ({100*n_fail/N:.2f}%)")
print()

# ============================================================
# success_step distribution: CLEAN vs DIRTY vs full
# ============================================================
ss_succ = succ_step[succ_mask].float()
ss_clean = succ_step[clean_mask].float()
ss_dirty = succ_step[succ_mask & ~clean_mask].float()
print("=== success_step distribution ===")
print(f"  All success  : mean={ss_succ.mean():.1f}  median={ss_succ.median():.0f}  "
      f"min={ss_succ.min():.0f}  max={ss_succ.max():.0f}")
if n_clean > 0:
    print(f"  CLEAN only   : mean={ss_clean.mean():.1f}  median={ss_clean.median():.0f}  "
          f"min={ss_clean.min():.0f}  max={ss_clean.max():.0f}")
if n_dirty > 0:
    print(f"  DIRTY only   : mean={ss_dirty.mean():.1f}  median={ss_dirty.median():.0f}  "
          f"min={ss_dirty.min():.0f}  max={ss_dirty.max():.0f}")
print()

# ============================================================
# Early (s<50, DIRECT-only possible) vs Late (s>=50, counter candidate)
# ============================================================
COUNTER_TH = 50
early_mask = succ_mask & (succ_step < COUNTER_TH)
late_mask = succ_mask & (succ_step >= COUNTER_TH)
clean_early = clean_mask & (succ_step < COUNTER_TH)
clean_late = clean_mask & (succ_step >= COUNTER_TH)
print(f"=== Early (s<{COUNTER_TH}) vs Late split ===")
print(f"  Early success    : {int(early_mask.sum())} / {n_succ} "
      f"({100*int(early_mask.sum())/max(n_succ,1):.1f}%)")
print(f"    └─ CLEAN early : {int(clean_early.sum())} ({100*int(clean_early.sum())/N:.2f}% of trials)")
print(f"  Late success     : {int(late_mask.sum())} / {n_succ} "
      f"({100*int(late_mask.sum())/max(n_succ,1):.1f}%)")
print(f"    └─ CLEAN late  : {int(clean_late.sum())} ({100*int(clean_late.sum())/N:.2f}% of trials)")
print()

# ============================================================
# gripper_q@s distribution
# ============================================================
gq_succ = gq_at_s[succ_mask]
print(f"=== gripper_q@s distribution (rad; grasp_thresh={GRASP_TH:.3f}, clean_thresh={CLEAN_TH:.2f}) ===")
print(f"  All success : mean={gq_succ.mean():.3f}  median={gq_succ.median():.3f}  "
      f"std={gq_succ.std():.3f}  min={gq_succ.min():.3f}  max={gq_succ.max():.3f}")
bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20, 1.60]
hist, _ = np.histogram(gq_succ.numpy(), bins=bins)
print(f"  Histogram   :")
for h, e0, e1 in zip(hist, bins[:-1], bins[1:]):
    marker = " <-- CLEAN" if e1 <= CLEAN_TH else ""
    bar = "#" * h
    print(f"    [{e0:.2f}, {e1:.2f}): {h:>3} {bar}{marker}")
print()

# ============================================================
# GOR (gripper_open_rate) correlation with success types
# ============================================================
gor_clean = gor[clean_mask]
gor_dirty = gor[succ_mask & ~clean_mask]
gor_fail = gor[~succ_mask]
print("=== gripper_open_rate (episode mean) by success type ===")
def _stats(t, label):
    if t.numel() == 0:
        return f"  {label}: <empty>"
    return f"  {label:<15}: mean={t.mean():.3f}  median={t.median():.3f}  std={t.std():.3f}  n={t.numel()}"
print(_stats(gor_clean, "CLEAN success"))
print(_stats(gor_dirty, "DIRTY success"))
print(_stats(gor_fail, "Failure"))
print()

# ============================================================
# Baseline comparison & decision gate
# ============================================================
BASELINE = 0.0781  # P6v14a 20/256
print("=== BASELINE COMPARE ===")
print(f"  P6v14a alone (prior)        : 20/256 = {100*BASELINE:.2f}%")
print(f"  release_bc nominal (v2)     : {n_succ}/{N} = {100*n_succ/N:.2f}%  "
      f"(Δ {100*(n_succ/N - BASELINE):+.2f}pp — inflated by counter-path)")
print(f"  release_bc CLEAN (v2)       : {n_clean}/{N} = {100*n_clean/N:.2f}%  "
      f"(Δ {100*(n_clean/N - BASELINE):+.2f}pp — apples-to-apples)")
print()

# Apples-to-apples vs P6v14a needs same filter applied to baseline. Without P6v14a's
# gripper_q@s data, we approximate with v1's "early (s<50) ~6.25%" as upper bound for
# P6v14a direct-path fraction.
print("=== DECISION (user matrix) ===")
clean_rate = n_clean / N
if clean_rate >= 0.50:
    print(f"  CLEAN {100*clean_rate:.2f}% ≥ 50% → PATH D real PASS, publishable")
elif clean_rate >= 0.30:
    print(f"  CLEAN {100*clean_rate:.2f}% in [30%,50%) → real PASS, subskill expansion")
elif clean_rate >= 0.10:
    print(f"  CLEAN {100*clean_rate:.2f}% in [10%,30%) → BC capacity 확장 또는 demo source 개선")
else:
    print(f"  CLEAN {100*clean_rate:.2f}% < 10% → PATH D FAIL")
    print(f"  Pivot recommendation: SkillGen/MimicGen procedural release demo")
    print(f"  (IK pose +5cm above target, scripted gripper-open 10 step, gravity drop)")
print()

# ============================================================
# Failure diagnostics
# ============================================================
print("=== Failure analysis ===")
print(_stats(gor_fail, "gor failure"))
n_fail_low_gor = int((gor_fail < 0.05).sum())
n_fail_mid_gor = int(((gor_fail >= 0.05) & (gor_fail < 0.2)).sum())
n_fail_high_gor = int((gor_fail >= 0.2).sum())
print(f"  Failure GOR breakdown:")
print(f"    GOR < 0.05 (gripper always closed): {n_fail_low_gor}/{n_fail}")
print(f"    GOR 0.05-0.20                      : {n_fail_mid_gor}/{n_fail}")
print(f"    GOR ≥ 0.20 (opened but missed)     : {n_fail_high_gor}/{n_fail}")
print()

# ============================================================
# Bottom-line summary
# ============================================================
print("=" * 72)
print("VERDICT")
print("=" * 72)
inflate_factor = (n_succ / max(n_clean, 1))
print(f"  Counter-path inflation factor : {inflate_factor:.1f}× (nominal/CLEAN)")
print(f"  Δ CLEAN vs P6v14a baseline    : {100*(n_clean/N - BASELINE):+.2f} pp")
print(f"  BC's learned skill            : 'hover near target with gripper closed'")
print(f"                                  (gq@s mean {gq_succ.mean():.2f} rad ≈ "
      f"{np.rad2deg(gq_succ.mean().item()):.1f}°, well above {np.rad2deg(CLEAN_TH):.0f}° clean threshold)")
print(f"  D.1 demo source contaminated  : P6v14a rollout itself = same hover pattern")
print(f"  → procedural (IK+scripted) demos likely necessary for real release skill")
