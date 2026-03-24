---
name: feedback_experiment_design
description: Confirmed design constraints for this project's experiments
type: feedback
---

Rule: Use open-loop 4-chunk deployment for evaluation (not closed-loop n=1).
**Why:** Closed-loop n=1 failed in real deployment due to per-step noise at grasp moment. Open-loop 4-chunk was the validated approach achieving 5/5 success.
**How to apply:** All evaluation scripts must use --open-loop --n-chunks 4 --start-pos init.

Rule: Evaluate from dataset_mean start position, not zero-position.
**Why:** Zero-position is OOD → mean regression → silent deployment failure. Learned from 2026-02-11 failures.
**How to apply:** Every eval protocol must enforce init/dataset_mean start.

Rule: N=20 trials minimum per condition for statistical validity.
**Why:** Binomial CI at N=20 for p=0.8 vs p=0.9 overlap — cannot distinguish. N=50 is preferred for CoRL claims.
**How to apply:** Design eval grids with N=20 as floor, N=50 for primary claims.
