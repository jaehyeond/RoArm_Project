---
name: feedback-p0-substep-ordering
description: P0 cube calib substep ordering rule: measurement substeps must precede gate substep; scripted gripper sweep vs L-F teleop boundary
metadata:
  type: feedback
---

# P0 Substep Ordering Rule

Rule: Gate substep (N/N success criterion) must come AFTER all measurement substeps that define the parameters being used in that gate. Putting a 5/5 gate before sweep/angle/z calibration is a logical inversion.

**Why:** identified during 2026-05-26 P0 substep cross-validation. Original order put 1.3 (5/5 grasp gate) before 1.4 jaw sweep, 1.5 grasp z, 1.6 approach angle — undefined parameters at gate time.

**How to apply:** reordered as: HW sanity → jaw sweep (scripted) → approach angle compare (scripted) → grasp z measure (combined) → consolidated gate with locked-in params.

Scripted vs L-F boundary:
- Jaw cmd precision sweep = follower-only scripted (L-F operator cannot hold exact cmd angle reliably)
- Approach angle compare = follower-only scripted (controlled variable, one angle at a time)
- Grasp z measurement = combined: scripted for TCP z readout, L-F for natural grip depth
- Consolidated gate 5/5 = L-F teleop (realistic operating condition test)
