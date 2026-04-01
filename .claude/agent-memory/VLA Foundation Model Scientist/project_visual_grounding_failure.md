---
name: Visual Grounding Failure Analysis (v5 deployment)
description: Root cause analysis of base≈10° failure despite PASS offline eval — data imbalance + L2 metric blindness
type: project
---

V5 model (136ep, 200K steps) always goes to base≈10° regardless of sponge position.

**Why:** 3 compounding root causes:

1. DATA IMBALANCE (primary)
   - LEFT: 2/136 eps (1.5%) → model never learned left
   - CENTER: 109/136 eps (80.1%) → model memorized center
   - LEFT:CENTER gradient ratio = 1:53 over 200K steps
   - Bayes-optimal given training distribution IS base=10°

2. L2 METRIC IS BLIND TO THIS
   - Base contributes only 19.5% of total MSE (vs Elbow 25.9%, WristP 22.1%)
   - A 50° base error adds ~13% to a ~57° non-base L2 floor
   - Constant predictor RMS L2 ≈ 63.5° → reported model 3.80° = 94% improvement
   - But model is NOT predicting base correctly for non-center zones

3. ZONE L2 RATIO = 1.19 IS MISLEADING
   - LEFT n=7 in eval → 95%CI spans ±1.6° → insufficient power
   - Ratio tests overall error magnitude, NOT positional accuracy
   - A model always predicting base=10° could still show ratio ≈ 1.0
     because non-base joints dominate the L2

**SigLIP can distinguish position in principle** (729 tokens, 27px/patch, spatial positional encoding preserved). Failure is training dynamics, not architecture.

**Fix priority:**
1. Collect 28+ LEFT episodes (from 2 → 30)
2. Add 5 RIGHT episodes (from 25 → 30)
3. Rebalance CENTER to ~50 or use weighted_sampling=3x on L/R
4. Replace L2-based zone eval with per-zone base-angle error metric

**New required eval metric:**
E[|pred_base - gt_base| | zone] per zone, reported separately.
Pass criterion: max(per-zone base error) < 20°.

**How to apply:** Before declaring "zone balanced", always check per-zone BASE-ANGLE error separately from total L2. L2 zone ratio masks positional failures when non-base joints dominate.
