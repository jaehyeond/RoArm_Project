---
name: project_v5_visual_grounding_failure
description: V5 136ep 200K training — base always ≈10°, sponge position ignored. Full root-cause chain documented.
type: project
---

## V5 Failure: Visual Grounding Missing (2026-03-31)

**Fact**: After 136ep/200K steps, model executes correct pick sequence but Base always ≈10° regardless of sponge position.

**Why**: Zone design had 3/5 zones sharing base≈0-15°. Data: 80.1% CENTER, 1.5% LEFT, 18.4% RIGHT. The 5-zone system gave appearance of diversity but was angular monoculture. Eval metrics (L2, std, zone-L2 ratio 1.19) all passed because they tested against the same biased distribution — not visual grounding.

**How to apply**: Any future eval script MUST include `base_prediction_variance_by_image_context` test. Zone L2 ratio measured against ground-truth is not sufficient — it measures self-consistency, not input-sensitivity. The key metric is: does the model's base prediction change as a function of sponge position in the image?

## Critical Eval Gap Identified

The `eval_v5_checkpoints.py` zone analysis (lines 157-174) splits by ground-truth base angle — which means LEFT zone samples already have base≈-39°. The model can appear "calibrated" per-zone even if it has zero visual grounding — if it learned "when GT base is -39°, predict -39°" via some other cue (temporal position in episode, not image content).

**The correct test**: Place sponge at position A, run inference. Move sponge to position B (same episode start), run inference again. If base prediction doesn't shift ≥ 20°, model has no visual grounding.

## Required New Metric: Visual Grounding Test

```
For N=5 known positions across ≥ 60° base range:
  - Capture single frame with sponge at each position
  - Run model inference on each frame independently
  - Report: base_prediction per position
  - Pass condition: base_prediction range ≥ 30° (model responds to spatial variation)
  - Fail condition: base_prediction range < 10° (mean-regressor behavior)
```

This test takes ~10 minutes to run and would have caught the failure before any real-world deployment.
