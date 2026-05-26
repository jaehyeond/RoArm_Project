---
name: project-roarm-cube-task-context
description: Track B task pivot sponge → cube 3×3×3cm, P0 calib plan context for A1 agent
metadata:
  type: project
---

# Track B Cube Task Pivot Context (2026-05-26)

Task: 5 cube 3+2 pyramid stacking (L1=3, L2=2). P0 = single cube grasp calibration.
Gripper linear fit: `jaw_mm ≈ 0.75 * state°` validated 0~30° only. Cube 30mm width → cmd ~40° is extrapolation (outside validated range).
Sponge anchors #19/#20/#24 SUPERSEDED. Cube anchors pending P0 measurement.
HARD RULE #18: user-stated corrections take absolute priority.

**Why:** task switched from foam sponge (compressible, friction-stabilised) to rigid cube (slippage risk ↑, no compression grip assist). All prior grasp anchors invalid.
**How to apply:** treat tech_gripper_grasp_anchors.md as historical-only; cube anchors come from P0 measured sweep.
