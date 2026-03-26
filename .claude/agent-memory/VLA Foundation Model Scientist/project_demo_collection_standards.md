---
name: VLA Demo Collection Standards
description: VLA demonstration data collection standards — episode structure, gripper behavior, quality criteria, minimum lengths from SmolVLA/ALOHA/ACT/BridgeData V2 literature vs RoArm-M3 current practice
type: project
---

Standard episode pattern (all major VLA papers):
Phase 1 (0-15%): Home → approach [arm down+out]
Phase 2 (15-35%): Pre-grasp [GRIPPER OPENS HERE, arm still descending]
Phase 3 (35-55%): Arm at object height, gripper open
Phase 4 (55-65%): Gripper closes [THE grasp moment]
Phase 5 (65-85%): Lift [arm up with object]
Phase 6 (85-100%): Return to home or handoff

Key numbers from literature:
- SmolVLA official: 393 frames avg @ 30fps = 13sec
- ALOHA/ACT: 400 timesteps @ 50Hz = 8sec = ~240 frames @ 30fps
- Diffusion Policy: 100 frames minimum cited
- LeRobot default episode_time_s=60 is a ceiling not target

Minimum frame threshold:
- SmolVLA chunk_size=50 → need 3 chunks minimum = 150 frames (5 sec)
- Current code FAIL at 90, WARN at 120 — too permissive
- Recommendation: FAIL at 120, WARN at 150

Gripper standards:
- Opens during approach phase (Phase 2), stays open through Phase 3-4
- Open threshold: 40° is minimum floor; 60°+ is target for RoArm-M3
- Close detection: absolute threshold (< 25°) better than relative (< 50% of max)
- v1 failure root cause: 58% of episodes closed gripper too early (before reaching object)

Shoulder vs Z check at grip close:
- Shoulder > 50° at close: reasonable but zone-dependent (NEAR zone may close at lower shoulder)
- Z < 130mm at close: more reliable primary check (physically meaningful)
- Recommendation: Z > 130mm = FAIL, shoulder < 40° = WARN only

Starting position:
- ALOHA/ACT: fixed home between episodes (scripted reset)
- BridgeData V2: varied (but 60K+ episodes scale)
- At 150ep scale: fixed home preferred
- Current practice: varied start (wherever previous ep ended) — acceptable but confirm dataset_mean is in-distribution for deployment

Missing validation checks:
- No post-grasp lift detection in current code
- Recommendation: check z_after_close > z_at_grip_close + 20mm within 1sec of grip close

Episode start position note:
- collect_data_manual.py calls move_init() only at startup
- Between episodes arm stays at end position
- deploy_smolvla.py uses dataset_mean start — must ensure this is in-distribution

**Why:** Research synthesis to guide data collection quality standards for Stage 1 (150ep) collection.
**How to apply:** When reviewing episode quality criteria or adjusting validate_episode() thresholds in collect_data_manual.py.
