---
name: V5 Dataset Cross-Validation Analysis (2026-03-26)
description: Full quantitative analysis of collected_data_v5/ — 136 episodes, 5 zones, key findings and risks
type: project
---

## V5 Dataset State (2026-03-26 analysis)
- Location: `collected_data_v5/` (136 episodes: episode_0000 to episode_0137, some IDs skipped)
- Task: "Pick up the sponge" (sponge on various surface heights/positions)
- Total: 136 episodes, 13,470 frames, mean 99.0 frames/ep (3.3s @ 30 FPS)
- Script: `data_v5_crossvalidation_v2.py` → output: `claudedocs/DATASET_V5_ANALYSIS.md`
- Training readiness: **9/10** (by automated score), **8/10** (adjusted for constraints)

## Zone Distribution
- FAR_CENTER: 39 eps (28.7%) — READY
- NEAR: 30 eps (22.1%) — READY
- MID_RIGHT: 27 eps (19.9%) — READY
- MID_LEFT: 25 eps (18.4%) — READY
- OVERHEAD: 15 eps (11.0%) — MARGINAL (collect 5 more)

## dataset_mean (for deployment starting position)
- dataset_mean = [9.93, 44.10, 40.94, 67.18, 0.20, 28.08]
- dataset_std  = [30.96, 16.05, 32.33, 28.55, 26.60, 20.39]
- Joints: [Base, Shoulder, Elbow, WristP, WristR, Gripper]

## CRITICAL: Episodes Start at Approach Pose (NOT home)
- V5 episodes start with shoulder~44°, elbow~36° (already at approach)
- V3 episodes started at init (shoulder~2.5°, elbow~90°)
- Deployment MUST use `--start-pos dataset_mean` and pre-position arm to approach pose
- NEVER use `move_init()` before inference — places arm at 2.6σ OOD from start

## CRITICAL: Gripper Closed Frames Underrepresented
- Gripper <15° (strict closed): only 7.5% of frames
- Gripper <20° (sponge-gripped): 57.8% of frames — CORRECT threshold for soft objects
- Gripper >40° (open): 23.0% of frames
- Gripper 15-40° (mid/ambiguous): 69.5% of frames
- KEY: sponge grasp = ~18-20° (not <10°), so <20° is the correct "gripped" criterion
- Risk: model may learn to stop at ~18° without firmly gripping if closed-signal is weak

## Gripper Temporal Pattern (big change from v3)
- Grip opens at frame ~9.4 (9.5% into episode) — EARLY (vs v3: frame 58.6, 33%)
- Grip closes at frame ~29 (vs v3: frame 98.7)
- Open duration: ~20 frames (0.67s)
- Episodes start ALREADY positioned → gripper opens immediately
- Entire open→close transition fits within first 50-step chunk

## Elbow Bimodality (MODERATE RISK)
- Bimodal distribution: cluster at 0-30° (deep grasp, 57%) and 72-115° (return, 37%)
- Dead zone at 42-60° — only ~5% of frames
- mean=40.9° SITS IN THE DEAD ZONE
- If mean regression occurs, elbow will stall at ~40-50° (physically unusual)
- Mitigation: use open-loop n-chunks=4 to commit through transitions

## WristR Zone Compensation
- Near-neutral dominant (0°): center zones
- -54° cluster: MID_LEFT compensation
- +54° cluster: MID_RIGHT compensation
- Model must learn WristR shift from visual observation of object position

## Phase Completeness
- 136/136 = 100% with zone-aware criteria (OVERHEAD: positive Z is correct)
- 0 flagged episodes (zero quality failures)
- 100% grip-close detection rate across ALL 5 zones

## OVERHEAD Zone Special Properties
- 15 episodes, all with z_at_grip_close > 0mm (elevated grasp, not table-surface)
- 10/15 eps: elbow_range < 5° (arm uses wrist pitch, not elbow extension)
- Kinematically different from other zones — needs sufficient representation

## V3 → V5 Key Regressions
- Gripper <15% fell from 31.6% → 7.5% (shorter episodes, no return phase)
- Frames/ep fell from 177.6 → 99.0 (episodes 44% shorter)
- Start position changed from home to approach pose (deployment constraint)

## V3 → V5 Key Improvements
- Zone diversity: 1 zone → 5 zones
- Phase completion: ~80% → 100%
- Quality flags: several → 0
- Grip-close detection: ~65% → 100%
- Duration consistency: std=1.0s → std=0.3s (much tighter)
