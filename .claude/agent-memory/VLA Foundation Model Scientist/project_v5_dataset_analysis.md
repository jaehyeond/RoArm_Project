---
name: v5 Dataset Analysis vs Published VLA Benchmarks
description: Cross-validated comparison of collected_data_v5 (136 ep) against 12 VLA datasets. Actual stats verified by script. 6 risk questions answered with evidence. 2026-03-26.
type: project
---

## Actual Dataset Stats (collected_data_v5, verified 2026-03-26)

- Total episodes: 136
- Total frames: 13,470
- Frames/episode: min=90, mean=99.0, max=152
- Duration (stored @ 30fps): min=3.0s, mean=3.3s, max=5.1s
- Zone distribution: FAR_CENTER=39, MID_LEFT=25, MID_RIGHT=27, NEAR=30, OVERHEAD=15
- Dual-cam episodes: 4 (ZED wrist), Single-cam: 132
- Mean gripper range: 62.1 degrees
- Mean elbow range: 50.7 degrees

**Why:** Baseline numbers needed for training config decisions and paper claims.
**How to apply:** Use these exact figures when discussing data sufficiency, training estimates, or paper claims.

## Zone Imbalance

FAR_CENTER=39 (28.7%) is the dominant zone. OVERHEAD=15 (11%) is underrepresented.
- Balanced target: 27/zone (136/5)
- Actual deviation: FAR_CENTER is 44% above balanced; OVERHEAD is 44% below
- Flag this in training analysis — model may over-fit to FAR_CENTER

## Critical Risk Verdicts (6 questions)

1. **136 episodes sufficient?** MARGINAL-OK. 27/zone < 50 threshold for OOD embodiment. Pooling helps.
2. **10fps recording?** REQUIRES VERIFICATION. If metadata claims 30fps but physical was 10fps via triplication, duration analysis is misleading. Mean 3.3s stored = 9.9s real at 10fps.
3. **Episode duration (3-5s stored)?** SHORT vs industry (13-27s). BUT if physical is ~10s real, maps to ~33-50 real frames at 10fps — borderline OK. Verify convert_to_lerobot_v3.py logic.
4. **Single camera?** ACCEPTABLE for sponge pick (gross manipulation). -8-15% penalty is for fine insertion tasks.
5. **Gripper range 0-122°?** REASONABLE. 122° may indicate over-extension (sponge compliance). Mean range 62.1° across 136 eps = sufficient variety.
6. **Red flags?** TWO: (a) Zone imbalance (FAR_CENTER=39), (b) Short stored durations (3.3s avg). Both manageable.

## SmolVLA Scheduler Issue (HIGH PRIORITY)

Default config: warmup=1K steps, decay=30K steps → decays to min LR (2.5e-6) by step 30K.
For 200K training: 170K steps at near-zero LR = effectively stopped learning.
**Fix:** Override `--policy.scheduler_decay_steps=180000` in run_official_train.py.

## Key Comparison Numbers

| Dataset | Ep/task | Duration | FPS | Cameras |
|---------|---------|----------|-----|---------|
| SmolVLA official | 50 | ~13s | 30 | 1-3 |
| ACT/ALOHA | 50 | ~15-20s | 50 | 4 |
| Diffusion Policy | 100-200 | ~10-30s | 10-25 | 1-2 |
| OURS | 136 (5-zone) | 3.3s stored | 30 | 1 |
| BridgeData V2 | 30-100 | ~15-25s | 5 | 2 |

## Key File

Full analysis: `/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/VLA_DATASET_COMPARISON.md`
Script: `/home/cgxr/Documents/Robotics/RoArm_Project/model_dataset_comparison.py`
