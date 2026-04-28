# Step 5b + 5c — Full-episode Sim Replay + SigLIP Re-measurement (2026-04-24)

## Goal
- 5b: render all 103 frames of ep0 with calibrated Kinect pose + table + sponge, and log per-frame joint tracking error.
- 5c: recompute SigLIP cosine similarity vs the v6 MP4 and compare to the 2026-04-13 baseline (0.668) that triggered the NO-GO.

Gate: `mean cosine >= 0.70` → proceed to Step 8 (LeRobot v3 parquet writer) + Step 9 (A/B/C training). `< 0.70` → apply table white override + HDRI background and re-measure.

## Step 5b — Render All 103 Frames

### Script change
`sim_scripts/replay_v6_sim.py` `--all-frames`:
- Single Sim boot (~120s), then loop target joints per frame.
- Output layout now matches `sim_real_compare.py`: `sim_renders_v2/episode_<EEE>/frame_<FFFF>.png`.
- Per-frame joint tracking dumped to `sim_renders_v2/episode_<EEE>/tracking_rmse.json` (target, actual, err, rmse, max|err|, per-joint stats).

### Run
```
conda run -n isaaclab python sim_scripts/replay_v6_sim.py \
    --episode 0 --all-frames --output-dir sim_renders_v2
```
Exit 0. 103 PNGs written. Total disk: 31M.

### Tracking results

| Metric | Value |
|---|---|
| N | 103 |
| RMSE mean | 0.4265° |
| RMSE max | 0.7172° |
| global max \|err\| | 1.5546° |

Per-joint mean abs error (deg): base=0.0006, shoulder=0.1197, elbow=0.5345, **wrist_pitch=0.5799**, wrist_roll=0.271, gripper=0.2776.
Per-joint max abs error (deg):  base=0.0016, shoulder=0.1438, elbow=0.7526, **wrist_pitch=1.5546**, wrist_roll=0.6885, gripper=0.7125.

**Verdict**: tracking is tight. Wrist_pitch worst-case 1.55° is from physics stepping (`world.step(render=True) × 3` + `rep.orchestrator.step()` per frame) slightly displacing the direct `set_joint_positions` target. Not a runaway.

Note: 2026-04-13 `replay_trajectory_sim.py` hit 4e-6° because it skipped `world.step()`. Stepping is needed for the USD stage + annotator to stay in sync at each frame; 1.55° wrist_pitch drift is acceptable.

## Step 5c — SigLIP Cosine Similarity

### Run
```
conda run -n roarm python sim_real_compare.py \
    --sim-dir sim_renders_v2 --episode 0 --max-frames 103 \
    --output sim_real_compare_v2_ep0.json
```

### Result

| Stat | Value |
|---|---|
| N frames | 103 |
| **mean** | **0.7159** |
| std | 0.0270 |
| min | 0.6404 (frame 44) |
| max | 0.7755 (frame 16) |
| frames ≥ 0.70 | 82/103 (79.6%) |
| frames ≥ 0.75 | 13/103 (12.6%) |
| frames < 0.65 | 1/103 (1.0%) |

**Verdict: GO (0.7159 ≥ 0.70)**.

### Comparison vs 2026-04-13 baseline

| | 2026-04-13 | 2026-04-24 (this) |
|---|---|---|
| Sim env | blank ground, default material | SeattleLabTable + pink sponge cuboid at per-episode pose |
| Camera | approximate pose | **Kinect calibrated (RMSE 10.13mm)** |
| Table Z | n/a | fit from 25 ep depth, -12.12mm URDF world |
| Robot joints | replay (joint positions only) | replay + physics stepping per frame |
| **Mean SigLIP** | **0.668** | **0.7159 (+0.048)** |
| Verdict | NO-GO | **GO** |

### Gap sources that remain
Looking at low-scoring frames (frame 44 = 0.640):
- Background mismatch: sim has dome-light white void; real has black couch + wall + cables.
- Table color: SeattleLabTable is dark gray; real is white.
- Robot materials: URDF meshes render with default gray; real is black plastic.

These are knowable from Step 5a visual inspection. The GO margin is thin (+0.016). An optional hardening pass would push the mean higher with modest effort:
- **Table top material override** → white `UsdPreviewSurface`. ~5 min, high value.
- **HDRI dome light** (indoor office/lab HDRI) → replace solid intensity. ~10 min, moderate value.
- (Lower priority) Robot link colors.

## Decision

Gate is met → **user approves proceeding to Step 8**. But 0.716 is only +0.016 above threshold with 20% of frames below threshold. Surfaced to user:

- **Option A (proceed as-is)**: Step 8 parquet writer + Step 9 A/B/C training with current renders.
- **Option B (harden first)**: table white + HDRI → likely mean > 0.75 → more robust sim-aug signal for Step 9, ~15 min delta.

User to choose before Step 8 starts.

## Files Produced

- `sim_renders_v2/episode_000/frame_0000.png .. frame_0102.png` (103 files, 31M)
- `sim_renders_v2/episode_000/tracking_rmse.json` — per-frame joint error log
- `sim_real_compare_v2_ep0.json` — SigLIP cosine per frame + summary
- `logs/step5b_ep0_render.log / .err` — Isaac Sim stdout/stderr (stdout mostly swallowed by `conda run`; reconstruct via output files)
