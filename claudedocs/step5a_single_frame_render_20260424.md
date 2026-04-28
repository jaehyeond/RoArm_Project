# Step 5a — Single-frame Sim Render Verification (2026-04-24)

## Goal
Boot Isaac Sim with calibrated Kinect pose and render ep0 frame 0 to validate:
- Scene geometry (table + robot + sponge placement)
- Calibrated camera extrinsics produce reasonable perspective
- URDF articulation loads and joints can be set

Before running full-episode loop (5b) or SigLIP (5c).

## Setup
- Script: `sim_scripts/replay_v6_sim.py`
- Conda env: `isaaclab` (Isaac Sim 5.1)
- Inputs:
  - URDF: `isaac_roarm_m3/.../urdf/roarm_m3.urdf` (wrist_pitch +/-110° fix from 2026-04-14)
  - Calib: `sim_scripts/kinect_calib.yaml` (RMSE 10.13mm)
  - Sponge pose: `sim_scripts/sponge_poses.json` (depth-based, 50 eps)
  - Table Z: `sim_scripts/table_plane.json` (-12.12 mm URDF world, std 0.40mm)

## Results
- Render completed, `sim_renders_v2/ep0000_frame0000.png` saved (294 KB).
- Isaac Sim boot time: ~120 s (URDF parse ~5s, articulation load + settle ~110s).
- Visual comparison vs real ep0 frame 0: see adjacent files.

| Aspect | Sim | Real | Match |
|---|---|---|---|
| Camera perspective (front-above) | ✓ | ✓ | ✓ |
| Robot at HOME, centered | ✓ | ✓ | ✓ |
| Sponge position on table | ✓ (center zone) | ✓ | ✓ |
| Table color | Dark gray (SeattleLabTable) | White | **mismatch** |
| Background | White void (dome light) | Black couch + wall + cables | **mismatch** |
| Robot materials | Default gray | Black plastic | mismatch |
| Sponge orientation | Identity quat (upright) | Upright | likely OK |

## Findings

1. **Frame coordinate chain is self-consistent**:
   - Calib extrinsics = URDF world frame (confirmed Step 3 + Step 4).
   - Sponge_poses.json positions load correctly into sim.
   - URDF import places `base_link` at world Z = +70.1mm automatically (fixed joint).
   - Table placed at Z = -12.1mm matches measured surface height.

2. **Gaps for SigLIP (not yet measured)**:
   - Background: single biggest source. Real has textured indoor environment.
   - Table material: SeattleLabTable is industrial dark gray, real is flat white.
   - Robot materials: URDF has STL meshes but no colors → default material.

3. **Minor issue noted**: `compute_sponge_poses.py` flagged ep0 as "tilt 61.4deg > 25.0" and emitted identity quaternion. Real ep0 sponge appears upright. The tilt was likely a false alarm from PCA on partial-visibility depth points. Ep0 fallback to identity worked visually.

## Gotchas Resolved

- `from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR` **fails** — the base `isaaclab` Python package is not editable-installed (only `isaaclab_assets/tasks/rl/mimic/contrib`). Workaround: read Nucleus root directly from `carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")` after `SimulationApp()` boots.

- `conda run -n isaaclab python` swallows script stdout in this workflow (stderr emits only verbose Isaac logs). For debugging, always check output files, not terminal output.

## Next — Before Step 5b
Optional quick fixes to improve SigLIP before running full loop (~10-15 min work):
- **Table material override**: apply white diffuse PreviewSurface to SeattleLabTable top face. **High value** (real table is white).
- **HDRI dome light**: replace solid dome with an office/lab HDRI. Moderate value, moderate effort.
- **Robot material colors**: optional, lower priority.

## Files
- Script: `sim_scripts/replay_v6_sim.py`
- Output: `sim_renders_v2/ep0000_frame0000.png`
- Real reference: `collected_data_v6/episode_0000/rgb_0000.jpg`
