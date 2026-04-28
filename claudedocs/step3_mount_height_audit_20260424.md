# Step 3 — Robot Mount Height Audit (2026-04-24)

## Goal
Verify URDF FK consistency with ESP32's built-in FK, and confirm robot `base_link` world Z placement for Isaac Sim replay.

## Method
For all 50 v6 episodes, compare:
- `URDF_FK(angles)` — our software FK in URDF world frame (Z=70.1mm offset to base_link)
- `ESP32 pose[3]` — ESP32's built-in FK from `collected_data_v6/episode_XXXX/metadata.json`

Samples: frame 0 (~HOME), frame mid, frame last → 150 total.

## Results (URDF − ESP32, mm)

| Where | dX mean±std | dY mean±std | dZ mean±std |
|---|---|---|---|
| frame0 (N=50) | −3.30 ± 0.81 | +0.00 ± 0.63 | **+122.37 ± 1.88** |
| frame_mid | −1.34 ± 0.67 | −0.18 ± 1.02 | **+124.88 ± 1.09** |
| frame_last | −2.66 ± 0.26 | −0.14 ± 0.25 | **+121.90 ± 1.05** |

**Verdict**: PASS (std ≤ 5mm).

## Interpretation — ESP32 Uses Shoulder Joint as Z Origin

URDF world → shoulder joint Z chain:
- `world_to_base_link`: +70.1 mm
- `link1 → link2` (shoulder): +51.959 mm
- **Sum: 122.059 mm**

Empirical offset: **122.37 mm** (frame0, HOME-close). Δ = 0.31 mm.

**ESP32's `pose[3]` Z origin = shoulder joint Z height** (standard robotics convention).
Horizontal (X,Y) agreement within ±3mm confirms geometric chain is correct.

## Frame-to-frame dZ drift (±3 mm)

frame0=122.4 / frame_mid=124.9 / frame_last=121.9. Drift ≤3mm across pose configurations. Causes:
- Servo compliance under load (~1-2° → ~1-2mm at 100-200mm lever arm)
- URDF nominal vs actual link length (sub-mm manufacturing tolerance)
- Not a bug — within expected hardware noise

## Isaac Sim Implications

1. **No URDF change needed** — `world_to_base_link` Z=+70.1mm is built into URDF and used in Kinect calib (kinect_calib.yaml extrinsics are in URDF world frame).
2. **`sponge_poses.json` frame = URDF world frame** → directly usable in Isaac Sim if sim world ≡ URDF world.
3. **`deploy_smolvla.py` DIST_MAX (420mm) uses `fk_roarm_m3` URDF world Z**. Not comparable to ESP32 pose Z logs without +122mm correction.

## Open Question — Table Surface Z in URDF World Frame

Not resolved by Step 3. Options for Step 5:
- (A) Live Kinect + fit ground plane to non-sponge depth points (5 min, robust).
- (B) Use sponge_poses Z stats + assumed geometry (mean Z=-9mm, but depth centroids ≠ volumetric centroids for thin objects → unreliable).
- (C) Physical tape measure of robot base plate thickness.

Recommend (A) before Step 5 replay. Without correct table Z, sim physics will let sponge fall through or float.

## Files Used
- `collected_data_v6/episode_XXXX/metadata.json` — ESP32 pose + angles
- `data_z_vs_elbow_analysis.py::fk_roarm_m3()` — URDF FK (RPY bug fix applied 2026-04-24)
- `sim_scripts/kinect_calib.yaml` — calib in URDF world frame (extrinsics source validated)
