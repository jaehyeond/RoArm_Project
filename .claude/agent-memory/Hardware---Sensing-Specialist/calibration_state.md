---
name: calibration-state
description: Current Kinect-RoArm calibration quality: hand-eye RMSE 10.13mm (MARGINAL), table plane RMSE 1.24mm
metadata:
  type: project
---

Hand-eye calibration (sim_scripts/kinect_calib.yaml, 2026-04-24):
- RMSE: 10.13mm (verdict: MARGINAL — acceptable for sim pipeline)
- n_poses: 31 total, 27 used, 4 outliers
- Extrinsic translation: [0.720, -0.001, 0.623] m
- Intrinsics (actual pyk4a): fx=608.33, fy=608.28, cx=638.31, cy=365.26

Table plane (sim_scripts/table_plane.json, 2026-04-24):
- z_world = -12.12mm, RMSE = 1.24mm (high quality, 450K+ inlier pts)
- Tilt: 2.5° from vertical (negligible)
- Source: v6 depth archives, 25 episodes pooled

**Why:** RMSE 10.13mm means camera-based 3D localization has ~10mm 1-sigma uncertainty in robot frame. This is significant relative to a 30mm cube.

**How to apply:** For cube top detection via Kinect depth, expected z measurement error ~10-15mm (1-2 sigma). A 30mm cube top sits at ~+18mm world (table=-12mm, cube_bottom~=table_top, cube_top=+18mm); 10mm uncertainty means cube top z range overlaps with table noise. Robot FK grasp z is more reliable for P0 anchor.

Related: [[project-roarm-cube-task]]
