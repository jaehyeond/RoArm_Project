---
name: feedback-grasp-z-method
description: Robot FK is the recommended primary method for grasp z measurement in P0; Kinect depth is secondary verification only
metadata:
  type: feedback
---

For cube P0 grasp z measurement, use robot FK as primary, Kinect depth as secondary sanity check.

**Why:** Hand-eye RMSE 10.13mm is comparable to half the cube height (15mm). Propagating this through coordinate transformation compounds error. Robot FK (pose_get()) returns z directly in robot base frame with sub-mm mechanical repeatability, bypassing the camera-robot transform entirely. Table plane fit at RMSE 1.24mm provides an independent ground truth reference.

**How to apply:** P0 substep 1.5 should: (1) position gripper at target grasp height manually, (2) call pose_get() to record TCP z, (3) optionally compare against Kinect depth-derived estimate as a sanity check (not as primary measurement). If the two disagree by >15mm, re-examine camera mounting.

Related: [[calibration-state]]
