---
name: project-roarm-cube-task
description: RoArm-M3 + Azure Kinect cube task (Track B) P0 hardware/sensing context as of 2026-05-26
metadata:
  type: project
---

Cube task pivot confirmed 2026-05-26: sponge → cube 3×3×3cm × 5개, 3+2 pyramid stacking.
P0 = cube + gripper calibration. L-F teleop: Leader USB0, Follower USB1, Azure Kinect fixed v6 viewpoint.
GPU CUDA mismatch (NVML 580.159 / kernel 580.126.09) blocks torch inference, but robot serial + Kinect are GPU-independent.
Sponge HARD RULES #19/#20 superseded by user explicit correction (HARD RULE #18).

**Why:** Task change requires fresh grasp z measurement because cube geometry (30mm rigid) differs from sponge (47mm edge-stand).

**How to apply:** P0 must measure cube-specific grasp z and gripper jaw angle. Do not reuse sponge anchors from tech_gripper_grasp_anchors.md.

Related: [[calibration-state]], [[feedback-grasp-z-method]]
