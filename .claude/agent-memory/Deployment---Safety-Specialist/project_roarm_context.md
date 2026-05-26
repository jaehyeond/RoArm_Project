---
name: project-roarm-context
description: RoArm-M3 SmolVLA pipeline key safety constants, Track B cube task P0 context, and deployment history
metadata:
  type: project
---

Track B cube task pivot (sponge→3×3×3cm rigid cube). P0 = follower(USB1) scripted calibration before any VLA deployment.

**Why:** Cube is rigid, OOD from v6 sponge training. P0.2 = gripper jaw cmd sweep (0~40° in 5° steps), P0.3 = approach angle sweep (wrist_p 75/60/45°).

**How to apply:** Safety analysis for P0 must account for: rigid cube jam risk (no compliance), ESP32 serial buffer limits, observer-effect on servo speed, and lack of workspace bounds in gripper-only control path.

## Key safety constants (from deploy_smolvla.py:80-93)
- Z_FLOOR_DEPLOY = -130mm (table -120 + 10mm margin)
- DIST_MAX_DEPLOY = 420mm
- JOINT_LIMITS = [(-190,190),(-110,110),(-70,190),(-110,110),(-190,190),(-10,100)]
- JOINT_SPEED_CAPS = [500,500,500,300,300,300] — distal joints capped at 300
- INIT_POS = [0, 0, 90, 0, 0, 5]

## Deployment history
- v1 FAIL: closed-loop n=1 → Wrist_R runaway -3→-92° (4σ OOD drift)
- v1 FAIL: Elbow 13→36° monotonic drift (DEEP data shortage)
- Plan3 SUCCESS (4/9): gripper-only speed unlock via second `gripper_angle_ctrl(speed=1000,acc=0)` call after `joints_angle_ctrl`

## Observer effect (gripper_calibrate_v4.py verified)
- V3 (1 read/cmd, sleep 6s) → ~0.87 deg/s effective, never reached cmd
- V2/V4 (poll every 1s) → ~17 deg/s effective, reaches cmd by 3-6s
- Read frequency directly affects servo motion speed — must poll during settle

## ESP32 serial (scan_servos.py / reset_robot.py)
- T:106 = ESP32 crash → auto-reset → motor bus re-init
- T:604 = reset settings, T:603 = move to init, T:210 = torque ON
- scan_servos.py uses 0.8s inter-command delay

Source: claudedocs/session_20260526_track_b_cube_task_pivot_plan.md, CLAUDE.md hardware section
