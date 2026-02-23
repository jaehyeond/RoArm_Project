# Deploy Agent Memory

## Key File Paths
- Main deploy script: `/home/cgxr/Documents/Robotics/RoArm_Project/deploy_smolvla.py`
- Keyboard teleop (existing, torque-ON): `/home/cgxr/Documents/Robotics/RoArm_Project/collect_data.py`
- Hand-guide teleop (torque-OFF): `/home/cgxr/Documents/Robotics/RoArm_Project/collect_data_manual.py`
- LeRobot keyboard teleop: `/home/cgxr/Documents/Robotics/RoArm_Project/lerobot/src/lerobot/teleoperators/keyboard/teleop_keyboard.py`
- LeRobot keyboard config: `/home/cgxr/Documents/Robotics/RoArm_Project/lerobot/src/lerobot/teleoperators/keyboard/configuration_keyboard.py`
- RoArm backup (has KeyboardTeleopStrategy): `/home/cgxr/Documents/Robotics/RoArm_Project/lerobot_backup/roarm_m3.py`
- LF configs: `lf_teleop_config.yaml`, `lf_teleop_nocam_config.yaml`, `lf_teleop_camera_config.yaml`

## Project Deploy State
- Current checkpoint default: `outputs/smolvla_v2_cleaned/checkpoints/last/pretrained_model`
- DATASET_MEAN_POS (v2): [3, 41, 13, 61, -2, 10]
- deploy_smolvla.py already has: action scaling, convergence detection, CSV logging, multi-checkpoint, open/closed-loop modes
- JOINT_LIMITS in deploy_smolvla.py uses list-of-tuples format (NOT dict format from CLAUDE.md — both are correct)

## Teleop Method Summary (for data collection)
- Hand-guide (collect_data_manual.py): torque-OFF, physically move arm, 30fps auto-record
- Keyboard (collect_data.py): torque-ON, QWERTY joint control, manual frame save (Space)
- Leader-follower: 2nd RoArm on /dev/ttyUSB1 as leader, mirrors to follower on /dev/ttyUSB0

## Keyboard Teleop Key Map (collect_data.py / lerobot_backup)
- Q/A: Base (+/-), W/S: Shoulder, E/D: Elbow, R/F: Wrist pitch, T/G: Wrist roll, Y/H: Gripper
- -/=: Speed down/up (2°/5°/10° per step), step_size default=5°
- Space: save frame, Enter: save episode, Backspace: cancel, ESC: quit

## LeRobot Keyboard Teleop Architecture
- `KeyboardTeleop`: raw key-set output (no joint mapping, requires robot integration)
- `KeyboardEndEffectorTeleop`: Cartesian EE delta (arrow keys + shift + ctrl_r/l) for SO100FollowerEndEffector
- LeRobot keyboard teleop is NOT joint-space for arm robots — it outputs delta_x/y/z for EE control
- For RoArm M3 joint-space keyboard: use collect_data.py pattern (already written, pynput-based)

## Verified Patterns
- collect_data.py loop runs at ~50Hz, pressed_keys set checked every iteration
- Key hold-down works: keys stay in pressed_keys until release event fires
- step_size=5° per iteration cycle at 50Hz = very fast; better to use smaller step or increase delay
- Actual robot command rate limited by motor speed=500, acc=200

## Important Warnings
- collect_data.py has a bug: frame save is manual (Space press), NOT automatic time-based like collect_data_manual.py
  → For 30fps data collection, collect_data_manual.py's time-based approach is better
- collect_data.py missing: Z-height tracking, gripper validation, episode quality check, dataset progress
- pynput requires DISPLAY set on Linux (both scripts check this implicitly)
