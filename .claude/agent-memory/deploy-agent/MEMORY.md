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
- [project_v5_deploy_state.md](project_v5_deploy_state.md) — V5 audit, start-pos deep analysis, HOME_POS_V5, echo rate, deploy commands
- **V5 CORRECT START-POS**: HOME_POS_V5=[0,44,36,81,0,2] (from actual episode start data). Neither INIT_POS nor dataset_mean work for v5.
- **Why dataset_mean fails**: gripper=28° (mid-open), wrist_pitch=67° (mid-trajectory) → echo → no motion
- **Why INIT_POS fails for v5**: L2=4.17 OOD, model moves init→HOME only, then echo at HOME
- **Why INIT_POS worked for v3**: v3_mean elbow=59° ≈ actual grasp elbow → INIT created approach gradient
- **Echo rate**: 93% of v5 episodes have a[0]≈state[0] within 1° — severe proprioceptive echo
- **V5 commands**: --n-chunks 2, --start-pos home (once HOME_POS_V5 added to script), always --checkpoint explicit
- DATASET_MEAN_POS (line 91): currently [10,44,41,67,0,28] — correct for v5, but WRONG for start-pos (mid-trajectory)
- v3 normalization: action.mean=[-0.47,30.18,58.88,40.72,-2.33,26.48], action.std=[25.81,18.81,24.83,30.07,20.22,24.15]
- v5 normalization: action.mean=[9.95,43.94,41.31,66.57,0.21,28.25], action.std=[30.74,16.13,32.53,29.13,26.36,20.24]
- v5 state: state.mean=[9.93,44.10,40.94,67.18,0.20,28.08], state.std=[30.96,16.05,32.33,28.55,26.60,20.39]
- JOINT_LIMITS: list-of-tuples format, base=(-180,180) NOT (-190,190) — verify against CLAUDE.md if needed
- n_action_steps default=5; EMA default=1.0 (off)

## EMA Smoothing Pattern (added 2026-02-25)
- `ema_smoothed = alpha * new_action + (1-alpha) * ema_smoothed` applied AFTER unnormalize, BEFORE clamp
- ema_smoothed initialized to None → first step bootstraps from raw action (no cold-start bias)
- alpha=1.0: off (raw), alpha=0.4: moderate (recommended for n_action_steps=1), alpha=0.2: heavy
- Only in closed-loop path; open-loop chunk replay intentionally skips EMA
- Root cause of jitter: flow matching starts from DIFFERENT random noise each call → nearby outputs differ
- With n_action_steps >= 5: within-chunk motion already smooth (same noise realization), EMA helps at boundaries

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

## Deployment Failure Analysis (2026-02-25, v3 25K checkpoint)
### CSV Log: logs/deploy_20260225_154420.csv (300 steps, 10Hz)
### Root Cause 1: n_action_steps=50 (open-loop chunking, NOT closed-loop)
- Confirmed by inference pattern: only 6 real inferences (steps 1,51,101,151,201,251 ~109ms each)
- All other 294 steps used cached action from 50-step chunk (~11ms)
- Despite `policy.config.n_action_steps = args.n_action_steps`, default=50 means open-loop re-inference every 50 steps
- For true closed-loop: must pass `--n-action-steps 1`

### Root Cause 2: Mean Regression (25K underfitted checkpoint)
- All z-scores stayed within ±1.0 throughout entire 300 steps
- z-scores at step 72 (convergence): base=+0.11, shoulder=-0.01, elbow=+0.44, wp=-0.86, wr=+0.04, gripper=-0.06
- Model predicted actions near dataset mean → robot barely moved
- 25K steps = 500 gradient updates (batch_size=64) — likely too few for v3 dataset

### Root Cause 3: Wrong Elbow Direction
- Elbow went UP: 60.1° → 78.5° (z: +0.05 → +0.79)
- Expected: elbow should DECREASE for grasping (approach sponge)
- Gripper NEVER opened: stayed at 26.5°→23.4° (z near 0 throughout)
- Wrist_pitch dropped 30° but in isolation (no coordinated arm descent)

### Per-Joint Freeze Steps (5-window all deltas < 0.5°)
- base: step 13,  shoulder: step 14,  elbow: step 21
- wrist_roll: step 6,  gripper: step 6,  wrist_pitch: step 43

### Recommended Fixes
1. Use `--n-action-steps 1` for TRUE closed-loop (re-infer every step)
2. Test 50K checkpoint (not 25K) before changing other settings
3. If still mean regression at 50K: need more/better training data
4. Action scale >1.0 (e.g., 1.5-2.0) can amplify weak z-scores but won't fix wrong direction

## Deployment FK Analysis (2026-02-25, v3 50K checkpoint, n_action_steps=1)
### Source: logs/deploy_20260225_163302.csv (300 steps, 10Hz, --n-action-steps 1)
### FK approximation: L1=95, L2=145, L3=145, L4=60mm (no wrist_roll projection)

### Trajectory Phases
1. Rotation (steps 1-48): base 1.5→51° (fast), shoulder barely moves
2. Approach (steps 49-100): shoulder 22→37°, elbow 80→62°, gripper opens 3→50°
3. Max open (step 116): gripper=56.6°, FK x=-2.2mm y=-2.7mm z=347.5mm
4. Convergence (steps 120-300): all joints plateau, gripper slowly closes 56→38°

### Critical Finding: Model NEVER Descends to Table Level
- FK z at max gripper open: 347mm (arm is 347mm ABOVE table surface)
- For table grasp (z~100mm), elbow must go to -40° to -60°
- All 5 runs show z=299-365mm at max open — consistently too high
- Elbow range across ALL runs: 55° to 91° (NEVER negative, never near table)
- dataset mean elbow=58.88° → model regresses to mean without descending

### Convergence Position (all runs end here)
- Joints: base~35-51°, shld~40-52°, elbow~58-87°, wr~45°
- FK: x~-13mm, y~-16mm, z~348mm
- This is the dataset action mean in joint space (elbow=58.88°)

### Required Elbow for Grasp
- Shoulder=30°, elbow=-40° → z=132mm, x=328mm (approx grasp zone)
- Shoulder=30°, elbow=-50° → z=97mm,  x=318mm (deep grasp zone)
- Dataset mean elbow=+58.88° → z=359mm (hover, not grasp)
- Model needs to learn elbow < -40° for actual table contact

### Why: Training Data Deficit
- v3 DEEP episodes (elbow <-40°): estimated 25-30% only
- Model interpolates toward mean (elbow~58°) → never reaches negative elbow range
- Gripper opens in mid-air (z~340-360mm) → no contact with sponge

## New Log Analysis (2026-02-25 afternoon sessions)
### THREE LOGS: 173740 (open-loop 50 steps), 174334 (closed-loop 200 steps), 163302 (closed-loop 300 steps)

### Gripper at ~24° IS a valid gripping state (sponge width prevents full close)
- This corrects previous error: 64/74 episodes "settling at 24°" = successful grasps, NOT failures
- 0° = fully closed (no object), 24° = sponge gripped, 56° = fully open

### Log 173740 (open-loop, 50 steps)
- ONLY 50 steps → gripper reached 55.9° at step 50 (still rising! hit episode end)
- Hypothesis A CONFIRMED: 50 steps is NOT enough for a complete grasp cycle
- At step 50: shld=33.7, elbow=53.4, z_FK=~317mm (still mid-air)
- Model was clearly still in "opening" phase when stopped

### Log 174334 (closed-loop, 200 steps)
- Gripper only reached 34.3° MAX (step 200) — never fully opened
- Much slower gripper opening than 163302 run
- Convergence detected at step 85 (shoulder ~12°, elbow ~92°, gripper ~1.5°)
- Then convergence RESET and arm resumed moving slowly
- Gripper z-score at end: +0.323 (weakly open signal) vs +1.248 in 163302
- Elbow stayed 61-93° the whole time, z_FK=340-370mm (all mid-air)

### Log 163302 (closed-loop, 300 steps) — most informative
- Gripper peak: 56.6° at step 116 (shld=46.3, elbow=61.6, z_FK=347mm)
- Gripper slowly DECREASING after step 116: 56→53→50→46→43→38° by step 300
- z_gripper at peak: +1.248 → at step 300: +0.516 (model slowly predicting "close")
- BUT elbow NEVER goes below 58° — z_elbow stays flat at +0.01 to +0.17 throughout
- The model correctly sequences: base rotate → approach → open gripper → ???
- Missing step: descend (elbow must go negative) then close gripper

### Root Cause Confirmed: Missing Descent Phase in Training Data
- In training data: gripper OPENS (going from ~24° to ~56°) requires descending arm
- The model learned "open gripper" but NOT "while descending"
- In deployment: arm opens gripper at z=347mm (mid-air) because elbow never goes negative
- The close phase (z_gripper going negative) does start slowly after step 116
  but by step 300, gripper only fell to 38° — still open

### Why gripper doesn't complete the close (even at 300 steps):
- The close signal is weak (z_gripper: +1.25 → +0.52 over 184 steps = 0.004/step)
- The arm is not descending simultaneously (elbow flat at +0.01-0.17 z-score)
- Without physical contact with sponge, closed-loop has no corrective signal
- Model enters attractor state: shld≈51°, elbow≈59°, grip≈38-45° = local mean

### Would more steps help? Probably not significantly:
- Rate of gripper decrease: ~18° over 184 steps = ~0.1°/step
- To reach 24° (grip): need ~(38-24)/0.1 = 140 more steps = ~500 total
- But elbow would need to decrease simultaneously — it's not doing that
- The missing physics: sponge contact → proprioceptive signal → trigger close
