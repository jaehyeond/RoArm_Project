# Data Agent Memory - RoArm M3 SmolVLA

## Dataset State (v4, as of 2026-02-23)
- Location: `lerobot_dataset_v4/` + `collected_data/`
- Episodes: 43 total (indices 3-49, with gaps at 0-2, 14, 21, 22, 45)
- Total frames: 9,747  |  Mean: 226.7 frames/ep  |  FPS: 30
- Parquet: `lerobot_dataset_v4/data/chunk-000/file-000.parquet` (single file, 9747 rows x 7 cols)

## Joint Conventions
- [0]=base, [1]=shoulder, [2]=elbow, [3]=wrist_pitch, [4]=wrist_roll, [5]=gripper
- Elbow DEEP = min < -30° -- WRONG METRIC (see Elbow vs Z Analysis below)
- Gripper open = > 20°, closed = < 5° (hardware max = 100°, actual max seen = 55.6°)

## CRITICAL: Elbow Angle is NOT a Valid Grasp Depth Proxy (2026-02-23)
- Analysis script: `/home/cgxr/Documents/Robotics/RoArm_Project/data_z_vs_elbow_analysis.py`
- Frame-level correlation: Shoulder-Z r=-0.814 (STRONG), Elbow-Z r=+0.287 (WEAK, 2.8x weaker)
- 34/43 episodes misclassified if using "elbow < -30 = DEEP" criterion
- SHOULDER is the dominant depth factor: shoulder > 60-65° = arm going DOWN (deep grasp)
- Elbow controls reach DISTANCE, not depth (moves arm forward/back, not up/down)
- URDF FK geometry confirms: shoulder joint rotation axis = horizontal (-X in world), so higher shoulder angle = TCP goes DOWN

## Grasp Depth Classification (CORRECTED, from URDF FK)
Using Relative Z (from home position, more negative = deeper):
- DEEP: Z_rel < -150mm (arm >150mm below home) -- 22/43 (51%)
- APPROACH: Z_rel -50 to -150mm -- 17/43 (40%)
- SHALLOW: Z_rel > -50mm -- 4/43 (9%)
Home position URDF FK Z = -106.2mm (reference baseline)
Best episodes: ep_0030 (Z_rel=-228mm, sh=69°), ep_0035 (-216mm, sh=74°), ep_0039 (-209mm, sh=81°)

## ESP32 pose_get() Z Calibration (CONFIRMED, 2026-02-23)
- pose_get() returns value[0:4]+[value[8],value[9]] = [x,y,z,tilt,wrist_roll,gripper]
- value[0:3] = x,y,z are computed by ESP32 onboard FK (positive Z = up from base plate)
- USER CONFIRMED: Z=30mm when arm is fully extended straight down to table surface
- Z convention: 0=base plate level, 30=table touch, 80=typical object grasp, 160=approach, 230+=home
- CALIBRATED thresholds in collect_data_manual.py (updated 2026-02-23):
  - DEEP: Z < 80mm  (gripper at or below object top surface)
  - APPROACH: Z 80-160mm (arm descending toward object)
  - SHALLOW: Z > 160mm (arm at home height or above)
  - Grip-close Z validation: fail if Z_at_grip_close > 130mm
  - Episode SHALLOW warning: if min_z > 160mm (never descended to approach zone)

## Elbow Distribution (CORRECTED CONTEXT)
- Old elbow DEEP (<-30°): 8 eps (19%) -- WRONG metric, ignores shoulder contribution
- Revised by URDF FK: 22/43 (51%) DEEP (arm truly low), 17/43 APPROACH, 4/43 SHALLOW
- Key insight: episodes 3-20 (early dataset) have POSITIVE elbow but still reach DOWN via high shoulder

## Critical Finding: Gripper-Elbow Timing Mismatch
- 58% of episodes: gripper opens BEFORE reaching deepest elbow (EARLY_OPEN)
- 40% of episodes: gripper opens AFTER deepest elbow (LATE_OPEN)
- Only 2% (1 episode): gripper open exactly when elbow is deepest (CORRECT_SYNCED)
- AT GRASP CLOSE: only 4/43 (9%) have elbow in DEEP range (<-30°)
- The model is learning to open gripper mid-trajectory, not at the grasp point

## Elbow Distribution
- DEEP (<-30°): 8 eps (19%), only 8.9% of frames
- APPROACH (-30 to -10°): 7 eps (16%), 14.6% of frames
- SHALLOW (>-10°): 28 eps (65%), 76.6% of frames

## Gripper Data Quality
- Max gripper angle: mean=40.6°, std=6.1° (very narrow range!)
- Frames > 50° (wide open): only 0.68% of total frames
- Frames > 20° (open): 21.4%  |  Frames < 5° (closed): 70.2%
- Gripper always opens in middle third (T2) of episode: 43/43 (100%)
- Training data teaches: gripper stays closed → opens briefly in middle → closes

## Object Position Variation
- Base angle std across episodes: 17.91° (good variation, box at different positions)
- Grasp-phase base std: 15.56° (genuine spatial variation)
- Frames with |base| < 5°: 19% (not always centered)

## Static Frame Fraction
- 29.1% of frame transitions show no joint moving more than 0.5°
- This inflates "closed gripper" and "stationary pose" in training

## Key Scripts
- `data_z_vs_elbow_analysis.py` - FK Z vs elbow analysis, confirms shoulder is depth proxy (2026-02-23)
- `data_training_quality_analysis.py` - Full quantitative analysis (created 2026-02-23)
- `data_episode_quality.py` - Existing episode quality script
- `data_distribution_simple.py` - Existing distribution visualizer

## RoArm M3 URDF Geometry (for FK)
- File: `/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf`
- world_to_base: fixed +70.1mm Z
- J0 (base): Z-axis rotation, no offset
- J1 (shoulder): +51.959mm Z offset, rpy=[-90,-90,0], rotation axis = world -X
- J2 (elbow): +236.815mm X, +30.002mm Y offset, rpy=[0,0,+90]
- J3 (wrist_pitch): -144.586mm Y offset
- J4 (wrist_roll): +15.147mm X, -53.653mm Y, rpy=[+90,+90,0]
- TCP (hand_tcp): +115.428mm Z, rpy=[+90,-90,0]

## Hand Occlusion Analysis (2026-02-23)
- Method: bimodal depth signature (>5% pixels <300mm AND <2% pixels 300-500mm = hand in frame)
- Overall occlusion rate: 6.5% of deep grasp frames, 1.0% of shallow frames
- Only 2/43 episodes show notable occlusion: episode_0031 (52% frames) and episode_0011 (28% frames)
- Spatial location of hand: UPPER-LEFT quadrant of frame (not center), cols 62-430 of 1280
- ep_0031 is the outlier: hand stays in frame from pre-grasp through entire deep grasp phase
- 6/8 deep episodes: ZERO detectable occlusion during grasp phase
- Occlusion location (when present): upper-left corner, NOT over the grasp site (object is center-frame)
- CONCLUSION: occlusion is a minor issue, concentrated in 1-2 episodes, and does NOT block grasp-site view

## Patterns Learned
- metadata.json structure: {episode_id, num_frames, timestamp, fps, min_elbow, frames:[{timestamp, angles[6], frame_idx, rgb_path, depth_path}]}
- New metadata has min_z, max_z, z_range, gripper_min, gripper_max, gripper_range fields (from upgraded script)
- NO object position metadata stored (no box_xyz in frames)
- conda env: `roarm`, must use `conda run -n roarm python3` or activate first
- Working directory matters: use absolute paths always
- Depth bimodal test: valid<300mm >5% AND valid 300-500mm <2% = hand/arm in frame foreground
