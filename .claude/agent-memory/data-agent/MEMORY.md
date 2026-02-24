# Data Agent Memory - RoArm M3 SmolVLA

## Dataset State - Sponge Collection (2026-02-24, LATEST)
- Location: `collected_data/` (51 episodes: episode_0000 to episode_0050)
- Task: "Pick up the sponge" (black sponge on white table)
- Total: 51 episodes, 7841 frames, mean 153.7 frames/ep (5.1s), FPS=30
- ALL 51 episodes: DEEP grasp (min_z < 80mm, actually all negative Z)
- Training readiness: 6/10 (PARTIALLY READY)

## Position Diversity (51 episodes, 2026-02-24)
- Base angle: mean=2.2 deg, std=15.7 deg, range=[-30.6, 38.2]
- Zone distribution: LEFT_FAR=1, LEFT=8, CENTER=28, RIGHT=12, RIGHT_FAR=2
- OLD (ep 0-30): base_std=7.3 deg (CENTER-heavy, range [-10.8, 16.0])
- NEW (ep 31-50): base_std=23.3 deg (range [-30.6, 38.2]) -- MAJOR IMPROVEMENT
- Remaining gap: LEFT_FAR only 1 episode, RIGHT_FAR only 2

## Grasp Depth (ESP32 FK Z, all 51 episodes)
- ALL 51 episodes = DEEP (min_z < 80mm, all negative)
- Min Z: mean=-87.6mm, std=14.5mm, range=[-106.5, -43.1]
- Z at grip close (19 eps with data): mean=-90.4mm, std=6.6mm
- Home Z: mean=220.2mm

## CRITICAL: Gripper Pattern Discovery (2026-02-24)
- 96.1% of episodes (49/51): OPENS_STAYS_OPEN (gripper opens to 60-108 deg, then settles to ~24 deg)
- Only 3.9% (2/51): OPEN_THEN_CLOSE (gripper fully closes below 15 deg after peak)
- Gripper start: mean=1.7 deg (closed), End: mean=23.8 deg (still partially open!)
- Min gripper after peak: mean=22.6 deg, std=4.1 deg -- concentrates at 20-25 deg
- This is NOT a data quality bug -- sponge is soft, ~24 deg = sponge gripped
- 25.1% of frames have gripper >30 deg (open), 31.6% <15 deg (closed), 43.3% mid (15-30)
- Grip opens at 33.2% into episode (mean), peak at ~45% into episode
- Episode-level gripper max: mean=66.8 deg (range 41.1-108.1)

## Joint Statistics (frame-level, 51 episodes)
- Base:         range 124.5 deg, mean=1.6, std=19.0
- Shoulder:     range 73.7 deg, mean=31.6, std=18.9
- Elbow:        range 100.5 deg, mean=59.8, std=24.9
- Wrist_pitch:  range 122.1 deg, mean=39.5, std=29.0
- Wrist_roll:   range 108.4 deg, mean=-2.7, std=16.3
- Gripper:      range 106.9 deg, mean=25.8, std=22.4
- Action mean: [1.58, 31.61, 59.82, 39.52, -2.68, 25.82]
- Action std:  [18.97, 18.93, 24.89, 29.02, 16.26, 22.42]

## Shoulder Distribution
- Max shoulder per episode: mean=55.3, range=[43.1, 68.6]
- DEEP(>60deg)=17ep(33%), APPROACH(40-60)=34ep(67%), SHALLOW(<40)=0
- Shoulder never exceeds 68.6 deg (no extreme deep shoulder)

## Temporal Quality
- Static frames: 33.5% of transitions (max joint change <0.5 deg)
- Episode mean: 32.9%, max: 59.6% (ep_0002 only outlier >50%)
- Duration: mean=5.1s, range=[3.4s, 9.9s], no too-short or too-long episodes

## OLD (ep 0-30) vs NEW (ep 31-50) Comparison
- OLD: 31 eps, mean 169.5 frames, base_std=7.3, grip_max=63.3
- NEW: 20 eps, mean 129.3 frames, base_std=23.3, grip_max=72.1
- NEW episodes are shorter, more diverse in position, wider gripper opening
- NEW min_z: mean=-92.2mm (deeper than OLD -84.7mm)

## Key Scripts
- `data_comprehensive_50ep_analysis.py` - Full 51-episode analysis (2026-02-24)
- `data_grip_close_investigation.py` - Gripper trajectory pattern analysis
- `data_gripper_trajectory_detail.py` - Detailed per-episode gripper+Z trajectory
- `data_z_vs_elbow_analysis.py` - FK Z vs elbow analysis (2026-02-23)
- `data_training_quality_analysis.py` - Full quantitative analysis (2026-02-23)
- `data_episode_quality.py` - Existing episode quality script
- `data_distribution_simple.py` - Existing distribution visualizer

## Joint Conventions
- [0]=base, [1]=shoulder, [2]=elbow, [3]=wrist_pitch, [4]=wrist_roll, [5]=gripper
- Shoulder is dominant depth factor (r=-0.814 with Z)
- Elbow controls reach distance, not depth
- ESP32 FK Z: positive=high, negative=deep, 0=base plate, 30mm=table, 220mm=home

## Patterns Learned
- metadata.json: {episode_id, num_frames, timestamp, fps, min_z, max_z, gripper_min/max/range, grip_open_frame, grip_close_frame, frames:[{timestamp, angles[6], pose[3], frame_idx}]}
- grip_close_frame = None for 32/51 episodes (gripper settles at ~24 deg, threshold >15 never crossed)
- Sponge grasp = partial close (~24 deg), NOT full close -- this is correct behavior for soft object
- conda env: `roarm`, must use `conda run -n roarm python3`
- Working directory matters: use absolute paths always

## RoArm M3 URDF Geometry (for FK)
- File: `/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/robots/roarm_m3/urdf/roarm_m3.urdf`
- world_to_base: +70.1mm Z, J1(shoulder): +51.959mm Z, rpy=[-90,-90,0]
- J2(elbow): +236.815mm X, J3(wrist): -144.586mm Y, TCP: +115.428mm Z
