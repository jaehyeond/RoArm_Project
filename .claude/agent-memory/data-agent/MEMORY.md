# Data Agent Memory - RoArm M3 SmolVLA

## L-F Pipeline Critical Review (2026-04-01) ← LATEST
- Full report: `data_lf_pipeline_review.md`
- B3 HIGH: `_safe_angle_read` returns `[0,0,0,0,0,0]` on failure → silent corrupt action in dataset + Follower home-snap
- B2 MEDIUM: raw leader_angles saved as action, but clamped values sent to Follower (mismatch for out-of-limit poses)
- B1 LOW: zone classification uses Follower base angle, should use Leader base angle
- Timing correct: Follower read → Leader read → mirror command → save_frame = standard IL (s_t, a_t) convention
- convert: action=leader_angles[t], state=follower_angles[t], same timestep, no offset — CORRECT

## Official Pipeline Analysis (2026-03-31)
- File: `project_official_pipeline_analysis.md`, script: `data_official_pipeline_analysis.py`
- lerobot-record has ZERO validation — operator manually re-records bad episodes
- Reference dataset: 50ep, 393fr/ep (13.1s), 20K steps, batch=64
- Our v5: 99fr/ep (3.3s), 200K steps — wrong on both counts
- C0a (HOME start) and C5 (Z check) = JUSTIFIED. C0b, C2, C3 = OVERCONSTRAINED/REDUNDANT
- Recommended: remove C0b, C2; relax C1→30deg; C3→WARNING; C4 FAIL→90fr

## V5 Deployment Failure Root Cause (2026-03-31)
- File: `project_v5_proprioceptive_echo.md`, script: `data_v5_deployment_failure_analysis.py`
- r(state.base, action.base)=0.9996 (V5) vs 0.9992 (V3) — nearly identical echo strength
- ROOT CAUSE: V5 episodes start at approach pose (shoulder=44.1°, 0% near home) → no home→approach trajectory
- V3 success: 100% episodes start at home (shoulder=2.8°) → large motion from init forces model commitment
- V5 with --start-pos dataset_mean = FIXED-POINT TRAP: arm already at approach, predicts "stay still"
- V3 with --start-pos init = forces shoulder to rise 41°, breaking fixed-point loop
- Gripper q10: V5=16.2° (mid-grip) vs V3=1.7° (firmly closed) — V5 gripper distribution missing low-closed state
- Fix: collect episodes starting from HOME position (shoulder<5°, base=0) so model learns full approach sequence

## V5 Zone System Design Flaw (2026-03-31)
- File: `project_v5_zone_bias_analysis.md`
- 3/5 zones (NEAR, FAR_CENTER, OVERHEAD) all map to |base|<30° — effectively same angular region
- Quota system is advisory only: no blocking, no FAIL, no hard enforcement
- Frame dist: 55.2% in -5° to +15°, mean=+9.93°, 80% within [-3°, +23°] (26° window)
- Fix: replace with 5 base-angle zones (FAR_LEFT/LEFT/CENTER/RIGHT/FAR_RIGHT × 27 eps each)
- User followed system correctly — this was a design failure, not a user error

## V5 Dataset Analysis (2026-03-26)
- File: `project_v5_dataset_analysis.md`
- 136 eps, 13,470 frames, 5 zones, 3.3s/ep, 100% phase completion, 0 flagged
- CRITICAL: episodes start at approach pose (sh~44°), NOT home — deployment constraint
- CRITICAL: gripper <15% only 7.5%, but <20% = 57.8% (correct sponge-grasp threshold)
- dataset_mean = [9.93, 44.10, 40.94, 67.18, 0.20, 28.08]
- Elbow bimodality: dead zone 42-60°, mean=40.9° is IN the dead zone
- OVERHEAD zone: 15 eps (marginal), z>0 by design (elevated grasp)
- Zone: FAR_CENTER=39, NEAR=30, MID_RIGHT=27, MID_LEFT=25, OVERHEAD=15

## collect_data_manual.py FAIL Flow Analysis (2026-03-26)
- File: `project_collect_fail_flow_analysis.md`
- Root cause: pynput background thread + conda no-PTY = fully buffered stdout (4096B)
  → print() FAIL reasons don't appear in terminal until buffer flushes
- Fix: flush=True on all print() in save_episode() + sys.stdout.flush() + OSD reason display
- All 5 FAIL conditions documented (F1-F5), all 6 WARNING conditions (W1-W6)
- Most likely cause for Grip:2°, Sh:45°, F:127, Z:-110mm = F1 (gripper never opened >40°)

## Dataset State - Sponge Collection (2026-02-24, LATEST)
- Location: `collected_data/` (51 episodes: episode_0000 to episode_0050)
- Task: "Pick up the sponge" (black sponge on white table)
- Total: 51 episodes, 7841 frames, mean 153.7 frames/ep (5.1s), FPS=30
- ALL 51 episodes: DEEP grasp (min_z < 80mm, actually all negative Z)
- Training readiness: 7.5/10 (PARTIALLY READY) -- updated 2026-02-24 with full analysis

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

## V3 Deployment Failure Analysis (2026-02-25)
- Deploy log: `logs/deploy_20260225_154420.csv` (300 steps, 25K checkpoint)
- Script: `data_v3_deployment_failure_analysis.py` (full root cause analysis)
- Convergence: first detected at step 72, fully locked in by step ~100
- Converged position: [2.5, 30, 70, 14, -1.7, 25] deg

### V3 Deployment Root Causes
1. FIXED-POINT LOOP: model predicts mean → robot goes to mean → model sees mean → repeat
2. GRIPPER STUCK AT MEAN (26.5 deg): bimodal dataset (closed OR open), mean=transition zone
   - Deployment gripper range: 23.1 to 26.6 deg (3.5 deg = 3.6% of offline 95 deg range!)
   - Offline evaluation showed 94 deg range -- completely misleading for sequential deployment
3. WRIST_PITCH COLLAPSE: converged to 14 deg vs mean 40.7 / median 54.9 deg
   - Offline pred mean=23 deg is BELOW training data q10=-3 → wrist folds in deployment
   - Wrist at 10 deg = gripper points sideways, not down
4. SHOULDER NEVER DESCENDS: locked at 30 deg (dataset mean), needs <10 deg for deep grasp
5. ELBOW DRIFTS UP: converged at 70 deg vs mean 58.9 / median 51.2 (wrong direction!)

### Key Insight: Why Offline Evaluation Lies
- Offline pred_std looks good (~25 deg per joint), but tests CROSS-SAMPLE (random timesteps)
- Deployment is SEQUENTIAL: robot state becomes correlated with predictions
- Mean regression causes state to converge → all subsequent states look similar → predict mean
- Solution: temporal evaluation (simulate sequential inference chain), not random sampling

### V3 Dataset (lerobot_dataset_v3) Stats
- 74 episodes, 13145 frames
- action.mean: [-0.47, 30.18, 58.88, 40.72, -2.33, 26.48]
- action.std:  [25.81, 18.81, 24.83, 30.07, 20.22, 24.15]
- Wrist_pitch q50=54.9 vs mean=40.7 → heavily bimodal
- Gripper q50=24.1 vs q90=68.5 → bimodal (mostly closed + opens briefly)

### Collection Requirements for Next Dataset
- Each episode MUST show complete temporal grasp cycle (7 phases):
  Phase 1 (start): gripper closed, arm at home
  Phase 2 (approach): arm moves toward sponge, gripper OPENS (>40 deg)
  Phase 3 (hover): gripper open, above sponge (shoulder <30 deg)
  Phase 4 (descend): gripper open, shoulder descending to <10 deg
  Phase 5 (grasp): gripper CLOSES (~5-10 deg = gripped)
  Phase 6 (lift): shoulder rises with sponge, gripper closed
  Phase 7 (return): return to start with sponge
- Min 40% of frames: gripper > 40 deg (currently 25.1%)
- Min 30% of frames: gripper < 15 deg (currently 31.6%, OK)
- Shoulder must reach < 10 deg in each episode

## V3 Base Joint Distribution Analysis (2026-02-25)
- Script: `data_base_joint_analysis.py` - lerobot_dataset_v3 base joint distribution deep dive
- Dataset: 74 episodes, 13145 frames, lerobot_dataset_v3/data/chunk-000/file-000.parquet

### Key Finding: MASSIVE Base=0 Spike
- 10.7% of ALL frames sit exactly at base=0 deg (1405/13145 frames)
- 37.2% of frames in z=[0.0, +0.2] (base 0 to +6 deg)
- This spike is NOT one episode -- it's spread across 73/74 episodes (all start from home)
- 100% of episodes START at base near 0 (range -1.05 to +0.62, mean=0.22, std=0.28)
- Home position creates a massive mode at base~0 in every episode's first few frames

### Why Model Outputs Base=51 (NOT mean regression)
- Base=51 deg = z-score of +1.99 (2-sigma outlier) -- only 5.4% of frames above 51
- This is NOT a data cluster -- there is NO attractor at 51 deg
- Probable cause: model learned a right-side sponge trajectory
  - 19/74 episodes (25.7%) have sponge placed RIGHT_FAR (max_base > 30 deg)
  - Episode 51 (max_base=64.2), ep 52 (66.1), ep 56 (66.4) -- these reach 51+ on approach
  - Base sweeps from 0 -> 50+ in these episodes; model replays this approach
  - 44/74 episodes (59.5%) have sponge in CENTER (max_base <10 deg)
  - 11/74 (14.9%) have sponge in RIGHT (max_base 10-30)
- This suggests model sees initial state (base~0) and activates "right-side approach" behavior

### Right/Left Asymmetry
- 60.6% of all frames have base > 0 (vs 37.1% below 0) -- 23.5% asymmetry
- Old episodes (0-39): base_mean=-0.17, centered
- New episodes (40-73): base_mean=-0.78, also centered BUT contain right-heavy episodes
- RIGHT_FAR episodes (40-57): ep_mean 22-41 deg, create the 50-66 deg cluster

### V3 Dataset Sponge Position Coverage
- CENTER (max_base 0-10): 44 eps (59.5%)
- RIGHT (max_base 10-30): 11 eps (14.9%)
- FAR_RIGHT (max_base >30): 19 eps (25.7%)
- FAR_LEFT (min_base < -30): 16 eps -- these have LEFT placements
- LEFT (-30 to -10): 0 episodes (no pure left zone episodes by max_base metric)

## V3 CENTER Episode Analysis (2026-02-25)
- Script: `data_center_episodes_analysis.py`, `data_center_grasp_detail.py`
- Dataset: lerobot_dataset_v3, 74 episodes total

### CENTER Episode Count
- CENTER (max|base| < 10 deg): 18 / 74 episodes (24%)
- Episode IDs: [0,1,2,3,4,17,18,19,21,25,26,27,28,29,30,50,65,73]

### CENTER Grasp Trajectory (consistent across all 18 eps)
- ALL 18 episodes: GRIP_OPEN -> DESCEND order (gripper opens ~35% into ep, descend ~50%)
- Phase 2 (approach at ~35%): Shoulder ~39 deg, Elbow ~48 deg, Gripper opens to ~68 deg
- Phase 3 (deepest at ~50%): Shoulder ~56 deg, Elbow ~46 deg (range 11-81), Gripper still open ~64 deg
- Phase 4 (lift at ~57%): Gripper settles to ~28 deg (sponge gripped), shoulder stays ~56 deg
- Phase 5 (return): Elbow returns to ~90 deg (home), shoulder drops

### Elbow at Grasp Point (CENTER episodes)
- At shoulder_peak (deepest): mean=46.0, std=22.9, range=[11.2, 80.9]
- TWO TRAJECTORY TYPES exist within CENTER:
  1. DEEP arm (eps 17,18,19,21,65): shoulder_max=64-67, elbow_at_deep=11-21 deg (high shoulder + very low elbow)
  2. MODERATE arm (eps 0-4,26-30): shoulder_max=43-53, elbow_at_deep=51-81 deg (less shoulder, less elbow drop)
- Elbow NEVER goes negative in CENTER episodes (min = 11.2 deg)

### Important Clarification
- "Elbow < -30 deg" target from project notes = NOT applicable to v3 dataset
- V3 uses different joint convention OR the collection data never reached negative elbow
- In v3: deep grasp = high shoulder (60+ deg) + low elbow (10-20 deg), NOT negative elbow
- The 18 CENTER episodes are all DEEP by shoulder criterion (shoulder_min all < 5 deg = arm returns to home)

## V3 Temporal Phase Analysis (2026-02-25)
- Script: `data_temporal_phase_analysis.py`
- Dataset: 74 episodes, all start from init (gripper~1.7, shoulder~2.5, elbow~90)

### Phase Timing (mean across all 74 episodes)
- Phase A (init):          frame 0      (0.00s, 0%)
- Phase B (approach):      frame ~5-10  (~0.2s, ~5%)
- Phase C (gripper >40°):  frame 58.6   (1.95s, 33.2%)  std=22.2, range=[29,128]
- Phase D (gripped ~24°):  frame 98.7   (3.29s, 56.1%)  std=33.9, range=[58,186]
- Phase F (end):           frame 177.6  (5.92s, 100%)

### Open Phase Duration (Phase C to Phase D)
- Mean: 40.1 frames (1.34s), range=[18, 69 frames] = [0.60s, 2.30s]

### 50-Step Chunk Coverage Problem
- 50 steps = frames 0-49 = 1.67s = only 28% of mean episode
- 56.8% of episodes: Phase C (gripper open) starts AFTER frame 50
- Phase C never occurs before frame 29 (earliest episode)
- Re-plan cycles for full grasp: ~3.5 cycles (cycle 1=approach, cycle 2=open+grip, cycle 3=lift+return)

### Why Bimodal Gripper Causes Mean Regression
- At frame 50-70 (re-plan boundary), ~43% of episodes have gripper >40 (open) and ~57% have it <10 (closed)
- Model sees same arm position but different gripper states across training batch
- Regression to mean = 26 deg = stuck in transition zone
- Solution: longer episodes OR temporal-aware training (context chunks)

### Key Structural Pattern (all 74 episodes)
- Arm descends DURING gripper opening (simultaneous, not sequential)
- Shoulder reaches max ~5-10 frames AFTER gripper peak
- Return to init: elbow rises back to 90 (home) during Phase E/F
- 100% of episodes end at shoulder ~10-25 deg (not fully at init 2.5 deg)

## Key Scripts
- `data_full_analysis_v4.py` - DEFINITIVE full analysis script (all 11 sections, v1 comparison, quality flags)
- `data_v3_deployment_failure_analysis.py` - V3 deployment failure root cause analysis (2026-02-25)
- `data_comprehensive_50ep_analysis.py` - Full 51-episode analysis (2026-02-24, superseded by v4)
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
