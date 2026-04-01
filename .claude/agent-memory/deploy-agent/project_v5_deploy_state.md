---
name: V5 Deployment State
description: deploy_smolvla.py audit for v5 model, known bugs, deployment commands, normalization stats
type: project
---

## V5 Model Deployment State (updated 2026-03-31)

Training complete: smolvla_v5_multipos, 200K steps, 136 episodes, 5 zones.
All 20 checkpoints at: `outputs/smolvla_v5_multipos/checkpoints/{010000..200000,last}/`

### V5 Normalization Stats (verified from 200K checkpoint safetensors)
- action.mean: [9.95, 43.94, 41.31, 66.57, 0.21, 28.25]
- action.std:  [30.74, 16.13, 32.53, 29.13, 26.36, 20.24]
- state.mean:  [9.93, 44.10, 40.94, 67.18, 0.20, 28.08]
- Note: 50K and 200K have identical normalization (dataset-derived)

### deploy_smolvla.py v5 Audit Result

All features present and working:
- --start-pos dataset_mean: supported
- --checkpoint: supported
- JOINT_LIMITS: correct
- open-loop + n-chunks: present
- CSV logging, EMA, convergence: present
- Workspace safety (Z_FLOOR=-130, DIST_MAX=420): present

**BUG (line 91): DATASET_MEAN_POS = v3 values [0, 30, 59, 41, -2, 26]**
- V5 correct value: [10, 44, 41, 67, 0, 28]
- Wrist_pitch difference is large: 41 → 67 (26° off)
- MUST fix before using --start-pos dataset_mean with v5 model
- Workaround: --start-pos current + manual arm positioning

**Suboptimal: CHECKPOINT_PATH default still points to v3 25K checkpoint**
- Always pass --checkpoint explicitly for v5

**Suboptimal: --n-chunks default=4 produces 200 steps for v5 episodes ~99 frames**
- Recommended: --n-chunks 2 (100 steps, matches v5 episode length)

### Recommended Deployment Commands

Phase 0 (dry-run, no robot):
```
python deploy_smolvla.py --checkpoint outputs/smolvla_v5_multipos/checkpoints/050000/pretrained_model --dry-run --start-pos current --max-steps 5 --log-csv auto
```

Phase 1 (first real run, 50K):
```
python deploy_smolvla.py --checkpoint outputs/smolvla_v5_multipos/checkpoints/050000/pretrained_model --start-pos current --open-loop --n-chunks 2 --hz 10 --log-csv auto --convergence-action warn
```

Phase 2 (200K best checkpoint):
```
python deploy_smolvla.py --checkpoint outputs/smolvla_v5_multipos/checkpoints/200000/pretrained_model --start-pos current --open-loop --n-chunks 2 --hz 10 --log-csv auto --convergence-action warn
```

### Success Criteria
- Gripper 15-20° with sponge = success (NOT 0°; sponge compliance)
- FK z < 150mm = arm reached table level
- Stage 1 target: 60%+ across 5 zones

### Plan file: /home/cgxr/Documents/Robotics/RoArm_Project/deploy_v5_plan.md

**Why:** v5 has completely different normalization from v3 (dataset_mean changed substantially,
especially shoulder 30→44, wrist_pitch 41→67). Using v3 DATASET_MEAN_POS with v5 model
would start the robot OOD and likely produce poor trajectories.

**How to apply:** When v5 deployment is discussed, remind user to patch DATASET_MEAN_POS
before using --start-pos dataset_mean. Always use --n-chunks 2 for v5.

---

## V5 Start Position Deep Analysis (2026-03-31)

### V5 Episode Start Statistics (136 episodes, from parquet)
- base: mean=8.7, std=37.3 (varies by zone: 5-zone collection)
- shoulder: mean=44.1, std=13.5 (consistent)
- elbow: mean=36.0, std=29.3 (varies by object position)
- wrist_pitch: mean=80.9, std=16.1 (consistent, arm upright at start)
- wrist_roll: mean=-0.1, std=35.5
- gripper: mean=1.9, std=1.2 (always CLOSED at start)

### Actual HOME position (derived from data):
`HOME_POS_V5 = [0, 44, 36, 81, 0, 2]`  (base=0 for center zone, others from data median)

### Why --start-pos dataset_mean FAILS for v5

DATASET_MEAN_POS=[10,44,41,67,0,28] is the MID-TRAJECTORY average (gripper=28, mid-open).
Episodes START with gripper=2 (fully closed) and wrist_pitch=81 (arm upright).
Dataset mean has wrist_pitch=67 (arm already partially lowered).
→ dataset_mean puts robot in a mid-grasp state → model sees current≈action_mean → echo → no motion.

### Why INIT_POS [0,0,90,0,0,0] ALSO FAILS for v5

INIT_POS in v5 state z-space:
- shoulder z=-2.748 (INIT=0, v5_mean=44.1, delta=-44°)
- wrist_pitch z=-2.353 (INIT=0, v5_mean=67.2, delta=-67°)
- elbow z=+1.517 (INIT=90, v5_mean=40.9, delta=+49°)
- L2 norm = 4.17 (highly OOD)

Predicted model behavior from INIT_POS with v5 model:
- Model must bring arm toward distribution center: shoulder +44, elbow -49, wrist_pitch +67
- This trajectory is INIT→HOME (arm-up to work position), NOT HOME→GRASP
- After reaching HOME (=dataset_mean), echo kicks in → no further motion
- v5 INIT_POS deployment: robot moves to HOME position then stops. No grasping.

### Why INIT_POS worked for v3 but not v5

V3 state distribution: shoulder_mean=30, elbow_mean=59, wrist_pitch_mean=41
V3 INIT_POS z-scores: shoulder z=-1.59, elbow z=+1.25, wrist_pitch z=-1.35, L2=2.67

In v3, elbow=59 is very close to the grasp position (elbow~58 throughout entire v3 trajectory).
From INIT_POS, v3 model moves: shoulder +30, elbow -31 → coincidentally matches first half of grasp approach.
The v3 dataset mean elbow (58.88°) IS roughly the grasp elbow angle.

In v5, elbow_mean=41 represents mid-trajectory (still descending) and wrist_pitch=67 is mid-way.
From INIT_POS, v5 model moves: shoulder +44, elbow -49 → reaches HOME (elbow~36), not grasp position.

### Correct start position for v5 deployment

**Use HOME_POS_V5 = [0, 44, 36, 81, 0, 2]** (actual episode start position from data)

This is in-distribution for the beginning of trajectories. The model will see:
- state z ≈ [-0.04, 0.00, -0.15, +0.48, -0.01, -1.28] (L2=1.375, close to distribution start)
- Model should predict: move arm toward object (lower elbow, rotate base, eventually open gripper)

Add to deploy_smolvla.py:
```python
HOME_POS_V5 = [0, 44, 36, 81, 0, 2]  # Actual episode start (v5 data-derived)
```
And add "home" as a --start-pos option.

### Proprioceptive Echo Rate in v5 Data
- 127/136 episodes (93%): a[0] == state[0] within 1° (max diff < 1°)
- 133/136 episodes (98%): a[0] ≈ state[0] within 5°
- This confirms severe echo in v5 training data. Model has learned "copy state → action" for step 0.
- Impact: whatever start position is used, the first ~few steps will essentially copy the state.
- After the initial echo, the model must diverge from state toward learned trajectory.
- HOME_POS_V5 start → first action ≈ HOME_POS_V5 (echo), then model diverges toward grasp.
