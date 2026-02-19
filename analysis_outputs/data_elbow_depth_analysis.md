# Elbow Depth Analysis: Is min_elbow < -30° a Valid Grasp Depth Metric?

## Investigation

**Question**: Does elbow angle < -30° actually indicate deep grasps, or could it be horizontal arm extension?

**Hypothesis**: For a 6-DOF arm, end-effector height depends on:
- Shoulder angle (lifts/lowers the upper arm)
- Elbow angle (extends/folds the forearm)
- Wrist pitch (tilts the gripper)

A negative elbow could mean:
- (a) Arm reaching DOWN to table (shoulder forward, elbow extending down) — GOOD for grasping
- (b) Arm extending HORIZONTALLY (shoulder up, elbow straightened) — NOT useful for grasping

## DEEP Episodes Analysis

| Episode | min_elbow | Shoulder @ min | Wrist_pitch @ min | Gripper @ min | Start→End Elbow | Start→End Shoulder |
|---------|-----------|----------------|-------------------|---------------|-----------------|--------------------|
|  2 |  -65.4° |   59.5° |  108.4° |   5.7 | -29.8→-65.4 (-35.6) |  22.9→ 21.6 ( -1.3) |
| 23 |  -45.4° |   27.6° |  107.7° |   2.4 |   7.4→-45.4 (-52.7) |  14.9→ 27.4 (+12.5) |
| 24 |  -49.1° |   27.4° |  107.7° |   5.0 | -49.1→-39.1 (+10.0) |  27.4→  6.4 (-21.0) |
| 25 |  -39.1° |    6.9° |   92.0° |   1.9 | -39.1→-10.0 (+29.1) |   6.9→  8.3 ( +1.4) |
| 31 |  -55.9° |   85.7° |   59.9° |   3.6 |  13.5→ 61.5 (+48.0) |   2.8→-14.5 (-17.3) |
| 34 |  -37.4° |   67.9° |   80.4° |   1.9 |  18.3→  7.3 (-11.0) |  20.2→ 17.5 ( -2.7) |
| 41 |  -47.0° |   91.9° |   27.8° |   2.4 | -28.2→ 40.0 (+68.2) |  19.2→ -4.3 (-23.6) |
| 48 |  -34.0° |   72.8° |   80.5° |   2.2 | -28.5→-33.9 ( -5.4) |   5.9→ 11.1 ( +5.2) |
| 49 |  -34.0° |   77.4° |   99.3° |   4.3 | -33.9→-33.9 ( +0.0) |  20.0→ 15.8 ( -4.1) |

## SHALLOW Episodes Analysis (sample)

| Episode | min_elbow | Shoulder @ min | Wrist_pitch @ min | Gripper @ min | Start→End Elbow | Start→End Shoulder |
|---------|-----------|----------------|-------------------|---------------|-----------------|--------------------|
|  0 |   94.1° |   39.8° |   -0.3° |   1.8 |  94.1→ 94.1 ( +0.0) |  39.8→ 39.8 ( +0.0) |
|  1 |   34.5° |   75.9° |  -20.7° |   6.6 |  34.5→ 34.5 ( +0.0) |  75.9→ 75.9 ( +0.0) |
|  3 |   31.0° |   10.3° |   50.6° |   1.8 |  31.6→ 31.5 ( -0.2) |  -3.2→-12.9 ( -9.8) |
|  4 |    3.8° |   -9.6° |   83.5° |   2.5 |  48.9→  3.9 (-45.0) |  -5.0→ -9.5 ( -4.5) |
|  5 |   24.2° |   47.5° |   80.3° |   5.0 |  34.7→ 24.9 ( -9.8) |  13.5→ -7.8 (-21.4) |
|  6 |   20.1° |  -10.5° |   71.8° |   1.5 |  34.9→ 20.7 (-14.2) |  -7.3→-10.5 ( -3.3) |
|  7 |   11.0° |   -2.9° |   64.8° |   4.1 |  35.4→ 11.0 (-24.4) | -11.0→ -2.9 ( +8.1) |
|  8 |   -0.5° |   -8.3° |   82.1° |   2.6 |  11.6→ -0.5 (-12.1) |  -2.5→ -8.3 ( -5.8) |
|  9 |    6.1° |   -2.2° |   79.5° |   1.8 |   6.1→115.0 (+108.9) |  -2.2→ -7.5 ( -5.3) |
| 10 |   22.6° |   55.4° |   65.8° |  30.8 | 123.4→ 22.6 (-100.8) |  -7.5→ -7.2 ( +0.3) |

## Statistical Summary

**DEEP episodes** (n=9):
- Shoulder at min_elbow: mean=57.5°, range=[6.9°, 91.9°]

**SHALLOW episodes** (n=10 sampled):
- Shoulder at min_elbow: mean=19.5°, range=[-10.5°, 75.9°]

## Findings

### Pattern Detection

- **Descend-grasp-lift pattern**: 1/9 DEEP episodes show shoulder rising >10° (lift after grasp)

- **Shoulder positioning**: DEEP episodes have mean shoulder=57.5° vs SHALLOW mean=19.5° at min_elbow

### Interpretation

✓ DEEP episodes show distinct shoulder positioning compared to SHALLOW.

DEEP episodes tend to have higher shoulder angles, suggesting upright posture.

## Recommendations

1. **Visual inspection**: Review video frames at min_elbow for DEEP episodes 2, 23, 24, 25, 31, 34, 41, 48, 49
   - Check if gripper is actually near the table surface
   - Verify if arm is reaching DOWN vs extending OUTWARD

2. **Better metric**: Consider using a combination of:
   - Shoulder angle (forward lean indicator)
   - Elbow angle (extension indicator)
   - Wrist pitch (downward tilt indicator)
   - Gripper trajectory (open→close→lift pattern)

3. **Data collection strategy**:
   - If current DEEP episodes are NOT actually deep grasps, the 77-episode collection plan needs revision
   - Focus on ensuring the gripper actually contacts the table before grasping
   - Record Z-height from Kinect depth to validate grasp depth

