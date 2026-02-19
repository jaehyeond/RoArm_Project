# Data Collection Strategy - Phase 1 Diagnosis (Step 4)

## Executive Summary

**Root Cause Identified:** 68% SHALLOW data → model learned *wrong* direction
- Model thinks: APPROACH position = LIFT (go up)
- Should be: APPROACH position = REACH DOWN (go down)
- Gripper bias: Most frames have gripper closed → model never learned to open

**Solution:** Rebalance dataset to 50% DEEP, 30% APPROACH, 20% SHALLOW

## Current Dataset Analysis (50 episodes)

| Grade | Count | Percentage |
|-------|-------|------------|
| DEEP | 9 | 18.0% |
| APPROACH | 7 | 14.0% |
| SHALLOW | 34 | 68.0% |

**Problematic Episodes:** 7 episodes should be removed
- Static (no motion): 2
- No gripping: 7

### Gripper Analysis

- Average gripper range: 35.8°
- Average gripper min: 1.8°
- Average gripper max: 37.6°

**Problem:** Gripper stays mostly closed (min ≈ 1-2°), rarely opens wide.

## Target Distribution (120 total episodes)

| Grade | Target Count | Percentage | Requirements |
|-------|--------------|------------|--------------|
| DEEP | 60 | 50% | See below |
| APPROACH | 36 | 30% | See below |
| SHALLOW | 24 | 20% | See below |

### DEEP Requirements

**Description:** Elbow < -30°, full reach down

**Requirements:**
- Elbow min < -30°
- Gripper opens (>20°) then closes (<10°)
- Smooth descent trajectory
- No static frames (elbow range > 10°)

### APPROACH Requirements

**Description:** Elbow -30° to -10°, medium depth

**Requirements:**
- -30° < Elbow min < -10°
- Gripper activity (range > 15°)
- Clear reaching motion

### SHALLOW Requirements

**Description:** Elbow > -10°, surface operations

**Requirements:**
- Elbow min > -10°
- Gripper activity (range > 10°)
- Functional motion (not static)

## Episodes to Remove

Remove 7 episodes from the dataset:

```
Episode 0
Episode 1
Episode 2
Episode 14
Episode 21
Episode 22
Episode 45
```

## Collection Protocol

### Before Starting

1. **Camera Position:** Ensure camera is FIXED (clamp/tripod). Any movement = all data invalid!
2. **Workspace Setup:** Clear workspace, consistent object placement
3. **Robot State:** Power cycle, run `scan_servos.py` if needed
4. **Lighting:** Consistent lighting (no shadows/glare)

### During Collection

**Real-Time Display (to be added to collect_data_manual.py):**

```
Current Progress:
  DEEP:     [####------] 25/60  (42%)
  APPROACH: [###-------] 12/36  (33%)
  SHALLOW:  [##--------]  8/24  (33%)

NEXT: Collect DEEP grasp (elbow < -30°)
```

### Per-Episode Checklist

Before accepting an episode, verify:

- [ ] Elbow moved > 10° (not static)
- [ ] Gripper opened (>20°) AND closed (<10°)
- [ ] Smooth, controlled motion (no jerks)
- [ ] Target elbow depth achieved:
  - DEEP: elbow < -30°
  - APPROACH: -30° < elbow < -10°
  - SHALLOW: elbow > -10°
- [ ] RGB camera captured all frames

### Collection Tips

**For DEEP grasps (elbow < -30°):**
- Start with arm raised (elbow ≈ 30-50°)
- Open gripper WIDE (>30°)
- Slowly lower arm (bend elbow DOWN to -40° ~ -60°)
- Close gripper around object (<5°)
- Lift back up

**For APPROACH grasps (-30° < elbow < -10°):**
- Medium height start
- Open gripper (>20°)
- Lower to -15° ~ -25°
- Close gripper
- Slight lift

**For SHALLOW grasps (elbow > -10°):**
- Surface-level operations
- Still need gripper activity!
- Avoid static hovering

## Post-Collection Validation

After collecting all episodes, run:

```bash
python data_episode_quality.py
python data_distribution_simple.py
```

Verify:

- Distribution matches target (50% DEEP, 30% APPROACH, 20% SHALLOW)
- All episodes have gripper activity (range > 15°)
- No static episodes (elbow range > 10°)
- Gripper open/close cycle visible in most episodes

## Implementation Notes

To add real-time guidance to `collect_data_manual.py`:

1. Load `analysis_corrected.csv` at startup
2. Count existing DEEP/APPROACH/SHALLOW episodes
3. Display progress bars after each episode
4. Recommend next episode type based on targets
5. Compute metrics immediately after recording (min_elbow, gripper_range)
6. Show ACCEPT/RETRY prompt with reasons
