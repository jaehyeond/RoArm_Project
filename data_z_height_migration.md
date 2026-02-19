# Z-Height Migration for Data Collection

## Overview
Modified `collect_data_manual.py` to use Z-height from forward kinematics (FK) instead of elbow angle for depth classification.

## Changes Made (2026-02-12)

### 1. Added `get_robot_pose()` method
- Safe FK reading with 5 retries (like `get_robot_angles()`)
- Returns `[x_mm, y_mm, z_mm, tilt_deg, roll_deg, gripper_deg]`
- Returns `None` on failure

### 2. Added Z-height tracking variables
- `self.min_z = 9999` (lowest Z reached during episode)
- `self.max_z = -9999` (highest Z reached during episode)
- `self.current_pose = None` (cache for display)

### 3. Updated main loop to get pose
- Calls `get_robot_pose()` alongside `get_robot_angles()`
- Tracks min/max Z during recording
- Caches pose for OSD display

### 4. Updated OSD display
- **OLD**: Big "Elbow" display with DEEP/APPROACH/SHALLOW based on elbow angle
- **NEW**: Big "Height: XXXmm [ZONE]" display with Z-based zones
- Z zones:
  - **DEEP**: Z < 100mm (GREEN)
  - **APPROACH**: 100-200mm (YELLOW)
  - **SHALLOW**: > 200mm (ORANGE)
- Added X, Y coordinates display (smaller)
- Kept elbow angle display but smaller

### 5. Updated `save_frame()`
- Added `pose` parameter
- Saves `pose[:3]` (X, Y, Z in mm) to frame data
- **IMPORTANT**: Still saves angles for backward compatibility with `convert_to_lerobot_v3.py`

### 6. Updated `validate_episode()`
- **OLD**: `min_elbow > -10` → warning
- **NEW**: `min_z > 200mm` → warning
- Kept gripper range check (< 15°)
- Kept frame count check (< 50)

### 7. Updated `save_episode()` metadata
- Added: `min_z`, `max_z`, `z_range` fields
- Kept: `min_elbow`, `max_elbow`, `elbow_range` (backward compat)
- Z-based quality classification:
  - DEEP: min_z < 100mm
  - APPROACH: 100-200mm
  - SHALLOW: > 200mm
- Print summary shows both Z-height and elbow angle

### 8. Updated `DatasetStats.analyze_existing_episodes()`
- Checks for `min_z` field first (new episodes)
- Falls back to `min_elbow` field (old episodes)
- Classifies by Z when available, elbow otherwise
- **Backward compatible** with 50-episode dataset

### 9. Updated reset logic
- `cancel_episode()`: resets min_z, max_z
- After `save_episode()`: resets min_z, max_z

### 10. Updated recording stats display
- **OLD**: "Min Elbow: XX° | Gripper: XX°"
- **NEW**: "Min Z: XXmm | Gripper: XX°"

## Z-Height Thresholds (Initial Values)

| Zone | Z Range | Color | Target % |
|------|---------|-------|----------|
| DEEP | < 100mm | GREEN | 50% |
| APPROACH | 100-200mm | YELLOW | 30% |
| SHALLOW | > 200mm | ORANGE | 20% |

**NOTE**: These thresholds are initial estimates and can be tuned after testing with actual hardware.

## Backward Compatibility

### Old Episodes (50 episodes, elbow-based)
- `metadata.json` has `min_elbow`, NO `min_z`
- `DatasetStats` falls back to elbow classification
- Episodes remain valid for analysis

### New Episodes (Z-based)
- `metadata.json` has BOTH `min_z` AND `min_elbow`
- `DatasetStats` uses Z classification
- Frame data has BOTH `angles` AND `pose`

### Training Pipeline
- `convert_to_lerobot_v3.py` still uses `angles` field (unchanged)
- Pose data is stored but not required for training
- Can be used for future analysis/debugging

## Testing Checklist

Before data collection:
- [ ] Verify `pose_get()` works with torque OFF
- [ ] Check Z=0 is at base level
- [ ] Measure table surface Z (should be 50-80mm)
- [ ] Test one episode, verify metadata has min_z/max_z
- [ ] Verify OSD shows Z-height correctly
- [ ] Test backward compat: old episodes still load

During collection:
- [ ] Monitor Z values match physical height
- [ ] Adjust thresholds if needed (100mm, 200mm)
- [ ] Verify gripper validation still works

After collection:
- [ ] Run `data_episode_quality.py` on new dataset
- [ ] Verify distribution shows Z-based classification

## Known Constraints

1. **pose_get() may be slow (~10ms)**: Called alongside joints_angle_get(), not as replacement
2. **Z is absolute from base**: 0mm = base level, higher = further up
3. **Table surface varies**: ~50-80mm depending on setup
4. **Thresholds are initial**: May need tuning after hardware testing

## Files Modified

- `/home/cgxr/Documents/Robotics/RoArm_Project/collect_data_manual.py`

## Files Created

- `/home/cgxr/Documents/Robotics/RoArm_Project/data_z_height_migration.md` (this file)
