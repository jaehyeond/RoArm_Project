# Ideal Data Collection Protocol for SmolVLA Pick-and-Place Training
# RoArm M3 + Azure Kinect

**Pipeline Agent - 2026-02-23**

---

## Research Basis

This protocol is grounded in:
1. SmolVLA official documentation (`lerobot/docs/source/smolvla.mdx`, `il_robots.mdx`)
2. SmolVLA paper dataset structure (50 eps, 5 positions x 10 reps)
3. v1 deployment failures (2x) diagnosed in `train_deployment_failure_analysis.md`
4. Previous gripper timing failure: 58% too early, 40% too late, 2% correct

---

## 1. Official SmolVLA Dataset Structure (Reference Standard)

From the official docs and paper (SVLA SO100 PickPlace dataset):

```
50 episodes total
5 distinct cube positions
10 episodes per position (repetitions)
Full pick-and-place cycle (approach → grasp → lift → place → return)
Episode time: ~60 seconds per episode
```

Key quote from docs:
> "In this dataset, we recorded 50 episodes across 5 distinct cube positions. For each position,
> we collected 10 episodes of pick-and-place interactions. This structure, repeating each variation
> several times, helped the model generalize better. We tried similar dataset with 25 episodes,
> and it was not enough leading to bad performance."

**Our target**: 100 episodes (100+ to fix v1 failures). Structure: 5 positions x 20 reps.

---

## 2. The 7-Phase Perfect Episode

At 30fps, a pick episode should take 5-8 seconds total (~150-240 frames).
NEVER record more than 300 frames (10s) per episode — excess creates static frames.

### Phase 1: START POSITION
**Duration**: 0.5s (15 frames)
**What**: Robot at the dataset_mean position, gripper CLOSED
**Joint targets**:
```
Base:        0° to +15° (facing object zone)
Shoulder:   +35° to +45°
Elbow:      +10° to +20°
Wrist Pitch: +55° to +70°
Wrist Roll:  -10° to +10°
Gripper:     0° to 5° (CLOSED)
```
**Z-height**: 250-350mm (arm is HIGH, in standby)
**Why this matters**: Dataset_mean start reduces OOD gap at deployment.

### Phase 2: APPROACH (moving toward object)
**Duration**: 1.0s (30 frames)
**What**: Arm moves horizontally toward object, elbow starts bending down
**Joints moving**: Base (coarse alignment), Shoulder (down), Elbow (starting to flex)
**Gripper**: OPEN — open the gripper EARLY during approach, not at grasp point
```
Base:        adjusts to position (see position table below)
Shoulder:    from +40° toward +25°
Elbow:       from +15° toward -10°
Wrist Pitch: from +60° toward +70° (maintains orientation)
Wrist Roll:  adjusts to object orientation
Gripper:     OPENS from 5° → 50° to 70° (WIDE OPEN)
```
**Z-height**: 200-300mm (descending)
**CRITICAL**: Gripper MUST be opening during this phase, not at Phase 4.
This is where v1 failed — gripper stayed closed until too late.

### Phase 3: PRE-GRASP (hovering above object)
**Duration**: 0.5s (15 frames)
**What**: Arm is positioned directly above the object, gripper fully open
**All joints**: Nearly static (small adjustments)
**Gripper**: FULLY OPEN at 60-80°
```
Z-height: 150-200mm (low, above object)
Elbow:    -20° to -35°
```
**Why hold here**: Gives the model clear frames of "arm above object, gripper open"
This phase is the temporal anchor for grasp timing.

### Phase 4: GRASP DESCENT (lowering onto object)
**Duration**: 0.5s (15 frames)
**What**: Arm moves DOWN to contact object level, gripper STILL OPEN
**Joints moving**: Elbow flexes more, Shoulder continues down
```
Elbow:    -35° to -55° (DEEP EXTENSION — this is the DEEP zone)
Shoulder: +20° to +15°
Z-height: 80-120mm (object contact level)
Gripper:  STILL OPEN at 60-80°
```
**CRITICAL**: DO NOT close gripper yet. The gripper must arrive OPEN at object level.
This is the most common mistake — closing too early while still descending.

### Phase 5: GRASP (gripper closes)
**Duration**: 0.3s (10 frames)
**What**: Gripper closes firmly on object
**All other joints**: STATIC (arm does not move during close)
```
Elbow:    -55° to -65° (deep, around object)
Gripper:  CLOSES from 70° → 5° to 15° (firm grip)
Z-height: 80-120mm
```
**CRITICAL timing requirement**:
- Close gripper ONLY when arm is at object level
- All other joints must be static or nearly static
- This creates a clean "gripper close" signal in data
- Takes only 10 frames (~0.3s) — quick and decisive

### Phase 6: LIFT
**Duration**: 1.0s (30 frames)
**What**: Arm moves UP with object in gripper
**Joints moving**: Shoulder, Elbow (reversing the descent)
**Gripper**: STAYS CLOSED at 5-15° throughout
```
Elbow:    -65° → -30° → 0° (extending back up)
Shoulder: +15° → +30° → +40°
Z-height: 80mm → 200mm → 300mm
```
**Why lift matters**: Without lift frames, model learns "grasp and stop" not "grasp and lift".
Every episode MUST include a full lift to 200mm+ Z-height.

### Phase 7: PLACE / RETURN
**Duration**: 1.0s (30 frames)
**What**: Arm returns to start position (or moves to a place target)
**Option A (pick-only)**: Return to start position with object, then open gripper
**Option B (pick-and-place)**: Move to place target, lower, open gripper, retract
```
Recommended: Option A for simplicity
Gripper opens at return position: 5° → 60° (release)
Return to start: Elbow back to +15°, Shoulder back to +40°
```
**Why return matters**: Provides return trajectory data, avoids dataset bias toward "end pose".
The official SmolVLA dataset always includes the return phase.

### Phase Summary Table

| Phase | Duration | Frames | Gripper State | Z-height | Elbow Range |
|-------|----------|--------|--------------|----------|-------------|
| 1. Start | 0.5s | 15 | CLOSED (0-5°) | 250-350mm | +10° to +20° |
| 2. Approach | 1.0s | 30 | OPENING (5→70°) | 200-300mm | +15° to -10° |
| 3. Pre-grasp | 0.5s | 15 | OPEN (60-80°) | 150-200mm | -20° to -35° |
| 4. Descent | 0.5s | 15 | OPEN (60-80°) | 80-120mm | -35° to -55° |
| 5. Grasp | 0.3s | 10 | CLOSING (70→5°) | 80-120mm | -55° to -65° |
| 6. Lift | 1.0s | 30 | CLOSED (5-15°) | 80→300mm | -65° to +15° |
| 7. Return | 1.0s | 30 | OPEN at end (→60°) | 200-350mm | +10° to +20° |
| **Total** | **5.3s** | **145** | — | — | — |

---

## 3. 5 Object Positions Layout

### Physical Setup

Place colored tape markers on the table. Robot base is the origin.

```
                        [AZURE KINECT CAMERA]
                         (fixed, front-facing)
                              |
                              | ~600mm distance
                              |
         ←————————————————————↓————————————————————→
                           [ROBOT]
                           BASE

        TABLE SURFACE (viewed from above):

                    P3 (center-far)
                   300mm from base

        P2 (left-mid)    P1 (center)    P4 (right-mid)
        250mm, -30°     250mm, 0°      250mm, +30°

                    P5 (center-near)
                   200mm from base
```

### Exact Position Coordinates

All distances are from the robot base center, measured on the table surface.

| Position | Distance | Angle | Tape Color | Expected Elbow at Grasp | Use |
|----------|----------|-------|------------|------------------------|-----|
| P1 center | 250mm | 0° | Blue | -50° to -60° | Primary (most reps) |
| P2 left | 250mm | -30° | Yellow | -45° to -55° | Base variation |
| P3 far | 300mm | 0° | Red | -60° to -70° | DEEP reach practice |
| P4 right | 250mm | +30° | Green | -45° to -55° | Base variation |
| P5 near | 200mm | 0° | White | -35° to -45° | Easier reach |

### Episode Distribution (100 total)

| Position | Episodes | Notes |
|----------|----------|-------|
| P1 center | 30 | Most important — establish baseline |
| P2 left | 20 | Trains base left rotation |
| P3 far | 20 | DEEP episodes (critical for fixing v1 failure) |
| P4 right | 20 | Trains base right rotation |
| P5 near | 10 | Shallower reach variation |

**IMPORTANT**: P3 (far, red tape) produces the deepest elbow extensions (-60° to -70°).
We need at least 20 P3 episodes to fix the "DEEP grasp 9/50" failure from v1.

### Tape Setup Instructions

1. Place robot on a fixed non-slip surface (use rubber mat or clamp to table).
2. Mark robot base center with a cross-hair on the table.
3. Use a ruler and protractor to place tape markers:
   - P1: 250mm directly in front of base, centered
   - P2: 250mm, rotate tape marker 30° left from center line
   - P3: 300mm directly in front of base
   - P4: 250mm, rotate tape marker 30° right from center line
   - P5: 200mm directly in front of base
4. Photograph the setup for reference before each session.
5. Verify positions match after any disturbance (bumping table, etc.).

---

## 4. Gripper Timing Protocol (Critical Fix for v1 Failure)

**v1 failure stats**: 58% opened too early, 40% too late, only 2% correct.

The correct gripper timeline is:

```
Frame:   0    15    45    60    75    85    95    125   145
Phase:  [Start][-------Approach------][Pre][Desc][GRSP][Lift][Return]
Gripper:[  0°  ][  0° → 70°          ][ 70° ][ 70° ][ 5° ][  5°  ][ 5°→60°]

Key rule: Open during APPROACH (frames 15-45)
          Close during GRASP (frames 85-95, ~10 frames)
          Never close before reaching object Z-height
```

### Gripper Opening Cue
Open the gripper at the START of the approach, not at the object.
Physical cue: "When you start moving the arm forward/down, simultaneously open the gripper."

### Gripper Closing Cue
Close the gripper ONLY when:
- The arm has stopped descending (Z-height is stable at 80-120mm)
- The gripper fingers are at the sides of the object
- The arm is NOT still moving downward

Physical cue: "Feel the object between the gripper fingers before closing."

### Common Mistake Prevention Table

| Mistake | What It Looks Like | Why It Hurts Training | Prevention |
|---------|-------------------|----------------------|------------|
| Early close | Gripper closes at Z=200mm, before reaching object | Model learns "close while descending" | Open cue: begin opening at arm start |
| Late close | Gripper closes after lift has begun (Z increasing) | Model learns "close while rising" | Close cue: stable Z, object contacted |
| Partial open | Gripper only opens to 30° | Small margin for grasping; model learns low gripper range | Must reach 60°+ before descent |
| Slow close | Gripper takes 1.5s to fully close | Model spreads close signal over many frames; noisy label | Close decisively in 0.3s (10 frames) |
| No return | Episode ends at lift peak | Dataset biased toward "stay up" | Always complete full return to start |
| Too slow | Episode takes 15s+ (450+ frames) | 50-60% static frames; "freeze" bias | Keep total under 300 frames (10s) |
| Jerky moves | Fast, discontinuous arm motion | Non-smooth trajectory; hard to learn | Move at consistent moderate pace |
| Starting wrong | Start at arbitrary arm position | Dataset inconsistent start distribution | Always start from I key (init position) |

---

## 5. Session-by-Session Collection Plan

### Pre-Session Checklist (do every session)

```
[ ] Camera position unchanged (compare to reference photo)
[ ] Robot is at init position (press I key to verify)
[ ] Table markers (tape) in correct positions
[ ] Object (white box) centered on current position tape
[ ] Terminal shows current stats (DEEP/APPROACH/SHALLOW counts)
[ ] Goal for this session noted (e.g., "20 P3 DEEP episodes")
```

### Session Schedule (to reach 100 episodes)

**Session 1 (Day 1 morning): P1 Center, 20 episodes**
- Goal: Establish baseline. FOCUS on correct gripper timing.
- Verify gripper opens during approach (check terminal output).
- Target Z at grasp: 80-120mm (watch camera OSD).

**Session 2 (Day 1 afternoon): P3 Far (red tape), 20 episodes**
- Goal: Fill the DEEP gap. This is the most important session.
- Elbow will reach -60° to -70° — this is correct and expected.
- Watch for Z < 100mm (green DEEP zone indicator).

**Session 3 (Day 2 morning): P2 Left + P4 Right, 20 episodes each**
- Alternate between P2 and P4 every 5 episodes.
- Base joint will rotate left (-30°) and right (+30°).
- Verify gripper timing is consistent across positions.

**Session 4 (Day 2 afternoon): P5 Near + P1 mixed, 10+10 episodes**
- P5: Shallower reach (Z 120-150mm at grasp is acceptable).
- P1: Additional reps for robustness.

**Total: 100 episodes across 4 sessions (2 days)**

---

## 6. Real-Time Quality Check During Collection

The `collect_data_manual.py` script shows live OSD. Watch for:

### GREEN indicators (good):
- `Height: XXmm [DEEP]` during grasp phase
- `Gripper range > 40°` after episode
- `Min Z < 100mm` at save confirmation

### YELLOW/ORANGE indicators (acceptable but warn):
- `Height: XXmm [APPROACH]` — Z is 100-200mm, OK but should go lower for P3
- `Gripper range: 30-40°`

### RED flags (discard episode with Backspace):
- Gripper range < 15° at episode end (gripper never opened properly)
- Min Z > 200mm for P1/P2/P3/P4 episodes (too shallow)
- Frame count < 50 (too short — missed phases)
- Frame count > 350 (too long — too many static frames)

### After each episode, verify in terminal:
```
Episode XX saved!
  Frames: [target: 120-280]
  Min Z-Height: [target: < 120mm for P1-P4, < 150mm for P5]
  Gripper Range: [target: > 40°]
```

---

## 7. What the Official SmolVLA Dataset Does (Reference)

From the SVLA SO100 PickPlace dataset (visualized at the HF Space):

- Full pick-and-place: approach, grasp, lift, translate, place, return
- Gripper opens BEFORE reaching the cube
- ~200-250 frames per episode at 30fps
- Cube placed at precise positions with tape markers
- 5 positions, 10 reps each = 50 episodes
- Task text: "Pick up the [object] and place it in [location].\n" (note: \n required for SmolVLANewLineProcessor)

For our task, we use only the pick + lift + return (not place to new location), which is:
- Simpler (one location)
- Less prone to position drift at deployment

---

## 8. Episode Quality Standards

### Minimum Acceptable (keep but flag):
- Gripper range >= 20°
- Min Z <= 200mm
- Frames: 80-350
- Elbow reaches at least -20°

### Good (target most episodes):
- Gripper range >= 40°
- Min Z <= 120mm
- Frames: 120-250
- Elbow reaches -40° to -60°

### Excellent (DEEP, count carefully):
- Gripper range >= 50°
- Min Z <= 100mm
- Frames: 140-220
- Elbow reaches -55° to -70°

### Discard immediately (Backspace):
- Gripper range < 15° (model will learn "stay closed")
- Min Z > 200mm AND position is P1/P2/P3/P4 (too shallow)
- Frames < 50 (incomplete episode, missed grasp phase)
- Frames > 400 (too many static frames, "freeze" bias)
- Any collision, jerky movement, or arm hitting table

---

## 9. Dataset Distribution Target (100 episodes)

### By Position:
```
P1 center (250mm, 0°):    30 episodes  (30%)
P2 left   (250mm, -30°):  20 episodes  (20%)
P3 far    (300mm, 0°):    20 episodes  (20%)
P4 right  (250mm, +30°):  20 episodes  (20%)
P5 near   (200mm, 0°):    10 episodes  (10%)
```

### By Z-Height Classification:
```
DEEP (Z < 100mm):         50+ episodes (50%+)  ← most important fix
APPROACH (Z 100-200mm):   30 episodes  (30%)
SHALLOW (Z > 200mm):      20 episodes  (20%)
```

Collect_data_manual.py already tracks these ratios and shows the recommendation:
- Terminal will say "Next: DEEP GRASP" when DEEP count is below 50%
- Always follow this recommendation

### By Gripper Timing:
ALL episodes must have gripper opening during Approach phase (not at grasp).
Track this manually: if the DEEP zone indicator appears before gripper is open,
discard the episode.

---

## 10. Environment Setup Requirements

### Camera (DO NOT CHANGE after first session):
```
Position: Front-facing, ~600mm from robot base
Height:   Roughly at table level +300mm (sees both arm and table)
Angle:    Tilted down ~15-20° to see table surface
Clamp:    MUST be on tripod or table clamp — NOT hand-held
```

### Lighting:
- Consistent across all sessions
- Avoid strong shadows on object (confuses VLM backbone)
- If possible, use same time of day for each session

### Object (white box):
- Same box across ALL sessions
- Always place on center of tape marker
- Always same orientation (e.g., long side facing camera)
- White box provides good contrast — do not use red/dark objects

### Table surface:
- Non-slip surface preferred
- Keep table clear of other objects (confused VLM)
- Clean table surface before each session

---

## 11. Gripper Angle Reference for RoArm M3

Based on joint specs and v1 dataset observations:

| State | Angle Range | When |
|-------|-------------|------|
| CLOSED | 0° to 5° | Start position, during lift, return |
| PARTIAL | 5° to 30° | Transition (minimize time here) |
| OPEN | 30° to 60° | Approach, pre-grasp, descent |
| WIDE OPEN | 60° to 80° | Pre-grasp hold, descent to object |
| GRASPING | closing from 70° to 5°-15° | Only at object level |

Note: Gripper range 0-100° per CLAUDE.md, but practical range for pick-and-place is 0-80°.
The `data_episode_quality.py` threshold of 30° for "open" is the minimum — target 60°+.

---

## 12. Script Usage During Collection

```bash
# Start collection
conda activate roarm
cd /home/cgxr/Documents/Robotics/RoArm_Project
python collect_data_manual.py

# Controls:
# Space:    Start/stop recording (toggle)
# Enter:    Save episode
# Backspace: Discard episode (bad quality)
# T:        Toggle torque (off for guiding)
# I:        Return to init position (use between episodes)
# ESC:      Exit

# Recommended workflow per episode:
# 1. Press I → arm returns to init (standard start)
# 2. Torque OFF (or already off from T key)
# 3. Place white box on current position tape
# 4. Press Space → recording starts
# 5. Guide arm through all 7 phases
# 6. Press Space → stop (or Enter → stop+save)
# 7. Review Z and Gripper stats in terminal
# 8. Press Enter to save, or Backspace to discard
# 9. Repeat from step 1
```

---

## 13. Post-Collection Verification

After each session, run the quality analysis:

```bash
# Quality analysis (after convert_to_lerobot_v3.py)
python data_episode_quality.py

# Check output:
# - Good episode rate > 80%
# - DEEP episodes (elbow < -40°) > 40% of total
# - Mean max gripper > 50°
# - Episodes with gripper > 30° = 100%
```

Target output after 100 episodes:
```
Total episodes: 100
Good episodes:  >80 (>80%)
DEEP episodes:  >50 (>50%)

ELBOW STATISTICS:
Mean min elbow: < -35° (v1 was -13°, we must improve)
Episodes with elbow < -40°: > 50

GRIPPER STATISTICS:
Mean max gripper: > 55°
Episodes with gripper > 30°: 100 (100%)
Episodes with gripper > 50°: > 80 (80%)
```

---

## Summary: The 3 Most Important Rules

### Rule 1: Gripper opens DURING APPROACH, not at the object
Start opening the gripper immediately when you start moving the arm toward the object.
The gripper must be WIDE OPEN (60°+) before the arm begins descending.

### Rule 2: Every episode must reach Z < 120mm (DEEP zone)
The arm must fully lower to object level. Watch the OSD — wait for green "DEEP" indicator
before closing the gripper. If you close at APPROACH zone, discard the episode.

### Rule 3: Complete all 7 phases in every episode
Start from init position, open gripper during approach, descend, close at object,
lift to 300mm+, return to start. Never record a partial episode.

---

**[PIPELINE AGENT] Protocol Version**: 1.0
**Based on**: SmolVLA docs + v1 failure analysis + official dataset structure
**Target**: 100 episodes, 5 positions x 20 reps, 50%+ DEEP, 100% correct gripper timing
