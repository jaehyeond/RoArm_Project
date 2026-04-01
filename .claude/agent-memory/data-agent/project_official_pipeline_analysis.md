---
name: Official SmolVLA Pipeline Analysis
description: Official lerobot-record has zero validation. Reference dataset stats. Our overconstrained checks identified.
type: project
---

Official lerobot-record script has ZERO episode validation — no gripper check, no frame count, no joint travel.
The only quality gate is the operator pressing Left Arrow to re-record manually.

**Why:** Leader-follower setup means operator physically controls arm — bad episodes are obvious and re-recorded.
Our torque-OFF manual setup has different failure modes: arm not returned to HOME, gripper not opened, etc.

**How to apply:** Our C0a (HOME start) and C5 (Z at grasp) are justified for our setup.
Checks C0b, C2, C3 are overconstrained and should be relaxed or removed.

## Reference Dataset (lerobot/svla_so100_pickplace)
- 50 episodes, 5 positions, 10 reps/position
- ~393 frames/ep = 13.1 seconds @ 30fps
- Training: 20K steps, batch=64
- Our v5: 99fr/ep (3.3s), 200K steps — wrong on both counts

## Our Validation Check Verdicts (2026-03-31)

| Check | Description | Verdict |
|-------|-------------|---------|
| C0a | HOME start distance <30deg | JUSTIFIED — unique to torque-OFF |
| C0b | Approach phase (base+shoulder travel >5deg) | OVERCONSTRAINED — redundant if C0a passes |
| C1 | Gripper open >40deg | JUSTIFIED but relax to 30deg |
| C2 | Gripper range >15deg | REDUNDANT — remove |
| C3 | Shoulder at grasp >40deg | OVERCONSTRAINED — WARNING only |
| C4 | Frame count >=120 (FAIL) | OVERCONSTRAINED — relax to 90 |
| C5 | Z at grasp <130mm | JUSTIFIED — physical check |

## Recommended Changes
1. Remove C0b
2. Relax C1: 40deg → 30deg
3. Remove C2
4. Convert C3 to WARNING only (threshold 40 → 30deg FAIL)
5. Relax C4 FAIL threshold: 120 → 90 frames
6. Keep C0a and C5 as FAIL

## Key Insight
The validation checks were NOT the cause of v5 failure.
v5 failed because episodes were collected starting at approach pose — C0a check was missing.
C0a was added correctly after v5 failure.
The over-aggressive OTHER checks may cause good episodes to be rejected unnecessarily.
