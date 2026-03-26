---
name: Isaac Lab RoArm M3 Failure Diagnosis
description: Root cause analysis of the failed 100-iter Isaac Lab reaching policy deployment on real RoArm M3 (2026-03-26)
type: project
---

# Isaac Lab → RoArm M3 Transfer Failure: Root Cause Analysis

**Artifact examined**: `/home/cgxr/Documents/Robotics/isaac_roarm_m3/logs/rsl_rl/roarm_m3_reach/2026-02-20_15-30-43/`

## Confirmed Root Causes (from code inspection)

### 1. Training not converged (CRITICAL)
- Only 100 iterations run (26 seconds at 49K steps/sec)
- Position error at 100 iter = 0.097m — still in early exploration
- Minimum needed: 1000 iter (~4.3 min), recommended: 2000 iter (~9 min)
- Deploying model_99.pt was equivalent to deploying random noise

### 2. Action space mismatch (CRITICAL)
- Sim outputs: `JointPositionAction(scale=0.5, use_default_offset=True)` → relative delta in radians
- Real API: `arm.joints_angle_ctrl(angles=[...])` → absolute degrees
- No conversion code was built or mentioned

### 3. Control frequency not synchronized (HIGH)
- Sim policy frequency: 30Hz (sim.dt=1/60, decimation=2)
- Real loop: no rate control, USB serial latency variable
- Required: `time.sleep(dt - elapsed)` loop at 30Hz

### 4. No domain randomization (HIGH)
- Only initial position randomized (reset_robot_joints)
- NO mass randomization, NO joint friction randomization, NO actuator delay
- Default physics: static_friction=0.5, dynamic_friction=0.5 (fixed, unrealistic)

### 5. Observation vector incompatible with real deployment (HIGH)
- Sim obs includes `pose_command` (7-dim target EE pose)
- In real deployment, this requires external object detection → coordinate transform
- Without camera-based object localization, the RL policy has no target to reach

## Key Insight: Why VLA was the right move

RL reach policy requires explicit EE target coordinates as input — essentially
requiring an object detection pipeline. VLA integrates this: image → action directly.
For vision-based pick-and-place, VLA solves the full problem; RL only solves the
"move to given coordinates" subproblem.

## Decision: Do NOT return to Isaac Lab now

**Why:** Stage 1 bottleneck is data quantity, not physics simulation.
Isaac Lab would require building: convergence training + action conversion +
DR + sysid + deployment wrapper + separate object detection pipeline.

**Why:** VLA already handles the full vision-to-action pipeline.

**When to revisit:** Stage 2+ after >60% pick-and-place success with VLA.
Potential use: sim data augmentation, trajectory priors, contact-rich tasks.

## Files
- Analysis: `/home/cgxr/Documents/Robotics/RoArm_Project/sim_isaac_failure_analysis.py`
- Isaac Lab source: `/home/cgxr/Documents/Robotics/isaac_roarm_m3/src/isaac_roarm_m3/`
