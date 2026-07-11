---
name: project_safety_status
description: Known failure modes, JOINT_LIMITS status, and deployment safety history for RoArm-M3 SmolVLA
type: project
---

Deployment safety history as of 2026-03-24:

**Successes**: open-loop 4-chunk, init start, 50K checkpoint → 5/5 (100%) on 2026-02-25

**Failure modes logged**:
- Wrist_R runaway: -3 → -92 degrees (4-sigma OOD drift, 2026-02-11 Run 1)
- Elbow unidirectional drift: 13 → 36 degrees (data imbalance, DEEP episodes too few)
- Gripper never opened: stayed 2-4 degrees (training data imbalance, gripper-open underrepresented)
- Closed-loop n=1: per-step noise accumulated → OOD → drift

**Current safety measures**: JOINT_LIMITS hardcoded (NEVER remove). ESP32 T:106 reset for motor bus recovery.

**Why open-loop 4-chunk succeeded**: Each 50-step chunk commits to trajectory within chunk, re-observes between chunks. Avoids per-step noise accumulation while still adapting every ~1.67 seconds.

**Files created**:
- monitor_camera_shift.py (2026-03-24): Pre-deployment diagnostic for camera repositioning.
  Computes SSIM, MAE, histogram correlation, and model inference z-score delta.
  Issues GO/CAUTION/STOP verdict. Run before any deployment after camera was moved.

**Files to create (not yet done)**: monitor_ood_detector.py, safety_joint_monitor.py

**Camera shift risk profile (2026-03-24 analysis)**:
- SmolVLA 74-ep model has narrow visual coverage: sensitive to observation shift
- Open-loop 4-chunk amplifies spatial offset errors (no per-step correction within chunk)
- Safe SSIM threshold: > 0.80 (CAUTION), > 0.65 (STOP)
- Safe action z-delta: < 1.0 (OK), 1.0-2.0 (CAUTION), > 2.0 (STOP)
- Expected failure mode from camera shift: spatial offset -> wrong base angle -> object miss

**Dual-camera wrist-mount analysis (2026-03-24)**:
- VERDICT: NOT RECOMMENDED for CoRL 2026 (May 28 deadline)
- Root problem: camera was moved → data is OOD. The fix is to fix the camera, not add more cameras.
- Wrist camera adds: cable entanglement risk, weight on wrist affecting precision, USB 3.0 bandwidth
  conflict between Azure Kinect + ZED Mini, SmolVLA architecture change needed (currently single-image),
  full data recollection + retraining required, 4-6 week engineering risk.
- Safe alternative: fix Azure Kinect position + recollect 100+ episodes (1-2 days).
- If dual-camera is genuinely needed for CoRL novelty, treat as v2 after baseline is re-established.

**Why:** Real deployment failures caused physical hardware risk and task failure. Safety monitoring prevents recurrence.
**How to apply:** Any new deployment scripts must have these monitors available. Cross-validate with deploy-agent.
Run monitor_camera_shift.py before deploying after any physical change to the setup.
