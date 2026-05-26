# 2026-05-22 Track A Contact-RL Stage 0 Preflight

## Scope

Track A only. This session did not run Isaac, PPO training, rollout collection,
dataset generation, hold-lift, transport/release, constraints, SurfaceGripper,
gate tuning, or any success claim.

The user's four-stage plan is valid in principle:

1. RL learns a task from random action + reward.
2. The trained policy becomes the expert.
3. Expert rollouts record state/action/observation.
4. Rollouts become LeRobot/RLDS-style demos for IL/VLA training.

The critical correction is that this pipeline needs a Stage 0 gate: the RL env
must expose a Track A-valid no-attach contact primitive before B200 PPO can
produce a valid expert.

## Reverified Evidence

Copied the v4 B200 stdout/audit logs into local `/tmp` and reverified md5s:

- stdout `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_v7_close26_b200.out`
  md5 `fe6a733727a6eeb288c6c6464c178af1`
- audit `/tmp/p7_branch_b_cube2cm_target_guarded_micro_close_v4_recovery_audit_b200.out`
  md5 `47f4ec7b78298fde0a46ac57105a6e6c`

The Stage 0 preflight script rechecked:

- stdout line 37: diagnostic-only, close_26-only, no training, no posewrite.
- stdout line 391: first hard freeze at `target_error_m=0.003035` and
  counter gap `0.002050m`.
- stdout line 421: final gripper `7.977deg`, command `8.000deg`,
  remaining close `18.023deg`.
- stdout line 423: `close_reached=NO`, hard freezes `31`, attach/posewrite `0`.
- audit lines 16/28/54: close, hard-freeze, final criteria all fail.

## Code Added

Added:

- `sim_scripts/p7_branch_b_contact_rl_stage0_preflight_static_analysis.py`
  md5 `73fa3e8dc18fcc4a0e5a4cf702985eee`

This script verifies v4 md5s and checks existing PPO env semantics:

- `roarm_rl/train_ppo.py` uses `RoArm-Pick-Direct-v0` and
  `RoArm-Stack-Direct-v0`.
- `roarm_rl/roarm_pick_env.py` and `roarm_rl/roarm_stack_env.py` use
  kinematic attach / `write_root_pose_to_sim`.
- Therefore direct B200 PPO on the existing envs is not Track A-valid no-attach
  contact evidence.

Added:

- `sim_scripts/p7_branch_b_cube2cm_contact_rl_v5_static_design.py`
  md5 `ab1b5c0b1b0655ebef4dc9c42d3e8de1`

This script quantifies the v5 preemptive recovery point:

- v4 line 390 is the last safe pre-freeze step:
  target error `0.002891m`, target margin `0.000109m`, counter gap
  `0.001969m`, support margin `0.000031m`, recovery hold YES.
- v4 line 391 is already too late:
  target error `0.003035m > 0.003m`, counter gap `0.002050m > 0.002m`.
- Therefore v5 must recover target/support before line 391, not just hold.

## Verification

Local/static verification:

- `python -m py_compile` for both new scripts: PASS.
- `python sim_scripts/p7_branch_b_contact_rl_stage0_preflight_static_analysis.py`:
  PASS, returns direct B200 PPO now `NO`.
- `python sim_scripts/p7_branch_b_cube2cm_contact_rl_v5_static_design.py`:
  PASS, identifies line 390 preemptive trigger.
- `git diff --check`: PASS.

## Current Decision

The four-stage RL-to-expert-to-demo plan is the right high-level pipeline, but
not executable yet on current Track A evidence.

Do not run PPO on the existing default Pick/Stack envs as Track A expert
generation. They are attach-based and would not prove no-attach contact grasp.

Next Track A work is still local/static/code-first:

- implement a no-attach contact RL Stage 0 env or a v5 contact-close gate;
- v5 must preempt at line 390-like margins, not wait for line 391 hard freeze;
- only robot joint targets may be written; object attach/posewrite/constraints
  remain forbidden;
- fixed target/support gates remain unchanged;
- random-action sanity, PPO training, expert rollout, and dataset generation all
  require separate approval after this Stage 0 readiness exists.
