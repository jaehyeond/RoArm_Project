# Session 2026-05-18 - P7 Branch B approach-stage target-delivery probe

## Scope Guard

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper to the RoArm chain.
- Did not go to the transport target.
- Did not execute release or scripted release variants.
- Did not run P7 training or tune scalar/threshold/release guidance.
- Did not edit env/train/chain defaults.

## Boot / Cross-Checks

- Read `CLAUDE.md` Current-State Protocol first.
- Read `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D037,
  `claudedocs/EXPERIMENT_LEDGER.md`, and the requested Branch B session docs.
- Ran `git status --short` before coding; it had no output.
- Verified requested md5s before coding; all matched the prompt.
- B200 v3 target-delivery evidence was rechecked directly before coding:
  `/tmp/p7_branch_b_roarm_chain_post_latch_target_delivery_v3_b200.out`
  lines 83-85, 110-112, 134-135.

## Code Added

- Added `sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
  md5 `ebe8eddafd4c6f35c28e5b79a82511b3`.
- The script compares the same +5deg shoulder nudge at:
  settled HOME, early PRE_MOVE, high, hover, and grasp-before-CLOSE/open-gripper.
- It prints set-target call counts/diffs, Articulation target fields, current
  joints, shoulder-nudge error reduction, TCP target reduction, `_grasped`,
  gripper angle, action/null-action, soft limits, and env-step versus direct
  set+sim-step results.
- It keeps `_update_grasp_attach` marker-only inside the diagnostic. No
  constraint, SurfaceGripper, transport target, release, training, or default
  edit is used.

Local checks:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py`
- `python sim_scripts/p7_branch_b_roarm_chain_approach_target_delivery_probe.py --help`

## B200 Run

Authoritative final run:

- `/tmp/p7_branch_b_roarm_chain_approach_target_delivery_v2_b200.out`
- `/tmp/p7_branch_b_roarm_chain_approach_target_delivery_v2_b200.err`

Remote md5 matched local:

- `ebe8eddafd4c6f35c28e5b79a82511b3`

Evidence from v2 stdout:

- Line 41 confirms strict scope: no constraint prim insertion, no fixed/dynamic
  integration, no SurfaceGripper, no attached transport, no transport target, no
  release, no P7 training, no default edits, and explicitly
  `attach_physics_validated=NO`, `release_physics_validated=NO`,
  `claim_attach_success=NO`.
- Line 43 confirms execution remains pre-transport:
  source `events_total=44`, executed pre-moves `38`, `move_cmds_executed=0`,
  `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Line 72 reports controller context:
  `action_scale=0.100000`, `null_action_max_abs=0.000000`, plus soft limits.
- Lines 75 and 87: HOME +5deg shoulder nudge is nonzero/within limits and
  realizes under env-step, with `final_nudge_joint_error_deg=0.109396`,
  `final_target_tcp_error_m=0.001977`, and `target_realized=YES`.
- Lines 103 and 115: early PRE_MOVE +5deg shoulder nudge is nonzero/within
  limits and realizes under env-step, with final nudge error `0.106804deg`,
  final TCP error `0.001955m`, and `target_realized=YES`.
- Lines 131 and 143: high-pose +5deg shoulder nudge is nonzero/within limits and
  realizes under env-step, with final nudge error `0.084780deg`, final TCP error
  `0.001028m`, and `target_realized=YES`.
- Lines 159 and 171: hover-pose +5deg shoulder nudge is nonzero/within limits
  and realizes under env-step, with final nudge error `0.105476deg`, final TCP
  error `0.000884m`, and `target_realized=YES`.
- Line 187 proves the grasp-before-CLOSE/open-gripper nudge is nonzero and
  within soft/analytic limits: `expected_tcp_delta_m=0.024271`.
- Line 199 reports grasp-before-CLOSE env-step failure despite target delivery:
  `set_target_seen=YES`, `best_data_target_attr_diff_rad=0.00000004`,
  `final_target_tcp_error_m=0.023947`,
  `final_nudge_joint_error_deg=5.042476`,
  `tcp_target_reduced=NO`, `nudge_joint_error_reduced=NO`,
  `target_realized=NO`, and `grasped=NO`.
- Line 211 reports direct set+sim-step does not rescue the same grasp pose:
  `set_target_seen=YES`, `max_realized_tcp_delta_m=0.000108`,
  `final_target_tcp_error_m=0.023927`,
  `final_nudge_joint_error_deg=5.044027`, and `target_realized=NO`.
- Line 212 skips after-CLOSE by design because preclose all-realized was false.
- Lines 213-214 aggregate:
  `env_realized_stages=['settled_home', 'early_pre_move', 'high', 'hover']`,
  `env_failed_stages=['grasp_before_close_open']`,
  `direct_rescue_stages=[]`, `home_high_realize_grasp_fails=YES`,
  `broader_command_realization_blocker=NO`,
  `local_grasp_pose_only_blocker=YES`, and `latch_seen=NO`.
- Line 215 reports diagnostic completion success. This is diagnostic success
  only, not attach/transport/release/P7 success.

stderr:

- Lines 1-4 are the known cpufreq/NVML/Fabric warnings. There is no Python
  traceback.

## Interpretation

- D037 is refined: the target-delivery path is not broadly broken. HOME,
  early PRE_MOVE, high, and hover realize the same +5deg shoulder nudge.
- The surviving blocker is local to the grasp-before-CLOSE/open-gripper pose:
  target delivery succeeds, the target is within limits, but env-step and direct
  set+sim-step both fail to reduce the shoulder/TCP error.
- This is not offset-preserve moving evidence. That behavior remains untested
  because the local grasp pose command still does not realize.
- This is not P7 success, not attach physics, not release physics, not attached
  transport, and not constraint integration.

## Next Step

- Stay pre-integration. Inspect why the grasp-before-CLOSE pose is a local
  command-realization dead zone: drive/limit behavior, contact/proximity,
  low-grasp controller state, or whether a slightly different pre-grasp/grasp
  staging pose avoids the failure.
- Do not proceed to transport, release, SurfaceGripper, or RoArm chain constraint
  insertion without explicit approval.

## Verification

- Local `py_compile` and `--help` passed.
- B200 `py_compile` passed.
- B200 final run exit code was 0.
- Local and B200 md5 for the new script matched:
  `ebe8eddafd4c6f35c28e5b79a82511b3`.
