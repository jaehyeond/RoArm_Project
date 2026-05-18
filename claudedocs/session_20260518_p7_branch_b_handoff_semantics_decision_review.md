# Session 2026-05-18 - P7 Branch B handoff semantics decision review

## Scope Guard

- Continued Track A P7/Branch B only.
- Stayed pre-integration.
- Did not train.
- Did not integrate constraints into the RoArm chain.
- Did not insert RoArm-chain constraint prims.
- Did not attach SurfaceGripper.
- Did not go to the transport target.
- Did not execute release or scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.
- Did not use `HANDOFF.md` or `TASKS.md`.

## Boot / Cross-Checks

- Read `CLAUDE.md` Current-State Protocol first.
- Read `START_HERE.md`.
- Read `claudedocs/DECISIONS.md` D034-D036 and D047-D049, then rechecked D015,
  D025-D030 because the user selected Branch B.
- Read `claudedocs/EXPERIMENT_LEDGER.md` rows 49-51 and 65-69.
- Read latest session:
  `claudedocs/session_20260518_p7_branch_b_preclose_coverage_audit.md`.
- Read as needed:
  `claudedocs/session_20260518_p7_branch_b_post_close_latch_boundary.md`,
  `claudedocs/session_20260518_p7_branch_b_handoff_model_probe.md`,
  `claudedocs/session_20260517_p7_branch_b_fixed_constraint_unit.md`,
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_constraint.md`,
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_target_tracking.md`,
  `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_interface_probe.md`,
  and `claudedocs/session_20260517_p7_branch_b_dynamic_anchor_chain_contract.md`.
- `git status --short` had no output before this review.
- Required local md5s matched:
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_admissible_region_probe.py`
    `89ad48b6ebdec076d6f58e330a9131f9`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
    `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

## Decision

Prioritize Branch B authored physics handoff semantics over offset-preserving
pose-write.

The surviving Branch B subpath is not generic SurfaceGripper and not the old
kinematic fixed-joint pose-write. It is:

- dynamic, gravity-disabled anchor;
- USD fixed joint between anchor and sponge;
- closed-loop velocity target tracking;
- explicit TCP-to-anchor offset mapping;
- strict command ordering contract.

This path remains pre-chain only. It is not P7 success, not RoArm chain
integration, and not permission to insert constraints into the RoArm chain.

## Evidence Against Current TCP-Center Pose-Write

- `roarm_rl/roarm_stack_env.py` lines 491-498 call `_update_grasp_attach()` when
  `_grasped.any()`. Lines 1184-1195 latch `_grasped` from distance plus gripper
  threshold. Lines 1216-1236 write sponge root pose to TCP and optionally zero
  velocity. This is a kinematic pose-write boundary, not authored attach physics.
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_posewrite_tcp_v2_b200.out`
  line 41 confirmed strict local scope:
  `constraint_prim_insertion=NO`, `surface_gripper=NO`,
  `attached_transport=NO`, `release_marker=NO`,
  `attach_physics_validated=NO`, `claim_attach_success=NO`.
- Same B200 line 83 killed first stationary post-latch hold:
  `target_error_m=0.015684`, `tcp_step_m=0.016131`,
  `pose_drift_m=0.017552`, `sponge_speed_mps=1.696947`,
  `quat_angle_deg=21.267`, `early_kill=YES`.
- Lines 85-86 reported `post_latch_hold_ok=NO` and
  `ROARM_POST_CLOSE_HANDOFF_MODEL_SUCCESS=NO`.

Interpretation: the current TCP-center pose-write should not be used as a
handoff surface for learning data or chain transport.

## Offset-Preserving Pose-Write Status

- B200 `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_offset_preserve_posewrite_v2_b200.out`
  line 41 confirmed the same strict local scope and explicitly
  `attach_physics_validated=NO`, `release_physics_validated=NO`,
  `claim_attach_success=NO`.
- Lines 83-88 passed sampled stationary post-latch hold steps.
- Line 89 reported `post_latch_hold_steps_done=20`,
  `hold_max_target_error_m=0.000817`, `max_sim_tcp_step_m=0.001947`,
  `hold_max_offset_error_m=0.000001`, `hold_max_speed_mps=0.000869`,
  `posewrite_calls=40`, and `offset_initialized=YES`.
- Line 90 reported `post_latch_hold_ok=YES` but still
  `attach_physics_validated=NO` and `release_physics_validated=NO`.
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_offset_preserve_posewrite_b200.out`
  line 41 again reported no constraints, no SurfaceGripper, no transport target,
  no release, and `claim_attach_success=NO`.
- Lines 88-91 attempted a 4mm `plus_x` micro target, but target error stayed
  about `0.004764m`.
- Lines 92-94 reported `micro_events_done=1`, `micro_motion_ok=NO`, and
  `ROARM_POST_CLOSE_HANDOFF_MICRO_MOTION_SUCCESS=NO`.
- B200 `/tmp/p7_branch_b_roarm_chain_post_latch_target_delivery_v3_b200.out`
  lines 94, 121, and 134 show that the 5deg shoulder nudge target reaches
  target buffers but is not realized before CLOSE, after CLOSE/latch through
  env-step, or after direct set+sim-step.

Interpretation: offset-preserving pose-write is only a local stationary
diagnostic/data-generation candidate. It is not attach physics and is not yet
validated under micro-motion, transport, or release. It should not be used to
generate learning data unless it first passes micro-motion and realism gates.

## SurfaceGripper And Constraint Evidence

- D015 remains active. Quick SurfaceGripper v2/v3 failed to attach:
  `/tmp/roarm_surface_gripper_transport_probe_v2.out` line 143
  `close_detect_step=-1`, line 152 sponge stayed at source with
  `d_xy_pre_release=166.1mm`, line 164 `SURFACE_PROBE_SUCCESS=NO`;
  `/tmp/roarm_surface_gripper_transport_probe_v3_gripperlink.out` line 144
  `close_detect_step=-1`, line 153 same source-stay result, line 165
  `SURFACE_PROBE_SUCCESS=NO`.
- This does not kill physics attach. D015 explicitly says the quick retrofit is
  not proof that physical constraints cannot work.
- D025 fixed-constraint static unit passed close/hold/release only:
  `/tmp/p7_branch_b_fixed_constraint_unit_smoke_v3.out` line 49 created the joint,
  lines 50 and 66 kept hold `rel=0.000000`, line 67 removed the joint, lines
  68-84 showed release/fall, and lines 86-87 reported `FIXED_UNIT_SUCCESS=YES`.
- D026 killed the kinematic-anchor micro-move method:
  `/tmp/p7_branch_b_fixed_constraint_micro_move_smoke.out` lines 59-71 showed
  anchor moved but sponge stayed; lines 104-105 reported `move_ok=NO`,
  `post_move_ok=NO`, and `FIXED_MICRO_MOVE_SUCCESS=NO`.
- D027 rescued motion with a dynamic, gravity-disabled, velocity-driven anchor,
  but the open-loop displacement was about 2x requested, so it was not calibrated.
- D028 closed-loop dynamic-anchor target tracking passed:
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_smoke.out` line 83
  reported `final_anchor_target_error=0.001426` and
  `final_sponge_target_error=0.001426`; lines 102-104 reported all gates YES and
  `FIXED_DYNAMIC_ANCHOR_TARGET_SUCCESS=YES`.
- D029 mock-TCP interface with nonzero TCP-to-anchor offset passed:
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_offset_smoke.out`
  lines 58, 75, and 93 showed `transform_error=0.000000`; line 128 reported
  `max_move_rel=0.000000`, `max_hold_rel=0.000000`,
  `max_final_anchor_target_error=0.001468`,
  `max_final_sponge_target_error=0.001468`; lines 129-130 reported
  `DYNAMIC_ANCHOR_INTERFACE_SUCCESS=YES`.
- D030 mock chain-command contract passed:
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_smoke.out`
  line 42 rejected unsafe command order cases; line 49 accepted `CLOSE`; lines
  59/76/94 accepted three `MOVE` commands with `transform_error=0.000000`; line
  111 accepted `RELEASE` after target reached; lines 129-131 reported
  `contract_negative_ok=YES`, `max_attached_rel=0.000000`,
  `max_final_anchor_target_error=0.001468`,
  `max_final_sponge_target_error=0.001468`, `release_drop=0.338178`, and
  `DYNAMIC_ANCHOR_CHAIN_CONTRACT_SUCCESS=YES`.
- Direct stderr hygiene recheck for D028-D030 B200 logs found only known
  cpufreq/NVML warnings on lines 1-3 and no Python traceback.

Interpretation: B is the stronger primary research path, but only the isolated
dynamic-anchor physics handoff semantics are currently validated. The missing
bridge is actual RoArm-chain signal generation and contact-safe handoff timing,
not constraint coupling in isolation.

## Minimum Pass Criteria Before Any Learning

No learning, dataset generation, chain transport, or release claim should start
until a candidate handoff model passes all of the following:

1. Stationary hold:
   post-latch or post-close attached hold remains stable under target-error,
   sim-step, relative-transform, object-speed, angular-speed, and upright gates.
2. Micro-motion hold:
   a small local command is actually realized by both carrier and object, with
   low relative error and no hidden pose-write artifact.
3. Relative TCP-object transform preservation:
   the intended TCP-to-object or TCP-to-anchor offset remains bounded during
   hold and micro-motion, not only at the first CLOSE step.
4. Upright preservation:
   object uprightness remains high during attached hold, micro-motion, and before
   release.
5. Release feasibility:
   detach/open/removal produces physical separation and settling without relying
   on a kinematic teleport as the success mechanism.
6. Dataset realism:
   the mechanism must be physics-authored or otherwise explicitly justified as
   realistic; logs with `kinematic_env_latch_only=YES`,
   `attach_physics_validated=NO`, or `claim_attach_success=NO` are not sufficient
   for learning data.

## Next Scoped Diagnostic Candidate

Do not integrate constraints into the RoArm chain yet.

The next narrow B diagnostic should be a pre-integration signal-compatibility
review/probe that maps the already validated dynamic-anchor contract against the
current RoArm command stream and pre-close selector evidence without executing
CLOSE->MOVE transport.

## Static Contract Compatibility Table

| Contract state | Existing B evidence | Existing RoArm evidence | Current verdict |
|---|---|---|---|
| `CLOSE` command ordering | D030 dynamic-anchor contract rejects unsafe order and accepts `CLOSE`: B200 `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_smoke.out` lines 42 and 49. | Passive contact close reaches the close marker without attach physics: `/tmp/p7_branch_b_roarm_chain_passive_contact_close_timing_probe_b200.out` lines 81-84. | Compatible as a marker only. Do not treat `_grasped` as attach physics. |
| Pre-close command stream | D030 expects small ordered command targets before attached state. | D031-D032 show chain-side stream works only with explicit resampling; B200 command stream line 81 has `max_pre_move_tcp_step_m=0.009525`, `max_move_tcp_step_m=0.007691`, and raw gaps still fail. D033 line 86 shows realized-TCP gating is required because `one_step_target_ok=NO`. | Compatible only with conservative resampling plus realized-TCP gates. Raw waypoints and one-step assumptions are invalid. |
| Stationary attached hold | D028-D030 dynamic-anchor units hold `rel=0.000000` in isolation; D030 line 129 reports `max_attached_rel=0.000000`. | Current TCP-center env pose-write fails first hold: `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_posewrite_tcp_v2_b200.out` line 83. Offset-preserve passes stationary hold but logs `attach_physics_validated=NO`: `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_offset_preserve_posewrite_v2_b200.out` lines 89-90. | B mechanism passes in isolation; current RoArm env pose-write does not. Offset-preserve is useful only as a stationary diagnostic, not as physics attach. |
| Local micro-motion | D029/D030 dynamic-anchor interface moves through three mock-TCP waypoints with max final target error `0.001468m` and `rel=0`. | RoArm post-CLOSE micro-motion did not realize 4mm/8mm targets; target-delivery instrumentation shows the 5deg nudge reaches buffers but fails at the grasp pose before and after CLOSE. D037 lines 1564-1581 and B200 target-delivery lines 94/121/134 confirm this. | Blocked. Before any B chain transition, the real RoArm local handoff signal must realize tiny safe motions. |
| Contact-safe pre-close geometry | B contract itself is geometry-agnostic. | D043-D049 constrain admissible RoArm pre-close geometry: below-top inside-footprint is banned; side-edge below-top needs realized outside-AABB margin; conservative wrapper requires 2mm side margin and depth no deeper than about -3mm. | Any future local B signal probe must use above/top-tangent or conservative side-edge geometry. Do not use below-top inside-footprint targets. |
| Release feasibility | D030 line 111 accepts release only after target reached, and lines 129-131 report `release_drop=0.338178` with success. | RoArm release physics is not validated in this B handoff context; current review did not execute release. | Keep as isolated B evidence only. No release run until local hold and micro-motion pass. |
| Dataset realism | Dynamic-anchor units are physics-authored isolated probes. | Logs with `kinematic_env_latch_only=YES`, `attach_physics_validated=NO`, or `claim_attach_success=NO` are not realistic data sources. | No dataset generation yet. |

Static review result:

- No new contradiction justifies a broad diagnostic matrix.
- The next blocker is not B constraint coupling in isolation. It is actual RoArm
  local signal generation near CLOSE/contact.
- The next runnable diagnostic, if approved later, must be narrower than transport:
  local post-CLOSE or just-before-CLOSE micro signal only, using a safe
  above/tangent or conservative side-edge geometry, with dynamic-anchor-style
  relative-transform and upright gates.

- Input evidence:
  - D030 dynamic-anchor command contract requirements;
  - D031-D033 RoArm chain-side command stream, resampling, and realized-TCP
    gating requirements;
  - D034-D036 post-CLOSE handoff failures;
  - D043-D049 valid pre-close geometry constraints.
- Required output:
  - a static contract table that says which real RoArm signals can satisfy
    `CLOSE`, local attached hold, local micro-motion, and future `RELEASE`;
  - explicit blockers where the real articulation/contact evidence contradicts
    the mock dynamic-anchor assumptions;
  - no Isaac run unless the static table exposes one narrow contradiction that
    cannot be resolved from existing logs.

If a run is later approved, the first runnable diagnostic should not go to the
transport target and should not execute release. It should only test whether a
real RoArm local post-CLOSE signal can drive a dynamic-anchor-style carrier
through a tiny local micro-motion while preserving the relative transform and
uprightness. That would still be pre-integration and not a chain success claim.
