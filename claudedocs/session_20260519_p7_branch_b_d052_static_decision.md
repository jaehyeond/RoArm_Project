# Session 2026-05-19 - P7 Branch B D052 static decision review

## Scope

- Continued Track A P7/Branch B only.
- Static evidence review and decision narrowing only.
- Did not train.
- Did not run Isaac.
- Did not run post_close_marker+side_edge.
- Did not run another matrix.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute attached transport, go to a transport target, run release, or add
  scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not edit env/train/chain defaults.
- Did not use `HANDOFF.md` or `TASKS.md`.

## Verified Inputs

Protocol / dashboard:

- `CLAUDE.md` lines 5-31 require the Current-State Protocol: read the rolling
  state docs, run `git status --short`, and verify metrics from source logs before
  citing them.
- `CLAUDE.md` lines 51-55 require critical/skeptical cross-verification and forbid
  treating `HANDOFF.md` or `TASKS.md` as current state.
- `START_HERE.md` lines 11-18 keep the project split into Track A P7/Branch B and
  Track B CoRL, and repeat the HANDOFF/TASKS ban.

Dirty / md5:

- `git status --short` showed existing dirty state:
  `START_HERE.md`, `claudedocs/DECISIONS.md`,
  `claudedocs/EXPERIMENT_LEDGER.md`, untracked
  `claudedocs/session_20260519_p7_branch_b_close_near_local_signal.md`, and
  untracked `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`.
  Nothing was reverted.
- Required local md5s matched:
  - `sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`
    `2b63df20972ad1e923f24e05c2810957`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_admissible_region_probe.py`
    `89ad48b6ebdec076d6f58e330a9131f9`
  - `sim_scripts/p7_branch_b_roarm_chain_preclose_candidate_selector_probe.py`
    `aa24ef00acbb9d8cd0aeee061b08f85f`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`

## Dynamic-Anchor Branch B Contract

The isolated Branch B mechanics are alive, but remain pre-chain:

- Dynamic-anchor target tracking B200
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_smoke.out` line 83:
  final anchor/sponge target errors both `0.001426`; lines 102-104 report all gates
  YES and `FIXED_DYNAMIC_ANCHOR_TARGET_SUCCESS=YES`.
- Mock TCP offset interface B200
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_offset_smoke.out`
  lines 58, 75, and 93 report `transform_error=0.000000`; lines 128-130 report
  `max_move_rel=0.000000`, max final target error `0.001468`, release drop
  `0.338178`, and `DYNAMIC_ANCHOR_INTERFACE_SUCCESS=YES`.
- Chain-command contract B200
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_smoke.out`
  line 42 rejects unsafe command order, line 49 accepts `CLOSE`, lines 59/76/94
  accept MOVE targets with zero transform error, line 111 accepts release after
  target-reached state, and lines 129-131 report all gates YES with
  `DYNAMIC_ANCHOR_CHAIN_CONTRACT_SUCCESS=YES`.

Interpretation:

- D030 is still correct: isolated constraint coupling, target tracking,
  TCP-to-anchor offset mapping, and command ordering are not the current blocker.
- The unresolved boundary is actual RoArm chain signal generation and future
  attach/constraint handoff, not the isolated dynamic-anchor contract itself.
- This is not chain integration evidence and not permission to run transport or
  release.

## RoArm Handoff Evidence Boundary

Old failures still matter:

- Current TCP-center env pose-write fails first post-latch hold:
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_posewrite_tcp_v2_b200.out`
  line 83 has `target_error_m=0.015684`, `tcp_step_m=0.016131`,
  `pose_drift_m=0.017552`, `sponge_speed_mps=1.696947`, and
  `quat_angle_deg=21.267`; lines 85-86 report success NO.
- Offset-preserving pose-write passes only stationary hold:
  `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_offset_preserve_posewrite_v2_b200.out`
  lines 89-91 report `post_latch_hold_ok=YES` and success YES, but line 90 still
  has `attach_physics_validated=NO` and `release_physics_validated=NO`.
- Offset-preserving micro-motion did not realize the first 4mm target:
  `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_offset_preserve_posewrite_b200.out`
  lines 88-91 keep target error around `0.004764`; lines 92-94 report
  `micro_motion_ok=NO` and success NO.
- The +5deg nudge is not broadly impossible, but it fails locally at the
  grasp-before-CLOSE/open-gripper pose:
  `/tmp/p7_branch_b_roarm_chain_approach_target_delivery_v2_b200.out` lines
  87/115/143/171 realize HOME/early/high/hover, while lines 199/211 fail the
  grasp-before-CLOSE/open target and lines 213-214 set
  `local_grasp_pose_only_blocker=YES`.

Interpretation:

- Offset-preserving pose-write is not the primary Branch B path. It is only a local
  stationary diagnostic result.
- B remains primary because the dynamic-anchor contract survives in isolation and
  the RoArm has now shown a top-tangent local signal carrier, but no attach surface
  has been validated.

## Pre-Close Geometry Boundary

The existing admissible-region evidence is internally consistent:

- Conservative wrapper B200
  `/tmp/p7_branch_b_roarm_chain_preclose_admissible_region_b200.out` line 3 states
  the non-deployed rule: `min_side_margin_m=0.002000`, max below-depth `-0.003000`,
  reject below-top inside-footprint, reject zero-margin boundary, require
  realized/final outside-AABB for below-top side-edge, and keep far-sponge below-top
  as no-contact control.
- Lines 4-12 classify all eight compact cases as expected, with
  `expected_matches=8/8`, `attach_calls_all_zero=YES`, and diagnostic success YES.
- Depth sweep at 2mm side margin:
  - `neg3p0` line 1055 is exact-clean (`final_target_tcp_error_m=0.002409`,
    exact YES, top clamp NO, mechanically valid YES, clean YES) and lines 1058-1059
    report success YES.
  - `neg4p0` line 1055 stays outside AABB and mechanically valid, but exact fails
    (`0.003346`, exact NO, clean NO), and lines 1058-1059 report success NO.

Interpretation:

- Side-edge below-top remains admissible only as a pre-close pose/hold geometry
  class inside this conservative evidence boundary.
- The wrapper is a conservative explanation of prior evidence, not a deployed chain
  policy and not a license to accept every observed clean side-edge case.
- No file/log contradiction was found, so a new pre-close matrix is not justified.

## D050 / D051 / D052 Decision

What D050/D051 establish:

- Default top-tangent B200
  `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_default_b200.out` lines
  41-43 confirm strict virtual-carrier/signal-only scope, no MOVE commands,
  `geometry=top_tangent`, `signal_stage=just_before_close`, and
  `micro_delta_m=0.004000`.
- Lines 279/285/291/297/300 show safe clearance, top-tangent signal pose,
  stationary hold, 4mm `micro_plus_x`, and return all reached.
- Lines 301-303 report `attach_calls=0`, `posewrite_calls=0`,
  `transport_target=NO`, `release_marker=NO`, all gates YES, and success YES.
- Post-close-marker top-tangent B200
  `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_post_close_marker_b200.out`
  lines 274-276 show the close-marker-only/no-posewrite step reached with
  `attach_calls=0`, `posewrite_calls=0`, and `claim_attach_success=NO`.
- Lines 299/302 show 4mm `micro_plus_x` and return reached; lines 303-305 report
  all gates YES and success YES.

What D052 changes:

- D052 changes the geometry generalization. Top-tangent 4mm local signal evidence
  cannot be generalized to conservative side-edge 4mm local micro-motion.
- Side-edge B200
  `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_side_edge_b200.out` lines
  41-43 confirm the same strict scope with `geometry=side_edge`,
  `signal_stage=just_before_close`, and `micro_delta_m=0.004000`.
- Lines 279/285/291 show side-edge clearance, side-edge signal pose, and stationary
  hold reached.
- Lines 292-295 show the 4mm `micro_plus_x` remained around
  `0.005342-0.005379m` target error through 60 steps.
- Line 296 reports `reached=NO`, `final_target_error_m=0.005342`,
  `set_target_seen=YES`, and `early_kill=YES`.
- Lines 297-299 report `attach_calls=0`, `posewrite_calls=0`,
  `transport_target=NO`, `release_marker=NO`,
  `micro_motion_realized_ok=NO`, `target_error_ok=NO`, and success NO.

What D052 does not change:

- It does not invalidate top-tangent D050/D051.
- It does not invalidate side-edge as a conservative pre-close pose/hold geometry
  class.
- It does not validate `_grasped` attach physics, dynamic/fixed constraint
  insertion, SurfaceGripper, attached transport, transport target, or release.
- It does not justify post_close_marker+side_edge, another matrix, transport,
  release, SurfaceGripper, or constraint integration.

## Static Decision

Current decision:

- Treat top-tangent D050/D051 as the only accepted RoArm CLOSE-near 4mm local
  signal prerequisite for future Branch B handoff design.
- Exclude conservative side-edge as a 4mm local signal carrier for now.
- Keep conservative side-edge only as pre-close pose/hold geometry evidence, not as
  a local micro-motion carrier.
- Keep B primary over offset-preserving pose-write because B has a living isolated
  physics-authored contract and top-tangent RoArm signal evidence, while
  offset-preserving pose-write is kinematic and stationary-only.

This is a narrowing decision, not a run approval.

## When To Move On

Move from static decision to the next runnable diagnostic only when all are true:

1. The intended carrier is top-tangent, not side-edge.
2. The scope is still local handoff only: no transport target, no release, no
   SurfaceGripper, no training, no scripted release variant.
3. The diagnostic has a falsifiable attach/constraint handoff gate before any MOVE:
   stationary hold, tiny local micro-motion, bounded TCP/anchor offset, bounded
   sponge drift/speed, upright preservation, no hidden kinematic pose-write
   artifact, and explicit `attach_physics_validated` semantics.
4. The run is separately approved, because it would leave pure static review.
5. It does not use the current env TCP-center pose-write as success evidence.

Enough evidence to proceed later would be:

- A top-tangent local handoff diagnostic with actual authored attach/constraint
  semantics, still before transport/release, where:
  - close/handoff succeeds under a stated gate;
  - stationary hold succeeds;
  - 4mm local micro-motion succeeds and returns;
  - relative TCP/anchor/object transform stays bounded;
  - `attach_calls`/constraint semantics are explicit and not hidden pose-write;
  - `transport_target=NO` and `release_marker=NO`;
  - stderr/process hygiene is clean.

Only after that should the next boundary be discussed. Transport/release remains a
later gate, not the next step.

## Narrowest Next Candidate

No Isaac run is recommended right now.

The next candidate, if the user explicitly approves a runnable diagnostic later, is
a top-tangent-only local attach/constraint handoff smoke with no MOVE transport and
no release. It should test only the missing boundary between:

- D030 isolated dynamic-anchor command contract, and
- D050/D051 real RoArm top-tangent CLOSE-near local signal.

It should not include side-edge, transport, release, SurfaceGripper, curriculum
training, scalar tuning, diagnostic gate tuning, or a pre-close matrix.
