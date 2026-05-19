# Session 2026-05-19 - P7 Branch B close-near local signal probe

## Scope

- Continued Track A P7/Branch B only.
- Did not train.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper.
- Did not execute attached transport, go to a transport target, run release, or
  add scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not tune diagnostic gates.
- Did not start structured A curriculum long-training.

## Boot / Evidence Hygiene

- Read `CLAUDE.md` Current-State Protocol first.
- Read `START_HERE.md`.
- Read `claudedocs/DECISIONS.md` D015, D025-D030, D031-D038, D043-D049.
- Read `claudedocs/EXPERIMENT_LEDGER.md` rows 27, 36, 38, 41, 42, 44-52, 65-69.
- Read latest decision review
  `claudedocs/session_20260518_p7_branch_b_handoff_semantics_decision_review.md`
  lines 198-218.
- Rechecked required B200 `/tmp` log lines before claiming state, including
  isolated dynamic-anchor target/interface/contract, handoff model, handoff
  micro-motion, target-delivery, approach target-delivery, admissible wrapper,
  and side-edge depth sweep logs.
- `git status --short` before patching showed only:
  `?? sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`.
- Required local md5s before patching matched the prompt, including the new
  script md5 `45dd29f7986df29336b4c07a1c5dd5c5`.

## Static Review and Patch

Reviewed requested sections of
`sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`:

- lines 1-9 scope;
- lines 215-250 args and conservative side-edge guards;
- lines 277-306 printed no-overclaim scope/gates;
- lines 410-516 per-event realization and artifact metrics;
- lines 551-646 local events and pass/fail gates.

Two narrow bugs were found and patched only in this new script:

1. `--reassert_sponge_z_m` was used for target geometry but the actual sponge
   root pose write still used the env constant. The patch now writes the
   argument value into the sponge root pose.
2. `--signal_stage post_close_marker` ran a close marker but did not include the
   close-marker result in the success gate before local signal execution. The
   patch added `close_marker_ok`, includes it in success, and skips local signal
   if the close marker fails.

After patch:

- Local `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py`
  passed.
- Local `python sim_scripts/p7_branch_b_roarm_chain_close_near_local_signal_probe.py --help`
  passed.
- Patched md5:
  `2b63df20972ad1e923f24e05c2810957`.

## B200 Sync / Static Verification

- Synced only the patched script to:
  `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/`.
- Remote md5 matched:
  `2b63df20972ad1e923f24e05c2810957`.
- Remote `py_compile` passed.
- Remote `--help` passed.

## B200 Run

Command scope:

- Ran only the default close-near local signal diagnostic.
- Default arguments mean:
  `geometry=top_tangent`, `signal_stage=just_before_close`,
  `micro_delta_m=0.004000`.
- Used B200 IsaacLab env with per-run:
  `OMNI_KIT_ACCEPT_EULA=YES`,
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`,
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

Logs:

- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_default_b200.out`
- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_default_b200.err`

The command printed `exit_code:0`.

## B200 Evidence

Stdout:

- Line 40 starts the probe.
- Line 41 confirms strict scope:
  `close_near_local_signal_only=YES`,
  `virtual_dynamic_anchor_style_carrier=YES`,
  `virtual_carrier_only=YES`, no constraint prim insertion, no fixed/dynamic
  constraint integration, no SurfaceGripper, no attached transport, no transport
  target, no release marker, no scripted release variant, no P7 training/tuning,
  no diagnostic gate tuning, no env/train/chain default edits, and
  `claim_attach_success=NO`.
- Line 42 confirms gates and defaults:
  `target_error_gate_m=0.003000`, `max_tcp_step_m=0.010000`,
  `home_fk_gate_m=0.003000`, `geometry=top_tangent`,
  `signal_stage=just_before_close`, `micro_delta_m=0.004000`.
- Line 43 confirms no MOVE execution:
  `source_events_total=44`, `pre_move_cmds=38`, `move_cmds_executed=0`,
  `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Lines 44-46 confirm IK convergence for the clearance/final/micro targets.
- Line 74 confirms HOME FK and sponge baseline:
  `home_fk_error_m=0.001894`, `home_fk_ok=YES`,
  settled sponge z `0.023500`, upright z `1.000000`.
- Lines 279 and 285 reach the safe clearance and top-tangent signal pose with
  final target errors `0.002505m` and `0.002050m`.
- Line 291 passes the stationary hold:
  final target error `0.000922m`, set target seen, no early kill.
- Line 297 reaches the 4mm `micro_plus_x` target:
  final target error `0.002267m`, set target seen, no early kill.
- Line 300 reaches `micro_return_x`:
  final target error `0.001351m`, set target seen, no early kill.
- Line 301 aggregate:
  `prep_events_done=38`, `prep_events_planned=38`,
  `max_final_target_error_m=0.002505`,
  `max_tcp_step_m=0.003353`,
  `max_tcp_anchor_offset_error_m=0.00000000`,
  `max_sponge_drift_m=0.000000`,
  `max_sponge_speed_mps=0.000540`,
  `max_quat_angle_deg=0.000`,
  `min_upright_z=1.000000`,
  `attach_calls=0`, `posewrite_calls=0`,
  `virtual_carrier_only=YES`, `transport_target=NO`, `release_marker=NO`.
- Line 302 reports all intended gates YES:
  `home_fk_ok`, `prep_ok`, `close_marker_ok`, `stationary_hold_ok`,
  `micro_motion_realized_ok`, `relative_tcp_anchor_transform_ok`,
  `upright_preservation_ok`, `no_hidden_kinematic_posewrite_artifact`,
  `no_attach_release_transport_overclaim`, `target_error_ok`, `sim_step_ok`,
  with `attach_physics_validated=NO`, `release_physics_validated=NO`, and
  `claim_attach_success=NO`.
- Line 303 reports:
  `ROARM_CLOSE_NEAR_LOCAL_SIGNAL_SUCCESS=YES`.

Stderr:

- Lines 1-4 contain the known cpufreq/NVML/Fabric messages seen in other B200
  Isaac diagnostics.
- A stdout/stderr scan found no Python traceback or exception.

Process hygiene:

- Post-run process check found no matching P7/Isaac/training process.

## Interpretation

- This is the first positive evidence that the real RoArm can generate a
  CLOSE-near 4mm-class local TCP micro signal under admissible top-tangent
  geometry.
- It directly narrows the prior blocker: the problem is no longer "no local
  signal near CLOSE" in general.
- This result does not validate dynamic-anchor constraint integration. The
  dynamic-anchor-style carrier is virtual, the TCP-to-anchor offset is algebraic,
  and no USD fixed joint or dynamic anchor was inserted.
- It does not validate attach physics, attached transport, release physics,
  SurfaceGripper, or chain-ready constraints.
- It does not justify a new pre-close matrix. The run used the already-approved
  top-tangent admissible geometry and a single default diagnostic.

## Next Step

- Stay pre-integration.
- Treat this as a narrow signal-only PASS, not P7 success.
- A follow-up, if explicitly approved, should remain in the same script and same
  no-overclaim envelope: either `post_close_marker` or conservative side-edge
  geometry. Do not proceed to transport/release or constraint integration.

## Approved Follow-Up: `post_close_marker`

After explicit approval, ran the same diagnostic script with only:

```bash
--signal_stage post_close_marker
```

Scope stayed unchanged:

- no training;
- no constraint prim insertion;
- no fixed/dynamic constraint integration;
- no SurfaceGripper;
- no attached transport;
- no transport target;
- no release marker;
- no env/train/chain default edits.

Logs:

- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_post_close_marker_b200.out`
- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_post_close_marker_b200.err`

The command printed `exit_code:0`.

Stdout:

- Line 41 confirms the same strict signal-only scope and
  `claim_attach_success=NO`.
- Line 42 confirms `geometry=top_tangent`,
  `signal_stage=post_close_marker`, and `micro_delta_m=0.004000`.
- Line 43 confirms `move_cmds_executed=0`, raw planner gap `0.211271`, and
  `raw_gap_ok=NO`.
- Lines 274-276 show the close-marker-only/no-posewrite step reached:
  final target error `0.001131`, `attach_calls=0`, `posewrite_calls=0`, and
  `claim_attach_success=NO`.
- Lines 282, 288, 294, 299, and 302 show safe clearance, top-tangent signal
  pose, stationary hold, `micro_plus_x`, and `micro_return_x` all reached.
- Line 303 aggregate:
  `prep_events_done=38`, `prep_events_planned=38`,
  `max_final_target_error_m=0.002576`,
  `max_tcp_step_m=0.003432`,
  `max_tcp_anchor_offset_error_m=0.00000000`,
  `max_sponge_drift_m=0.000000`,
  `max_sponge_speed_mps=0.000341`,
  `max_quat_angle_deg=0.000`,
  `min_upright_z=1.000000`,
  `attach_calls=0`, `posewrite_calls=0`,
  `virtual_carrier_only=YES`, `transport_target=NO`, `release_marker=NO`.
- Lines 304-305 report all intended gates YES and
  `ROARM_CLOSE_NEAR_LOCAL_SIGNAL_SUCCESS=YES`.

Stderr:

- Lines 1-4 contain only the known cpufreq/NVML/Fabric messages seen in other
  B200 Isaac diagnostics.
- No Python traceback/exception was found in the stdout/stderr scan.

Process hygiene:

- Post-run process check found no matching P7/Isaac/training process.

Interpretation:

- D050 is now strengthened: top-tangent 4mm-class local TCP signal is available
  both just before CLOSE and after a close-marker-only/no-posewrite step.
- This is still not `_grasped` attach physics, not dynamic-anchor constraint
  integration, not SurfaceGripper validation, not attached transport, not a
  transport target, and not release validation.
- No new pre-close matrix is justified by this result.

## Approved Follow-Up: `side_edge`

After explicit approval in the new session, ran the same diagnostic script with
only:

```bash
--geometry side_edge
```

Scope stayed unchanged:

- no training;
- no constraint prim insertion;
- no fixed/dynamic constraint integration;
- no SurfaceGripper;
- no attached transport;
- no transport target;
- no release marker;
- no env/train/chain default edits;
- no diagnostic gate tuning.

Logs:

- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_side_edge_b200.out`
- B200 `/tmp/p7_branch_b_roarm_chain_close_near_local_signal_side_edge_b200.err`

The command printed `exit_code:0`.

Stdout:

- Lines 41-43 confirm the same strict signal-only scope,
  `geometry=side_edge`, `signal_stage=just_before_close`,
  `micro_delta_m=0.004000`, `move_cmds_executed=0`, raw planner gap
  `0.211271`, and `raw_gap_ok=NO`.
- Lines 44-46 show IK convergence for the side-edge clearance/final/micro
  targets.
- Line 279 shows side-edge clearance reached with final target error
  `0.002567m`.
- Line 285 shows the conservative side-edge signal pose reached with final
  target error `0.002879m`, target below top by about `-0.003005m`, and outside
  the sponge AABB.
- Line 291 shows the 5-step stationary hold reached with final target error
  `0.002875m`.
- Lines 292-295 show the 4mm `micro_plus_x` target did not converge. It remained
  around `0.005342-0.005379m` target error through 60 steps.
- Line 296 reports `micro_plus_x` `reached=NO`, steps `60`,
  `final_target_error_m=0.005342`, `set_target_seen=YES`, and
  `early_kill=YES`.
- Line 297 aggregate:
  `prep_events_done=38`, `prep_events_planned=38`,
  `max_final_target_error_m=0.005342`,
  `max_tcp_step_m=0.003899`,
  `max_tcp_anchor_offset_error_m=0.00000000`,
  `max_sponge_drift_m=0.000040`,
  `max_sponge_speed_mps=0.013705`,
  `max_quat_angle_deg=0.043`,
  `min_upright_z=0.999999`,
  `attach_calls=0`, `posewrite_calls=0`,
  `virtual_carrier_only=YES`, `transport_target=NO`, `release_marker=NO`.
- Lines 298-299 report `micro_motion_realized_ok=NO`, `target_error_ok=NO`, and
  `ROARM_CLOSE_NEAR_LOCAL_SIGNAL_SUCCESS=NO`.

Stderr:

- Lines 1-4 contain only the known cpufreq/NVML/Fabric messages seen in other
  B200 Isaac diagnostics.
- No Python traceback/exception was found in the stdout/stderr scan.

Process hygiene:

- Post-run process check found no matching P7/Isaac/training process.

Interpretation:

- Top-tangent D050/D051 should not be generalized to conservative side-edge
  4mm local micro-motion.
- The conservative side-edge signal pose and short stationary hold are reachable,
  but the 4mm local `micro_plus_x` signal is not realized under this single-point
  diagnostic.
- This is still not `_grasped` attach physics, not dynamic-anchor constraint
  integration, not SurfaceGripper validation, not attached transport, not a
  transport target, and not release validation.
- No new pre-close matrix is justified by this result. The result is a narrow
  single-point failure in the already-approved side-edge diagnostic envelope.
