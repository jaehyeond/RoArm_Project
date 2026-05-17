# Session 2026-05-18 - P7 Branch B post-close handoff micro-motion probe

## Scope

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper to the RoArm chain.
- Did not execute attached transport, release, or scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not run P7 training.
- Added only a diagnostic-local post-CLOSE micro-motion probe.

## Boot Verification

- Read `CLAUDE.md`, `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D036,
  latest Branch B ledger rows, and the requested Branch B session docs.
- `git status --short` before coding had no output.
- Required local md5s before coding matched the prompt:
  - `sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`
    `938a94b3b856dcc5a48527991a87c1e9`
  - `sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`
    `58b628682a536535d3d9a6790c51974d`
  - `sim_scripts/p7_branch_b_roarm_chain_passive_contact_close_timing_probe.py`
    `6cb899ca124ff588fcc011d2805fa605`
  - `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`
    `339bdfd2ced7cf05b4ce87d2cd92128a`
  - `sim_scripts/p7_branch_b_roarm_chain_command_stream_probe.py`
    `d9a07b43bed44f6061144234d7f6ec36`
  - `sim_scripts/p7_branch_b_roarm_chain_timing_resample_probe.py`
    `fe2b227d2a111bf1c7acfe82e8f43133`
  - `sim_scripts/p7_branch_b_roarm_chain_contract_dryrun_probe.py`
    `88b4b8b33cd7aeecd6a18f78bf144283`
  - `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_probe.py`
    `6af24284baef540f190b762e5da164a5`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
  - `launch_chain_topdown.sh` `b34ef3853ac993a1e2adbaddb420adab`
  - `launch_p6v17_transport_release.sh` `2acd462042d0997610fca25ff7a41e21`

## Prior Evidence Re-Checked

- `roarm_rl/roarm_stack_env.py` lines 491-498 set joint targets and call
  `_update_grasp_attach()` only when `_grasped.any()`.
- `roarm_rl/roarm_stack_env.py` lines 1184-1195 latch `_grasped` from
  distance plus gripper threshold.
- `roarm_rl/roarm_stack_env.py` lines 1216-1236 show current kinematic attach:
  sponge root pose is written to TCP, quaternion is preserved or reset by config,
  and velocity is zeroed by default.
- B200 `posewrite_tcp` handoff-model v2 line 83 reproduced the D034 failure:
  `target_error_m=0.015684`, `tcp_step_m=0.016131`,
  `pose_drift_m=0.017552`, `sponge_speed_mps=1.696947`,
  `quat_angle_deg=21.267`.
- B200 `offset_preserve_posewrite` v2 lines 83-91 passed stationary hold only,
  with line 90 still reporting `attach_physics_validated=NO` and
  `release_physics_validated=NO`.

## Script

- `sim_scripts/p7_branch_b_roarm_chain_handoff_micro_motion_probe.py`
- md5 `a7ed4387e0ab1ce5b95de08f59c2eb52`

Design:

- Imports the existing conservative command stream builder from
  `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`.
- Places the sponge at nominal pick after reset and executes only
  `PRE_MOVE* -> CLOSE`.
- Uses realized-TCP gated execution; no one-step command assumption.
- Monkey-patches `_update_grasp_attach` inside this diagnostic only; env/train/
  chain defaults remain untouched.
- Compares:
  - `posewrite_tcp`: current env baseline, expected to fail stationary hold.
  - `marker_only`: no pose-write negative control.
  - `offset_preserve_posewrite`: latch-time TCP-to-sponge offset candidate.
- After a short 5-step stationary hold, attempts only tiny TCP perturbations
  around the grasp pose: `+x`, return, `+z`, return.
- Does not go to the transport target and does not release.

Falsifiable gates:

- HOME sim TCP vs analytic FK `<= 0.003m`
- all executed PRE_MOVE/CLOSE events reach target under gated execution
- stationary hold target error `<= 0.003m`
- per-sim TCP step `<= 0.010m`
- stationary hold speed `<= 0.050m/s`
- micro offset error `<= 0.002m`
- micro speed `<= 0.500m/s`
- quaternion change `<= 5deg`
- minimum sponge upright z axis `>= 0.90`
- no NaN and no episode truncation

## Runs

Local:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_handoff_micro_motion_probe.py`
  passed.
- `python sim_scripts/p7_branch_b_roarm_chain_handoff_micro_motion_probe.py --help`
  passed.

B200:

- Synced only the new script to
  `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/`.
- Remote md5 matched local:
  `a7ed4387e0ab1ce5b95de08f59c2eb52`.
- Remote `py_compile` passed.
- Runs used `OMNI_KIT_ACCEPT_EULA=YES`,
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`, and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

Logs:

- B200 `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_posewrite_tcp_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_marker_only_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_offset_preserve_posewrite_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_offset_preserve_posewrite_d8mm_b200.{out,err}`

## Evidence

Common B200 scope:

- Line 41 in each stdout confirms no constraint prim insertion, no fixed/dynamic
  integration, no SurfaceGripper, no attached transport, no release marker, no P7
  training, no default edits, `transport_target=NO`, `micro_motion_not_transport=YES`,
  and `claim_attach_success=NO`.
- Line 43 in each stdout confirms the conservative source stream was truncated
  before MOVE: source `events_total=44`, executed events `39`,
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.

Baseline current pose-write:

- `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_posewrite_tcp_b200.out`
  line 83 failed first stationary hold with `target_error_m=0.015684`,
  `tcp_step_m=0.016131`, `pose_drift_m=0.017552`,
  `sponge_speed_mps=1.696947`, `quat_angle_deg=21.267`.
- Line 84 skipped micro-motion because stationary hold was not OK.
- Lines 85-87 reported `post_latch_hold_ok=NO`, `micro_motion_ok=NO`, and
  `ROARM_POST_CLOSE_HANDOFF_MICRO_MOTION_SUCCESS=NO`.

Marker-only, 4mm perturbation:

- `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_marker_only_b200.out`
  lines 83-87 passed the short stationary hold with negligible drift.
- Lines 88-91 attempted `plus_x` with `micro_delta_m=0.004`, but target error
  stayed around `0.004764-0.004765` through 60 steps, while TCP step remained
  tiny and the sponge stayed essentially stationary.
- Lines 92-94 reported only one of four micro events done,
  `micro_max_target_error_m=0.004764`, `target_error_ok=NO`,
  `micro_motion_ok=NO`, and success `NO`.
- This remains a negative control and is not attach evidence.

Offset-preserving candidate, 4mm perturbation:

- `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_offset_preserve_posewrite_b200.out`
  lines 83-87 passed the short stationary hold with negligible drift.
- Lines 88-91 attempted the same 4mm `plus_x`, but target error stayed
  `0.004764-0.004765`; the micro target was not reached.
- Lines 92-94 reported `micro_events_done=1`, `micro_events_planned=4`,
  `micro_motion_ok=NO`, and success `NO`.

Offset-preserving candidate, 8mm perturbation cross-check:

- `/tmp/p7_branch_b_roarm_chain_handoff_micro_motion_probe_offset_preserve_posewrite_d8mm_b200.out`
  line 42 changed only `micro_delta_m=0.008000`.
- Lines 83-87 again passed the short stationary hold.
- Lines 88-91 attempted 8mm `plus_x`, but target error stayed
  `0.008699-0.008703`; the micro target was still not reached.
- Lines 92-94 reported `micro_events_done=1`, `micro_events_planned=4`,
  `target_error_ok=NO`, `micro_motion_ok=NO`, and success `NO`.

B200 stderr:

- The three 4mm stderr files and the 8mm stderr file had only the known
  cpufreq/NVML/Fabric warnings on lines 1-4 and no Python traceback.

## Interpretation

- The current TCP-center pose-write baseline is still killed before any
  micro-motion, reproducing D034/D036.
- Marker-only and offset-preserve both survive the short stationary hold, but
  neither 4mm nor 8mm post-close `plus_x` perturbation reached the requested TCP
  target in this diagnostic.
- Therefore this run does **not** validate offset-preserving behavior under MOVE,
  object attachment physics, attached transport, release physics, SurfaceGripper,
  or constraint insertion.
- The immediate new blocker is lower-level: this diagnostic's post-latch
  micro-target execution did not produce realized TCP motion. The next work
  should instrument or redesign the post-latch micro-command executor before
  interpreting offset-preserving micro-motion behavior.
- No D037 was appended in this session because the result may be a diagnostic
  execution-path issue rather than a durable mechanics rule.

## Verification

- Local py_compile passed.
- B200 py_compile passed.
- B200 md5 matched local.
- Protected md5s after coding still matched:
  - `sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`
    `938a94b3b856dcc5a48527991a87c1e9`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
