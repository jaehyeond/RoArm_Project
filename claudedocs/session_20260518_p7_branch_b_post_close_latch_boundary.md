# Session 2026-05-18 - P7 Branch B post-close latch-boundary probe

## Scope

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper to the RoArm chain.
- Did not execute attached transport, release, or scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not run structured A curriculum training.
- Added only a narrow Isaac/RoArm post-close env latch-boundary diagnostic.

## Boot Verification

- Read `CLAUDE.md`, `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D033,
  latest Branch B ledger rows, and the requested session docs.
- `git status --short` before coding had no output.
- Required local md5s before coding matched the prompt:
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

- Command-stream conservative pass logs, local and B200:
  - line 2 confirms no Isaac integration, constraint insertion, SurfaceGripper,
    P7 training, env default edits, or chain default edits;
  - lines 19-24 show raw planner gaps fail the `0.010m` gate, including
    `home->high=0.211271m`;
  - line 81 reports `events_total=44`, `pre_move_cmds=38`, `move_cmds=3`,
    `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`,
    `max_pre_move_tcp_step_m=0.009525`,
    `max_move_tcp_step_m=0.007691`, max FK errors `0.000997/0.000655`,
    `transport_final_error_m=0.000655`, and zero IK failures.
- Command-stream no-margin failure logs, local and B200:
  - line 3 uses `resample_fraction=1.000`;
  - line 39 rejects one `PRE_MOVE` because realized `tcp_step_m=0.010351`;
  - lines 77-79 report `max_pre_move_tcp_step_m=0.010351`,
    `pre_move_stream_ok=NO`, `close_ok=NO`, `command_order_ok=NO`, and
    `ROARM_CHAIN_COMMAND_STREAM_SUCCESS=NO`.
- B200 RoArm articulation timing:
  - line 40 confirms no constraint prim insertion, no fixed/dynamic integration,
    no SurfaceGripper, no P7 training, no default edits, and release marker only;
  - line 86 reports `total_sim_steps=311`, `max_event_steps=16`,
    `event_timeouts=0`, `max_first_step_target_error_m=0.009291`,
    `one_step_target_ok=NO`, `max_final_target_error_m=0.002705`,
    `max_sim_tcp_step_m=0.001947`, `grasped_seen=NO`, and
    `release_gripper_open_ok=YES`;
  - lines 87-88 report scoped gates YES but `one_step_target_ok=NO`.
- B200 passive-contact close timing:
  - line 41 confirms marker-only close timing scope, with no constraint prim
    insertion, no SurfaceGripper, no attached transport, no release marker, and
    no default edits;
  - line 81 reports CLOSE reached in 15 steps with `gripper_q_deg=+23.02`,
    `d_tcp_sponge_m=0.023599`, `sponge_xy_drift_m=0.000005`,
    `min_upright_z=1.000000`, and latch seen;
  - lines 82-84 report all scoped gates YES but explicitly
    `attach_physics_validated=NO`, `release_physics_validated=NO`.

## Script

- `sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`
- md5 `58b628682a536535d3d9a6790c51974d`

Design:

- Imports the existing conservative command stream builder from
  `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`.
- Places the sponge at nominal pick after reset and executes only
  `PRE_MOVE* -> CLOSE`.
- Uses realized-TCP gated execution; no one-step command assumption.
- Captures pre-latch and latch-step state, then holds the same CLOSE/grasp pose
  for a short stationary post-close window.
- Measures latch pose jump, TCP/sponge separation jump, sponge velocity and
  angular velocity, quaternion angle change, uprightness, target error, TCP step,
  and hold drift.
- Treats env `_grasped` as current env kinematic latch only. It is not authored
  constraint attach and not attach/release physics evidence.
- Does not insert constraints, use SurfaceGripper, execute attached transport,
  run release, run P7 training, or edit env/train/chain defaults.
- After the first B200 run, the script was extended with attribution-only CLI
  options for `attach_quat_mode`, `attach_velocity_mode`, and a marker-only
  `--disable_attach_posewrite` control. These options change only this
  diagnostic run, not env/train/chain defaults.

Falsifiable gates:

- HOME sim TCP vs analytic FK `<= 0.003m`
- all executed events reach target gate under gated execution
- max final target error `<= 0.003m`
- max per-sim TCP step `<= 0.010m`
- no pre-close latch
- pre-close sponge XY drift `<= 0.005m`
- close-phase sponge XY drift `<= 0.020m`
- latch pose jump `<= 0.005m`
- latch/hold TCP-sponge separation jump `<= 0.010m`
- post-latch sponge speed `<= 0.050m/s`
- post-latch quaternion change `<= 5deg`
- minimum sponge upright z axis `>= 0.90`
- no NaN and no episode truncation

## Runs

Local:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_post_close_latch_boundary_probe.py`
  passed.

B200:

- Synced new script to
  `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/`.
- Remote md5 matched local:
  `58b628682a536535d3d9a6790c51974d`.
- Remote `py_compile` passed.
- Final run used:
  `OMNI_KIT_ACCEPT_EULA=YES`,
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`, and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

Logs:

- B200 `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_b200.out`
- B200 `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_b200.err`

## Evidence

B200 stdout:

- Line 40 confirms the probe header.
- Line 41 confirms scope: post-close latch-boundary only, no constraint prim
  insertion, no fixed/dynamic integration, no SurfaceGripper, no attached
  transport, no release marker, no P7 training, no default edits, and explicitly
  `attach_physics_validated=NO`, `release_physics_validated=NO`.
- Line 42 reports gates:
  `target_error_gate_m=0.003000`, `max_tcp_step_m=0.010000`,
  `post_latch_pose_jump_gate_m=0.005000`,
  `post_latch_hold_drift_gate_m=0.005000`,
  `post_latch_d_tcp_jump_gate_m=0.010000`,
  `post_latch_speed_gate_mps=0.050000`,
  `post_latch_quat_angle_gate_deg=5.00`, and hold steps `20`.
- Line 43 reports source stream and truncation:
  source `events_total=44`, executed events `39`, `pre_move_cmds=38`,
  `close_index=39`, `move_cmds_executed=0`, `raw_max_gap_m=0.211271`,
  `raw_gap_ok=NO`.
- Line 71 reports the baseline:
  `home_fk_error_m=0.001894`, settled sponge position
  `(+0.266020, -0.034486, +0.023500)`, `settled_upright_z=1.000000`,
  `attach_quat_mode=preserve`, and `attach_velocity_mode=zero`.
- Lines 72-80 sample PRE_MOVE events. All reached under gated execution with
  zero measurable sponge XY drift and no latch.
- Line 81 reports CLOSE:
  `steps=15`, `final_target_error_m=0.000817`,
  `max_sim_tcp_step_m=0.000457`, `gripper_q_deg=+23.02`,
  `d_tcp_sponge_m=0.023599`, `sponge_xy_drift_m=0.000005`,
  `max_sponge_speed_mps=0.000660`, `min_upright_z=1.000000`,
  `latch_seen=YES`, `latch_step=15`, `reached=YES`.
- Line 82 reports the immediate latch boundary:
  pre-step `274`, latch step `275`, threshold step `275`,
  `pre_d_tcp_sponge_m=0.023599`, `latch_d_tcp_sponge_m=0.023599`,
  `pose_jump_m=0.000000`, `xy_jump_m=0.000000`,
  `d_tcp_sponge_jump_m=0.000000`, `quat_angle_deg=0.000`,
  `upright_latch=1.000000`, and no velocity spike.
- Line 83 kills the first stationary post-latch hold step:
  `target_error_m=0.015684`, `tcp_step_m=0.016131`,
  `pose_drift_m=0.017552`, `xy_drift_m=0.006564`,
  `d_tcp_sponge_m=0.019945`,
  `sponge_speed_mps=1.696947`,
  `sponge_ang_speed_rps=17.195574`,
  `quat_angle_deg=21.267`, `upright_z=0.931860`,
  `grasped=YES`, `early_kill=YES`.
- Line 84 aggregate:
  `executed_events=39`, `total_sim_steps=276`, hold steps done `1`,
  `hold_max_target_error_m=0.015684`,
  `max_sim_tcp_step_m=0.016131`,
  `max_preclose_sponge_xy_drift_m=0.000000`,
  `max_close_sponge_xy_drift_m=0.006564`,
  `hold_max_pose_drift_m=0.017552`,
  `hold_max_speed_mps=1.696947`,
  `hold_max_ang_speed_rps=17.195574`,
  `hold_max_quat_angle_deg=21.267`,
  `latch_seen=YES`, `preclose_latch_seen=NO`,
  `hold_early_kill=YES`.
- Lines 85-86 report gates:
  `target_error_ok=NO`, `sim_step_ok=NO`, `post_latch_hold_ok=NO`,
  `attach_physics_validated=NO`, `release_physics_validated=NO`, and
  `ROARM_POST_CLOSE_LATCH_BOUNDARY_SUCCESS=NO`.

B200 stderr:

- Lines 1-3 are known cpufreq/NVML warnings.
- Line 4 reports `Failed to clone in Fabric`; stdout line 63 confirms one env
  still ran, and the probe completed.

Relevant env code:

- `roarm_rl/roarm_stack_env.py` lines 1184-1195 latch `_grasped` from
  distance+gripper threshold.
- `roarm_rl/roarm_stack_env.py` lines 1216-1236 implement kinematic pose-write
  attach: sponge root pose is written to TCP, quaternion is preserved by default,
  and velocity is zeroed by default.

## Interpretation

- The previous passive-contact/close timing result remains valid only through
  the latch marker step.
- The latch step itself was quiet in this run, but the first stationary
  post-latch hold step immediately violated the TCP target and sim-step gates
  while producing sponge pose drift, high linear/angular velocity, and a large
  quaternion change.
- The current env `_grasped` kinematic attach boundary is therefore not a stable
  post-close handoff surface for chain transport.
- This is not P7 success and not constraint integration.
- This does not validate object attachment physics, release physics, attached
  transport, SurfaceGripper, or constraint insertion.
- A new D034 decision was appended because this changes the durable
  do-not-proceed rule for the current env kinematic latch boundary.

## Attribution Matrix

Question:

- Is the post-latch failure caused by quaternion mode, velocity mode, or by the
  pose-write attach boundary itself?

Design:

- Reused the same `PRE_MOVE* -> CLOSE -> stationary grasp-pose hold` diagnostic.
- Ran four pose-write enabled modes plus one marker-only/no-posewrite control.
- Still no constraint prim insertion, no fixed/dynamic integration, no
  SurfaceGripper, no attached transport, no release marker, no P7 training, and
  no env/train/chain default edits.

B200 logs:

- `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_default_b200.{out,err}`
- `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_keep_b200.{out,err}`
- `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_identity_zero_b200.{out,err}`
- `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_identity_keep_b200.{out,err}`
- `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_no_posewrite_b200.{out,err}`

Evidence:

- Default `preserve+zero`: line 42 confirms
  `attach_posewrite_enabled=YES`; line 83 kills first hold step with
  `target_error_m=0.015684`, `tcp_step_m=0.016131`,
  `pose_drift_m=0.017552`, `quat_angle_deg=21.267`; lines 84-86 report
  `post_latch_hold_ok=NO` and success `NO`.
- `preserve+keep`: line 42 confirms `attach_velocity_mode=keep`; line 83 still
  kills first hold step with `target_error_m=0.013359`,
  `tcp_step_m=0.013831`, `pose_drift_m=0.014504`,
  `quat_angle_deg=16.664`; lines 84-86 report success `NO`.
- `identity+zero`: line 42 confirms `attach_quat_mode=identity`; line 83 still
  kills first hold step with `target_error_m=0.015831`,
  `tcp_step_m=0.016265`, `pose_drift_m=0.016712`,
  `quat_angle_deg=10.393`; lines 84-86 report success `NO`.
- `identity+keep`: line 42 confirms identity+keep; line 83 still kills first
  hold step with `target_error_m=0.012996`, `tcp_step_m=0.013450`,
  `pose_drift_m=0.013716`, `quat_angle_deg=8.849`; lines 84-86 report
  success `NO`.
- Marker-only/no-posewrite control: line 42 confirms
  `attach_posewrite_enabled=NO`; lines 83-88 show hold steps 1-20 with no early
  kill; line 89 reports `post_latch_hold_steps_done=20`,
  `hold_max_target_error_m=0.000817`, `max_sim_tcp_step_m=0.001947`,
  `hold_max_pose_drift_m=0.000000`, `hold_max_speed_mps=0.000604`; lines 90-91
  report `post_latch_hold_ok=YES` and success `YES`.

Interpretation:

- The proximate trigger is `_update_grasp_attach` pose-write to TCP.
- Quaternion reset and velocity keep reduce some secondary metrics but do not
  rescue the boundary; all pose-write enabled variants remain killed.
- The no-posewrite pass is only a marker-only negative control. It is not attach
  physics evidence and must not be used to justify attached transport.
- A new D035 decision was appended for this attribution result.

## Verification

- Local `py_compile` passed.
- B200 `py_compile` passed.
- B200 md5 matched local.
- Initial B200 run completed and internally reported
  `ROARM_POST_CLOSE_LATCH_BOUNDARY_SUCCESS=NO`.
- B200 attribution matrix completed: four pose-write enabled variants reported
  success `NO`; marker-only/no-posewrite control reported success `YES`.
