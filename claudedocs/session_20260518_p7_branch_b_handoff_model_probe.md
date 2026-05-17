# Session 2026-05-18 - P7 Branch B post-close handoff-model probe

## Scope

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper to the RoArm chain.
- Did not execute attached transport, release, or scripted release variants.
- Did not tune P7 scalar/threshold/release guidance.
- Did not run structured A curriculum training.
- Added only a narrow Isaac/RoArm CLOSE handoff-model diagnostic.

## Boot Verification

- Read `CLAUDE.md`, `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D035,
  latest Branch B ledger rows, and the requested session docs.
- `git status --short` before coding had no output.
- Required local md5s before coding matched the prompt, including:
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

- `roarm_rl/roarm_stack_env.py` lines 491-498 show `_apply_action` sets joint
  targets and calls `_update_grasp_attach()` only when `_grasped.any()`.
- `roarm_rl/roarm_stack_env.py` lines 1184-1195 show `_compute_intermediate_values`
  latches `_grasped` from distance plus gripper threshold and releases it when
  the gripper opens.
- `roarm_rl/roarm_stack_env.py` lines 1216-1236 show current kinematic attach:
  sponge root pose is written to TCP, quaternion is preserved or reset by config,
  and velocity is zeroed by default.
- B200 D034 baseline
  `/tmp/p7_branch_b_roarm_chain_post_close_latch_boundary_probe_b200.out` lines
  41-43 confirmed scope/truncated stream, line 82 showed quiet latch, and line
  83 killed the first stationary hold:
  `target_error_m=0.015684`, `tcp_step_m=0.016131`,
  `pose_drift_m=0.017552`, `sponge_speed_mps=1.696947`,
  `quat_angle_deg=21.267`.
- B200 D035 attribution matrix confirmed all pose-write enabled quaternion/
  velocity variants failed first hold, while marker-only/no-posewrite passed 20
  holds. This remained marker-only negative control, not attach physics.

## Script

- `sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`
- md5 `938a94b3b856dcc5a48527991a87c1e9`

Design:

- Imports the existing conservative command stream builder from
  `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`.
- Places the sponge at nominal pick after reset and executes only
  `PRE_MOVE* -> CLOSE`.
- Uses realized-TCP gated execution; no one-step command assumption.
- Monkey-patches `_update_grasp_attach` inside this diagnostic only; env/train/
  chain defaults remain untouched.
- Compares:
  - `posewrite_tcp`: current env baseline, snap sponge center to TCP.
  - `marker_only`: keep `_grasped` marker but no pose-write.
  - `delayed_posewrite`: wait 3 stationary env steps, then use current TCP-center
    pose-write.
  - `oneshot_align`: TCP-center pose-write once, then no continuous pose-write.
  - `offset_preserve_posewrite`: preserve latch-time TCP-to-sponge offset and
    continuously write sponge pose to `tcp + offset`.
- Measures target error, TCP step, sponge pose drift, `d_tcp_sponge` jump,
  TCP-to-sponge offset error, sponge linear/angular velocity, quaternion angle,
  upright z, latch timing, and early kill.
- Stops before any MOVE/transport and before release.
- Does not insert constraints, use SurfaceGripper, execute attached transport,
  run release, run P7 training, or edit env/train/chain defaults.

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
- latch/hold offset error `<= 0.010m`
- post-latch sponge speed `<= 0.050m/s`
- post-latch quaternion change `<= 5deg`
- minimum sponge upright z axis `>= 0.90`
- no NaN and no episode truncation

## Runs

Local:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_handoff_model_probe.py`
  passed.

B200:

- Synced new script to
  `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/`.
- Remote md5 matched local:
  `938a94b3b856dcc5a48527991a87c1e9`.
- Remote `py_compile` passed.
- Final v2 runs used:
  `OMNI_KIT_ACCEPT_EULA=YES`,
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`, and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

Logs:

- B200 `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_posewrite_tcp_v2_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_marker_only_v2_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_delayed_posewrite_v2_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_oneshot_align_v2_b200.{out,err}`
- B200 `/tmp/p7_branch_b_roarm_chain_handoff_model_probe_offset_preserve_posewrite_v2_b200.{out,err}`

## Evidence

Common B200 stdout:

- Line 41 in every v2 log confirms scope: no constraint prim insertion, no
  fixed/dynamic integration, no SurfaceGripper, no attached transport, no release
  marker, no P7 training, no default edits, and explicitly
  `attach_physics_validated=NO`, `release_physics_validated=NO`,
  `claim_attach_success=NO`.
- Line 43 in every v2 log confirms the conservative stream was truncated before
  MOVE: source `events_total=44`, executed events `39`, `pre_move_cmds=38`,
  `move_cmds_executed=0`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Line 71 in every v2 log reports the same HOME FK/sponge baseline:
  `home_fk_error_m=0.001894`, settled sponge
  `(+0.266020, -0.034486, +0.023500)`, `settled_upright_z=1.000000`.
- Line 81 in every v2 log reports CLOSE reached in 15 steps with
  `gripper_q_deg=+23.02`, `d_tcp_sponge_m=0.023599`,
  `sponge_xy_drift_m=0.000005`, `min_upright_z=1.000000`, latch seen.
- Line 82 in baseline and offset-preserve logs shows the latch step itself was
  quiet: `pose_jump_m=0.000000`, `d_tcp_sponge_jump_m=0.000000`,
  `quat_angle_deg=0.000`, latch step and gripper-threshold step both `275`.

Variant results:

- Current `posewrite_tcp` baseline:
  - line 83 failed first hold step with `target_error_m=0.015684`,
    `tcp_step_m=0.016131`, `pose_drift_m=0.017552`,
    `xy_drift_m=0.006564`, `offset_error_m=0.006043`,
    `sponge_speed_mps=1.696947`, `sponge_ang_speed_rps=17.195574`,
    `quat_angle_deg=21.267`, `early_kill=YES`;
  - line 84 aggregate had `hold_early_kill=YES`, `posewrite_calls=2`;
  - lines 85-86 reported `post_latch_hold_ok=NO` and
    `ROARM_POST_CLOSE_HANDOFF_MODEL_SUCCESS=NO`.
- `marker_only`:
  - lines 83-88 showed sampled hold steps 1-20 with no early kill;
  - line 89 reported `post_latch_hold_steps_done=20`,
    `hold_max_target_error_m=0.000817`, `max_sim_tcp_step_m=0.001947`,
    `hold_max_pose_drift_m=0.000000`, `hold_max_offset_error_m=0.000001`,
    `hold_max_speed_mps=0.000604`, `posewrite_calls=0`;
  - lines 90-91 reported `post_latch_hold_ok=YES` and success `YES`.
- `delayed_posewrite`:
  - lines 83-85 showed the first three stationary env steps passed without
    pose-write failure;
  - line 86 failed when the delayed TCP-center pose-write began:
    `target_error_m=0.015686`, `tcp_step_m=0.016133`,
    `pose_drift_m=0.017553`, `offset_error_m=0.006043`,
    `sponge_speed_mps=1.693429`, `quat_angle_deg=21.266`,
    `early_kill=YES`;
  - line 87 reported `delay_env_steps_seen=4`, `posewrite_calls=2`;
  - lines 88-89 reported `post_latch_hold_ok=NO` and success `NO`.
- `oneshot_align`:
  - line 83 failed first hold step with `target_error_m=0.005097`,
    `pose_drift_m=0.005682`, `sponge_speed_mps=0.823018`,
    `sponge_ang_speed_rps=16.806622`, `quat_angle_deg=7.080`,
    `early_kill=YES`;
  - line 84 reported `posewrite_calls=1`;
  - lines 85-86 reported `post_latch_hold_ok=NO` and success `NO`.
- `offset_preserve_posewrite`:
  - lines 83-88 showed sampled hold steps 1-20 with no early kill;
  - line 89 reported `post_latch_hold_steps_done=20`,
    `hold_max_target_error_m=0.000817`, `max_sim_tcp_step_m=0.001947`,
    `hold_max_pose_drift_m=0.000000`, `hold_max_offset_error_m=0.000001`,
    `hold_max_speed_mps=0.000869`, `posewrite_calls=40`,
    `offset_initialized=YES`;
  - lines 90-91 reported `post_latch_hold_ok=YES` and success `YES`.

B200 stderr:

- For all five v2 runs, stderr lines 1-4 were the known cpufreq/NVML/Fabric
  warnings. There was no Python traceback.

## Interpretation

- D034/D035 remain valid: the current TCP-center pose-write is killed at the
  first stationary post-latch hold step, and marker-only pass is not attach
  physics.
- The new split narrows the actual local trigger: it is specifically the
  center-snap geometry, not the mere existence of a post-latch marker and not
  only quaternion/velocity mode.
- Delaying the same center snap only delays failure; once the snap starts, the
  old failure reappears.
- One-shot center align is also too disruptive under the stationary hold gates.
- Continuous TCP-offset-preserving pose-write survives the stationary local
  hold, while still being a kinematic pose-write model. This is a local handoff
  candidate only.
- This is not P7 success, not constraint integration, not object attachment
  physics validation, not attached transport, and not release physics.

## Verification

- Local `py_compile` passed.
- B200 `py_compile` passed.
- B200 md5 matched local.
- Existing protected md5s after coding still matched:
  `roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`,
  `chain_skills.py` `c6e610216197994c6b7d2b6625d87560`,
  `train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`, and the previous Branch B
  probe md5s listed in Boot Verification.
- `git status --short` after implementation showed the new script plus state-doc
  updates only.
