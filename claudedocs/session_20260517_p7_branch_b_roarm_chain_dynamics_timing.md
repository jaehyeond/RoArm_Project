# Session 2026-05-17 - P7 Branch B RoArm chain dynamics/timing probe

## Scope

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper to the RoArm chain.
- Did not tune P7 scalar/threshold/release guidance.
- Did not run structured A curriculum training.
- Did not add scripted release variants.
- Added only a narrow Isaac/RoArm articulation timing diagnostic.

## Boot Verification

- Read `CLAUDE.md`, `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D032,
  latest Branch B ledger rows, and the requested session docs.
- `git status --short` before coding had no output.
- Required local md5s before coding matched the prompt:
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
  - line 2 confirms `command_stream_only=YES`, `chain_side_only=YES`,
    `isaac_chain_integration=NO`, `constraint_prim_insertion=NO`,
    `surface_gripper=NO`, `p7_training=NO`;
  - lines 19-24 show raw planner gaps fail the `0.010m` gate; max raw gap is
    `0.211271m` and `grasp -> transport_hover` is `0.022913m`;
  - line 81 reports `events_total=44`, `pre_move_cmds=38`, `move_cmds=3`,
    `max_pre_move_tcp_step_m=0.009525`, `max_move_tcp_step_m=0.007691`,
    max FK errors `0.000997/0.000655`, `transport_final_error_m=0.000655`,
    and zero IK failures;
  - lines 82-83 report all command-stream gates YES and
    `ROARM_CHAIN_COMMAND_STREAM_SUCCESS=YES`.
- Command-stream no-margin failure logs, local and B200:
  - line 3 uses `resample_fraction=1.000`;
  - line 39 rejects one `PRE_MOVE` with `tcp_step_m=0.010351`;
  - line 77 reports `max_pre_move_tcp_step_m=0.010351`;
  - lines 78-79 report `pre_move_stream_ok=NO`, `close_ok=NO`,
    `command_order_ok=NO`, and `ROARM_CHAIN_COMMAND_STREAM_SUCCESS=NO`.
- B200 dynamic-anchor mock chain-command contract:
  - lines 40-42 confirm pre-chain scope and negative contract checks;
  - line 49 accepts `CLOSE`;
  - lines 59, 76, 94 accept `MOVE`;
  - line 111 accepts `RELEASE`;
  - lines 129-131 report max target error `0.001468`, release drop `0.338178`,
    and `DYNAMIC_ANCHOR_CHAIN_CONTRACT_SUCCESS=YES`.

## Script

- `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`
- md5 `339bdfd2ced7cf05b4ce87d2cd92128a`

Design:

- Builds the same conservative `PRE_MOVE* -> CLOSE -> MOVE* -> HOLD -> RELEASE`
  stream from `TrajectoryPlanner` and `roarm_kinematics.py`.
- Runs Isaac/RoArm articulation/controller through those command targets.
- Keeps the env sponge far away so the existing kinematic `_grasped` path cannot
  attach; `CLOSE` and `RELEASE` are marker/gripper timing checks only.
- Measures controller latency in sim steps, first-step target error,
  final target error, per-sim-step realized TCP motion, cached-vs-fresh TCP
  delta, and no-attach status.
- Does not insert constraint prims, use SurfaceGripper, integrate dynamic/fixed
  constraints, run P7 training, or edit env/train/chain defaults.

Falsifiable gates:

- HOME sim TCP vs analytic FK `<= 0.003m`
- all events reach target gate under gated execution
- final requested target error `<= 0.003m`
- per-sim-step realized TCP step `<= 0.010m`
- cached-vs-fresh TCP delta `<= 0.002m`
- no env `_grasped` attach
- release marker opens gripper below threshold
- no NaN and no episode truncation

## Runs

Local:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`
  passed.

B200:

- Synced new script to `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/`.
- Remote md5 matched local:
  `339bdfd2ced7cf05b4ce87d2cd92128a`.
- Remote `py_compile` passed.
- First B200 run without `OMNI_KIT_ACCEPT_EULA=YES` failed before probe logic:
  stdout lines 2-5 show EULA prompt; stderr line 1 is EOF.
- Final B200 run used:
  `OMNI_KIT_ACCEPT_EULA=YES`,
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`, and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

Logs:

- `/tmp/p7_branch_b_roarm_chain_dynamics_timing_probe_b200.out`
- `/tmp/p7_branch_b_roarm_chain_dynamics_timing_probe_b200.err`

## Evidence

B200 stdout:

- Lines 14-28 show B200 GPUs and Vulkan with driver `580.95.05`.
- Lines 39-40 confirm the probe and scope:
  articulation-only, no constraint prim insertion, no fixed/dynamic chain
  integration, no SurfaceGripper, no P7 training, no default edits, and
  `release_marker_only=YES`.
- Line 41 reports gates:
  target error `0.003000`, max TCP step `0.010000`, cache delta `0.002000`,
  home FK `0.003000`, and `resample_fraction=0.900`.
- Line 42 reports stream shape:
  `events_total=44`, `pre_move_cmds=38`, `move_cmds=3`,
  `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`.
- Lines 54-60 confirm env device/timing:
  `cuda:0`, physics step `0.005`, environment step `0.01`, one environment.
- Line 70 reports HOME sim-vs-FK baseline:
  `home_fk_error_m=0.000870`, `home_fk_ok=YES`.
- Lines 71-79 sample PRE_MOVE events. Each reaches under gated execution; first
  step errors are still often about `0.008-0.009m`, so single-step readiness is
  not guaranteed.
- Line 80 reports `CLOSE` marker reaches after 16 sim steps with
  `gripper_q_deg=+23.56`, `grasped=NO`.
- Lines 81-83 report MOVE marker events reach in 9/10/10 sim steps with final
  target errors `0.002606`, `0.002675`, and `0.002705`.
- Lines 84-85 report `HOLD` reaches in 2 steps and `RELEASE` marker reaches in
  3 steps with `gripper_q_deg=+20.60`, `grasped=NO`.
- Line 86 aggregate:
  `total_sim_steps=311`, `max_event_steps=16`, `event_timeouts=0`,
  `max_first_step_target_error_m=0.009291`, `one_step_target_ok=NO`,
  `max_final_target_error_m=0.002705`,
  `max_final_expected_error_m=0.002110`,
  `max_sim_tcp_step_m=0.001947`,
  `max_cache_fresh_delta_m=0.000000`,
  `grasped_seen=NO`, `release_gripper_open_ok=YES`.
- Lines 87-88 report all scoped gates YES except `one_step_target_ok=NO`, and
  `ROARM_CHAIN_DYNAMICS_TIMING_SUCCESS=YES`.

B200 stderr:

- Lines 1-3 are the known cpufreq/NVML warnings.
- Line 4 reports `Failed to clone in Fabric`; stdout line 62 confirms one env
  still ran, and the probe completed with exit code 0.

## Interpretation

- The conservative command stream can be executed by the real Isaac/RoArm
  articulation/controller in this no-contact/no-attach diagnostic, if the
  scheduler waits for realized TCP gates before advancing events.
- A one-sim-step-per-command assumption is false: aggregate line 86 reports
  `one_step_target_ok=NO`, and events needed up to 16 sim steps.
- Cached TCP and fresh TCP matched in this execution (`max_cache_fresh_delta=0`),
  but this only covers the post-step observation path used here.
- This is not P7 success, not contact validation, not object attach/release
  physics validation, and not constraint integration.
- Next narrow pre-integration risk is passive contact / close timing around the
  conservative stream, still without constraint prim insertion unless explicitly
  approved.

## Verification

- Local py_compile passed.
- B200 py_compile passed.
- B200 final run exit code was 0.
- B200 md5 matched local.
