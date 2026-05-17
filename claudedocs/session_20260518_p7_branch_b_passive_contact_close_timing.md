# Session 2026-05-18 - P7 Branch B passive-contact / close-timing probe

## Scope

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not insert constraint prims.
- Did not attach SurfaceGripper to the RoArm chain.
- Did not tune P7 scalar/threshold/release guidance.
- Did not run structured A curriculum training.
- Did not add scripted release variants.
- Added only a narrow Isaac/RoArm passive-contact and close-timing diagnostic.

## Boot Verification

- Read `CLAUDE.md`, `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D033,
  latest Branch B ledger rows, and the requested session docs.
- `git status --short` before coding showed an already dirty worktree:
  `START_HERE.md`, `claudedocs/DECISIONS.md`,
  `claudedocs/EXPERIMENT_LEDGER.md` modified; previous dynamics timing session
  doc and script untracked. These were not reverted.
- Required local md5s before coding matched the prompt:
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
  - line 2 confirms command-stream/dry-run scope with no Isaac integration,
    constraint insertion, SurfaceGripper, P7 training, env default edits, or chain
    default edits;
  - lines 19-24 show raw planner gaps fail the `0.010m` gate, including
    `home->high=0.211271m`, `high->hover=0.073074m`,
    `hover->1b1_z59=0.018075m`, and `grasp->transport_hover=0.022913m`;
  - line 81 reports `events_total=44`, `pre_move_cmds=38`, `move_cmds=3`,
    `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`,
    `max_pre_move_tcp_step_m=0.009525`,
    `max_move_tcp_step_m=0.007691`, max FK errors `0.000997/0.000655`,
    `transport_final_error_m=0.000655`, and zero IK failures;
  - lines 82-83 report all scoped command-stream gates YES.
- Command-stream no-margin failure logs, local and B200:
  - line 3 uses `resample_fraction=1.000`;
  - line 39 rejects one `PRE_MOVE` because realized `tcp_step_m=0.010351`;
  - line 77 reports `max_pre_move_tcp_step_m=0.010351`;
  - lines 78-79 report `pre_move_stream_ok=NO`, `close_ok=NO`,
    `command_order_ok=NO`, and `ROARM_CHAIN_COMMAND_STREAM_SUCCESS=NO`.
- B200 dynamic-anchor mock chain-command contract:
  - lines 40-42 confirm pre-chain scope and negative contract checks;
  - line 49 accepts `CLOSE`;
  - lines 59, 76, and 94 accept `MOVE`;
  - line 111 accepts `RELEASE`;
  - lines 129-131 report max target error `0.001468`, release drop `0.338178`,
    and all gates YES.
- B200 RoArm articulation timing:
  - line 40 confirms no constraint prim insertion, no fixed/dynamic integration,
    no SurfaceGripper, no P7 training, no default edits, and release marker only;
  - line 42 reports `events_total=44`, `pre_move_cmds=38`, `move_cmds=3`,
    `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`;
  - line 86 reports `total_sim_steps=311`, `max_event_steps=16`,
    `event_timeouts=0`, `max_first_step_target_error_m=0.009291`,
    `one_step_target_ok=NO`, `max_final_target_error_m=0.002705`,
    `max_sim_tcp_step_m=0.001947`, `grasped_seen=NO`, and
    `release_gripper_open_ok=YES`;
  - lines 87-88 report scoped gates YES but `one_step_target_ok=NO`.

## Script

- `sim_scripts/p7_branch_b_roarm_chain_passive_contact_close_timing_probe.py`
- md5 `6cb899ca124ff588fcc011d2805fa605`

Design:

- Imports the existing conservative command stream builder from
  `sim_scripts/p7_branch_b_roarm_chain_dynamics_timing_probe.py`.
- Places the sponge at nominal pick after reset and executes only
  `PRE_MOVE* -> CLOSE`.
- Uses realized-TCP gated execution; no one-step command assumption.
- Stops immediately once the env `_grasped` marker appears, to avoid continuing
  into kinematic attach transport.
- Measures target error, per-sim TCP step, passive sponge XY/z motion, sponge
  speed, uprightness, gripper threshold timing, and env latch timing.
- Treats env `_grasped` as marker-only. It is not attach physics evidence.
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
- minimum sponge upright z axis `>= 0.90`
- env latch, if seen, must not occur before gripper threshold
- no NaN and no episode truncation

## Runs

Local:

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_passive_contact_close_timing_probe.py`
  passed.

B200:

- Synced new script to
  `/NHNHOME/WORKSPACE/0526040060_A/JHPark/roarm_b200/code/sim_scripts/`.
- Remote md5 matched local:
  `6cb899ca124ff588fcc011d2805fa605`.
- Remote `py_compile` passed.
- Final run used:
  `OMNI_KIT_ACCEPT_EULA=YES`,
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.95.05`, and
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`.

Logs:

- B200 `/tmp/p7_branch_b_roarm_chain_passive_contact_close_timing_probe_b200.out`
- B200 `/tmp/p7_branch_b_roarm_chain_passive_contact_close_timing_probe_b200.err`

## Evidence

B200 stdout:

- Line 40 confirms the probe header.
- Line 41 confirms scope: passive-contact/close-timing only, no constraint prim
  insertion, no fixed/dynamic integration, no SurfaceGripper, no attached
  transport, no release marker, no P7 training, no default edits, and env
  kinematic latch is marker-only.
- Line 42 reports gates:
  `target_error_gate_m=0.003000`, `max_tcp_step_m=0.010000`,
  `home_fk_gate_m=0.003000`, `preclose_drift_gate_m=0.005000`,
  `close_drift_gate_m=0.020000`, `min_upright_z_gate=0.900`,
  `resample_fraction=0.900`.
- Line 43 reports source stream and truncated execution:
  source `events_total=44`, executed events `39`, `pre_move_cmds=38`,
  `close_index=39`, `move_cmds_executed=0`, `raw_max_gap_m=0.211271`,
  `raw_gap_ok=NO`.
- Lines 55-60 confirm B200 env device/timing:
  `cuda:0`, physics step `0.005`, env step `0.01`.
- Line 71 reports the baseline:
  `home_fk_error_m=0.001894`, `home_fk_ok=YES`, settled sponge position
  `(+0.266020, -0.034486, +0.023500)`, and `settled_upright_z=1.000000`.
- Lines 72-80 sample PRE_MOVE events. All sampled PRE_MOVE events reached under
  gated execution, with `sponge_xy_drift_m=0.000000`, `min_upright_z=1.000000`,
  and `latch_seen=NO`.
- Line 81 reports CLOSE:
  `steps=15`, `final_target_error_m=0.000817`,
  `max_sim_tcp_step_m=0.000457`, `gripper_q_deg=+23.02`,
  `d_tcp_sponge_m=0.023599`, `sponge_xy_drift_m=0.000005`,
  `sponge_z_delta_m=+0.000001`, `max_sponge_speed_mps=0.000660`,
  `min_upright_z=1.000000`, `latch_seen=YES`, `latch_step=15`, `reached=YES`.
- Line 82 aggregate:
  `executed_events=39`, `total_sim_steps=275`,
  `max_final_target_error_m=0.002399`, `max_sim_tcp_step_m=0.001947`,
  `max_preclose_sponge_xy_drift_m=0.000000`,
  `max_close_sponge_xy_drift_m=0.000005`,
  `max_sponge_speed_mps=0.000660`, `min_upright_z=1.000000`,
  `latch_seen=YES`, `latch_event_index=39`, `latch_global_step=275`,
  `gripper_threshold_global_step=275`, `preclose_latch_seen=NO`,
  `kinematic_env_latch_is_marker_only=YES`.
- Lines 83-84 report all scoped gates YES and explicitly
  `attach_physics_validated=NO`, `release_physics_validated=NO`.

B200 stderr:

- Lines 1-3 are known cpufreq/NVML warnings.
- Line 4 reports `Failed to clone in Fabric`; stdout line 63 confirms one env
  still ran, and the probe completed with exit code 0.

## Interpretation

- Nominal passive approach/close timing around the conservative stream did not
  show pre-close sponge push, pre-close latch, target-gate timeout, excessive
  realized TCP step, or sponge tipping in this one-sponge B200 diagnostic.
- CLOSE produced the existing env `_grasped` marker at the same global step as
  the gripper threshold (`275`). The script stopped there, so it did not execute
  attached transport.
- This is not P7 success and not constraint integration.
- This does not validate object attachment physics, release physics, attached
  transport, SurfaceGripper, or constraint insertion.
- No new DECISIONS entry was appended because this run narrows the current
  pre-integration state but does not change a durable do-not-repeat rule.

## Verification

- Local `py_compile` passed.
- B200 `py_compile` passed.
- B200 final run exit code was 0.
- B200 md5 matched local.
