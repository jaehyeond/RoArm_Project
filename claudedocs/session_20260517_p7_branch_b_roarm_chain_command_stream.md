# Session 2026-05-17 - P7 Branch B RoArm chain-side command stream

## Scope

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not use SurfaceGripper.
- Did not tune P7 scalar/threshold/release guidance.
- Did not run structured A curriculum training.
- Did not add scripted release variants.
- Added only a local/numpy chain-side command-stream abstraction diagnostic.

## Boot Verification

- Read `CLAUDE.md`, `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D032,
  latest Branch B ledger rows, and the three requested session docs.
- `git status --short` before coding showed an already dirty worktree:
  `START_HERE.md`, `claudedocs/DECISIONS.md`, `claudedocs/EXPERIMENT_LEDGER.md`
  modified; two session docs and two Branch B probe scripts untracked.
- Required local md5s before coding matched the prompt:
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
- Requested timing and dynamic-anchor B200/local logs existed. The timing `.err`
  files were empty; B200 dynamic-anchor contract `.err` had only cpufreq/NVML
  warnings on lines 1-3.

## Prior Evidence Re-Checked

- Existing conservative timing dry-run logs, local and B200:
  - line 2 confirms `chain_side_only=YES`, `isaac_chain_integration=NO`,
    `constraint_prim_insertion=NO`, `surface_gripper=NO`, `p7_training=NO`;
  - lines 11-16 show raw planner gaps fail, including `home->high=0.211271m`
    and `grasp->transport_hover=0.022913m`;
  - lines 65 and 71-72 accept `CLOSE`, `HOLD`, and `RELEASE` only after target
    reached;
  - line 73 reports `preclose_cmds=38`, `attached_cmds=3`,
    `max_preclose_tcp_step_m=0.009525`,
    `max_attached_tcp_step_m=0.007691`,
    max FK errors `0.000997/0.000655`, `transport_final_error_m=0.000655`,
    and zero IK failures;
  - lines 74-75 report all gates YES and
    `ROARM_CHAIN_TIMING_RESAMPLE_SUCCESS=YES`.
- Existing no-margin timing cross-check, local and B200:
  - line 3 uses `resample_fraction=1.000`;
  - line 31 has one `PRE_MOVE` with `tcp_step_m=0.010351` and `ok=NO`;
  - lines 69-71 fail the stream and command-order gates.
- Existing B200 dynamic-anchor chain contract:
  - lines 40-42 confirm pre-chain scope and negative contract checks;
  - line 49 accepts `CLOSE`;
  - lines 59, 76, 94 accept `MOVE`;
  - line 111 accepts `RELEASE`;
  - lines 129-131 report max target error `0.001468`, release drop `0.338178`,
    and `DYNAMIC_ANCHOR_CHAIN_CONTRACT_SUCCESS=YES`.

## Script

- `sim_scripts/p7_branch_b_roarm_chain_command_stream_probe.py`
- md5 `d9a07b43bed44f6061144234d7f6ec36`

Design:

- Local/numpy-only diagnostic.
- Imports `roarm_rl/chain_skills.py` directly by file path to avoid importing
  Isaac/gym through `roarm_rl/__init__.py`.
- Uses existing `TrajectoryPlanner` and `sim_scripts/roarm_kinematics.py`.
- Builds explicit command events:
  `PRE_MOVE* -> CLOSE -> MOVE* -> HOLD -> RELEASE`.
- Validates a small state machine:
  - `PRE_MOVE` only before attach/release;
  - `CLOSE` only after the grasp target gate;
  - `MOVE` only while attached;
  - `HOLD` only after the transport target gate;
  - `RELEASE` only after `HOLD`/target reached;
  - no moves after release.
- Does not run Isaac, insert constraint prims, use SurfaceGripper, change
  `roarm_stack_env.py`, `train_ppo.py`, `chain_skills.py`, or launch defaults.

Falsifiable gates:

- max realized FK TCP step `<= 0.010m`
- max FK target error `<= 0.003m`
- final transport target error `<= 0.003m`
- IK failures `0`
- `release_after_target_ok=YES`
- `no_move_after_release=YES`
- `stream_shape_ok=YES`

## Runs

Local:

- Pass: `/tmp/p7_branch_b_roarm_chain_command_stream_probe.{out,err}`
- No-margin fail: `/tmp/p7_branch_b_roarm_chain_command_stream_probe_nomargin_fail.{out,err}`
- Both `.err` files were empty.

B200 subset:

- Synced only:
  - `sim_scripts/p7_branch_b_roarm_chain_command_stream_probe.py`
  - `sim_scripts/roarm_kinematics.py`
  - `roarm_rl/chain_skills.py`
- Remote subset path: `/tmp/roarm_chain_command_stream_probe`
- B200 md5s:
  - new script `d9a07b43bed44f6061144234d7f6ec36`
  - `sim_scripts/roarm_kinematics.py` `d4d0b5d6f5d0057b7ff4aaa4c285190f`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
- Pass: `/tmp/p7_branch_b_roarm_chain_command_stream_probe_b200.{out,err}`
- No-margin fail:
  `/tmp/p7_branch_b_roarm_chain_command_stream_probe_nomargin_fail_b200.{out,err}`
- Both B200 `.err` files were empty.

Cross-machine verification:

- Local and B200 pass `.out` sha256 matched:
  `8cad0ccef79ecb915db85fcd5b8151d3cbefc9a73cdd95bdef951262ae06b471`.
- Local and B200 no-margin fail `.out` sha256 matched:
  `1141b3eb9c8d07f4d635da89e97a4b62f0d4051b578d94756693ab92ff79fa6a`.

## Evidence

Pass run, local and B200 identical:

- Line 2: `command_stream_only=YES`, `chain_side_only=YES`,
  `isaac_chain_integration=NO`, `constraint_prim_insertion=NO`,
  `surface_gripper=NO`, `p7_training=NO`, `env_default_edits=NO`,
  `chain_defaults_edits=NO`.
- Line 3: gates are `fk_error_gate_m=0.003000`,
  `endpoint_gate_m=0.003000`, `max_tcp_step_m=0.010000`,
  `resample_fraction=0.900`.
- Line 4: stream schema is `PRE_MOVE* CLOSE MOVE* HOLD RELEASE`.
- Lines 19-24: raw planner gaps still fail the 10mm step gate; max raw gap is
  `0.211271m`, and `grasp -> transport_hover` is `0.022913m`.
- Line 73: `CLOSE` is accepted only after the grasp target gate with
  `target_error_m=0.000392`.
- Lines 75-77: attached `MOVE` commands all have IK convergence YES; max realized
  attached step is `0.007691`.
- Line 79: `HOLD` is accepted after final transport target error `0.000655m`.
- Line 80: `RELEASE` is accepted with `release_after_target_ok=YES`.
- Line 81: aggregate reports `events_total=44`, `pre_move_cmds=38`,
  `move_cmds=3`, `raw_max_gap_m=0.211271`, `raw_gap_ok=NO`,
  `max_pre_move_tcp_step_m=0.009525`, `max_move_tcp_step_m=0.007691`,
  max FK errors `0.000997/0.000655`, `transport_final_error_m=0.000655`,
  and zero IK failures.
- Lines 82-83: all stream/order/release gates pass and
  `ROARM_CHAIN_COMMAND_STREAM_SUCCESS=YES`.

No-margin failure, local and B200 identical:

- Line 3 uses `resample_fraction=1.000`.
- Line 39 rejects one `PRE_MOVE` because realized `tcp_step_m=0.010351`.
- Line 69 rejects `CLOSE` because the pre-move stream already failed.
- Line 77 reports `max_pre_move_tcp_step_m=0.010351`.
- Lines 78-79 report `pre_move_stream_ok=NO`, `close_ok=NO`,
  `command_order_ok=NO`, and `ROARM_CHAIN_COMMAND_STREAM_SUCCESS=NO`.

## Interpretation

- A formal chain-side command-stream abstraction can be built from the existing
  planner/kinematics while staying dry-run only.
- This confirms D032 in a more interface-like shape: exact 10mm spacing is still
  unsafe, but default conservative `resample_fraction=0.9` produces an explicit
  event stream satisfying the command, FK, and release-timing gates.
- This is not P7 success and not constraint integration.
- The unresolved real problem remains actual Isaac/RoArm chain dynamics,
  controller latency, TCP estimation timing, contact, attach/release timing, and
  eventual constraint insertion after explicit user approval.

## Verification

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_command_stream_probe.py`
  passed locally.
- Local pass exit code was 0.
- Local no-margin cross-check exited 2 as intended.
- B200 pass exit code was 0.
- B200 no-margin cross-check exited 2 as intended.
