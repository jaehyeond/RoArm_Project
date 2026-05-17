# Session 2026-05-17 — P7 Branch B RoArm chain-side contract dry-run

## Scope

- Continued Track A P7/Branch B only.
- Did not integrate fixed/dynamic constraints into the RoArm chain.
- Did not use SurfaceGripper.
- Did not tune P7 scalar/threshold/release guidance.
- Did not run structured A curriculum training.
- Did not add scripted release variants.
- Added only a local/numpy dry-run diagnostic for real RoArm planner/kinematics
  command compatibility.

## Boot Verification

- Read `CLAUDE.md`, `START_HERE.md`, `claudedocs/DECISIONS.md` D024-D030,
  latest Branch B ledger rows, and the three requested session docs.
- `git status --short` before coding was clean.
- Required md5s before coding matched:
  - `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_probe.py`
    `6af24284baef540f190b762e5da164a5`
  - `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_interface_probe.py`
    `eb81372d78828730e63879a996911bbd`
  - `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_target_probe.py`
    `4706cdd555de659833df6756f95a4cb0`
  - `sim_scripts/p7_branch_b_fixed_constraint_dynamic_anchor_probe.py`
    `082f20f84eac10b76b3d678845321243`
  - `roarm_rl/roarm_stack_env.py` `e2748144034d5a09d6c7a0f6c0da6906`
  - `roarm_rl/train_ppo.py` `795ee48b1bfdd83e8c9735efd01f6920`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `launch_chain_topdown.sh` `b34ef3853ac993a1e2adbaddb420adab`
  - `launch_p6v17_transport_release.sh` `2acd462042d0997610fca25ff7a41e21`
- Requested prior B200 logs existed on `JHPark` under `/tmp`.

## Prior B200 Evidence Re-Checked

- Chain-command contract:
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_chain_contract_smoke.out`
  lines 40-41 confirm CPU/no chain/no transport/no SurfaceGripper/no P7
  training; line 42 rejects unsafe ordering; line 49 accepts `CLOSE`; lines 59,
  76, 94 accept `MOVE`; line 111 accepts `RELEASE` after target reached; lines
  129-131 report all gates YES with max target error `0.001468` and
  `release_drop=0.338178`.
- Mock-TCP interface:
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_interface_smoke.out` lines
  40-41 confirm pre-chain scope; lines 58, 75, 93 have
  `transform_error=0.000000`; line 128 reports max target error `0.001468`,
  `release_drop=0.338178`; lines 129-130 pass. Offset cross-check lines 48, 58,
  75, 93, 128-130 show the same with nonzero offset.
- Target tracking:
  `/tmp/p7_branch_b_fixed_constraint_dynamic_anchor_target_smoke.out` lines
  40-41 confirm pre-chain scope; line 83 reports final target errors
  `0.001426`; line 102 reports `max_move_rel=0.000000` and
  `release_drop=0.335825`; lines 103-104 pass. Half-command log lines 81 and
  100-102 report final target error `0.001429`, `release_drop=0.330823`, and
  pass.
- All five requested `.err` files contained only three warning lines each
  (cpufreq/NVML); no Python traceback.

## Script

- `sim_scripts/p7_branch_b_roarm_chain_contract_dryrun_probe.py`
- md5 `88b4b8b33cd7aeecd6a18f78bf144283`

Design:

- Imports `roarm_rl/chain_skills.py` directly by file path to avoid pulling
  `roarm_rl/__init__.py` and Isaac/gym dependencies.
- Uses existing `TrajectoryPlanner` waypoints and `sim_scripts/roarm_kinematics.py`
  FK/IK.
- Does not run Isaac, does not touch `roarm_stack_env.py`, `train_ppo.py`,
  `chain_skills.py`, or launch defaults.
- Checks:
  - waypoint FK TCP error against `0.003m`;
  - raw planner waypoint gaps against `0.010m`;
  - a proposed attached-transport contract stream resampled from grasp TCP to
    transport TCP;
  - `CLOSE` only after grasp target reached;
  - `RELEASE` only after transport target gate;
  - no `MOVE` after `RELEASE`.

## Runs

Local:

- `/tmp/p7_branch_b_roarm_chain_contract_dryrun_probe.out`
- `/tmp/p7_branch_b_roarm_chain_contract_dryrun_probe.err`

B200:

- Synced only the diagnostic subset to `/tmp/roarm_chain_contract_probe`:
  the new script, `roarm_rl/chain_skills.py`, and `sim_scripts/roarm_kinematics.py`.
- B200 md5s:
  - new script `88b4b8b33cd7aeecd6a18f78bf144283`
  - `roarm_rl/chain_skills.py` `c6e610216197994c6b7d2b6625d87560`
  - `sim_scripts/roarm_kinematics.py` `d4d0b5d6f5d0057b7ff4aaa4c285190f`
- Logs:
  - `/tmp/p7_branch_b_roarm_chain_contract_dryrun_probe_b200.out`
  - `/tmp/p7_branch_b_roarm_chain_contract_dryrun_probe_b200.err`

Local and B200 output matched on all reported metrics.

Evidence from B200/local line numbers:

- Line 2: `chain_side_only=YES`, `isaac_chain_integration=NO`,
  `constraint_prim_insertion=NO`, `surface_gripper=NO`, `p7_training=NO`,
  `env_default_edits=NO`, `chain_defaults_edits=NO`.
- Lines 12-17: all six planner waypoints satisfy the `0.003m` FK TCP gate.
  Max waypoint FK TCP error is line 30 aggregate `0.000551m`.
- Lines 19-23: raw planner waypoint gaps fail the conservative `0.010m`
  command-step criterion:
  - high -> hover `0.073074m`
  - hover -> 1b1 `0.018075m`
  - grasp -> transport `0.022913m`
- Line 24: `CLOSE` accepted only with `target_reached=YES`.
- Lines 25-27: resampled attached transport uses three `MOVE` events; all have
  `ik_converged=YES`, TCP steps `0.007048`, `0.007605`, `0.007648`, and FK
  errors `0.000611`, `0.000649`, `0.000646`.
- Line 28: `HOLD` accepted with final target error `0.000231m`.
- Line 29: `RELEASE` accepted with `release_after_target_ok=YES`.
- Line 30: aggregate `contract_move_steps=3`,
  `max_contract_tcp_step_m=0.007648`,
  `max_contract_fk_error_m=0.000649`,
  `final_transport_target_error_m=0.000231`.
- Lines 31-32: contract stream gates YES and
  `ROARM_CHAIN_CONTRACT_DRYRUN_SUCCESS=YES`; raw planner gap remains NO.
- Both stderr logs were empty.

## Interpretation

- Current RoArm planner/kinematics can produce a contract-compatible TCP command
  stream if attached transport is explicitly resampled into small TCP steps.
- The current raw planner waypoint/target spacing is too coarse for a
  conservative `0.010m` command-step contract. Do not treat raw `q_grasp ->
  q_transport` as directly compatible with the mock command contract.
- This remains a dry-run only. It does not validate articulation dynamics,
  controller latency, sim TCP estimate timing, contact, or attach/release timing.
- This is not P7 success and not chain-ready.

## Verification

- `python -m py_compile sim_scripts/p7_branch_b_roarm_chain_contract_dryrun_probe.py`
  passed locally.
- Local run exit code was 0.
- B200 run exit code was 0.
