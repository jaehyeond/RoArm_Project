# D319 RL data-factory conveyor pilot

Date: 2026-07-07 KST

Scope: professor 10cm / 0.72kg cube top-view branch only. Local host GPU for
D290 conveyor runs. No B200/SSH/pull/RoArm/VLA/PPO/controller hand-condition.

Verdict:

`D319_DATA_CONVEYOR_LOW_MID_PRODUCER_UPPER_RL_TARGET_RENDER_REPLAY_GAP`

## Current-state checks

Followed the repo current-state protocol:

- Read `CLAUDE.md`.
- Read `START_HERE.md`.
- Read `claudedocs/DECISIONS.md`.
- Read `claudedocs/EXPERIMENT_LEDGER.md`.
- Read D318 session doc:
  `claudedocs/session_20260707_cube10cm_top_view_d318_train_eval_contract_hybrid_stop.md`.
- Checked `git status --short --untracked-files=all --branch`.

No existing dirty/untracked/ahead state was reverted. `HANDOFF.md` and
`TASKS.md` were not used. No B200/JHPark/SSH/pull was used.

## Step 0 - direction document

Added durable direction doc:

- `claudedocs/direction_20260708_rl_data_factory.md`

The recorded research chain is:

```text
script push -> rendered pair dataset -> RL training -> RL policy large-scale
data generation -> VLA training at the end
```

Interpretation:

- RL is the data-factory engine, not the final artifact.
- The existing 0-999 rendered script dataset is the script-only baseline/control
  corpus, not discarded data.
- RLDG (`arXiv:2412.09858`) is the literature anchor for specialist RL policies
  generating data for generalist policies.

## Step 1 - high-friction D314 audit

Audited the D314 high-friction row `2.2/1.8` from existing per-env/step data.

Output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm/high_friction_audit/high_friction_audit_d319.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/tap10cm/high_friction_audit/high_friction_audit_d319.md`

Result:

| metric | value |
|---|---:|
| strict useful | `0/32` |
| overshoot | `32/32` |
| contact/reaction | `8/32` |
| mean/max XY | `3.7688m / 11.9895m` |
| envs `>=1m` | `13/32` |
| max speed | `10.0m/s` |
| primitive stop unique | `[1]` |

Decision: this row remains solver/runaway-suspect and is not a valid immediate
training target.

## Step 2 - baseline-v2 generation pilot

Ran baseline v2:

```text
candidate8 zero-action + candidate8_hybrid_stop_after_useful
```

Runtime script:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/run_d319_conveyor_chunks.sh`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/run_d319_conveyor_chunks_envcsv.sh`

Common D290 settings:

- `--exec_source zero`
- `--rl_action_mode candidate8_diffik_target_residual`
- `--policy_action_space 3`
- `--candidate8_hybrid_stop_after_useful`
- `--reset_pose_source env_hook`
- `--d256_reset_sample_mode random`
- `--num_envs 100`
- `--steps 580`

The first run produced summary/dataset artifacts. The second run repeated the
same seeds/settings with `--out_env_csv` because exact data-entry filtering
requires env-level provenance.

## Step 3 - label filter and pass-rate audit

Offline audit script:

- `sim_scripts/cube10cm_top_view_d319_data_conveyor_audit.py`

Audit outputs:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_data_conveyor_audit_summary.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_data_conveyor_audit_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_all_env_filter_rows.csv`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_accepted_env_rows.csv`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_selected_200_for_replay_manifest.csv`

Filter rule:

```text
contact=1 AND reaction=1 AND useful=1 AND overshoot=0 AND max XY >= 1mm
```

Bin result:

| bin | generated | accepted | contact | reaction | useful | overshoot | mean XY | max XY | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0.7-0.9` | `300` | `289` (`96.3%`) | `300` | `300` | `289` | `11` | `8.64mm` | `57.05mm` | producer bin |
| `0.9-1.2` | `200` | `193` (`96.5%`) | `200` | `200` | `193` | `7` | `15.70mm` | `295.96mm` | producer bin |
| `1.2-1.6` | `300` | `58` (`19.3%`) | `294` | `294` | `58` | `242` | `232.40mm` | `11140.39mm` | RL contribution candidate |

The generator criterion is `>=30%` pass rate. Therefore low/mid bins can feed
the data conveyor; upper bin must be frozen as an RL contribution target rather
than patched by another hand-written controller condition.

## Step 5 - diversity audit

Comparison against the existing script-only 0-999 visual dataset labels:

| corpus | accepted | mean accepted XY | accepted XY variance | direction histogram |
|---|---:|---:|---:|---|
| script 0-999 accepted | `812` | `7.12mm` | `14.21mm^2` | `{"+x":496,"+x/+y":139,"+x/-y":167,"+y":6,"-x":4}` |
| D319 accepted | `540` | `10.33mm` | `11.37mm^2` | `{"+x_object_frame_commanded":540}` |

Interpretation:

- D319 low/mid bins are useful for a data-conveyor pilot.
- D319 accepted trajectories are directionally narrow. This is acceptable as a
  fixture-level pilot, but it is not POSCO-style generalization evidence.
- Goal-conditioned push direction/displacement remains necessary before making
  a broader VLA data claim.

## Step 4 status - replay render and LeRobot append

The 200-row replay selection manifest exists:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d319/audit/d319_selected_200_for_replay_manifest.csv`

But LeRobot append was not run. Reason:

- Existing visual renderers are manifest-fed `cube_x_m/cube_y_m` renderers for
  the original scripted top-view dataset.
- They do not replay D319's D256 reset episode, friction-bin material, and
  candidate8 hybrid-stop baseline trajectory into frames.
- Disk is also tight: local filesystem was `95%` used with about `34G`
  available after D319 runtime artifacts.

Decision: the next implementation needed for the conveyor is a dedicated D319
replay renderer, not a fake LeRobot append using the older 0-999 videos.

## Verification

- `python3 -m py_compile sim_scripts/cube10cm_top_view_d319_data_conveyor_audit.py`
  passed.
- `git diff --check` passed.
- D319 env CSV runtime completed for all 8 chunks.
- D319 audit script completed and wrote JSON/CSV/Markdown artifacts.
- `pgrep -af 'isaaclab|Isaac-Sim|train_cube_push_ppo|tensorboard|torchrun|rl_games|python .*cube10cm'`
  returned no residual experiment processes.
- `nvidia-smi` showed only Xorg graphics processes, but GPU utilization was
  `19%`, not `0%`. Therefore the "no residual compute process" check passed,
  while the strict "GPU utilization 0%" check did not pass on the desktop GPU.

## Next steps

1. Implement a D319 replay renderer that consumes
   `d319_selected_200_for_replay_manifest.csv` and reproduces D319 baseline-v2
   trajectories with friction/provenance metadata.
2. Render a small 5-10 episode smoke before attempting 200 episodes.
3. Convert that smoke to LeRobot v3 and validate decode/metadata.
4. Keep upper bin `1.2-1.6` frozen as the next RL contribution target.
5. Do not run same-setting longer PPO; future RL must control a parameter that
   zero-action cannot solve, such as direction, target displacement, stop
   margin, or approach/contact offset.
