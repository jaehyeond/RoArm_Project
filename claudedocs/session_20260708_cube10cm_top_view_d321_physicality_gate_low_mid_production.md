# D321 physicality gate + low/mid production conveyor

Date: 2026-07-08 KST

Verdict: `D321_PHYSICALITY_GATE_LOW_MID_LEROBOT_PASS_DESIGN_DRAFT`

## Scope

This session followed `CLAUDE.md` Current-State Protocol for the professor
10cm / 0.72kg cube top-view branch. It did not use B200/SSH/pull/RoArm/VLA/PPO,
and it did not add controller hand-conditions.

The failable experiment was the D321 low/mid data-conveyor production path:
state generation, label filtering with the new physicality gate, replay render,
LeRobot v3 append, and DataLoader one-batch load. Failure conditions were any
bin pass rate below `90%`, render/conversion/load failure, or disk pressure
forcing an early stop.

## Code Added Or Changed

- `sim_scripts/cube10cm_top_view_d321_data_conveyor_audit.py`
  - Audits D321 env-level conveyor rows.
  - Applies the physicality gate: `max_disp_xy_m >= 0.300` is
    `solver_outlier` and is excluded from accepted rows.
  - Joins D256 reset provenance and writes all/accepted CSVs plus summary JSON.
- `sim_scripts/cube10cm_top_view_d321_render_lerobot_conveyor.py`
  - Reads accepted D321 rows, writes per-chunk replay manifests, calls the D320
    replay renderer, appends to a LeRobot v3 dataset, validates each chunk, and
    deletes raw PNG frames after append.
  - Performs final DataLoader one-batch validation with `video_backend=pyav`.
- `sim_scripts/cube10cm_top_view_d320_replay_render.py`
  - Extended to support capped manifest reading and generic provenance fields
    needed by D321 replay manifests.
- `claudedocs/design_d321_goal_conditioned_primitive.md`
  - Design-only draft for the next goal-conditioned primitive action space.

## Step 0 - Disk And GPU Preflight

Disk before production:

- `df -h .`: `590G` total, `528G` used, `32G` available, `95%` used.
- Free bytes: `34291871744`.

GPU before production:

- File:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/nvidia_smi_before_d321.txt`
- RTX 4090 Laptop GPU visible.
- Memory `25MiB / 16376MiB`; GPU util `18%`.

The full requested 2,000 state episodes and 1,920 accepted replay-render
episodes fit under the disk guard. No scale-down was applied.

## Step 1 - Physicality Gate

Durable gate added to the D321 label filter:

- Rule: if `max XY displacement >= 300mm`, classify as `solver_outlier`.
- `solver_outlier` rows are rejected from accepted data.
- Rationale: D320 upper-bin audit found 6 meter-scale solver outliers, with a
  max displacement of `11.14m`. The 300mm threshold is three cube widths for a
  100mm cube, so it is a task-scale physicality bound rather than a tuned
  controller condition.

## Step 2 - State Production + Filter

Runtime:

- Script:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/run_d321_conveyor_chunks_envcsv.sh`
- Output root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/tap10cm_envcsv`
- Baseline: candidate8 zero-action + `candidate8_hybrid_stop_after_useful`.
- Direction: `+x` only.
- Reset: D256 random reset.
- Upper `1.2-1.6` bin was not produced.

Audit output:

- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/audit/d321_data_conveyor_audit_summary.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/audit/d321_data_conveyor_audit_summary.md`
- Accepted rows:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/audit/d321_accepted_env_rows.csv`

Filter rule:

`contact=1 AND reaction=1 AND useful=1 AND overshoot=0 AND max XY >=1mm AND max XY <300mm`

| bin | generated | accepted | contact | reaction | useful | overshoot | solver outlier | reject reasons | delta vs D319 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `0.7-0.9` | 1000 | 954 (95.4%) | 1000 | 1000 | 954 | 46 | 0 | `accepted=954`, `not_useful=46` | -0.93pp |
| `0.9-1.2` | 1000 | 966 (96.6%) | 999 | 999 | 966 | 34 | 1 | `accepted=966`, `not_useful=33`, `solver_outlier=1` | +0.10pp |
| combined | 2000 | 1920 (96.0%) | 1999 | 1999 | 1920 | 80 | 1 | `accepted=1920`, `not_useful=79`, `solver_outlier=1` | n/a |

Interpretation:

- Both bins passed the D321 `>=90%` production gate.
- Low/mid pass rates remained close to D319.
- The mid-bin solver-outlier row proves the D320 physicality gate is not just an
  upper-bin patch; rare nonphysical rows can appear below the upper bin too.

## Step 3 - Replay Render, LeRobot Append, DataLoader

Preflight:

- 2-episode preflight passed before full production:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_preflight_2ep/d321_render_lerobot_summary.json`

Full production:

- Output root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1`
- Summary:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1/d321_render_lerobot_summary.json`
- Dataset root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/render_lerobot_v1/lerobot_dataset`
- Chunks: `10` chunks (`9 x 200ep`, `1 x 120ep`).
- Episodes: `1920`.
- Frames: `280320`.
- Dataset bytes reported by validation: `782627028`.
- DataLoader validation: `PASS`.
- Video backend: `pyav`.
- Batch keys:
  `action`, `episode_index`, `frame_index`, `index`, `observation.images.top`,
  `observation.state`, `task`, `task_index`, `timestamp`.
- Batch shapes:
  - `observation.images.top`: `[2, 3, 720, 1280]`
  - `observation.state`: `[2, 6]`
  - `action`: `[2, 6]`

Raw frame handling:

- The chunk renderer wrote raw PNG frames only transiently.
- The D321 conveyor deleted raw PNG frames after each chunk append.
- Post-run `find .../renders -name '*.png'` returned no PNG files.
- Render directories remain with render summaries/manifests for provenance.

Disk after production:

- `df -h .`: `590G` total, `530G` used, `30G` available, `95%` used.
- Summary `disk_after.free_gb`: `31.771250688`.

GPU/process after production:

- File:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/data_conveyor_d321/nvidia_smi_after_d321.txt`
- RTX 4090 Laptop GPU, memory `25MiB / 16376MiB`, util `19%`.
- `pgrep -af 'isaaclab|Isaac-Sim|train_ppo|torchrun|rl_games|tensorboard'`
  returned no matches.

Warnings:

- The LeRobot append path emitted SVT-AV1 and torchvision video deprecation
  warnings. They did not block conversion or DataLoader validation.
- The local GPU's reported utilization remains around `18-19%` with only Xorg
  listed; there was no residual Isaac/PPO/torchrun/TensorBoard process.

## Diversity Audit

| corpus | accepted | mean accepted XY | accepted XY variance | direction histogram |
| --- | ---: | ---: | ---: | --- |
| script 0-999 accepted | 812 | 7.12mm | 14.21mm^2 | `{"+x":496, "+x/+y":139, "+x/-y":167, "+y":6, "-x":4}` |
| D321 accepted | 1920 | 9.83mm | 8.14mm^2 | `{"+x_object_frame_commanded":1920}` |

Interpretation:

- D321 is a larger low/mid producer dataset, not a direction-diverse dataset.
- It increases accepted low/mid replay-render volume, but it narrows the
  direction distribution to commanded `+x`.
- Direction diversity remains a D322+ goal-conditioned primitive or learned
  primitive-parameter problem.

## Step 4 - Goal-Conditioned Primitive Design Draft

Created:

- `claudedocs/design_d321_goal_conditioned_primitive.md`

Contents:

- Direction conditions: `{+x, -x, +y, -y}`.
- Target displacement bands.
- Candidate learnable primitive parameters: approach offset, lateral offset,
  push depth, stop margin, height offset.
- Reward draft aligned with strict useful, target matching, and the 300mm
  physicality gate.
- Zero-action baseline requirements for every direction.
- Curriculum: `+x` first, then `-x/-y`, then `+y`; upper friction after
  direction stability.
- Evaluation protocol: fresh32 x 4 directions x 2 friction bins with explicit
  train/eval contract fields.

This was design only. No PPO/RL training was run in D321.

## Verification

- `python -m py_compile` passed for:
  - `sim_scripts/cube10cm_top_view_d321_data_conveyor_audit.py`
  - `sim_scripts/cube10cm_top_view_d321_render_lerobot_conveyor.py`
  - `sim_scripts/cube10cm_top_view_d320_replay_render.py`
- `git diff --check` passed.
- No residual Isaac/PPO/TensorBoard/torchrun/rl_games processes were found.
- GPU compute processes were not listed in `nvidia-smi`; only Xorg was present.

## Decision

- The D321 low/mid producer path passed: state generation, physicality-gated
  filtering, replay render, LeRobot append, and DataLoader load all completed.
- The physicality gate is now required for data acceptance: `>=300mm` XY
  displacement is `solver_outlier`.
- Low/mid bins can be used as producer bins for script-v2 data.
- Upper bin production remains blocked and reserved as an RL contribution target.
- D321 is still `+x` only. It must not be presented as multi-direction data.
- Next work should use the goal-conditioned primitive design to create a
  direction-conditioned generator or learned primitive-parameter experiment,
  while preserving script-only and D321 script-v2 baselines.
