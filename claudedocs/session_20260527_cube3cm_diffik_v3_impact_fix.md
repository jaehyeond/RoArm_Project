# Session 2026-05-27 - Professor cube3cm DiffIK v3 impact fix

## Scope

- Branch/task: professor 2026-05-26 cube3cm push/tap branch only.
- Do not mix with Track A grasp/dataset/training.
- No B200/SSH/pull/.ssh operations were used.
- GPU/IsaacLab runtime commands were run locally with escalated sandbox access
  because default Codex sandbox hides `/dev/nvidia*`.

## Boot Verification

- `git status --short --untracked-files=all --branch` was run first and initially
  reported clean `## master...origin/master`.
- `CLAUDE.md:5-31` Current-State Protocol was read: use START_HERE,
  DECISIONS, EXPERIMENT_LEDGER, session docs, git status, and verify metrics from
  logs before citing.
- `START_HERE.md:3,86-124,709-729` was re-read before edits. It said v2 reduced
  low-motion but worsened `(1,0)` impact/tip and that the next step was v3.
- `claudedocs/DECISIONS.md:5070-5134` D102 was re-read. It recorded v2 as mixed
  and not teacher/scale-up ready.
- `claudedocs/EXPERIMENT_LEDGER.md` latest row was re-read. It recorded the v2
  mixed result.

## v2 Static Diagnosis

Files/logs checked before code edit:

- `sim_scripts/cube3cm_push_diffik_probe.py:35-52`: v2 parameters and
  `--trajectory_variant` before v3.
- `sim_scripts/cube3cm_push_diffik_probe.py:221-233`: target path moves from
  precontact to through point around the cube.
- `roarm_rl/roarm_cube_push_env.py:377-413`: audit definitions. `impact` is a
  boolean outlier flag if final speed, total XY displacement, or tip angle exceeds
  audit p99 thresholds; it is not force or impulse.
- `diffik_probe_v2_eval1024_seed779_audit.out:1-6`: v2 mechanism PASS, controlled
  `0.932617188`, impact `0.038085938`, low-motion `0.051757812`.
- `diffik_probe_v2_eval1024_seed779_posthoc.out:6`: `(1,0)` controlled
  `0.785185185`, impact `0.144444444`, low-motion `0.085185185`.
- `diffik_probe_v2_eval1024_seed779_compare_to_v1.out:3`: v2 improved `(1,0)`
  controlled/low-motion but worsened impact and tip p95/max.

CSV-level diagnosis of v2 `(1,0)`:

- Rows: 270 `(1,0)` trials out of 1024.
- Impact rows: 39/270 = `0.144444444`.
- All 39 impact rows were caused by tip angle exceeding `tip_p99_deg`, not by
  final speed or total XY displacement.
- Therefore v3 targeted tip moment reduction, not harder/faster push.

## Code Change

Modified file:

- `sim_scripts/cube3cm_push_diffik_probe.py`

Change:

- Added default-preserving `--trajectory_variant v3`.
- Added v3 `(1,0)` parameters:
  - `--v3_posx_precontact_clearance_m=0.014`
  - `--v3_posx_push_through_m=0.020`
  - `--v3_posx_tcp_top_margin_m=-0.011`
  - `--v3_posx_lateral_offset_m=0.0`
  - `--v3_posx_approach_steps=300`
  - `--v3_posx_push_steps=220`
  - `--v3_posx_post_steps=60`
  - `--v3_posx_max_diffik_joint_step_rad=0.020`
- Added v3 stdout/summary/CSV markers:
  `v3_posx_steps`, `v3_posx_max_diffik_joint_step_rad`,
  `posx_variant_applied`, `v3_posx_applied`, `v3_posx_env_count`.

Validation:

- `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_audit.py sim_scripts/cube3cm_push_diffik_posthoc.py` PASS.
- `git diff --check` PASS.
- Current `sim_scripts/cube3cm_push_diffik_probe.py` md5:
  `f4c8dfe7d9117d733ec38a0ac68e4019`.

## Runtime Sequence

### v3 smoke16 seed780

Artifacts:

- `diffik_probe_v3_smoke16_seed780_stdout.out`
- `diffik_probe_v3_smoke16_seed780_audit.out`
- `diffik_probe_v3_smoke16_seed780_posthoc.out`
- `diffik_probe_v3_smoke16_seed780_summary.json`

Evidence:

- stdout lines 20-21: local IsaacLab run, 16 envs, built-in
  `DifferentialIKController`, no RoArm-local IK loop, no training/dataset/grasp/
  attach/posewrite, `trajectory_variant=v3`, v3 pos-x steps `300/220/60`.
- audit lines 1-6 PASS: rows `16`, controlled `1.000000000`, impact `0`,
  low-motion `0`, success marker `0.812500000`.
- summary lines 37/49: `posx_variant_env_count=0`, `v3_posx_env_count=0`, so this
  was mechanism smoke only, not weak-direction evidence.

### v3 reach16 seed779

Artifacts:

- `diffik_probe_v3_reach16_seed779_stdout.out`
- `diffik_probe_v3_reach16_seed779_audit.out`
- `diffik_probe_v3_reach16_seed779_posthoc.out`
- `diffik_probe_v3_reach16_seed779_summary.json`

Evidence:

- stdout lines 20-21: local IsaacLab run, 16 envs, same mechanism constraints.
- summary lines 37/49: `posx_variant_env_count=6`, `v3_posx_env_count=6`.
- audit lines 1-6 PASS: controlled `1.000000000`, impact `0`,
  low-motion `0.062500000`, success marker `0.562500000`, final TCP error
  `0.032624591`.
- posthoc line 6: `(1,0)` n=6, controlled `1.000000000`, impact `0`,
  low-motion `0.166666667`, success marker `0.666666667`.

### v3 frozen 1024 seed779

Artifacts:

- `diffik_probe_v3_eval1024_seed779_stdout.out`
- `diffik_probe_v3_eval1024_seed779_audit.out`
- `diffik_probe_v3_eval1024_seed779_posthoc.out`
- `diffik_probe_v3_eval1024_seed779_summary.json`
- `diffik_probe_v3_eval1024_seed779_compare_to_v1_v2.out`

Evidence:

- stdout lines 20-21: local IsaacLab run, `num_envs=1024`, built-in
  `DifferentialIKController`, no RoArm-local IK loop, no training/dataset/grasp/
  attach/posewrite, `trajectory_variant=v3`, v3 pos-x steps `300/220/60`.
- audit line 1: `csv_rows=1024`, `summary_trials=1024`, row count match.
- audit line 2: mechanism OK, zero posewrite during rollout, auto-reset disabled,
  env joint-delta action loop bypassed.
- audit line 3: controlled `0.969726562`, impact `0.004882812`,
  low-motion `0.035156250`, success marker `0.604492188`.
- audit line 4: mean push displacement `0.037237558m`, mean XY displacement
  `0.038506947m`, max speed `1.960922837m/s`.
- audit line 5: final TCP target error `0.023551417m`, clip mean `0.656992523`.
- audit line 6: learned policy `NO`, Track A grasp success `NO`, dataset ready
  `NO`.
- posthoc line 6: `(1,0)` n=270, controlled `0.929629630`, impact
  `0.014814815`, low-motion `0.088888889`, success marker `0.314814815`.
- posthoc line 8: worst grid is `(0,2)` by low+impact `0.095238095`, not the old
  v1 `(1,1)` pocket.

## Same-Seed v1/v2/v3 Comparison

Source:

- `diffik_probe_v3_eval1024_seed779_compare_to_v1_v2.out:1-10`

Key rows:

- line 1: rows are `1024/1024/1024` for v1/v2/v3, same seed 779, grid quantiles
  from v1.
- line 2 overall:
  - controlled `0.892578125 -> 0.932617188 -> 0.969726562`
  - impact `0.023437500 -> 0.038085938 -> 0.004882812`
  - low `0.136718750 -> 0.051757812 -> 0.035156250`
  - success `0.520507812 -> 0.580078125 -> 0.604492188`
  - final TCP error `0.028779610 -> 0.024324538 -> 0.023551417`
  - tip p95 `138.731293 -> 142.095001 -> 128.647156`
- line 3 `(1,0)`:
  - controlled `0.633333333 -> 0.785185185 -> 0.929629630`
  - impact `0.088888889 -> 0.144444444 -> 0.014814815`
  - low `0.274074074 -> 0.085185185 -> 0.088888889`
  - success `0.533333333 -> 0.440740741 -> 0.314814815`
  - final TCP error `0.043867240 -> 0.033135812 -> 0.039558476`
  - clip `0.900232803 -> 0.977681103 -> 1.000000000`
  - tip p95 `153.082306 -> 161.068298 -> 140.676743`
- line 4 grid `(1,1)`:
  - impact `0 -> 0.031250000 -> 0.007812500`
  - low `0.304687500 -> 0.023437500 -> 0`
  - controlled `0.796875000 -> 0.914062500 -> 0.945312500`
- line 9: remaining v3 `(1,0)` impacts are still tip-angle-only outliers.

## Interpretation

- Yes, in the 1024-v3 run the scripted IsaacLab Differential IK controller
  physically reached/pushed the cube in most trials: average XY displacement was
  about 3.85cm and controlled push rate was 96.97%.
- This is not learning. It is scripted physics evidence using IsaacLab built-in
  Differential IK with live Jacobian control.
- v3 fixes the main v2 failure mode: `(1,0)` tip impact drops from 14.44% to
  1.48%, and overall impact drops from 3.81% to 0.49%.
- Critical caveat: `(1,0)` success marker falls to 31.48%, final TCP error is
  worse than v2 in `(1,0)`, and clip is 1.0. That means v3 is a strong 10k
  robustness-test candidate, but not automatically a clean teacher trajectory.

## Next Step

Recommended next action:

1. If the professor's goal is "run 10,000 IsaacLab tests and report push/tap
   statistics", run a 10,240-env v3 scripted Differential IK robustness audit.
2. If the goal is to create teacher/dataset trajectories, run a small v3.1 sweep
   first to recover `(1,0)` displacement/success while preserving low impact.
3. Continue to keep this separate from Track A grasp/dataset/training.

