# Session 2026-05-27 - Professor cube3cm DiffIK v3 10k audit

## Scope

- Branch/task: professor 2026-05-26 cube3cm push/tap branch only.
- Objective: run the professor-style 10,000-scale scripted IsaacLab Differential
  IK push/tap robustness test after v3 fixed the v2 tip-impact regression.
- Not Track A grasp, not PPO/VLA learning, not dataset generation.
- No B200/SSH/pull/.ssh operations were used.

## Preflight

- `git status --short --untracked-files=all --branch` was run before work. The
  tree already contained expected v3 docs/code/log changes from the prior step.
- `START_HERE.md:727-746` was re-read. It said v3 had passed the 1024 scripted
  DiffIK impact gate and that the next scale action for the professor's 10,000
  test was a 10,240-trial v3 scripted robustness audit.
- `claudedocs/DECISIONS.md:5140-5189` D103 was re-read. It said v3 is the
  preferred scripted DiffIK scale candidate, but not teacher/dataset readiness.
- `sim_scripts/cube3cm_push_diffik_probe.py:279-302` was checked before runtime:
  `episodes` loops call `env.reset()` each episode, so `num_envs=1024`,
  `episodes=10` produces 10,240 trials while keeping GPU memory at 1024-env scale.
- GPU preflight showed RTX 4090 Laptop with 13,088 MiB free and 0% utilization.

## Accounting Patch

Before the 10-episode run, the probe had an accounting issue for multi-episode
runs:

- `posewrite_calls_during_rollout` was reset inside each episode.
- `posx_variant_env_count` only represented the last episode.

Patch:

- Removed per-episode reset of `posewrite_calls_during_rollout`.
- Added `posx_variant_trial_count`, `v2_posx_trial_count`, and
  `v3_posx_trial_count`.
- Physics trajectory behavior was not changed.

Validation:

- `python -m py_compile sim_scripts/cube3cm_push_diffik_probe.py sim_scripts/cube3cm_push_diffik_audit.py sim_scripts/cube3cm_push_diffik_posthoc.py` PASS.
- `git diff --check` PASS.
- Current `sim_scripts/cube3cm_push_diffik_probe.py` md5:
  `dc6ca5a222f0bd9437d5f83bf5449729`.

## Runtime

Command shape:

- Local IsaacLab/GPU, escalated sandbox access.
- `num_envs=1024`
- `episodes=10`
- `seed=779`
- `trajectory_variant=v3`
- base steps `220/90/40`
- v3 pos-x steps `300/220/60`

Artifacts:

- `diffik_probe_v3_eval10240_seed779_stdout.out`
- `diffik_probe_v3_eval10240_seed779_stderr.out`
- `diffik_probe_v3_eval10240_seed779.csv`
- `diffik_probe_v3_eval10240_seed779_summary.json`
- `diffik_probe_v3_eval10240_seed779_audit.out`
- `diffik_probe_v3_eval10240_seed779_posthoc.out`
- `diffik_probe_v3_eval10240_seed779_compare_to_1024.out`

md5:

- audit `f4752641d1e8c7e9fa5d888ffdf1aa65`
- posthoc `798e2d0cb438953bd597cdf058ac90a0`
- compare `159e532cd0f432ce7040885317411afa`
- summary `9af2de1813605ab98d421b0a5b75be33`

## Mechanism Verification

- stdout line 20: `isaac_run=YES`, `num_envs=1024`, `episodes=10`,
  `total_trials=10240`, controller `IsaacLab_DifferentialIKController`,
  `local_roarm_ik_dls_control_loop=NO`, training `NO`, dataset generation `NO`,
  grasp `NO`, attach/object posewrite `NO`.
- stdout line 21: `trajectory_variant=v3`, base steps `220/90/40`, v3 pos-x
  steps `300/220/60`, `episode_length_s=6.080`.
- audit line 1: `csv_rows=10240`, `summary_trials=10240`, row count match.
- audit line 2: mechanism OK, zero posewrite during rollout, env auto-reset
  disabled, env joint-delta action loop bypassed.
- summary lines 36/39/53: posewrite calls `0`, `posx_variant_trial_count=2566`,
  `v3_posx_trial_count=2566`.

## Main Metrics

Source: `diffik_probe_v3_eval10240_seed779_audit.out:3-6`

- controlled push rate: `0.943164062`
- impact outlier rate: `0.007519531`
- low-motion rate: `0.042480469`
- success marker rate: `0.594824219`
- mean push displacement: `0.035298955m`
- mean XY displacement: `0.038060439m`
- max XY displacement: `0.386441052m`
- max cube speed: `2.300552368m/s`
- final TCP target error mean: `0.023529604m`
- diffik clip mean: `0.652074015`
- learned policy: `NO`
- Track A grasp success: `NO`
- dataset ready: `NO`

## Direction Breakdown

Source: `diffik_probe_v3_eval10240_seed779_posthoc.out:3-6`

- `(-1,0)` n=2558: controlled `0.933932760`, impact `0.011727912`,
  low-motion `0`, success `0.913604378`.
- `(0,-1)` n=2533: controlled `0.985392815`, impact `0.001579155`,
  low-motion `0.025661271`, success `0.454796684`.
- `(0,1)` n=2583: controlled `0.979094077`, impact `0.003871467`,
  low-motion `0.021293070`, success `0.712737127`.
- `(1,0)` n=2566: controlled `0.874512860`, impact `0.012860483`,
  low-motion `0.122759158`, success `0.296570538`.

Critical read:

- `(1,0)` is still the weakest direction.
- v3 solved the high impact/tip issue enough for robustness statistics, but not
  enough for a clean teacher/dataset trajectory.

## 1024 vs 10,240 Comparison

Source: `diffik_probe_v3_eval10240_seed779_compare_to_1024.out:1-25`

- line 1: rows `1024 -> 10240`, seed 779, implemented as
  `num_envs1024xepisodes10`.
- line 2 overall:
  - controlled `0.969726562 -> 0.943164062`
  - impact `0.004882812 -> 0.007519531`
  - low `0.035156250 -> 0.042480469`
  - success `0.604492188 -> 0.594824219`
  - final TCP `0.023551417 -> 0.023529604`
- line 3 `(1,0)`:
  - controlled `0.929629630 -> 0.874512860`
  - impact `0.014814815 -> 0.012860483`
  - low `0.088888889 -> 0.122759158`
  - success `0.314814815 -> 0.296570538`
- lines 10-14 impact causes:
  - overall impact rows: 77/10240
  - 65 tip-only
  - 9 displacement-only
  - 3 tip+displacement
  - no impact rows from final speed alone
- lines 15-24 per-episode impact stayed below about 1.2% in all 10 episodes.
- line 25 verdict: rows confirmed, mechanism pass, overall impact below 1%,
  `(1,0)` impact below 2%, `(1,0)` low-motion caveat remains, not learned policy,
  not dataset-ready.

## Interpretation

- The professor-style 10,000-scale scripted DiffIK push/tap test is now complete.
- The result is scientifically useful physics evidence: IsaacLab built-in
  Differential IK physically pushed/tapped the cube across 10,240 trials with low
  overall impact.
- This is not PPO/VLA learning and not Track A grasp success.
- This is not dataset readiness because `(1,0)` still has weak success and
  elevated low-motion.

## Next Step

- For professor reporting: use the 10,240 v3 result directly, with caveats.
- For teacher/dataset: do not scale again yet; design v3.1 for `(1,0)`
  low-motion/success recovery while keeping impact near or below the v3 1-2%
  direction-level range.

