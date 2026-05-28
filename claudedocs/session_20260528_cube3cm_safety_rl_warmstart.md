# Session 2026-05-28 - Cube3cm Safety-Aware RL Warm-Start Smoke

## Scope

- Professor cube3cm push/tap branch only.
- Explicitly approved small safety-aware RL warm-start pilot after D108.
- Not Track A grasp, not Track A runtime, not dataset generation, not 10k/100k
  learned-policy robustness.
- No B200, no SSH, no pull, no `.ssh` copy.

## Starting Evidence

- D108 allowed a small safety-aware RL warm-start pilot from the safety BC v3
  checkpoint, followed by frozen 1024 overall/per-bucket audit.
- Safety BC v3 had passed two direct BC rollouts, but low_x motion quality remained
  weak:
  - seed883 audit: controlled `0.953125000`, impact `0.004882812`,
    low-motion `0.030273438`, success `0.662109375`.
  - seed883 low_x bucket: impact `0.041095890`, low-motion `0.315068493`,
    success `0.315068493`.
  - seed884 audit: controlled `0.943359375`, impact `0.010742188`,
    low-motion `0.024414062`, success `0.662109375`.

## Code Changes

- `roarm_rl/roarm_cube_push_env.py`
  - md5 `9806c1fcfb4666355f825418da5b7d75`
  - Added default-off BC teacher checkpoint loading.
  - Added BC teacher normalized-action bridge and optional imitation reward.
  - Added logs for BC teacher blend, imitation MSE, and teacher action magnitude.
- `roarm_rl/train_cube_push_ppo.py`
  - md5 `5466a9c9d40a7f09d397fbffa7cdb878`
  - Added CLI controls for BC teacher blend, imitation reward, and PPO smoke
    `num_steps_per_env` / `save_interval`.
- `roarm_rl/eval_cube_push_policy.py`
  - md5 `fa68ee654c969aff7938867894acf125`
  - Added teacher-off/on BC bridge eval controls, `episode_length_s`, summary fields,
    and `posx_x_bucket` in CSV.
- `sim_scripts/cube3cm_push_diffik_bucket_audit.py`
  - md5 `62f74ce38c9a44f0f0790e00559f634a`
  - Made final TCP target error optional so PPO eval CSVs can still be bucket-audited.
- Added `sim_scripts/cube3cm_push_ppo_rollout_audit.py`
  - md5 `b92260c8f0986c1b6bfe233fcf417d01`
  - Mechanism audit for frozen PPO rollouts.

## Static And Environment Checks

- `python -m py_compile roarm_rl/roarm_cube_push_env.py roarm_rl/train_cube_push_ppo.py roarm_rl/eval_cube_push_policy.py sim_scripts/cube3cm_push_diffik_bucket_audit.py sim_scripts/cube3cm_push_ppo_rollout_audit.py` PASS.
- `git diff --check` PASS.
- Direct system `python -m roarm_rl.train_cube_push_ppo ...` failed before Isaac:
  `ppo_bc_teacher_warmstart_smoke12_seed885_stderr.out:1-6` shows
  `ModuleNotFoundError: No module named 'gymnasium'`.
- Local `conda run -n isaaclab python -c "import gymnasium; import roarm_rl"` passed,
  so IsaacLab runs used `conda run -n isaaclab`.

## PPO Warm-Start Smoke

Command class:

- Local IsaacLab/GPU, `num_envs=256`, `max_iterations=12`, seed `885`.
- BC teacher checkpoint:
  `diffik_state_action_dataset_v3_bucket_1024_seed779/bc_mlp_joint_delta_v3_safety_l2.pt`.
- PPO action semantics stayed 6D normalized joint delta.
- Safety limits used short 1.2s episode, IK endpoint reset, action smoothing `0.25`,
  max joint delta `0.006rad`, contact slowdown `0.25`, fast-cube slowdown `0.10`.
- BC teacher blend `0.35`, imitation reward scale `0.30`.

Evidence:

- Training stdout lines 47-78 confirm `cuda:0`, no-attach scope, action semantics,
  BC teacher checkpoint path, `bc_teacher_blend=0.35`, and
  `bc_teacher_imitation_reward_scale=0.3`.
- Lines 80-86 confirm environment device `cuda:0`, physics step `0.005`, env step
  `0.01`.
- Iteration 0 lines 121-155 confirm the actor/critic loop ran and logged
  `cube_push_bc_teacher_blend_mean=0.3500`,
  `cube_push_bc_teacher_imitation_mse=0.5454`, and
  `bc_teacher_imitation_penalty=-0.1636`.
- Final iteration lines 573-614 completed `model_11.pt`.
- Checkpoint md5: `c9f945a4d1eacd817d4733e7d9b7e48e`.

## Teacher-Off Frozen 1024 Audit

This is the only learned-policy performance check from the PPO smoke.

- Eval stdout lines 48-54 confirm no training, no dataset generation, no grasp/attach,
  no rollout object posewrite, checkpoint `model_11.pt`, seed `886`, and
  `bc_teacher_blend=0.0`.
- Lines 56-66 confirm `cuda:0` and 1024 envs.
- Line 100 summary: controlled `0.4707`, impact `0.0879`, low-motion `0.3447`,
  success `0.0781`, grasped marker `0`.
- PPO rollout audit lines 1-5 PASS mechanism only:
  - csv rows `1024`, row count match true.
  - learned policy true, training false, dataset generation false, grasp false,
    rollout object posewrite false.
  - controlled `0.470703125`, impact `0.087890625`, low-motion `0.344726562`,
    success `0.078125000`.
- Bucket audit lines 1-10 FAIL performance:
  - overall impact `0.087890625`, success `0.078125000`.
  - `(-1,0)` impact `0.230483271`.
  - `(1,0)` success `0.070833333`.
  - low_x success `0.197530864`; mid_x/high_x success `0` / `0.014492754`.

## Teacher-On Bridge Diagnostic

Purpose: check whether the BC teacher bridge itself helps when enabled, not to claim a
learned policy.

Short/safety-limited teacher-on:

- Eval stdout lines 48-53 confirm `bc_teacher_blend=1.0`, same short 1.2s/safety
  action-loop limits, and safety BC checkpoint path.
- Line 100 summary: controlled `0.4648`, impact `0.0977`, low-motion `0.3965`,
  success `0.0508`.
- Audit lines 1-5 PASS mechanism only; bucket lines 1-10 FAIL performance.

Direct-like teacher-on diagnostic:

- Eval stdout lines 48-53 confirm `bc_teacher_blend=1.0`, home reset
  (`ik_endpoint_reset=False`), longer `episode_length_s=6.0`, action smoothing `1.0`,
  max joint delta `0.04rad`, contact slowdown `1.0`, fast-cube slowdown `1.0`, and
  safety BC checkpoint path.
- Lines 56-74 confirm `cuda:0`, 256 envs, max episode length `600`.
- Line 100 summary: controlled `0.7930`, impact `0.0547`, low-motion `0.1758`,
  success `0.4180`, grasped marker `0`.
- Audit lines 1-5 PASS mechanism only with controlled `0.792968750`, impact
  `0.054687500`, low-motion `0.175781250`, success `0.417968750`.
- Bucket audit lines 1-10 PASS the loose posx bucket screen:
  - overall controlled `0.792968750`, impact `0.054687500`, success `0.417968750`.
  - `(1,0)` controlled `0.900000000`, impact `0`, success `0.516666667`.
  - low_x success `0.640000000`, mid_x `0.352941176`, high_x `0.500000000`.

## Interpretation

- The PPO/RL training path is connected and ran on GPU.
- The 12-iteration PPO policy is not a useful learned policy yet. Teacher-off 1024
  success is only `0.078125000` and impact is `0.087890625`.
- The BC teacher-to-PPO action bridge is not equivalent to the direct BC rollout:
  short episode + endpoint reset + action smoothing/slowdown/max-delta safety limits
  make the teacher-on bridge worse than the prior direct BC rollout.
- The direct-like bridge diagnostic partially recovers performance, so the BC
  checkpoint is not dead. The likely blocker is the mismatch between direct BC
  joint-target replay and PPO env normalized action-loop/safety curriculum.
- Therefore more PPO iterations right now would be poorly motivated. The next gate
  should redesign the warm-start environment/curriculum first.

## Decision

- Do not call `model_11.pt` a successful learned cube-push policy.
- Do not run 10k/100k learned-policy scaling from this PPO smoke.
- Keep the safety BC v3 direct rollout as the best current learned-policy artifact for
  this branch.
- Next valid work:
  1. make BC teacher bridge faithful enough under a controlled horizon before PPO,
  2. then use teacher decay/residual PPO with teacher-off frozen 1024 audit,
  3. only if teacher-off 1024 passes overall and bucket gates, consider larger learned
     robustness tests.

## Updated State

- Updated `START_HERE.md`.
- Appended `claudedocs/EXPERIMENT_LEDGER.md`.
- Appended `claudedocs/DECISIONS.md` D109.
